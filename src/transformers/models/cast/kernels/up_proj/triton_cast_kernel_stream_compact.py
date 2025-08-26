import torch
import triton
import triton.language as tl


@triton.jit
def compute_backward_gradients_stream_compact_tile(
    grad_inter_block, up_proj_cached_ptr, 
    grad_gate_ptr, grad_up_proj_ptr,
    act_indices, gate_vals, global_bs_indices, block_idx, global_cols,
    mask_bs, mask_ls, row_active,
    stride_up_cached_actidx, stride_up_cached_ls,
    stride_grad_gate_bs, stride_grad_gate_nb,
    stride_grad_up_proj_actidx, stride_grad_up_proj_ls,
    out_dtype: tl.constexpr,
    line_size: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """Compute gate and up_proj gradients for current tile.
    
    Gate gradient = sum(grad_inter * up_proj_cached, dim=LS) for each block
    Up_proj gradient = grad_inter * gate_vals (broadcasted)
    """
    # Load up_proj_cached for current tile using sparse indexing
    # Only load where gate values are non-zero to avoid uninitialized memory
    gate_mask = gate_vals > 0.0  # (BLOCK_SIZE_BS,)
    up_proj_cached_ptrs = up_proj_cached_ptr + act_indices[:, None] * stride_up_cached_actidx + global_cols[None, :] * stride_up_cached_ls
    up_proj_cached_block = tl.load(up_proj_cached_ptrs, mask=mask_bs[:, None] & mask_ls[None, :] & gate_mask[:, None], other=0.0)    
    
    # ============ Gate Gradient Calculation ============
    # Compute elementwise product: grad_inter * up_proj_cached
    grad_product = grad_inter_block * up_proj_cached_block
    
    # Reduce over line_size dimension (sum across columns for each row)
    grad_gate_tile = tl.sum(grad_product, axis=1)  # (BLOCK_SIZE_BS,)
    
    # Store gate gradients using dense indexing (global_bs_indices + block_idx)
    grad_gate_ptrs = grad_gate_ptr + global_bs_indices * stride_grad_gate_bs + block_idx * stride_grad_gate_nb
    
    tl.atomic_add(grad_gate_ptrs, grad_gate_tile, mask=mask_bs & row_active)
    
    # ============ Up_proj Gradient Calculation ============
    # Broadcast gate_vals to match grad_inter_block shape: (BLOCK_SIZE_BS, BLOCK_SIZE_LS)
    gate_vals_broadcasted = gate_vals[:, None]  # (BLOCK_SIZE_BS, 1)
    
    # Compute grad_up_proj = grad_inter * gate_vals (broadcasted over LS dimension)
    grad_up_proj_block = grad_inter_block * gate_vals_broadcasted
    
    # Apply ReLU mask: only where up_proj_cached > 0
    relu_mask = up_proj_cached_block > 0.0
    grad_up_proj_block = tl.where(relu_mask, grad_up_proj_block, 0.0)
    
    # Store up_proj gradients using sparse indexing (same as cached input)
    grad_up_proj_ptrs = grad_up_proj_ptr + act_indices[:, None] * stride_grad_up_proj_actidx + global_cols[None, :] * stride_grad_up_proj_ls
    tl.store(grad_up_proj_ptrs, grad_up_proj_block.to(out_dtype), mask=row_active[:, None] & mask_ls[None, :])


def get_triton_autotune_config():
    """
    Autotune configurations for stream compact up-projection kernel.
    Grid is based on (NB, max_rows, col_chunks) so we need good block sizes.
    """
    return [
        # ------------------------------------------------------------------
        # 1) Square / very large matrices
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R': 16},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=32, num_stages=1),

        # ------------------------------------------------------------------
        # 2) One skinny dimension (64) – three permutations
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=2),

        # ------------------------------------------------------------------
        # 3) Two skinny dimensions (64) – again all permutations
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=4,  num_stages=2),

        # ------------------------------------------------------------------
        # 4) Fallback / edge cases
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  1},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R': 16},  num_warps=16, num_stages=2),
    ]


@triton.autotune(
    configs=get_triton_autotune_config(),
    key=['hidden_size', 'line_size', 'max_rows'],
)
@triton.jit
def fused_up_proj_stream_compact_kernel(
    # Input tensors
    x_ptr,                          # (BS, H) input
    up_weight_ptr,                  # (H, I) weights
    
    # Index mappings from stream compact preprocessing
    nb_maxrows_to_bs_ptr,           # (NB, max_rows) -> BS index
    nb_maxrows_to_local_idx_ptr,    # (NB, max_rows) -> local row index
    block_offsets_ptr,              # (NB,) -> block start offsets
    nb_maxrows_gate_vals_ptr,       # (NB, max_rows) -> gate values
    
    # Output tensor
    output_ptr,                     # (act_idx, LS) dense output
    up_proj_ptr,                    # (act_idx, LS) pre-gating values (optional)
    up_proj_cached_ptr,             # (act_idx, LS) cached values from forward pass
    grad_gate_ptr,                  # (BS, NB) gate gradient output
    grad_up_proj_ptr,               # (act_idx, LS) up_proj gradient output
    
    # Sizes
    hidden_size: tl.constexpr,
    line_size: tl.constexpr, 
    max_rows: tl.constexpr,
    
    # Strides
    stride_x_bs, stride_x_h,
    stride_w_h, stride_w_i,
    stride_out_actidx, stride_out_ls,
    stride_up_proj_actidx, stride_up_proj_ls,
    stride_up_cached_actidx, stride_up_cached_ls,
    stride_grad_gate_bs, stride_grad_gate_nb,
    stride_grad_up_proj_actidx, stride_grad_up_proj_ls,
    
    # Meta-params
    out_dtype: tl.constexpr,
    apply_gate: tl.constexpr,
    apply_relu: tl.constexpr,
    save_up_proj: tl.constexpr,
    calculate_grad_gate_up_proj: tl.constexpr,
    
    # Block sizes from autotune
    BLOCK_SIZE_BS: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_LS: tl.constexpr, 
    GROUP_SIZE_R: tl.constexpr,
):
    """
    Stream compact up-projection kernel.
    
    Grid: (NB, row_groups, col_chunks) where
    - NB: number of blocks  
    - row_groups: ceil(max_rows / (BLOCK_SIZE_BS * GROUP_SIZE_R))
    - col_chunks: ceil(line_size / BLOCK_SIZE_LS)
    
    Each thread block processes:
    - A specific block (nb_idx)
    - A chunk of rows within that block (row_chunk) 
    - A chunk of columns (col_chunk)
    
    Key insight: Same weight slice is reused across all rows in the same block.
    """
    
    pid = tl.program_id(0)
    
    # ------------------------------------------------------------------
    # Decompose pid into (block_idx, row_chunk, col_chunk)
    # ------------------------------------------------------------------
    num_col_chunks = tl.cdiv(line_size, BLOCK_SIZE_LS)  # column tiles per block

    row_chunks = tl.cdiv(max_rows, BLOCK_SIZE_BS)
    row_groups = tl.cdiv(row_chunks, GROUP_SIZE_R)
    num_pid_per_block = num_col_chunks * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem   = pid % num_pid_per_block

    col_chunk    = (pid_rem // GROUP_SIZE_R) % num_col_chunks
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group    = pid_rem // (num_col_chunks * GROUP_SIZE_R)
    row_chunk    = row_group * GROUP_SIZE_R + row_in_group

    if row_chunk >= row_chunks:
        return  # out-of-bounds

    # ------------------------------------------------------------------
    # Row and column offsets for this tile
    # ------------------------------------------------------------------
    row_start = row_chunk * BLOCK_SIZE_BS
    offs_bs   = tl.arange(0, BLOCK_SIZE_BS)
    rows_in_block = row_start + offs_bs
    mask_bs   = rows_in_block < max_rows

    offs_ls   = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls   = offs_ls < line_size
    col_offset = block_idx * line_size
    global_cols = col_offset + offs_ls

    # ------------------------------------------------------------------
    # Load index mappings for this tile
    # ------------------------------------------------------------------
    base_ptr  = block_idx * max_rows  # stride between blocks
    
    # Load global row indices (BS dimension)
    global_rows = tl.load(nb_maxrows_to_bs_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0)
    
    # Load local row indices and reconstruct act_idx values (output storage indices)
    local_indices = tl.load(nb_maxrows_to_local_idx_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0)
    block_offset = tl.load(block_offsets_ptr + block_idx)
    act_indices = block_offset + local_indices.to(tl.int64)
    
    # Load gate values and ensure they are non-negative (safety check)
    gate_vals_raw = tl.load(nb_maxrows_gate_vals_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0.0)
    gate_vals = tl.where(gate_vals_raw > 0, gate_vals_raw, 0.0)
    
    # Check which rows are actually active (gate > 0)
    row_active = gate_vals > 0.0  # bool mask per row
    
    # Early exit if tile is fully padded or all gate values are zero
    if tl.sum(gate_vals) == 0:
        return

    # ------------------------------------------------------------------
    # Matrix multiplication: x @ up_weight slice
    # ------------------------------------------------------------------
    # Accumulator initialization
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)

    # Main dot-product loop over hidden_size
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size

        # Load input x for selected global rows
        x_ptrs = x_ptr + global_rows[:, None] * stride_x_bs + curr_offs_h[None, :] * stride_x_h
        x_block = tl.load(x_ptrs, mask=row_active[:, None] & mask_h[None, :], other=0.0)

        # Load weight slice for this block and column chunk
        w_ptrs = up_weight_ptr + curr_offs_h[:, None] * stride_w_h + global_cols[None, :] * stride_w_i
        w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_ls[None, :], other=0.0)

        # Accumulate: x @ w
        acc += tl.dot(x_block, w_block)

    # ------------------------------------------------------------------
    # Apply activation and gating
    # ------------------------------------------------------------------
    # Apply ReLU if requested
    if apply_relu:
        acc = tl.where(acc > 0, acc, 0.0)

    # Store post-ReLU, pre-gating values if requested (for backward pass caching)
    if save_up_proj:
        up_proj_ptrs = up_proj_ptr + act_indices[:, None] * stride_up_proj_actidx + offs_ls[None, :] * stride_up_proj_ls
        tl.store(up_proj_ptrs, acc.to(out_dtype), mask=row_active[:, None] & mask_ls[None, :])

    # Apply gating if requested
    if apply_gate:
        acc *= gate_vals[:, None]

    # ------------------------------------------------------------------
    # Store to dense output format (act_idx, LS)
    # ------------------------------------------------------------------
    out_ptrs = output_ptr + act_indices[:, None] * stride_out_actidx + offs_ls[None, :] * stride_out_ls
    tl.store(out_ptrs, acc.to(out_dtype), mask=row_active[:, None] & mask_ls[None, :])
    
    # Conditionally compute backward gradients if requested
    if calculate_grad_gate_up_proj:
        compute_backward_gradients_stream_compact_tile(
            acc, up_proj_cached_ptr,
            grad_gate_ptr, grad_up_proj_ptr,
            act_indices, gate_vals, global_rows, block_idx, offs_ls,
            mask_bs, mask_ls, row_active,
            stride_up_cached_actidx, stride_up_cached_ls,
            stride_grad_gate_bs, stride_grad_gate_nb,
            stride_grad_up_proj_actidx, stride_grad_up_proj_ls,
            out_dtype,
            line_size,
            BLOCK_SIZE_BS, BLOCK_SIZE_LS,
        )


def fused_up_proj_gate_activation_sparse_triton_stream_compact(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    *,
    mappings: dict,
    zero_init: bool = False,
    out_dtype: torch.dtype = torch.float32,
    apply_gate: bool = True,
    apply_relu: bool = True,
    save_up_proj: bool = False,
    calculate_grad_gate_up_proj: bool = False,
    up_proj_cached: torch.Tensor = None, 
    grad_gate_output: torch.Tensor = None,
    grad_up_proj_output: torch.Tensor = None,
):
    """
    Stream compact sparse up-projection using precomputed index mappings.
    
    Args:
        x: (BS, H) input tensor
        up_weight: (H, I) weight tensor where I = num_blocks * line_size
        gate: (BS, NB) gate tensor (for compatibility, not used in computation)
        num_blocks: number of blocks
        line_size: size per block
        mappings: dict from create_stream_compact_index() containing:
            - nb_maxrows_to_bs: (NB, max_rows) -> BS mapping
            - nb_maxrows_to_actidx: (NB, max_rows) -> act_idx mapping
            - nb_maxrows_gate_vals: (NB, max_rows) -> gate values
            - max_rows: maximum active rows per block
            - total_act_idx: total number of active elements
        zero_init: whether to zero-initialize output (for compatibility)
        out_dtype: output data type
        apply_gate: whether to apply gating
        apply_relu: whether to apply ReLU activation
        save_up_proj: save pre-gating values to up_proj_output
        calculate_grad_gate_up_proj: compute gradients (not implemented yet)
        
    Returns:
        output: (act_idx, LS) dense tensor containing only active computations
        up_proj_output: if save_up_proj=True, (act_idx, LS) pre-gating values
    """
    
    batch_seq_size, hidden_size = x.shape
    intermediate_size = num_blocks * line_size
    
    # Validate inputs
    assert up_weight.shape == (hidden_size, intermediate_size)
    assert gate.shape == (batch_seq_size, num_blocks)
    
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in supported_dtypes, f"x must be fp16/bf16/fp32, got {x.dtype}"
    assert up_weight.dtype in supported_dtypes, f"up_weight must be fp16/bf16/fp32, got {up_weight.dtype}"
    
    # Extract mappings
    nb_maxrows_to_bs = mappings['nb_maxrows_to_bs']           # (NB, max_rows)
    nb_maxrows_to_local_idx = mappings['nb_maxrows_to_local_idx'] # (NB, max_rows)  
    nb_maxrows_gate_vals = mappings['nb_maxrows_gate_vals']   # (NB, max_rows)
    block_offsets = mappings['block_offsets']                 # (NB,)
    max_rows = mappings['max_rows']
    total_act_idx = mappings['total_act_idx']
    
    # Early exit if no active elements
    if max_rows == 0 or total_act_idx == 0:
        return torch.zeros((0, line_size), device=x.device, dtype=out_dtype)
    
    # Assert that input tensors have efficient memory layout (at least one stride ≤ 1)
    assert min(x.stride()) <= 1, f"x has inefficient stride pattern: {x.stride()}"
    assert min(up_weight.stride()) <= 1, f"up_weight has inefficient stride pattern: {up_weight.stride()}"
    x_contiguous = x  # No copy needed
    up_weight_contiguous = up_weight  # No copy needed
    
    # Allocate output tensor in dense format (act_idx, LS)
    output = torch.empty((total_act_idx, line_size), device=x.device, dtype=out_dtype)
    
    # Conditionally allocate up_proj buffer for pre-gating caching
    if save_up_proj:
        up_proj_output = torch.empty((total_act_idx, line_size), device=x.device, dtype=out_dtype)
        up_proj_ptr = up_proj_output
        stride_up_proj_actidx, stride_up_proj_ls = up_proj_output.stride()
    else:
        up_proj_output = None
        up_proj_ptr = output  # Use dummy pointer (won't be accessed due to save_up_proj=False)
        stride_up_proj_actidx, stride_up_proj_ls = 0, 0  # Dummy strides
    
    # Handle gradient computation parameters
    if calculate_grad_gate_up_proj:
        if up_proj_cached is None or grad_gate_output is None or grad_up_proj_output is None:
            raise ValueError("up_proj_cached, grad_gate_output, and grad_up_proj_output must be provided when calculate_grad_gate_up_proj=True")
        up_proj_cached_ptr = up_proj_cached
        grad_gate_ptr = grad_gate_output
        grad_up_proj_ptr = grad_up_proj_output
        
        # Zero-initialize grad_gate_output since we use atomic_add for accumulation
        grad_gate_output.zero_()
        
        stride_up_cached_actidx, stride_up_cached_ls = up_proj_cached.stride()
        stride_grad_gate_bs, stride_grad_gate_nb = grad_gate_output.stride()
        stride_grad_up_proj_actidx, stride_grad_up_proj_ls = grad_up_proj_output.stride()
    else:
        up_proj_cached_ptr = output  # Dummy pointer
        grad_gate_ptr = output  # Dummy pointer
        grad_up_proj_ptr = output  # Dummy pointer
        stride_up_cached_actidx, stride_up_cached_ls = 0, 0
        stride_grad_gate_bs, stride_grad_gate_nb = 0, 0
        stride_grad_up_proj_actidx, stride_grad_up_proj_ls = 0, 0
    
    # Grid calculation
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_LS = meta["BLOCK_SIZE_LS"]
        G_SIZE_R = meta["GROUP_SIZE_R"]

        num_col_chunks = triton.cdiv(line_size, BLK_LS)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, G_SIZE_R)
        num_pid_per_block = num_col_chunks * G_SIZE_R * row_groups

        return (num_pid_per_block * num_blocks,)

    # Map torch dtypes to triton dtypes
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    # Launch kernel
    fused_up_proj_stream_compact_kernel[grid](
        # Input tensors
        x_contiguous,
        up_weight_contiguous,
        
        # Index mappings
        nb_maxrows_to_bs,
        nb_maxrows_to_local_idx,
        block_offsets,
        nb_maxrows_gate_vals,
        
        # Output
        output,
        up_proj_ptr,
        up_proj_cached_ptr,
        grad_gate_ptr,
        grad_up_proj_ptr,
        
        # Sizes
        hidden_size, line_size, max_rows,
        
        # Strides
        x_contiguous.stride(0), x_contiguous.stride(1),
        up_weight_contiguous.stride(0), up_weight_contiguous.stride(1),
        output.stride(0), output.stride(1),
        stride_up_proj_actidx, stride_up_proj_ls,
        stride_up_cached_actidx, stride_up_cached_ls,
        stride_grad_gate_bs, stride_grad_gate_nb,
        stride_grad_up_proj_actidx, stride_grad_up_proj_ls,
        
        # Meta-params
        out_dtype=triton_out_dtype,
        apply_gate=apply_gate,
        apply_relu=apply_relu,
        save_up_proj=save_up_proj,
        calculate_grad_gate_up_proj=calculate_grad_gate_up_proj,
    )
    
    if save_up_proj:
        return output, up_proj_output  # (gated + ReLU, pre-gated + ReLU)
    else:
        return output  # (gated + ReLU)


if __name__ == "__main__":
    """Basic test of the stream compact up-projection kernel"""
    
    print("✅ Stream compact up-projection kernel loaded successfully")
    
    # Simple test with random data
    BS, NB, LS, H = 16, 8, 32, 64
    
    x = torch.randn(BS, H, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(H, NB * LS, device="cuda", dtype=torch.float16)
    gate = torch.rand(BS, NB, device="cuda", dtype=torch.float32)
    
    # Make gate sparse (70% zeros)
    mask = torch.rand_like(gate) < 0.7
    gate[mask] = 0.0
    
    print(f"Input shapes: x={x.shape}, up_weight={up_weight.shape}, gate={gate.shape}")
    print(f"Gate sparsity: {(gate == 0).float().mean().item():.1%}")
    
    # Create stream compact mappings
    from kernels.stream_compact_index import create_stream_compact_index
    mappings = create_stream_compact_index(gate)
    
    print(f"Max rows: {mappings['max_rows']}, Total act_idx: {mappings['total_act_idx']}")
    
    # Test kernel
    try:
        output = fused_up_proj_gate_activation_sparse_triton_stream_compact(
            x, up_weight, gate, NB, LS, mappings=mappings
        )
        print(f"Output shape: {output.shape}")
        print("✅ Stream compact up-projection test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise