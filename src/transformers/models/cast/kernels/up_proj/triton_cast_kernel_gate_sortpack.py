import torch
import triton
import triton.language as tl

# =============================================================================
# Sort-pack (ELLPACK-like) fused up-proj + gate + activation kernel
#   – no CSR writer kernel, we simply sort the gate columns so that all
#     non-zero rows are packed contiguously at the top of each block column
#   – each block column therefore has exactly `max_rows` entries; rows past the
#     true number of non-zeros are zero-padded and can be skipped cheaply in the
#     kernel.
# =============================================================================

@triton.jit
def compute_backward_gradients_tile(
    grad_inter_block, up_proj_cached_ptr, 
    grad_gate_ptr, grad_up_proj_ptr,
    row_indices, gate_vals, block_idx, global_cols,
    mask_bs, mask_ls,
    stride_up_cached_bs, stride_up_cached_i,
    stride_grad_gate_bs, stride_grad_gate_nb,
    stride_grad_up_proj_bs, stride_grad_up_proj_i,
    out_dtype: tl.constexpr,
    line_size: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """Compute gate and up_proj gradients for current tile.
    
    Gate gradient = sum(grad_inter * up_proj_cached, dim=LS) for each block
    Up_proj gradient = grad_inter * gate_vals (broadcasted)
    """
    # Load up_proj_cached for current tile (grad_inter_block is already passed in)
    # Only load where gate values are non-zero to avoid uninitialized memory
    gate_mask = gate_vals > 0.0  # (BLOCK_SIZE_BS,)
    up_proj_cached_ptrs = up_proj_cached_ptr + row_indices[:, None] * stride_up_cached_bs + global_cols[None, :] * stride_up_cached_i
    up_proj_cached_block = tl.load(up_proj_cached_ptrs, mask=mask_bs[:, None] & mask_ls[None, :] & gate_mask[:, None], other=0.0)    
    
    # ============ Gate Gradient Calculation ============
    # Compute elementwise product: grad_inter * up_proj_cached
    grad_product = grad_inter_block * up_proj_cached_block
    
    # Reduce over line_size dimension (sum across columns for each row)
    grad_gate_tile = tl.sum(grad_product, axis=1)  # (BLOCK_SIZE_BS,)
    
    # Store gate gradients - each row contributes to one gate element for this block
    grad_gate_ptrs = grad_gate_ptr + row_indices * stride_grad_gate_bs + block_idx * stride_grad_gate_nb
    
    tl.atomic_add(grad_gate_ptrs, grad_gate_tile, mask=mask_bs)
    
    # ============ Up_proj Gradient Calculation ============
    # Broadcast gate_vals to match grad_inter_block shape: (BLOCK_SIZE_BS, BLOCK_SIZE_LS)
    gate_vals_broadcasted = gate_vals[:, None]  # (BLOCK_SIZE_BS, 1)
    
    # Compute grad_up_proj = grad_inter * gate_vals (broadcasted over LS dimension)
    grad_up_proj_block = grad_inter_block * gate_vals_broadcasted
    
    # Apply ReLU mask: only where up_proj_cached > 0
    relu_mask = up_proj_cached_block > 0.0
    grad_up_proj_block = tl.where(relu_mask, grad_up_proj_block, 0.0)
    
    # Store up_proj gradients directly (no atomic add needed since each tile writes to unique locations)
    grad_up_proj_ptrs = grad_up_proj_ptr + row_indices[:, None] * stride_grad_up_proj_bs + global_cols[None, :] * stride_grad_up_proj_i
    tl.store(grad_up_proj_ptrs, grad_up_proj_block.to(out_dtype), mask=mask_bs[:, None] & mask_ls[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 16, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_LS': 16, 'GROUP_SIZE_R': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_BS': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_LS': 32, 'GROUP_SIZE_R': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_LS': 64, 'GROUP_SIZE_R': 4}, num_warps=8, num_stages=2),
    ],
    key=['hidden_size', 'line_size', 'save_up_proj', 'calculate_grad_gate_up_proj'],
    reset_to_zero=['grad_gate_ptr'],
)
@triton.jit
def fused_up_proj_gate_sortpack_kernel(
    x_ptr, w_ptr,
    row_idx_ptr, gate_vals_ptr,        # (NB, max_rows) column-major (row-major in memory)
    output_ptr,
    up_proj_ptr,                       # optional output for pre-ReLU values
    up_proj_cached_ptr,                # pointer to cached up_proj values  
    grad_gate_ptr,                     # pointer to gate gradient output
    grad_up_proj_ptr,                  # pointer to up_proj gradient output
    stride_up_cached_bs, stride_up_cached_i,       # strides for up_proj_cached
    stride_grad_gate_bs, stride_grad_gate_nb,      # strides for grad_gate output
    stride_grad_up_proj_bs, stride_grad_up_proj_i, # strides for grad_up_proj output
    # sizes
    hidden_size: tl.constexpr, line_size: tl.constexpr, max_rows: tl.constexpr,
    # strides / leading dimensions
    stride_x_bs, stride_x_h,
    stride_w_h, stride_w_i,
    stride_row_blk,                    # max_rows – distance between consecutive blocks in row_idx / gate_vals
    stride_out_bs, stride_out_i,
    stride_up_proj_bs, stride_up_proj_i,  # strides for up_proj_ptr
    # meta-params
    out_dtype: tl.constexpr,
    apply_gate: tl.constexpr,
    apply_relu: tl.constexpr,
    save_up_proj: tl.constexpr,        # whether to save pre-ReLU values to up_proj_ptr
    calculate_grad_gate_up_proj: tl.constexpr, # flag to enable gradient computation
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_LS: tl.constexpr, GROUP_SIZE_R: tl.constexpr,
):
    """Each CTA processes a tile of size (BLOCK_SIZE_BS rows × BLOCK_SIZE_LS cols)
    for a given (block column, row chunk, column chunk) triple encoded into the
    linear program id (same packing scheme as the CSR kernel to ease tuning).
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
    # Row gather & gate values
    # ------------------------------------------------------------------
    row_start = row_chunk * BLOCK_SIZE_BS
    offs_bs   = tl.arange(0, BLOCK_SIZE_BS)
    rows_in_block = row_start + offs_bs
    mask_bs   = rows_in_block < max_rows

    base_ptr  = block_idx * stride_row_blk  # stride_row_blk == max_rows
    row_indices = tl.load(row_idx_ptr  + base_ptr + rows_in_block, mask=mask_bs, other=0)
    gate_vals  = tl.load(gate_vals_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0.0)
    # Clamp negative gate values to zero
    gate_vals = tl.where(gate_vals > 0, gate_vals, 0.0)

    # ------------------------------------------------------------------
    # Column offsets for the current tile
    # ------------------------------------------------------------------
    offs_ls   = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls   = offs_ls < line_size
    col_offset = block_idx * line_size
    global_cols = col_offset + offs_ls

    # Early exit if this tile is fully padded
    if tl.sum(gate_vals) == 0:
        # If computing gradients, zero out grad_up_proj for this tile before returning
        if calculate_grad_gate_up_proj:
            grad_up_proj_ptrs = grad_up_proj_ptr + row_indices[:, None] * stride_grad_up_proj_bs + global_cols[None, :] * stride_grad_up_proj_i
            zeros = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)
            tl.store(grad_up_proj_ptrs, zeros.to(out_dtype), mask=mask_bs[:, None] & mask_ls[None, :])
        return

    # ------------------------------------------------------------------
    # Accumulator initialisation
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main dot-product loop over hidden_size
    # ------------------------------------------------------------------
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size

        x_ptrs = x_ptr + row_indices[:, None] * stride_x_bs + curr_offs_h[None, :] * stride_x_h
        w_ptrs = w_ptr + curr_offs_h[:, None] * stride_w_h + global_cols[None, :] * stride_w_i

        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_h[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_ls[None, :], other=0.0)
        acc += tl.dot(x_block, w_block)

    # Apply activation (ReLU) if requested
    if apply_relu:
        acc = tl.where(acc > 0, acc, 0.0)
    
    # Store post-ReLU, pre-gating values if requested (for backward pass caching)
    if save_up_proj:
        up_proj_ptrs = up_proj_ptr + row_indices[:, None] * stride_up_proj_bs + global_cols[None, :] * stride_up_proj_i
        tl.store(up_proj_ptrs, acc.to(out_dtype), mask=mask_bs[:, None] & mask_ls[None, :])

    if apply_gate:
        acc *= gate_vals[:, None]

    out_ptrs = output_ptr + row_indices[:, None] * stride_out_bs + global_cols[None, :] * stride_out_i
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_bs[:, None] & mask_ls[None, :])
    
    # Conditionally compute backward gradients if requested
    if calculate_grad_gate_up_proj:
        compute_backward_gradients_tile(
            acc, up_proj_cached_ptr, 
            grad_gate_ptr, grad_up_proj_ptr,
            row_indices, gate_vals, block_idx, global_cols,
            mask_bs, mask_ls,
            stride_up_cached_bs, stride_up_cached_i,
            stride_grad_gate_bs, stride_grad_gate_nb,
            stride_grad_up_proj_bs, stride_grad_up_proj_i,
            out_dtype,
            line_size,
            BLOCK_SIZE_BS, BLOCK_SIZE_LS,
        )


# =============================================================================
# Python helper – mirrors the original CSR wrapper but uses torch.sort packing
# =============================================================================

def fused_up_proj_gate_activation_sparse_triton_sortpack(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    zero_init: bool = True,
    out_dtype: torch.dtype = torch.float32,
    apply_gate: bool = True,
    apply_relu: bool = True,
    # New preprocessed gate parameters
    gate_vals: torch.Tensor = None,
    row_idx: torch.Tensor = None, 
    block_counts: torch.Tensor = None,
    max_rows: int = None,
    # Optimization parameter - moved to end for backward compatibility
    save_up_proj: bool = False,        # whether to save pre-ReLU values
    calculate_grad_gate_up_proj: bool = False,
    up_proj_cached: torch.Tensor = None, 
    grad_gate_output: torch.Tensor = None,
    grad_up_proj_output: torch.Tensor = None,
):
    """Sort-pack (ELLPACK) sparse fused MLP helper.

    This is a drop-in replacement for *_csr* variants but relies on a single
    Triton kernel and a `torch.sort` pre-pass instead of a CSR writer kernel.
    """

    batch_seq_size, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Shape / dtype validation
    assert up_weight.shape == (hidden_size, intermediate_size)
    assert gate.shape == (batch_seq_size, num_blocks)

    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert (
        x.dtype in supported_dtypes and up_weight.dtype in supported_dtypes
    ), f"x, up_weight must be fp16/bf16/fp32, got x={x.dtype}, up_weight={up_weight.dtype}"

    # Promote gate to fp32 for better precision in sorting / multiplication
    if gate.dtype != torch.float32:
        gate = gate.float()

    # Use inputs directly (already flattened)
    # Assert that input tensors have efficient memory layout (at least one stride ≤ 1)
    assert min(x.stride()) <= 1, f"x has inefficient stride pattern: {x.stride()}"
    assert min(up_weight.stride()) <= 1, f"up_weight has inefficient stride pattern: {up_weight.stride()}"
    x_reshaped = x  # (BS, H)

    # ------------------------------------------------------------------
    # Require preprocessed gate data - no longer do internal preprocessing
    # ------------------------------------------------------------------
    if gate_vals is None or row_idx is None or block_counts is None or max_rows is None:
        raise ValueError("gate_vals, row_idx, block_counts, and max_rows must all be provided")
    
    # Early exit: gate is entirely zero
    if max_rows == 0:
        return torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)

    # Flatten for raw pointer access in Triton
    gate_vals_flat = gate_vals.view(-1)
    row_idx_flat   = row_idx.view(-1)

    # ------------------------------------------------------------------
    # Allocate / zero-initialise output buffer
    # ------------------------------------------------------------------
    if zero_init:
        output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)
    else:
        output = torch.empty((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)

    # ------------------------------------------------------------------
    # Conditionally allocate up_proj buffer for pre-ReLU caching
    # ------------------------------------------------------------------
    if save_up_proj:
        if zero_init:
            up_proj_output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)
        else:
            up_proj_output = torch.empty((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)
        up_proj_ptr = up_proj_output
        stride_up_proj_bs, stride_up_proj_i = up_proj_output.stride()
    else:
        up_proj_output = None
        up_proj_ptr = output  # Use dummy pointer (won't be accessed due to save_up_proj=False)
        stride_up_proj_bs, stride_up_proj_i = 0, 0  # Dummy strides

    # ------------------------------------------------------------------
    # Handle gradient computation parameters
    # ------------------------------------------------------------------
    if calculate_grad_gate_up_proj:
        if up_proj_cached is None or grad_gate_output is None or grad_up_proj_output is None:
            raise ValueError("up_proj_cached, grad_gate_output, and grad_up_proj_output must be provided when calculate_grad_gate_up_proj=True")
        up_proj_cached_ptr = up_proj_cached
        grad_gate_ptr = grad_gate_output
        grad_up_proj_ptr = grad_up_proj_output
        
        # Zero-initialize grad_gate_output since we use atomic_add for accumulation
        grad_gate_output.zero_()
        
        stride_up_cached_bs, stride_up_cached_i = up_proj_cached.stride()
        stride_grad_gate_bs, stride_grad_gate_nb = grad_gate_output.stride()
        stride_grad_up_proj_bs, stride_grad_up_proj_i = grad_up_proj_output.stride()
    else:
        up_proj_cached_ptr = output  # Dummy pointer
        grad_gate_ptr = output  # Dummy pointer
        grad_up_proj_ptr = output  # Dummy pointer
        stride_up_cached_bs, stride_up_cached_i = 0, 0
        stride_grad_gate_bs, stride_grad_gate_nb = 0, 0
        stride_grad_up_proj_bs, stride_grad_up_proj_i = 0, 0

    # ------------------------------------------------------------------
    # Grid size helper (same logic as CSR variant)
    # ------------------------------------------------------------------
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_LS = meta["BLOCK_SIZE_LS"]
        G_SIZE_R = meta["GROUP_SIZE_R"]

        num_col_chunks = triton.cdiv(line_size, BLK_LS)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, G_SIZE_R)
        num_pid_per_block = num_col_chunks * G_SIZE_R * row_groups

        return (num_pid_per_block * num_blocks,)

    # ------------------------------------------------------------------
    # Launch Triton kernel
    # ------------------------------------------------------------------
    # Map torch dtypes to triton dtypes
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    fused_up_proj_gate_sortpack_kernel[grid](
        x_reshaped,
        up_weight,
        row_idx_flat,
        gate_vals_flat,
        output,
        up_proj_ptr,
        up_proj_cached_ptr,
        grad_gate_ptr,
        grad_up_proj_ptr,
        stride_up_cached_bs, stride_up_cached_i,
        stride_grad_gate_bs, stride_grad_gate_nb,
        stride_grad_up_proj_bs, stride_grad_up_proj_i,
        hidden_size, line_size, max_rows,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        max_rows,  # stride between blocks in row_idx / gate_vals
        output.stride(0), output.stride(1),
        stride_up_proj_bs, stride_up_proj_i,
        out_dtype=triton_out_dtype,
        apply_gate=apply_gate,
        apply_relu=apply_relu,
        save_up_proj=save_up_proj,
        calculate_grad_gate_up_proj=calculate_grad_gate_up_proj,
    )

    if save_up_proj:
        return output, up_proj_output  # (gated + ReLU, pre-ReLU gated)
    else:
        return output  # already (BS, I) 