import torch
import triton
import triton.language as tl

# =============================================================================
# Stream compact sparse down-proj kernel
#   – Uses stream compact format: intermediate is (act_idx, LS), not (BS, I).
#   – Leverages existing stream compact index mappings for efficient sparse access.
#   – For now, uses atomic adds like sort pack (future: separate sum optimization).
# =============================================================================

# ----------------------------- autotune configs -----------------------------
def get_triton_autotune_config():
    """
    Autotune configurations adapted from sort pack down proj kernel.
    Same tile size strategies work for stream compact format.
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
    key=["hidden_size", "line_size", "two_stage_reduction"],
    reset_to_zero=['output_ptr'],
)
@triton.jit
def fused_down_proj_stream_compact_kernel(
    # Input tensors
    x_ptr,                          # (act_idx, LS) sparse intermediate
    w_ptr,                          # (I, H) down projection weight
    
    # Index mappings from stream compact preprocessing  
    nb_maxrows_to_bs_ptr,           # (NB, max_rows) -> BS index
    nb_maxrows_to_actidx_ptr,       # (NB, max_rows) -> sequential act_idx
    nb_maxrows_gate_vals_ptr,       # (NB, max_rows) -> gate values
    max_rows_per_block_ptr,         # (NB,) number of active rows per block
    
    # Output tensor
    output_ptr,                     # (BS, H) or (act_idx, H) depending on two_stage_reduction
    
    # Sizes
    hidden_size: tl.constexpr, 
    line_size: tl.constexpr, 
    max_rows: tl.constexpr,
    
    # Strides
    stride_x_actidx, stride_x_ls,
    stride_w_ls, stride_w_h,
    stride_out_dim0, stride_out_h,  # dim0 = BS or act_idx depending on mode
    
    # Meta-params
    out_dtype: tl.constexpr,
    two_stage_reduction: tl.constexpr,  # New flag
    
    # Block sizes from autotune
    BLOCK_SIZE_BS: tl.constexpr, 
    BLOCK_SIZE_LS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, 
    GROUP_SIZE_R: tl.constexpr,
):
    """CTA processes tile (BLOCK_SIZE_BS rows × BLOCK_SIZE_H cols) for a given
    (block column, row chunk, hidden-chunk) triple encoded in the program id.
    Accumulates fp32 and atomically adds into the global output tensor.
    
    Key difference from sort pack: intermediate input is in sparse (act_idx, LS) format.
    """

    pid = tl.program_id(0)

    # ------------------------------------------------------------------
    # Decompose pid into (block_idx, row_chunk, col_chunk)
    # ------------------------------------------------------------------
    num_col_chunks = tl.cdiv(hidden_size, BLOCK_SIZE_H)  # hidden chunks per block

    row_chunks = tl.cdiv(max_rows, BLOCK_SIZE_BS)
    row_groups = tl.cdiv(row_chunks, GROUP_SIZE_R)
    num_pid_per_block = num_col_chunks * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem = pid % num_pid_per_block

    col_chunk = (pid_rem // GROUP_SIZE_R) % num_col_chunks  # hidden chunk
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group = pid_rem // (num_col_chunks * GROUP_SIZE_R)
    row_chunk = row_group * GROUP_SIZE_R + row_in_group

    if row_chunk >= row_chunks:
        return

    # ------------------------------------------------------------------
    # Gather rows from stream compact mappings
    # ------------------------------------------------------------------
    row_start = row_chunk * BLOCK_SIZE_BS
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)
    rows_in_block = row_start + offs_bs
    mask_bs = rows_in_block < max_rows

    base_ptr = block_idx * max_rows
    
    # Load sequential act_indices directly (no reconstruction needed)
    act_indices = tl.load(nb_maxrows_to_actidx_ptr + base_ptr + rows_in_block,
                         mask=mask_bs, other=0)
    bs_indices = tl.load(nb_maxrows_to_bs_ptr + base_ptr + rows_in_block,
                        mask=mask_bs, other=0)
    gate_vals = tl.load(nb_maxrows_gate_vals_ptr + base_ptr + rows_in_block,
                       mask=mask_bs, other=0.0)

    # Check if this block has active rows
    blk_rows = tl.load(max_rows_per_block_ptr + block_idx)
    if blk_rows == 0:
        return

    # Determine which rows are actually active based on gate values
    gate_vals = tl.where(gate_vals > 0, gate_vals, 0.0)  # Set negative gate values to zero
    row_active = gate_vals > 0.0  # bool mask per row

    # If tile is fully padded, exit early
    if tl.sum(row_active) == 0:
        return

    # ------------------------------------------------------------------
    # Column (hidden) offsets for this tile
    # ------------------------------------------------------------------
    offs_h = col_chunk * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < hidden_size

    # The LS offsets are the full [0, LS) for every iteration of the K-loop
    offs_ls = tl.arange(0, BLOCK_SIZE_LS)

    # ------------------------------------------------------------------
    # Accumulator
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_H), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main dot-product loop over line_size (K dimension)
    # ------------------------------------------------------------------
    for k in range(0, line_size, BLOCK_SIZE_LS):
        curr_offs_ls = k + offs_ls
        mask_ls = curr_offs_ls < line_size

        # x_ptr shape (act_idx, LS). Load using sparse act_indices
        x_ptrs = x_ptr + act_indices[:, None] * stride_x_actidx + curr_offs_ls[None, :] * stride_x_ls

        # weight slice: (LS, H) from current block
        w_ptrs = w_ptr + (block_idx * line_size + curr_offs_ls)[:, None] * stride_w_ls + offs_h[None, :] * stride_w_h

        x_block = tl.load(x_ptrs, mask=row_active[:, None] & mask_ls[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_ls[:, None] & mask_h[None, :], other=0.0)

        acc += tl.dot(x_block, w_block)

    # ------------------------------------------------------------------
    # Write output: atomic add (single-stage) or direct store (two-stage)
    # ------------------------------------------------------------------
    if two_stage_reduction:
        # Two-stage: store directly to (act_idx, H) without atomics
        out_ptrs = output_ptr + act_indices[:, None] * stride_out_dim0 + offs_h[None, :] * stride_out_h
        tl.store(out_ptrs, acc.to(out_dtype), mask=row_active[:, None] & mask_h[None, :])
    else:
        # Single-stage: atomic add into final (BS, H) output 
        out_ptrs = output_ptr + bs_indices[:, None] * stride_out_dim0 + offs_h[None, :] * stride_out_h
        tl.atomic_add(out_ptrs, acc.to(out_dtype), mask=row_active[:, None] & mask_h[None, :], sem="relaxed")


# =============================================================================
# Stage 2: Custom summation kernel (act_idx, H) -> (BS, H)
# =============================================================================

def get_summation_autotune_config():
    """Autotune configurations for sequential summation kernel"""
    return [
        triton.Config({'BLOCK_SIZE_H': 1024}),
        triton.Config({'BLOCK_SIZE_H': 2048}),
        triton.Config({'BLOCK_SIZE_H': 512}),
        triton.Config({'BLOCK_SIZE_H': 4096}),
        triton.Config({'BLOCK_SIZE_H': 256}),
    ]


@triton.autotune(
    configs=get_summation_autotune_config(),
    key=["batch_seq_size", "hidden_size"],
)
@triton.jit
def stream_compact_summation_kernel(
    # Input tensors
    intermediate_ptr,        # (act_idx, H) intermediate results - now sequential!
    bs_start_indices_ptr,    # (BS,) -> start act_idx for each BS row
    bs_counts_ptr,           # (BS,) -> count of active elements per BS row
    
    # Output tensor
    output_ptr,              # (BS, H) final output
    
    # Sizes
    batch_seq_size: tl.constexpr,
    hidden_size: tl.constexpr, 
    
    # Strides
    stride_inter_actidx, stride_inter_h,
    stride_out_bs, stride_out_h,
    
    # Meta-params
    out_dtype: tl.constexpr,
    
    # Block sizes from autotune
    BLOCK_SIZE_H: tl.constexpr,
):
    """Sum intermediate (act_idx, H) results to final (BS, H) output.
    
    Sequential access: Each CTA processes one BS row and a chunk of H dimension.
    Uses simple sequential range iteration for perfect memory coalescing.
    """
    
    pid = tl.program_id(0)
    
    # ------------------------------------------------------------------  
    # 1D grid indexing: map PID to (BS, H_chunk)
    # ------------------------------------------------------------------
    num_h_chunks = tl.cdiv(hidden_size, BLOCK_SIZE_H)
    
    bs_idx = pid // num_h_chunks
    h_chunk = pid % num_h_chunks
    
    # Guard against out-of-bounds grid launches
    if bs_idx >= batch_seq_size or h_chunk >= num_h_chunks:
        return
    
    # ------------------------------------------------------------------
    # Calculate this CTA's H range
    # ------------------------------------------------------------------
    h_start = h_chunk * BLOCK_SIZE_H
    offs_h = h_start + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < hidden_size
    
    # ------------------------------------------------------------------
    # Load sequential range for this BS row
    # ------------------------------------------------------------------
    start_idx = tl.load(bs_start_indices_ptr + bs_idx)
    count = tl.load(bs_counts_ptr + bs_idx)
    
    # ------------------------------------------------------------------
    # Initialize 1D accumulator for this H chunk
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    
    # ------------------------------------------------------------------
    # Simple sequential loop - perfect memory coalescing!
    # ------------------------------------------------------------------
    for i in range(count):
        act_idx = start_idx + i
        
        # Load intermediate values - sequential access pattern
        inter_ptrs = intermediate_ptr + act_idx * stride_inter_actidx + offs_h * stride_inter_h
        inter_vals = tl.load(inter_ptrs, mask=mask_h, other=0.0)
        
        # Accumulate contributions
        acc += inter_vals
    
    # ------------------------------------------------------------------
    # Store final results for this BS row
    # ------------------------------------------------------------------
    out_ptrs = output_ptr + bs_idx * stride_out_bs + offs_h * stride_out_h
    
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_h)


# =============================================================================
# Python helper
# =============================================================================

def fused_down_proj_sparse_triton_stream_compact(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    num_blocks: int,
    line_size: int,
    *,
    mappings: dict,
    out_dtype: torch.dtype = torch.float32,
    two_stage_reduction: bool = False,
):
    """Stream compact sparse helper for Cast down-projection.

    Expected input layout:
        x            : (act_idx, LS)  – fp16/bf16/fp32 sparse format from stream compact up_proj
        down_weight  : (I, H)         – fp16/bf16/fp32 (pass Linear.weight)
        mappings     : dict from create_stream_compact_index() containing:
            - nb_maxrows_to_bs: (NB, max_rows) -> BS mapping
            - nb_maxrows_to_actidx: (NB, max_rows) -> sequential act_idx mapping
            - max_rows: maximum active rows per block
            - max_rows_per_block: (NB,) active rows per block
        two_stage_reduction : bool, default False
            - False: Use atomic adds to produce final result
            - True: Generate intermediate (act_idx, H), then sum to final result (avoids atomics)

    Returns:
        output: (BS, H) down projection result
    """

    # Basic validations
    assert x.ndim == 2 and down_weight.ndim == 2, "Inputs must be 2-D tensors"

    total_act_idx, LS = x.shape
    intermediate_size, hidden_size = down_weight.shape
    
    assert LS == line_size, f"x line_size {LS} != expected {line_size}"
    assert intermediate_size == num_blocks * line_size, f"down_weight intermediate_size {intermediate_size} != num_blocks * line_size {num_blocks * line_size}"

    supported = (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in supported and down_weight.dtype in supported, f"Unsupported dtype: x={x.dtype}, down_weight={down_weight.dtype}. Supported: {supported}"

    # Extract mappings
    nb_maxrows_to_bs = mappings['nb_maxrows_to_bs']           # (NB, max_rows)
    nb_maxrows_to_actidx = mappings['nb_maxrows_to_actidx']   # (NB, max_rows) -> sequential act_idx  
    nb_maxrows_gate_vals = mappings['nb_maxrows_gate_vals']   # (NB, max_rows)
    max_rows = mappings['max_rows']
    max_rows_per_block = mappings['max_rows_per_block']       # (NB,)
    bs_start_indices = mappings['bs_start_indices']           # (BS,) -> start act_idx for each BS row
    bs_counts = mappings['bs_counts']                         # (BS,) -> count of active elements per BS row
    
    # Get batch_seq_size from bs_start_indices shape
    batch_seq_size = bs_start_indices.shape[0]
    
    # Early exit if no active elements
    if max_rows == 0 or total_act_idx == 0:
        return torch.zeros((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)

    # Assert that input tensors have efficient memory layout (at least one stride ≤ 1)
    assert min(x.stride()) <= 1, f"x has inefficient stride pattern: {x.stride()}"
    assert min(down_weight.stride()) <= 1, f"down_weight has inefficient stride pattern: {down_weight.stride()}"

    if two_stage_reduction:
        # Stage 1: Generate intermediate (act_idx, H) results without atomic adds
        output_tensor = torch.empty((total_act_idx, hidden_size), device=x.device, dtype=out_dtype)
        stride_out_dim0 = output_tensor.stride(0)
    else:
        # Original single-stage approach with atomic adds
        output_tensor = torch.zeros((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)
        stride_out_dim0 = output_tensor.stride(0)

    # Grid helper
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_H = meta["BLOCK_SIZE_H"]
        G_SIZE_R = meta["GROUP_SIZE_R"]

        num_col_chunks = triton.cdiv(hidden_size, BLK_H)
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
    fused_down_proj_stream_compact_kernel[grid](
        # Input tensors
        x,
        down_weight,
        
        # Index mappings
        nb_maxrows_to_bs,
        nb_maxrows_to_actidx,
        nb_maxrows_gate_vals,
        max_rows_per_block,
        
        # Output
        output_tensor,
        
        # Sizes
        hidden_size, line_size, max_rows,
        
        # Strides
        x.stride(0), x.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        stride_out_dim0, output_tensor.stride(1),
        
        # Meta-params
        out_dtype=triton_out_dtype,
        two_stage_reduction=two_stage_reduction,
    )

    if two_stage_reduction:
        # Stage 2: Sum intermediate results to final (BS, H) output
        final_output = torch.empty((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)
        
        # Grid for 1D summation kernel
        def summation_grid(meta):
            BLK_H = meta["BLOCK_SIZE_H"]
            num_h_chunks = triton.cdiv(hidden_size, BLK_H)
            return (batch_seq_size * num_h_chunks,)
        
        # Launch summation kernel
        stream_compact_summation_kernel[summation_grid](
            # Input tensors
            output_tensor,       # (act_idx, H) intermediate results
            bs_start_indices,    # (BS,) -> start act_idx for each BS row
            bs_counts,           # (BS,) -> count of active elements per BS row
            
            # Output tensor
            final_output,        # (BS, H) final output
            
            # Sizes
            batch_seq_size, hidden_size,
            
            # Strides
            output_tensor.stride(0), output_tensor.stride(1),        # intermediate strides
            final_output.stride(0), final_output.stride(1),         # output strides
            
            # Meta-params
            out_dtype=triton_out_dtype,
        )
        
        return final_output
    else:
        return output_tensor


if __name__ == "__main__":
    """Basic test of the stream compact down proj kernel"""
    
    print("✅ Stream compact down proj kernel loaded successfully")
    
    # Simple test with random data
    BS, NB, LS, H = 16, 8, 32, 64
    
    # Create test data in stream compact format
    gate = torch.rand(BS, NB, device="cuda", dtype=torch.float32)
    # Make gate sparse (70% zeros)
    mask = torch.rand_like(gate) < 0.7
    gate[mask] = 0.0
    
    # Create stream compact mappings
    from kernels.stream_compact_index import create_stream_compact_index
    mappings = create_stream_compact_index(gate)
    
    total_act_idx = mappings['total_act_idx']
    
    # Create sparse intermediate tensor (act_idx, LS)
    x = torch.randn(total_act_idx, LS, device="cuda", dtype=torch.float16)
    down_weight = torch.randn(NB * LS, H, device="cuda", dtype=torch.float16)
    
    print(f"Input shapes: x={x.shape}, down_weight={down_weight.shape}, gate={gate.shape}")
    print(f"Gate sparsity: {(gate == 0).float().mean().item():.1%}")
    print(f"Max rows: {mappings['max_rows']}, Total act_idx: {total_act_idx}")
    
    # Test kernel
    try:
        output = fused_down_proj_sparse_triton_stream_compact(
            x, down_weight, NB, LS, mappings=mappings
        )
        print(f"Output shape: {output.shape}")
        print("✅ Stream compact down proj test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise