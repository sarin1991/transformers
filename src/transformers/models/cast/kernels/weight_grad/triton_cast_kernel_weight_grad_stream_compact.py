import torch
import triton
import triton.language as tl

# =============================================================================
# Stream compact sparse weight-gradient kernel
#  - Computes dW = intermediateᵀ · other for a single MLP block column.
#  - Uses stream compact format: intermediate is (act_idx, LS), not (BS, I).
#  - Leverages existing stream compact index mappings for efficient sparse access.
# =============================================================================

# ----------------------------- autotune configs -----------------------------
def get_triton_autotune_config():
    """
    Autotune configurations adapted from sort pack weight grad kernel.
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
    key=["hidden_size", "line_size"],
)
@triton.jit
def fused_weight_grad_stream_compact_kernel(
    # Input tensors
    intermediate_ptr,               # (act_idx, LS) sparse intermediate
    other_ptr,                      # (BS, H) other tensor
    
    # Index mappings from stream compact preprocessing  
    nb_maxrows_to_bs_ptr,           # (NB, max_rows) -> BS index
    nb_maxrows_to_actidx_ptr,       # (NB, max_rows) -> sequential act_idx
    max_rows_per_block_ptr,         # (NB,) number of active rows per block
    
    # Output tensor
    output_ptr,                     # (I, H) weight gradients
    
    # Sizes
    hidden_size: tl.constexpr, 
    line_size: tl.constexpr, 
    max_rows: tl.constexpr,
    
    # Strides
    stride_inter_actidx, stride_inter_ls,
    stride_other_bs, stride_other_h,
    stride_out_i, stride_out_h,
    
    # Meta-params
    out_dtype: tl.constexpr,
    
    # Block sizes from autotune
    BLOCK_SIZE_BS: tl.constexpr, 
    BLOCK_SIZE_LS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, 
    GROUP_SIZE_R: tl.constexpr,
):
    """CTA computes one (LS × H) tile of dW for a block column.

    Each CTA owns its output tile exclusively and therefore stores its result
    with tl.store (no atomic add). The full reduction over the packed row list
    is executed inside the CTA.
    
    Key difference from sort pack: intermediate input is in sparse (act_idx, LS) format.
    """

    pid = tl.program_id(0)

    # ------------------------------------------------------------------
    # Decompose program-id → (block_idx, ls_chunk_in_block, h_chunk)
    # ------------------------------------------------------------------
    num_hidden_chunks = tl.cdiv(hidden_size, BLOCK_SIZE_H)
    num_ls_groups     = tl.cdiv(line_size, GROUP_SIZE_R * BLOCK_SIZE_LS)
    ls_chunks_per_blk = num_ls_groups * GROUP_SIZE_R
    num_pid_per_block = num_hidden_chunks * ls_chunks_per_blk

    block_idx  = pid // num_pid_per_block
    pid_in_blk = pid %  num_pid_per_block

    ls_group        = pid_in_blk // (num_hidden_chunks * GROUP_SIZE_R)
    pid_in_group    = pid_in_blk %  (num_hidden_chunks * GROUP_SIZE_R)

    h_chunk         = pid_in_group // GROUP_SIZE_R
    ls_chunk_in_grp = pid_in_group %  GROUP_SIZE_R
    ls_chunk_in_blk = ls_group * GROUP_SIZE_R + ls_chunk_in_grp

    # Guard CTAs that map outside of the valid LS/H ranges
    if (h_chunk * BLOCK_SIZE_H >= hidden_size) or (
        ls_chunk_in_blk * BLOCK_SIZE_LS >= line_size):
        return

    # ------------------------------------------------------------------
    # Sequential act_idx layout - no block offset needed
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    offs_ls = tl.arange(0, BLOCK_SIZE_LS)
    offs_h  = tl.arange(0, BLOCK_SIZE_H)
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)

    ls_global  = ls_chunk_in_blk * BLOCK_SIZE_LS + offs_ls
    h_global   = h_chunk * BLOCK_SIZE_H + offs_h

    mask_ls = ls_global < line_size
    mask_h  = h_global  < hidden_size

    col_offset  = block_idx * line_size
    global_cols = col_offset + ls_global

    # Pointer into the stream compact index mappings
    base_ptr = block_idx * max_rows

    # ------------------------------------------------------------------
    # Accumulator for this (LS × H) tile
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_LS, BLOCK_SIZE_H), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main reduction loop over packed rows
    # ------------------------------------------------------------------
    blk_rows = tl.load(max_rows_per_block_ptr + block_idx)
    # If block has no active rows, store zeros and return
    if blk_rows == 0:
        out_ptrs = (
            output_ptr
            + global_cols[:, None] * stride_out_i
            + h_global[None, :]   * stride_out_h
        )
        tl.store(out_ptrs, acc.to(out_dtype), mask=mask_ls[:, None] & mask_h[None, :])
        return

    for r in range(0, blk_rows, BLOCK_SIZE_BS):
        row_offs   = r + offs_bs
        # Mask rows using **per-block** active count instead of global max_rows
        mask_rows  = row_offs < blk_rows

        # Load sequential act_indices directly (no reconstruction needed)
        act_indices = tl.load(nb_maxrows_to_actidx_ptr + base_ptr + row_offs,
                             mask=mask_rows, other=0)
        bs_indices = tl.load(nb_maxrows_to_bs_ptr + base_ptr + row_offs,
                            mask=mask_rows, other=0)

        # Load slice from intermediate → shape (LS, BS) - transposed loading
        inter_ptrs_t = (
            intermediate_ptr
            + ls_global[:, None] * stride_inter_ls
            + act_indices[None, :] * stride_inter_actidx
        )
        inter_blk_t = tl.load(inter_ptrs_t,
                            mask=mask_ls[:, None] & mask_rows[None, :],
                            other=0.0)

        # Load slice from other → shape (BS, H)
        other_ptrs = (
            other_ptr
            + bs_indices[:, None] * stride_other_bs
            + h_global[None, :] * stride_other_h
        )
        other_blk = tl.load(other_ptrs,
                            mask=mask_rows[:, None] & mask_h[None, :],
                            other=0.0)

        # Dot product: (LS, BS) @ (BS, H) = (LS, H)
        acc += tl.dot(inter_blk_t, other_blk)

    # ------------------------------------------------------------------
    # Write back (no atomics required)
    # ------------------------------------------------------------------
    out_ptrs = (
        output_ptr
        + global_cols[:, None] * stride_out_i
        + h_global[None, :]   * stride_out_h
    )
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_ls[:, None] & mask_h[None, :])


# =============================================================================
# Python helper
# =============================================================================

def fused_weight_grad_sparse_triton_stream_compact(
    intermediate: torch.Tensor,
    other: torch.Tensor,
    num_blocks: int,
    line_size: int,
    *,
    mappings: dict,
    out_dtype: torch.dtype = torch.float32,
):
    """Stream compact sparse helper to compute dW = intermediateᵀ · other.

    Expected input layout:
        intermediate : (act_idx, LS)  – fp16/bf16/fp32 sparse format from stream compact up_proj
        other        : (BS, H)        – fp16/bf16/fp32
        mappings     : dict from create_stream_compact_index() containing:
            - nb_maxrows_to_bs: (NB, max_rows) -> BS mapping
            - nb_maxrows_to_actidx: (NB, max_rows) -> sequential act_idx mapping
            - max_rows: maximum active rows per block
            - max_rows_per_block: (NB,) active rows per block

    Returns:
        dW: (I, H) weight gradients where I = num_blocks * line_size
    """

    # Basic validations
    assert intermediate.ndim == 2 and other.ndim == 2, "Inputs must be 2-D tensors"

    total_act_idx, LS = intermediate.shape
    batch_seq, hidden_size = other.shape
    
    assert other.shape == (batch_seq, hidden_size)
    assert LS == line_size, f"intermediate line_size {LS} != expected {line_size}"

    supported = (torch.float16, torch.bfloat16, torch.float32)
    assert intermediate.dtype in supported and other.dtype in supported, f"Unsupported dtype: intermediate={intermediate.dtype}, other={other.dtype}. Supported: {supported}"

    # Extract mappings
    nb_maxrows_to_bs = mappings['nb_maxrows_to_bs']           # (NB, max_rows)
    nb_maxrows_to_actidx = mappings['nb_maxrows_to_actidx']   # (NB, max_rows) -> sequential act_idx
    max_rows = mappings['max_rows']
    max_rows_per_block = mappings['max_rows_per_block']       # (NB,)
    
    # Early exit if no active elements
    if max_rows == 0 or total_act_idx == 0:
        intermediate_size = num_blocks * line_size
        return torch.zeros((intermediate_size, hidden_size), device=intermediate.device, dtype=out_dtype)

    # Assert that input tensors have efficient memory layout (at least one stride ≤ 1)
    assert min(intermediate.stride()) <= 1, f"intermediate has inefficient stride pattern: {intermediate.stride()}"
    assert min(other.stride()) <= 1, f"other has inefficient stride pattern: {other.stride()}"

    # Output tensor
    intermediate_size = num_blocks * line_size
    output = torch.empty((intermediate_size, hidden_size), device=intermediate.device, dtype=out_dtype)

    # Grid calculation - same as sort pack
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_LS = meta["BLOCK_SIZE_LS"]
        BLK_H  = meta["BLOCK_SIZE_H"]
        G_R    = meta["GROUP_SIZE_R"]

        hidden_chunks = triton.cdiv(hidden_size, BLK_H)
        ls_groups     = triton.cdiv(line_size, G_R * BLK_LS)
        num_pid_per_block = hidden_chunks * ls_groups * G_R
        return (num_pid_per_block * num_blocks,)

    # Map torch dtypes to triton dtypes
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16, 
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    # Launch kernel
    fused_weight_grad_stream_compact_kernel[grid](
        # Input tensors
        intermediate,
        other,
        
        # Index mappings
        nb_maxrows_to_bs,
        nb_maxrows_to_actidx,
        max_rows_per_block,
        
        # Output
        output,
        
        # Sizes
        hidden_size, line_size, max_rows,
        
        # Strides
        intermediate.stride(0), intermediate.stride(1),
        other.stride(0), other.stride(1),
        output.stride(0), output.stride(1),
        
        # Meta-params
        out_dtype=triton_out_dtype,
    )

    return output


if __name__ == "__main__":
    """Basic test of the stream compact weight grad kernel"""
    
    print("✅ Stream compact weight grad kernel loaded successfully")
    
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
    intermediate = torch.randn(total_act_idx, LS, device="cuda", dtype=torch.float16)
    other = torch.randn(BS, H, device="cuda", dtype=torch.float16)
    
    print(f"Input shapes: intermediate={intermediate.shape}, other={other.shape}, gate={gate.shape}")
    print(f"Gate sparsity: {(gate == 0).float().mean().item():.1%}")
    print(f"Max rows: {mappings['max_rows']}, Total act_idx: {total_act_idx}")
    
    # Test kernel
    try:
        output = fused_weight_grad_sparse_triton_stream_compact(
            intermediate, other, NB, LS, mappings=mappings
        )
        print(f"Output shape: {output.shape}")
        print("✅ Stream compact weight grad test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise