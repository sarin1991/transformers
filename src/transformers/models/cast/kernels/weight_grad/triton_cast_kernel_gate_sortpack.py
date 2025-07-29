import torch
import triton
import triton.language as tl

# =============================================================================
# Sort-pack (ELLPACK-like) sparse weight-gradient kernel
#  - Computes dW = intermediateᵀ · other for a single MLP block column.
#  - Row packing identical to down_proj and up_proj sort-pack helpers.
# =============================================================================

# ----------------------------- autotune configs -----------------------------
_CONFIG_WARPS  = (4, 8, 16)
_TILE_SIZES    = (64, 128)
_NUM_STAGES    = (1, 2, 3)

CONFIGS = []
for tile in _TILE_SIZES:
    for warps in _CONFIG_WARPS:
        for stages in _NUM_STAGES:
            CONFIGS.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_BS": tile,
                        "BLOCK_SIZE_LS": tile,
                        "BLOCK_SIZE_H" : tile,
                        "GROUP_SIZE_R" : 16,
                    },
                    num_warps=warps,
                    num_stages=stages,
                )
            )


@triton.autotune(
    configs=CONFIGS,
    key=["hidden_size", "line_size"],
)
@triton.jit
def fused_weight_grad_sortpack_kernel(
    inter_ptr, other_ptr,
    row_idx_ptr,               # (NB, max_rows)
    row_counts_ptr,            # (NB,) – number of active rows per block
    output_ptr,                # (I, H) – fp32/fp16
    # sizes
    hidden_size: tl.constexpr, line_size: tl.constexpr, max_rows: tl.constexpr,
    # strides
    stride_inter_bs, stride_inter_ls,
    stride_other_bs, stride_other_h,
    stride_row_blk,
    stride_out_i, stride_out_h,
    # meta
    out_dtype: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, GROUP_SIZE_R: tl.constexpr,
):
    """CTA computes one (LS × H) tile of dW for a block column.

    Each CTA owns its output tile exclusively and therefore stores its result
    with tl.store (no atomic add). The full reduction over the packed row list
    is executed inside the CTA.
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

    # Pointer into the packed row index buffer (column-major, max_rows per block)
    base_row_ptr = block_idx * stride_row_blk  # stride_row_blk == max_rows

    # ------------------------------------------------------------------
    # Accumulator for this (LS × H) tile
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_LS, BLOCK_SIZE_H), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main reduction loop over packed rows
    # ------------------------------------------------------------------
    blk_rows = tl.load(row_counts_ptr + block_idx)
    # Early exit if block has no active rows
    if blk_rows == 0:
        return

    for r in range(0, max_rows, BLOCK_SIZE_BS):
        row_offs   = r + offs_bs
        # Mask rows using **per-block** active count instead of global max_rows
        mask_rows  = row_offs < blk_rows
        # Skip this tile entirely if it contains no active rows
        if tl.sum(mask_rows) == 0:
            continue

        row_idx = tl.load(row_idx_ptr + base_row_ptr + row_offs,
                          mask=mask_rows, other=0)

        # Load slice from intermediate → shape (BS, LS)
        inter_ptrs_t = (
            inter_ptr
            + row_idx[None, :] * stride_inter_bs
            + global_cols[:, None] * stride_inter_ls
        )
        inter_blk_t = tl.load(inter_ptrs_t,
                            mask=mask_rows[None, :] & mask_ls[:, None],
                            other=0.0)

        # Load slice from other → shape (BS, H)
        other_ptrs = (
            other_ptr
            + row_idx[:, None] * stride_other_bs
            + h_global[None, :] * stride_other_h
        )
        other_blk = tl.load(other_ptrs,
                            mask=mask_rows[:, None] & mask_h[None, :],
                            other=0.0)

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

def fused_weight_grad_sparse_triton_sortpack(
    intermediate: torch.Tensor,
    other: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    GROUP_SIZE_R: int = 4,
    out_dtype: torch.dtype = torch.float32,
):
    """Sort-pack sparse helper to compute dW = intermediateᵀ · other.

    Expected input layout (flattened batch-sequence):
        intermediate : (BS, I)  – fp16/bf16 with I = NB·LS (already gated)
        other        : (BS, H)  – fp16/bf16
        gate         : (BS, NB) – fp32 mask (non-zero → active)

    where BS = batch_size × sequence_length.
    """

    # Basic validations
    assert intermediate.ndim == 2 and other.ndim == 2 and gate.ndim == 2, "Inputs must be 2-D (batch_seq, …) tensors"

    batch_seq, I = intermediate.shape
    hidden_size = other.shape[1]

    assert other.shape == (batch_seq, hidden_size)
    assert gate.shape == (batch_seq, num_blocks)
    assert I == num_blocks * line_size, "line_size must divide intermediate size"

    supported = (torch.float16, torch.bfloat16)
    assert intermediate.dtype in supported and other.dtype in supported

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Contiguous views (ensure memory stride is compact)
    inter_flat = intermediate.contiguous()
    other_flat = other.contiguous()
    gate_flat  = gate.contiguous()

    # Build packed buffers
    mask = gate_flat > 0
    block_counts = mask.sum(dim=0, dtype=torch.int32)
    max_rows = int(block_counts.max().item())
    if max_rows == 0:
        return torch.zeros((I, hidden_size), device=intermediate.device, dtype=out_dtype)

    _, row_idx_sorted = torch.sort(gate_flat, dim=0, descending=True)
    row_idx = row_idx_sorted[:max_rows, :].t().contiguous().to(torch.int32)  # (NB, max_rows)

    row_idx_flat = row_idx.view(-1)

    # Output tensor
    output = torch.empty((I, hidden_size), device=intermediate.device, dtype=out_dtype)

    # Strides
    stride_inter_bs, stride_inter_ls = inter_flat.stride()
    stride_other_bs, stride_other_h = other_flat.stride()

    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_LS = meta["BLOCK_SIZE_LS"]
        BLK_H  = meta["BLOCK_SIZE_H"]
        G_R    = meta["GROUP_SIZE_R"]

        hidden_chunks = triton.cdiv(hidden_size, BLK_H)
        ls_groups     = triton.cdiv(line_size, G_R * BLK_LS)
        num_pid_per_block = hidden_chunks * ls_groups * G_R
        return (num_pid_per_block * num_blocks,)

    fused_weight_grad_sortpack_kernel[grid](
        inter_flat,
        other_flat,
        row_idx_flat,
        block_counts.contiguous(),
        output,
        hidden_size,
        line_size,
        max_rows,
        stride_inter_bs, stride_inter_ls,
        stride_other_bs, stride_other_h,
        max_rows,
        output.stride(0), output.stride(1),
        out_dtype=tl.float16 if out_dtype == torch.float16 else tl.float32,
    )

    return output 