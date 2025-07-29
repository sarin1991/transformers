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
    reset_to_zero=["output_ptr"],
)
@triton.jit
def fused_weight_grad_sortpack_kernel(
    inter_ptr, other_ptr,
    row_idx_ptr,               # (NB, max_rows)
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
    """CTA computes (LS × Htile) slice of dW for one block column."""

    pid = tl.program_id(0)

    # Decompose pid → (block_idx, row_chunk, col_chunk)
    num_col_chunks = tl.cdiv(hidden_size, BLOCK_SIZE_H)

    row_chunks  = tl.cdiv(max_rows, BLOCK_SIZE_BS)
    row_groups  = tl.cdiv(row_chunks, GROUP_SIZE_R)

    num_pid_per_block = num_col_chunks * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem   = pid %  num_pid_per_block

    # --- indices inside the row-group ---
    pid_row_group = pid_rem % (num_col_chunks * GROUP_SIZE_R)
    row_in_group  = pid_row_group % GROUP_SIZE_R
    col_chunk     = pid_row_group // GROUP_SIZE_R  # 0 … num_col_chunks-1

    row_group = pid_rem // (num_col_chunks * GROUP_SIZE_R)
    row_chunk = row_group * GROUP_SIZE_R + row_in_group

    if row_chunk >= row_chunks:
        return

    # ----- gather packed rows -----
    row_start = row_chunk * BLOCK_SIZE_BS
    offs_bs   = tl.arange(0, BLOCK_SIZE_BS)
    rows_in_block = row_start + offs_bs
    mask_bs   = rows_in_block < max_rows

    base_ptr  = block_idx * stride_row_blk  # stride_row_blk == max_rows
    row_indices = tl.load(row_idx_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0)

    # Early exit if tile is fully padded.
    if tl.sum(row_indices) == 0:
        return

    # ----- column offsets -----
    offs_ls = col_offs = tl.arange(0, BLOCK_SIZE_LS)
    col_chunk_offset = block_idx * line_size
    global_cols = col_chunk_offset + col_offs + col_chunk * 0  # placeholder

    # Create proper offs_ls considering col_chunk
    offs_ls = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls = offs_ls < line_size
    global_cols = col_chunk_offset + offs_ls

    # ----- accumulator -----
    acc = tl.zeros((BLOCK_SIZE_LS, BLOCK_SIZE_H), dtype=tl.float32)

    # K-loop over rows in BS tiles
    for b in range(0, BLOCK_SIZE_BS):
        row_mask = mask_bs[b]
        row_idx  = row_indices[b]
        if not row_mask:
            continue

    # We will iterate over hidden_size in H-chunks
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size

        # ---- load inter_block: shape (LS, 1) for current row subset ----
        # We accumulate over rows, so use tl.dot later with transpose.
        inter_ptrs = inter_ptr + row_indices[:, None] * stride_inter_bs + (col_chunk_offset + offs_ls)[None, :] * stride_inter_ls
        inter_block = tl.load(inter_ptrs, mask=mask_bs[:, None] & mask_ls[None, :], other=0.0)  # (BS, LS)

        other_ptrs = other_ptr + row_indices[:, None] * stride_other_bs + curr_offs_h[None, :] * stride_other_h
        other_block = tl.load(other_ptrs, mask=mask_bs[:, None] & mask_h[None, :], other=0.0)   # (BS, Htile)

        acc += tl.dot(tl.trans(inter_block), other_block)  # (LS, Htile)

    # ----- write back (atomic add) -----
    out_ptrs = output_ptr + global_cols[:, None] * stride_out_i + offs_h[None, :] * 0  # we write later per H tile
    off_h_start = 0
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size
        tile = acc[:, :]
        out_tile_ptrs = output_ptr + global_cols[:, None] * stride_out_i + curr_offs_h[None, :] * stride_out_h
        tl.atomic_add(out_tile_ptrs, tile.to(out_dtype), mask=mask_ls[:, None] & mask_h[None, :])


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

    intermediate : (B, S, I) fp16/bf16 with I = NB·LS (already gated)
    other        : (B, S, H) fp16/bf16
    gate         : (B, S, NB) fp32 – same sparsity pattern as forward
    """

    B, S, I = intermediate.shape
    hidden_size = other.shape[2]
    batch_seq = B * S

    assert I == num_blocks * line_size
    assert gate.shape == (B, S, num_blocks)

    supported = (torch.float16, torch.bfloat16)
    assert intermediate.dtype in supported and other.dtype in supported

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Flatten views
    inter_flat = intermediate.contiguous().view(batch_seq, I)
    other_flat = other.contiguous().view(batch_seq, hidden_size)
    gate_flat  = gate.contiguous().view(batch_seq, num_blocks)

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
    output = torch.zeros((I, hidden_size), device=intermediate.device, dtype=out_dtype)

    # Strides
    stride_inter_bs, stride_inter_ls = inter_flat.stride()
    stride_other_bs, stride_other_h = other_flat.stride()

    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_H  = meta["BLOCK_SIZE_H"]
        G_R    = meta["GROUP_SIZE_R"]
        C = triton.cdiv(hidden_size, BLK_H)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, G_R)
        num_pid_per_block = C * G_R * row_groups
        return (num_pid_per_block * num_blocks,)

    fused_weight_grad_sortpack_kernel[grid](
        inter_flat,
        other_flat,
        row_idx_flat,
        output,
        line_size,
        hidden_size,
        max_rows,
        stride_inter_bs, stride_inter_ls,
        stride_other_bs, stride_other_h,
        max_rows,
        output.stride(0), output.stride(1),
        out_dtype=tl.float16 if out_dtype == torch.float16 else tl.float32,
    )

    return output 