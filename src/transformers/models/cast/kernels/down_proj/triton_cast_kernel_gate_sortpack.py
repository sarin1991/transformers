import torch
import triton
import triton.language as tl

# =============================================================================
# Sort-pack (ELLPACK-like) sparse down-proj kernel
#   – Uses same packing buffers (row_idx, gate_vals) produced for up-proj but
#     **does not multiply by gate_vals** (intermediate already includes gating).
#   – Each block column has exactly `max_rows` active rows. Zero-padded rows
#     are skipped cheaply. The kernel accumulates into the hidden dimension and
#     writes results via tl.atomic_add to merge contributions of multiple blocks.
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_BS": 16,
                "BLOCK_SIZE_LS": 16,
                "BLOCK_SIZE_H": 16,
                "GROUP_SIZE_R": 4,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BS": 32,
                "BLOCK_SIZE_LS": 32,
                "BLOCK_SIZE_H": 32,
                "GROUP_SIZE_R": 4,
            },
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BS": 64,
                "BLOCK_SIZE_LS": 64,
                "BLOCK_SIZE_H": 64,
                "GROUP_SIZE_R": 4,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BS": 32,
                "BLOCK_SIZE_LS": 128,
                "BLOCK_SIZE_H": 64,
                "GROUP_SIZE_R": 4,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BS": 64,
                "BLOCK_SIZE_LS": 128,
                "BLOCK_SIZE_H": 64,
                "GROUP_SIZE_R": 4,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BS": 32,
                "BLOCK_SIZE_LS": 256,
                "BLOCK_SIZE_H": 64,
                "GROUP_SIZE_R": 4,
            },
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BS": 64,
                "BLOCK_SIZE_LS": 256,
                "BLOCK_SIZE_H": 64,
                "GROUP_SIZE_R": 4,
            },
            num_warps=16,
            num_stages=3,
        ),
    ],
    key=["hidden_size", "line_size"],
    reset_to_zero=['output_ptr'],
)
@triton.jit
def fused_down_proj_sortpack_kernel(
    x_ptr, w_ptr,
    row_idx_ptr, gate_vals_ptr,        # (NB, max_rows) column-major buffers
    output_ptr,
    # sizes
    line_size: tl.constexpr, hidden_size: tl.constexpr, max_rows: tl.constexpr,
    # strides
    stride_x_bs, stride_x_ls,
    stride_w_ls, stride_w_h,
    stride_row_blk,
    stride_out_bs, stride_out_h,
    # meta
    out_dtype: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, GROUP_SIZE_R: tl.constexpr,
):
    """CTA processes tile (BLOCK_SIZE_BS rows × BLOCK_SIZE_H cols) for a given
    (block column, row chunk, hidden-chunk) triple encoded in the program id.
    Accumulates fp32 and atomically adds into the global output tensor.
    """

    pid = tl.program_id(0)

    # --------------------------------------------------------------
    # Decompose pid into (block_idx, row_chunk, col_chunk)
    # --------------------------------------------------------------
    C = (hidden_size + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H  # hidden chunks per block

    row_chunks = (max_rows + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    row_groups = (row_chunks + GROUP_SIZE_R - 1) // GROUP_SIZE_R
    num_pid_per_block = C * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem = pid % num_pid_per_block

    col_chunk = (pid_rem // GROUP_SIZE_R) % C  # hidden chunk
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group = pid_rem // (C * GROUP_SIZE_R)
    row_chunk = row_group * GROUP_SIZE_R + row_in_group

    if row_chunk >= row_chunks:
        return

    # --------------------------------------------------------------
    # Gather rows & gate values
    # --------------------------------------------------------------
    row_start = row_chunk * BLOCK_SIZE_BS
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)
    rows_in_block = row_start + offs_bs
    mask_bs = rows_in_block < max_rows

    base_ptr = block_idx * stride_row_blk  # stride_row_blk == max_rows
    row_indices = tl.load(row_idx_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0)
    gate_vals = tl.load(gate_vals_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0.0)
    row_active = gate_vals > 0.0  # bool mask per row

    # If tile is fully padded or all gate values are zero, exit early
    if tl.sum(gate_vals) == 0:
        return

    # --------------------------------------------------------------
    # Column (hidden) offsets for this tile
    # --------------------------------------------------------------
    offs_h = col_chunk * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offs_h < hidden_size

    # The LS offsets are the full [0, LS) for every iteration of the K-loop
    offs_ls = tl.arange(0, BLOCK_SIZE_LS)

    # --------------------------------------------------------------
    # Accumulator
    # --------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_H), dtype=tl.float32)

    # --------------------------------------------------------------
    # Main dot-product loop over line_size (K dimension)
    # --------------------------------------------------------------
    for k in range(0, line_size, BLOCK_SIZE_LS):
        curr_offs_ls = k + offs_ls
        mask_ls = curr_offs_ls < line_size

        # x_ptr shape (batch_seq, intermediate_size). Need LS slice inside block.
        x_ptrs = x_ptr + row_indices[:, None] * stride_x_bs + (block_idx * line_size + curr_offs_ls)[None, :] * stride_x_ls

        # weight slice: (LS, H)
        w_ptrs = w_ptr + (block_idx * line_size + curr_offs_ls)[:, None] * stride_w_ls + offs_h[None, :] * stride_w_h

        x_block = tl.load(x_ptrs, mask=row_active[:, None] & mask_ls[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_ls[:, None] & mask_h[None, :], other=0.0)

        acc += tl.dot(x_block, w_block)

    # --------------------------------------------------------------
    # Atomic add into output tensor
    # --------------------------------------------------------------
    out_ptrs = output_ptr + row_indices[:, None] * stride_out_bs + offs_h[None, :] * stride_out_h
    tl.atomic_add(out_ptrs, acc.to(out_dtype), mask=row_active[:, None] & mask_h[None, :])


# =============================================================================
# Python helper – build sort-pack buffers and launch kernel
# =============================================================================


def fused_down_proj_sparse_triton_sortpack(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    GROUP_SIZE_R: int = 4,
    out_dtype: torch.dtype = torch.float32,
):
    """Sort-pack sparse helper for Cast down-projection.

    Args:
        x:           (B, S, I)        – fp16, I = NB·LS
        down_weight: (I, H)           – fp16 (pass Linear.weight)
        gate:        (B, S, NB)       – fp32 gate tensor (values > 0 indicate active)
        num_blocks:  NB
        line_size:   LS
    """

    batch_size, seq_len, intermediate_size = x.shape
    hidden_size = down_weight.shape[1]

    assert intermediate_size == num_blocks * line_size
    assert gate.shape == (batch_size, seq_len, num_blocks)
    assert x.dtype == torch.float16 and down_weight.dtype == torch.float16

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Flatten
    x_reshaped = x.contiguous().view(-1, intermediate_size)  # (B·S, I)
    gate_reshaped = gate.contiguous().view(-1, num_blocks)    # (B·S, NB)
    batch_seq_size = x_reshaped.size(0)

    # Build packed buffers
    mask = gate_reshaped > 0
    block_counts = mask.sum(dim=0, dtype=torch.int32)
    max_rows = int(block_counts.max().item())

    if max_rows == 0:
        return torch.zeros((batch_size, seq_len, hidden_size), device=x.device, dtype=out_dtype)

    gate_vals_sorted, row_idx_sorted = torch.sort(gate_reshaped, dim=0, descending=True)
    gate_vals = gate_vals_sorted[:max_rows, :].t().contiguous()  # (NB, max_rows)
    row_idx = row_idx_sorted[:max_rows, :].t().contiguous().to(torch.int32)

    gate_vals_flat = gate_vals.view(-1)
    row_idx_flat = row_idx.view(-1)

    # Allocate output tensor – must start at zeros because kernel writes via atomic_add.
    output = torch.zeros((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)

    # Grid helper
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_H = meta["BLOCK_SIZE_H"]
        C = triton.cdiv(hidden_size, BLK_H)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, GROUP_SIZE_R)
        num_pid_per_block = C * GROUP_SIZE_R * row_groups
        return (num_pid_per_block * num_blocks,)

    fused_down_proj_sortpack_kernel[grid](
        x_reshaped,
        down_weight,
        row_idx_flat,
        gate_vals_flat,  # still needed for early-exit check
        output,
        line_size,
        hidden_size,
        max_rows,
        # strides
        x_reshaped.stride(0), x_reshaped.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        max_rows,
        output.stride(0), output.stride(1),
        out_dtype=tl.float16 if out_dtype == torch.float16 else tl.float32,
    )

    return output.view(batch_size, seq_len, hidden_size) 