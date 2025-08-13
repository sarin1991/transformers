import torch
import triton
import triton.language as tl


# =============================================================================
# Fused sort-pack (ELLPACK-like) up-proj (+optional ReLU/gate) followed by
# down-proj that accumulates directly into Y via atomic adds.
#   – Reuses preprocessed sort-pack buffers: row_idx, gate_vals, block_counts, max_rows
#   – Avoids materializing the intermediate (BS, I) activation tensor
# =============================================================================


def _get_triton_autotune_config():
    """Curated configs reused from existing kernels for broad coverage."""
    return [
        # 1) Square / large tiles
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R': 16},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=32, num_stages=1),

        # 2) One skinny dim
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=2),

        # 3) Two skinny dims
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H': 128,
                       'GROUP_SIZE_R':  4},  num_warps=4,  num_stages=2),

        # 4) Fallback / edge cases
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  4},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE_BS':  64, 'BLOCK_SIZE_LS':  64, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R':  1},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_LS': 128, 'BLOCK_SIZE_H':  64,
                       'GROUP_SIZE_R': 16},  num_warps=16, num_stages=2),
    ]


@triton.autotune(
    configs=_get_triton_autotune_config(),
    key=['hidden_size', 'line_size', 'apply_gate', 'apply_relu', 'save_up_proj'],
    reset_to_zero=['y_ptr'],
)
@triton.jit
def fused_up_down_proj_sortpack_kernel(
    x_ptr, w_up_ptr, w_down_ptr,
    row_idx_ptr, gate_vals_ptr,        # (NB, max_rows) column-major buffers
    y_ptr, up_proj_ptr,                # y_ptr is output; up_proj_ptr optional
    # sizes
    hidden_size: tl.constexpr, line_size: tl.constexpr, max_rows: tl.constexpr,
    # strides
    stride_x_bs, stride_x_h,
    stride_wup_h, stride_wup_i,
    stride_wd_ls, stride_wd_h,
    stride_row_blk,
    stride_y_bs, stride_y_h,
    stride_up_bs, stride_up_i,
    # meta
    out_dtype: tl.constexpr,
    apply_gate: tl.constexpr,
    apply_relu: tl.constexpr,
    save_up_proj: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_LS: tl.constexpr, GROUP_SIZE_R: tl.constexpr,
):
    """Each CTA computes a tile (BS_tile x LS_tile) of the up-proj result for a given
    block column and row-chunk, then immediately multiplies with W_down in chunks of H
    and atomically accumulates into Y.
    """

    pid = tl.program_id(0)

    # Decompose pid into (block_idx, row_chunk, col_chunk)
    num_col_chunks = tl.cdiv(line_size, BLOCK_SIZE_LS)

    row_chunks = tl.cdiv(max_rows, BLOCK_SIZE_BS)
    row_groups = tl.cdiv(row_chunks, GROUP_SIZE_R)
    num_pid_per_block = num_col_chunks * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem = pid % num_pid_per_block

    col_chunk = (pid_rem // GROUP_SIZE_R) % num_col_chunks
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group = pid_rem // (num_col_chunks * GROUP_SIZE_R)
    row_chunk = row_group * GROUP_SIZE_R + row_in_group

    if row_chunk >= row_chunks:
        return

    # Row gather
    row_start = row_chunk * BLOCK_SIZE_BS
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)
    rows_in_block = row_start + offs_bs
    mask_bs = rows_in_block < max_rows

    base_ptr = block_idx * stride_row_blk  # == max_rows
    row_indices = tl.load(row_idx_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0)
    gate_vals = tl.load(gate_vals_ptr + base_ptr + rows_in_block, mask=mask_bs, other=0.0)
    gate_vals = tl.where(gate_vals > 0, gate_vals, 0.0)

    # Column offsets for current LS tile within block
    offs_ls = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls = offs_ls < line_size
    col_offset = block_idx * line_size
    global_cols = col_offset + offs_ls

    # Early exit if tile fully padded
    if tl.sum(gate_vals) == 0:
        return

    # Accumulator for up-proj tile (BS_tile, LS_tile)
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)

    # Main dot over H for up-proj
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size

        x_ptrs = x_ptr + row_indices[:, None] * stride_x_bs + curr_offs_h[None, :] * stride_x_h
        wup_ptrs = w_up_ptr + curr_offs_h[:, None] * stride_wup_h + global_cols[None, :] * stride_wup_i

        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_h[None, :], other=0.0)
        wup_block = tl.load(wup_ptrs, mask=mask_h[:, None] & mask_ls[None, :], other=0.0)
        acc += tl.dot(x_block, wup_block)

    if apply_relu:
        acc = tl.where(acc > 0, acc, 0.0)

    if save_up_proj:
        up_ptrs = up_proj_ptr + row_indices[:, None] * stride_up_bs + global_cols[None, :] * stride_up_i
        tl.store(up_ptrs, acc.to(out_dtype), mask=mask_bs[:, None] & mask_ls[None, :])

    if apply_gate:
        acc *= gate_vals[:, None]

    # Down-proj accumulation into Y in chunks of H
    for h_out in range(0, hidden_size, BLOCK_SIZE_H):
        offs_h2 = h_out + tl.arange(0, BLOCK_SIZE_H)
        mask_h2 = offs_h2 < hidden_size

        wd_ptrs = w_down_ptr + global_cols[:, None] * stride_wd_ls + offs_h2[None, :] * stride_wd_h
        wdown_block = tl.load(wd_ptrs, mask=mask_ls[:, None] & mask_h2[None, :], other=0.0)

        y_tile_out = tl.dot(acc, wdown_block)  # (BS_tile, H_tile)

        y_ptrs = y_ptr + row_indices[:, None] * stride_y_bs + offs_h2[None, :] * stride_y_h
        tl.atomic_add(y_ptrs, y_tile_out.to(out_dtype), mask=mask_bs[:, None] & mask_h2[None, :], sem="relaxed")


def _grid_fn(num_blocks: int, max_rows: int):
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_LS = meta["BLOCK_SIZE_LS"]
        G_SIZE_R = meta["GROUP_SIZE_R"]

        num_col_chunks = triton.cdiv(meta["line_size"], BLK_LS) if "line_size" in meta else 1
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, G_SIZE_R)
        num_pid_per_block = num_col_chunks * G_SIZE_R * row_groups
        return (num_pid_per_block * num_blocks,)

    return grid


def fused_up_down_proj_sparse_triton_sortpack(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    out_dtype: torch.dtype = torch.float32,
    apply_gate: bool = True,
    apply_relu: bool = True,
    # Preprocessed sort-pack buffers
    gate_vals: torch.Tensor | None = None,
    row_idx: torch.Tensor | None = None,
    block_counts: torch.Tensor | None = None,
    max_rows: int | None = None,
    # Debugging/inspection
    save_up_proj: bool = False,
):
    """Fused sort-pack up+down projection helper.

    Args:
        x:           (BS, H)
        up_weight:   (H, I) where I = NB * LS
        down_weight: (I, H)
        gate:        (BS, NB)
        num_blocks:  NB
        line_size:   LS
        gate_vals, row_idx, block_counts, max_rows: precomputed sort-pack buffers
        out_dtype:   torch.float16/bfloat16/float32 output dtype
        apply_gate:  whether to multiply by gate after activation
        apply_relu:  whether to apply ReLU to up-proj before gating
        save_up_proj: if True, stores the up-proj tile (post-activation, pre-gate) into a buffer
    """

    batch_seq_size, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    assert up_weight.shape == (hidden_size, intermediate_size)
    assert down_weight.shape == (intermediate_size, hidden_size)
    assert gate.shape == (batch_seq_size, num_blocks)

    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert (
        x.dtype in supported_dtypes and up_weight.dtype in supported_dtypes and down_weight.dtype in supported_dtypes
    ), f"Unsupported dtype: x={x.dtype}, up_weight={up_weight.dtype}, down_weight={down_weight.dtype}. Supported: {supported_dtypes}"

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Assert reasonable strides
    assert min(x.stride()) <= 1, f"x has inefficient stride pattern: {x.stride()}"
    assert min(up_weight.stride()) <= 1, f"up_weight has inefficient stride pattern: {up_weight.stride()}"
    assert min(down_weight.stride()) <= 1, f"down_weight has inefficient stride pattern: {down_weight.stride()}"
    x_reshaped = x

    # Require preprocessed buffers
    if gate_vals is None or row_idx is None or block_counts is None or max_rows is None:
        raise ValueError("gate_vals, row_idx, block_counts, and max_rows must all be provided")

    if max_rows == 0:
        if save_up_proj:
            up_proj_zero = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)
            y_zero = torch.zeros((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)
            return y_zero, up_proj_zero
        else:
            return torch.zeros((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)

    # Flatten pack buffers for raw pointer access
    gate_vals_flat = gate_vals.view(-1)
    row_idx_flat = row_idx.view(-1)

    # Output Y must be zero-initialized because kernel uses atomic_add
    y = torch.zeros((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)

    # Optional up-proj buffer
    if save_up_proj:
        up_proj_out = torch.empty((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)
        up_proj_ptr = up_proj_out
        stride_up_bs, stride_up_i = up_proj_out.stride()
    else:
        up_proj_out = None
        up_proj_ptr = y  # dummy
        stride_up_bs, stride_up_i = 0, 0

    # dtype mapping
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    # Grid function needs knowledge of NB and max_rows; line_size is captured by autotune meta
    def grid(meta):
        BLK_BS = meta["BLOCK_SIZE_BS"]
        BLK_LS = meta["BLOCK_SIZE_LS"]
        G_SIZE_R = meta["GROUP_SIZE_R"]
        num_col_chunks = triton.cdiv(line_size, BLK_LS)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, G_SIZE_R)
        num_pid_per_block = num_col_chunks * G_SIZE_R * row_groups
        return (num_pid_per_block * num_blocks,)

    fused_up_down_proj_sortpack_kernel[grid](
        x_reshaped,
        up_weight,
        down_weight,
        row_idx_flat,
        gate_vals_flat,
        y,
        up_proj_ptr,
        hidden_size,
        line_size,
        max_rows,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        max_rows,
        y.stride(0), y.stride(1),
        stride_up_bs, stride_up_i,
        out_dtype=triton_out_dtype,
        apply_gate=apply_gate,
        apply_relu=apply_relu,
        save_up_proj=save_up_proj,
    )

    if save_up_proj:
        return y, up_proj_out
    else:
        return y


