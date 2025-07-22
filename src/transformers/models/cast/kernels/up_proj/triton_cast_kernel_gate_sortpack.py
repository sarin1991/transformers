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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 16, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_LS': 16, 'GROUP_SIZE_R': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_BS': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_LS': 32, 'GROUP_SIZE_R': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_LS': 64, 'GROUP_SIZE_R': 4}, num_warps=8, num_stages=2),
    ],
    key=['hidden_size', 'line_size'],
)
@triton.jit
def fused_up_proj_gate_sortpack_kernel(
    x_ptr, w_ptr, b_ptr,
    row_idx_ptr, gate_vals_ptr,        # (NB, max_rows) column-major (row-major in memory)
    output_ptr,
    # sizes
    hidden_size: tl.constexpr, line_size: tl.constexpr, max_rows: tl.constexpr,
    # strides / leading dimensions
    stride_x_bs, stride_x_h,
    stride_w_h, stride_w_i,
    stride_row_blk,                    # max_rows – distance between consecutive blocks in row_idx / gate_vals
    stride_out_bs, stride_out_i,
    # meta-params
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
    C = (line_size + BLOCK_SIZE_LS - 1) // BLOCK_SIZE_LS  # column tiles per block

    row_chunks = (max_rows + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    row_groups = (row_chunks + GROUP_SIZE_R - 1) // GROUP_SIZE_R
    num_pid_per_block = C * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem   = pid % num_pid_per_block

    col_chunk    = (pid_rem // GROUP_SIZE_R) % C
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group    = pid_rem // (C * GROUP_SIZE_R)
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

    # Early exit if this tile is fully padded
    if tl.sum(gate_vals) == 0:
        return

    # ------------------------------------------------------------------
    # Column offsets for the current tile
    # ------------------------------------------------------------------
    offs_ls   = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls   = offs_ls < line_size
    col_offset = block_idx * line_size
    global_cols = col_offset + offs_ls

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

    # Add bias and activation (ReLU)
    b_ptrs = b_ptr + global_cols
    bias   = tl.load(b_ptrs, mask=mask_ls, other=0.0)
    acc += bias[None, :]
    acc = tl.where(acc > 0, acc, 0.0)

    # Apply gate and write back
    acc *= gate_vals[:, None]

    out_ptrs = output_ptr + row_indices[:, None] * stride_out_bs + global_cols[None, :] * stride_out_i
    tl.store(out_ptrs, acc, mask=mask_bs[:, None] & mask_ls[None, :])


# =============================================================================
# Python helper – mirrors the original CSR wrapper but uses torch.sort packing
# =============================================================================

def fused_up_proj_gate_activation_sparse_triton_sortpack(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    GROUP_SIZE_R: int = 4,
    zero_init: bool = True,
):
    """Sort-pack (ELLPACK) sparse fused MLP helper.

    This is a drop-in replacement for *_csr* variants but relies on a single
    Triton kernel and a `torch.sort` pre-pass instead of a CSR writer kernel.
    """

    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Shape / dtype validation (same as the original helpers)
    assert up_weight.shape == (hidden_size, intermediate_size)
    assert gate.shape == (batch_size, seq_len, num_blocks)
    assert x.dtype == torch.float16 and up_weight.dtype == torch.float16 and up_bias.dtype == torch.float16

    # Promote gate to fp32 for better precision in sorting / multiplication
    if gate.dtype != torch.float32:
        gate = gate.float()

    # Flatten views (contiguous)
    x_reshaped    = x.contiguous().view(-1, hidden_size)        # (B*S, H)
    gate_reshaped = gate.contiguous().view(-1, num_blocks)       # (B*S, NB)
    batch_seq_size = x_reshaped.size(0)

    # ------------------------------------------------------------------
    # Build sort-packed buffers on the GPU (entirely with PyTorch ops)
    # ------------------------------------------------------------------
    mask = gate_reshaped > 0
    block_counts = mask.sum(dim=0, dtype=torch.int32)          # (NB,)
    max_rows = int(block_counts.max().item())

    # Early exit: gate is entirely zero
    if max_rows == 0:
        return torch.zeros((batch_size, seq_len, intermediate_size), device=x.device, dtype=torch.float32)

    # Sort each column in descending order – positive values first.
    gate_vals_sorted, row_idx_sorted = torch.sort(gate_reshaped, dim=0, descending=True)

    # Truncate to max_rows and transpose so that blocks are contiguous in memory
    gate_vals = gate_vals_sorted[:max_rows, :].t().contiguous()   # (NB, max_rows)
    row_idx   = row_idx_sorted[:max_rows, :].t().contiguous().to(torch.int32)  # (NB, max_rows)

    # Flatten for raw pointer access in Triton
    gate_vals_flat = gate_vals.view(-1)
    row_idx_flat   = row_idx.view(-1)

    # ------------------------------------------------------------------
    # Allocate / zero-initialise output buffer
    # ------------------------------------------------------------------
    if zero_init:
        output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=torch.float32)
    else:
        output = torch.empty((batch_seq_size, intermediate_size), device=x.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Grid size helper (same logic as CSR variant)
    # ------------------------------------------------------------------
    def grid(meta):
        BLK_BS = meta['BLOCK_SIZE_BS']
        BLK_LS = meta['BLOCK_SIZE_LS']
        C = triton.cdiv(line_size, BLK_LS)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, GROUP_SIZE_R)
        num_pid_per_block = C * GROUP_SIZE_R * row_groups
        return (num_pid_per_block * num_blocks,)

    # ------------------------------------------------------------------
    # Launch Triton kernel
    # ------------------------------------------------------------------
    fused_up_proj_gate_sortpack_kernel[grid](
        x_reshaped,
        up_weight,
        up_bias,
        row_idx_flat,
        gate_vals_flat,
        output,
        hidden_size, line_size, max_rows,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        max_rows,  # stride between blocks in row_idx / gate_vals
        output.stride(0), output.stride(1),
    )

    return output.view(batch_size, seq_len, intermediate_size) 