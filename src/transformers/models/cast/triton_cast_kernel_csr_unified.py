import torch
import triton
import triton.language as tl

# =============================================================================
# Unified CSR builder: counts and writes on GPU (two parts)
#   1. PyTorch ops (mask, per-block counts, cumsum)
#   2. Triton kernel that writes rows_index & gate_values with per-block atomic
#
# CTA tiles:  BLOCK_SIZE_BS rows  ×  BLOCK_SIZE_NB blocks
# Grid dims:  (row_chunks, block_chunks)
# =============================================================================


# Wider NB tiles for better vectorised loads
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_NB': 8},  num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 256, 'BLOCK_SIZE_NB': 8},  num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 128, 'BLOCK_SIZE_NB': 16}, num_warps=8, num_stages=2),
    ],
    key=['BS', 'NB'],
)
@triton.jit
def _csr_writer_kernel(
    gate_ptr,                 # (BS, NB) fp32
    rows_index_ptr,           # (N,)     int32  – output
    gates_val_ptr,            # (N,)     fp32   – output
    block_offsets_ptr,        # (NB+1,)  int32  – CSR starts (exclusive prefix)
    block_write_ptrs_ptr,     # (NB,)    int32  – per-block atomic cursor
    BS: tl.constexpr, NB: tl.constexpr,
    stride_g_bs, stride_g_nb,  # gate strides
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_NB: tl.constexpr,
):
    """Each CTA handles a tile of size (BLOCK_SIZE_BS rows × BLOCK_SIZE_NB blocks).
    It writes all non-zero (row, block) pairs for its tile into the global CSR
    buffers, using one atomicAdd per block to reserve a contiguous slice.
    """
    pid_row = tl.program_id(0)  # fast-varying – rows
    pid_blk = tl.program_id(1)  # slow-varying – blocks

    row_start = pid_row * BLOCK_SIZE_BS
    blk_start = pid_blk * BLOCK_SIZE_NB

    # Offsets within the tile
    offs_bs = row_start + tl.arange(0, BLOCK_SIZE_BS)
    offs_nb = blk_start + tl.arange(0, BLOCK_SIZE_NB)

    mask_bs = offs_bs < BS
    mask_nb = offs_nb < NB

    # ---------------- Vectorised per-tile processing -------------------
    g_tile_ptrs = gate_ptr + offs_bs[:, None] * stride_g_bs + offs_nb[None, :] * stride_g_nb
    gate_tile = tl.load(g_tile_ptrs, mask=mask_bs[:, None] & mask_nb[None, :], other=0.0)  # (rows, nb)

    g_mask = gate_tile != 0                                # (rows, nb)
    nnz = tl.sum(g_mask, axis=0)                   # (nb,)

    # Reserve slices atomically per block (vectorised)
    tile_offsets = tl.atomic_add(block_write_ptrs_ptr + offs_nb, nnz, mask=mask_nb)

    block_bases = tl.load(block_offsets_ptr + offs_nb, mask=mask_nb, other=0)  # (nb,)
    write_bases = block_bases + tile_offsets                                    # (nb,)

    # Prefix inside each column
    prefix = tl.cumsum(g_mask.to(tl.int32), axis=0) - 1    # (rows, nb)

    row_vals = offs_bs[:, None]                            # broadcast
    tl.store(rows_index_ptr + write_bases[None, :] + prefix, row_vals, mask=g_mask)
    tl.store(gates_val_ptr + write_bases[None, :] + prefix, gate_tile,  mask=g_mask)


# =============================================================================
# Python helper
# =============================================================================

def build_csr_buffers_triton(
    gate_reshaped: torch.Tensor,
):
    """Build CSR buffers for the *gate* matrix on GPU using PyTorch + Triton.

    Args:
        gate_reshaped: ``(BS, NB)`` tensor, *float32* on CUDA.

    Returns:
        rows_index  (N,)    int32
        gates_val   (N,)    float32
        block_offsets (NB+1,) int32  CSR offsets
        max_rows    int  – maximum #rows per block (needed by compute kernel)
    """
    assert gate_reshaped.is_cuda, "Gate tensor must be on CUDA for Triton path"
    BS, NB = gate_reshaped.shape

    # ---------------------------------------------------------------------
    # 1) Per-block counts & offsets via plain PyTorch ops (GPU kernels)
    # ---------------------------------------------------------------------
    mask = gate_reshaped != 0
    block_counts = mask.sum(dim=0, dtype=torch.int32)  # (NB,)
    block_offsets = torch.empty(NB + 1, dtype=torch.int32, device=gate_reshaped.device)
    block_offsets[0] = 0
    block_offsets[1:] = torch.cumsum(block_counts, dim=0)

    N_total = int(block_offsets[-1].item())
    max_rows = int(block_counts.max().item())

    # Early exit: entirely zero gate tensor
    if N_total == 0:
        rows_index = torch.empty((0,), dtype=torch.int32, device=gate_reshaped.device)
        gates_val = torch.empty((0,), dtype=torch.float32, device=gate_reshaped.device)
        return rows_index, gates_val, block_offsets, max_rows

    # ---------------------------------------------------------------------
    # 2) Allocate output buffers & per-block cursors
    # ---------------------------------------------------------------------
    rows_index = torch.empty(N_total, dtype=torch.int32, device=gate_reshaped.device)
    gates_val = torch.empty(N_total, dtype=torch.float32, device=gate_reshaped.device)
    block_write_ptrs = torch.zeros(NB, dtype=torch.int32, device=gate_reshaped.device)

    # ---------------------------------------------------------------------
    # 3) Launch Triton kernel to fill the buffers
    # ---------------------------------------------------------------------
    # Grid dimensions
    BLOCK_SIZE_BS = 128  # will be overridden by autotune configs
    BLOCK_SIZE_NB = 4
    row_chunks = triton.cdiv(BS, BLOCK_SIZE_BS)
    block_chunks = triton.cdiv(NB, BLOCK_SIZE_NB)
    grid = (row_chunks, block_chunks)

    _csr_writer_kernel[grid](
        gate_reshaped,
        rows_index,
        gates_val,
        block_offsets,
        block_write_ptrs,
        BS, NB,
        gate_reshaped.stride(0), gate_reshaped.stride(1),
    )

    # Optional sanity check (debug builds only): ensure writes match counts
    # assert torch.all(block_write_ptrs == block_counts)

    return rows_index, gates_val, block_offsets, max_rows 


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 16, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_LS': 16, 'GROUP_SIZE_R': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_BS': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_LS': 32, 'GROUP_SIZE_R': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_LS': 64, 'GROUP_SIZE_R': 4}, num_warps=8, num_stages=2),
    ],
    key=['hidden_size', 'line_size'],
)
@triton.jit
def fused_up_proj_gate_csr_kernel(
    x_ptr, w_ptr, b_ptr, rows_index_ptr, gates_val_ptr, block_start_offsets_ptr,
    output_ptr, max_rows,
    # sizes
    hidden_size, line_size,
    # strides
    stride_x_bs, stride_x_h,
    stride_w_h, stride_w_i,
    stride_out_bs, stride_out_i,
    # meta
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_LS: tl.constexpr, GROUP_SIZE_R: tl.constexpr,
):
    pid = tl.program_id(0)

    C = (line_size + BLOCK_SIZE_LS - 1) // BLOCK_SIZE_LS

    row_chunks = (max_rows + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    row_groups = (row_chunks + GROUP_SIZE_R - 1) // GROUP_SIZE_R
    num_pid_per_block = C * GROUP_SIZE_R * row_groups

    block_idx = pid // num_pid_per_block
    pid_rem = pid % num_pid_per_block

    col_chunk    = (pid_rem // GROUP_SIZE_R) % C
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group    = pid_rem // (C * GROUP_SIZE_R)
    row_chunk    = row_group * GROUP_SIZE_R + row_in_group

    # Offsets within row
    block_start = tl.load(block_start_offsets_ptr + block_idx) + row_chunk * BLOCK_SIZE_BS
    block_end   = tl.load(block_start_offsets_ptr + block_idx + 1)

    if block_start >= block_end:
        return

    offs_bs = tl.arange(0, BLOCK_SIZE_BS)
    row_indices = block_start + offs_bs
    mask_bs = row_indices < block_end
    row_indices = tl.load(rows_index_ptr + row_indices, mask=mask_bs, other=0)

    gate_vals = tl.load(gates_val_ptr + block_start + offs_bs, mask=mask_bs, other=0.0)

    if tl.sum(gate_vals) == 0:
        return

    offs_ls = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls = offs_ls < line_size
    col_offset = block_idx * line_size
    global_cols = col_offset + offs_ls

    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)

    offs_h = tl.arange(0, BLOCK_SIZE_H)

    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size

        x_ptrs = x_ptr + row_indices[:, None] * stride_x_bs + curr_offs_h[None, :] * stride_x_h
        w_ptrs = w_ptr + curr_offs_h[:, None] * stride_w_h + global_cols[None, :] * stride_w_i

        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_h[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_ls[None, :], other=0.0)
        acc += tl.dot(x_block, w_block)

    b_ptrs = b_ptr + global_cols
    bias = tl.load(b_ptrs, mask=mask_ls, other=0.0)
    acc += bias[None, :]
    acc = tl.where(acc > 0, acc, 0.0)

    acc *= gate_vals[:, None]

    out_ptrs = output_ptr + row_indices[:, None] * stride_out_bs + global_cols[None, :] * stride_out_i
    tl.store(out_ptrs, acc, mask=mask_bs[:, None] & mask_ls[None, :])


def fused_up_proj_gate_activation_sparse_triton_csr_unified(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    GROUP_SIZE_R: int = 4,
    zero_init: bool = True,
):
    """Fast CSR path that builds CSR on the GPU using Triton.

    Args mirror the original *fused_up_proj_gate_activation_sparse_triton_csr* but
    the slow Python CSR construction has been replaced by *build_csr_buffers_triton*.
    """

    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Validate shapes / dtypes (same rules as original helper)
    assert up_weight.shape == (hidden_size, intermediate_size)
    assert gate.shape == (batch_size, seq_len, num_blocks)
    assert x.dtype == torch.float16 and up_weight.dtype == torch.float16 and up_bias.dtype == torch.float16

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Flatten views (contiguous)
    x_reshaped = x.contiguous().view(-1, hidden_size)           # (B*S, H)
    gate_reshaped = gate.contiguous().view(-1, num_blocks)      # (B*S, NB)
    batch_seq_size = x_reshaped.size(0)

    # ------------------------------------------------------------------
    # Build CSR buffers on the GPU
    # ------------------------------------------------------------------
    rows_index, gates_val, block_offsets, max_rows = build_csr_buffers_triton(gate_reshaped)

    N_total = rows_index.numel()
    if N_total == 0:
        # gate is entirely zero
        return torch.zeros((batch_size, seq_len, intermediate_size), device=x.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Allocate / zero-initialise output buffer as requested
    # ------------------------------------------------------------------
    if zero_init:
        output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=torch.float32)
    else:
        output = torch.empty((batch_seq_size, intermediate_size), device=x.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Helper to pick grid size for compute kernel (same formula as original)
    # ------------------------------------------------------------------
    def grid(meta):
        BLK_BS = meta['BLOCK_SIZE_BS']
        BLK_LS = meta['BLOCK_SIZE_LS']
        C = triton.cdiv(line_size, BLK_LS)
        row_chunks = triton.cdiv(max_rows, BLK_BS)
        row_groups = triton.cdiv(row_chunks, GROUP_SIZE_R)
        num_pid_per_block = C * GROUP_SIZE_R * row_groups
        return (num_pid_per_block * num_blocks,)

    fused_up_proj_gate_csr_kernel[grid](
        x_reshaped,
        up_weight,
        up_bias,
        rows_index,
        gates_val,
        block_offsets,
        output,
        max_rows,
        hidden_size, line_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        output.stride(0), output.stride(1),
    )

    return output.view(batch_size, seq_len, intermediate_size) 