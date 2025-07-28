import torch
import triton
import triton.language as tl
from typing import Tuple


# =============================================================================
# Sparse CSR-style kernel – each CTA works on ONE block and ONE row-chunk
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
def fused_up_proj_gate_csr_kernel(
    x_ptr,                       # (B·S, H)             fp16
    w_ptr,                       # (H, I)               fp16   (full matrix)
    b_ptr,                       # (I,)                 fp16
    rows_index_ptr,              # (N,)                 int32
    gates_val_ptr,               # (N,)                 fp32
    block_start_offsets_ptr,     # (NB+1,)              int32
    output_ptr,                  # (B·S, I)             fp32   (pre-allocated, zero)
    max_rows,                    # int32 – maximum rows per block (host pre-computed)
    # Sizes
    hidden_size, line_size,
    # Strides
    stride_x_bs, stride_x_h,     # x
    stride_w_h, stride_w_i,      # w  (H major)
    stride_out_bs, stride_out_i, # output
    # Meta
    out_dtype: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_LS: tl.constexpr, GROUP_SIZE_R: tl.constexpr,
):
    """CTA computes a tile (row_chunk, col_chunk) for **one block**.

    grid is 1-D; we derive (block_idx, row_chunk, col_chunk) in software
    following grouping logic: row_chunk fast, col_chunk medium, block slow.
    """

    pid = tl.program_id(0)

    C = (line_size + BLOCK_SIZE_LS - 1) // BLOCK_SIZE_LS  # column chunks per block

    # Reconstruct num_pid_per_block from max_rows & meta – same formula as host
    row_chunks = (max_rows + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    row_groups = (row_chunks + GROUP_SIZE_R - 1) // GROUP_SIZE_R
    num_pid_per_block = C * GROUP_SIZE_R * row_groups

    # Derive block index (slowest-changing)
    block_idx = pid // num_pid_per_block
    pid_rem   = pid %  num_pid_per_block

    col_chunk    = (pid_rem // GROUP_SIZE_R) % C
    row_in_group = pid_rem % GROUP_SIZE_R
    row_group    = pid_rem // (C * GROUP_SIZE_R)
    row_chunk    = row_group * GROUP_SIZE_R + row_in_group

    # Offsets within row
    block_start = tl.load(block_start_offsets_ptr + block_idx) + row_chunk * BLOCK_SIZE_BS
    block_end   = tl.load(block_start_offsets_ptr + block_idx + 1)

    if block_start >= block_end:
        return  # nothing to do for this CTA

    # Prepare row indices tensor
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)
    row_indices = block_start + offs_bs
    mask_bs = row_indices < block_end
    row_indices = tl.load(rows_index_ptr + row_indices, mask=mask_bs, other=0)  # int32 indices into x/out

    # Gate values
    gate_vals = tl.load(gates_val_ptr + block_start + offs_bs, mask=mask_bs, other=0.0)

    # Early exit if entire tile is zero – caller is responsible for zero-init
    if tl.sum(gate_vals) == 0:
        return

    # Column-related offsets (only evaluated for non-zero tiles)
    offs_ls = col_chunk * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls = offs_ls < line_size
    col_offset = block_idx * line_size
    global_cols = col_offset + offs_ls

    # Prepare acc
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

    # Bias, relu
    b_ptrs = b_ptr + global_cols
    bias = tl.load(b_ptrs, mask=mask_ls, other=0.0)
    acc += bias[None, :]
    acc = tl.where(acc > 0, acc, 0.0)

    # Apply gate
    acc *= gate_vals[:, None]

    # Store
    out_ptrs = output_ptr + row_indices[:, None] * stride_out_bs + global_cols[None, :] * stride_out_i
    tl.atomic_add(out_ptrs, acc.to(out_dtype), mask=mask_bs[:, None] & mask_ls[None, :]) 

# =============================================================================
# Python helper – CSR sparse path
# =============================================================================

def fused_up_proj_gate_activation_sparse_triton_csr(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    GROUP_SIZE_R: int = 4,
    zero_init: bool = True,
    out_dtype: torch.dtype = torch.float32,
):
    """Sparse helper using CSR buffers and single-axis grid launch."""

    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Validate
    assert up_weight.shape == (hidden_size, intermediate_size)
    assert gate.shape == (batch_size, seq_len, num_blocks)

    supported_dtypes = (torch.float16, torch.bfloat16)
    assert x.dtype in supported_dtypes, "x must be fp16 or bf16"
    assert up_weight.dtype in supported_dtypes, "up_weight must be fp16 or bf16"
    assert up_bias.dtype in supported_dtypes, "up_bias must be fp16 or bf16"

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Flatten views
    x_reshaped = x.contiguous().view(-1, hidden_size)           # (B*S, H)
    gate_reshaped = gate.contiguous().view(-1, num_blocks)      # (B*S, NB)
    batch_seq_size = x_reshaped.size(0)

    # Build CSR buffers
    rows_list = []
    gates_list = []
    block_start_offsets = [0]
    for nb in range(num_blocks):
        mask_nb = gate_reshaped[:, nb] != 0
        rows_nb = mask_nb.nonzero(as_tuple=False).squeeze(1)
        if rows_nb.numel() == 0:
            block_start_offsets.append(block_start_offsets[-1])
            continue
        gates_nb = gate_reshaped[rows_nb, nb]
        rows_list.append(rows_nb.int())
        gates_list.append(gates_nb)
        block_start_offsets.append(block_start_offsets[-1] + rows_nb.numel())

    N_total = block_start_offsets[-1]
    if N_total == 0:
        # all gates zero
        return torch.zeros((batch_size, seq_len, intermediate_size), device=x.device, dtype=out_dtype)

    rows_index = torch.cat(rows_list).contiguous().to(torch.int32)
    gates_index = torch.cat(gates_list).contiguous().to(torch.float32)
    block_start_offsets = torch.tensor(block_start_offsets, device=x.device, dtype=torch.int32)

    # Output buffer initialisation policy controlled by caller
    if zero_init:
        output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)
    else:
        output = torch.empty((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)

    # Pre-compute maximum rows per block (CPU side)
    max_rows = 0
    for nb in range(num_blocks):
        max_rows = max(max_rows, int(block_start_offsets[nb + 1] - block_start_offsets[nb]))

    # Helper to compute grid size depending on meta-parameters
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
        gates_index,
        block_start_offsets,
        output,
        max_rows,
        hidden_size, line_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        output.stride(0), output.stride(1),
        out_dtype=tl.float16 if out_dtype == torch.float16 else tl.float32,
    )

    return output.view(batch_size, seq_len, intermediate_size) 