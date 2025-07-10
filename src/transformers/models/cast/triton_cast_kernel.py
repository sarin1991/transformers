import torch
import triton
import triton.language as tl
from typing import Optional


# =============================================================================
# Triton kernel – indexed rows version (no host gather/scatter)
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 16, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_LS': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_BS': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_LS': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_LS': 64}, num_warps=8, num_stages=2),
    ],
    key=['num_rows', 'hidden_size', 'line_size'],
)
@triton.jit
def fused_up_proj_gate_activation_kernel_optimized(
    x_ptr,                       # (B·S, H)               float16
    up_weight_ptr,               # (H, line_size)         float16 – **single block slice**
    up_bias_ptr,                 # (line_size,)           float16 – slice for this block
    gate_col_ptr,                # (B·S,)                 float32 – gate[:, nb]
    rows_ptr,                    # (K,)                   int32 row indices for this block
    output_ptr,                  # (B·S, num_blocks*L)    float32 – full output tensor
    num_rows, hidden_size, line_size,
    stride_x_bs, stride_x_h,     # x strides
    stride_w_h,                 # w_slice stride(0)
    stride_out_bs, stride_out_ls,# output strides
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """Compute fused up-proj * gate for selected rows of **one** block.

    rows_ptr supplies the (1-D) int32 indices of the batch-seq rows to
    process for this invocation.  We load x and gate directly via those
    indices and write the results back into the full output tensor.
    """

    pid = tl.program_id(0)

    # Number of row blocks we need for the input set
    num_pid_bs = (num_rows + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    num_pid_ls = (line_size + BLOCK_SIZE_LS - 1) // BLOCK_SIZE_LS

    pid_ls = pid % num_pid_ls
    pid_rblk = pid // num_pid_ls  # block id among row-blocks

    offs_ls = pid_ls * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    mask_ls = offs_ls < line_size

    # Determine the starting row for this thread-block
    row_start = pid_rblk * BLOCK_SIZE_BS
    offs_bs = row_start + tl.arange(0, BLOCK_SIZE_BS)
    mask_bs_valid = offs_bs < num_rows

    # Gather the *global* row indices to operate on
    row_indices = tl.load(rows_ptr + offs_bs, mask=mask_bs_valid, other=0).to(tl.int32)

    # Load gate values for these rows (gate_col_ptr is already offset to the correct column)
    g_ptrs = gate_col_ptr + row_indices
    g_vals = tl.load(g_ptrs, mask=mask_bs_valid, other=0.0)

    # Early exit if whole sub-block has zero gates
    if tl.sum(g_vals) == 0:
        return

    # -------------------- matrix multiply --------------------
    # We compute accumulator (BLOCK_SIZE_BS, BLOCK_SIZE_LS)
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)

    # Offsets along hidden dim
    offs_h = tl.arange(0, BLOCK_SIZE_H)

    # pointers base for X and W
    # Will be updated inside the k-loop
    # x_ptr matrix:  row_index * stride_x_bs + h * stride_x_h
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        curr_offs_h = h + offs_h
        mask_h = curr_offs_h < hidden_size

        # Build pointer arrays for x and w
        x_ptrs = x_ptr + (row_indices[:, None] * stride_x_bs) + (curr_offs_h[None, :] * stride_x_h)
        w_ptrs = up_weight_ptr + (curr_offs_h[:, None] * stride_w_h) + (offs_ls[None, :] * 1)
        # Note: up_weight is already sliced so stride_w_h = line_size, stride_w_ls = 1

        x_block = tl.load(x_ptrs, mask=mask_bs_valid[:, None] & mask_h[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_ls[None, :], other=0.0)
        acc = tl.dot(x_block, w_block, acc)

    # Add bias & ReLU
    bias_ptrs = up_bias_ptr + offs_ls
    b = tl.load(bias_ptrs, mask=mask_ls, other=0.0)
    acc += b[None, :]
    acc = tl.where(acc > 0, acc, 0.0)

    # Apply gate
    acc *= g_vals[:, None]

    # Store back to output – need global row index
    out_ptrs = output_ptr + (row_indices[:, None] * stride_out_bs) + (offs_ls[None, :] * stride_out_ls)
    tl.store(out_ptrs, acc, mask=mask_bs_valid[:, None] & mask_ls[None, :])


# =============================================================================
# Python helper – sparse path using the indexed kernel
# =============================================================================

def fused_up_proj_gate_activation_sparse_triton_optimized(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    density_threshold: float = 0.5,
):
    """Sparse variant that offloads gather & scatter into the Triton kernel.

    Signature is identical to *fused_up_proj_gate_activation_sparse_triton* so
    it can be dropped into existing debug / benchmark helpers.
    """

    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Validations
    assert up_weight.shape == (hidden_size, intermediate_size)
    assert gate.shape == (batch_size, seq_len, num_blocks)
    assert x.dtype == torch.float16
    assert up_weight.dtype == torch.float16
    assert up_bias.dtype == torch.float16

    if gate.dtype != torch.float32:
        gate = gate.float()

    x_reshaped = x.contiguous().view(-1, hidden_size)          # (B·S, H)
    gate_reshaped = gate.contiguous().view(-1, num_blocks)     # (B·S, NB)
    batch_seq_size = x_reshaped.size(0)

    # ------------------------------------------------------------------
    # Determine sparsity – fallback if too dense
    # ------------------------------------------------------------------
    gate_mask = gate_reshaped != 0
    nnz = int(gate_mask.sum().item())
    if nnz == 0:
        return torch.zeros((batch_size, seq_len, intermediate_size), device=x.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Sparse path using indexed kernel
    # ------------------------------------------------------------------
    output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=torch.float32)

    rows, blocks = gate_mask.nonzero(as_tuple=True)
    # Iterate over blocks – still one call per block, but no gather/scatter
    unique_blocks = blocks.unique(sorted=False)

    for nb in unique_blocks.tolist():
        rows_mask = blocks == nb
        rows_nb = rows[rows_mask].contiguous()
        K = rows_nb.numel()
        if K == 0:
            continue

        # Build weight & bias slice for this block (contiguous)
        col_start = nb * line_size
        col_end = col_start + line_size
        w_slice = up_weight[:, col_start:col_end].contiguous()
        b_slice = up_bias[col_start:col_end].contiguous()

        # Pointers we need
        gate_col_ptr = gate_reshaped[:, nb].contiguous()  # (B·S,) – contiguous column view

        grid = lambda meta: (
            triton.cdiv(K, meta['BLOCK_SIZE_BS']) * triton.cdiv(line_size, meta['BLOCK_SIZE_LS']),
        )

        fused_up_proj_gate_activation_kernel_optimized[grid](
            x_reshaped,
            w_slice,
            b_slice,
            gate_col_ptr,
            rows_nb,
            output,
            K, hidden_size, line_size,
            x_reshaped.stride(0), x_reshaped.stride(1),
            w_slice.stride(0),
            output.stride(0), output.stride(1),
        )

    return output.view(batch_size, seq_len, intermediate_size) 