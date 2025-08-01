import torch
import triton
import triton.language as tl
from triton_cast_kernel_gate_sortpack import fused_weight_grad_sparse_triton_sortpack

# -----------------------------------------------------------------------------
# Autotuning configurations – reuse the same philosophy as up_proj/down_proj
# -----------------------------------------------------------------------------
_CONFIG_WARPS = (4, 8, 16)
_TILE_SIZES   = (64, 128)
_NUM_STAGES   = (1, 2, 3)
_GROUP_SIZE_I = 16  # grouping factor along the I-dimension

CONFIGS = []
for tile in _TILE_SIZES:
    for warps in _CONFIG_WARPS:
        for stages in _NUM_STAGES:
            CONFIGS.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_I": tile,
                        "BLOCK_SIZE_H": tile,
                        "BLOCK_SIZE_BS": tile,
                        "GROUP_SIZE_I": _GROUP_SIZE_I,
                    },
                    num_warps=warps,
                    num_stages=stages,
                )
            )


@triton.autotune(configs=CONFIGS, key=["intermediate_size", "hidden_size", "batch_seq_size"])
@triton.jit
def fused_weight_grad_kernel(
    y_ptr, x_ptr, out_ptr,  # pointers; y = dy or activations, x = x or dy
    batch_seq_size, intermediate_size, hidden_size, line_size,
    stride_s_bs, stride_s_i,
    stride_d_bs, stride_d_h,
    stride_out_i, stride_out_h,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    """Compute Gram matrix G = yᵀ · x used as weight-gradient.

    Inputs (all row-major):
        y_ptr :  (batch_seq_size, intermediate_size)  fp16/bf16/fp32  – sparse or dense activations/gradients
        x_ptr :  (batch_seq_size, hidden_size)        fp16/bf16/fp32  – dense activations/gradients

    Output:
        out_ptr : (intermediate_size, hidden_size)    fp16/bf16/fp32 – accumulated result

    The kernel tiles along I and H dimensions and reduces over the batch_seq axis.
    GROUP_SIZE_I allows several I-tiles to stay resident in the same CTA, mirroring the
    `GROUP_SIZE_BS` strategy used in the down-projection kernel.
    """

    pid = tl.program_id(0)

    # ------------------------------------------------------------------
    # Decompose program id into (i_chunk, h_chunk) with grouping on i
    # ------------------------------------------------------------------
    num_hidden_chunks = tl.cdiv(hidden_size, BLOCK_SIZE_H)
    num_ls_groups     = tl.cdiv(line_size, GROUP_SIZE_I * BLOCK_SIZE_I)
    ls_chunks_per_blk = num_ls_groups * GROUP_SIZE_I
    num_blocks        = tl.cdiv(intermediate_size, line_size)

    # Total CTAs per block column
    num_pid_per_block = ls_chunks_per_blk * num_hidden_chunks

    # -------- program-id decomposition --------
    block_idx = pid // num_pid_per_block
    pid_blk   = pid %  num_pid_per_block                    # inside this block column

    # 1) identify which LS-group (coarse) we are in
    ls_group       = pid_blk // (num_hidden_chunks * GROUP_SIZE_I)

    # 2) remainder inside that LS-group encodes (h_chunk, ls_chunk_in_group)
    pid_in_lsg     = pid_blk %  (num_hidden_chunks * GROUP_SIZE_I)

    h_chunk            = pid_in_lsg // GROUP_SIZE_I           # hidden-dimension tile
    ls_chunk_in_group  = pid_in_lsg %  GROUP_SIZE_I           # fine LS tile inside group

    # absolute LS chunk index inside this block column
    ls_chunk_in_block = ls_group * GROUP_SIZE_I + ls_chunk_in_group  # 0 … ls_chunks_per_blk-1

    # Guard – ensure indices map inside the valid ranges (avoid chained boolean ops)
    invalid_h  = h_chunk * BLOCK_SIZE_H >= hidden_size
    invalid_ls = ls_chunk_in_block * BLOCK_SIZE_I >= line_size
    invalid_b  = block_idx >= num_blocks

    if (invalid_h or invalid_ls) or invalid_b:
        return

    i_start = block_idx * line_size + ls_chunk_in_block * BLOCK_SIZE_I

    # update offs (row indices this CTA is responsible for)
    offs_i = i_start + tl.arange(0, BLOCK_SIZE_I)
    offs_h = h_chunk * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)

    mask_i = offs_i < intermediate_size
    mask_h = offs_h < hidden_size

    # Accumulator for this tile (I,H)
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_H), dtype=tl.float32)

    # Loop over batchSeq (reduction axis)
    for b in range(0, batch_seq_size, BLOCK_SIZE_BS):
        curr_bs = b + offs_bs
        mask_bs = curr_bs < batch_seq_size

        # Load tile from Y  -> shape (I, BS)
        y_ptrs = y_ptr + offs_i[:, None] * stride_s_i + curr_bs[None, :] * stride_s_bs
        y_block = tl.load(y_ptrs, mask=mask_i[:, None] & mask_bs[None, :], other=0.0)

        # Load tile from X  -> shape (BS, H)
        x_ptrs = x_ptr + curr_bs[:, None] * stride_d_bs + offs_h[None, :] * stride_d_h
        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_h[None, :], other=0.0)

        # y_block: (I, BS) , x_block : (BS, H)
        acc += tl.dot(y_block, x_block)

    # Write-back
    out_ptrs = out_ptr + offs_i[:, None] * stride_out_i + offs_h[None, :] * stride_out_h
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_i[:, None] & mask_h[None, :])


def fused_weight_grad_triton(
    intermediate: torch.Tensor,
    other: torch.Tensor,
    line_size: int,
    out_dtype: torch.dtype = torch.float32,
):
    """Compute weight gradient `dW = intermediateᵀ · other` with Triton.

    Args:
        intermediate: (batch_seq, intermediate_size) – *float16*/*bfloat16*/*float32*  (U or dZ)
        other:        (batch_seq, hidden_size)       – *float16*/*bfloat16*/*float32*  (dY or X)
        line_size: size of each block line (LS)
        out_dtype: dtype of the returned matrix (default fp32, can be fp16)

    Returns:
        Tensor of shape (intermediate_size, hidden_size)
    """

    # Basic validations
    assert intermediate.ndim == 2 and other.ndim == 2, "Input tensors must be 2-D"
    assert intermediate.shape[0] == other.shape[0], "Batch dimension mismatch"

    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert intermediate.dtype in supported_dtypes and other.dtype in supported_dtypes, f"Unsupported dtype: intermediate={intermediate.dtype}, other={other.dtype}. Supported: {supported_dtypes}"

    batch_seq_size, intermediate_size = intermediate.shape
    assert intermediate_size % line_size == 0, "line_size must divide intermediate_size"
    hidden_size = other.shape[1]

    out = torch.empty((intermediate_size, hidden_size), device=intermediate.device, dtype=out_dtype)

    def grid(meta):
        BLK_I = meta["BLOCK_SIZE_I"]
        BLK_H = meta["BLOCK_SIZE_H"]
        G_I   = meta["GROUP_SIZE_I"]

        num_ls_groups = triton.cdiv(line_size, G_I * BLK_I)
        num_blocks    = intermediate_size // line_size
        h_chunks      = triton.cdiv(hidden_size, BLK_H)

        return (num_blocks * num_ls_groups * h_chunks * G_I,)

    # Map torch dtypes to triton dtypes
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    fused_weight_grad_kernel[grid](
        intermediate, other, out,
        batch_seq_size,
        intermediate_size,
        hidden_size,
        line_size,
        intermediate.stride(0), intermediate.stride(1),
        other.stride(0), other.stride(1),
        out.stride(0), out.stride(1),
        out_dtype=triton_out_dtype,
    )

    return out

# -----------------------------------------------------------------------------
# Debug utilities
# -----------------------------------------------------------------------------


def _make_sparse_gate(batch_seq: int, num_blocks: int, sparsity: float = 0.9):
    """Build a 1-D boolean mask of active blocks for each row (size N × NB)."""
    gate = torch.rand(batch_seq, num_blocks, device="cuda", dtype=torch.float32)
    mask = torch.rand_like(gate) < sparsity
    gate[mask] = 0.0
    return gate


def debug_large_scale():
    """Run numerical correctness checks on several shapes with block sparsity."""

    configs = [
        # (N, hidden_size, NB, LS)  -> I = NB*LS
        (64,   256, 4,  32),
        (256,  512, 8,  32),
        (512,  768, 8,  64),
        (1024,  1024, 4,  1024),
    ]

    overall_max_diff_dense = 0.0
    overall_max_diff_sortpack = 0.0

    for batch_seq, hidden_size, num_blocks, line_size in configs:
        intermediate_size = num_blocks * line_size
        print(
            f"\nConfig: N={batch_seq} | H={hidden_size} | I={intermediate_size} | NB={num_blocks} | LS={line_size}"
        )

        # Random activations and gradients
        other = torch.randn(batch_seq, hidden_size, device="cuda", dtype=torch.float16)  # acts or grads (N,H)

        # Build sparse intermediate matrix following gate mask
        gate = _make_sparse_gate(batch_seq, num_blocks, sparsity=0.9)
        intermediate = torch.randn(batch_seq, num_blocks, line_size, device="cuda", dtype=torch.float16)
        intermediate = intermediate * gate.unsqueeze(-1).to(dtype=intermediate.dtype)
        intermediate = intermediate.view(batch_seq, intermediate_size)

        # Reference & Triton (dense)
        ref = intermediate.transpose(0, 1).float() @ other.float()
        tri_dense = fused_weight_grad_triton(intermediate, other, line_size)

        # Triton SortPack sparse helper
        tri_sortpack = fused_weight_grad_sparse_triton_sortpack(
            intermediate,   # (BS, I)
            other,          # (BS, H)
            gate,           # (BS, NB)
            num_blocks,
            line_size,
        )

        # Diffs
        max_diff_dense = torch.max(torch.abs(ref - tri_dense)).item()
        mean_diff_dense = torch.mean(torch.abs(ref - tri_dense)).item()

        max_diff_sort = torch.max(torch.abs(ref - tri_sortpack)).item()
        mean_diff_sort = torch.mean(torch.abs(ref - tri_sortpack)).item()

        overall_max_diff_dense = max(overall_max_diff_dense, max_diff_dense)
        overall_max_diff_sortpack = max(overall_max_diff_sortpack, max_diff_sort)

        print(f"Dense   → max diff {max_diff_dense:.6e} | mean diff {mean_diff_dense:.6e}")
        print(f"SortPk  → max diff {max_diff_sort:.6e} | mean diff {mean_diff_sort:.6e}")

    print(
        f"\nOverall max diff across configs | Dense: {overall_max_diff_dense:.6e} | SortPk: {overall_max_diff_sortpack:.6e}"
    )


if __name__ == "__main__":
    debug_large_scale() 