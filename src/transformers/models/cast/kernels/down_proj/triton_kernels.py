import torch
import triton
import triton.language as tl
from typing import Optional
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 16, 'BLOCK_SIZE_I': 16, 'BLOCK_SIZE_H': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_BS': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_H': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_H': 64}, num_warps=8, num_stages=2),
    ],
    key=['batch_seq_size', 'intermediate_size', 'hidden_size'],
)
@triton.jit
def fused_down_proj_kernel(
    x_ptr, w_ptr, row_mask_ptr, out_ptr,
    batch_seq_size, intermediate_size, hidden_size,
    stride_x_bs, stride_x_i,
    stride_w_i, stride_w_h,
    stride_mask_bs,
    stride_out_bs, stride_out_h,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_H: tl.constexpr,
):
    """Dense down-projection kernel (matmul) with optional early exit when an entire
    intermediate vector is zero.  *row_mask_ptr* is a 1-D array containing 1 for
    active rows and 0 for fully-zero rows.
    ``x``:   (B·S, I)   – float16
    ``w``:   (I,  H)   – float16  (transposed weight)
    ``out``: (B·S, H)   – float32
    """

    pid = tl.program_id(0)

    num_pid_bs = (batch_seq_size + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    num_pid_h  = (hidden_size     + BLOCK_SIZE_H  - 1) // BLOCK_SIZE_H

    pid_h  = pid % num_pid_h
    pid_bs = pid // num_pid_h

    if pid_bs >= num_pid_bs:
        return

    offs_bs = pid_bs * BLOCK_SIZE_BS + tl.arange(0, BLOCK_SIZE_BS)
    offs_h  = pid_h  * BLOCK_SIZE_H  + tl.arange(0, BLOCK_SIZE_H)
    offs_i  = tl.arange(0, BLOCK_SIZE_I)

    mask_bs = offs_bs < batch_seq_size
    mask_h  = offs_h  < hidden_size

    # ------------------------------------------------------------------
    # Row activity mask – if all selected rows are inactive, write zeros.
    # ------------------------------------------------------------------
    row_mask_vals = tl.load(row_mask_ptr + offs_bs * stride_mask_bs, mask=mask_bs, other=0.0)
    if tl.sum(row_mask_vals) == 0:
        out_ptrs = out_ptr + offs_bs[:, None] * stride_out_bs + offs_h[None, :] * stride_out_h
        tl.store(out_ptrs, tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_H), dtype=tl.float32), mask=mask_bs[:, None] & mask_h[None, :])
        return

    # ------------------------------------------------------------------
    # Accumulator initialisation (float32)
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_H), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main matmul loop over intermediate_size (K-dimension)
    # ------------------------------------------------------------------
    for k in range(0, intermediate_size, BLOCK_SIZE_I):
        curr_offs_i = k + offs_i
        mask_i = curr_offs_i < intermediate_size

        x_ptrs = x_ptr + offs_bs[:, None] * stride_x_bs + curr_offs_i[None, :] * stride_x_i
        w_ptrs = w_ptr + curr_offs_i[:, None] * stride_w_i + offs_h[None, :] * stride_w_h

        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_i[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_i[:, None] & mask_h[None, :], other=0.0)

        acc += tl.dot(x_block, w_block)

    # ------------------------------------------------------------------
    # Multiply by row mask to zero-out inactive rows (cheaper than branching per row).
    # ------------------------------------------------------------------
    acc *= row_mask_vals[:, None]

    # ------------------------------------------------------------------
    # Write back
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + offs_bs[:, None] * stride_out_bs + offs_h[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_bs[:, None] & mask_h[None, :])


# ===============================================================
# Python wrapper
# ===============================================================

def fused_down_proj_triton(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
):
    """Fused dense down-projection using Triton.

    Args:
        x:            (batch, seq_len, intermediate_size) – *float16*
        down_weight:  (intermediate_size, hidden_size)    – *float16* (note: pass ``model.down_proj.weight.t()``)
        gate:         (batch, seq_len, num_blocks)        – *float32* (or any type, will be upcast)
        num_blocks:   ``NB``
        line_size:    ``LS`` (``intermediate_size = NB·LS``)

    Returns:
        Tensor (batch, seq_len, hidden_size) in *float32*
    """
    batch_size, seq_len, intermediate_size = x.shape
    hidden_size = down_weight.shape[1]

    assert intermediate_size == num_blocks * line_size, "Mismatch intermediate size"
    assert down_weight.shape == (intermediate_size, hidden_size)
    assert gate.shape == (batch_size, seq_len, num_blocks)
    assert x.dtype == torch.float16 and down_weight.dtype == torch.float16, "x and weight must be fp16"

    # ------------------------------------------------------------------
    # Build per-row activity mask: 1 if any gate > 0 else 0.
    # ------------------------------------------------------------------
    gate_reshaped = gate.contiguous().view(-1, num_blocks)
    row_mask = (gate_reshaped.abs().sum(dim=1) != 0).float().contiguous()

    # Flatten x to 2-D (B·S, I)
    batch_seq_size = batch_size * seq_len
    x_reshaped = x.contiguous().view(batch_seq_size, intermediate_size)

    # Allocate output (fp32)
    output = torch.empty((batch_seq_size, hidden_size), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(batch_seq_size, meta['BLOCK_SIZE_BS']) * triton.cdiv(hidden_size, meta['BLOCK_SIZE_H']),
    )

    fused_down_proj_kernel[grid](
        x_reshaped,
        down_weight,
        row_mask,
        output,
        batch_seq_size,
        intermediate_size,
        hidden_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        row_mask.stride(0),
        output.stride(0), output.stride(1),
    )

    return output.view(batch_size, seq_len, hidden_size)


# ===============================================================
# Debug / numerical accuracy checker (run with `python triton_kernels.py`)
# ===============================================================

def _make_sparse_gate(batch_size: int, seq_len: int, num_blocks: int, sparsity: float = 0.9):
    gate = torch.rand(batch_size, seq_len, num_blocks, device="cuda", dtype=torch.float32)
    mask = torch.rand_like(gate) < sparsity
    gate[mask] = 0.0
    return gate


def debug_large_scale(use_sparse_gate: bool = False):
    """Run numerical correctness checks on several shapes."""
    configs = [
        (4, 8, 128, 4, 32),      # (B, S, H, NB, LS)
        (8, 16, 256, 8, 32),
        (32, 32, 512, 8, 64),
    ]

    overall_max_diff = 0.0

    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        intermediate_size = num_blocks * line_size
        print(f"\nConfig: {batch_size}×{seq_len} / H={hidden_size} / NB={num_blocks} / LS={line_size}")

        # Random tensors
        x_fp16 = torch.randn(batch_size, seq_len, intermediate_size, device="cuda", dtype=torch.float16)
        down_weight_fp16 = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
        gate = _make_sparse_gate(batch_size, seq_len, num_blocks, sparsity=0.9 if use_sparse_gate else 0.0)

        # Zero-out fully gated rows in x (simulate real pipeline)
        row_mask = (gate.view(-1, num_blocks).abs().sum(dim=1) != 0).view(batch_size, seq_len, 1)
        x_fp16 = x_fp16 * row_mask.to(dtype=torch.float16)

        # Reference PyTorch result (fp32)
        ref_fp32 = F.linear(x_fp16.float(), down_weight_fp16.t().float()).float()

        # Triton
        out_triton = fused_down_proj_triton(
            x_fp16,
            down_weight_fp16,
            gate,
            num_blocks,
            line_size,
        )

        max_diff = torch.max(torch.abs(ref_fp32 - out_triton)).item()
        mean_diff = torch.mean(torch.abs(ref_fp32 - out_triton)).item()
        overall_max_diff = max(overall_max_diff, max_diff)

        print(f"max diff {max_diff:.6e} | mean diff {mean_diff:.6e}")

    print(f"\nOverall max diff across configs: {overall_max_diff:.6e}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA is not available – exiting.")
        exit(1)
    try:
        import triton
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    print("✅ Triton down-proj kernel loaded successfully.")
    debug_large_scale(use_sparse_gate=False)
    debug_large_scale(use_sparse_gate=True) 