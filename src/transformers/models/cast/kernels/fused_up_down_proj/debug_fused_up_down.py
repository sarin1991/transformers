import argparse
from typing import Tuple

import torch
import torch.nn.functional as F

# Absolute imports to reference sort-pack up/down kernels
from transformers.models.cast.kernels.up_proj.triton_cast_kernel_gate_sortpack import (
    fused_up_proj_gate_activation_sparse_triton_sortpack,
)
from transformers.models.cast.kernels.down_proj.triton_cast_kernel_gate_sortpack import (
    fused_down_proj_sparse_triton_sortpack,
)

# Local fused helper
from .triton_cast_kernel_gate_sortpack_fused import (
    fused_up_down_proj_sparse_triton_sortpack,
)


def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9) -> torch.Tensor:
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    if sparsity > 0.0:
        mask = torch.rand_like(gate) < sparsity
        gate[mask] = 0.0
    return gate


def _preprocess_gate(gate_tensor: torch.Tensor, num_blocks: int):
    mask = gate_tensor > 0
    block_counts = mask.sum(dim=0, dtype=torch.int32)
    max_rows = int(block_counts.max().item())
    if max_rows == 0:
        return None, None, block_counts, max_rows
    gate_vals_sorted, row_idx_sorted = torch.sort(gate_tensor, dim=0, descending=True)
    gate_vals = gate_vals_sorted[:max_rows, :].t().contiguous()
    row_idx = row_idx_sorted[:max_rows, :].t().contiguous().to(torch.int32)
    return gate_vals, row_idx, block_counts, max_rows


def _reference_two_kernels(
    x: torch.Tensor,
    gate: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    num_blocks: int,
    line_size: int,
    *,
    gate_vals: torch.Tensor,
    row_idx: torch.Tensor,
    block_counts: torch.Tensor,
    max_rows: int,
    out_dtype: torch.dtype,
    apply_relu: bool = True,
    apply_gate: bool = True,
) -> torch.Tensor:
    # Up-proj (post-activation, post-gate) → (BS, I)
    up_intermediate = fused_up_proj_gate_activation_sparse_triton_sortpack(
        x,
        up_weight,
        gate,
        num_blocks,
        line_size,
        zero_init=False,
        out_dtype=out_dtype,
        apply_gate=apply_gate,
        apply_relu=apply_relu,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
    )

    # Down-proj (consumes gated intermediate) → (BS, H)
    y_ref = fused_down_proj_sparse_triton_sortpack(
        up_intermediate,
        down_weight,
        gate,
        num_blocks,
        line_size,
        out_dtype=out_dtype,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
    )

    return y_ref


def run_accuracy_once(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_blocks: int,
    line_size: int,
    sparsity: float = 0.9,
    dtype: torch.dtype = torch.float16,
    out_dtype: torch.dtype = torch.float32,
    apply_relu: bool = True,
    apply_gate: bool = True,
) -> Tuple[float, float, float, float]:
    bs = batch_size * seq_len
    inter = num_blocks * line_size

    x = torch.randn(bs, hidden_size, device="cuda", dtype=dtype)
    gate = _make_sparse_gate(bs, num_blocks, sparsity=sparsity)
    up_weight = torch.randn(hidden_size, inter, device="cuda", dtype=dtype)
    down_weight = torch.randn(inter, hidden_size, device="cuda", dtype=dtype)

    gate_vals, row_idx, block_counts, max_rows = _preprocess_gate(gate, num_blocks)

    # Fused
    y_fused = fused_up_down_proj_sparse_triton_sortpack(
        x,
        up_weight,
        down_weight,
        gate,
        num_blocks,
        line_size,
        out_dtype=out_dtype,
        apply_gate=apply_gate,
        apply_relu=apply_relu,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
        save_up_proj=False,
    )

    # Reference (two kernels)
    y_ref = _reference_two_kernels(
        x,
        gate,
        up_weight,
        down_weight,
        num_blocks,
        line_size,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
        out_dtype=out_dtype,
        apply_relu=apply_relu,
        apply_gate=apply_gate,
    )

    # Compare
    diff = (y_fused - y_ref).float().abs()
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    ref_abs = y_ref.float().abs()
    rel_max = abs_max / (ref_abs.max().item() + 1e-6)
    rel_mean = abs_mean / (ref_abs.mean().item() + 1e-6)
    return rel_max, rel_mean, abs_max, abs_mean


def main():
    parser = argparse.ArgumentParser(description="Debug fused up+down vs two-kernel reference (sort-pack)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Input/weight dtype")
    parser.add_argument("--out-dtype", default="float32", choices=["float16", "bfloat16", "float32"], help="Output dtype for kernels")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Fraction of zeros in gate")
    parser.add_argument("--apply-relu", action="store_true", help="Apply ReLU in up-proj before gate")
    parser.add_argument("--no-gate", action="store_true", help="Disable gating multiply")
    parser.add_argument("--configs", type=str, default="2,4,128,4,32;4,8,256,8,32;8,16,512,8,64",
                        help="Semicolon-separated configs as B,S,H,NB,LS")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available – exiting")
        return

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    out_dtype = dtype_map[args.out_dtype]
    apply_gate = not args.no_gate

    print("=== Fused Up+Down: Accuracy vs Two-Kernel Reference ===")
    for cfg in [c.strip() for c in args.configs.split(";") if c.strip()]:
        b, s, h, nb, ls = map(int, cfg.split(","))
        rel_max, rel_mean, abs_max, abs_mean = run_accuracy_once(
            batch_size=b,
            seq_len=s,
            hidden_size=h,
            num_blocks=nb,
            line_size=ls,
            sparsity=args.sparsity,
            dtype=dtype,
            out_dtype=out_dtype,
            apply_relu=args.apply_relu,
            apply_gate=apply_gate,
        )
        print(f"Config B={b} S={s} H={h} NB={nb} LS={ls} | rel_max={rel_max:.3e} rel_mean={rel_mean:.3e} | abs_max={abs_max:.3e} abs_mean={abs_mean:.3e}")


if __name__ == "__main__":
    main()


