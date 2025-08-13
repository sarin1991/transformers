import argparse
from typing import Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from .triton_cast_kernel_gate_sortpack_fused import (
    fused_up_down_proj_sparse_triton_sortpack,
)
from kernels.up_proj.triton_cast_kernel_gate_sortpack import (
    fused_up_proj_gate_activation_sparse_triton_sortpack,
)
from kernels.down_proj.triton_cast_kernel_gate_sortpack import (
    fused_down_proj_sparse_triton_sortpack,
)


def _make_tensors(batch: int, seq: int, hidden: int, nb: int, ls: int, sparsity: float | str):
    bs = batch * seq
    inter = nb * ls
    x = torch.randn(bs, hidden, device="cuda", dtype=torch.float16)
    up_w = torch.randn(hidden, inter, device="cuda", dtype=torch.float16)
    down_w = torch.randn(inter, hidden, device="cuda", dtype=torch.float16)
    gate = torch.rand(bs, nb, device="cuda", dtype=torch.float32)
    # Dynamic sparsity: expected 1 active block per row
    if isinstance(sparsity, str) and sparsity == "dynamic":
        zeros_frac = 1.0 - (1.0 / nb)
    else:
        zeros_frac = float(sparsity)
    if zeros_frac > 0:
        gate[torch.rand_like(gate) < zeros_frac] = 0.0
    return x, up_w, down_w, gate


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


def _run_fused(x, up_w, down_w, gate, nb, ls, gate_vals, row_idx, block_counts, max_rows):
    fused_up_down_proj_sparse_triton_sortpack(
        x,
        up_w,
        down_w,
        gate,
        nb,
        ls,
        out_dtype=torch.float16,
        apply_gate=True,
        apply_relu=True,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
        save_up_proj=False,
    )


def _run_two_step(x, up_w, down_w, gate, nb, ls, gate_vals, row_idx, block_counts, max_rows):
    inter = fused_up_proj_gate_activation_sparse_triton_sortpack(
        x,
        up_w,
        gate,
        nb,
        ls,
        zero_init=False,
        out_dtype=torch.float16,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
    )
    fused_down_proj_sparse_triton_sortpack(
        inter,
        down_w,
        gate,
        nb,
        ls,
        out_dtype=torch.float16,
        gate_vals=gate_vals,
        row_idx=row_idx,
        block_counts=block_counts,
        max_rows=max_rows,
    )


def main():
    parser = argparse.ArgumentParser(description="Profile fused up+down vs two-step sort-pack kernels")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--line-size", type=int, default=64)
    parser.add_argument("--big-config", action="store_true", help="Use (batch=128, seq=128, hidden=4096, num_blocks=8, line_size=4096)")
    parser.add_argument("--sparsity", type=str, default="dynamic", help="'dynamic' or a float in [0,1]")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available – exiting")
        return

    if args.big_config:
        args.batch = 128
        args.seq = 128
        args.hidden = 4096
        args.num_blocks = 8
        args.line_size = 4096

    x, up_w, down_w, gate = _make_tensors(args.batch, args.seq, args.hidden, args.num_blocks, args.line_size, args.sparsity)
    gate_vals, row_idx, block_counts, max_rows = _preprocess_gate(gate, args.num_blocks)

    # Warm-up
    _run_fused(x, up_w, down_w, gate, args.num_blocks, args.line_size, gate_vals, row_idx, block_counts, max_rows)
    _run_two_step(x, up_w, down_w, gate, args.num_blocks, args.line_size, gate_vals, row_idx, block_counts, max_rows)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(args.iters):
            with record_function("FUSED_UP_DOWN_SORTPACK"):
                _run_fused(x, up_w, down_w, gate, args.num_blocks, args.line_size, gate_vals, row_idx, block_counts, max_rows)
            with record_function("TWO_STEP_SORTPACK"):
                _run_two_step(x, up_w, down_w, gate, args.num_blocks, args.line_size, gate_vals, row_idx, block_counts, max_rows)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.row_limit))


if __name__ == "__main__":
    main()


