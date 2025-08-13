import argparse
import torch

from .triton_cast_kernel_gate_sortpack_fused import (
    fused_up_down_proj_sparse_triton_sortpack,
)
from kernels.up_proj.triton_cast_kernel_gate_sortpack import (
    fused_up_proj_gate_activation_sparse_triton_sortpack,
)
from kernels.down_proj.triton_cast_kernel_gate_sortpack import (
    fused_down_proj_sparse_triton_sortpack,
)


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


def time_cuda(fn, iters: int = 100):
    # Warm-up
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream())
    for _ in range(iters):
        fn()
    end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused up+down vs two-step sort-pack kernels")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--sparsity",
        type=str,
        default="dynamic",
        help="Fraction of zeros in gate (e.g., 0.9) or 'dynamic' to use 1 - 1/num_blocks per config",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available – exiting")
        return

    torch.manual_seed(0)

    # Always use the full config suite (matches other kernels)
    configs = [
        (128, 128, 4096, 64, 64),
        (128, 128, 4096, 256, 128),
        (128, 128, 4096, 128, 256),
        (128, 128, 4096, 64, 512),
        (128, 128, 4096, 32, 1024),
        (128, 128, 4096, 8, 4096),
    ]

    print("\n=== Fused Up+Down Benchmark (SortPack) ===")
    for b, s, h, nb, ls in configs:
        bs = b * s
        inter = nb * ls

        x = torch.randn(bs, h, device="cuda", dtype=torch.float16)
        up_w = torch.randn(h, inter, device="cuda", dtype=torch.float16)
        down_w = torch.randn(inter, h, device="cuda", dtype=torch.float16)

        if args.sparsity == "dynamic":
            zeros_frac = 1.0 - (1.0 / nb)
        else:
            try:
                zeros_frac = float(args.sparsity)
            except ValueError as e:
                raise ValueError(
                    f"Invalid --sparsity value '{args.sparsity}'. Use 'dynamic' or a float between 0 and 1."
                ) from e
            zeros_frac = max(0.0, min(1.0, zeros_frac))

        gate = torch.rand(bs, nb, device="cuda", dtype=torch.float32)
        gate[torch.rand_like(gate) < zeros_frac] = 0.0

        gate_vals, row_idx, block_counts, max_rows = _preprocess_gate(gate, nb)

        # Prepare callables
        def fused_call():
            return fused_up_down_proj_sparse_triton_sortpack(
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

        def two_step_call():
            inter_act = fused_up_proj_gate_activation_sparse_triton_sortpack(
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
            return fused_down_proj_sparse_triton_sortpack(
                inter_act,
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

        for _ in range(5):
            fused_call()
        torch.cuda.synchronize()
        fused_ms = time_cuda(fused_call, iters=args.iters)

        for _ in range(5):
            two_step_call()
        torch.cuda.synchronize()
        two_step_ms = time_cuda(two_step_call, iters=args.iters)

        print(
            f"Config B={b} S={s} H={h} NB={nb} LS={ls} | Fused: {fused_ms:.3f} ms | Two-step: {two_step_ms:.3f} ms | Speedup: {two_step_ms / fused_ms:.2f}x"
        )


if __name__ == "__main__":
    main()


