import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from triton_kernels import fused_weight_grad_triton
from triton_cast_kernel_gate_sortpack import fused_weight_grad_sparse_triton_sortpack


# -----------------------------------------------------------------------------
# Utility – create random tensors
# -----------------------------------------------------------------------------

def _generate_tensors(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_blocks: int,
    line_size: int,
    sparsity: float,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return intermediate_fp16, other_fp16, gate_fp32, intermediate_fp16_t.

    *sparsity* – fraction of zeros in gate (0.9 → 10 % nnz)
    """
    inter_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len
    intermediate_fp16 = torch.randn(batch_seq_size, inter_size, device=device, dtype=torch.float16)
    other_fp16 = torch.randn(batch_seq_size, hidden_size, device=device, dtype=torch.float16)

    gate = torch.rand(batch_seq_size, num_blocks, device=device, dtype=torch.float32)
    if sparsity > 0.0:
        gate[torch.rand_like(gate) < sparsity] = 0.0

    # Zero out fully gated rows to mimic real pipeline
    row_mask = (gate.view(-1, num_blocks).abs().sum(dim=1) != 0).view(batch_seq_size, 1)
    intermediate_fp16 = (intermediate_fp16 * row_mask.to(dtype=torch.float16)).contiguous()
    
    # Pre-transpose and make contiguous for PyTorch
    intermediate_fp16_t = intermediate_fp16.t().contiguous()
    
    # Ensure all tensors are contiguous
    other_fp16 = other_fp16.contiguous()
    gate = gate.contiguous()

    return intermediate_fp16, other_fp16, gate, intermediate_fp16_t


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------

def _run_dense(intermediate: torch.Tensor, other: torch.Tensor, num_blocks: int, line_size: int):
    """Triton dense helper – mirrors down_proj naming (_run_dense)."""
    fused_weight_grad_triton(intermediate, other, line_size, out_dtype=torch.float16)


def _run_pytorch(intermediate_t: torch.Tensor, other: torch.Tensor):
    intermediate_t @ other


# SortPack
def _run_sortpack(intermediate: torch.Tensor, other: torch.Tensor, gate: torch.Tensor, num_blocks: int, line_size: int):
    fused_weight_grad_sparse_triton_sortpack(intermediate, other, gate, num_blocks, line_size, out_dtype=torch.float16)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile Triton vs PyTorch for Cast weight-gradient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--line-size", type=int, default=64, help="Line size",)
    parser.add_argument(
        "--big-config",
        action="store_true",
        help="Shortcut – use (batch=128, seq=128, hidden=4096, num_blocks=8, line_size=4096)",
    )
    parser.add_argument("--sparsity", type=float, default=0.9, help="Fraction of zeros in gate tensor")
    parser.add_argument("--steps", type=int, default=50, help="Profiler steps")
    parser.add_argument("--profile-dense", action="store_true", help="Profile dense Triton helper")
    parser.add_argument("--profile-sortpack", action="store_true", help="Profile SortPack Triton helper")
    parser.add_argument("--profile-pytorch", action="store_true", help="Profile PyTorch baseline")
    args = parser.parse_args()

    # Apply --big-config preset (overrides individual size flags)
    if args.big_config:
        args.batch = 128
        args.seq = 128
        args.hidden = 4096
        args.num_blocks = 8
        args.line_size = 4096

    if not torch.cuda.is_available():
        print("CUDA not available – exiting.")
        return

    intermediate_fp16, other_fp16, gate, intermediate_fp16_t = _generate_tensors(
        args.batch, args.seq, args.hidden, args.num_blocks, args.line_size, args.sparsity
    )

    # Ensure at least one kernel selected
    if not (args.profile_dense or args.profile_sortpack or args.profile_pytorch):
        args.profile_dense = True  # default to dense

    def _profile(name: str, run_fn):
        # Warm-up
        for _ in range(10):
            run_fn()
        torch.cuda.synchronize()

        activities = [ProfilerActivity.CUDA]
        print(f"\n===== Profiling {name} =====")
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
            with_flops=True,
            profile_memory=True,
            with_modules=False,
        ) as prof:
            for _ in range(args.steps):
                with record_function(name):
                    run_fn()
            torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20, max_name_column_width=80))

    if args.profile_dense:
        _profile("triton_weight_grad_dense", lambda: _run_dense(intermediate_fp16, other_fp16, args.num_blocks, args.line_size))

    if args.profile_sortpack:
        _profile("triton_weight_grad_sortpack", lambda: _run_sortpack(intermediate_fp16, other_fp16, gate, args.num_blocks, args.line_size))

    if args.profile_pytorch:
        _profile("pytorch_weight_grad", lambda: _run_pytorch(intermediate_fp16_t, other_fp16))

    if args.profile_dense and args.profile_sortpack:
        print("\nTip: compare triton_weight_grad_dense vs triton_weight_grad_sortpack blocks above.")


if __name__ == "__main__":
    try:
        import triton  # noqa: F401
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    main()