import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from triton_kernels import fused_down_proj_triton
from triton_cast_kernel_gate_sortpack import fused_down_proj_sparse_triton_sortpack


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x_fp16, weight_fp16, gate_fp32.

    *sparsity* – fraction of zeros in gate (0.9 → 10 % nnz)
    """
    inter_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len
    x_fp16 = torch.randn(batch_seq_size, inter_size, device=device, dtype=torch.float16)
    weight_fp16 = torch.randn(inter_size, hidden_size, device=device, dtype=torch.float16)

    gate = torch.rand(batch_seq_size, num_blocks, device=device, dtype=torch.float32)
    if sparsity > 0.0:
        gate[torch.rand_like(gate) < sparsity] = 0.0

    # Zero out gated blocks to mimic real pipeline
    x_fp16 = x_fp16.view(batch_seq_size, num_blocks, line_size)
    x_fp16 = x_fp16 * gate.unsqueeze(-1).to(dtype=torch.float16)
    x_fp16 = x_fp16.view(batch_seq_size, inter_size)

    return x_fp16, weight_fp16, gate


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------

def _run_dense(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor, num_blocks: int, line_size: int,
               gate_vals: torch.Tensor = None, row_idx: torch.Tensor = None, block_counts: torch.Tensor = None, max_rows: int = None):
    """Triton dense helper – mirrors up_proj naming (_run_dense)."""
    fused_down_proj_triton(x, w, gate, num_blocks, line_size, out_dtype=torch.float16,
                          gate_vals=gate_vals, row_idx=row_idx, block_counts=block_counts, max_rows=max_rows)


def _run_pytorch(x: torch.Tensor, w: torch.Tensor):
    F.linear(x.float(), w.t().float())


# SortPack
def _run_sortpack(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor, num_blocks: int, line_size: int,
                  gate_vals: torch.Tensor = None, row_idx: torch.Tensor = None, block_counts: torch.Tensor = None, max_rows: int = None):
    fused_down_proj_sparse_triton_sortpack(x, w, gate, num_blocks, line_size, out_dtype=torch.float16,
                                          gate_vals=gate_vals, row_idx=row_idx, block_counts=block_counts, max_rows=max_rows)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile Triton vs PyTorch for Cast down-projection.",
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

    x_fp16, w_fp16, gate = _generate_tensors(
        args.batch, args.seq, args.hidden, args.num_blocks, args.line_size, args.sparsity
    )

    # Preprocess gate data once for all kernels
    def preprocess_gate(gate_tensor, num_blocks):
        mask = gate_tensor > 0
        block_counts = mask.sum(dim=0, dtype=torch.int32)
        max_rows = int(block_counts.max().item())
        if max_rows == 0:
            return None, None, block_counts, max_rows
        gate_vals_sorted, row_idx_sorted = torch.sort(gate_tensor, dim=0, descending=True)
        gate_vals = gate_vals_sorted[:max_rows, :].t().contiguous()
        row_idx = row_idx_sorted[:max_rows, :].t().contiguous().to(torch.int32)
        return gate_vals, row_idx, block_counts, max_rows

    gate_vals, row_idx, block_counts, max_rows = preprocess_gate(gate, args.num_blocks)

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

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    if args.profile_dense:
        _profile("triton_down_proj_dense", lambda: _run_dense(x_fp16, w_fp16, gate, args.num_blocks, args.line_size,
                                                              gate_vals=gate_vals, row_idx=row_idx, block_counts=block_counts, max_rows=max_rows))

    if args.profile_sortpack:
        _profile("triton_down_proj_sortpack", lambda: _run_sortpack(x_fp16, w_fp16, gate, args.num_blocks, args.line_size,
                                                                    gate_vals=gate_vals, row_idx=row_idx, block_counts=block_counts, max_rows=max_rows))

    if args.profile_pytorch:
        _profile("pytorch_down_proj", lambda: _run_pytorch(x_fp16, w_fp16))

    if args.profile_dense and args.profile_sortpack:
        print("\nTip: compare triton_down_proj_dense vs triton_down_proj_sortpack blocks above.")


if __name__ == "__main__":
    try:
        import triton  # noqa: F401
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    main() 