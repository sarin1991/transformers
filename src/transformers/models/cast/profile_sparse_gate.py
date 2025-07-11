import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

# Import the dense & sparse helpers from the same package
from triton_kernels import (
    fused_up_proj_gate_activation_triton,
    fused_up_proj_gate_activation_sparse_triton,
)
from triton_cast_kernel import (
    fused_up_proj_gate_activation_sparse_triton_optimized as fused_up_proj_gate_activation_sparse_triton_opt,
)
from triton_cast_kernel_csr import (
    fused_up_proj_gate_activation_sparse_triton_csr as fused_up_proj_gate_activation_sparse_triton_csr,
)
# Unified CSR helper
from triton_cast_kernel_csr_unified import (
    fused_up_proj_gate_activation_sparse_triton_csr_unified as fused_up_proj_gate_activation_sparse_triton_csr_unified,
)


# -----------------------------------------------------------------------------
# Utility – random test data
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
    """Create fp16 input, fp16 weight/bias and a gate tensor with *sparsity*.

    *sparsity*  – fraction of **zeros** in the gate tensor (e.g. 0.9 → 10 % nnz).
    """

    inter_size = num_blocks * line_size

    x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(hidden_size, inter_size, device=device, dtype=torch.float16)
    b_fp16 = torch.randn(inter_size, device=device, dtype=torch.float16)

    gate = torch.rand(batch_size, seq_len, num_blocks, device=device, dtype=torch.float32)
    if sparsity > 0.0:
        mask_zero = torch.rand_like(gate) < sparsity
        gate[mask_zero] = 0.0

    return x_fp16, w_fp16, b_fp16, gate


# -----------------------------------------------------------------------------
# Profiling helpers
# -----------------------------------------------------------------------------

def _run_dense(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
):
    fused_up_proj_gate_activation_triton(x, w, b, gate, num_blocks, line_size)


def _run_sparse(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
):
    fused_up_proj_gate_activation_sparse_triton(
        x,
        w,
        b,
        gate,
        num_blocks,
        line_size,
    )


# Optimized sparse helper
def _run_opt(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    zero_init: bool,
):
    fused_up_proj_gate_activation_sparse_triton_opt(
        x,
        w,
        b,
        gate,
        num_blocks,
        line_size,
        zero_init=zero_init,
    )

# CSR sparse helper
def _run_csr(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    zero_init: bool,
):
    fused_up_proj_gate_activation_sparse_triton_csr(
        x,
        w,
        b,
        gate,
        num_blocks,
        line_size,
        zero_init=zero_init,
    )

# Unified CSR helper
def _run_csr_unified(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    zero_init: bool,
):
    fused_up_proj_gate_activation_sparse_triton_csr_unified(
        x,
        w,
        b,
        gate,
        num_blocks,
        line_size,
        zero_init=zero_init,
    )


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile dense vs sparse Triton helpers to locate bottlenecks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Default: large config used in benchmark_triton.py → (128, 128, 4096, 64, 64)
    parser.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--seq", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden size H (default: 4096)")
    parser.add_argument("--blocks", type=int, default=64, help="Number of blocks NB (default: 64)")
    parser.add_argument("--line", type=int, default=64, help="Line size L (default: 64)")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.9,
        help="Fraction of zeros in the gate tensor (0.9 → 10 % nnz)",
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Iterations per helper inside the profiler",
    )
    parser.add_argument(
        "--profile-dense", action="store_true", help="Include dense helper in the profile",
    )
    parser.add_argument(
        "--profile-sparse", action="store_true", help="Include baseline sparse helper in the profile",
    )
    parser.add_argument(
        "--profile-optimized", action="store_true", help="Include optimized sparse helper in the profile",
    )
    parser.add_argument(
        "--profile-csr", action="store_true", help="Include CSR sparse helper in the profile",
    )
    parser.add_argument(
        "--profile-csr-unified", action="store_true", help="Include unified CSR sparse helper in the profile",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=40,
        help="Rows to display in the profiler table",
    )

    # Convenience preset for the large line-size config used in the benchmark
    parser.add_argument(
        "--big-config",
        action="store_true",
        help="Shortcut – use (batch=128, seq=128, hidden=4096, blocks=8, line=4096)",
    )

    parser.add_argument(
        "--zero-init",
        action="store_true",
        help="Pre-zero the output buffer in sparse helpers (default: off for fastest path)",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Apply --big-config preset (overrides individual size flags)
    # --------------------------------------------------------------
    if args.big_config:
        args.batch = 128
        args.seq = 128
        args.hidden = 4096
        args.blocks = 8
        args.line = 4096

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton profiling, but torch.cuda.is_available() == False")

    # ------------------------------------------------------------------
    # Profiling mode selection
    # ------------------------------------------------------------------
    # By **default** we always profile the sparse helper.  If neither helper
    # is requested explicitly, we profile *both*.  This guarantees that the
    # sparse path is included unless the script is modified to disable it.
    if not (args.profile_dense or args.profile_sparse or args.profile_optimized or args.profile_csr or args.profile_csr_unified):
        # No flags → profile all helpers
        args.profile_dense = True
        args.profile_sparse = True
        args.profile_optimized = True
        args.profile_csr = True
        args.profile_csr_unified = True
    elif args.profile_dense and not (args.profile_sparse or args.profile_optimized or args.profile_csr or args.profile_csr_unified):
        # User asked for dense only – still include all sparse paths by default
        args.profile_sparse = True
        args.profile_optimized = True
        args.profile_csr = True
        args.profile_csr_unified = True
    elif args.profile_sparse and not (args.profile_optimized or args.profile_csr or args.profile_csr_unified):
        # baseline sparse only → also add optimized and csr for comparison
        args.profile_optimized = True
        args.profile_csr = True
        args.profile_csr_unified = True

    device = torch.device("cuda")

    x_fp16, w_fp16, b_fp16, gate_fp32 = _generate_tensors(
        args.batch,
        args.seq,
        args.hidden,
        args.blocks,
        args.line,
        args.sparsity,
        device=device,
    )

    # Warm-up each helper once outside the profiler
    if args.profile_dense:
        _run_dense(x_fp16, w_fp16, b_fp16, gate_fp32, args.blocks, args.line)
    if args.profile_sparse:
        _run_sparse(
            x_fp16,
            w_fp16,
            b_fp16,
            gate_fp32,
            args.blocks,
            args.line,
        )
    if args.profile_optimized:
        _run_opt(
            x_fp16,
            w_fp16,
            b_fp16,
            gate_fp32,
            args.blocks,
            args.line,
            args.zero_init,
        )
    if args.profile_csr:
        _run_csr(
            x_fp16,
            w_fp16,
            b_fp16,
            gate_fp32,
            args.blocks,
            args.line,
            args.zero_init,
        )
    if args.profile_csr_unified:
        _run_csr_unified(
            x_fp16,
            w_fp16,
            b_fp16,
            gate_fp32,
            args.blocks,
            args.line,
            args.zero_init,
        )
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(args.iters):
            if args.profile_dense:
                with record_function("DENSE_HELPER"):
                    _run_dense(x_fp16, w_fp16, b_fp16, gate_fp32, args.blocks, args.line)
            if args.profile_sparse:
                with record_function("SPARSE_HELPER"):
                    _run_sparse(
                        x_fp16,
                        w_fp16,
                        b_fp16,
                        gate_fp32,
                        args.blocks,
                        args.line,
                    )
            if args.profile_optimized:
                with record_function("OPTIMIZED_HELPER"):
                    _run_opt(
                        x_fp16,
                        w_fp16,
                        b_fp16,
                        gate_fp32,
                        args.blocks,
                        args.line,
                        args.zero_init,
                    )
            if args.profile_csr:
                with record_function("CSR_HELPER"):
                    _run_csr(
                        x_fp16,
                        w_fp16,
                        b_fp16,
                        gate_fp32,
                        args.blocks,
                        args.line,
                        args.zero_init,
                    )
            if args.profile_csr_unified:
                with record_function("CSR_UN_HELPER"):
                    _run_csr_unified(
                        x_fp16,
                        w_fp16,
                        b_fp16,
                        gate_fp32,
                        args.blocks,
                        args.line,
                        args.zero_init,
                    )
        torch.cuda.synchronize()

    print("\n========= PROFILER SUMMARY =========")
    print(
        prof.key_averages()
        .table(sort_by="cuda_time_total", row_limit=args.row_limit, header="Profiler Key Averages")
    )

    if args.profile_dense and args.profile_sparse:
        print(
            "\nTip: look for DENSE_HELPER vs SPARSE_HELPER blocks in the table to compare kernel counts/time.",
        )
    if args.profile_optimized:
        print("Also compare OPTIMIZED_HELPER for the improved path.")
    if args.profile_csr:
        print("CSR_HELPER shows the single-kernel CSR path.")


if __name__ == "__main__":
    main() 