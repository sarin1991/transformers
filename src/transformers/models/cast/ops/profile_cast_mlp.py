import argparse
from typing import Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Local imports – run from cast root directory
#   cd src/transformers/models/cast/
#   python -m ops.profile_cast_mlp
from ops.cast_mlp_fused import cast_mlp_fused
from ops.debug_cast_mlp_fused import reference_cast_mlp_pytorch, _make_sparse_gate


# -----------------------------------------------------------------------------
# Utility – tensor factory
# -----------------------------------------------------------------------------

def _generate_tensors(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_blocks: int,
    line_size: int,
    sparsity: float,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x, gate, up_weight, down_weight tensors for a single config."""
    intermediate_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=sparsity).view(
        batch_size, seq_len, num_blocks
    )
    up_weight = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)
    down_weight = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)

    return x, gate, up_weight, down_weight


# -----------------------------------------------------------------------------
# Runner helpers
# -----------------------------------------------------------------------------

def _run_fused(x, g, up_w, down_w):
    return cast_mlp_fused(x, g, up_w, down_w)


def _run_pytorch(x, g, up_w, down_w, compute_dtype=torch.float32):
    return reference_cast_mlp_pytorch(x, g, up_w, down_w, compute_dtype=compute_dtype)


def _run_fused_backward(output_template):
    output = output_template.detach().requires_grad_(True)
    loss = output.sum()
    loss.backward()


def _run_pytorch_backward(output_template):
    output = output_template.detach().requires_grad_(True)
    loss = output.sum()
    loss.backward()


# -----------------------------------------------------------------------------
# Main CLI entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile CAST fused MLP vs PyTorch baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--line-size", type=int, default=4096)
    parser.add_argument("--sparsity", type=float, default=0.9, help="Fraction of zeros in gate tensor")
    parser.add_argument("--steps", type=int, default=50, help="Profiler steps")
    parser.add_argument("--profile-fused", action="store_true", help="Profile Triton fused kernel")
    parser.add_argument("--profile-pytorch", action="store_true", help="Profile PyTorch reference")
    parser.add_argument("--profile-backward", action="store_true", help="Profile backward pass instead of forward")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype for forward pass",
    )
    args = parser.parse_args()

    # Ensure at least one target selected
    if not (args.profile_fused or args.profile_pytorch):
        args.profile_fused = True  # default

    if not torch.cuda.is_available():
        print("CUDA not available – exiting.")
        return

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    x, gate, up_w, down_w = _generate_tensors(
        args.batch,
        args.seq,
        args.hidden,
        args.num_blocks,
        args.line_size,
        args.sparsity,
        dtype=dtype,
    )
    
    if args.profile_backward:
        x = x.requires_grad_(True)
        gate = gate.requires_grad_(True)
        up_w = up_w.requires_grad_(True)
        down_w = down_w.requires_grad_(True)

    def _profile(name: str, fn):
        # Warm-up
        for _ in range(10):
            fn()
        torch.cuda.synchronize()

        activities = [ProfilerActivity.CUDA]
        print(f"\n===== Profiling {name} =====")
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            for _ in range(args.steps):
                with record_function(name):
                    fn()
            torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

    if args.profile_backward:
        if args.profile_fused:
            # Pre-compute forward pass output as template
            fused_output_template = _run_fused(x, gate, up_w, down_w)
            _profile("cast_mlp_fused_backward", lambda: _run_fused_backward(fused_output_template))

        if args.profile_pytorch:
            # Pre-compute forward pass output as template
            pytorch_output_template = _run_pytorch(x, gate, up_w, down_w, compute_dtype=dtype)
            _profile("cast_mlp_pytorch_backward", lambda: _run_pytorch_backward(pytorch_output_template))
    else:
        if args.profile_fused:
            _profile("cast_mlp_fused", lambda: _run_fused(x, gate, up_w, down_w))

        if args.profile_pytorch:
            _profile("cast_mlp_pytorch", lambda: _run_pytorch(x, gate, up_w, down_w, compute_dtype=dtype))


if __name__ == "__main__":
    try:
        import triton  # noqa: F401 – import to check availability and to align with other scripts
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it for full performance.")

    main() 