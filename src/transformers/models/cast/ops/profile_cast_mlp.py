import argparse
from typing import Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Local imports – run from cast root directory
#   cd src/transformers/models/cast/
#   python -m ops.profile_cast_mlp
from ops.cast_mlp_fused import cast_mlp_fused, _CastMLPFusedFunction
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
    sparsity,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x, gate, up_weight, down_weight tensors for a single config."""
    intermediate_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len

    # Calculate sparsity
    if sparsity == "dynamic":
        zeros_frac = 1.0 - (1.0 / num_blocks)
    else:
        try:
            zeros_frac = float(sparsity)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid --sparsity value '{sparsity}'. Use 'dynamic' or a float between 0 and 1."
            ) from e
        zeros_frac = max(0.0, min(1.0, zeros_frac))

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=zeros_frac).view(
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

def _run_fused_no_grad(x, g, up_w, down_w):
    with torch.no_grad():
        return cast_mlp_fused(x, g, up_w, down_w)

def _run_pytorch(x, g, up_w, down_w, compute_dtype=torch.float32):
    return reference_cast_mlp_pytorch(x, g, up_w, down_w, compute_dtype=compute_dtype)

def _run_pytorch_no_grad(x, g, up_w, down_w, compute_dtype=torch.float32):
    with torch.no_grad():
        return reference_cast_mlp_pytorch(x, g, up_w, down_w, compute_dtype=compute_dtype)


def _run_fused_backward(loss):
    loss.backward(retain_graph=True)


def _run_pytorch_backward(loss):
    loss.backward(retain_graph=True)


def _setup_direct_backward_context(x, gate, up_w, down_w):
    """Setup context for direct kernel backward profiling."""
    # Create mock context
    class MockContext:
        def __init__(self):
            pass
            
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors
    
    ctx = MockContext()
    
    # Run actual forward pass to populate context with real intermediate tensors
    fused_output = _CastMLPFusedFunction.forward(ctx, x, gate, up_w, down_w)
    
    # Create gradient output identical to fused_loss.sum() case
    grad_out = torch.tensor(1.0, device=fused_output.device, dtype=fused_output.dtype).expand_as(fused_output)
    
    return ctx, grad_out


def _run_direct_kernel_backward(ctx, grad_out):
    """Direct kernel backward call for profiling."""
    _CastMLPFusedFunction.backward(ctx, grad_out)


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
    parser.add_argument(
        "--sparsity",
        default="dynamic",
        help="Fraction of zeros in gate (e.g., 0.9) or 'dynamic' to use 1 - 1/num_blocks per config",
    )
    parser.add_argument("--steps", type=int, default=50, help="Profiler steps")
    parser.add_argument("--profile-fused", action="store_true", help="Profile Triton fused kernel")
    parser.add_argument("--profile-pytorch", action="store_true", help="Profile PyTorch reference")
    parser.add_argument("--forward-grad", action="store_true", help="Profile forward pass with gradient computation")
    parser.add_argument("--profile-backward", action="store_true", help="Profile backward pass instead of forward")
    parser.add_argument(
        "--measure-total-backward",
        action="store_true",
        help="Measure total backward (autograd) instead of direct kernel backward when profiling backward",
    )
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
    
    if args.profile_backward or args.forward_grad:
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

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=120))

    if args.profile_backward:
        if args.profile_fused:
            if args.measure_total_backward:
                # Total autograd path (includes PyTorch overhead)
                fused_output_template = _run_fused(x, gate, up_w, down_w)
                fused_loss = fused_output_template.sum()
                _profile("cast_mlp_fused_backward_total", lambda: _run_fused_backward(fused_loss))
            else:
                # Direct kernel backward (no PyTorch autograd overhead)
                ctx, grad_out = _setup_direct_backward_context(x, gate, up_w, down_w)
                _profile("cast_mlp_fused_backward_kernel", lambda: _run_direct_kernel_backward(ctx, grad_out))

        if args.profile_pytorch:
            # PyTorch reference always uses total autograd path
            pytorch_output_template = _run_pytorch(x, gate, up_w, down_w, compute_dtype=dtype)
            pytorch_loss = pytorch_output_template.sum()
            _profile("cast_mlp_pytorch_backward", lambda: _run_pytorch_backward(pytorch_loss))
    elif args.forward_grad:
        # Forward pass with gradient computation (old default behavior)
        if args.profile_fused:
            _profile("cast_mlp_fused_forward_grad", lambda: _run_fused(x, gate, up_w, down_w))

        if args.profile_pytorch:
            _profile("cast_mlp_pytorch_forward_grad", lambda: _run_pytorch(x, gate, up_w, down_w, compute_dtype=dtype))
    else:
        # Default: Forward pass with no gradient computation (fastest)
        if args.profile_fused:
            _profile("cast_mlp_fused_forward", lambda: _run_fused_no_grad(x, gate, up_w, down_w))

        if args.profile_pytorch:
            _profile("cast_mlp_pytorch_forward", lambda: _run_pytorch_no_grad(x, gate, up_w, down_w, compute_dtype=dtype))


if __name__ == "__main__":
    try:
        import triton  # noqa: F401 – import to check availability and to align with other scripts
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it for full performance.")

    main() 