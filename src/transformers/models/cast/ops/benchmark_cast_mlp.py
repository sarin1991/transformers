import argparse
import torch

# Local imports – run from cast root directory
#   cd src/transformers/models/cast/
#   python -m ops.benchmark_cast_mlp
from ops.cast_mlp_fused import cast_mlp_fused
from ops.debug_cast_mlp_fused import reference_cast_mlp_pytorch, _make_sparse_gate



def benchmark_cast_mlp(num_iters: int = 100, sparsity: float = 0.9, dtype: torch.dtype = torch.float16):
    """Benchmark CAST fused MLP against a PyTorch reference implementation.

    The reference is implemented in *debug_cast_mlp_fused.py* to keep results consistent across
    scripts.  The benchmark closely mirrors *kernels/down_proj/benchmark_triton.py*.
    """

    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("\n=== CAST MLP Performance Benchmark ===")

    # (batch, seq_len, hidden_size, num_blocks, line_size)
    configs = [
        (4, 8, 128, 4, 32),
        (8, 16, 256, 8, 32),
        (128, 32, 512, 8, 64),
        (128, 256, 4096, 64, 64),
        (128, 256, 4096, 256, 128),
        (128, 256, 4096, 128, 256),
        (128, 256, 4096, 64, 512),
        (128, 256, 4096, 32, 1024),
        (128, 256, 4096, 8, 4096),
    ]

    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        intermediate_size = num_blocks * line_size
        batch_seq_size = batch_size * seq_len

        cfg_str = (
            f"B={batch_size} S={seq_len} H={hidden_size} "
            f"NB={num_blocks} LS={line_size} | I={intermediate_size}"
        )
        print(f"\nConfig: {cfg_str}  |  Iters: {num_iters}")

        # ------------------------------------------------------------------
        # Create random tensors
        # ------------------------------------------------------------------
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=sparsity).view(
            batch_size, seq_len, num_blocks
        )
        up_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
        down_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)

        # ------------------------------------------------------------------
        # PyTorch reference timing
        # ------------------------------------------------------------------
        for _ in range(5):
            reference_cast_mlp_pytorch(x, gate, up_weight, down_weight, compute_dtype=dtype)
        torch.cuda.synchronize()

        t_start_ref = torch.cuda.Event(enable_timing=True)
        t_end_ref = torch.cuda.Event(enable_timing=True)
        t_start_ref.record()
        for _ in range(num_iters):
            reference_cast_mlp_pytorch(x, gate, up_weight, down_weight, compute_dtype=dtype)
        t_end_ref.record()
        torch.cuda.synchronize()
        ref_ms = t_start_ref.elapsed_time(t_end_ref) / num_iters

        print(f"PyTorch reference: {ref_ms:.3f} ms")

        # ------------------------------------------------------------------
        # Fused kernel timing
        # ------------------------------------------------------------------
        for _ in range(5):
            cast_mlp_fused(x, gate, up_weight, down_weight)
        torch.cuda.synchronize()

        t_start_fused = torch.cuda.Event(enable_timing=True)
        t_end_fused = torch.cuda.Event(enable_timing=True)
        t_start_fused.record()
        for _ in range(num_iters):
            cast_mlp_fused(x, gate, up_weight, down_weight)
        t_end_fused.record()
        torch.cuda.synchronize()
        fused_ms = t_start_fused.elapsed_time(t_end_fused) / num_iters

        speedup = ref_ms / fused_ms if fused_ms > 0.0 else float('inf')
        print(f"Triton fused:     {fused_ms:.3f} ms | Speed-up: {speedup:.2f}×")

        # ------------------------------------------------------------------
        # Backward pass timing
        # ------------------------------------------------------------------
        print(f"\nBackward pass:")
        
        # Create tensors that require gradients
        x_grad = x.clone().requires_grad_(True)
        gate_grad = gate.clone().requires_grad_(True)
        up_weight_grad = up_weight.clone().requires_grad_(True)
        down_weight_grad = down_weight.clone().requires_grad_(True)
        ref_output = reference_cast_mlp_pytorch(x_grad, gate_grad, up_weight_grad, down_weight_grad, compute_dtype=dtype)
        ref_loss = ref_output.sum()

        # PyTorch reference backward
        for _ in range(5):
            ref_loss.backward(retain_graph=True)
        torch.cuda.synchronize()

        t_start_ref_bwd = torch.cuda.Event(enable_timing=True)
        t_end_ref_bwd = torch.cuda.Event(enable_timing=True)
        t_start_ref_bwd.record()
        for _ in range(num_iters):
            ref_loss.backward(retain_graph=True)
        t_end_ref_bwd.record()
        torch.cuda.synchronize()
        ref_bwd_ms = t_start_ref_bwd.elapsed_time(t_end_ref_bwd) / num_iters

        print(f"PyTorch reference: {ref_bwd_ms:.3f} ms")

        # Triton fused backward
        fused_output = cast_mlp_fused(x_grad, gate_grad, up_weight_grad, down_weight_grad)
        fused_loss = fused_output.sum()
        for _ in range(5):
            fused_loss.backward(retain_graph=True)
        torch.cuda.synchronize()

        t_start_fused_bwd = torch.cuda.Event(enable_timing=True)
        t_end_fused_bwd = torch.cuda.Event(enable_timing=True)
        t_start_fused_bwd.record()
        for _ in range(num_iters):
            fused_loss.backward(retain_graph=True)
        t_end_fused_bwd.record()
        torch.cuda.synchronize()
        fused_bwd_ms = t_start_fused_bwd.elapsed_time(t_end_fused_bwd) / num_iters

        speedup_bwd = ref_bwd_ms / fused_bwd_ms if fused_bwd_ms > 0.0 else float('inf')
        print(f"Triton fused:     {fused_bwd_ms:.3f} ms | Speed-up: {speedup_bwd:.2f}×")

        # Combined forward + backward
        total_ref = ref_ms + ref_bwd_ms
        total_fused = fused_ms + fused_bwd_ms
        total_speedup = total_ref / total_fused if total_fused > 0.0 else float('inf')
        print(f"\nTotal (fwd + bwd):")
        print(f"PyTorch reference: {total_ref:.3f} ms")
        print(f"Triton fused:     {total_fused:.3f} ms | Speed-up: {total_speedup:.2f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CAST fused MLP kernel")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per configuration")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Fraction of zeros in gate tensor")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype for forward pass",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    try:
        import triton  # noqa: F401 – ensure Triton is available
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it for full performance.")

    benchmark_cast_mlp(num_iters=args.iters, sparsity=args.sparsity, dtype=dtype_map[args.dtype]) 