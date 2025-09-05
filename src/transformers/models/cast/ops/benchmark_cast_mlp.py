import argparse
import torch

# Local imports – run from cast root directory
#   cd src/transformers/models/cast/
#   python -m ops.benchmark_cast_mlp
from .cast_mlp_fused import cast_mlp_fused, _CastMLPFusedFunction
from .cast_mlp_fused_stream_compact import cast_mlp_fused_stream_compact, _CastMLPFusedStreamCompactFunction
from .debug_cast_mlp_fused import reference_cast_mlp_pytorch, _make_sparse_gate


def _make_sparse_gate_skewed(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9):
    """Build a gate tensor with skewed (concentrated) sparsity pattern.
    
    Instead of evenly distributing active gates across blocks, this creates
    maximum concentration by filling blocks sequentially (block 0 first, 
    then block 1, etc.) to stress-test load balancing in the adaptive kernel.
    
    Args:
        batch_seq_size: Total batch×seq positions 
        num_blocks: Number of blocks (NB)
        sparsity: Fraction of zeros (e.g. 0.9 → 10% non-zero)
        
    Returns:
        torch.float32 tensor (batch_seq_size, num_blocks) on CUDA
    """
    # Start with all zeros
    gate = torch.zeros(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    
    # Calculate how many total elements should be active
    total_elements = batch_seq_size * num_blocks
    active_elements = int(total_elements * (1.0 - sparsity))
    
    if active_elements == 0:
        return gate
    
    # Fill blocks greedily: all positions in block 0 first, then block 1, etc.
    elements_filled = 0
    
    for block_idx in range(num_blocks):
        if elements_filled >= active_elements:
            break
            
        # How many elements can we fill in this block?
        elements_in_this_block = min(batch_seq_size, active_elements - elements_filled)
        
        # Vectorized fill: set first `elements_in_this_block` positions in this block
        gate[:elements_in_this_block, block_idx] = torch.rand(
            elements_in_this_block, device="cuda", dtype=torch.float32
        )
            
        elements_filled += elements_in_this_block
    
    return gate



def benchmark_cast_mlp(num_iters: int = 100, sparsity = "dynamic", dtype: torch.dtype = torch.float16, measure_total_backward: bool = False):
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
        (128, 256, 4096, 256, 64),
        (128, 256, 4096, 128, 128),
        (128, 256, 4096, 64, 256),
        (128, 256, 4096, 32, 512),
        (128, 256, 4096, 16, 1024),
        (128, 256, 4096, 4, 4096),
    ]

    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        intermediate_size = num_blocks * line_size
        batch_seq_size = batch_size * seq_len

        # Calculate sparsity
        if sparsity == "dynamic" or sparsity == "dynamic-skewed":
            zeros_frac = 1.0 - (1.0 / num_blocks)
        else:
            try:
                zeros_frac = float(sparsity)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid --sparsity value '{sparsity}'. Use 'dynamic', 'dynamic-skewed', or a float between 0 and 1."
                ) from e
            zeros_frac = max(0.0, min(1.0, zeros_frac))

        sparsity_pattern = "skewed" if sparsity == "dynamic-skewed" else "even"
        cfg_str = (
            f"B={batch_size} S={seq_len} H={hidden_size} "
            f"NB={num_blocks} LS={line_size} | I={intermediate_size} | Sparsity={zeros_frac:.3f} ({sparsity_pattern})"
        )
        print(f"\nConfig: {cfg_str}  |  Iters: {num_iters}")

        # ------------------------------------------------------------------
        # Create random tensors (no gradients for forward-only benchmarking)
        # ------------------------------------------------------------------
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        # Create gate tensor with appropriate pattern
        if sparsity == "dynamic-skewed":
            gate = _make_sparse_gate_skewed(batch_seq_size, num_blocks, sparsity=zeros_frac).view(
                batch_size, seq_len, num_blocks
            )
        else:
            gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=zeros_frac).view(
                batch_size, seq_len, num_blocks
            )
        up_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
        down_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)

        # ------------------------------------------------------------------
        # PyTorch reference timing
        # ------------------------------------------------------------------
        with torch.no_grad():
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
        # Fused SortPack kernel timing
        # ------------------------------------------------------------------
        with torch.no_grad():
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
        print(f"Triton fused (SortPack): {fused_ms:.3f} ms | Speed-up: {speedup:.2f}×")

        # ------------------------------------------------------------------
        # Fused Stream Compact kernel timing
        # ------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(5):
                cast_mlp_fused_stream_compact(x, gate, up_weight, down_weight)
            torch.cuda.synchronize()

            t_start_fused_sc = torch.cuda.Event(enable_timing=True)
            t_end_fused_sc = torch.cuda.Event(enable_timing=True)
            t_start_fused_sc.record()
            for _ in range(num_iters):
                cast_mlp_fused_stream_compact(x, gate, up_weight, down_weight)
            t_end_fused_sc.record()
            torch.cuda.synchronize()
            fused_sc_ms = t_start_fused_sc.elapsed_time(t_end_fused_sc) / num_iters

        speedup_sc = ref_ms / fused_sc_ms if fused_sc_ms > 0.0 else float('inf')
        sortpack_vs_sc = fused_ms / fused_sc_ms if fused_sc_ms > 0.0 else float('inf')
        print(f"Triton fused (StreamCmpt): {fused_sc_ms:.3f} ms | Speed-up vs PyTorch: {speedup_sc:.2f}× | vs SortPack: {sortpack_vs_sc:.2f}×")

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

        # SortPack backward timing
        if measure_total_backward:
            # Triton fused backward - total autograd path (includes PyTorch overhead)
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
        else:
            # Direct kernel backward
            class MockContext:
                def __init__(self):
                    pass
                def save_for_backward(self, *tensors):
                    self.saved_tensors = tensors
            
            ctx = MockContext()
            fused_output = _CastMLPFusedFunction.forward(ctx, x_grad, gate_grad, up_weight_grad, down_weight_grad)
            grad_out = torch.tensor(1.0, device=fused_output.device, dtype=fused_output.dtype).expand_as(fused_output)
            
            for _ in range(5):
                _CastMLPFusedFunction.backward(ctx, grad_out)
            torch.cuda.synchronize()

            t_start_fused_bwd = torch.cuda.Event(enable_timing=True)
            t_end_fused_bwd = torch.cuda.Event(enable_timing=True)
            t_start_fused_bwd.record()
            for _ in range(num_iters):
                _CastMLPFusedFunction.backward(ctx, grad_out)
            t_end_fused_bwd.record()
            torch.cuda.synchronize()
            fused_bwd_ms = t_start_fused_bwd.elapsed_time(t_end_fused_bwd) / num_iters

        speedup_bwd = ref_bwd_ms / fused_bwd_ms if fused_bwd_ms > 0.0 else float('inf')
        kernel_type = "Total (autograd)" if measure_total_backward else "Direct kernel"
        print(f"Triton fused SortPack ({kernel_type}): {fused_bwd_ms:.3f} ms | Speed-up: {speedup_bwd:.2f}×")

        # Stream Compact backward timing
        if measure_total_backward:
            # Stream compact total autograd path
            fused_sc_output = cast_mlp_fused_stream_compact(x_grad, gate_grad, up_weight_grad, down_weight_grad)
            fused_sc_loss = fused_sc_output.sum()
            for _ in range(5):
                fused_sc_loss.backward(retain_graph=True)
            torch.cuda.synchronize()

            t_start_fused_sc_bwd = torch.cuda.Event(enable_timing=True)
            t_end_fused_sc_bwd = torch.cuda.Event(enable_timing=True)
            t_start_fused_sc_bwd.record()
            for _ in range(num_iters):
                fused_sc_loss.backward(retain_graph=True)
            t_end_fused_sc_bwd.record()
            torch.cuda.synchronize()
            fused_sc_bwd_ms = t_start_fused_sc_bwd.elapsed_time(t_end_fused_sc_bwd) / num_iters
        else:
            # Direct kernel backward
            ctx_sc = MockContext()
            fused_sc_output = _CastMLPFusedStreamCompactFunction.forward(ctx_sc, x_grad, gate_grad, up_weight_grad, down_weight_grad)
            grad_out_sc = torch.tensor(1.0, device=fused_sc_output.device, dtype=fused_sc_output.dtype).expand_as(fused_sc_output)
            
            for _ in range(5):
                _CastMLPFusedStreamCompactFunction.backward(ctx_sc, grad_out_sc)
            torch.cuda.synchronize()

            t_start_fused_sc_bwd = torch.cuda.Event(enable_timing=True)
            t_end_fused_sc_bwd = torch.cuda.Event(enable_timing=True)
            t_start_fused_sc_bwd.record()
            for _ in range(num_iters):
                _CastMLPFusedStreamCompactFunction.backward(ctx_sc, grad_out_sc)
            t_end_fused_sc_bwd.record()
            torch.cuda.synchronize()
            fused_sc_bwd_ms = t_start_fused_sc_bwd.elapsed_time(t_end_fused_sc_bwd) / num_iters

        speedup_sc_bwd = ref_bwd_ms / fused_sc_bwd_ms if fused_sc_bwd_ms > 0.0 else float('inf')
        sortpack_vs_sc_bwd = fused_bwd_ms / fused_sc_bwd_ms if fused_sc_bwd_ms > 0.0 else float('inf')
        print(f"Triton fused StreamCmpt ({kernel_type}): {fused_sc_bwd_ms:.3f} ms | Speed-up vs PyTorch: {speedup_sc_bwd:.2f}× | vs SortPack: {sortpack_vs_sc_bwd:.2f}×")

        # Combined forward + backward
        total_ref = ref_ms + ref_bwd_ms
        total_fused_sp = fused_ms + fused_bwd_ms
        total_fused_sc = fused_sc_ms + fused_sc_bwd_ms
        total_speedup_sp = total_ref / total_fused_sp if total_fused_sp > 0.0 else float('inf')
        total_speedup_sc = total_ref / total_fused_sc if total_fused_sc > 0.0 else float('inf')
        total_sp_vs_sc = total_fused_sp / total_fused_sc if total_fused_sc > 0.0 else float('inf')
        print(f"\nTotal (fwd + bwd):")
        print(f"PyTorch reference: {total_ref:.3f} ms")
        print(f"Triton SortPack:   {total_fused_sp:.3f} ms | Speed-up: {total_speedup_sp:.2f}×")
        print(f"Triton StreamCmpt: {total_fused_sc:.3f} ms | Speed-up vs PyTorch: {total_speedup_sc:.2f}× | vs SortPack: {total_sp_vs_sc:.2f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CAST fused MLP kernel")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per configuration")
    parser.add_argument(
        "--sparsity",
        default="dynamic",
        help="Fraction of zeros in gate (e.g., 0.9), 'dynamic' for even distribution (1 - 1/num_blocks), or 'dynamic-skewed' for concentrated load",
    )
    parser.add_argument(
        "--measure-total-backward",
        action="store_true",
        help="Measure total backward (autograd) instead of direct kernel backward",
    )
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

    benchmark_cast_mlp(
        num_iters=args.iters, 
        sparsity=args.sparsity, 
        dtype=dtype_map[args.dtype], 
        measure_total_backward=args.measure_total_backward
    ) 