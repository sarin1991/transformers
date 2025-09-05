import torch
import argparse
import time

from .triton_stream_compact_index import create_stream_compact_index
from .triton_stream_compact_index_fused import create_stream_compact_index_adaptive, should_use_fused_kernel


def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9):
    """Build a gate tensor with given sparsity on CUDA.
    
    sparsity denotes the fraction of zeros (e.g. 0.9 → 10% non-zero).
    Returns a torch.float32 tensor on the current CUDA device.
    """
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    if sparsity > 0.0:
        mask = torch.rand_like(gate) < sparsity  # True for zeros
        gate[mask] = 0.0
    return gate


def benchmark_indexing_kernels(num_iters: int = 100, sparsity: str = "dynamic", verbose: bool = False):
    """Benchmark original vs adaptive stream compact indexing kernel performance.
    
    Focus on small batch sizes where the fused kernel should show speedups.
    """
    
    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("\n=== Stream Compact Indexing Performance Benchmark ===")
    
    # Test configs focused on various scenarios
    # Format: (batch_size, seq_len, num_blocks)
    configs = [
        # Decoding scenarios (small batch sizes) 
        (1, 1, 64),     # 1×64 - typical decoding
        (1, 1, 128),    # 1×128 - larger blocks  
        (1, 1, 256),    # 1×256 - even larger
        (2, 1, 128),    # 2×128 - small batch
        (4, 1, 64),     # 4×64 - small batch
        (8, 1, 32),     # 8×32 - small batch
        (16, 1, 32),    # 16×32 - borderline
        (32, 1, 32),    # 32×32 - borderline
        
        # Larger cases 
        (64, 1, 64),    # 64×64 - larger case
        (128, 1, 32),   # 128×32 - large batch
        (128, 2, 64),   # 256×64 - training-like
        (256, 4, 32),   # 1024×32 - large case
    ]
    
    for batch_size, seq_len, num_blocks in configs:
        batch_seq_size = batch_size * seq_len
        
        # Calculate sparsity
        if sparsity == "dynamic":
            zeros_frac = 1.0 - (1.0 / num_blocks)  # Typical MoE sparsity
        else:
            try:
                zeros_frac = float(sparsity)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid --sparsity value '{sparsity}'. Use 'dynamic' or a float between 0 and 1."
                ) from e
            zeros_frac = max(0.0, min(1.0, zeros_frac))
        
        # Create gate tensor
        gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=zeros_frac)
        
        cfg = f"BS={batch_seq_size:4d} × NB={num_blocks:3d} | Sparsity={zeros_frac:.3f}"
        print(f"\nConfig: {cfg} | Iters: {num_iters}")
        
        if verbose:
            active_elements = (gate > 0).sum().item()
            print(f"  Active elements: {active_elements}/{gate.numel()} ({100*(1-zeros_frac):.1f}%)")
        
        # ---- Original Kernel Timing ----
        # Warmup
        for _ in range(5):
            _ = create_stream_compact_index(gate)
        torch.cuda.synchronize()
        
        start_orig = torch.cuda.Event(enable_timing=True)
        end_orig = torch.cuda.Event(enable_timing=True)
        start_orig.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = create_stream_compact_index(gate)
        end_orig.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        orig_ms = start_orig.elapsed_time(end_orig) / num_iters
        
        # ---- Adaptive Kernel Timing ----  
        # Warmup
        for _ in range(5):
            _ = create_stream_compact_index_adaptive(gate)
        torch.cuda.synchronize()
        
        start_adaptive = torch.cuda.Event(enable_timing=True)
        end_adaptive = torch.cuda.Event(enable_timing=True)
        start_adaptive.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = create_stream_compact_index_adaptive(gate)
        end_adaptive.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        adaptive_ms = start_adaptive.elapsed_time(end_adaptive) / num_iters
        
        # Results
        speedup = orig_ms / adaptive_ms if adaptive_ms > 0 else 1.0
        speedup_str = f"{speedup:.2f}×" if speedup > 1.0 else f"{1/speedup:.2f}× slower"
        
        print(f"Original:  {orig_ms:.3f} ms")
        print(f"Adaptive:  {adaptive_ms:.3f} ms | Speed-up: {speedup_str}")
        
        # Color-code significant speedups/slowdowns
        if speedup >= 1.5:
            print(f"  🟢 Significant speedup!")
        elif speedup <= 0.8:
            print(f"  🟡 Slower than original")


def benchmark_batch_scaling(num_iters: int = 100, sparsity: float = 0.9, num_blocks: int = 64):
    """Show how performance scales with batch size for fixed num_blocks."""
    
    print(f"\n=== Batch Size Scaling (NB={num_blocks}, Sparsity={sparsity:.3f}) ===")
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    for batch_size in batch_sizes:
        gate = _make_sparse_gate(batch_size, num_blocks, sparsity=sparsity)
        uses_fused = should_use_fused_kernel(batch_size, num_blocks, gate.device)
        
        # Quick timing
        # Original
        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = create_stream_compact_index(gate)
        torch.cuda.synchronize()
        orig_time = (time.perf_counter() - start_time) * 1000 / num_iters
        
        # Adaptive  
        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = create_stream_compact_index_adaptive(gate)
        torch.cuda.synchronize()
        adaptive_time = (time.perf_counter() - start_time) * 1000 / num_iters
        
        speedup = orig_time / adaptive_time if adaptive_time > 0 else 1.0
        kernel_used = "FUSED" if uses_fused else "FALLBACK"
        
        print(f"BS={batch_size:3d}: {orig_time:.3f}ms → {adaptive_time:.3f}ms ({speedup:.2f}×) [{kernel_used}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Stream Compact Indexing kernel")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per config")  
    parser.add_argument("--sparsity", type=str, default="dynamic", help="Sparsity for the gate (e.g., 0.9, 'dynamic')")
    parser.add_argument("--batch-scaling", action="store_true", help="Run batch size scaling benchmark")
    args = parser.parse_args()
    
    try:
        import triton  # noqa: F401
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)
    
    # Main benchmark
    benchmark_indexing_kernels(
        num_iters=args.iters, 
        sparsity=args.sparsity
    )
    
    # Optional batch scaling benchmark
    if args.batch_scaling:
        sparsity_val = 0.9 if args.sparsity == "dynamic" else float(args.sparsity)
        benchmark_batch_scaling(
            num_iters=args.iters//2,  # Fewer iters for scaling test
            sparsity=sparsity_val
        )