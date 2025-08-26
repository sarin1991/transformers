import torch
import torch.nn.functional as F
from .triton_kernels import (
    fused_up_proj_gate_activation_triton,
    fused_up_proj_gate_activation_sparse_triton,
)
from .triton_cast_kernel import (
    fused_up_proj_gate_activation_sparse_triton_optimized as fused_up_proj_gate_activation_sparse_triton_opt,
)
from .triton_cast_kernel_csr import (
    fused_up_proj_gate_activation_sparse_triton_csr as fused_up_proj_gate_activation_sparse_triton_csr,
)  # New CSR helper
from .triton_cast_kernel_gate_sortpack import (
    fused_up_proj_gate_activation_sparse_triton_sortpack as fused_up_proj_gate_activation_sparse_triton_sortpack,
)
from .triton_cast_kernel_stream_compact import (
    fused_up_proj_gate_activation_sparse_triton_stream_compact as fused_up_proj_gate_activation_sparse_triton_stream_compact,
)
from kernels.stream_compact_index import create_stream_compact_index
import argparse


def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9):
    """Build a gate tensor with given sparsity on CUDA.
    
    sparsity denotes the fraction of zeros (e.g. 0.9 → 10 % non-zero).
    Returns a torch.float32 tensor on the current CUDA device.
    """
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    if sparsity > 0.0:
        mask = torch.rand_like(gate) < sparsity  # True for zeros
        gate[mask] = 0.0
    return gate


def benchmark_fused_vs_pytorch(num_iters: int = 100, run_all: bool = False, sparsity: str = "dynamic"):
    """Time PyTorch reference vs Triton fused kernel on several shapes."""
    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("\n=== Performance Benchmark (fp16 inputs, fp32 gate/output) ===")

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
        
        cfg = f"{batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size} | Sparsity={zeros_frac:.3f}"
        print(f"\nConfig: {cfg}  |  Iters: {num_iters}")

        # Random tensors
        batch_seq_size = batch_size * seq_len
        x_fp16 = torch.randn(batch_seq_size, hidden_size, device="cuda", dtype=torch.float16)
        up_weight_fp16 = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
        
        # Create sparse gate with calculated sparsity
        gate_fp32 = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=zeros_frac)

        # ---- PyTorch timing ----
        # Warm-up for PyTorch (5 runs)
        for _ in range(5):
            up_proj_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16)).float()
            up_proj_reshaped = up_proj_fp16.view(batch_seq_size, num_blocks, line_size)
            _ = (up_proj_reshaped * gate_fp32.unsqueeze(-1)).view(batch_seq_size, intermediate_size)
        torch.cuda.synchronize()

        start_pt = torch.cuda.Event(enable_timing=True)
        end_pt = torch.cuda.Event(enable_timing=True)
        start_pt.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            up_proj_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16)).float()
            up_proj_reshaped = up_proj_fp16.view(batch_seq_size, num_blocks, line_size)
            _ = (up_proj_reshaped * gate_fp32.unsqueeze(-1)).view(batch_seq_size, intermediate_size)
        end_pt.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch_ms = start_pt.elapsed_time(end_pt) / num_iters

        print(f"PyTorch: {torch_ms:.3f} ms")
        
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

        gate_vals_dense, row_idx_dense, block_counts_dense, max_rows_dense = preprocess_gate(gate_fp32, num_blocks)
        
        # Preprocess stream compact mappings once for all stream compact kernels
        def preprocess_stream_compact(gate_tensor):
            """Create stream compact mappings once for reuse"""
            return create_stream_compact_index(gate_tensor)
        
        if run_all:
            # ---- Triton timing (autotuned) ----
            
            for _ in range(5):
                _ = fused_up_proj_gate_activation_triton(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_fp32,
                    num_blocks,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_dense,
                    row_idx=row_idx_dense,
                    block_counts=block_counts_dense,
                    max_rows=max_rows_dense,
                )
            torch.cuda.synchronize()

            start_tri = torch.cuda.Event(enable_timing=True)
            end_tri   = torch.cuda.Event(enable_timing=True)
            start_tri.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_up_proj_gate_activation_triton(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_fp32,
                    num_blocks,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_dense,
                    row_idx=row_idx_dense,
                    block_counts=block_counts_dense,
                    max_rows=max_rows_dense,
                )
            end_tri.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            triton_ms = start_tri.elapsed_time(end_tri) / num_iters

            print(f"Triton (autotuned): {triton_ms:.3f} ms | Speed-up: {torch_ms/triton_ms:.2f}x")

        # Create sparse gate data for sparse benchmarks using the same sparsity
        gate_sparse = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=zeros_frac)
        gate_vals_sparse, row_idx_sparse, block_counts_sparse, max_rows_sparse = preprocess_gate(gate_sparse, num_blocks)
        
        # Preprocess stream compact mappings for sparse gate
        mappings_sparse = preprocess_stream_compact(gate_sparse)

        if run_all:
            # ---- Sparse benchmark (10% non-zero gate) ----

            # Warm-up sparse helper (5 runs)
            for _ in range(5):
                _ = fused_up_proj_gate_activation_sparse_triton(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_sparse,
                    num_blocks,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_sparse,
                    row_idx=row_idx_sparse,
                    block_counts=block_counts_sparse,
                    max_rows=max_rows_sparse,
                )
            torch.cuda.synchronize()

            start_sp = torch.cuda.Event(enable_timing=True)
            end_sp   = torch.cuda.Event(enable_timing=True)
            start_sp.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_up_proj_gate_activation_sparse_triton(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_sparse,
                    num_blocks,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_sparse,
                    row_idx=row_idx_sparse,
                    block_counts=block_counts_sparse,
                    max_rows=max_rows_sparse,
                )
            end_sp.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            sparse_ms = start_sp.elapsed_time(end_sp) / num_iters

            print(f"Triton (sparse helper, {zeros_frac:.1%} nnz): {sparse_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/sparse_ms:.2f}x")

            # ---- Optimized sparse benchmark (10% non-zero gate) ----
            for _ in range(5):
                _ = fused_up_proj_gate_activation_sparse_triton_opt(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_sparse,
                    num_blocks,
                    line_size,
                    zero_init=False,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_sparse,
                    row_idx=row_idx_sparse,
                    block_counts=block_counts_sparse,
                    max_rows=max_rows_sparse,
                )
            torch.cuda.synchronize()

            start_opt = torch.cuda.Event(enable_timing=True)
            end_opt   = torch.cuda.Event(enable_timing=True)
            start_opt.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_up_proj_gate_activation_sparse_triton_opt(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_sparse,
                    num_blocks,
                    line_size,
                    zero_init=False,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_sparse,
                    row_idx=row_idx_sparse,
                    block_counts=block_counts_sparse,
                    max_rows=max_rows_sparse,
                )
            end_opt.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            opt_ms = start_opt.elapsed_time(end_opt) / num_iters

            print(f"Triton (optimized sparse, {zeros_frac:.1%} nnz): {opt_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/opt_ms:.2f}x | Speed-up vs baseline sparse: {sparse_ms/opt_ms:.2f}x")

            # ---- CSR sparse benchmark ({zeros_frac:.1%} non-zero gate) ----
            for _ in range(5):
                _ = fused_up_proj_gate_activation_sparse_triton_csr(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_sparse,
                    num_blocks,
                    line_size,
                    zero_init=False,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_sparse,
                    row_idx=row_idx_sparse,
                    block_counts=block_counts_sparse,
                    max_rows=max_rows_sparse,
                )
            torch.cuda.synchronize()

            start_csr = torch.cuda.Event(enable_timing=True)
            end_csr   = torch.cuda.Event(enable_timing=True)
            start_csr.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_up_proj_gate_activation_sparse_triton_csr(
                    x_fp16,
                    up_weight_fp16.t(),
                    gate_sparse,
                    num_blocks,
                    line_size,
                    zero_init=False,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals_sparse,
                    row_idx=row_idx_sparse,
                    block_counts=block_counts_sparse,
                    max_rows=max_rows_sparse,
                )
            end_csr.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            csr_ms = start_csr.elapsed_time(end_csr) / num_iters

        # ---- SortPack benchmark (always) ----
        # Reuse preprocessed sparse gate data

        for _ in range(5):
            _ = fused_up_proj_gate_activation_sparse_triton_sortpack(
                x_fp16,
                up_weight_fp16.t(),
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
                out_dtype=torch.float16,
                gate_vals=gate_vals_sparse,
                row_idx=row_idx_sparse,
                block_counts=block_counts_sparse,
                max_rows=max_rows_sparse,
            )
        torch.cuda.synchronize()

        start_sortpack = torch.cuda.Event(enable_timing=True)
        end_sortpack   = torch.cuda.Event(enable_timing=True)
        start_sortpack.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_sparse_triton_sortpack(
                x_fp16,
                up_weight_fp16.t(),
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
                out_dtype=torch.float16,
                gate_vals=gate_vals_sparse,
                row_idx=row_idx_sparse,
                block_counts=block_counts_sparse,
                max_rows=max_rows_sparse,
            )
        end_sortpack.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        sortpack_ms = start_sortpack.elapsed_time(end_sortpack) / num_iters

        base_line = f"Triton (SortPack, {zeros_frac:.1%} nnz): {sortpack_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/sortpack_ms:.2f}x"
        if run_all:
            base_line += f" | vs baseline sparse: {sparse_ms/sortpack_ms:.2f}x | vs optimized: {opt_ms/sortpack_ms:.2f}x | vs old CSR: {csr_ms/sortpack_ms:.2f}x"
        print(base_line)

        # ---- StreamCompact benchmark (always) ----
        # Reuse preprocessed sparse gate data and mappings

        for _ in range(5):
            _ = fused_up_proj_gate_activation_sparse_triton_stream_compact(
                x_fp16,
                up_weight_fp16.t(),
                gate_sparse,
                num_blocks,
                line_size,
                mappings=mappings_sparse,
                out_dtype=torch.float16,
            )
        torch.cuda.synchronize()

        start_stream_compact = torch.cuda.Event(enable_timing=True)
        end_stream_compact = torch.cuda.Event(enable_timing=True)
        start_stream_compact.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_sparse_triton_stream_compact(
                x_fp16,
                up_weight_fp16.t(),
                gate_sparse,
                num_blocks,
                line_size,
                mappings=mappings_sparse,
                out_dtype=torch.float16,
            )
        end_stream_compact.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        stream_compact_ms = start_stream_compact.elapsed_time(end_stream_compact) / num_iters

        stream_compact_line = f"Triton (StreamCompact, {zeros_frac:.1%} nnz): {stream_compact_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/stream_compact_ms:.2f}x | vs SortPack: {sortpack_ms/stream_compact_ms:.2f}x"
        if run_all:
            stream_compact_line += f" | vs baseline sparse: {sparse_ms/stream_compact_ms:.2f}x | vs optimized: {opt_ms/stream_compact_ms:.2f}x | vs old CSR: {csr_ms/stream_compact_ms:.2f}x"
        print(stream_compact_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cast up-proj kernels")
    parser.add_argument("--all", action="store_true", help="Run all helper variants (dense, sparse, etc.)")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per config")
    parser.add_argument("--sparsity", type=str, default="dynamic", help="Sparsity for sparse benchmarks (e.g., 0.1, 0.9, dynamic)")
    args = parser.parse_args()

    try:
        import triton
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    benchmark_fused_vs_pytorch(num_iters=args.iters, run_all=args.all, sparsity=args.sparsity) 