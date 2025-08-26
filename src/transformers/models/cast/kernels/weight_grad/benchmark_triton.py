import torch
import torch.nn.functional as F
import argparse

from .triton_kernels import fused_weight_grad_triton
from .triton_cast_kernel_gate_sortpack import fused_weight_grad_sparse_triton_sortpack
from .triton_cast_kernel_weight_grad_stream_compact import fused_weight_grad_sparse_triton_stream_compact
from kernels.stream_compact_index import create_stream_compact_index


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


def convert_dense_to_stream_compact(intermediate_dense, mappings, num_blocks, line_size):
    """Convert (BS, I) dense → (act_idx, LS) sparse format using stream compact mappings"""
    BS, I = intermediate_dense.shape
    
    # Early exit if no active elements
    if mappings['total_act_idx'] == 0:
        return torch.empty(0, line_size, device=intermediate_dense.device, dtype=intermediate_dense.dtype)
    
    # Reshape to block format: (BS, I) → (BS, NB, LS)
    intermediate_reshaped = intermediate_dense.view(BS, num_blocks, line_size)
    
    # Extract stream compact mappings (new format with local indices + block offsets)
    bs_nb_to_local_idx = mappings['bs_nb_to_local_idx']  # (BS, NB) → local row index
    block_offsets = mappings['block_offsets']             # (NB,) → block start offsets
    total_act_idx = mappings['total_act_idx']
    
    # Build sparse tensor: (act_idx, LS)
    intermediate_sparse = torch.zeros(total_act_idx, line_size, 
                                    device=intermediate_dense.device, 
                                    dtype=intermediate_dense.dtype)
    
    # Fill sparse tensor using mappings
    active_mask = bs_nb_to_local_idx != 65535
    if active_mask.sum() > 0:
        active_bs, active_nb = torch.where(active_mask)
        # Convert to int32 for indexing, then reconstruct act_indices
        bs_nb_to_local_idx_int32 = bs_nb_to_local_idx.to(torch.int32)
        local_indices = bs_nb_to_local_idx_int32[active_bs, active_nb]
        act_indices = block_offsets[active_nb] + local_indices
        intermediate_sparse[act_indices] = intermediate_reshaped[active_bs, active_nb]
    
    return intermediate_sparse


def benchmark_fused_vs_pytorch(num_iters: int = 100, run_all: bool = False, sparsity: str = "dynamic"):
    """Time PyTorch reference vs Triton fused weight-grad kernel on several shapes.

    The signature mirrors *down_proj/benchmark_triton.py* for consistency.
    """

    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("\n=== Weight-Grad Performance Benchmark (fp16 inputs → fp16 output) ===")

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
        
        cfg = f"{batch_size}x{seq_len} | H={hidden_size} NB={num_blocks} LS={line_size} | Sparsity={zeros_frac:.3f}"
        print(f"\nConfig: {cfg}  |  Iters: {num_iters}")

        # Random tensors
        intermediate_fp16 = torch.randn(batch_size * seq_len, intermediate_size, device="cuda", dtype=torch.float16)
        other_fp16 = torch.randn(batch_size * seq_len, hidden_size, device="cuda", dtype=torch.float16)
        
        # Create sparse gate with calculated sparsity
        gate_fp32 = _make_sparse_gate(batch_size * seq_len, num_blocks, sparsity=zeros_frac)

        # Apply sparsity to intermediate tensor
        gate_expanded = gate_fp32.unsqueeze(-1).expand(-1, -1, line_size).reshape(batch_size * seq_len, intermediate_size)
        intermediate_fp16_masked = (intermediate_fp16 * gate_expanded.to(dtype=torch.float16)).contiguous()

        # Ensure all inputs are contiguous
        intermediate_fp16 = intermediate_fp16.contiguous()
        other_fp16 = other_fp16.contiguous()
        gate_fp32 = gate_fp32.contiguous()

        # Pre-transpose and make contiguous for fair comparison
        intermediate_fp16_masked_t = intermediate_fp16_masked.t().contiguous()

        # ---- PyTorch timing (half×half→half) ----
        for _ in range(5):
            _ = intermediate_fp16_masked_t @ other_fp16
        torch.cuda.synchronize()

        start_pt = torch.cuda.Event(enable_timing=True)
        end_pt = torch.cuda.Event(enable_timing=True)
        start_pt.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = intermediate_fp16_masked_t @ other_fp16
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

        gate_vals, row_idx, block_counts, max_rows = preprocess_gate(gate_fp32, num_blocks)
        
        # Stream compact preprocessing
        stream_compact_mappings = create_stream_compact_index(gate_fp32)
        intermediate_sparse = convert_dense_to_stream_compact(
            intermediate_fp16, stream_compact_mappings, num_blocks, line_size)

        if run_all:
            # ---- Triton timing ----
            for _ in range(5):
                _ = fused_weight_grad_triton(
                    intermediate_fp16_masked,
                    other_fp16,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals,
                    row_idx=row_idx,
                    block_counts=block_counts,
                    max_rows=max_rows,
                )
            torch.cuda.synchronize()

            start_tri = torch.cuda.Event(enable_timing=True)
            end_tri = torch.cuda.Event(enable_timing=True)
            start_tri.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_weight_grad_triton(
                    intermediate_fp16_masked,
                    other_fp16,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals,
                    row_idx=row_idx,
                    block_counts=block_counts,
                    max_rows=max_rows,
                )
            end_tri.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            triton_ms = start_tri.elapsed_time(end_tri) / num_iters

            print(f"Triton (dense):  {triton_ms:.3f} ms | Speed-up: {torch_ms / triton_ms:.2f}×")

            # ---- Triton SortPack sparse (10% nnz) ----
            for _ in range(5):
                _ = fused_weight_grad_sparse_triton_sortpack(
                    intermediate_fp16,
                    other_fp16,
                    gate_fp32,
                    num_blocks,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals,
                    row_idx=row_idx,
                    block_counts=block_counts,
                    max_rows=max_rows,
                )
            torch.cuda.synchronize()

            start_sp = torch.cuda.Event(enable_timing=True)
            end_sp = torch.cuda.Event(enable_timing=True)
            start_sp.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_weight_grad_sparse_triton_sortpack(
                    intermediate_fp16,
                    other_fp16,
                    gate_fp32,
                    num_blocks,
                    line_size,
                    out_dtype=torch.float16,
                    gate_vals=gate_vals,
                    row_idx=row_idx,
                    block_counts=block_counts,
                    max_rows=max_rows,
                )
            end_sp.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            sortpack_ms = start_sp.elapsed_time(end_sp) / num_iters

            print(
                f"Triton (SortPack): {sortpack_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms / sortpack_ms:.2f}× | vs dense: {triton_ms / sortpack_ms:.2f}×"
            )
            
            # ---- Triton StreamCompact sparse (10% nnz) ----
            for _ in range(5):
                _ = fused_weight_grad_sparse_triton_stream_compact(
                    intermediate_sparse,
                    other_fp16,
                    num_blocks,
                    line_size,
                    mappings=stream_compact_mappings,
                    out_dtype=torch.float16,
                )
            torch.cuda.synchronize()

            start_sc = torch.cuda.Event(enable_timing=True)
            end_sc = torch.cuda.Event(enable_timing=True)
            start_sc.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_weight_grad_sparse_triton_stream_compact(
                    intermediate_sparse,
                    other_fp16,
                    num_blocks,
                    line_size,
                    mappings=stream_compact_mappings,
                    out_dtype=torch.float16,
                )
            end_sc.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            streamcompact_ms = start_sc.elapsed_time(end_sc) / num_iters

            print(
                f"Triton (StreamCmpt): {streamcompact_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms / streamcompact_ms:.2f}× | vs dense: {triton_ms / streamcompact_ms:.2f}× | vs SortPack: {sortpack_ms / streamcompact_ms:.2f}×"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cast weight-grad kernel")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per config")
    parser.add_argument(
        "--sparsity",
        default="dynamic",
        help="Fraction of zeros in gate (e.g., 0.9) or 'dynamic' to use 1 - 1/num_blocks per config",
    )
    args = parser.parse_args()

    try:
        import triton  # noqa: F401  – import to trigger availability message
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    benchmark_fused_vs_pytorch(num_iters=args.iters, run_all=True, sparsity=args.sparsity)