import torch
import torch.nn.functional as F
import argparse

from .triton_kernels import fused_down_proj_triton
from .triton_cast_kernel_gate_sortpack import fused_down_proj_sparse_triton_sortpack
from .triton_cast_kernel_down_proj_stream_compact import fused_down_proj_sparse_triton_stream_compact
from kernels.stream_compact_index import create_stream_compact_index


def convert_dense_to_stream_compact(x_dense, mappings, num_blocks, line_size):
    """Convert (BS, I) dense → (act_idx, LS) sparse format using stream compact mappings"""
    BS, I = x_dense.shape
    
    # Early exit if no active elements
    if mappings['total_act_idx'] == 0:
        return torch.empty(0, line_size, device=x_dense.device, dtype=x_dense.dtype)
    
    # Reshape to block format: (BS, I) → (BS, NB, LS)
    x_reshaped = x_dense.view(BS, num_blocks, line_size)
    
    # Extract stream compact mappings (new format with local indices + block offsets)
    bs_nb_to_local_idx = mappings['bs_nb_to_local_idx']  # (BS, NB) → local row index
    block_offsets = mappings['block_offsets']             # (NB,) → block start offsets
    total_act_idx = mappings['total_act_idx']
    
    # Build sparse tensor: (act_idx, LS)
    x_sparse = torch.zeros(total_act_idx, line_size, 
                          device=x_dense.device, 
                          dtype=x_dense.dtype)
    
    # Fill sparse tensor using mappings
    active_mask = bs_nb_to_local_idx != 65535
    if active_mask.sum() > 0:
        active_bs, active_nb = torch.where(active_mask)
        # Reconstruct act_indices from local indices + block offsets
        local_indices = bs_nb_to_local_idx[active_bs, active_nb]
        act_indices = block_offsets[active_nb] + local_indices
        x_sparse[act_indices] = x_reshaped[active_bs, active_nb]
    
    return x_sparse


def benchmark_fused_vs_pytorch(num_iters: int = 100, run_all: bool = False):
    """Time PyTorch reference vs Triton fused down-proj kernel on several shapes.

    The signature mirrors *up_proj/benchmark_triton.py* for consistency.
    """

    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("\n=== Down-Proj Performance Benchmark (fp16 inputs → fp32 output) ===")

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
        cfg = f"{batch_size}x{seq_len} | H={hidden_size} NB={num_blocks} LS={line_size}"
        print(f"\nConfig: {cfg}  |  Iters: {num_iters}")

        # Random tensors
        batch_seq_size = batch_size * seq_len
        x_fp16 = torch.randn(batch_seq_size, intermediate_size, device="cuda", dtype=torch.float16)
        weight_fp16 = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
        # Pre-compute transposed weight (fp16) – F.linear will return fp16, we cast to fp32 once.
        weight_t_fp16 = weight_fp16.t().contiguous()
        gate_fp32 = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
        # Introduce sparsity (90 % zeros) to emulate typical gating pattern
        mask_sparse = torch.rand_like(gate_fp32) < 0.9
        gate_fp32[mask_sparse] = 0.0

        # Zero-out gated blocks in x for the PyTorch reference
        x_fp16_masked = x_fp16.view(batch_seq_size, num_blocks, line_size)
        x_fp16_masked = x_fp16_masked * gate_fp32.unsqueeze(-1).to(dtype=torch.float16)
        x_fp16_masked = x_fp16_masked.view(batch_seq_size, intermediate_size)

        # ---- PyTorch timing (half×half→half) ----
        for _ in range(5):
            _ = F.linear(x_fp16_masked, weight_t_fp16)
        torch.cuda.synchronize()

        start_pt = torch.cuda.Event(enable_timing=True)
        end_pt = torch.cuda.Event(enable_timing=True)
        start_pt.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = F.linear(x_fp16_masked, weight_t_fp16)
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
        x_sparse = convert_dense_to_stream_compact(x_fp16, stream_compact_mappings, num_blocks, line_size)

        if run_all:
            # ---- Triton timing ----
            for _ in range(5):
                _ = fused_down_proj_triton(
                    x_fp16,
                    weight_fp16,
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

            start_tri = torch.cuda.Event(enable_timing=True)
            end_tri = torch.cuda.Event(enable_timing=True)
            start_tri.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_down_proj_triton(
                    x_fp16,
                    weight_fp16,
                    gate_fp32,
                    num_blocks,
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
                _ = fused_down_proj_sparse_triton_sortpack(
                    x_fp16,
                    weight_fp16,
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
                _ = fused_down_proj_sparse_triton_sortpack(
                    x_fp16,
                    weight_fp16,
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
                _ = fused_down_proj_sparse_triton_stream_compact(
                    x_sparse,
                    weight_fp16,
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
                _ = fused_down_proj_sparse_triton_stream_compact(
                    x_sparse,
                    weight_fp16,
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
            
            # ---- Triton StreamCompact with two-stage reduction (atomic-free) ----
            for _ in range(5):
                _ = fused_down_proj_sparse_triton_stream_compact(
                    x_sparse,
                    weight_fp16,
                    num_blocks,
                    line_size,
                    mappings=stream_compact_mappings,
                    out_dtype=torch.float16,
                    two_stage_reduction=True,
                )
            torch.cuda.synchronize()

            start_sc2 = torch.cuda.Event(enable_timing=True)
            end_sc2 = torch.cuda.Event(enable_timing=True)
            start_sc2.record(torch.cuda.current_stream())
            for _ in range(num_iters):
                _ = fused_down_proj_sparse_triton_stream_compact(
                    x_sparse,
                    weight_fp16,
                    num_blocks,
                    line_size,
                    mappings=stream_compact_mappings,
                    out_dtype=torch.float16,
                    two_stage_reduction=True,
                )
            end_sc2.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            streamcompact_2stage_ms = start_sc2.elapsed_time(end_sc2) / num_iters

            print(
                f"Triton (StreamCmpt2): {streamcompact_2stage_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms / streamcompact_2stage_ms:.2f}× | vs dense: {triton_ms / streamcompact_2stage_ms:.2f}× | vs SortPack: {sortpack_ms / streamcompact_2stage_ms:.2f}× | vs StreamCmpt: {streamcompact_ms / streamcompact_2stage_ms:.2f}×"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cast down-proj kernel")
    parser.add_argument("--iters", type=int, default=100, help="Iterations per config")
    args = parser.parse_args()

    try:
        import triton  # noqa: F401  – import to trigger availability message
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    benchmark_fused_vs_pytorch(num_iters=args.iters, run_all=True) 