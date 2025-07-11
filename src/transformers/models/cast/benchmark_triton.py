import torch
import torch.nn.functional as F
from triton_kernels import (
    fused_up_proj_gate_activation_triton,
    fused_up_proj_gate_activation_sparse_triton,
)
from triton_cast_kernel import (
    fused_up_proj_gate_activation_sparse_triton_optimized as fused_up_proj_gate_activation_sparse_triton_opt,
)
from triton_cast_kernel_csr import (
    fused_up_proj_gate_activation_sparse_triton_csr as fused_up_proj_gate_activation_sparse_triton_csr,
)  # New CSR helper
# Unified CSR helper (GPU-built buffers)
from triton_cast_kernel_csr_unified import (
    fused_up_proj_gate_activation_sparse_triton_csr_unified as fused_up_proj_gate_activation_sparse_triton_csr_unified,
)


def benchmark_fused_vs_pytorch(num_iters: int = 100):
    """Time PyTorch reference vs Triton fused kernel on several shapes."""
    if not torch.cuda.is_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("\n=== Performance Benchmark (fp16 inputs, fp32 gate/output) ===")

    configs = [
        (4, 8, 128, 4, 32),
        (8, 16, 256, 8, 32),
        (128, 32, 512, 8, 64),
        (128, 128, 4096, 64, 64),
        (128, 128, 4096, 8, 4096),
    ]

    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        intermediate_size = num_blocks * line_size
        cfg = f"{batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}"
        print(f"\nConfig: {cfg}  |  Iters: {num_iters}")

        # Random tensors
        x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
        up_weight_fp16 = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
        up_bias_fp16 = torch.randn(intermediate_size, device="cuda", dtype=torch.float16)
        gate_fp32 = torch.rand(batch_size, seq_len, num_blocks, device="cuda", dtype=torch.float32)

        # ---- PyTorch timing ----
        # Warm-up for PyTorch (5 runs)
        for _ in range(5):
            up_proj_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16, up_bias_fp16)).float()
            up_proj_reshaped = up_proj_fp16.view(batch_size, seq_len, num_blocks, line_size)
            _ = (up_proj_reshaped * gate_fp32.unsqueeze(-1)).view(batch_size, seq_len, intermediate_size)
        torch.cuda.synchronize()

        start_pt = torch.cuda.Event(enable_timing=True)
        end_pt = torch.cuda.Event(enable_timing=True)
        start_pt.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            up_proj_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16, up_bias_fp16)).float()
            up_proj_reshaped = up_proj_fp16.view(batch_size, seq_len, num_blocks, line_size)
            _ = (up_proj_reshaped * gate_fp32.unsqueeze(-1)).view(batch_size, seq_len, intermediate_size)
        end_pt.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        torch_ms = start_pt.elapsed_time(end_pt) / num_iters

        print(f"PyTorch: {torch_ms:.3f} ms")
        
        # ---- Triton timing (autotuned) ----
        # Warm-up for Triton (5 runs)
        for _ in range(5):
            _ = fused_up_proj_gate_activation_triton(x_fp16, up_weight_fp16.t(), up_bias_fp16, gate_fp32, num_blocks, line_size)
        torch.cuda.synchronize()

        start_tri = torch.cuda.Event(enable_timing=True)
        end_tri   = torch.cuda.Event(enable_timing=True)
        start_tri.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_triton(x_fp16, up_weight_fp16.t(), up_bias_fp16, gate_fp32, num_blocks, line_size)
        end_tri.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        triton_ms = start_tri.elapsed_time(end_tri) / num_iters

        speedup = torch_ms / triton_ms
        print(f"Triton (autotuned): {triton_ms:.3f} ms | Speed-up: {speedup:.2f}x")

        # ---- Sparse benchmark (10% non-zero gate) ----
        gate_sparse = gate_fp32.clone()
        mask_sparse = torch.rand_like(gate_sparse) < 0.9  # 90% zeros
        gate_sparse[mask_sparse] = 0.0

        # Warm-up sparse helper (5 runs)
        for _ in range(5):
            _ = fused_up_proj_gate_activation_sparse_triton(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
            )
        torch.cuda.synchronize()

        start_sp = torch.cuda.Event(enable_timing=True)
        end_sp   = torch.cuda.Event(enable_timing=True)
        start_sp.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_sparse_triton(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
            )
        end_sp.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        sparse_ms = start_sp.elapsed_time(end_sp) / num_iters

        print(f"Triton (sparse helper, 10% nnz): {sparse_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/sparse_ms:.2f}x")

        # ---- Optimized sparse benchmark (10% non-zero gate) ----
        for _ in range(5):
            _ = fused_up_proj_gate_activation_sparse_triton_opt(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
            )
        torch.cuda.synchronize()

        start_opt = torch.cuda.Event(enable_timing=True)
        end_opt   = torch.cuda.Event(enable_timing=True)
        start_opt.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_sparse_triton_opt(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
            )
        end_opt.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        opt_ms = start_opt.elapsed_time(end_opt) / num_iters

        print(f"Triton (optimized sparse, 10% nnz): {opt_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/opt_ms:.2f}x | Speed-up vs baseline sparse: {sparse_ms/opt_ms:.2f}x")

        # ---- CSR sparse benchmark (10% non-zero gate) ----
        for _ in range(5):
            _ = fused_up_proj_gate_activation_sparse_triton_csr(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
            )
        torch.cuda.synchronize()

        start_csr = torch.cuda.Event(enable_timing=True)
        end_csr   = torch.cuda.Event(enable_timing=True)
        start_csr.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_sparse_triton_csr(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
            )
        end_csr.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        csr_ms = start_csr.elapsed_time(end_csr) / num_iters

        # ---- Unified CSR sparse benchmark ----
        for _ in range(5):
            _ = fused_up_proj_gate_activation_sparse_triton_csr_unified(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
            )
        torch.cuda.synchronize()

        start_csr_u = torch.cuda.Event(enable_timing=True)
        end_csr_u   = torch.cuda.Event(enable_timing=True)
        start_csr_u.record(torch.cuda.current_stream())
        for _ in range(num_iters):
            _ = fused_up_proj_gate_activation_sparse_triton_csr_unified(
                x_fp16,
                up_weight_fp16.t(),
                up_bias_fp16,
                gate_sparse,
                num_blocks,
                line_size,
                zero_init=False,
            )
        end_csr_u.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        csr_u_ms = start_csr_u.elapsed_time(end_csr_u) / num_iters

        print(
            f"Triton (CSR sparse, 10% nnz):      {csr_ms   :.3f} ms | Speed-up vs PyTorch: {torch_ms/csr_ms   :.2f}x"
        )
        print(
            f"Triton (CSR-Unified, 10% nnz): {csr_u_ms:.3f} ms | Speed-up vs PyTorch: {torch_ms/csr_u_ms:.2f}x | "
            f"vs baseline sparse: {sparse_ms/csr_u_ms:.2f}x | vs optimized: {opt_ms/csr_u_ms:.2f}x | vs old CSR: {csr_ms/csr_u_ms:.2f}x"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA is not available – exiting.")
        exit(1)

    try:
        import triton
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)

    # Run benchmark
    benchmark_fused_vs_pytorch(num_iters=100) 