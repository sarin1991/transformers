import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from triton_kernels import fused_up_proj_gate_activation_kernel, fused_up_proj_gate_activation_triton


def debug_small_scale():
    """
    Debug at a very small scale to isolate the issue.
    """
    print("=== Small Scale Debug ===")
    
    # Very small configuration
    batch_size, seq_len, hidden_size, num_blocks, line_size = 1, 2, 8, 2, 4
    intermediate_size = num_blocks * line_size
    
    print(f"Config: {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}")
    print(f"Intermediate size: {intermediate_size}")
    
    # Create simple test data
    x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    up_weight_fp16 = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float16)
    up_bias_fp16 = torch.randn(intermediate_size, device='cuda', dtype=torch.float16)
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
    
    print(f"x_fp16 shape: {x_fp16.shape}")
    print(f"up_weight_fp16 shape: {up_weight_fp16.shape}")
    print(f"up_bias_fp16 shape: {up_bias_fp16.shape}")
    print(f"gate_fp32 shape: {gate_fp32.shape}")
    
    # PyTorch reference
    ref_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16, up_bias_fp16))
    ref_fp32 = ref_fp16.float()
    
    print(f"PyTorch reference shape: {ref_fp32.shape}")
    print(f"PyTorch reference values:\n{ref_fp32}")
    
    # Triton kernel
    up_weight_triton_fp16 = up_weight_fp16.t()
    out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
    
    print(f"Triton output shape: {out_fp32.shape}")
    print(f"Triton output values:\n{out_fp32}")
    
    # Compare
    max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
    mean_diff = torch.mean(torch.abs(ref_fp32 - out_fp32)).item()
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    # Print differences
    diff = torch.abs(ref_fp32 - out_fp32)
    print(f"Differences:\n{diff}")
    
    return max_diff


def debug_step_by_step():
    """
    Debug step by step to see where the issue occurs.
    """
    print("\n=== Step by Step Debug ===")
    
    batch_size, seq_len, hidden_size, num_blocks, line_size = 1, 1, 4, 1, 2
    intermediate_size = num_blocks * line_size
    
    print(f"Config: {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}")
    
    # Create very simple data
    x_fp16 = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device='cuda', dtype=torch.float16)
    up_weight_fp16 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], 
                                 device='cuda', dtype=torch.float16)
    up_bias_fp16 = torch.tensor([0.1, 0.2], device='cuda', dtype=torch.float16)
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
    
    print(f"x_fp16: {x_fp16}")
    print(f"up_weight_fp16: {up_weight_fp16}")
    print(f"up_bias_fp16: {up_bias_fp16}")
    print(f"gate_fp32: {gate_fp32}")
    
    # PyTorch reference
    ref_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16, up_bias_fp16))
    ref_fp32 = ref_fp16.float()
    
    print(f"PyTorch reference: {ref_fp32}")
    
    # Triton kernel
    up_weight_triton_fp16 = up_weight_fp16.t()
    out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
    
    print(f"Triton output: {out_fp32}")
    
    # Compare
    max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
    print(f"Max diff: {max_diff:.6f}")
    
    return max_diff


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        exit(1)
    
    try:
        import triton
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available.")
        exit(1)
    
    # Run debug tests
    debug_small_scale()
    debug_step_by_step() 