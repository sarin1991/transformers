import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from triton_kernels import fused_up_proj_gate_activation_kernel, fused_up_proj_gate_activation_triton


def debug_small_scale():
    """
    Debug at a smaller scale to reproduce the issue.
    """
    print("=== Small Scale Debug (Zero Bias, Ones Gate) ===")
    
    # Smaller configuration
    batch_size, seq_len, hidden_size, num_blocks, line_size = 1, 2, 16, 2, 8
    intermediate_size = num_blocks * line_size
    
    print(f"Config: {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}")
    print(f"Intermediate size: {intermediate_size}")
    
    # Create test data with random weights but zero bias and ones gate
    x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    up_weight_fp16 = torch.ones(intermediate_size, hidden_size, device='cuda', dtype=torch.float16)  # Ones weights
    up_bias_fp16 = torch.zeros(intermediate_size, device='cuda', dtype=torch.float16)  # Zero bias
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)  # Ones gate
    
    print(f"x_fp16 shape: {x_fp16.shape}")
    print(f"up_weight_fp16 shape: {up_weight_fp16.shape}")
    print(f"up_bias_fp16 shape: {up_bias_fp16.shape}")
    print(f"gate_fp32 shape: {gate_fp32.shape}")
    
    # Show input values
    print(f"Input x values:\n{x_fp16}")
    
    # PyTorch reference (compute in fp32)
    ref_pre_relu = F.linear(x_fp16.float(), up_weight_fp16.float(), up_bias_fp16.float())
    print(f"PyTorch pre-ReLU values (fp32):\n{ref_pre_relu}")

    ref_fp32 = F.relu(ref_pre_relu)
    
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
    up_weight_fp16 = torch.ones(2, 4, device='cuda', dtype=torch.float16)
    up_bias_fp16 = torch.zeros(2, device='cuda', dtype=torch.float16)
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
    
    print(f"x_fp16: {x_fp16}")
    print(f"up_weight_fp16: {up_weight_fp16}")
    print(f"up_bias_fp16: {up_bias_fp16}")
    print(f"gate_fp32: {gate_fp32}")
    
    # PyTorch reference (fp32)
    ref_fp32 = F.relu(F.linear(x_fp16.float(), up_weight_fp16.float(), up_bias_fp16.float()))
    
    print(f"PyTorch reference: {ref_fp32}")
    
    # Triton kernel
    up_weight_triton_fp16 = up_weight_fp16.t()
    out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
    
    print(f"Triton output: {out_fp32}")
    
    # Compare
    max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
    print(f"Max diff: {max_diff:.6f}")
    
    return max_diff


def debug_simple_matmul():
    """
    Debug simple matrix multiplication without complex blocking.
    """
    print("\n=== Simple Matrix Multiplication Debug ===")
    
    # Simple configuration
    batch_size, seq_len, hidden_size, num_blocks, line_size = 1, 1, 8, 1, 8
    intermediate_size = num_blocks * line_size
    
    print(f"Config: {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}")
    
    # Create simple data
    x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    up_weight_fp16 = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float16)
    up_bias_fp16 = torch.zeros(intermediate_size, device='cuda', dtype=torch.float16)
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
    
    print(f"x_fp16: {x_fp16}")
    print(f"up_weight_fp16: {up_weight_fp16}")
    
    # PyTorch reference - matrix multiplication in fp32
    x_reshaped = x_fp16.view(batch_size * seq_len, hidden_size).float()
    ref_matmul = F.linear(x_reshaped, up_weight_fp16.float(), up_bias_fp16.float())
    ref_fp32 = F.relu(ref_matmul)
    
    print(f"PyTorch matmul result: {ref_matmul}")
    print(f"PyTorch final result: {ref_fp32}")
    
    # Triton kernel
    up_weight_triton_fp16 = up_weight_fp16.t()
    out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
    
    print(f"Triton output: {out_fp32}")
    
    # Compare
    max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
    print(f"Max diff: {max_diff:.6f}")
    
    return max_diff


def debug_weight_layout():
    """
    Debug weight matrix layout and transposition.
    """
    print("\n=== Weight Layout Debug ===")
    
    # Simple configuration
    batch_size, seq_len, hidden_size, num_blocks, line_size = 1, 1, 4, 1, 4
    intermediate_size = num_blocks * line_size
    
    print(f"Config: {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}")
    print(f"Intermediate size: {intermediate_size}")
    
    # Create simple data
    x_fp16 = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device='cuda', dtype=torch.float16)
    up_weight_fp16 = torch.tensor([[1.0, 2.0, 3.0, 4.0], 
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0],
                                  [13.0, 14.0, 15.0, 16.0]], device='cuda', dtype=torch.float16)
    up_bias_fp16 = torch.zeros(intermediate_size, device='cuda', dtype=torch.float16)
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
    
    print(f"Original weight shape: {up_weight_fp16.shape}")
    print(f"Original weight:\n{up_weight_fp16}")
    
    # Transpose for Triton
    up_weight_triton_fp16 = up_weight_fp16.t()
    print(f"Transposed weight shape: {up_weight_triton_fp16.shape}")
    print(f"Transposed weight:\n{up_weight_triton_fp16}")
    
    # PyTorch reference (fp32)
    ref_fp32 = F.relu(F.linear(x_fp16.float(), up_weight_fp16.float(), up_bias_fp16.float()))
    
    print(f"PyTorch reference: {ref_fp32}")
    
    # Manual calculation to verify
    x_reshaped = x_fp16.view(1, 4)
    manual_result = torch.zeros(1, 4, device='cuda', dtype=torch.float32)
    
    # Manual matrix multiplication
    for i in range(4):  # output dimension
        for j in range(4):  # input dimension
            manual_result[0, i] += x_reshaped[0, j].float() * up_weight_fp16[i, j].float()
    
    # Apply ReLU
    manual_result = torch.relu(manual_result)
    print(f"Manual calculation: {manual_result}")
    
    # Triton kernel
    out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
    
    print(f"Triton output: {out_fp32}")
    
    # Compare
    max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
    manual_diff = torch.max(torch.abs(manual_result - out_fp32)).item()
    print(f"Max diff (PyTorch vs Triton): {max_diff:.6f}")
    print(f"Max diff (Manual vs Triton): {manual_diff:.6f}")
    
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

    # 1. Weight-layout sanity check (deterministic)
    weight_diff = debug_weight_layout()

    print("\n=== Summary ===")
    print(f"Weight layout diff : {weight_diff:.6f}")

    if weight_diff < 1e-6:
        print("✅ Weight layout check passed!")
    elif weight_diff < 1e-3:
        print("⚠️  Small differences, but likely acceptable.")
    else:
        print("❌ Significant differences - issue needs investigation.") 