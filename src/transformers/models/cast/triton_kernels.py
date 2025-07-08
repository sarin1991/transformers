import torch
import triton
import triton.language as tl
from typing import Optional
import time
import torch.nn.functional as F


@triton.jit
def fused_up_proj_gate_activation_kernel(
    x_ptr, up_weight_ptr, up_bias_ptr, gate_ptr, output_ptr,
    batch_seq_size, hidden_size, num_blocks, line_size,
    stride_x_bs, stride_x_h,
    stride_w_h, stride_w_ls,
    stride_g_bs, stride_g_nb,
    stride_out_bs, stride_out_ls,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. up_proj = F.relu(x @ up_weight + up_bias)
    2. output = up_proj * gate (reshaped and applied)
    
    Following proper Triton matrix multiplication pattern with gating logic added.
    Blocks across batch_seq, num_blocks, and line_size dimensions.
    All operations in float16 for consistency.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block indices - block across batch_seq, num_blocks, and line_size
    # Similar to reference Triton pattern: pid = pid_m * num_pid_n + pid_n
    num_pid_bs = (batch_seq_size + BLOCK_SIZE_BS - 1) // BLOCK_SIZE_BS
    num_pid_nb = num_blocks
    num_pid_ls = (line_size + BLOCK_SIZE_LS - 1) // BLOCK_SIZE_LS
    
    # Calculate 3D block indices
    pid_ls = pid % num_pid_ls
    pid_nb = (pid // num_pid_ls) % num_pid_nb
    pid_bs = (pid // num_pid_ls) // num_pid_nb
    
    if pid_bs >= num_pid_bs:
        return
    
    # Create offsets for the block - following Triton pattern
    offs_bs = (pid_bs * BLOCK_SIZE_BS + tl.arange(0, BLOCK_SIZE_BS)) % batch_seq_size
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_ls = (pid_ls * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)) % line_size
    
    # Create masks
    mask_bs = offs_bs < batch_seq_size
    mask_h = offs_h < hidden_size
    mask_ls = offs_ls < line_size
    
    # Load pre-calculated gate values for this block
    # gate: (batch_seq_size, num_blocks)
    g_ptrs = gate_ptr + (offs_bs * stride_g_bs + pid_nb * stride_g_nb)
    g = tl.load(g_ptrs, mask=mask_bs, other=0.0)
    
    # Check if all gates are zero - early return
    g_sum = tl.sum(g)
    zero_gates = (g_sum == 0.0)
    
    # Early return if all gates are zero
    if zero_gates:
        # Store zeros in output for this block
        out_ptrs = output_ptr + (offs_bs[:, None] * stride_out_bs + 
                               (pid_nb * line_size + offs_ls)[None, :] * stride_out_ls)
        tl.store(out_ptrs, tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32), 
                mask=(mask_bs[:, None] & mask_ls[None, :]))
        return
    
    # Initialize accumulator for matrix multiplication - use float32 for tl.dot
    accumulator = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=tl.float32)
    
    # Create pointers for the first blocks of x and up_weight
    # Following the Triton pattern exactly
    x_ptrs = x_ptr + (offs_bs[:, None] * stride_x_bs + offs_h[None, :] * stride_x_h)
    w_ptrs = up_weight_ptr + (offs_h[:, None] * stride_w_h + 
                            (pid_nb * line_size + offs_ls)[None, :] * stride_w_ls)
    
    # Matrix multiplication: x @ up_weight
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        # Load blocks with proper masking
        x_block = tl.load(x_ptrs, mask=(mask_bs[:, None] & mask_h[None, :]), other=0.0)
        w_block = tl.load(w_ptrs, mask=(mask_h[:, None] & mask_ls[None, :]), other=0.0)
        
        # Matrix multiplication: (BLOCK_SIZE_BS, BLOCK_SIZE_H) @ (BLOCK_SIZE_H, BLOCK_SIZE_LS) -> (BLOCK_SIZE_BS, BLOCK_SIZE_LS)
        accumulator = tl.dot(x_block, w_block, accumulator)
        
        # Advance pointers to next K block - following Triton pattern
        x_ptrs += BLOCK_SIZE_H * stride_x_h
        w_ptrs += BLOCK_SIZE_H * stride_w_h
    
    # Add bias and apply ReLU - keep in float32 for precision
    # Load bias for the current block's line_size elements
    bias_ptrs = up_bias_ptr + (pid_nb * line_size + offs_ls)
    up_bias = tl.load(bias_ptrs, mask=mask_ls, other=0.0)
    accumulator += up_bias[None, :]
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    
    # Apply gate activation: (BLOCK_SIZE_BS, BLOCK_SIZE_LS) * (BLOCK_SIZE_BS,) -> (BLOCK_SIZE_BS, BLOCK_SIZE_LS)
    output = accumulator * g[:, None]
    
    # Store output
    out_ptrs = output_ptr + (offs_bs[:, None] * stride_out_bs + 
                           (pid_nb * line_size + offs_ls)[None, :] * stride_out_ls)
    tl.store(out_ptrs, output, mask=(mask_bs[:, None] & mask_ls[None, :]))


def fused_up_proj_gate_activation_triton(x, up_weight, up_bias, gate, num_blocks, line_size):
    """
    Fused Triton implementation that performs up projection and gate activation in one kernel.
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size) - must be float16
        up_weight: Up projection weight of shape (hidden_size, intermediate_size) - must be float16
        up_bias: Up projection bias of shape (intermediate_size,) - must be float16
        gate: Pre-calculated gate tensor of shape (batch_size, seq_len, num_blocks) - must be float32
        num_blocks: Number of blocks
        line_size: Size of each line within a block
    Returns:
        Output tensor of shape (batch_size, seq_len, intermediate_size), always float32
    """
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size
    assert up_weight.shape == (hidden_size, intermediate_size), "Incompatible up_weight shape"
    assert gate.shape == (batch_size, seq_len, num_blocks), "Incompatible gate shape"
    
    # Check that inputs are correct dtypes
    assert x.dtype == torch.float16, f"Input x must be float16, got {x.dtype}"
    assert up_weight.dtype == torch.float16, f"up_weight must be float16, got {up_weight.dtype}"
    assert up_bias.dtype == torch.float16, f"up_bias must be float16, got {up_bias.dtype}"
    
    # Optionally upcast gate to float32 if not already
    if gate.dtype != torch.float32:
        gate = gate.float()

    # Ensure all inputs are contiguous
    x = x.contiguous()
    up_weight = up_weight.contiguous()
    up_bias = up_bias.contiguous()
    gate = gate.contiguous()

    # Reshape inputs to combine batch_size and seq_len
    batch_seq_size = batch_size * seq_len
    x_reshaped = x.view(batch_seq_size, hidden_size)
    gate_reshaped = gate.view(batch_seq_size, num_blocks)

    # Allocate output as 2D tensor (float32)
    output = torch.empty((batch_seq_size, num_blocks * line_size), 
                        device=x.device, dtype=torch.float32)

    # Launch kernel with 3D blocking across batch_seq, num_blocks, and line_size
    num_pid_bs = (batch_seq_size + 15) // 16
    num_pid_nb = num_blocks
    num_pid_ls = (line_size + 15) // 16
    grid = (num_pid_bs * num_pid_nb * num_pid_ls,)

    fused_up_proj_gate_activation_kernel[grid](
        x_reshaped, up_weight, up_bias, gate_reshaped, output,
        batch_seq_size, hidden_size, num_blocks, line_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        gate_reshaped.stride(0), gate_reshaped.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_BS=16, BLOCK_SIZE_H=16, BLOCK_SIZE_LS=16,
    )

    # Reshape back to original shape
    return output.view(batch_size, seq_len, intermediate_size)


def debug_test():
    """
    Debug test to isolate the issue by testing just the matrix multiplication part.
    """
    print("Debug test - testing matrix multiplication only...")
    
    batch_size, seq_len, hidden_size, num_blocks, line_size = 2, 8, 512, 4, 16
    intermediate_size = num_blocks * line_size
    
    # Create test data
    x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    up_weight_fp16 = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float16)
    up_bias_fp16 = torch.randn(intermediate_size, device='cuda', dtype=torch.float16)
    
    # PyTorch reference - just the up projection
    ref_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16, up_bias_fp16))
    ref_fp32 = ref_fp16.float()
    
    # Test our kernel without gate (set gate to all ones)
    gate_fp32 = torch.ones(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
    up_weight_triton_fp16 = up_weight_fp16.t()
    out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
    
    max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
    mean_diff = torch.mean(torch.abs(ref_fp32 - out_fp32)).item()
    print(f"Debug - Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")
    
    # Check if the issue is in the matrix multiplication
    if max_diff > 1e-3:
        print("❌ Issue is in the matrix multiplication part")
        return False
    else:
        print("✅ Matrix multiplication is correct")
        return True


def test_fused_up_proj_gate_activation_triton():
    """
    Test the correctness of fused_up_proj_gate_activation_triton against a PyTorch reference implementation.
    Inputs are float16, gate is float32, output is float32.
    """
    print("Testing fused_up_proj_gate_activation_triton correctness...")
    # Test a few configurations
    test_configs = [
        (2, 8, 512, 4, 16),   # (batch_size, seq_len, hidden_size, num_blocks, line_size)
        (4, 16, 1024, 8, 32),
        (1, 4, 256, 2, 8),
    ]
    
    for batch_size, seq_len, hidden_size, num_blocks, line_size in test_configs:
        intermediate_size = num_blocks * line_size
        print(f"\n--- Testing Config {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size} ---")
        
        # Test with float16 inputs and float32 gate
        print("Testing implementation:")
        x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
        up_weight_fp16 = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float16)
        up_bias_fp16 = torch.randn(intermediate_size, device='cuda', dtype=torch.float16)
        gate_fp32 = torch.randn(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float32)
        
        # PyTorch reference (float16 inputs, float32 gate, float32 output)
        up_proj_fp16 = F.relu(F.linear(x_fp16, up_weight_fp16, up_bias_fp16)).float()  # Convert to float32
        up_proj_reshaped_fp32 = up_proj_fp16.view(batch_size, seq_len, num_blocks, line_size)
        gate_expanded_fp32 = gate_fp32.unsqueeze(-1).expand_as(up_proj_reshaped_fp32)
        ref_fp32 = (up_proj_reshaped_fp32 * gate_expanded_fp32).view(batch_size, seq_len, intermediate_size)
        
        # Triton kernel (float16 inputs, float32 gate, float32 output)
        up_weight_triton_fp16 = up_weight_fp16.t()
        out_fp32 = fused_up_proj_gate_activation_triton(x_fp16, up_weight_triton_fp16, up_bias_fp16, gate_fp32, num_blocks, line_size)
        max_diff = torch.max(torch.abs(ref_fp32 - out_fp32)).item()
        mean_diff = torch.mean(torch.abs(ref_fp32 - out_fp32)).item()
        print(f"  Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")
        
        # Assertions - tight tolerances since we're comparing float32 outputs
        assert max_diff < 1e-5, f"Test failed for config {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}"
        
        print(f"✅ Config {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size} passed!")
    
    print("\n🎉 All tests passed!")
    print("Note: Inputs are float16, gate is float32, output is float32.")
    print("Kernel uses float32 accumulation for precision.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Triton kernels require CUDA.")
        exit(1)
    try:
        import triton
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install triton.")
        exit(1)
    
    debug_test() 