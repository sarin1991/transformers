import torch
import triton
import triton.language as tl
from typing import Optional
import time
import torch.nn.functional as F


@triton.jit
def fused_up_proj_gate_activation_kernel(
    x_ptr, up_weight_ptr, up_bias_ptr, gate_ptr, output_ptr,
    batch_size, seq_len, hidden_size, num_blocks, line_size,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. up_proj = F.relu(x @ up_weight + up_bias)
    2. output = up_proj * gate (reshaped and applied)
    Gate is pre-calculated and passed as input.
    
    Parameters:
    -----------
    x_ptr: Pointer to input tensor of shape (batch_size, seq_len, hidden_size) - contiguous
    up_weight_ptr: Pointer to up projection weight of shape (hidden_size, intermediate_size) - contiguous
    up_bias_ptr: Pointer to up projection bias of shape (intermediate_size,) - contiguous
    gate_ptr: Pointer to pre-calculated gate tensor of shape (batch_size, seq_len, num_blocks) - contiguous
    output_ptr: Pointer to output tensor of shape (batch_size, seq_len, num_blocks, line_size) - contiguous
    
    Dimensions:
    -----------
    batch_size: Number of batches
    seq_len: Sequence length
    hidden_size: Input hidden dimension
    num_blocks: Number of blocks for gate activation
    line_size: Size of each line within a block (intermediate_size = num_blocks * line_size)
    
    Block Sizes (compile-time constants):
    ------------------------------------
    BLOCK_SIZE_H: Number of hidden dimensions processed per block (minimum 16)
    BLOCK_SIZE_LS: Number of line size elements processed per block (minimum 16)
    
    The block sizes determine the granularity of parallelization and memory access patterns.
    Larger block sizes can improve memory bandwidth utilization but may reduce parallelism.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block indices - process one (batch, seq, block) element at a time
    total_blocks = batch_size * seq_len * num_blocks
    block_idx = pid
    
    if block_idx >= total_blocks:
        return
    
    # Calculate indices
    b = block_idx // (seq_len * num_blocks)
    l = (block_idx // num_blocks) % seq_len
    nb = block_idx % num_blocks
    
    # Create offsets for the block
    offs_ls = tl.arange(0, BLOCK_SIZE_LS)
    
    # Create masks
    mask_ls = offs_ls < line_size
    
    # Load pre-calculated gate value for this specific (batch, seq, block) element early
    # gate: (batch_size, seq_len, num_blocks)
    g_ptr = gate_ptr + (b * seq_len * num_blocks + l * num_blocks + nb)
    g = tl.load(g_ptr)
    
    # Early return if gate is zero - no need to compute matrix multiplication
    if g == 0.0:
        # Store zeros in output
        # output: (batch_size, seq_len, num_blocks, line_size)
        out_ptrs = output_ptr + (b * seq_len * num_blocks * line_size + 
                               l * num_blocks * line_size + nb * line_size + offs_ls)
        tl.store(out_ptrs, tl.zeros((BLOCK_SIZE_LS,), dtype=tl.float32), mask=mask_ls)
        return
    
    # Create offsets for the block (only needed if gate is non-zero)
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    
    # Create masks
    mask_h = offs_h < hidden_size
    
    # Compute up projection: up_proj = F.relu(x @ up_weight + up_bias)
    up_proj = tl.zeros((BLOCK_SIZE_LS,), dtype=tl.float32)
    
    # Matrix multiplication for up projection
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        # Load up_weight block for this specific (batch, seq, block) element
        # up_weight: (hidden_size, intermediate_size) -> (hidden_size, num_blocks * line_size)
        up_w_ptrs = up_weight_ptr + (offs_h[:, None] * (num_blocks * line_size) + 
                                   (nb * line_size + offs_ls)[None, :])
        up_w = tl.load(up_w_ptrs, mask=(mask_h[:, None] & mask_ls[None, :]), other=0.0)
        
        # Load x block for this specific (batch, seq) element
        # x: (batch_size, seq_len, hidden_size)
        x_ptrs = x_ptr + (b * seq_len * hidden_size + l * hidden_size + offs_h + h)
        x_block = tl.load(x_ptrs, mask=(offs_h + h) < hidden_size, other=0.0)
        
        # Accumulate matrix multiplication
        up_proj += tl.dot(x_block, up_w)
    
    # Add bias and apply ReLU
    # up_bias: (intermediate_size,) -> (num_blocks * line_size,)
    bias_ptrs = up_bias_ptr + (nb * line_size + offs_ls)
    up_bias = tl.load(bias_ptrs, mask=mask_ls, other=0.0)
    up_proj += up_bias
    up_proj = tl.where(up_proj > 0, up_proj, 0.0)
    
    # Apply gate activation
    output = up_proj * g
    
    # Store output
    # output: (batch_size, seq_len, num_blocks, line_size)
    out_ptrs = output_ptr + (b * seq_len * num_blocks * line_size + 
                           l * num_blocks * line_size + nb * line_size + offs_ls)
    tl.store(out_ptrs, output, mask=mask_ls)


def fused_up_proj_gate_activation_triton(x, up_weight, up_bias, gate, num_blocks, line_size):
    """
    Fused Triton implementation that performs up projection and gate activation in one kernel.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size)
        up_weight: Up projection weight of shape (hidden_size, intermediate_size)
        up_bias: Up projection bias of shape (intermediate_size,)
        gate: Pre-calculated gate tensor of shape (batch_size, seq_len, num_blocks)
        num_blocks: Number of blocks
        line_size: Size of each line within a block
    
    Returns:
        Output tensor of shape (batch_size, seq_len, intermediate_size)
    """
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size
    assert up_weight.shape == (hidden_size, intermediate_size), "Incompatible up_weight shape"
    assert gate.shape == (batch_size, seq_len, num_blocks), "Incompatible gate shape"
    
    # Ensure all inputs are contiguous
    x = x.contiguous()
    up_weight = up_weight.contiguous()
    up_bias = up_bias.contiguous()
    gate = gate.contiguous()
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, num_blocks, line_size), 
                        device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = (batch_size * seq_len * num_blocks,)
    
    fused_up_proj_gate_activation_kernel[grid](
        x, up_weight, up_bias, gate, output,
        batch_size, seq_len, hidden_size, num_blocks, line_size,
        BLOCK_SIZE_H=16, BLOCK_SIZE_LS=16,
    )
    
    # Reshape back to original shape
    return output.view(batch_size, seq_len, intermediate_size)


def test_fused_up_proj_gate_activation_triton():
    """
    Test the correctness of fused_up_proj_gate_activation_triton against a PyTorch reference implementation.
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
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
        # Create weight in the format expected by F.linear: (out_features, in_features)
        up_weight = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float16)
        up_bias = torch.randn(intermediate_size, device='cuda', dtype=torch.float16)
        gate = torch.randn(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float16)
        
        # PyTorch reference
        up_proj = F.relu(F.linear(x, up_weight, up_bias))
        up_proj_reshaped = up_proj.view(batch_size, seq_len, num_blocks, line_size)
        gate_expanded = gate.unsqueeze(-1).expand_as(up_proj_reshaped)
        ref = (up_proj_reshaped * gate_expanded).view(batch_size, seq_len, intermediate_size)
        
        # For Triton kernel, we need weight in (hidden_size, intermediate_size) format
        up_weight_triton = up_weight.t()
        
        # Triton kernel
        out = fused_up_proj_gate_activation_triton(x, up_weight_triton, up_bias, gate, num_blocks, line_size)
        max_diff = torch.max(torch.abs(ref - out)).item()
        mean_diff = torch.mean(torch.abs(ref - out)).item()
        print(f"Config {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")
        assert max_diff < 1e-2, f"Test failed for config {batch_size}x{seq_len}x{hidden_size}x{num_blocks}x{line_size}"
    print("✅ fused_up_proj_gate_activation_triton correctness test passed!")


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
    
    test_fused_up_proj_gate_activation_triton() 