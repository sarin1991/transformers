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
    stride_xb, stride_xl, stride_xh,
    stride_upwb, stride_upwh,
    stride_gb, stride_gl, stride_gnb,
    stride_outb, stride_outl, stride_outnb, stride_outls,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_L: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_NB: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. up_proj = F.relu(x @ up_weight + up_bias)
    2. output = up_proj * gate (reshaped and applied)
    Gate is pre-calculated and passed as input.
    
    Parameters:
    -----------
    x_ptr: Pointer to input tensor of shape (batch_size, seq_len, hidden_size)
    up_weight_ptr: Pointer to up projection weight of shape (hidden_size, intermediate_size)
    up_bias_ptr: Pointer to up projection bias of shape (intermediate_size,)
    gate_ptr: Pointer to pre-calculated gate tensor of shape (batch_size, seq_len, num_blocks)
    output_ptr: Pointer to output tensor of shape (batch_size, seq_len, num_blocks, line_size)
    
    Dimensions:
    -----------
    batch_size: Number of batches
    seq_len: Sequence length
    hidden_size: Input hidden dimension
    num_blocks: Number of blocks for gate activation
    line_size: Size of each line within a block (intermediate_size = num_blocks * line_size)
    
    Strides:
    --------
    stride_xb, stride_xl, stride_xh: Strides for input tensor x
    stride_upwb, stride_upwh: Strides for up projection weight
    stride_gb, stride_gl, stride_gnb: Strides for gate tensor
    stride_outb, stride_outl, stride_outnb, stride_outls: Strides for output tensor
    
    Block Sizes (compile-time constants):
    ------------------------------------
    BLOCK_SIZE_B: Number of batch elements processed per block (typically 1)
    BLOCK_SIZE_L: Number of sequence elements processed per block (typically 1)
    BLOCK_SIZE_H: Number of hidden dimensions processed per block (typically 64)
    BLOCK_SIZE_NB: Number of block indices processed per block (typically 1)
    BLOCK_SIZE_LS: Number of line size elements processed per block (typically 64)
    
    The block sizes determine the granularity of parallelization and memory access patterns.
    Larger block sizes can improve memory bandwidth utilization but may reduce parallelism.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block indices
    total_blocks = batch_size * seq_len * num_blocks
    block_idx = pid
    
    if block_idx >= total_blocks:
        return
    
    # Calculate indices
    b = block_idx // (seq_len * num_blocks)
    l = (block_idx // num_blocks) % seq_len
    nb = block_idx % num_blocks
    
    # Create offsets for the block
    offs_b = b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_l = l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_nb = nb * BLOCK_SIZE_NB + tl.arange(0, BLOCK_SIZE_NB)
    offs_ls = tl.arange(0, BLOCK_SIZE_LS)
    
    # Create masks
    mask_b = offs_b < batch_size
    mask_l = offs_l < seq_len
    mask_h = offs_h < hidden_size
    mask_nb = offs_nb < num_blocks
    mask_ls = offs_ls < line_size
    
    # Compute up projection: up_proj = F.relu(x @ up_weight + up_bias)
    up_proj = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_L, num_blocks * line_size), dtype=tl.float32)
    
    # Matrix multiplication for up projection
    for h in range(0, hidden_size, BLOCK_SIZE_H):
        # Load up_weight block
        up_w_ptrs = up_weight_ptr + (offs_h[:, None] * stride_upwb + 
                                   tl.arange(0, num_blocks * line_size)[None, :] * stride_upwh)
        up_w = tl.load(up_w_ptrs, mask=(mask_h[:, None] & tl.arange(0, num_blocks * line_size)[None, :] < num_blocks * line_size), other=0.0)
        
        # Load x block
        x_block_ptrs = x_ptr + (offs_b[:, None, None] * stride_xb + 
                              offs_l[None, :, None] * stride_xl + 
                              (offs_h + h)[None, None, :] * stride_xh)
        x_block = tl.load(x_block_ptrs, mask=(mask_b[:, None, None] & mask_l[None, :, None] & 
                                            (offs_h + h)[None, None, :] < hidden_size), other=0.0)
        
        # Accumulate matrix multiplication
        up_proj += tl.dot(x_block, up_w)
    
    # Add bias and apply ReLU
    up_bias = tl.load(up_bias_ptr + tl.arange(0, num_blocks * line_size), 
                     mask=tl.arange(0, num_blocks * line_size) < num_blocks * line_size, other=0.0)
    up_proj += up_bias[None, None, :]
    up_proj = tl.where(up_proj > 0, up_proj, 0.0)
    
    # Load pre-calculated gate values
    g_ptrs = gate_ptr + (offs_b[:, None, None] * stride_gb + 
                        offs_l[None, :, None] * stride_gl + 
                        offs_nb[None, None, :] * stride_gnb)
    g = tl.load(g_ptrs, mask=(mask_b[:, None, None] & mask_l[None, :, None] & mask_nb[None, None, :]), other=0.0)
    
    # Reshape up_proj to (batch, seq, num_blocks, line_size) and apply gate
    up_proj_reshaped = up_proj.view(BLOCK_SIZE_B, BLOCK_SIZE_L, num_blocks, line_size)
    gate_expanded = g[:, :, :, None]  # Expand to match up_proj_reshaped shape
    
    # Apply gate activation
    output = up_proj_reshaped * gate_expanded
    
    # Store output
    out_ptrs = output_ptr + (offs_b[:, None, None, None] * stride_outb + 
                           offs_l[None, :, None, None] * stride_outl + 
                           offs_nb[None, None, :, None] * stride_outnb + 
                           offs_ls[None, None, None, :] * stride_outls)
    tl.store(out_ptrs, output, mask=(mask_b[:, None, None, None] & mask_l[None, :, None, None] & 
                                   mask_nb[None, None, :, None] & mask_ls[None, None, None, :]))


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
    
    # Allocate output
    output = torch.empty((batch_size, seq_len, num_blocks, line_size), 
                        device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = (batch_size * seq_len * num_blocks,)
    
    fused_up_proj_gate_activation_kernel[grid](
        x, up_weight, up_bias, gate, output,
        batch_size, seq_len, hidden_size, num_blocks, line_size,
        x.stride(0), x.stride(1), x.stride(2),
        up_weight.stride(0), up_weight.stride(1),
        gate.stride(0), gate.stride(1), gate.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_B=1, BLOCK_SIZE_L=1, BLOCK_SIZE_H=64, BLOCK_SIZE_NB=1, BLOCK_SIZE_LS=64,
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
        up_weight = torch.randn(intermediate_size, hidden_size, device='cuda', dtype=torch.float16).t()
        up_bias = torch.randn(intermediate_size, device='cuda', dtype=torch.float16)
        gate = torch.randn(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float16)
        
        # PyTorch reference
        up_proj = F.relu(F.linear(x, up_weight, up_bias))
        up_proj_reshaped = up_proj.view(batch_size, seq_len, num_blocks, line_size)
        gate_expanded = gate.unsqueeze(-1).expand_as(up_proj_reshaped)
        ref = (up_proj_reshaped * gate_expanded).view(batch_size, seq_len, intermediate_size)
        
        # Triton kernel
        out = fused_up_proj_gate_activation_triton(x, up_weight, up_bias, gate, num_blocks, line_size)
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