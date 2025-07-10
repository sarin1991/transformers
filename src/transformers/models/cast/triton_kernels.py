import torch
import triton
import triton.language as tl
from typing import Optional
import time
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BS': 16, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_LS': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_BS': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_LS': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_BS': 64, 'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_LS': 64}, num_warps=8, num_stages=2),
    ],
    key=['batch_seq_size', 'hidden_size', 'line_size']   # pick by problem size
)
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
    # Use linear offsets; mask will guard out-of-bounds indices. Using modulo here
    # would alias multiple threads to the same in-bounds element and cause races.
    offs_bs = pid_bs * BLOCK_SIZE_BS + tl.arange(0, BLOCK_SIZE_BS)
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_ls = pid_ls * BLOCK_SIZE_LS + tl.arange(0, BLOCK_SIZE_LS)
    
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
        # Build a fresh mask for the K-dimension tail.
        curr_offs_h = h + offs_h
        mask_h_iter = curr_offs_h < hidden_size

        # Load blocks with proper masking to avoid OOB reads.
        x_block = tl.load(x_ptrs, mask=(mask_bs[:, None] & mask_h_iter[None, :]), other=0.0)
        w_block = tl.load(w_ptrs, mask=(mask_h_iter[:, None] & mask_ls[None, :]), other=0.0)

        # (BLOCK_SIZE_BS x BLOCK_SIZE_H) @ (BLOCK_SIZE_H x BLOCK_SIZE_LS) → (BLOCK_SIZE_BS x BLOCK_SIZE_LS)
        accumulator = tl.dot(x_block, w_block, accumulator)

        # Advance pointers along K dimension
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

    # Launch kernel with a grid derived from the chosen autotune config.
    grid = lambda meta: (
        triton.cdiv(batch_seq_size, meta['BLOCK_SIZE_BS'])
        * num_blocks
        * triton.cdiv(line_size, meta['BLOCK_SIZE_LS']),
    )

    fused_up_proj_gate_activation_kernel[grid](
        x_reshaped, up_weight, up_bias, gate_reshaped, output,
        batch_seq_size, hidden_size, num_blocks, line_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        gate_reshaped.stride(0), gate_reshaped.stride(1),
        output.stride(0), output.stride(1),
        # BLOCK_SIZE_BS=block_size, BLOCK_SIZE_H=block_size, BLOCK_SIZE_LS=block_size, # Removed as Triton picks
    )

    # Reshape back to original shape
    return output.view(batch_size, seq_len, intermediate_size)


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

    print("✅ Triton kernel loaded successfully.") 