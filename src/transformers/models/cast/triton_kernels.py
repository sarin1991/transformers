import torch
import triton
import triton.language as tl
from typing import Optional
import time


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is the change in `a_ptr`
    # when moving along the M dimension (axis 0).
    stride_am, stride_ak,  # A matrix strides
    stride_bk, stride_bn,  # B matrix strides
    stride_cm, stride_cn,  # C matrix strides
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Number of columns in A and rows in B
    # Number of rows in A and columns in C
    # Number of columns in B and rows in C
    # Number of warps
    num_warps: tl.constexpr,
    # Number of stages for pipeline
    num_stages: tl.constexpr,
):
    """
    Compute C = A x B.
    A is of shape (M, K), B is of shape (K, N) and C is of shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def matmul_kernel_fp32(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is the change in `a_ptr`
    # when moving along the M dimension (axis 0).
    stride_am, stride_ak,  # A matrix strides
    stride_bk, stride_bn,  # B matrix strides
    stride_cm, stride_cn,  # C matrix strides
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Number of warps
    num_warps: tl.constexpr,
    # Number of stages for pipeline
    num_stages: tl.constexpr,
):
    """
    Compute C = A x B.
    A is of shape (M, K), B is of shape (K, N) and C is of shape (M, N)
    All matrices are in fp32
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.jit
def matmul_kernel_with_bias(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is the change in `a_ptr`
    # when moving along the M dimension (axis 0).
    stride_am, stride_ak,  # A matrix strides
    stride_bk, stride_bn,  # B matrix strides
    stride_cm, stride_cn,  # C matrix strides
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Number of warps
    num_warps: tl.constexpr,
    # Number of stages for pipeline
    num_stages: tl.constexpr,
):
    """
    Compute C = A x B + bias.
    A is of shape (M, K), B is of shape (K, N), bias is of shape (N,) and C is of shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    accumulator += bias[None, :]

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b, bias=None, use_fp32=False):
    """
    Performs matrix multiplication C = A @ B + bias using Triton kernels.
    
    Args:
        a: Input matrix of shape (M, K)
        b: Input matrix of shape (K, N) 
        bias: Optional bias vector of shape (N,)
        use_fp32: Whether to use fp32 precision for computation
    
    Returns:
        Output matrix of shape (M, N)
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    M, K = a.shape
    K, N = b.shape
    
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Choose kernel based on precision and bias
    if bias is not None:
        kernel = matmul_kernel_with_bias
        kernel[grid](
            a, b, bias, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            num_warps=4,
            num_stages=3,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
    elif use_fp32:
        kernel = matmul_kernel_fp32
        kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            num_warps=4,
            num_stages=3,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
    else:
        kernel = matmul_kernel
        kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            num_warps=4,
            num_stages=3,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
    
    return c


@triton.jit
def gate_activation_kernel(
    x_ptr, g_ptr, output_ptr,
    batch_size, seq_len, num_blocks, line_size,
    stride_xb, stride_xl, stride_xnb, stride_xls,
    stride_gb, stride_gl, stride_gnb,
    stride_outb, stride_outl, stride_outnb, stride_outls,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_L: tl.constexpr, 
    BLOCK_SIZE_NB: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """
    Custom gate activation kernel for CastMLP.
    Computes: output[b, l, nb, ls] = x[b, l, nb, ls] * g[b, l, nb]
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
    offs_nb = nb * BLOCK_SIZE_NB + tl.arange(0, BLOCK_SIZE_NB)
    offs_ls = tl.arange(0, BLOCK_SIZE_LS)
    
    # Create masks
    mask_b = offs_b < batch_size
    mask_l = offs_l < seq_len
    mask_nb = offs_nb < num_blocks
    mask_ls = offs_ls < line_size
    
    # Load gate values
    g_ptrs = g_ptr + (offs_b[:, None, None] * stride_gb + 
                     offs_l[None, :, None] * stride_gl + 
                     offs_nb[None, None, :] * stride_gnb)
    g = tl.load(g_ptrs, mask=(mask_b[:, None, None] & mask_l[None, :, None] & mask_nb[None, None, :]), other=0.0)
    
    # Load input values
    x_ptrs = x_ptr + (offs_b[:, None, None, None] * stride_xb + 
                     offs_l[None, :, None, None] * stride_xl + 
                     offs_nb[None, None, :, None] * stride_xnb + 
                     offs_ls[None, None, None, :] * stride_xls)
    x = tl.load(x_ptrs, mask=(mask_b[:, None, None, None] & mask_l[None, :, None, None] & 
                             mask_nb[None, None, :, None] & mask_ls[None, None, None, :]), other=0.0)
    
    # Apply gate activation
    output = x * g[:, :, :, None]
    
    # Store output
    out_ptrs = output_ptr + (offs_b[:, None, None, None] * stride_outb + 
                           offs_l[None, :, None, None] * stride_outl + 
                           offs_nb[None, None, :, None] * stride_outnb + 
                           offs_ls[None, None, None, :] * stride_outls)
    tl.store(out_ptrs, output, mask=(mask_b[:, None, None, None] & mask_l[None, :, None, None] & 
                                   mask_nb[None, None, :, None] & mask_ls[None, None, None, :]))


def gate_activation_triton(x, g, num_blocks, line_size):
    """
    Triton implementation of gate activation for CastMLP.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, intermediate_size)
        g: Gate tensor of shape (batch_size, seq_len, num_blocks)
        num_blocks: Number of blocks
        line_size: Size of each line within a block
    
    Returns:
        Output tensor of shape (batch_size, seq_len, intermediate_size)
    """
    batch_size, seq_len, intermediate_size = x.shape
    assert intermediate_size == num_blocks * line_size, "Incompatible dimensions"
    
    # Reshape x to (batch_size, seq_len, num_blocks, line_size)
    x_reshaped = x.view(batch_size, seq_len, num_blocks, line_size)
    
    # Allocate output
    output = torch.empty_like(x_reshaped)
    
    # Launch kernel
    grid = (batch_size * seq_len * num_blocks,)
    
    gate_activation_kernel[grid](
        x_reshaped, g, output,
        batch_size, seq_len, num_blocks, line_size,
        x_reshaped.stride(0), x_reshaped.stride(1), x_reshaped.stride(2), x_reshaped.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_B=1, BLOCK_SIZE_L=1, BLOCK_SIZE_NB=1, BLOCK_SIZE_LS=64,
    )
    
    # Reshape back to original shape
    return output.view(batch_size, seq_len, intermediate_size)


def test_gate_activation_triton():
    """
    Test the correctness of gate_activation_triton against a PyTorch reference implementation.
    """
    print("Testing gate_activation_triton correctness...")
    # Test a few configurations
    test_configs = [
        (2, 8, 4, 16),   # (batch_size, seq_len, num_blocks, line_size)
        (4, 16, 8, 32),
        (1, 4, 2, 8),
    ]
    for batch_size, seq_len, num_blocks, line_size in test_configs:
        intermediate_size = num_blocks * line_size
        x = torch.randn(batch_size, seq_len, intermediate_size, device='cuda', dtype=torch.float16)
        g = torch.randn(batch_size, seq_len, num_blocks, device='cuda', dtype=torch.float16)
        # PyTorch reference
        x_reshaped = x.view(batch_size, seq_len, num_blocks, line_size)
        g_expanded = g.unsqueeze(-1).expand_as(x_reshaped)
        ref = (x_reshaped * g_expanded).view(batch_size, seq_len, intermediate_size)
        # Triton kernel
        out = gate_activation_triton(x, g, num_blocks, line_size)
        max_diff = torch.max(torch.abs(ref - out)).item()
        mean_diff = torch.mean(torch.abs(ref - out)).item()
        print(f"Config {batch_size}x{seq_len}x{num_blocks}x{line_size}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")
        assert max_diff < 1e-2, f"Test failed for config {batch_size}x{seq_len}x{num_blocks}x{line_size}"
    print("✅ gate_activation_triton correctness test passed!")


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
    test_gate_activation_triton() 