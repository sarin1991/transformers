import torch
import triton
import triton.language as tl
from typing import Optional
import time
import warnings
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


# ---------------------------------------------------------------
# Sparse helper
# ---------------------------------------------------------------


def fused_up_proj_gate_activation_sparse_triton(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    density_threshold: float = 0.5,
):
    """Optimized variant of *fused_up_proj_gate_activation_triton* for sparse ``gate`` tensors.

    The function detects non-zero gate entries, runs the Triton kernel only on the
    corresponding (row, block) pairs, and scatters the results back into a full-sized
    output tensor.  For low sparsity (density > *density_threshold*) it transparently
    falls back to the original dense implementation.

    Args:
        x:              ``(batch, seq_len, hidden_size)``, *float16*
        up_weight:      ``(hidden_size, intermediate_size)``, *float16*
        up_bias:        ``(intermediate_size,)``, *float16*
        gate:           ``(batch, seq_len, num_blocks)``, *float32* or *float16/bf16*
        num_blocks:     number of blocks in the gated FFN
        line_size:      size of each block line (``intermediate_size = num_blocks * line_size``)
        density_threshold: switch to dense path when *gate* is sufficiently dense.
    Returns:
        ``output`` – ``(batch, seq_len, intermediate_size)`` in *float32*
    """

    batch_size, seq_len, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Ensure the same validations as in the dense path
    assert up_weight.shape == (hidden_size, intermediate_size), "Incompatible up_weight shape"
    assert gate.shape == (batch_size, seq_len, num_blocks), "Incompatible gate shape"
    assert x.dtype == torch.float16, f"Input x must be float16, got {x.dtype}"
    assert up_weight.dtype == torch.float16, f"up_weight must be float16, got {up_weight.dtype}"
    assert up_bias.dtype == torch.float16, f"up_bias must be float16, got {up_bias.dtype}"

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Prepare contiguous flattened views
    x_reshaped = x.contiguous().view(-1, hidden_size)         # (B·S, H)
    gate_reshaped = gate.contiguous().view(-1, num_blocks)    # (B·S, NB)
    batch_seq_size = x_reshaped.size(0)

    # ------------------------------------------------------------------
    # Determine sparsity – decide whether to use sparse or dense path
    # ------------------------------------------------------------------
    gate_mask = gate_reshaped != 0
    nnz = int(gate_mask.sum().item())

    if nnz == 0:
        # Everything is zero – return all-zeros tensor fast
        return torch.zeros((batch_size, seq_len, intermediate_size), device=x.device, dtype=torch.float32)

    density = nnz / (batch_seq_size * num_blocks)
    if density >= density_threshold:
        # Not sparse enough – fall back to the dense implementation
        warnings.warn(
            f"[sparse_helper] Density {density:.2%} ≥ threshold {density_threshold:.2%}. Falling back to dense path.",
            stacklevel=2,
        )
        return fused_up_proj_gate_activation_triton(x, up_weight, up_bias, gate, num_blocks, line_size)

    # ------------------------------------------------------------------
    # Sparse path
    # ------------------------------------------------------------------

    # Pre-allocate full output (zero-initialised)
    output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=torch.float32)

    # Indices of (row, block) pairs that have non-zero gate values
    active_pairs = gate_mask.nonzero(as_tuple=False)          # (N, 2)
    rows = active_pairs[:, 0]
    blocks = active_pairs[:, 1]

    # Process each unique block separately
    unique_blocks = blocks.unique(sorted=False)

    for nb in unique_blocks.tolist():
        # Boolean mask & row indices for this block
        rows_mask = (blocks == nb)
        rows_nb = rows[rows_mask]

        if rows_nb.numel() == 0:
            continue  # Safety – should not happen

        # Gather the relevant slices – keep everything contiguous
        x_subset = x_reshaped.index_select(0, rows_nb).contiguous()       # (K, H)
        gate_subset = gate_reshaped[rows_nb, nb].unsqueeze(1).contiguous()  # (K, 1)

        # Slice weights & bias for this block only
        start_col = nb * line_size
        end_col = start_col + line_size
        w_slice = up_weight[:, start_col:end_col].contiguous()  # (H, L)
        b_slice = up_bias[start_col:end_col].contiguous()       # (L,)

        K = x_subset.size(0)

        # Allocate per-block output buffer
        out_subset = torch.empty((K, line_size), device=x.device, dtype=torch.float32)

        # Kernel grid – num_blocks = 1 for this invocation
        grid = lambda meta: (
            triton.cdiv(K, meta['BLOCK_SIZE_BS']) * 1 * triton.cdiv(line_size, meta['BLOCK_SIZE_LS']),
        )

        fused_up_proj_gate_activation_kernel[grid](
            x_subset,                     # x_ptr
            w_slice,                      # up_weight_ptr
            b_slice,                      # up_bias_ptr
            gate_subset,                  # gate_ptr
            out_subset,                   # output_ptr
            K,                            # batch_seq_size (== rows in this slice)
            hidden_size,
            1,                            # num_blocks (single block per call)
            line_size,
            x_subset.stride(0), x_subset.stride(1),
            w_slice.stride(0), w_slice.stride(1),
            gate_subset.stride(0), gate_subset.stride(1),
            out_subset.stride(0), out_subset.stride(1),
        )

        # Scatter results back into the full output tensor
        output[rows_nb, start_col:end_col] = out_subset

    # Reshape back to original 3-D shape
    return output.view(batch_size, seq_len, intermediate_size)


# ===============================================================
# Debug / numerical verification helpers
# ===============================================================


def _make_sparse_gate(batch_size: int, seq_len: int, num_blocks: int, sparsity: float = 0.9):
    """Utility: generate a gate tensor with given *sparsity* on CUDA.

    *sparsity* denotes the fraction of **zeros** (e.g. 0.9 → 10 % non-zero).
    Returns a `torch.float32` tensor on the current CUDA device.
    """
    gate = torch.rand(batch_size, seq_len, num_blocks, device="cuda", dtype=torch.float32)
    if sparsity > 0.0:
        mask = torch.rand_like(gate) < sparsity  # True for zeros
        gate[mask] = 0.0
    return gate


def debug_large_scale(use_sparse_gate: bool = False):
    """Run several larger shapes to verify correctness of dense & sparse helpers.

    If *use_sparse_gate* is True, create a gate tensor with ~10 % non-zero entries;
    otherwise use a fully dense random gate.
    """
    print("\n=== Large Scale Debug (triton_kernels) ===")

    test_configs = [
        (2, 4, 64, 4, 16),
        (4, 8, 128, 4, 32),
        (8, 16, 256, 8, 32),
        (16, 32, 512, 8, 64),
    ]

    overall_max_dense = 0.0
    overall_max_sparse = 0.0

    for batch_size, seq_len, hidden_size, num_blocks, line_size in test_configs:
        intermediate_size = num_blocks * line_size
        cfg = f"{batch_size}x{seq_len}x{hidden_size} | blocks={num_blocks}, line={line_size}"
        print(f"\nConfig: {cfg}")

        # Random fp16 data (CUDA)
        x_fp16 = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

        # Weight / bias: kernel expects (H, I); PyTorch linear expects (I, H)
        up_weight_fp16 = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float16)
        up_bias_fp16 = torch.randn(intermediate_size, device="cuda", dtype=torch.float16)

        # Gate
        if use_sparse_gate:
            gate_fp32 = _make_sparse_gate(batch_size, seq_len, num_blocks, sparsity=0.9)  # 10% non-zero
        else:
            gate_fp32 = torch.rand(batch_size, seq_len, num_blocks, device="cuda", dtype=torch.float32)

        # PyTorch reference (float32)
        up_proj_fp32 = F.relu(F.linear(x_fp16.float(), up_weight_fp16.t().float(), up_bias_fp16.float()))
        ref_reshaped = up_proj_fp32.view(batch_size, seq_len, num_blocks, line_size)
        ref_fp32 = (ref_reshaped * gate_fp32.unsqueeze(-1)).view(batch_size, seq_len, intermediate_size)

        # Dense Triton helper
        out_dense = fused_up_proj_gate_activation_triton(
            x_fp16,
            up_weight_fp16,
            up_bias_fp16,
            gate_fp32,
            num_blocks,
            line_size,
        )

        # Sparse Triton helper (may internally fall back)
        out_sparse = fused_up_proj_gate_activation_sparse_triton(
            x_fp16,
            up_weight_fp16,
            up_bias_fp16,
            gate_fp32,
            num_blocks,
            line_size,
        )

        # Compute diffs
        max_diff_dense = torch.max(torch.abs(ref_fp32 - out_dense)).item()
        mean_diff_dense = torch.mean(torch.abs(ref_fp32 - out_dense)).item()

        max_diff_sparse = torch.max(torch.abs(ref_fp32 - out_sparse)).item()
        mean_diff_sparse = torch.mean(torch.abs(ref_fp32 - out_sparse)).item()

        print(f"Dense   → max diff {max_diff_dense:.6e} | mean diff {mean_diff_dense:.6e}")
        print(f"Sparse  → max diff {max_diff_sparse:.6e} | mean diff {mean_diff_sparse:.6e}")

        overall_max_dense = max(overall_max_dense, max_diff_dense)
        overall_max_sparse = max(overall_max_sparse, max_diff_sparse)

    print(
        f"\nOverall max diff across configs | Dense: {overall_max_dense:.6e} | Sparse: {overall_max_sparse:.6e}"
    )


# ===============================================================
# Main (basic self-test)
# ===============================================================


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

    # Run numerical debug for dense and sparse gates
    debug_large_scale(use_sparse_gate=False)
    debug_large_scale(use_sparse_gate=True) 