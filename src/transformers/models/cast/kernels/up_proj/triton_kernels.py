import torch
import triton
import triton.language as tl
from typing import Optional
import torch.nn.functional as F
from .triton_cast_kernel import (
    fused_up_proj_gate_activation_sparse_triton_optimized as fused_up_proj_gate_activation_sparse_triton_opt,
)
from .triton_cast_kernel_csr import (
    fused_up_proj_gate_activation_sparse_triton_csr as fused_up_proj_gate_activation_sparse_triton_csr,
)
# Unified CSR builder+compute helper
from .triton_cast_kernel_gate_sortpack import (
    fused_up_proj_gate_activation_sparse_triton_sortpack as fused_up_proj_gate_activation_sparse_triton_sortpack,
)
# Stream compact helper
from .triton_cast_kernel_stream_compact import (
    fused_up_proj_gate_activation_sparse_triton_stream_compact as fused_up_proj_gate_activation_sparse_triton_stream_compact,
)
# Stream compact index preprocessing
from kernels.stream_compact_index import create_stream_compact_index


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
    x_ptr, up_weight_ptr, gate_ptr, output_ptr,
    batch_seq_size, hidden_size, num_blocks, line_size,
    stride_x_bs, stride_x_h,
    stride_w_h, stride_w_ls,
    stride_g_bs, stride_g_nb,
    stride_out_bs, stride_out_ls,
    out_dtype: tl.constexpr,
    apply_gate: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_LS: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. up_proj = F.relu(x @ up_weight)
    2. output = up_proj * gate (reshaped and applied)
    
    Following proper Triton matrix multiplication pattern with gating logic added.
    Blocks across batch_seq, num_blocks, and line_size dimensions.
    All operations in float16 for consistency.
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block indices - block across batch_seq, num_blocks, and line_size
    # Similar to reference Triton pattern: pid = pid_m * num_pid_n + pid_n
    num_pid_bs = tl.cdiv(batch_seq_size, BLOCK_SIZE_BS)
    num_pid_nb = num_blocks
    num_pid_ls = tl.cdiv(line_size, BLOCK_SIZE_LS)
    
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
    # Clamp negative gate values to zero
    g = tl.where(g > 0, g, 0.0)
    
    # Check if all gates are zero - early return
    g_sum = tl.sum(g)
    zero_gates = (g_sum == 0.0)
    
    # Early return if all gates are zero
    if zero_gates:
        # Store zeros in output for this block
        out_ptrs = output_ptr + (offs_bs[:, None] * stride_out_bs + 
                               (pid_nb * line_size + offs_ls)[None, :] * stride_out_ls)
        tl.store(out_ptrs, tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_LS), dtype=out_dtype), 
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
        accumulator += tl.dot(x_block, w_block)

        # Advance pointers along K dimension
        x_ptrs += BLOCK_SIZE_H * stride_x_h
        w_ptrs += BLOCK_SIZE_H * stride_w_h
    
    # Apply ReLU (optional) - keep in float32 for precision
    if apply_relu:
        accumulator = tl.where(accumulator > 0, accumulator, 0.0)
    
    # Optionally apply gate activation
    if apply_gate:
        output = accumulator * g[:, None]
    else:
        output = accumulator
    
    # Store output
    out_ptrs = output_ptr + (offs_bs[:, None] * stride_out_bs + 
                           (pid_nb * line_size + offs_ls)[None, :] * stride_out_ls)
    tl.store(out_ptrs, output.to(out_dtype), mask=(mask_bs[:, None] & mask_ls[None, :]))


def fused_up_proj_gate_activation_triton(x, up_weight, gate, num_blocks, line_size, out_dtype: torch.dtype = torch.float32, apply_gate: bool = True, apply_relu: bool = True, gate_vals: torch.Tensor = None, row_idx: torch.Tensor = None, block_counts: torch.Tensor = None, max_rows: int = None):
    """
    Fused Triton implementation that performs up projection and gate activation in one kernel.
    Args:
        x: Input tensor of shape (batch_seq_size, hidden_size) – *float16*/*bfloat16*/*float32*
        up_weight: Up projection weight of shape (hidden_size, intermediate_size) – *float16*/*bfloat16*/*float32*
        gate: Pre-calculated gate tensor of shape (batch_seq_size, num_blocks) - must be float32
        num_blocks: Number of blocks
        line_size: Size of each line within a block
    Returns:
        Output tensor of shape (batch_seq_size, intermediate_size), in requested dtype
    """
    batch_seq_size, hidden_size = x.shape
    intermediate_size = num_blocks * line_size
    assert up_weight.shape == (hidden_size, intermediate_size), "Incompatible up_weight shape"
    assert gate.shape == (batch_seq_size, num_blocks), "Incompatible gate shape"
    
    # Check that inputs are correct dtypes (fp16, bf16, or fp32)
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in supported_dtypes, f"Input x must be fp16/bf16/fp32, got {x.dtype}"
    assert up_weight.dtype in supported_dtypes, f"up_weight must be fp16/bf16/fp32, got {up_weight.dtype}"
    
    # Optionally upcast gate to float32 if not already
    if gate.dtype != torch.float32:
        gate = gate.float()

    # Ensure all inputs are contiguous
    x = x.contiguous()
    up_weight = up_weight.contiguous()
    gate = gate.contiguous()
    x_reshaped = x  # already (BS, H)
    gate_reshaped = gate  # already (BS, NB)

    # Allocate output as 2D tensor with requested dtype
    output = torch.empty((batch_seq_size, num_blocks * line_size), 
                        device=x.device, dtype=out_dtype)

    # Launch kernel with a grid derived from the chosen autotune config.
    grid = lambda meta: (
        triton.cdiv(batch_seq_size, meta['BLOCK_SIZE_BS'])
        * num_blocks
        * triton.cdiv(line_size, meta['BLOCK_SIZE_LS']),
    )

    # Map torch dtypes to triton dtypes
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    fused_up_proj_gate_activation_kernel[grid](
        x_reshaped, up_weight, gate_reshaped, output,
        batch_seq_size, hidden_size, num_blocks, line_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        gate_reshaped.stride(0), gate_reshaped.stride(1),
        output.stride(0), output.stride(1),
        out_dtype=triton_out_dtype,
        apply_gate=apply_gate,
        apply_relu=apply_relu,
    )

    return output  # already (BS, I)


# ---------------------------------------------------------------
# Sparse helper
# ---------------------------------------------------------------


def fused_up_proj_gate_activation_sparse_triton(
    x: torch.Tensor,
    up_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    out_dtype: torch.dtype = torch.float32,
    apply_gate: bool = True,
    apply_relu: bool = True,
    gate_vals: torch.Tensor = None,
    row_idx: torch.Tensor = None, 
    block_counts: torch.Tensor = None,
    max_rows: int = None,
):
    """Sparse Triton helper for up-projection + gating + activation.

    This is a drop-in replacement for the dense helper that processes only the
    non-zero entries in the gate tensor. It may internally fall back to the dense
    path if the gate is sufficiently dense.

    Args:
        x:              ``(batch_seq_size, hidden_size)``, *float16*/*bfloat16*/*float32*

        up_weight:      ``(hidden_size, intermediate_size)``, *float16*/*bfloat16*/*float32*

        gate:           ``(batch_seq_size, num_blocks)``, *float32* or *float16/bf16*
        num_blocks:     number of blocks in the gated FFN
        line_size:      size of each block line (``intermediate_size = num_blocks * line_size``)
        density_threshold: switch to dense path when *gate* is sufficiently dense.
    Returns:
        ``output`` – ``(batch_seq_size, intermediate_size)`` in requested dtype
    """

    batch_seq_size, hidden_size = x.shape
    intermediate_size = num_blocks * line_size

    # Ensure the same validations as in the dense path
    assert up_weight.shape == (hidden_size, intermediate_size), "Incompatible up_weight shape"
    assert gate.shape == (batch_seq_size, num_blocks), "Incompatible gate shape"
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert x.dtype in supported_dtypes, f"Input x must be fp16/bf16/fp32, got {x.dtype}"
    assert up_weight.dtype in supported_dtypes, f"up_weight must be fp16/bf16/fp32, got {up_weight.dtype}"


    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("out_dtype must be torch.float32, torch.float16, or torch.bfloat16")

    if gate.dtype != torch.float32:
        gate = gate.float()

    # Prepare contiguous views (already 2D)
    x_reshaped = x.contiguous()         # (batch_seq_size, H)
    gate_reshaped = gate.contiguous()   # (batch_seq_size, NB)

    # ------------------------------------------------------------------
    # Determine sparsity – decide whether to use sparse or dense path
    # ------------------------------------------------------------------
    gate_mask = gate_reshaped != 0
    nnz = int(gate_mask.sum().item())

    if nnz == 0:
        # Everything is zero – return all-zeros tensor fast
        return torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)


    # ------------------------------------------------------------------
    # Sparse path
    # ------------------------------------------------------------------

    # Pre-allocate full output (zero-initialised)
    output = torch.zeros((batch_seq_size, intermediate_size), device=x.device, dtype=out_dtype)

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

        # Slice weights for this block only
        start_col = nb * line_size
        end_col = start_col + line_size
        w_slice = up_weight[:, start_col:end_col].contiguous()  # (H, L)

        K = x_subset.size(0)

        # Allocate per-block output buffer
        out_subset = torch.empty((K, line_size), device=x.device, dtype=out_dtype)

        # Kernel grid – num_blocks = 1 for this invocation
        grid = lambda meta: (
            triton.cdiv(K, meta['BLOCK_SIZE_BS']) * 1 * triton.cdiv(line_size, meta['BLOCK_SIZE_LS']),
        )

        fused_up_proj_gate_activation_kernel[grid](
            x_subset,                     # x_ptr
            w_slice,                      # up_weight_ptr
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
            out_dtype=tl.float16 if out_dtype == torch.float16 else tl.float32,
            apply_gate=apply_gate,
            apply_relu=apply_relu,
        )

        # Scatter results back into the full output tensor
        output[rows_nb, start_col:end_col] = out_subset

    # Reshape back to original 3-D shape
    return output.view(batch_seq_size, intermediate_size)


# ===============================================================
# Debug / numerical verification helpers
# ===============================================================


def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9):
    """Utility: generate a gate tensor with given *sparsity* on CUDA.

    *sparsity* denotes the fraction of **zeros** (e.g. 0.9 → 10 % non-zero).
    Returns a `torch.float32` tensor on the current CUDA device.
    """
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    if sparsity > 0.0:
        mask = torch.rand_like(gate) < sparsity  # True for zeros
        gate[mask] = 0.0
    return gate


def stream_compact_to_dense_blocks(stream_compact_output, mappings, batch_seq_size, num_blocks, line_size):
    """
    Convert (act_idx, LS) to (BS, NB, LS) for easy comparison with reference.
    
    Args:
        stream_compact_output: (act_idx, LS) tensor from stream compact kernel
        mappings: dict from create_stream_compact_index
        batch_seq_size: BS dimension
        num_blocks: NB dimension  
        line_size: LS dimension
        
    Returns:
        dense_blocks: (BS, NB, LS) tensor with zeros for inactive blocks
    """
    bs_nb_to_local_idx = mappings['bs_nb_to_local_idx']  # (BS, NB) → local row index
    block_offsets = mappings['block_offsets']             # (NB,) → block start offsets
    
    # Create output tensor (zeros for inactive blocks)
    dense_blocks = torch.zeros((batch_seq_size, num_blocks, line_size), 
                              device=stream_compact_output.device, 
                              dtype=stream_compact_output.dtype)
    
    # Vectorized approach using advanced indexing
    active_mask = bs_nb_to_local_idx >= 0  # (BS, NB)
    
    # Get active positions and their corresponding act_idx values
    bs_indices, nb_indices = torch.nonzero(active_mask, as_tuple=True)
    # Reconstruct act_indices from local indices + block offsets
    local_indices = bs_nb_to_local_idx[active_mask]  # Only active local indices
    act_indices = block_offsets[nb_indices] + local_indices
    
    # Vectorized assignment
    dense_blocks[bs_indices, nb_indices, :] = stream_compact_output[act_indices, :]
    
    return dense_blocks


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
    overall_max_opt = 0.0
    overall_max_csr = 0.0
    overall_max_sortpack = 0.0
    overall_max_stream_compact = 0.0

    for batch_size, seq_len, hidden_size, num_blocks, line_size in test_configs:
        intermediate_size = num_blocks * line_size
        cfg = f"{batch_size}x{seq_len}x{hidden_size} | blocks={num_blocks}, line={line_size}"
        print(f"\nConfig: {cfg}")

        # Random fp16 data (CUDA)
        batch_seq_size = batch_size * seq_len
        x_fp16 = torch.randn(batch_seq_size, hidden_size, device="cuda", dtype=torch.float16)

        # Weight: kernel expects (H, I); PyTorch linear expects (I, H)
        up_weight_fp16 = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float16)

        # Gate
        if use_sparse_gate:
            gate_fp32 = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=0.9)  # 10% non-zero
        else:
            gate_fp32 = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)

        # PyTorch reference (float32)
        up_proj_fp32 = F.relu(F.linear(x_fp16.float(), up_weight_fp16.t().float()))
        ref_reshaped = up_proj_fp32.view(batch_seq_size, num_blocks, line_size)
        ref_fp32 = (ref_reshaped * gate_fp32.unsqueeze(-1)).view(batch_seq_size, intermediate_size)

        # Preprocess gate data once for all helpers
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

        # Dense Triton helper
        out_dense = fused_up_proj_gate_activation_triton(
            x_fp16,
            up_weight_fp16,
            gate_fp32,
            num_blocks,
            line_size,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # Baseline sparse Triton helper (may internally fall back)
        out_sparse = fused_up_proj_gate_activation_sparse_triton(
            x_fp16,
            up_weight_fp16,
            gate_fp32,
            num_blocks,
            line_size,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # Optimized sparse helper
        out_opt = fused_up_proj_gate_activation_sparse_triton_opt(
            x_fp16,
            up_weight_fp16,
            gate_fp32,
            num_blocks,
            line_size,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # CSR sparse helper (original two-pass)
        out_csr = fused_up_proj_gate_activation_sparse_triton_csr(
            x_fp16,
            up_weight_fp16,
            gate_fp32,
            num_blocks,
            line_size,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # SortPack helper - reuse preprocessed gate data
        out_sortpack = fused_up_proj_gate_activation_sparse_triton_sortpack(
            x_fp16,
            up_weight_fp16,
            gate_fp32,
            num_blocks,
            line_size,
            zero_init=True,  # Required for sparse data to avoid uninitialized output positions
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # Stream compact helper - NEW
        mappings = create_stream_compact_index(gate_fp32)
        out_stream_compact_raw = fused_up_proj_gate_activation_sparse_triton_stream_compact(
            x_fp16,
            up_weight_fp16,
            gate_fp32,
            num_blocks,
            line_size,
            mappings=mappings,
        )
        
        # Convert stream compact output to (BS, NB, LS) for comparison
        out_stream_compact = stream_compact_to_dense_blocks(out_stream_compact_raw, mappings, batch_seq_size, num_blocks, line_size)

        # Compute diffs
        max_diff_dense = torch.max(torch.abs(ref_fp32 - out_dense)).item()
        mean_diff_dense = torch.mean(torch.abs(ref_fp32 - out_dense)).item()

        max_diff_sparse = torch.max(torch.abs(ref_fp32 - out_sparse)).item()
        mean_diff_sparse = torch.mean(torch.abs(ref_fp32 - out_sparse)).item()

        max_diff_opt = torch.max(torch.abs(ref_fp32 - out_opt)).item()
        mean_diff_opt = torch.mean(torch.abs(ref_fp32 - out_opt)).item()

        max_diff_csr = torch.max(torch.abs(ref_fp32 - out_csr)).item()
        mean_diff_csr = torch.mean(torch.abs(ref_fp32 - out_csr)).item()

        max_diff_sortpack = torch.max(torch.abs(ref_fp32 - out_sortpack)).item()
        mean_diff_sortpack = torch.mean(torch.abs(ref_fp32 - out_sortpack)).item()

        # Convert reference to (BS, NB, LS) format for stream compact comparison
        ref_reshaped = ref_fp32.view(batch_seq_size, num_blocks, line_size)
        max_diff_stream_compact = torch.max(torch.abs(ref_reshaped - out_stream_compact)).item()
        mean_diff_stream_compact = torch.mean(torch.abs(ref_reshaped - out_stream_compact)).item()

        print(f"Dense   → max diff {max_diff_dense:.6e} | mean diff {mean_diff_dense:.6e}")
        print(f"Sparse  → max diff {max_diff_sparse:.6e} | mean diff {mean_diff_sparse:.6e}")
        print(f"OptSpa  → max diff {max_diff_opt  :.6e} | mean diff {mean_diff_opt  :.6e}")
        print(f"CSR     → max diff {max_diff_csr  :.6e} | mean diff {mean_diff_csr  :.6e}")
        print(f"SortPk  → max diff {max_diff_sortpack:.6e} | mean diff {mean_diff_sortpack:.6e}")
        print(f"StrmCmp → max diff {max_diff_stream_compact:.6e} | mean diff {mean_diff_stream_compact:.6e}")

        overall_max_dense = max(overall_max_dense, max_diff_dense)
        overall_max_sparse = max(overall_max_sparse, max_diff_sparse)
        overall_max_opt = max(overall_max_opt, max_diff_opt)
        overall_max_csr = max(overall_max_csr, max_diff_csr)
        overall_max_sortpack = max(overall_max_sortpack, max_diff_sortpack)
        overall_max_stream_compact = max(overall_max_stream_compact, max_diff_stream_compact)

    print(
        f"\nOverall max diff across configs | Dense: {overall_max_dense:.6e} | Sparse: {overall_max_sparse:.6e} | OptSpa: {overall_max_opt:.6e} | CSR: {overall_max_csr:.6e} | SortPk: {overall_max_sortpack:.6e} | StrmCmp: {overall_max_stream_compact:.6e}"
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