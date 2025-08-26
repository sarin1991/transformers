import torch
import triton
import triton.language as tl
from typing import Optional
import torch.nn.functional as F
from .triton_cast_kernel_gate_sortpack import fused_down_proj_sparse_triton_sortpack
from .triton_cast_kernel_down_proj_stream_compact import fused_down_proj_sparse_triton_stream_compact
from kernels.stream_compact_index import create_stream_compact_index

# -----------------------------------------------------------------------------
# Dynamically build Triton autotune configurations
#   – Two tile sizes: 64 and 128
#   – Try several warp counts (4, 8, 16, 32) for each size
#   – Try several num_stages (1, 2, 3) for each size
# -----------------------------------------------------------------------------
_CONFIG_WARPS = (4, 8, 16, 32)
_TILE_SIZES   = (64, 128)
_NUM_STAGES = (1, 2, 3)

CONFIGS = []
for tile in _TILE_SIZES:
    for warps in _CONFIG_WARPS:
        for num_stages in _NUM_STAGES:
            CONFIGS.append(
                triton.Config(
                    {
                        'BLOCK_SIZE_BS': tile,
                        'BLOCK_SIZE_I':  tile,
                        'BLOCK_SIZE_H':  tile,
                        'GROUP_SIZE_BS': 16,
                    },
                    num_warps=warps,
                    num_stages=num_stages,
                )
            )


@triton.autotune(
    configs=CONFIGS,
    key=['batch_seq_size', 'intermediate_size', 'hidden_size'],
)
@triton.jit
def fused_down_proj_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_seq_size, intermediate_size, hidden_size,
    stride_x_bs, stride_x_i,
    stride_w_i, stride_w_h,
    stride_out_bs, stride_out_h,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, GROUP_SIZE_BS: tl.constexpr,
):
    """Dense down-projection kernel (matmul) with optional early exit when an entire
    intermediate vector is zero.  *row_mask_ptr* is a 1-D array containing 1 for
    active rows and 0 for fully-zero rows.
    ``x``:   (B·S, I)   – float16/bf16/fp32
    ``w``:   (I,  H)   – float16/bf16/fp32  (transposed weight)
    ``out``: (B·S, H)   – float16/bf16/fp32
    """

    pid = tl.program_id(0)

    # --------------------------------------------------------------
    # Decompose pid into (row_chunk, hidden_chunk) with row grouping
    # --------------------------------------------------------------
    num_col_chunks = tl.cdiv(hidden_size, BLOCK_SIZE_H)  # hidden chunks
    row_chunks = tl.cdiv(batch_seq_size, BLOCK_SIZE_BS)

    col_chunk = pid % num_col_chunks  # hidden chunk index
    row_in_group = (pid // num_col_chunks) % GROUP_SIZE_BS
    row_group = (pid // (num_col_chunks * GROUP_SIZE_BS))
    row_chunk = row_group * GROUP_SIZE_BS + row_in_group

    if row_chunk >= row_chunks:
        return

    offs_bs = row_chunk * BLOCK_SIZE_BS + tl.arange(0, BLOCK_SIZE_BS)
    offs_h  = col_chunk * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_i  = tl.arange(0, BLOCK_SIZE_I)

    mask_bs = offs_bs < batch_seq_size
    mask_h  = offs_h  < hidden_size

    # ------------------------------------------------------------------
    # Accumulator initialisation (float32)
    # ------------------------------------------------------------------
    acc = tl.zeros((BLOCK_SIZE_BS, BLOCK_SIZE_H), dtype=tl.float32)

    # ------------------------------------------------------------------
    # Main matmul loop over intermediate_size (K-dimension)
    # ------------------------------------------------------------------
    for k in range(0, intermediate_size, BLOCK_SIZE_I):
        curr_offs_i = k + offs_i
        mask_i = curr_offs_i < intermediate_size

        x_ptrs = x_ptr + offs_bs[:, None] * stride_x_bs + curr_offs_i[None, :] * stride_x_i
        w_ptrs = w_ptr + curr_offs_i[:, None] * stride_w_i + offs_h[None, :] * stride_w_h

        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_i[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_i[:, None] & mask_h[None, :], other=0.0)

        acc += tl.dot(x_block, w_block)

    # ------------------------------------------------------------------
    # Write back
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + offs_bs[:, None] * stride_out_bs + offs_h[None, :] * stride_out_h

    # Cast accumulator to the requested output dtype and store
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_bs[:, None] & mask_h[None, :])


# ===============================================================
# Python wrapper
# ===============================================================

def fused_down_proj_triton(
    x: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    num_blocks: int,
    line_size: int,
    out_dtype: torch.dtype = torch.float32,
    # New preprocessed gate parameters (ignored for this implementation)
    gate_vals: torch.Tensor = None,
    row_idx: torch.Tensor = None,
    block_counts: torch.Tensor = None,
    max_rows: int = None,
):
    """Fused dense down-projection using Triton.

    Args:
        x:            (batch_seq_size, intermediate_size) – *float16*/*bfloat16*/*float32*
        down_weight:  (intermediate_size, hidden_size)    – *float16*/*bfloat16*/*float32*  (pass ``model.down_proj.weight.t()``)
        gate:         (batch_seq_size, num_blocks)        – *float32* (or any type, will be upcast)
        num_blocks:   ``NB``
        line_size:    ``LS`` (``intermediate_size = NB·LS``)

    Returns:
        Tensor (batch_seq_size, hidden_size) in *float16*/*bfloat16*/*float32*
    """
    batch_seq_size, intermediate_size = x.shape
    hidden_size = down_weight.shape[1]

    # --------------------------------------------------------------
    # Shape & dtype validations
    # --------------------------------------------------------------
    assert intermediate_size == num_blocks * line_size, "Mismatch intermediate size"
    assert down_weight.shape == (intermediate_size, hidden_size)
    assert gate.shape == (batch_seq_size, num_blocks)

    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    assert (
        x.dtype in supported_dtypes and down_weight.dtype in supported_dtypes
    ), f"Unsupported dtype: x={x.dtype}, down_weight={down_weight.dtype}. Supported: {supported_dtypes}"

    # Validate output dtype
    if out_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("out_dtype must be torch.float32, torch.float16, or torch.bfloat16")

    # Use input directly (already flattened)
    x_reshaped = x.contiguous()

    # Allocate output with desired dtype
    output = torch.empty((batch_seq_size, hidden_size), device=x.device, dtype=out_dtype)

    def grid(meta):
        BLK_BS = meta['BLOCK_SIZE_BS']
        BLK_H = meta['BLOCK_SIZE_H']
        GROUP_SIZE_BS = meta['GROUP_SIZE_BS']
        num_col_chunks = triton.cdiv(hidden_size, BLK_H)
        row_chunks = triton.cdiv(batch_seq_size, BLK_BS)
        row_groups = triton.cdiv(row_chunks, GROUP_SIZE_BS)
        return (num_col_chunks * GROUP_SIZE_BS * row_groups,)

    # Map torch dtypes to triton dtypes
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    triton_out_dtype = dtype_map[out_dtype]

    fused_down_proj_kernel[grid](
        x_reshaped,
        down_weight,
        output,
        batch_seq_size,
        intermediate_size,
        hidden_size,
        x_reshaped.stride(0), x_reshaped.stride(1),
        down_weight.stride(0), down_weight.stride(1),
        output.stride(0), output.stride(1),
        out_dtype=triton_out_dtype,
    )

    return output  # already (BS, H)


# ===============================================================
# Debug / numerical accuracy checker (run with `python triton_kernels.py`)
# ===============================================================

def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9):
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    mask = torch.rand_like(gate) < sparsity
    gate[mask] = 0.0
    return gate


def convert_dense_to_stream_compact(x_dense, mappings, num_blocks, line_size):
    """Convert (BS, I) dense → (act_idx, LS) sparse format using stream compact mappings"""
    BS, I = x_dense.shape
    
    # Early exit if no active elements
    if mappings['total_act_idx'] == 0:
        return torch.empty(0, line_size, device=x_dense.device, dtype=x_dense.dtype)
    
    # Reshape to block format: (BS, I) → (BS, NB, LS)
    x_reshaped = x_dense.view(BS, num_blocks, line_size)
    
    # Extract stream compact mappings (new format with local indices + block offsets)
    bs_nb_to_local_idx = mappings['bs_nb_to_local_idx']  # (BS, NB) → local row index
    block_offsets = mappings['block_offsets']             # (NB,) → block start offsets
    total_act_idx = mappings['total_act_idx']
    
    # Build sparse tensor: (act_idx, LS)
    x_sparse = torch.zeros(total_act_idx, line_size, 
                          device=x_dense.device, 
                          dtype=x_dense.dtype)
    
    # Fill sparse tensor using mappings
    active_mask = bs_nb_to_local_idx != 65535
    if active_mask.sum() > 0:
        active_bs, active_nb = torch.where(active_mask)
        # Convert to int32 for indexing, then reconstruct act_indices
        bs_nb_to_local_idx_int32 = bs_nb_to_local_idx.to(torch.int32)
        local_indices = bs_nb_to_local_idx_int32[active_bs, active_nb]
        act_indices = block_offsets[active_nb] + local_indices
        x_sparse[act_indices] = x_reshaped[active_bs, active_nb]
    
    return x_sparse


def debug_large_scale(use_sparse_gate: bool = False, use_fp32: bool = False):
    """Run numerical correctness checks on several shapes."""
    configs = [
        (4, 8, 128, 4, 32),      # (B, S, H, NB, LS)
        (8, 16, 256, 8, 32),
        (32, 32, 512, 8, 64),
    ]

    overall_max_diff_dense = 0.0
    overall_max_diff_sortpack = 0.0
    overall_max_diff_streamcompact = 0.0
    overall_max_diff_streamcompact_2stage = 0.0

    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        intermediate_size = num_blocks * line_size
        print(f"\nConfig: {batch_size}×{seq_len} / H={hidden_size} / NB={num_blocks} / LS={line_size}")

        # Random tensors
        batch_seq_size = batch_size * seq_len
        dtype = torch.float32 if use_fp32 else torch.float16
        x = torch.randn(batch_seq_size, intermediate_size, device="cuda", dtype=dtype)
        down_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)
        gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=0.9 if use_sparse_gate else 0.0)

        # Zero-out gated blocks in x (simulate real pipeline)
        # (BS, I) where I = NB · LS
        x = x.view(batch_seq_size, num_blocks, line_size)
        x = x * gate.unsqueeze(-1).to(dtype=x.dtype)   # element-wise multiply
        x = x.view(batch_seq_size, intermediate_size)

        # Reference PyTorch result (fp32)
        ref_fp32 = F.linear(x.float(), down_weight.t().float()).float()

        # Preprocess gate data once for all kernels
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

        gate_vals, row_idx, block_counts, max_rows = preprocess_gate(gate, num_blocks)
        
        # Stream compact preprocessing
        stream_compact_mappings = create_stream_compact_index(gate)
        x_sparse = convert_dense_to_stream_compact(x, stream_compact_mappings, num_blocks, line_size)

        # Triton dense helper
        out_dense = fused_down_proj_triton(
            x,
            down_weight,
            gate,
            num_blocks,
            line_size,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # Triton SortPack sparse helper
        out_sortpack = fused_down_proj_sparse_triton_sortpack(
            x,
            down_weight,
            gate,
            num_blocks,
            line_size,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )
        
        # Triton StreamCompact sparse helper
        out_streamcompact = fused_down_proj_sparse_triton_stream_compact(
            x_sparse,
            down_weight,
            num_blocks,
            line_size,
            mappings=stream_compact_mappings,
        )
        
        # Triton StreamCompact with two-stage reduction (atomic-free)
        out_streamcompact_2stage = fused_down_proj_sparse_triton_stream_compact(
            x_sparse,
            down_weight,
            num_blocks,
            line_size,
            mappings=stream_compact_mappings,
            two_stage_reduction=True,
        )

        max_diff_dense = torch.max(torch.abs(ref_fp32 - out_dense)).item()
        mean_diff_dense = torch.mean(torch.abs(ref_fp32 - out_dense)).item()

        max_diff_sort = torch.max(torch.abs(ref_fp32 - out_sortpack)).item()
        mean_diff_sort = torch.mean(torch.abs(ref_fp32 - out_sortpack)).item()
        
        max_diff_streamcompact = torch.max(torch.abs(ref_fp32 - out_streamcompact)).item()
        mean_diff_streamcompact = torch.mean(torch.abs(ref_fp32 - out_streamcompact)).item()
        
        max_diff_streamcompact_2stage = torch.max(torch.abs(ref_fp32 - out_streamcompact_2stage)).item()
        mean_diff_streamcompact_2stage = torch.mean(torch.abs(ref_fp32 - out_streamcompact_2stage)).item()

        overall_max_diff_dense = max(overall_max_diff_dense, max_diff_dense)
        overall_max_diff_sortpack = max(overall_max_diff_sortpack, max_diff_sort)
        overall_max_diff_streamcompact = max(overall_max_diff_streamcompact, max_diff_streamcompact)
        overall_max_diff_streamcompact_2stage = max(overall_max_diff_streamcompact_2stage, max_diff_streamcompact_2stage)

        print(f"Dense      → max diff {max_diff_dense:.6e} | mean diff {mean_diff_dense:.6e}")
        print(f"SortPk     → max diff {max_diff_sort:.6e} | mean diff {mean_diff_sort:.6e}")
        print(f"StreamCmpt → max diff {max_diff_streamcompact:.6e} | mean diff {mean_diff_streamcompact:.6e}")
        print(f"StreamCmpt2→ max diff {max_diff_streamcompact_2stage:.6e} | mean diff {mean_diff_streamcompact_2stage:.6e}")

    print(
        f"\nOverall max diff across configs | Dense: {overall_max_diff_dense:.6e} | SortPk: {overall_max_diff_sortpack:.6e} | StreamCmpt: {overall_max_diff_streamcompact:.6e} | StreamCmpt2: {overall_max_diff_streamcompact_2stage:.6e}"
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

    print("✅ Triton down-proj kernel loaded successfully.")
    
    print("\n=== Testing with FP16 ===")
    debug_large_scale(use_sparse_gate=False, use_fp32=False)
    debug_large_scale(use_sparse_gate=True, use_fp32=False)
    
    print("\n=== Testing with FP32 ===")
    debug_large_scale(use_sparse_gate=False, use_fp32=True)
    debug_large_scale(use_sparse_gate=True, use_fp32=True) 