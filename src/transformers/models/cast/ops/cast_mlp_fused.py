import torch
from torch.autograd import Function
import torch.nn.functional as F

# Import existing Triton helpers (sort-pack variants)
# Import Triton kernels - works with module execution from cast directory
from kernels.up_proj.triton_cast_kernel_gate_sortpack import (
    fused_up_proj_gate_activation_sparse_triton_sortpack as _up_sparse,
)
from kernels.down_proj.triton_cast_kernel_gate_sortpack import (
    fused_down_proj_sparse_triton_sortpack as _down_sparse,
)
from kernels.weight_grad.triton_cast_kernel_gate_sortpack import (
    fused_weight_grad_sparse_triton_sortpack as _wg_sparse,
)


__all__ = ["cast_mlp_fused"]


def _preprocess_gate_data(gate: torch.Tensor):
    """Preprocess gate data once to avoid redundant sorting in multiple kernel calls.
    
    Returns:
        gate_vals: (NB, max_rows) - sorted gate values, transposed for memory layout
        row_idx: (NB, max_rows) - corresponding row indices, transposed  
        block_counts: (NB,) - number of active rows per block
        max_rows: int - maximum number of active rows across all blocks
    """    
    # Build mask and counts
    mask = gate > 0
    block_counts = mask.sum(dim=0, dtype=torch.int32)  # (NB,)
    max_rows = int(block_counts.max().item())
    
    # Early exit: gate is entirely zero
    if max_rows == 0:
        return None, None, block_counts, max_rows
    
    # Sort each column in descending order – positive values first
    gate_vals_sorted, row_idx_sorted = torch.sort(gate, dim=0, descending=True)
    
    # Truncate to max_rows and transpose so that blocks are contiguous in memory
    gate_vals = gate_vals_sorted[:max_rows, :].t().contiguous()   # (NB, max_rows)
    row_idx = row_idx_sorted[:max_rows, :].t().contiguous().to(torch.int32)  # (NB, max_rows)
    
    return gate_vals, row_idx, block_counts, max_rows


class _CastMLPFusedFunction(Function):
    """Fuses sparse up-projection → gate → sparse down-projection.

    Forward inputs
        x           – (B, S, H)   fp16 / bf16
        gate        – (B, S, NB) float32  (after ReLU)
        up_weight   – (H, I)  – hidden × intermediate (matches Triton helper)
        down_weight – (I, H)  – intermediate × hidden
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor, kernel: str = "sortpack"):
        # ----- basic shapes -----
        assert x.ndim == 3 and gate.ndim == 3, "x and gate must be 3-D (B,S,...) tensors"
        B, S, H = x.shape
        NB = gate.shape[-1]
        assert gate.shape[:2] == (B, S), "gate batch/seq dims must match x"
        assert gate.dtype == torch.float32, "gate must be float32"

        # Expect up_weight (H, I) and down_weight (I, H); caller owns layout.
        I = up_weight.shape[1]  # intermediate dimension
        LS = I // NB
        assert LS * NB == I, "gate dim must divide intermediate size"
        # down_weight is expected to be (I, H) – transpose of up_weight
        assert down_weight.shape == (I, H), "down_weight must have shape (intermediate, hidden)"

        # Ensure contiguous for raw-pointer access
        up_w_T   = up_weight.contiguous()
        down_w_T = down_weight.contiguous()
        gate = F.relu(gate)

        # Up-projection + gating (sparse)
        x_flat = x.view(B * S, H)
        gate_flat = gate.view(B * S, NB)
        
        # Preprocess gate data once for all kernels
        gate_vals, row_idx, block_counts, max_rows = _preprocess_gate_data(gate_flat, NB)
        
        # Compute gated intermediate with real gate
        inter_flat = _up_sparse(
            x_flat,
            up_w_T,
            gate_flat,
            NB,
            LS,
            out_dtype=x.dtype,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )  # (BS, I) - post-gating intermediate

        # Down-projection (sparse)
        out_flat = _down_sparse(
            inter_flat,
            down_w_T,
            gate_flat,
            NB,
            LS,
            out_dtype=x.dtype,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )  # (BS, H)
        out = out_flat.view(B, S, H)

        # Save tensors for backward
        ctx.save_for_backward(x, gate, inter_flat, up_weight, down_weight, gate_vals, row_idx, block_counts)
        ctx.NB = NB
        ctx.LS = LS
        ctx.max_rows = max_rows
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gate, inter_flat, up_weight, down_weight, gate_vals, row_idx, block_counts = ctx.saved_tensors
        NB, LS, max_rows = ctx.NB, ctx.LS, ctx.max_rows

        B, S, H = x.shape
        BS = B * S
        I = inter_flat.shape[-1]

        grad_out_flat = grad_out.contiguous().view(BS, H)
        gate_flat     = gate.contiguous().view(BS, NB)
        x_flat        = x.contiguous().view(BS, H)

        # ---- Triton kernel path ----
        # ---------------- grad w.r.t. down_weight ----------------
        grad_down_w = _wg_sparse(
            inter_flat,
            grad_out_flat,
            gate_flat,
            NB,
            LS,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )  # (I, H)

        # ---------------- grad_inter_flat = grad_out · W_downᵀ ----------------
        down_w_transposed = down_weight.t()  # (H, I)

        grad_inter_flat = _up_sparse(
            grad_out_flat,
            down_w_transposed,
            gate_flat,
            NB,
            LS,
            out_dtype=grad_out.dtype,
            apply_gate=False,
            apply_relu=False,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )  # (BS, I)

        # ---------------- Recompute up_proj ----------------
        up_w_T = up_weight.contiguous()
        up_proj_flat = _up_sparse(
            x_flat,
            up_w_T,
            gate_flat,
            NB,
            LS,
            out_dtype=x.dtype,
            apply_gate=False,
            apply_relu=True,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )

        # ---------------- grad w.r.t. gate -------------------
        grad_gate_flat = (
            grad_inter_flat.view(BS, NB, LS) * up_proj_flat.view(BS, NB, LS)
        ).sum(dim=2)
        pos_mask = (gate_flat > 0).to(grad_gate_flat.dtype)
        grad_gate_flat = grad_gate_flat * pos_mask
        grad_gate = grad_gate_flat.view_as(gate)

        grad_up_relu = grad_inter_flat.view(BS, NB, LS) * gate_flat.to(grad_inter_flat.dtype).view(BS, NB, 1)
        relu_mask = (up_proj_flat.view(BS, NB, LS) > 0)
        grad_up_proj = (grad_up_relu * relu_mask).view(BS, I).to(x.dtype)

        # grad up weight
        grad_up_w_T = _wg_sparse(
            grad_up_proj,
            x_flat,
            gate_flat,
            NB,
            LS,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )
        grad_up_w = grad_up_w_T.t()

        grad_x_flat = _down_sparse(
            grad_up_proj,
            up_weight.t(),
            gate_flat,
            NB,
            LS,
            out_dtype=grad_out.dtype,
            gate_vals=gate_vals,
            row_idx=row_idx,
            block_counts=block_counts,
            max_rows=max_rows,
        )
        grad_x = grad_x_flat.view_as(x)

        grad_kernel = None

        return grad_x, grad_gate, grad_up_w, grad_down_w, grad_kernel


def cast_mlp_fused(
    x: torch.Tensor,
    gate: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    kernel: str = "sortpack",
) -> torch.Tensor:
    """Convenience wrapper around the autograd Function (Triton backend)."""
    return _CastMLPFusedFunction.apply(x, gate, up_weight, down_weight, kernel) 