import torch
from torch.autograd import Function

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

        # Up-projection + gating (sparse)
        x_flat = x.view(B * S, H)
        gate_flat = gate.view(B * S, NB)
        
        # Compute gated intermediate with real gate
        inter_flat = _up_sparse(
            x_flat,
            up_w_T,
            gate_flat,
            NB,
            LS,
            out_dtype=x.dtype,
        )  # (BS, I) - post-gating intermediate

        # Down-projection (sparse)
        out_flat = _down_sparse(
            inter_flat,
            down_w_T,
            gate_flat,
            NB,
            LS,
            out_dtype=x.dtype,
        )  # (BS, H)
        out = out_flat.view(B, S, H)

        # Save tensors for backward
        ctx.save_for_backward(x, gate, inter_flat, up_weight, down_weight)
        ctx.NB = NB
        ctx.LS = LS
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gate, inter_flat, up_weight, down_weight = ctx.saved_tensors
        NB, LS = ctx.NB, ctx.LS

        B, S, H = x.shape
        BS = B * S
        I = inter_flat.shape[-1]

        grad_out_flat = grad_out.contiguous().view(BS, H)
        gate_flat     = gate.contiguous().view(BS, NB)
        x_flat        = x.contiguous().view(BS, H)

        # ---------------- grad w.r.t. down_weight ----------------
        grad_down_w = _wg_sparse(
            inter_flat,
            grad_out_flat,
            gate_flat,
            NB,
            LS,
        )  # (I, H) matches down_weight layout

        # ---------------- grad_up_proj = grad_out · W_downᵀ ----------------
        # Use up-proj kernel with actual gates
        down_w_transposed = down_weight.t().contiguous()  # (H, I)
        
        grad_up_proj = _up_sparse(
            grad_out_flat,
            down_w_transposed,
            gate_flat,  # Use actual gates, not dummy
            NB,
            LS,
            out_dtype=grad_out.dtype,
        )  # (BS, I) - gradient w.r.t. pre-gating intermediate

        # ---------------- grad w.r.t. gate -------------------
        # Use epsilon threshold for numerical stability
        eps = 1e-9
        gate_expanded = gate_flat.repeat_interleave(LS, dim=1)
        grad_gate_flat = torch.where(
            gate_flat > eps,
            (grad_up_proj * inter_flat / (gate_expanded ** 2)).view(BS, NB, LS).sum(dim=2),
            0
        )  # (BS, NB)
        grad_gate = grad_gate_flat.view_as(gate)

        # ---------------- grad w.r.t. up_weight -------------------
        grad_up_w_T = _wg_sparse(
            grad_up_proj,
            x_flat,
            gate_flat,
            NB,
            LS,
        )  # (I, H)
        grad_up_w = grad_up_w_T.t().contiguous()  # (H,I)

        # ---------------- grad w.r.t. input ----------------------
        # Use down-proj kernel in reverse: grad_up_proj → grad_x
        grad_x_flat = _down_sparse(
            grad_up_proj,
            up_weight.t().contiguous(),  # (I, H) - transpose to match kernel expectation
            gate_flat,
            NB,
            LS,
            out_dtype=grad_out.dtype,
        )  # (BS, H)
        grad_x = grad_x_flat.view_as(x)

        grad_kernel = None

        return grad_x, grad_gate, grad_up_w, grad_down_w, grad_kernel


def cast_mlp_fused(x: torch.Tensor, gate: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor, *, kernel: str = "sortpack") -> torch.Tensor:
    """Convenience wrapper around the autograd Function."""
    return _CastMLPFusedFunction.apply(x, gate, up_weight, down_weight, kernel) 