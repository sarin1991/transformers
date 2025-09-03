import torch
from torch.autograd import Function
import torch.nn.functional as F

# Import stream compact Triton kernels
from kernels.up_proj.triton_cast_kernel_stream_compact import (
    fused_up_proj_gate_activation_sparse_triton_stream_compact as _up_sparse_stream_compact,
)
from kernels.down_proj.triton_cast_kernel_down_proj_stream_compact import (
    fused_down_proj_sparse_triton_stream_compact as _down_sparse_stream_compact,
)
from kernels.weight_grad.triton_cast_kernel_weight_grad_stream_compact import (
    fused_weight_grad_sparse_triton_stream_compact as _wg_sparse_stream_compact,
)

# Import stream compact index creation utility
from kernels.stream_compact_index import create_stream_compact_index


__all__ = ["cast_mlp_fused_stream_compact"]


class _CastMLPFusedStreamCompactFunction(Function):
    """Fuses sparse up-projection → gate → sparse down-projection using stream compact kernels.

    Forward inputs
        x           – (B, S, H)   fp16 / bf16
        gate        – (B, S, NB) float32  (after ReLU)
        up_weight   – (H, I)  – hidden × intermediate (matches Triton helper)
        down_weight – (I, H)  – intermediate × hidden
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, gate: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor):
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

        # Handle non-contiguous weights if necessary - only call contiguous() if ALL strides > 1
        if up_weight.stride(0) > 1 and up_weight.stride(1) > 1:
            up_weight = up_weight.contiguous()
        if down_weight.stride(0) > 1 and down_weight.stride(1) > 1:
            down_weight = down_weight.contiguous()

        # Up-projection + gating (sparse)
        x_flat = x.view(B * S, H)
        # Handle non-contiguous x_flat if necessary - only call contiguous() if ALL strides > 1
        if x_flat.stride(0) > 1 and x_flat.stride(1) > 1:
            x_flat = x_flat.contiguous()
        gate_flat = gate.view(B * S, NB)
        
        # Create stream compact index mappings
        mappings = create_stream_compact_index(gate_flat)
        max_rows = mappings['max_rows']
        
        # Short-circuit if all gate values are zero/negative (max_rows == 0)
        if max_rows == 0:
            # Return zeros with the same shape as x
            ctx.save_for_backward()  # Save empty context for backward
            ctx.NB = NB
            ctx.LS = LS
            ctx.max_rows = max_rows
            return torch.zeros_like(x)
        
        # Check if any input requires gradients (for backward pass optimization)
        needs_backward = any(t.requires_grad for t in [x, gate, up_weight, down_weight])
        
        # Compute gated intermediate with stream compact indexing
        if needs_backward:
            inter_sparse, up_proj_sparse = _up_sparse_stream_compact(
                x_flat,
                up_weight,
                gate_flat,
                NB,
                LS,
                mappings=mappings,
                zero_init=False,
                out_dtype=x.dtype,
                save_up_proj=True,  # Cache post-ReLU, pre-gating values for backward
            )  # (act_idx, LS) - post-gating intermediate, (act_idx, LS) - post-ReLU, pre-gating
        else:
            inter_sparse = _up_sparse_stream_compact(
                x_flat,
                up_weight,
                gate_flat,
                NB,
                LS,
                mappings=mappings,
                zero_init=False,
                out_dtype=x.dtype,
                save_up_proj=False,  # No caching for inference
            )  # (act_idx, LS) - post-gating intermediate
            up_proj_sparse = None  # No cached values

        # Down-projection (sparse)
        out_flat = _down_sparse_stream_compact(
            inter_sparse,
            down_weight,
            NB,
            LS,
            mappings=mappings,
            out_dtype=x.dtype,
        )  # (BS, H)
        out = out_flat.view(B, S, H)

        # Save tensors for backward
        ctx.save_for_backward(x_flat, gate_flat, inter_sparse, up_weight, down_weight, up_proj_sparse)
        ctx.NB = NB
        ctx.LS = LS
        ctx.max_rows = max_rows
        ctx.mappings = mappings
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_flat, gate_flat, inter_sparse, up_weight, down_weight, up_proj_sparse = ctx.saved_tensors
        NB, LS, max_rows = ctx.NB, ctx.LS, ctx.max_rows
        mappings = ctx.mappings
        
        # Short-circuit if max_rows == 0 (all gate values were zero/negative)
        if max_rows == 0:
            # Return zero gradients with correct shapes
            B, S, H = grad_out.shape
            I = NB * LS  # intermediate size
            return (
                torch.zeros_like(grad_out),  # grad_x
                torch.zeros((B, S, NB), device=grad_out.device, dtype=torch.float32),  # grad_gate 
                torch.zeros((H, I), device=grad_out.device, dtype=torch.float32),  # grad_up_weight
                torch.zeros((I, H), device=grad_out.device, dtype=torch.float32),  # grad_down_weight  
            )
        
        # Assert that up_proj_sparse was cached (should always be available in backward pass)
        assert up_proj_sparse is not None, "up_proj_sparse should be cached when requires_grad=True"

        B, S, H = grad_out.shape
        BS = B * S
        total_act_idx = mappings['total_act_idx']
            
        grad_out_flat = grad_out.view(BS, H)
        # Handle non-contiguous inputs if necessary - only call contiguous() if ALL strides > 1
        if grad_out.stride(0) > 1 and grad_out.stride(1) > 1:
            grad_out = grad_out.contiguous()

        # ---------------- grad w.r.t. down_weight ----------------
        grad_down_w = _wg_sparse_stream_compact(
            inter_sparse,      # (act_idx, LS)
            grad_out_flat,     # (BS, H)
            NB,
            LS,
            mappings=mappings,
        )  # (I, H)

        # ---------------- grad_inter_sparse = grad_out · W_downᵀ + grad w.r.t. gate/up_proj ----------------
        down_w_transposed = down_weight.t()  # (H, I)
        
        # Allocate output buffers for gradients
        grad_gate_flat = torch.empty((BS, NB), device=grad_out_flat.device, dtype=torch.float32)
        grad_up_proj_sparse = torch.empty((total_act_idx, LS), device=grad_out_flat.device, dtype=x_flat.dtype)
        
        # Compute grad_inter_sparse AND gate/up_proj gradients in single kernel call
        grad_inter_sparse = _up_sparse_stream_compact(
            grad_out_flat,
            down_w_transposed,
            gate_flat,
            NB,
            LS,
            mappings=mappings,
            out_dtype=grad_out.dtype,
            apply_gate=False,
            apply_relu=False,
            calculate_grad_gate_up_proj=True,
            up_proj_cached=up_proj_sparse,
            grad_gate_output=grad_gate_flat,
            grad_up_proj_output=grad_up_proj_sparse,
        )  # (act_idx, LS)
        
        grad_gate = grad_gate_flat.view(B, S, NB)

        # grad up weight
        grad_up_w_T = _wg_sparse_stream_compact(
            grad_up_proj_sparse,  # (act_idx, LS)
            x_flat,               # (BS, H)
            NB,
            LS,
            mappings=mappings,
        )
        grad_up_w = grad_up_w_T.t()

        grad_x_flat = _down_sparse_stream_compact(
            grad_up_proj_sparse,  # (act_idx, LS)
            up_weight.t(),        # (I, H)
            NB,
            LS,
            mappings=mappings,
            out_dtype=grad_out.dtype,
        )
        grad_x = grad_x_flat.view(B, S, H)

        return grad_x, grad_gate, grad_up_w, grad_down_w


def cast_mlp_fused_stream_compact(
    x: torch.Tensor,
    gate: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper around the stream compact autograd Function (Triton backend)."""
    return _CastMLPFusedStreamCompactFunction.apply(x, gate, up_weight, down_weight)