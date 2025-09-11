from typing import Tuple
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

def pytorch_mlp_op(
    x: torch.Tensor,
    l2_gate_proj,  # nn.Linear layer
    up_proj,       # nn.Linear layer  
    down_proj,     # nn.Linear layer
    l2_num_blocks: int,
    l2_line_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original PyTorch MLP implementation"""
    
    def gate_activation(x, g, num_blocks, line_size):
        x = rearrange(x, 'b l (nb ls) -> b l nb ls', nb=num_blocks, ls=line_size)
        x = einsum(x, g, 'b l nb ls, b l nb -> b l nb ls')
        x = rearrange(x, 'b l nb ls -> b l (nb ls)')
        return x
    
    up_proj_out = F.relu(up_proj(x))
    l2_gate = F.relu(l2_gate_proj(x))
    intermediate = gate_activation(up_proj_out, l2_gate, l2_num_blocks, l2_line_size)
    down_proj_out = down_proj(intermediate)
    l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
    l2_reg_loss = l2_gate.sum()
    return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss

def triton_sortpack_mlp_op(
    x: torch.Tensor,
    l2_gate_proj,  # nn.Linear layer
    up_proj,       # nn.Linear layer  
    down_proj,     # nn.Linear layer
    l2_num_blocks: int,
    l2_line_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton sort-pack optimized MLP implementation"""
    try:
        from cast_kernels import cast_mlp_fused
    except ImportError:
        raise ImportError("cast-kernels package not available. Install with: pip install cast-kernels")
    
    l2_gate = F.relu(l2_gate_proj(x))  # Apply ReLU for auxiliary losses
    
    # Use optimized fused kernel with TRANSPOSED weights from Linear layers
    # nn.Linear weights are (out_features, in_features), but kernels expect (in_features, out_features)
    down_proj_out = cast_mlp_fused(
        x, l2_gate, 
        up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
        down_proj.weight.t(),  # Transpose: (hidden, intermediate) -> (intermediate, hidden)
        kernel="sortpack"
    )
    
    l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
    l2_reg_loss = l2_gate.sum()
    return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss

def triton_stream_compact_mlp_op(
    x: torch.Tensor,
    l2_gate_proj,  # nn.Linear layer
    up_proj,       # nn.Linear layer  
    down_proj,     # nn.Linear layer
    l2_num_blocks: int,
    l2_line_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton stream compact optimized MLP implementation"""
    try:
        from cast_kernels import cast_mlp_fused_stream_compact
    except ImportError:
        raise ImportError("cast-kernels package not available. Install with: pip install cast-kernels")
    
    l2_gate = F.relu(l2_gate_proj(x))  # Apply ReLU for auxiliary losses
    
    # Use optimized stream compact kernel with TRANSPOSED weights from Linear layers
    # nn.Linear weights are (out_features, in_features), but kernels expect (in_features, out_features)
    down_proj_out = cast_mlp_fused_stream_compact(
        x, l2_gate, 
        up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
        down_proj.weight.t()   # Transpose: (hidden, intermediate) -> (intermediate, hidden)
    )
    
    l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
    l2_reg_loss = l2_gate.sum()
    return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss

# Operation registry
MLP_OPS = {
    "pytorch": pytorch_mlp_op,
    "triton_sortpack": triton_sortpack_mlp_op,
    "triton_stream_compact": triton_stream_compact_mlp_op,
}

def get_mlp_op(implementation: str):
    """Get the MLP operation function for the given implementation"""
    import os
    
    # Check environment variable for implementation override
    env_implementation = os.getenv("CAST_MLP_IMPLEMENTATION")
    if env_implementation:
        implementation = env_implementation.lower()
    else:
        implementation = implementation.lower()
    
    if implementation not in MLP_OPS:
        raise ValueError(f"Unknown MLP implementation: {implementation}. Available: {list(MLP_OPS.keys())}")
    
    # Try to get the operation, fallback to pytorch if failed
    try:
        return MLP_OPS[implementation]
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to load optimized MLP implementation '{implementation}': {e}. Falling back to PyTorch implementation.")
        return MLP_OPS["pytorch"]
