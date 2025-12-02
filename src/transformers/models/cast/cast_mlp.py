import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum
import os
import math
from .configuration_cast import CastConfig


MAX_NUM_SAMPLES = int(os.getenv("CAST_MLP_MAX_NUMBER_SAMPLES", 32768))

class CastMLPPyTorch(nn.Module):
    """Standard PyTorch MLP implementation with einops-based gate activation"""
    
    def __init__(self, config: CastConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.l2_line_size = config.l2_line_size
        self.l2_num_blocks = self.intermediate_size // self.l2_line_size
        self.l2_gate_proj = nn.Linear(self.hidden_size, self.l2_num_blocks, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def gate_activation(self, x, g, num_blocks, line_size):
        """Gate activation using einops"""
        x = rearrange(x, 'b l (nb ls) -> b l nb ls', nb=num_blocks, ls=line_size)
        x = einsum(x, g, 'b l nb ls, b l nb -> b l nb ls')
        x = rearrange(x, 'b l nb ls -> b l (nb ls)')
        return x
    
    def forward(self, x):
        up_proj_out = F.relu(self.up_proj(x))
        l2_gate = F.relu(self.l2_gate_proj(x))
        intermediate = self.gate_activation(up_proj_out, l2_gate, self.l2_num_blocks, self.l2_line_size)
        down_proj_out = self.down_proj(intermediate)
        l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
        l2_reg_loss = l2_gate.sum()
        return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss


class CastMLPTritonSortPack(nn.Module):
    """Triton sort-pack optimized MLP implementation"""
    
    def __init__(self, config: CastConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.l2_line_size = config.l2_line_size
        self.l2_num_blocks = self.intermediate_size // self.l2_line_size
        self.l2_gate_proj = nn.Linear(self.hidden_size, self.l2_num_blocks, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        try:
            from cast_kernels import cast_mlp_fused
        except ImportError:
            raise ImportError("cast-kernels package not available. Install with: pip install cast-kernels")
        
        l2_gate = F.relu(self.l2_gate_proj(x)).to(torch.float32)  # Apply ReLU for auxiliary losses
        
        # Use optimized fused kernel with TRANSPOSED weights from Linear layers
        # nn.Linear weights are (out_features, in_features), but kernels expect (in_features, out_features)
        down_proj_out = cast_mlp_fused(
            x, l2_gate, 
            self.up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
            self.down_proj.weight.t(),    # Transpose: (hidden, intermediate) -> (intermediate, hidden)
            kernel="sortpack"
        )
        
        l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
        l2_reg_loss = l2_gate.sum()
        return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss


class CastMLPTritonStreamCompact(nn.Module):
    """Triton stream compact optimized MLP implementation"""
    
    def __init__(self, config: CastConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.l2_line_size = config.l2_line_size
        self.l2_num_blocks = self.intermediate_size // self.l2_line_size
        self.l2_gate_proj = nn.Linear(self.hidden_size, self.l2_num_blocks, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        try:
            from cast_kernels import cast_mlp_fused_stream_compact
        except ImportError:
            raise ImportError("cast-kernels package not available. Install with: pip install cast-kernels")
        
        l2_gate = F.relu(self.l2_gate_proj(x)).to(torch.float32)  # Apply ReLU for auxiliary losses
        l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
        l2_reg_loss = l2_gate.sum()
        batch_seq_size = math.product(l2_gate.shape[:-1])
        num_samples = math.ceil(l2_act_ratio * batch_seq_size)
        num_chunks = math.ceil(num_samples / MAX_NUM_SAMPLES)
        chunk_size = math.ceil(batch_seq_size / num_chunks)

        if num_chunks > 1:
            down_proj_out_list = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, batch_seq_size)
                x_chunk = x[start_idx:end_idx]
                l2_gate_chunk = l2_gate[start_idx:end_idx]
                down_proj_out_chunk = cast_mlp_fused_stream_compact(
                    x_chunk, l2_gate_chunk, 
                    self.up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
                    self.down_proj.weight.t()   # Transpose: (hidden, intermediate) -> (intermediate, hidden)
                )
                down_proj_out_list.append(down_proj_out_chunk)
            down_proj_out = torch.cat(down_proj_out_list, dim=0)
        else:
            down_proj_out = cast_mlp_fused_stream_compact(
                x, l2_gate, 
                self.up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
                self.down_proj.weight.t()   # Transpose: (hidden, intermediate) -> (intermediate, hidden)
            )
        return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss


# Registry mapping implementation names to classes
CAST_MLP_CLASSES = {
    "pytorch": CastMLPPyTorch,
    "triton_sortpack": CastMLPTritonSortPack,
    "triton_stream_compact": CastMLPTritonStreamCompact,
}


def get_mlp_class(config: CastConfig):
    """Get the MLP class for the given implementation configuration"""
    import os
    import warnings
    
    # Check environment variable for implementation override
    env_implementation = os.getenv("CAST_MLP_IMPLEMENTATION")
    if env_implementation:
        implementation = env_implementation.lower()
    else:
        implementation = config.mlp_implementation.lower()
    
    if implementation not in CAST_MLP_CLASSES:
        raise ValueError(f"Unknown MLP implementation: {implementation}. Available: {list(CAST_MLP_CLASSES.keys())}")
    
    # Try to get the class, fallback to pytorch if Triton kernels unavailable
    mlp_class = CAST_MLP_CLASSES[implementation]
    
    # For Triton implementations, check if kernels are available at import time
    if implementation.startswith("triton"):
        try:
            if implementation == "triton_sortpack":
                from cast_kernels import cast_mlp_fused  # noqa: F401
            elif implementation == "triton_stream_compact":
                from cast_kernels import cast_mlp_fused_stream_compact  # noqa: F401
        except ImportError:
            warnings.warn(
                f"Failed to load optimized MLP implementation '{implementation}'. "
                "Falling back to PyTorch implementation."
            )
            return CAST_MLP_CLASSES["pytorch"]
    
    return mlp_class

