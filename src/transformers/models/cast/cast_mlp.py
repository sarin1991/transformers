import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum
import os
import math
from .configuration_cast import CastConfig
import threading

GLOBAL_LOCK = threading.Lock()


MAX_NUM_SAMPLES = int(os.getenv("CAST_MLP_MAX_NUMBER_SAMPLES", 262144))

SHRINK_WINDOW = 1000  # Number of steps to look back for usage
SHRINK_FACTOR = 0.7  # When to consider shrinking, e.g., consistently using less than 70% of alloc'd
shrink_history = []

MAX_ALLOCATED_ROWS = MAX_NUM_SAMPLES

def choose_chunk_size_from_gates(l2_gate: torch.Tensor):
    """
    Choose chunk_size as a power of 2, starting from the smallest power of 2
    >= dense_min_chunk, and grow while the max active rows per chunk stays
    <= MAX_NUM_SAMPLES.

    Returns:
        best_chunk_size, best_max_rows
    """
    # Flatten batch/sequence into a single token dimension: (T, num_blocks)
    gate_flat = l2_gate.reshape(-1, l2_gate.shape[-1])
    num_tokens, num_blocks = gate_flat.shape

    if num_tokens == 0:
        return 0, 0

    # Active blocks per token (rows along dim 0)
    active_blocks_per_token = (gate_flat > 0).sum(dim=1)  # shape: (T,)

    # Dense-safe minimum: assume all blocks active for all tokens in a chunk
    dense_min_chunk = math.ceil(MAX_NUM_SAMPLES / num_blocks)

    min_exp = math.ceil(math.log2(dense_min_chunk))

    # Largest power of 2 <= num_tokens
    max_exp = int(math.floor(math.log2(num_tokens)))

    best_chunk_size = None
    best_max_rows = None

    for exp in range(min_exp, max_exp + 1):
        candidate_chunk_size = 1 << exp  # 2**exp

        # Split into chunks of size candidate_chunk_size (last one may be shorter)
        splits = torch.split(
            active_blocks_per_token,
            split_size_or_sections=candidate_chunk_size
        )

        # Total active rows in each chunk = sum of active blocks per token
        max_active = torch.stack([split.sum() for split in splits]).max().item()

        if max_active <= MAX_NUM_SAMPLES:
            best_chunk_size = candidate_chunk_size
            best_max_rows = max_active
        else:
            # As chunk size grows, rows per chunk can only increase, so stop
            break

    if best_chunk_size is None:
        raise ValueError("No valid chunk_size found under MAX_NUM_SAMPLES.")

    return best_chunk_size, best_max_rows

def calc_rows_to_allocate(max_rows):
    global MAX_ALLOCATED_ROWS
    with GLOBAL_LOCK:
        if max_rows>MAX_ALLOCATED_ROWS:
            MAX_ALLOCATED_ROWS = max_rows
            shrink_history.clear()
        else:
            shrink_history.append(max_rows)
        if len(shrink_history)>=SHRINK_WINDOW:
            max_rows_shrink_window = max(shrink_history)
            if max_rows_shrink_window < SHRINK_FACTOR*MAX_ALLOCATED_ROWS:
                MAX_ALLOCATED_ROWS = max_rows_shrink_window
            shrink_history.clear()
    return MAX_ALLOCATED_ROWS


class CastMLPDense(nn.Module):
    """Standard Dense PyTorch MLP implementation"""
    
    def __init__(self, config: CastConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        intermediate = F.relu(self.up_proj(x))
        down_proj_out = self.down_proj(intermediate)
        l2_act_ratio = 0
        l2_reg_loss = 0
        l2_gate = None
        return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss


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
        self.l2_gate_proj = nn.Linear(self.hidden_size, self.l2_num_blocks, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        try:
            from cast_kernels import cast_mlp_fused_stream_compact, cast_mlp_fused_stream_compact_chunked
        except ImportError:
            raise ImportError("cast-kernels package not available. Install with: pip install cast-kernels")
        l2_gate = F.relu(self.l2_gate_proj(x))  # Apply ReLU for auxiliary losses
        l2_act_ratio = (l2_gate > 0).mean(dtype=torch.float32)
        l2_reg_loss = l2_gate.sum()
        batch_seq_size = math.prod(l2_gate.shape[:-1])
        num_blocks = l2_gate.shape[-1]
        max_intermediate_rows = math.ceil(l2_act_ratio * batch_seq_size * num_blocks)
        if max_intermediate_rows > MAX_NUM_SAMPLES:
            chunk_size, max_rows = choose_chunk_size_from_gates(l2_gate)
            rows_to_allocate = calc_rows_to_allocate(max_rows)
            down_proj_out = cast_mlp_fused_stream_compact_chunked(
                x, l2_gate, 
                self.up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
                self.down_proj.weight.t(),   # Transpose: (hidden, intermediate) -> (intermediate, hidden)
                rows_to_allocate,
                chunk_size,
            )
        else:
            rows_to_allocate = calc_rows_to_allocate(max_intermediate_rows)
            down_proj_out = cast_mlp_fused_stream_compact(
                x, l2_gate, 
                self.up_proj.weight.t(),    # Transpose: (intermediate, hidden) -> (hidden, intermediate)
                self.down_proj.weight.t(),   # Transpose: (hidden, intermediate) -> (intermediate, hidden)
                rows_to_allocate
            )
        return down_proj_out, l2_gate, l2_act_ratio, l2_reg_loss


# Registry mapping implementation names to classes
CAST_MLP_CLASSES = {
    "dense": CastMLPDense,
    "pytorch": CastMLPPyTorch,
    "triton_sortpack": CastMLPTritonSortPack,
    "triton_stream_compact": CastMLPTritonStreamCompact,
}


def get_mlp_class(config: CastConfig):
    """Get the MLP class for the given implementation configuration"""
    import os
    import warnings
    
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

