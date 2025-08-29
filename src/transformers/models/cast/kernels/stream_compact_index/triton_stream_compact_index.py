import torch
import triton
import triton.language as tl


def get_triton_autotune_config():
    """
    Autotune configurations for the index mapping kernel.
    Focus on different block sizes and warp configurations for the 1D processing pattern.
    """
    return [
        # Small block sizes for small problems
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        
        # Medium block sizes 
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        
        # Large block sizes for big problems
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
        
        # Alternative warp configurations
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=1),
    ]


@triton.autotune(
    configs=get_triton_autotune_config(),
    key=['BS', 'NB'],
)
@triton.jit
def stream_compact_index_kernel(
    # Inputs
    gate_ptr,                # (BS, NB) float32 - original gate values
    gate_mask_ptr,           # (BS, NB) bool - where gate > 0
    cumsum_ptr,              # (BS, NB) int32 - cumulative sum along dim=0
    
    # Outputs
    nb_maxrows_to_bs_ptr,    # (NB, max_rows) -> BS index
    nb_maxrows_to_actidx_ptr,# (NB, max_rows) -> sequential act_idx
    nb_maxrows_gate_vals_ptr,# (NB, max_rows) -> gate values
    
    # Sequential indexing inputs
    cumsum_nb_ptr,           # (BS, NB) int32 - cumulative sum along dim=1  
    bs_start_indices_ptr,    # (BS,) int32 - start act_idx for each BS row
    
    # Dimensions
    BS: tl.constexpr, 
    NB: tl.constexpr, 
    max_rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch-process multiple (bs, nb) pairs per thread for efficiency.
    Each thread processes BLOCK_SIZE consecutive elements.
    
    Creates three mapping tables:
    1. (NB, max_rows) -> BS: For loading input data in up-projection
    2. (NB, max_rows) -> act_idx: For storing to dense intermediate format
    3. (BS, NB) -> act_idx: For accumulating down-projection results
    
    Grid: (triton.cdiv(BS * NB, BLOCK_SIZE),)
    """
    
    pid = tl.program_id(0)
    
    # Calculate starting index for this thread
    start_idx = pid * BLOCK_SIZE
    
    # Process BLOCK_SIZE elements
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    total_elements = BS * NB
    mask = offsets < total_elements
    
    # Decompose linear indices to (bs, nb) coordinates
    bs_indices = offsets // NB
    nb_indices = offsets % NB
    
    # Load gate mask values for this batch
    is_active = tl.load(gate_mask_ptr + offsets, mask=mask, other=False)
    
    # Load cumsum values for active elements (local row indices within block)
    cumsum_vals = tl.load(cumsum_ptr + offsets, mask=mask & is_active, other=0)
    local_row_indices = cumsum_vals - 1  # Convert to 0-indexed
    
    # Load original gate values for active elements
    gate_vals_raw = tl.load(gate_ptr + offsets, mask=mask & is_active, other=0.0)
    gate_vals = tl.where(gate_vals_raw > 0, gate_vals_raw, 0.0)
    
    # Load sequential cumsum values for act_idx assignment
    cumsum_nb_vals = tl.load(cumsum_nb_ptr + offsets, mask=mask & is_active, other=0)
    bs_start_offsets = tl.load(bs_start_indices_ptr + bs_indices, mask=mask & is_active, other=0)
    sequential_act_indices = bs_start_offsets + (cumsum_nb_vals - 1)
    
    # Calculate offsets into (NB, max_rows) tensors
    nb_maxrows_offsets = nb_indices * max_rows + local_row_indices
    
    # Store mappings for active elements
    active_mask = mask & is_active
    
    # Store (NB, max_rows) -> BS mapping
    tl.store(nb_maxrows_to_bs_ptr + nb_maxrows_offsets, bs_indices, mask=active_mask)
    
    # Store (NB, max_rows) -> sequential act_idx mapping
    tl.store(nb_maxrows_to_actidx_ptr + nb_maxrows_offsets, sequential_act_indices, mask=active_mask)
    
    # Store (NB, max_rows) -> gate values mapping
    tl.store(nb_maxrows_gate_vals_ptr + nb_maxrows_offsets, gate_vals, mask=active_mask)


def create_stream_compact_index(gate: torch.Tensor):
    """
    Create all stream compaction index mapping tables using unified kernel.
    
    Args:
        gate: (BS, NB) tensor of gate values
        
    Returns:
        dict containing:
        - nb_maxrows_to_bs: (NB, max_rows) -> global row index mapping
        - nb_maxrows_to_actidx: (NB, max_rows) -> sequential act_idx mapping  
        - nb_maxrows_gate_vals: (NB, max_rows) -> gate values mapping
        - max_rows: maximum active rows across all blocks
        - total_act_idx: total number of active elements
        - max_rows_per_block: (NB,) active rows per block
        - bs_start_indices: (BS,) start act_idx for each BS row
        - bs_counts: (BS,) count of active elements per BS row
    """
    
    BS, NB = gate.shape
    
    # ===== Preprocessing =====
    # Create boolean mask for active positions
    mask = gate > 0  # (BS, NB)
    
    # Dual cumulative sums
    cumsum_bs = torch.cumsum(mask, dim=0, dtype=torch.int32)  # (BS, NB) - for NB-based kernels
    cumsum_nb = torch.cumsum(mask, dim=1, dtype=torch.int32)  # (BS, NB) - for sequential act_idx
    
    # Count active rows per block and totals
    max_rows_per_block = cumsum_bs[-1, :]  # (NB,) - final cumsum values
    max_rows = max_rows_per_block.max().item()
    total_act_idx = max_rows_per_block.sum().item()
    
    # Sequential indexing support
    bs_counts = cumsum_nb[:, -1].contiguous()  # (BS,) - active count per BS row
    bs_start_indices = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=gate.device),
        torch.cumsum(bs_counts[:-1], dim=0)
    ])  # (BS,) - start index for each BS row
    
    # Early exit if no active elements
    if max_rows == 0:
        return {
            'nb_maxrows_to_bs': torch.empty((NB, 0), dtype=torch.int32, device=gate.device),
            'nb_maxrows_to_actidx': torch.empty((NB, 0), dtype=torch.int32, device=gate.device),
            'nb_maxrows_gate_vals': torch.empty((NB, 0), dtype=torch.float32, device=gate.device),
            'bs_nb_to_gate_vals': gate,
            'max_rows': 0,
            'total_act_idx': 0,
            'max_rows_per_block': max_rows_per_block,
            'bs_start_indices': bs_start_indices,
            'bs_counts': bs_counts,
        }
    
    # ===== Allocate Output Tensors =====
    # Use empty for speed, will be filled by kernel
    nb_maxrows_to_bs = torch.empty((NB, max_rows), dtype=torch.int32, device=gate.device)
    nb_maxrows_to_actidx = torch.empty((NB, max_rows), dtype=torch.int32, device=gate.device)
    # Gate values need zeros initialization since kernel doesn't write to all positions
    nb_maxrows_gate_vals = torch.zeros((NB, max_rows), dtype=torch.float32, device=gate.device)
    
    # ===== Launch Autotune Kernel =====
    total_elements = BS * NB
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    stream_compact_index_kernel[grid](
        gate,                    # gate_ptr
        mask,                    # gate_mask_ptr
        cumsum_bs,               # cumsum_ptr  
        nb_maxrows_to_bs,        # nb_maxrows_to_bs_ptr
        nb_maxrows_to_actidx,    # nb_maxrows_to_actidx_ptr
        nb_maxrows_gate_vals,    # nb_maxrows_gate_vals_ptr
        cumsum_nb,               # cumsum_nb_ptr
        bs_start_indices,        # bs_start_indices_ptr
        BS=BS,
        NB=NB, 
        max_rows=max_rows,
    )
    
    return {
        'nb_maxrows_to_bs': nb_maxrows_to_bs,
        'nb_maxrows_to_actidx': nb_maxrows_to_actidx,
        'nb_maxrows_gate_vals': nb_maxrows_gate_vals,
        'bs_nb_to_gate_vals': gate,
        'max_rows': max_rows,
        'total_act_idx': total_act_idx,
        'max_rows_per_block': max_rows_per_block,
        'bs_start_indices': bs_start_indices,
        'bs_counts': bs_counts,
    }


if __name__ == "__main__":
    """Basic test of the index mapping kernel"""
    
    print("✅ Stream compact index kernel loaded successfully")
    
    # Simple test with random data
    BS, NB = 8, 4
    gate = torch.rand(BS, NB, device="cuda", dtype=torch.float32)
    # Make it sparse (60% zeros)
    mask = torch.rand_like(gate) < 0.6
    gate[mask] = 0.0
    
    print(f"Test gate shape: {gate.shape}")
    print(f"Sparsity: {(gate == 0).float().mean().item():.1%}")
    
    # Create mappings
    mappings = create_stream_compact_index(gate)
    
    print(f"Max rows: {mappings['max_rows']}")
    print(f"Total act_idx: {mappings['total_act_idx']}")
    print("✅ Basic test completed successfully!")