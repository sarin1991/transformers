import torch
import triton
import triton.language as tl
from .triton_stream_compact_index import create_stream_compact_index


# Global cache for shared memory info
_gpu_shared_memory_cache = {}

def get_gpu_shared_memory_size(device):
    """Query actual GPU shared memory size using CUDA driver"""
    # Convert to device index if tensor device passed
    device_idx = device.index if hasattr(device, 'index') else device
    
    # Check cache first
    if device_idx in _gpu_shared_memory_cache:
        return _gpu_shared_memory_cache[device_idx]
    
    # Use Triton's CUDA driver to get actual device properties
    import triton.runtime.driver as driver
    if not (hasattr(driver, 'active') and hasattr(driver.active, 'utils')):
        raise RuntimeError("Triton CUDA driver not available - cannot query actual shared memory size")
    
    # Get device properties from CUDA driver directly
    props_dict = driver.active.utils.get_device_properties(device_idx)
    if 'max_shared_memory_per_block' not in props_dict:
        raise RuntimeError(f"Could not get max_shared_memory_per_block for device {device_idx}")
    
    shared_mem = props_dict['max_shared_memory_per_block']
    _gpu_shared_memory_cache[device_idx] = shared_mem
    return shared_mem

def should_use_fused_kernel(BS, NB, device):
    """
    Dynamic threshold check based on actual GPU shared memory and kernel requirements.
    
    Kernel memory usage analysis:
    - bs_range: BS * 4 bytes (int32)  
    - nb_range: NB * 4 bytes (int32)
    - gate_vals: BS * NB * 4 bytes (float32, worst case all active)
    - bs_start_indices: BS * 4 bytes (int32)
    - Total: ~(BS * NB * 4) + (BS + NB) * 8 bytes
    
    Uses 25% of available shared memory for safety margin.
    """
    shared_mem = get_gpu_shared_memory_size(device)
    usable_mem = int(shared_mem * 0.25)  # 25% threshold, standard practice
    
    # Calculate actual memory requirements for this kernel
    estimated_usage = (BS * NB * 4) + ((BS + NB) * 8)  # Main arrays + indexing
    
    use_fused = estimated_usage <= usable_mem
    
    print(f"  Threshold check: {BS}×{NB} kernel")
    print(f"  Estimated usage: {estimated_usage} bytes ({estimated_usage//1024}KB)")  
    print(f"  GPU shared mem: {shared_mem} bytes ({shared_mem//1024}KB)")
    print(f"  Usable (25%): {usable_mem} bytes ({usable_mem//1024}KB)")
    print(f"  Decision: {'✅ FUSED' if use_fused else '❌ FALLBACK'}")
    
    return use_fused

@triton.jit
def fused_stream_compact_index_kernel_stage1(
    # Inputs
    gate_ptr,                    # (BS, NB) float32 - gate values
    gate_stride_bs,              # stride for BS dimension
    gate_stride_nb,              # stride for NB dimension
    
    # Outputs
    bs_start_indices_ptr,        # (BS,) -> start act_idx for each BS row
    bs_counts_ptr,              # (BS,) -> count of active elements per BS row
    max_rows_per_block_ptr,     # (NB,) -> active rows per block
    max_rows_ptr,               # (1,) -> max active rows
    total_act_idx_ptr,          # (1,) -> total active elements

    # Dimensions
    BS: tl.constexpr,
    NB: tl.constexpr,
):
    """
    Fused kernel that computes cumsum in shared memory and generates all mappings.
    Uses vectorized Triton operations for optimal performance on small BS×NB cases.
    """
    
    # Create index grids
    bs_range = tl.arange(0, BS)  # (BS,)
    nb_range = tl.arange(0, NB)  # (NB,)
    
    # Load entire gate matrix using strides
    gate_offsets = bs_range[:, None] * gate_stride_bs + nb_range[None, :] * gate_stride_nb
    gate_vals = tl.load(gate_ptr + gate_offsets)  # (BS, NB)
    
    # Create mask for active elements
    mask = gate_vals > 0  # (BS, NB)
    
    # Sum along BS dimension (dim=0) for each NB column
    sum_bs = tl.sum(mask, axis=0)  # (NB,)
    
    # Sum along NB dimension (dim=1) for each BS row  
    sum_nb = tl.sum(mask, axis=1)  # (BS,)
    
    # Store max_rows_per_block: last row of cumsum_bs
    tl.store(max_rows_per_block_ptr + nb_range, sum_bs)
    
    # Store bs_counts: last column of cumsum_nb
    tl.store(bs_counts_ptr + bs_range, sum_nb)
    
    # Compute bs_start_indices (prefix sum of bs_counts)
    bs_starts_idx = tl.cumsum(sum_nb, axis=0) # (BS,)
    tl.store(bs_start_indices_ptr + bs_range, tl.zeros_like(bs_starts_idx), mask=bs_range==0) # 0 in first row
    tl.store(bs_start_indices_ptr + bs_range + 1, bs_starts_idx, mask=bs_range<BS-1) # setting other rows

    # Save max rows
    tl.store(max_rows_ptr, tl.max(sum_bs))

    # Save total act_idx
    tl.store(total_act_idx_ptr, tl.sum(sum_bs))

@triton.jit
def fused_stream_compact_index_kernel_stage2(
    # Inputs
    gate_ptr,                    # (BS, NB) float32 - gate values
    gate_stride_bs,              # stride for BS dimension
    gate_stride_nb,              # stride for NB dimension
    bs_start_indices_ptr,        # (BS,) -> start act_idx for each BS row
    
    # Outputs
    nb_maxrows_to_bs_ptr,        # (NB, max_rows) -> BS index
    nb_maxrows_to_actidx_ptr,    # (NB, max_rows) -> sequential act_idx
    nb_maxrows_gate_vals_ptr,    # (NB, max_rows) -> gate values
        
    # Dimensions
    BS: tl.constexpr,
    NB: tl.constexpr,
    max_rows: tl.constexpr,
):
    """
    Fused kernel that computes cumsum in shared memory and generates all mappings.
    Uses vectorized Triton operations for optimal performance on small BS×NB cases.
    """
    
    # Create index grids
    bs_range = tl.arange(0, BS)  # (BS,)
    nb_range = tl.arange(0, NB)  # (NB,)
    
    # Load entire gate matrix using strides
    gate_offsets = bs_range[:, None] * gate_stride_bs + nb_range[None, :] * gate_stride_nb
    gate_vals = tl.load(gate_ptr + gate_offsets)  # (BS, NB)

    # Load bs_start_indices
    bs_start_indices = tl.load(bs_start_indices_ptr + bs_range)  # (BS,)
    
    # Create mask for active elements
    is_active = gate_vals > 0  # (BS, NB)
    gate_vals = tl.where(gate_vals > 0, gate_vals, 0.0)
    
    # Cumsum along BS dimension (dim=0) for each NB column
    cumsum_bs = tl.cumsum(is_active, axis=0)  # (BS, NB)
    
    # Cumsum along NB dimension (dim=1) for each BS row  
    cumsum_nb = tl.cumsum(is_active, axis=1)  # (BS, NB)

    # Output offsets for (NB, max_rows) layout
    nb_max_row_output_offsets = nb_range[None, :] * max_rows + cumsum_bs - 1

    # Compute mapping indices
    sequential_act_indices = bs_start_indices[:, None] + (cumsum_nb - 1)  # Sequential act_idx
    
    # Store mappings only for active elements using mask
    tl.store(nb_maxrows_to_bs_ptr + nb_max_row_output_offsets, bs_range[:, None], mask=is_active)
    tl.store(nb_maxrows_to_actidx_ptr + nb_max_row_output_offsets, sequential_act_indices, mask=is_active)
    tl.store(nb_maxrows_gate_vals_ptr + nb_max_row_output_offsets, gate_vals)

def create_stream_compact_index_fused(gate: torch.Tensor):
    """
    Fused version for small BS×NB cases that fits in shared memory.
    Uses single kernel to eliminate torch.cumsum overhead.
    """
    BS, NB = gate.shape

    max_rows_ptr = torch.empty(1, dtype=torch.int32, device=gate.device)
    total_act_idx_ptr = torch.empty(1, dtype=torch.int32, device=gate.device)
    bs_start_indices = torch.empty(BS, dtype=torch.int32, device=gate.device)
    bs_counts = torch.empty(BS, dtype=torch.int32, device=gate.device)
    max_rows_per_block = torch.empty(NB, dtype=torch.int32, device=gate.device)

    grid = (1,)

    fused_stream_compact_index_kernel_stage1[grid](
        gate,
        gate.stride(0),
        gate.stride(1),
        bs_start_indices,
        bs_counts,
        max_rows_per_block,
        max_rows_ptr,
        total_act_idx_ptr,
        BS=BS,
        NB=NB,
    )

    max_rows = max_rows_ptr.item()
    total_act_idx = total_act_idx_ptr.item()
    # Early exit if no active elements
    if max_rows == 0:
        return {
            'nb_maxrows_to_bs': torch.empty((NB, 0), dtype=torch.int32, device=gate.device),
            'nb_maxrows_to_actidx': torch.empty((NB, 0), dtype=torch.int32, device=gate.device),
            'nb_maxrows_gate_vals': torch.empty((NB, 0), dtype=torch.float32, device=gate.device),
            'max_rows': 0,
            'total_act_idx': 0,
            'max_rows_per_block': max_rows_per_block,
            'bs_start_indices': torch.zeros(BS, dtype=torch.int32, device=gate.device),
            'bs_counts': torch.zeros(BS, dtype=torch.int32, device=gate.device),
        }
    
    # Allocate output tensors
    nb_maxrows_to_bs = torch.empty((NB, max_rows), dtype=torch.int32, device=gate.device)
    nb_maxrows_to_actidx = torch.empty((NB, max_rows), dtype=torch.int32, device=gate.device)
    nb_maxrows_gate_vals = torch.zeros((NB, max_rows), dtype=torch.float32, device=gate.device)

    grid = (1,)

    fused_stream_compact_index_kernel_stage2[grid](
        gate,
        gate.stride(0),
        gate.stride(1),
        bs_start_indices,
        nb_maxrows_to_bs,
        nb_maxrows_to_actidx,
        nb_maxrows_gate_vals,
        BS=BS,
        NB=NB,
        max_rows=max_rows,
    )
    
    return {
        'nb_maxrows_to_bs': nb_maxrows_to_bs,
        'nb_maxrows_to_actidx': nb_maxrows_to_actidx,
        'nb_maxrows_gate_vals': nb_maxrows_gate_vals,
        'max_rows': max_rows,
        'total_act_idx': total_act_idx,
        'max_rows_per_block': max_rows_per_block,
        'bs_start_indices': bs_start_indices,
        'bs_counts': bs_counts,
    }


def create_stream_compact_index_adaptive(gate: torch.Tensor):
    """
    Adaptive dispatcher that chooses between fused and original implementation
    based on problem size and actual GPU shared memory constraints.
    """
    BS, NB = gate.shape
    
    if should_use_fused_kernel(BS, NB, gate.device):
        return create_stream_compact_index_fused(gate)
    else:
        # Fallback to original CUB-based implementation
        return create_stream_compact_index(gate)


if __name__ == "__main__":
    """Test the fused kernel implementation"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        exit(1)
        
    try:
        import triton
        print("✅ Triton is available")
    except ImportError:
        print("❌ Triton not available")
        exit(1)
    
    # Test 1bs×1seq×64NB case (decoding scenario)
    BS, NB = 1, 64
    gate = torch.rand(BS, NB, device="cuda", dtype=torch.float32)
    # Make sparse (simulate typical gate activation)
    mask = torch.rand_like(gate) < 0.7
    gate[mask] = 0.0
    
    print(f"Testing fused kernel: BS={BS}, NB={NB}")
    print(f"Sparsity: {(gate == 0).float().mean().item():.1%}")
    
    # Test fused version
    mappings_fused = create_stream_compact_index_fused(gate)
    print(f"Fused - Max rows: {mappings_fused['max_rows']}, Total act_idx: {mappings_fused['total_act_idx']}")
    
    # Test original for comparison
    mappings_orig = create_stream_compact_index(gate)
    print(f"Original - Max rows: {mappings_orig['max_rows']}, Total act_idx: {mappings_orig['total_act_idx']}")
    
    # Verify results match
    for key in ['max_rows', 'total_act_idx']:
        if mappings_fused[key] != mappings_orig[key]:
            print(f"❌ Mismatch in {key}: fused={mappings_fused[key]}, orig={mappings_orig[key]}")
        else:
            print(f"✅ {key} matches")
    
    print("✅ Fused kernel test completed!")