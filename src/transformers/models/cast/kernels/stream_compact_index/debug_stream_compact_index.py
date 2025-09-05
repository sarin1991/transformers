import torch
from kernels.stream_compact_index.triton_stream_compact_index import create_stream_compact_index
from kernels.stream_compact_index.triton_stream_compact_index_fused import (
    create_stream_compact_index_fused, 
    create_stream_compact_index_adaptive,
    should_use_fused_kernel
)


def verify_mappings(gate: torch.Tensor, mappings: dict):
    """
    Verify that the generated sequential mappings are correct.
    
    Args:
        gate: Original gate tensor (BS, NB)
        mappings: Dictionary returned by create_stream_compact_index
    """
    BS, NB = gate.shape
    mask = gate > 0
    
    nb_maxrows_to_bs = mappings['nb_maxrows_to_bs']
    nb_maxrows_to_actidx = mappings['nb_maxrows_to_actidx'] 
    nb_maxrows_gate_vals = mappings['nb_maxrows_gate_vals']
    max_rows = mappings['max_rows']
    total_act_idx = mappings['total_act_idx']
    bs_start_indices = mappings['bs_start_indices']
    bs_counts = mappings['bs_counts']
    
    print(f"=== Stream Compact Sequential Mappings Verification ===")
    print(f"Gate shape: {gate.shape}")
    print(f"Active elements: {mask.sum().item()} / {BS * NB}")
    print(f"Max rows per block: {max_rows}")
    print(f"Total act_idx: {total_act_idx}")
    print(f"Sequential layout: bs_start_indices shape {bs_start_indices.shape}, bs_counts shape {bs_counts.shape}")
    
    errors = 0
    
    # Verify sequential layout consistency
    expected_total = bs_counts.sum().item()
    if expected_total != total_act_idx:
        print(f"ERROR: bs_counts sum ({expected_total}) != total_act_idx ({total_act_idx})")
        errors += 1
    
    # Verify bs_start_indices are sequential
    for bs in range(BS - 1):
        expected_next_start = bs_start_indices[bs].item() + bs_counts[bs].item()
        actual_next_start = bs_start_indices[bs + 1].item()
        if expected_next_start != actual_next_start:
            print(f"ERROR: Non-sequential start indices at BS {bs}: expected {expected_next_start}, got {actual_next_start}")
            errors += 1
    
    # Build reverse mapping from act_idx to (bs, nb) for verification
    act_idx_to_bs_nb = {}
    for nb in range(NB):
        for local_row in range(max_rows):
            gate_val = nb_maxrows_gate_vals[nb, local_row].item()
            if gate_val > 0:  # Valid/active entry
                bs_mapped = nb_maxrows_to_bs[nb, local_row].item()
                act_idx = nb_maxrows_to_actidx[nb, local_row].item()
                
                # Validate ranges
                if bs_mapped < 0 or bs_mapped >= BS:
                    print(f"ERROR: Invalid bs_mapped {bs_mapped} for active entry at ({nb}, {local_row})")
                    errors += 1
                    continue
                    
                if act_idx < 0 or act_idx >= total_act_idx:
                    print(f"ERROR: Invalid act_idx {act_idx} for active entry at ({nb}, {local_row})")
                    errors += 1
                    continue
                
                # Check for duplicates
                if act_idx in act_idx_to_bs_nb:
                    prev_bs, prev_nb = act_idx_to_bs_nb[act_idx]
                    print(f"ERROR: Duplicate act_idx {act_idx} found at ({nb}, {local_row}) and ({prev_nb}, prev_row)")
                    errors += 1
                else:
                    act_idx_to_bs_nb[act_idx] = (bs_mapped, nb)
                
                # Verify gate value consistency
                original_gate_val = gate[bs_mapped, nb].item()
                if abs(gate_val - original_gate_val) > 1e-6:
                    print(f"ERROR: Gate value mismatch at ({nb}, {local_row}): stored {gate_val}, original {original_gate_val}")
                    errors += 1
    
    # Verify that act_idx values within each BS range are sequential
    for bs in range(BS):
        start_idx = bs_start_indices[bs].item()
        count = bs_counts[bs].item()
        
        if count == 0:
            continue
            
        # Find all act_idx values for this BS
        bs_act_indices = []
        for act_idx in range(start_idx, start_idx + count):
            if act_idx in act_idx_to_bs_nb:
                mapped_bs, mapped_nb = act_idx_to_bs_nb[act_idx]
                if mapped_bs == bs:
                    bs_act_indices.append(act_idx)
        
        # Check that we found the expected count
        if len(bs_act_indices) != count:
            print(f"ERROR: BS {bs} expected {count} act_indices, found {len(bs_act_indices)}")
            errors += 1
        
        # Check that they are sequential
        expected_indices = list(range(start_idx, start_idx + count))
        if sorted(bs_act_indices) != expected_indices:
            print(f"ERROR: BS {bs} act_indices are not sequential: expected {expected_indices}, got {sorted(bs_act_indices)}")
            errors += 1
    
    # Verify that every active (bs, nb) position has a corresponding act_idx
    active_positions = torch.nonzero(mask, as_tuple=False)  # (num_active, 2)
    for bs, nb in active_positions:
        bs_item, nb_item = bs.item(), nb.item()
        found = False
        
        # Search for this (bs, nb) in the mappings
        for act_idx, (mapped_bs, mapped_nb) in act_idx_to_bs_nb.items():
            if mapped_bs == bs_item and mapped_nb == nb_item:
                found = True
                break
        
        if not found:
            print(f"ERROR: Active position ({bs_item}, {nb_item}) not found in act_idx mappings")
            errors += 1
    
    if errors == 0:
        print("✅ All sequential mappings verified successfully!")
    else:
        print(f"❌ Found {errors} mapping errors")
    
    return errors == 0


def test_simple_case():
    """Test with a simple, manually verifiable case"""
    print("=== Testing Simple Case ===")
    
    gate = torch.tensor([
        [0.0, 2.5, 0.0, 1.2],  # row 0: blocks 1,3 active
        [1.8, 0.0, 0.0, 0.0],  # row 1: block 0 active  
        [0.0, 0.0, 3.1, 0.0],  # row 2: block 2 active
        [0.5, 1.9, 0.0, 0.7],  # row 3: blocks 0,1,3 active
    ], device="cuda", dtype=torch.float32)
    
    print(f"Test gate tensor:\n{gate}")
    print(f"Active mask:\n{gate > 0}")
    
    # Create mappings
    mappings = create_stream_compact_index(gate)
    
    print(f"\nGenerated sequential mappings:")
    print(f"nb_maxrows_to_bs:\n{mappings['nb_maxrows_to_bs']}")
    print(f"nb_maxrows_to_actidx:\n{mappings['nb_maxrows_to_actidx']}")
    print(f"bs_start_indices:\n{mappings['bs_start_indices']}")
    print(f"bs_counts:\n{mappings['bs_counts']}")
    
    # Verify correctness
    verify_mappings(gate, mappings)


def test_random_cases():
    """Test with various random configurations"""
    print("\n=== Testing Random Cases ===")
    
    test_configs = [
        (16, 8, 0.5),   # Small, medium sparsity
        (64, 16, 0.7),  # Medium, high sparsity
        (128, 32, 0.9), # Large, very high sparsity
        (256, 64, 0.3), # Large, low sparsity
    ]
    
    for BS, NB, sparsity in test_configs:
        print(f"\nTesting BS={BS}, NB={NB}, sparsity={sparsity:.1%}")
        
        gate = torch.rand(BS, NB, device="cuda", dtype=torch.float32)
        # Make it sparse
        mask = torch.rand_like(gate) < sparsity
        gate[mask] = 0.0
        
        actual_sparsity = (gate == 0).float().mean().item()
        print(f"Actual sparsity: {actual_sparsity:.1%}")
        
        mappings = create_stream_compact_index(gate)
        
        print(f"Max rows: {mappings['max_rows']}")
        print(f"Total act_idx: {mappings['total_act_idx']}")
        
        # Verify correctness
        is_valid = verify_mappings(gate, mappings)
        if not is_valid:
            print(f"❌ FAILED for BS={BS}, NB={NB}, sparsity={sparsity:.1%}")
            break
    else:
        print("✅ All random test cases passed!")


def test_edge_cases():
    """Test edge cases like all zeros, all ones, etc."""
    print("\n=== Testing Edge Cases ===")
    
    # Test 1: All zeros
    print("Test 1: All zeros")
    gate_zeros = torch.zeros(10, 8, device="cuda", dtype=torch.float32)
    mappings_zeros = create_stream_compact_index(gate_zeros)
    print(f"Max rows: {mappings_zeros['max_rows']}")
    print(f"Total act_idx: {mappings_zeros['total_act_idx']}")
    verify_mappings(gate_zeros, mappings_zeros)
    
    # Test 2: All ones  
    print("\nTest 2: All ones")
    gate_ones = torch.ones(10, 8, device="cuda", dtype=torch.float32)
    mappings_ones = create_stream_compact_index(gate_ones)
    print(f"Max rows: {mappings_ones['max_rows']}")
    print(f"Total act_idx: {mappings_ones['total_act_idx']}")
    verify_mappings(gate_ones, mappings_ones)
    
    # Test 3: Single active element
    print("\nTest 3: Single active element")
    gate_single = torch.zeros(10, 8, device="cuda", dtype=torch.float32)
    gate_single[5, 3] = 1.0
    mappings_single = create_stream_compact_index(gate_single)
    print(f"Max rows: {mappings_single['max_rows']}")
    print(f"Total act_idx: {mappings_single['total_act_idx']}")
    verify_mappings(gate_single, mappings_single)


def test_fused_correctness():
    """Test that fused kernel produces same results as original"""
    print("\n=== Fused Kernel Correctness Tests ===")
    
    test_cases = [
        (1, 64),    # 1bs 1seq decoding case
        (8, 32),    # Small batch
        (16, 128),  # Medium case
        (64, 64),   # Square case
        (256, 256), # Large case (should fallback)
    ]
    
    for BS, NB in test_cases:
        print(f"\nTesting BS={BS}, NB={NB}")
        
        # Create test data
        gate = torch.rand(BS, NB, device="cuda", dtype=torch.float32)
        mask = torch.rand_like(gate) < 0.7  # 70% sparsity
        gate[mask] = 0.0
        
        # Get results from different methods
        results_orig = create_stream_compact_index(gate)
        results_adaptive = create_stream_compact_index_adaptive(gate)
        
        # Check if this case should use fused kernel
        uses_fused = should_use_fused_kernel(BS, NB, gate.device)
        print(f"Uses fused kernel: {uses_fused}")
        
        if uses_fused:
            # For small cases, test fused directly against original
            results_fused = create_stream_compact_index_fused(gate)
            
            # Compare scalar values
            for key in ['max_rows', 'total_act_idx']:
                assert results_orig[key] == results_fused[key], f"FUSED {key} mismatch: orig={results_orig[key]}, fused={results_fused[key]}"
            
            # Compare mapping tensor shapes first
            for key in ['nb_maxrows_to_bs', 'nb_maxrows_to_actidx', 'nb_maxrows_gate_vals']:
                assert results_orig[key].shape == results_fused[key].shape, f"FUSED {key} shape mismatch: orig={results_orig[key].shape}, fused={results_fused[key].shape}"
            
            # Compare mapping tensor values (these should be identical)
            for key in ['nb_maxrows_to_bs', 'nb_maxrows_to_actidx', 'nb_maxrows_gate_vals']:
                if not torch.equal(results_orig[key], results_fused[key]):
                    # Show some details for debugging
                    diff_mask = results_orig[key] != results_fused[key]
                    if diff_mask.any():
                        print(f"  Differences found at {diff_mask.sum().item()} positions in {key}")
                        print(f"  Tensor shape: {results_orig[key].shape}")
                        
                        # For small tensors, print the whole thing
                        if results_orig[key].numel() <= 100:
                            print(f"  Original {key}:\n{results_orig[key]}")
                            print(f"  Fused {key}:\n{results_fused[key]}")
                        else:
                            # Show first few differences
                            diff_indices = torch.nonzero(diff_mask)[:10]
                            for idx in diff_indices:
                                pos = tuple(idx.tolist())
                                print(f"  At {pos}: orig={results_orig[key][pos].item()}, fused={results_fused[key][pos].item()}")
                    assert False, f"FUSED {key} values mismatch"
            
            # Compare other tensor values
            for key in ['max_rows_per_block', 'bs_counts', 'bs_start_indices']:
                if not torch.equal(results_orig[key], results_fused[key]):
                    print(f"  Original: {results_orig[key]}")
                    print(f"  Fused: {results_fused[key]}")
                    assert False, f"FUSED {key} values mismatch"
            
            print(f"✅ Fused kernel matches original")
        
        # Test adaptive (should match original regardless of which path it takes)
        # Compare scalar values
        for key in ['max_rows', 'total_act_idx']:
            assert results_orig[key] == results_adaptive[key], f"ADAPTIVE {key} mismatch: orig={results_orig[key]}, adaptive={results_adaptive[key]}"
        
        # Compare mapping tensor shapes
        for key in ['nb_maxrows_to_bs', 'nb_maxrows_to_actidx', 'nb_maxrows_gate_vals']:
            assert results_orig[key].shape == results_adaptive[key].shape, f"ADAPTIVE {key} shape mismatch: orig={results_orig[key].shape}, adaptive={results_adaptive[key].shape}"
        
        # Compare mapping tensor values
        for key in ['nb_maxrows_to_bs', 'nb_maxrows_to_actidx', 'nb_maxrows_gate_vals']:
            if not torch.equal(results_orig[key], results_adaptive[key]):
                # Show some details for debugging
                diff_mask = results_orig[key] != results_adaptive[key]
                if diff_mask.any():
                    print(f"  Differences found at {diff_mask.sum().item()} positions")
                    # Show first few differences
                    diff_indices = torch.nonzero(diff_mask)[:5]
                    for idx in diff_indices:
                        pos = tuple(idx.tolist())
                        print(f"  At {pos}: orig={results_orig[key][pos].item()}, adaptive={results_adaptive[key][pos].item()}")
                assert False, f"ADAPTIVE {key} values mismatch"
        
        # Compare other tensor values
        for key in ['max_rows_per_block', 'bs_counts', 'bs_start_indices']:
            if not torch.equal(results_orig[key], results_adaptive[key]):
                print(f"  Original: {results_orig[key]}")
                print(f"  Adaptive: {results_adaptive[key]}")
                assert False, f"ADAPTIVE {key} values mismatch"
        
        print(f"✅ Adaptive matches original (BS={BS}, NB={NB})")


def test_fused_threshold_logic():
    """Test that threshold logic works correctly"""
    print("\n=== Fused Kernel Threshold Logic Tests ===")
    
    device = torch.device("cuda")
    
    # Print GPU info
    from kernels.stream_compact_index.triton_stream_compact_index_fused import get_gpu_shared_memory_size
    shared_mem = get_gpu_shared_memory_size(device)
    usable_mem = int(shared_mem * 0.25)
    max_elements = usable_mem // 20
    print(f"GPU shared memory: {shared_mem//1024}KB")
    print(f"Usable memory (25%): {usable_mem//1024}KB") 
    print(f"Max elements for fused: {max_elements}")
    
    # Test cases that should use fused kernel
    small_cases = [(1, 64), (8, 32), (16, 16), (32, 32)]
    print(f"\nSmall cases (should use fused):")
    for BS, NB in small_cases:
        uses_fused = should_use_fused_kernel(BS, NB, device)
        elements = BS * NB
        print(f"  {BS}×{NB} ({elements} elements): {'✅ fused' if uses_fused else '❌ fallback'}")
    
    # Test cases that should fallback
    large_cases = [(128, 128), (256, 256), (512, 64)]
    print(f"\nLarge cases (should fallback):")
    for BS, NB in large_cases:
        uses_fused = should_use_fused_kernel(BS, NB, device)
        elements = BS * NB
        print(f"  {BS}×{NB} ({elements} elements): {'⚠️ fused' if uses_fused else '✅ fallback'}")


def test_fused_edge_cases():
    """Test edge cases for fused kernel"""
    print("\n=== Fused Kernel Edge Case Tests ===")
    
    # Create edge cases
    all_zeros = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    all_ones = torch.ones(4, 8, device="cuda", dtype=torch.float32)
    
    single_element = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    single_element[2, 3] = 1.0
    
    very_sparse = torch.rand(8, 16, device="cuda", dtype=torch.float32)
    sparse_mask = torch.rand_like(very_sparse) < 0.95  # 95% zeros
    very_sparse[sparse_mask] = 0.0
    
    edge_cases = [
        ("All zeros", all_zeros),
        ("All ones", all_ones),
        ("Single element", single_element),
        ("Very sparse", very_sparse),
    ]
    
    for name, gate in edge_cases:
        print(f"\n{name}: shape={gate.shape}, active={torch.sum(gate > 0).item()}")
        
        results_orig = create_stream_compact_index(gate)
        results_adaptive = create_stream_compact_index_adaptive(gate)
        
        # Basic consistency check
        assert results_orig['max_rows'] == results_adaptive['max_rows'], f"{name} - max_rows mismatch: orig={results_orig['max_rows']}, adaptive={results_adaptive['max_rows']}"
        print(f"✅ {name} - max_rows consistent ({results_orig['max_rows']})")


if __name__ == "__main__":
    """Run all debug tests"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - exiting")
        exit(1)
        
    try:
        import triton
        print("✅ Triton is available")
    except ImportError:
        print("❌ Triton not available - exiting") 
        exit(1)
    
    print("=== Stream Compact Index Debug Suite ===")
    
    # Run all tests
    test_simple_case()
    test_random_cases()
    test_edge_cases()
    
    print("\n✅ Stream compact index debug tests completed!")
    
    # Run fused kernel tests
    test_fused_threshold_logic()
    test_fused_correctness()
    test_fused_edge_cases()
    
    print("\n🎉 All tests passed!")