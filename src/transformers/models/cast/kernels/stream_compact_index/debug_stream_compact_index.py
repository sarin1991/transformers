import torch
from kernels.stream_compact_index.triton_stream_compact_index import create_stream_compact_index


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