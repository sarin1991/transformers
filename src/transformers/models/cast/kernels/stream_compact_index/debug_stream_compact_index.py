import torch
from triton_stream_compact_index import create_stream_compact_index


def verify_mappings(gate: torch.Tensor, mappings: dict):
    """
    Verify that the generated mappings are correct.
    
    Args:
        gate: Original gate tensor (BS, NB)
        mappings: Dictionary returned by create_stream_compact_index
    """
    BS, NB = gate.shape
    mask = gate > 0
    
    nb_maxrows_to_bs = mappings['nb_maxrows_to_bs']
    nb_maxrows_to_actidx = mappings['nb_maxrows_to_actidx']
    bs_nb_to_actidx = mappings['bs_nb_to_actidx']
    max_rows = mappings['max_rows']
    total_act_idx = mappings['total_act_idx']
    
    print(f"=== Stream Compact Index Mappings Verification ===")
    print(f"Gate shape: {gate.shape}")
    print(f"Active elements: {mask.sum().item()} / {BS * NB}")
    print(f"Max rows per block: {max_rows}")
    print(f"Total act_idx: {total_act_idx}")
    
    # Verify that all active (bs, nb) pairs have valid act_idx
    active_positions = torch.nonzero(mask, as_tuple=False)  # (num_active, 2)
    
    errors = 0
    for i, (bs, nb) in enumerate(active_positions):
        bs_item, nb_item = bs.item(), nb.item()
        
        # Check bs_nb_to_actidx mapping
        act_idx = bs_nb_to_actidx[bs_item, nb_item].item()
        if act_idx < 0 or act_idx >= total_act_idx:
            print(f"ERROR: Invalid act_idx {act_idx} for active position ({bs_item}, {nb_item})")
            errors += 1
    
    # Verify that all inactive (bs, nb) pairs have act_idx = -1
    inactive_mask = ~mask
    inactive_act_indices = bs_nb_to_actidx[inactive_mask]
    if (inactive_act_indices != -1).any():
        print(f"ERROR: Found non-(-1) act_idx for inactive positions")
        errors += 1
    
    # Verify nb_maxrows mappings consistency
    for nb in range(NB):
        for local_row in range(max_rows):
            bs_mapped = nb_maxrows_to_bs[nb, local_row].item()
            act_idx_mapped = nb_maxrows_to_actidx[nb, local_row].item()
            
            if bs_mapped >= 0:  # Valid entry
                if bs_mapped >= BS:
                    print(f"ERROR: Invalid bs_mapped {bs_mapped} >= {BS}")
                    errors += 1
                    continue
                    
                if act_idx_mapped < 0 or act_idx_mapped >= total_act_idx:
                    print(f"ERROR: Invalid act_idx_mapped {act_idx_mapped}")
                    errors += 1
                    continue
                
                # Check consistency with bs_nb_to_actidx
                expected_act_idx = bs_nb_to_actidx[bs_mapped, nb].item()
                if act_idx_mapped != expected_act_idx:
                    print(f"ERROR: Inconsistent act_idx mapping at ({nb}, {local_row}): "
                          f"nb_maxrows says {act_idx_mapped}, bs_nb says {expected_act_idx}")
                    errors += 1
    
    if errors == 0:
        print("✅ All mappings verified successfully!")
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
    
    print(f"\nGenerated mappings:")
    print(f"nb_maxrows_to_bs:\n{mappings['nb_maxrows_to_bs']}")
    print(f"nb_maxrows_to_actidx:\n{mappings['nb_maxrows_to_actidx']}")
    print(f"bs_nb_to_actidx:\n{mappings['bs_nb_to_actidx']}")
    
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