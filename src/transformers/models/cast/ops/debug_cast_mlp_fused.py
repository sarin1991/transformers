import argparse
import torch
import torch.nn.functional as F
import os
from typing import Tuple
# Import the fused operation - run from cast directory as:
# cd src/transformers/models/cast/
# python -m ops.debug_cast_mlp_fused
from ops.cast_mlp_fused import cast_mlp_fused
from ops.cast_mlp_fused_stream_compact import cast_mlp_fused_stream_compact


def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9) -> torch.Tensor:
    """Create a sparse gate tensor with specified sparsity level."""
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    mask = torch.rand_like(gate) < sparsity
    gate[mask] = 0.0
    return gate


def reference_cast_mlp_pytorch(
    x: torch.Tensor, 
    gate: torch.Tensor, 
    up_weight: torch.Tensor, 
    down_weight: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Reference PyTorch implementation of the CAST MLP operation.
    
    Args:
        x: (B, S, H) input tensor
        gate: (B, S, NB) gate tensor (after ReLU)
        up_weight: (H, I) up projection weight
        down_weight: (I, H) down projection weight
        compute_dtype: dtype for matrix multiplications (default: float32)
    
    Returns:
        (B, S, H) output tensor
    """
    B, S, H = x.shape
    NB = gate.shape[-1]
    I = up_weight.shape[1]  # up_weight is (H, I)
    LS = I // NB
    
    # Flatten for easier processing
    x_flat = x.view(B * S, H)  # (BS, H)
    gate_flat = F.relu(gate.view(B * S, NB))  # (BS, NB) after ReLU as in fused op
    
    # Up projection: (BS, H) @ (H, I) = (BS, I)
    up_proj = torch.matmul(x_flat.to(compute_dtype), up_weight.to(compute_dtype).contiguous())
    
    # Apply ReLU activation (as done in the up_proj kernels)
    up_proj = F.relu(up_proj)  # (BS, I)
    
    # Reshape for gating: (BS, I) -> (BS, NB, LS)
    up_proj_reshaped = up_proj.view(B * S, NB, LS)
    
    # Apply gating: multiply each block by its corresponding gate value
    # Cast gate to same dtype as up_proj to avoid upcasting
    gate_expanded = gate_flat.to(up_proj_reshaped.dtype).unsqueeze(-1)  # (BS, NB, 1)
    gated_intermediate = up_proj_reshaped * gate_expanded  # (BS, NB, LS)
    
    # Flatten back: (BS, NB, LS) -> (BS, I)
    gated_intermediate_flat = gated_intermediate.view(B * S, I).to(x.dtype).contiguous()
    
    # Down projection: (BS, I) @ (I, H) = (BS, H)
    output_flat = torch.matmul(gated_intermediate_flat.to(compute_dtype), down_weight.to(compute_dtype).contiguous())
    
    # Reshape back to original: (BS, H) -> (B, S, H)
    output = output_flat.view(B, S, H).to(x.dtype)
    
    return output


def run_accuracy_test(
    batch_size: int,
    seq_len: int, 
    hidden_size: int,
    num_blocks: int,
    line_size: int,
    sparsity: float = 0.9,
    dtype: torch.dtype = torch.float16
) -> Tuple[float, float, float, float]:
    """Run accuracy test for a single configuration.
    
    Returns:
        Tuple of (rel_max_diff, rel_mean_diff, abs_max_diff, abs_mean_diff)
    """
    intermediate_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len
    
    print(f"  Testing: B={batch_size}, S={seq_len}, H={hidden_size}, NB={num_blocks}, LS={line_size}")
    print(f"  Intermediate size: {intermediate_size}, Sparsity: {sparsity:.1%}")
    
    # Create input tensors
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=sparsity).view(batch_size, seq_len, num_blocks)
    up_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    down_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)
    
    # Reference tensors use the SAME dtype as the forward path
    x_ref = x.detach().clone()
    up_weight_ref = up_weight.detach().clone()
    down_weight_ref = down_weight.detach().clone()
    gate_ref = gate.detach().clone()
    
    ref_output = reference_cast_mlp_pytorch(x_ref, gate_ref, up_weight_ref, down_weight_ref)
    
    # Fused tensors use the SAME dtype as the forward path
    x_fused = x.detach().clone()
    up_weight_fused = up_weight.detach().clone()
    down_weight_fused = down_weight.detach().clone()
    gate_fused = gate.detach().clone()

    # Fused SortPack implementation
    fused_output = cast_mlp_fused(x_fused, gate_fused, up_weight_fused, down_weight_fused)
    
    # Fused Stream Compact implementation
    x_fused_sc = x.detach().clone()
    up_weight_fused_sc = up_weight.detach().clone()
    down_weight_fused_sc = down_weight.detach().clone()
    gate_fused_sc = gate.detach().clone()
    
    fused_sc_output = cast_mlp_fused_stream_compact(x_fused_sc, gate_fused_sc, up_weight_fused_sc, down_weight_fused_sc)
    
    # Compare outputs (convert fused to float32 for comparison)
    fused_output_f32 = fused_output.float()
    fused_sc_output_f32 = fused_sc_output.float()
    
    # Compare SortPack vs Reference
    diff_sp = torch.abs(ref_output - fused_output_f32)
    abs_max_diff_sp = torch.max(diff_sp).item()
    abs_mean_diff_sp = torch.mean(diff_sp).item()
    abs_ref_max = torch.max(torch.abs(ref_output)).item()
    abs_ref_mean = torch.mean(torch.abs(ref_output)).item()
    rel_max_diff_sp = abs_max_diff_sp / (abs_ref_max + 1e-6)
    rel_mean_diff_sp = abs_mean_diff_sp / (abs_ref_mean + 1e-6)
    
    print(f"  SortPack vs Reference:")
    print(f"    Abs Max diff: {abs_max_diff_sp:.6e} | Abs Mean diff: {abs_mean_diff_sp:.6e}")
    print(f"    Rel Max diff: {rel_max_diff_sp:.6e} | Rel Mean diff: {rel_mean_diff_sp:.6e}")
    
    # Compare StreamCompact vs Reference
    diff_sc = torch.abs(ref_output - fused_sc_output_f32)
    abs_max_diff_sc = torch.max(diff_sc).item()
    abs_mean_diff_sc = torch.mean(diff_sc).item()
    rel_max_diff_sc = abs_max_diff_sc / (abs_ref_max + 1e-6)
    rel_mean_diff_sc = abs_mean_diff_sc / (abs_ref_mean + 1e-6)
    
    print(f"  StreamCompact vs Reference:")
    print(f"    Abs Max diff: {abs_max_diff_sc:.6e} | Abs Mean diff: {abs_mean_diff_sc:.6e}")
    print(f"    Rel Max diff: {rel_max_diff_sc:.6e} | Rel Mean diff: {rel_mean_diff_sc:.6e}")
    
    # Compare SortPack vs StreamCompact
    diff_sp_sc = torch.abs(fused_output_f32 - fused_sc_output_f32)
    abs_max_diff_sp_sc = torch.max(diff_sp_sc).item()
    abs_mean_diff_sp_sc = torch.mean(diff_sp_sc).item()
    rel_max_diff_sp_sc = abs_max_diff_sp_sc / (abs_ref_max + 1e-6)
    rel_mean_diff_sp_sc = abs_mean_diff_sp_sc / (abs_ref_mean + 1e-6)
    
    print(f"  SortPack vs StreamCompact:")
    print(f"    Abs Max diff: {abs_max_diff_sp_sc:.6e} | Abs Mean diff: {abs_mean_diff_sp_sc:.6e}")
    print(f"    Rel Max diff: {rel_max_diff_sp_sc:.6e} | Rel Mean diff: {rel_mean_diff_sp_sc:.6e}")
    
    return rel_max_diff_sp, rel_mean_diff_sp, abs_max_diff_sp, abs_mean_diff_sp



def test_gradient_accuracy(
    batch_size: int = 2,
    seq_len: int = 4,
    hidden_size: int = 128,
    num_blocks: int = 4,
    line_size: int = 32,
    sparsity: float = 0.8
):
    """Test gradient accuracy comparing PyTorch vs fused kernel implementations."""
    intermediate_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len
    
    print(f"\n=== Gradient Accuracy Test ===")
    print(f"Config: B={batch_size}, S={seq_len}, H={hidden_size}, NB={num_blocks}, LS={line_size}")
    
    # Create input tensors (shared for both tests)
    x_base = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    gate_base = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=sparsity).view(batch_size, seq_len, num_blocks)
    up_weight_base = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float16)
    down_weight_base = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16)
    
    # Test 1: Reference PyTorch implementation
    print("  Testing against reference PyTorch implementation...")
    x_ref = x_base.detach().clone().requires_grad_(True)
    gate_ref = gate_base.detach().clone().requires_grad_(True) 
    up_weight_ref = up_weight_base.detach().clone().requires_grad_(True)
    down_weight_ref = down_weight_base.detach().clone().requires_grad_(True)
    
    ref_output = reference_cast_mlp_pytorch(x_ref, gate_ref, up_weight_ref, down_weight_ref)
    ref_loss = ref_output.sum()
    ref_loss.backward()
    
    # Store reference gradients immediately
    ref_grads = {
        'x': x_ref.grad.detach().clone(),
        'gate': gate_ref.grad.detach().clone(),
        'up_weight': up_weight_ref.grad.detach().clone(),
        'down_weight': down_weight_ref.grad.detach().clone(),
    }
    
    # Test 2: Fused SortPack implementation gradients
    print("  Testing fused SortPack gradients...")
    x_fused_grad = x_base.detach().clone().requires_grad_(True)
    gate_fused_grad = gate_base.detach().clone().requires_grad_(True)
    up_weight_fused_grad = up_weight_base.detach().clone().requires_grad_(True)
    down_weight_fused_grad = down_weight_base.detach().clone().requires_grad_(True)
    
    fused_grad_output = cast_mlp_fused(x_fused_grad, gate_fused_grad, up_weight_fused_grad, down_weight_fused_grad)
    fused_grad_loss = fused_grad_output.sum()
    fused_grad_loss.backward()
    
    fused_grads = {
        'x': x_fused_grad.grad.detach().clone(),
        'gate': gate_fused_grad.grad.detach().clone(),
        'up_weight': up_weight_fused_grad.grad.detach().clone(),
        'down_weight': down_weight_fused_grad.grad.detach().clone(),
    }
    
    # Test 3: Fused Stream Compact implementation gradients  
    print("  Testing fused Stream Compact gradients...")
    x_sc_grad = x_base.detach().clone().requires_grad_(True)
    gate_sc_grad = gate_base.detach().clone().requires_grad_(True)
    up_weight_sc_grad = up_weight_base.detach().clone().requires_grad_(True)
    down_weight_sc_grad = down_weight_base.detach().clone().requires_grad_(True)
    
    sc_grad_output = cast_mlp_fused_stream_compact(x_sc_grad, gate_sc_grad, up_weight_sc_grad, down_weight_sc_grad)
    sc_grad_loss = sc_grad_output.sum()
    sc_grad_loss.backward()
    
    sc_grads = {
        'x': x_sc_grad.grad.detach().clone(),
        'gate': gate_sc_grad.grad.detach().clone(),
        'up_weight': up_weight_sc_grad.grad.detach().clone(),
        'down_weight': down_weight_sc_grad.grad.detach().clone(),
    }
    
    # Compare gradients
    def compare_grads(name1, grads1, name2, grads2):
        print(f"  {name1} vs {name2} gradients:")
        for key in ['x', 'gate', 'up_weight', 'down_weight']:
            diff = torch.abs(grads1[key] - grads2[key])
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            ref_max = torch.max(torch.abs(grads1[key])).item()
            rel_max = max_diff / (ref_max + 1e-6)
            print(f"    {key}: Max diff: {max_diff:.6e} | Rel Max: {rel_max:.6e}")
    
    compare_grads("Reference", ref_grads, "SortPack", fused_grads)
    compare_grads("Reference", ref_grads, "StreamCompact", sc_grads)
    compare_grads("SortPack", fused_grads, "StreamCompact", sc_grads)
    
    # Test 3: Fused implementation with fused gradients (CAST_USE_FUSED_GRAD_KERNEL=1)  
    print("  Testing fused forward + fused gradients...")
    os.environ["CAST_USE_FUSED_GRAD_KERNEL"] = "1"
    
    # Completely fresh tensors for fused test
    x_fused_grad = x_base.detach().clone().requires_grad_(True)
    gate_fused_grad = gate_base.detach().clone().requires_grad_(True)
    up_weight_fused_grad = up_weight_base.detach().clone().requires_grad_(True)
    down_weight_fused_grad = down_weight_base.detach().clone().requires_grad_(True)
    
    # Explicitly zero gradients
    if x_fused_grad.grad is not None:
        x_fused_grad.grad.zero_()
    if gate_fused_grad.grad is not None:
        gate_fused_grad.grad.zero_()
    if up_weight_fused_grad.grad is not None:
        up_weight_fused_grad.grad.zero_()
    if down_weight_fused_grad.grad is not None:
        down_weight_fused_grad.grad.zero_()
    
    fused_grad_output = cast_mlp_fused(x_fused_grad, gate_fused_grad, up_weight_fused_grad, down_weight_fused_grad)
    fused_grad_loss = fused_grad_output.sum()
    fused_grad_loss.backward()
    
    # Store gradients immediately
    fused_grads = {
        'x': x_fused_grad.grad.detach().clone(),
        'gate': gate_fused_grad.grad.detach().clone(),
        'up_weight': up_weight_fused_grad.grad.detach().clone(), 
        'down_weight': down_weight_fused_grad.grad.detach().clone(),
    }
    
    # Compare gradients
    def compare_grads(name, ref_grad, test_grad):
        abs_diff = torch.max(torch.abs(ref_grad - test_grad.float())).item()
        abs_ref = torch.max(torch.abs(ref_grad)).item() 
        rel_diff = abs_diff / (abs_ref + 1e-6)
        print(f"    {name:<12} | abs diff: {abs_diff:.3e} | rel diff: {rel_diff:.3e}")
        return rel_diff
    
    grad_diffs = {}
    
    print("  SortPack gradients vs Reference:")
    names = ['x', 'gate', 'up_weight', 'down_weight']
    
    fused_diffs = {}
    for name in names:
        fused_diffs[name] = compare_grads(name, ref_grads[name], fused_grads[name])
    grad_diffs['fused'] = fused_diffs
    
    print("  StreamCompact gradients vs Reference:")
    sc_diffs = {}
    for name in names:
        sc_diffs[name] = compare_grads(name, ref_grads[name], sc_grads[name])
    grad_diffs['stream_compact'] = sc_diffs
    
    print("  SortPack vs StreamCompact gradients:")
    comparison_diffs = {}
    for name in names:
        comparison_diffs[name] = compare_grads(name, fused_grads[name], sc_grads[name])
    grad_diffs['fused_vs_pytorch'] = comparison_diffs
    
    return grad_diffs


def debug_cast_mlp_fused(test_gradients: bool = True, *, dtype: torch.dtype = torch.float16):
    """Run comprehensive accuracy tests on the fused CAST MLP operation."""
    
    print("=== CAST MLP Fused Operation Debug ===")
    
    # Test configurations: (B, S, H, NB, LS)
    configs = [
        (2, 4, 128, 4, 32),      # Small test
        (4, 8, 256, 8, 32),      # Medium test  
        (8, 16, 512, 8, 64),     # Larger test
        (16, 32, 1024, 16, 64),  # Large test
        (128, 512, 4096, 8, 4096),  # Very Large test
    ]
    
    overall_max_diff_dense = 0.0
    overall_max_diff_sparse = 0.0
    
    # Test with dense gates (sparsity = 0.0)
    print("\n=== Dense Gate Tests ===")
    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        rel_max_diff, *_ = run_accuracy_test(
            batch_size, seq_len, hidden_size, num_blocks, line_size,
            sparsity=0.0,
            dtype=dtype,
        )
        overall_max_diff_dense = max(overall_max_diff_dense, rel_max_diff)
    
    # Test with sparse gates (sparsity = 0.9)
    print("\n=== Sparse Gate Tests ===")
    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        rel_max_diff, *_ = run_accuracy_test(
            batch_size, seq_len, hidden_size, num_blocks, line_size,
            sparsity=0.9,
            dtype=dtype,
        )
        overall_max_diff_sparse = max(overall_max_diff_sparse, rel_max_diff)
    
    print(f"\n=== Overall Results ===")
    print(f"Max RELATIVE diff across all dense configs: {overall_max_diff_dense:.6e}")
    print(f"Max RELATIVE diff across all sparse configs: {overall_max_diff_sparse:.6e}")
    
    # Gradient testing
    if test_gradients:
        grad_diffs = test_gradient_accuracy()
        # Flatten nested dict to compute global maximum relative diff
        max_grad_diff = 0.0
        for backend_dict in grad_diffs.values():
            if isinstance(backend_dict, dict):
                backend_max = max(backend_dict.values()) if backend_dict else 0.0
                max_grad_diff = max(max_grad_diff, backend_max)
            else:
                max_grad_diff = max(max_grad_diff, backend_dict)
        print(f"Max gradient diff: {max_grad_diff:.6e}")
    
    # Success criteria (relative tolerance)
    REL_TOLERANCE = 3e-1  # 30% relative tolerance for fp16 operations
    
    success = (
        overall_max_diff_dense < REL_TOLERANCE and 
        overall_max_diff_sparse < REL_TOLERANCE
    )
    
    if success:
        print(f"✅ All tests passed! Max relative error {max(overall_max_diff_dense, overall_max_diff_sparse):.6e} < {REL_TOLERANCE}")
    else:
        print(f"❌ Tests failed! Max relative error {max(overall_max_diff_dense, overall_max_diff_sparse):.6e} >= {REL_TOLERANCE}")
    
    return success


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA is not available – exiting.")
        exit(1)
    
    try:
        import triton
        print("✅ Triton is available.")
    except ImportError:
        print("❌ Triton is not available. Please install it.")
        exit(1)
    
    print("✅ CAST MLP Fused operation loaded successfully.")
    print("Run from: cd src/transformers/models/cast/ && python -m ops.debug_cast_mlp_fused")
    
    # ---------------- CLI ----------------
    parser = argparse.ArgumentParser(description="Debug CAST MLP fused op accuracy.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Computation dtype for forward pass")
    parser.add_argument("--skip-grad", action="store_true", help="Skip gradient accuracy check")
    cli_args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype_arg = dtype_map[cli_args.dtype]

    success = debug_cast_mlp_fused(test_gradients=not cli_args.skip_grad, dtype=dtype_arg)
    
    if not success:
        exit(1) 