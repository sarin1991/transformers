import torch
import torch.nn.functional as F
from typing import Tuple
# Import the fused operation - works when run from cast directory as:
# cd src/transformers/models/cast/
# python ops/debug_cast_mlp_fused.py
try:
    from .cast_mlp_fused import cast_mlp_fused
except ImportError:
    # Fallback for running from cast directory
    from ops.cast_mlp_fused import cast_mlp_fused


def _make_sparse_gate(batch_seq_size: int, num_blocks: int, sparsity: float = 0.9) -> torch.Tensor:
    """Create a sparse gate tensor with specified sparsity level."""
    gate = torch.rand(batch_seq_size, num_blocks, device="cuda", dtype=torch.float32)
    mask = torch.rand_like(gate) < sparsity
    gate[mask] = 0.0
    # Apply ReLU to ensure non-negative values (as expected by the fused op)
    gate = F.relu(gate)
    return gate


def reference_cast_mlp_pytorch(
    x: torch.Tensor, 
    gate: torch.Tensor, 
    up_weight: torch.Tensor, 
    down_weight: torch.Tensor
) -> torch.Tensor:
    """Reference PyTorch implementation of the CAST MLP operation.
    
    Args:
        x: (B, S, H) input tensor
        gate: (B, S, NB) gate tensor (after ReLU)
        up_weight: (H, I) up projection weight
        down_weight: (I, H) down projection weight
    
    Returns:
        (B, S, H) output tensor
    """
    B, S, H = x.shape
    NB = gate.shape[-1]
    I = up_weight.shape[1]  # up_weight is (H, I)
    LS = I // NB
    
    # Flatten for easier processing
    x_flat = x.view(B * S, H)  # (BS, H)
    gate_flat = gate.view(B * S, NB)  # (BS, NB)
    
    # Up projection: (BS, H) @ (H, I) = (BS, I)
    up_proj = x_flat @ up_weight  # (BS, I)
    
    # Apply ReLU activation (as done in the up_proj kernels)
    up_proj = F.relu(up_proj)  # (BS, I)
    
    # Reshape for gating: (BS, I) -> (BS, NB, LS)
    up_proj_reshaped = up_proj.view(B * S, NB, LS)
    
    # Apply gating: multiply each block by its corresponding gate value
    gate_expanded = gate_flat.unsqueeze(-1)  # (BS, NB, 1)
    gated_intermediate = up_proj_reshaped * gate_expanded  # (BS, NB, LS)
    
    # Flatten back: (BS, NB, LS) -> (BS, I)
    gated_intermediate_flat = gated_intermediate.view(B * S, I)
    
    # Down projection: (BS, I) @ (I, H) = (BS, H)
    output_flat = gated_intermediate_flat @ down_weight  # (BS, H)
    
    # Reshape back to original: (BS, H) -> (B, S, H)
    output = output_flat.view(B, S, H)
    
    return output


def run_accuracy_test(
    batch_size: int,
    seq_len: int, 
    hidden_size: int,
    num_blocks: int,
    line_size: int,
    sparsity: float = 0.9,
    dtype: torch.dtype = torch.float16
) -> Tuple[float, float]:
    """Run accuracy test for a single configuration.
    
    Returns:
        Tuple of (max_diff, mean_diff) between reference and fused implementations
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
    
    # Reference implementation (use float32 for higher precision)
    x_ref = x.float()
    up_weight_ref = up_weight.float()
    down_weight_ref = down_weight.float()
    gate_ref = gate.float()
    
    ref_output = reference_cast_mlp_pytorch(x_ref, gate_ref, up_weight_ref, down_weight_ref)
    
    # Fused implementation
    fused_output = cast_mlp_fused(x, gate, up_weight, down_weight)
    
    # Compare outputs (convert fused to float32 for comparison)
    fused_output_f32 = fused_output.float()
    
    max_diff = torch.max(torch.abs(ref_output - fused_output_f32)).item()
    mean_diff = torch.mean(torch.abs(ref_output - fused_output_f32)).item()
    
    print(f"  Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}")
    
    return max_diff, mean_diff


def test_gradient_accuracy(
    batch_size: int = 2,
    seq_len: int = 4,
    hidden_size: int = 128,
    num_blocks: int = 4,
    line_size: int = 32,
    sparsity: float = 0.8
):
    """Test gradient accuracy by comparing against reference implementation."""
    intermediate_size = num_blocks * line_size
    batch_seq_size = batch_size * seq_len
    
    print(f"\n=== Gradient Accuracy Test ===")
    print(f"Config: B={batch_size}, S={seq_len}, H={hidden_size}, NB={num_blocks}, LS={line_size}")
    
    # Create input tensors (require gradients)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True)
    gate = _make_sparse_gate(batch_seq_size, num_blocks, sparsity=sparsity).view(batch_size, seq_len, num_blocks)
    gate.requires_grad_(True)
    up_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float16, requires_grad=True)
    down_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True)
    
    # Reference gradients
    x_ref = x.detach().float().requires_grad_(True)
    gate_ref = gate.detach().float().requires_grad_(True)
    up_weight_ref = up_weight.detach().float().requires_grad_(True)
    down_weight_ref = down_weight.detach().float().requires_grad_(True)
    
    ref_output = reference_cast_mlp_pytorch(x_ref, gate_ref, up_weight_ref, down_weight_ref)
    ref_loss = ref_output.sum()
    ref_loss.backward()
    
    # Fused gradients
    fused_output = cast_mlp_fused(x, gate, up_weight, down_weight)
    fused_loss = fused_output.sum()
    fused_loss.backward()
    
    # Compare gradients
    grad_diffs = {}
    
    # x gradients
    if x.grad is not None and x_ref.grad is not None:
        grad_x_diff = torch.max(torch.abs(x_ref.grad - x.grad.float())).item()
        grad_diffs['x'] = grad_x_diff
        print(f"  Grad x max diff: {grad_x_diff:.6e}")
    
    # gate gradients
    if gate.grad is not None and gate_ref.grad is not None:
        grad_gate_diff = torch.max(torch.abs(gate_ref.grad - gate.grad.float())).item()
        grad_diffs['gate'] = grad_gate_diff
        print(f"  Grad gate max diff: {grad_gate_diff:.6e}")
    
    # up_weight gradients
    if up_weight.grad is not None and up_weight_ref.grad is not None:
        grad_up_diff = torch.max(torch.abs(up_weight_ref.grad - up_weight.grad.float())).item()
        grad_diffs['up_weight'] = grad_up_diff
        print(f"  Grad up_weight max diff: {grad_up_diff:.6e}")
    
    # down_weight gradients
    if down_weight.grad is not None and down_weight_ref.grad is not None:
        grad_down_diff = torch.max(torch.abs(down_weight_ref.grad - down_weight.grad.float())).item()
        grad_diffs['down_weight'] = grad_down_diff
        print(f"  Grad down_weight max diff: {grad_down_diff:.6e}")
    
    return grad_diffs


def debug_cast_mlp_fused(test_gradients: bool = True):
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
        max_diff, mean_diff = run_accuracy_test(
            batch_size, seq_len, hidden_size, num_blocks, line_size,
            sparsity=0.0
        )
        overall_max_diff_dense = max(overall_max_diff_dense, max_diff)
    
    # Test with sparse gates (sparsity = 0.9)
    print("\n=== Sparse Gate Tests ===")
    for batch_size, seq_len, hidden_size, num_blocks, line_size in configs:
        max_diff, mean_diff = run_accuracy_test(
            batch_size, seq_len, hidden_size, num_blocks, line_size,
            sparsity=0.9
        )
        overall_max_diff_sparse = max(overall_max_diff_sparse, max_diff)
    
    print(f"\n=== Overall Results ===")
    print(f"Max diff across all dense configs: {overall_max_diff_dense:.6e}")
    print(f"Max diff across all sparse configs: {overall_max_diff_sparse:.6e}")
    
    # Gradient testing
    if test_gradients:
        try:
            grad_diffs = test_gradient_accuracy()
            max_grad_diff = max(grad_diffs.values()) if grad_diffs else 0.0
            print(f"Max gradient diff: {max_grad_diff:.6e}")
        except Exception as e:
            print(f"Gradient testing failed: {e}")
    
    # Success criteria
    TOLERANCE = 1e-3  # Reasonable tolerance for fp16 operations
    
    success = (
        overall_max_diff_dense < TOLERANCE and 
        overall_max_diff_sparse < TOLERANCE
    )
    
    if success:
        print(f"✅ All tests passed! Max error {max(overall_max_diff_dense, overall_max_diff_sparse):.6e} < {TOLERANCE}")
    else:
        print(f"❌ Tests failed! Max error {max(overall_max_diff_dense, overall_max_diff_sparse):.6e} >= {TOLERANCE}")
    
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
    print("Run from: cd src/transformers/models/cast/ && python ops/debug_cast_mlp_fused.py")
    
    # Run debug tests
    success = debug_cast_mlp_fused(test_gradients=True)
    
    if not success:
        exit(1) 