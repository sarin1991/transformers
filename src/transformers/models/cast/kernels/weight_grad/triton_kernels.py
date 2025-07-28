import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Autotuning configurations – reuse the same philosophy as up_proj/down_proj
# -----------------------------------------------------------------------------
_CONFIG_WARPS = (4, 8, 16)
_TILE_SIZES   = (64, 128)
_NUM_STAGES   = (1, 2, 3)
_GROUP_SIZE_I = 16  # grouping factor along the I-dimension

CONFIGS = []
for tile in _TILE_SIZES:
    for warps in _CONFIG_WARPS:
        for stages in _NUM_STAGES:
            CONFIGS.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_I": tile,
                        "BLOCK_SIZE_H": tile,
                        "BLOCK_SIZE_BS": tile,
                        "GROUP_SIZE_I": _GROUP_SIZE_I,
                    },
                    num_warps=warps,
                    num_stages=stages,
                )
            )


@triton.autotune(configs=CONFIGS, key=["intermediate_size", "hidden_size", "batch_seq_size"])
@triton.jit
def fused_weight_grad_kernel(
    y_ptr, x_ptr, out_ptr,  # pointers; y = dy or activations, x = x or dy
    batch_seq_size, intermediate_size, hidden_size,
    stride_s_bs, stride_s_i,
    stride_d_bs, stride_d_h,
    stride_out_i, stride_out_h,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_BS: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    """Compute Gram matrix G = yᵀ · x used as weight-gradient.

    Inputs (all row-major):
        y_ptr :  (batch_seq_size, intermediate_size)  fp16  – sparse or dense activations/gradients
        x_ptr :  (batch_seq_size, hidden_size)        fp16  – dense activations/gradients

    Output:
        out_ptr : (intermediate_size, hidden_size)    fp32/fp16 – accumulated result

    The kernel tiles along I and H dimensions and reduces over the batch_seq axis.
    GROUP_SIZE_I allows several I-tiles to stay resident in the same CTA, mirroring the
    `GROUP_SIZE_BS` strategy used in the down-projection kernel.
    """

    pid = tl.program_id(0)

    # ------------------------------------------------------------------
    # Decompose program id into (i_chunk, h_chunk) with grouping on i
    # ------------------------------------------------------------------
    num_h_chunks = (hidden_size + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    i_chunks = (intermediate_size + BLOCK_SIZE_I - 1) // BLOCK_SIZE_I

    i_chunk_in_group = (pid // num_h_chunks) % GROUP_SIZE_I
    i_group = (pid // num_h_chunks) // GROUP_SIZE_I
    i_chunk = i_group * GROUP_SIZE_I + i_chunk_in_group
    h_chunk = pid % num_h_chunks

    if i_chunk >= i_chunks:
        return

    # Offsets
    offs_i = i_chunk * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_h = h_chunk * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_bs = tl.arange(0, BLOCK_SIZE_BS)

    mask_i = offs_i < intermediate_size
    mask_h = offs_h < hidden_size

    # Accumulator for this tile (I,H)
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_H), dtype=tl.float32)

    # Loop over batchSeq (reduction axis)
    for b in range(0, batch_seq_size, BLOCK_SIZE_BS):
        curr_bs = b + offs_bs
        mask_bs = curr_bs < batch_seq_size

        # Load tile from Y  -> shape (I, BS)
        y_ptrs = y_ptr + offs_i[:, None] * stride_s_i + curr_bs[None, :] * stride_s_bs
        y_block = tl.load(y_ptrs, mask=mask_i[:, None] & mask_bs[None, :], other=0.0)

        # Load tile from X  -> shape (BS, H)
        x_ptrs = x_ptr + curr_bs[:, None] * stride_d_bs + offs_h[None, :] * stride_d_h
        x_block = tl.load(x_ptrs, mask=mask_bs[:, None] & mask_h[None, :], other=0.0)

        # y_block: (I, BS) , x_block : (BS, H)
        acc += tl.dot(y_block, x_block)

    # Write-back
    out_ptrs = out_ptr + offs_i[:, None] * stride_out_i + offs_h[None, :] * stride_out_h
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_i[:, None] & mask_h[None, :])


def fused_weight_grad_triton(
    y: torch.Tensor,
    x: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
):
    """Dense Triton wrapper that computes G = yᵀ · x.

    Args:
        y: (batch_seq, intermediate_size) *float16*/*bfloat16*
        x: (batch_seq, hidden_size)       *float16*/*bfloat16*
        out_dtype: dtype of the returned matrix (default fp32, can be fp16)
    Returns:
        Tensor (intermediate_size, hidden_size)
    """
    assert y.ndim == 2 and x.ndim == 2, "Input tensors must be 2-D"
    assert y.shape[0] == x.shape[0], "Batch dimension mismatch"

    supported_dtypes = (torch.float16, torch.bfloat16)
    assert y.dtype in supported_dtypes and x.dtype in supported_dtypes, "Inputs must be fp16 or bf16"

    batch_seq_size, intermediate_size = y.shape
    hidden_size = x.shape[1]

    out = torch.empty((intermediate_size, hidden_size), device=y.device, dtype=out_dtype)

    def grid(meta):
        I_chunks = triton.cdiv(intermediate_size, meta["BLOCK_SIZE_I"])
        H_chunks = triton.cdiv(hidden_size, meta["BLOCK_SIZE_H"])
        return (I_chunks * H_chunks,)

    fused_weight_grad_kernel[grid](
        y, x, out,
        batch_seq_size,
        intermediate_size,
        hidden_size,
        y.stride(0), y.stride(1),
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        out_dtype=tl.float16 if out_dtype == torch.float16 else tl.float32,
    )

    return out

# -----------------------------------------------------------------------------
# Debug utilities
# -----------------------------------------------------------------------------

def _debug_single_shape(batch_seq, intermediate_size, hidden_size):
    torch.manual_seed(0)
    y = torch.randn(batch_seq, intermediate_size, device="cuda", dtype=torch.float16)
    x = torch.randn(batch_seq, hidden_size, device="cuda", dtype=torch.float16)

    ref = y.transpose(0, 1).float() @ x.float()
    tri = fused_weight_grad_triton(y, x)

    max_diff = (ref - tri).abs().max().item()
    print(f"Shape (N={batch_seq}, I={intermediate_size}, H={hidden_size}) -> max diff {max_diff:.6e}")
    assert max_diff < 1e-2, "Numerical error too high!"


def debug_large_scale():
    shapes = [
        (64, 128, 256),
        (256, 256, 512),
        (512, 512, 768),
    ]
    for bs, I, H in shapes:
        _debug_single_shape(bs, I, H)


if __name__ == "__main__":
    debug_large_scale() 