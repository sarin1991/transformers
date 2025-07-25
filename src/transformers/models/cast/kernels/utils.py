import torch
import triton
import triton.language as tl


_TRITON_DTYPE_MAP = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }