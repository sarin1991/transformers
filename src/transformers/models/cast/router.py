import torch
import torch.nn as nn
from .configuration_cast import CastConfig

class Router(nn.Module):
    def __init__(self, config: CastConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.l2_line_size = config.l2_line_size
        self.l2_num_blocks = self.intermediate_size // self.l2_line_size
        self.n_experts = self.l2_num_blocks

        self.top_k = config.moe_topk
        self.bias_update_rate = config.router_bias_update_rate
        self.score_fn = config.router_score_fn
        self.bias_clamp = config.router_bias_clamp
        self.eps = 1e-9

        self.register_buffer("expert_bias", torch.zeros(self.n_experts, dtype=torch.float32))

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_fn == "sigmoid":
            return torch.sigmoid(logits)
        if self.score_fn == "softmax":
            return torch.softmax(logits, dim=-1)
        raise ValueError(f"Unknown score_fn={self.score_fn}")

    @torch.no_grad()
    def _update_bias_inplace(self, counts: torch.Tensor):
        counts = counts.to(device=self.expert_bias.device, dtype=torch.float32)
        avg = counts.mean()
        delta = torch.sign(avg - counts)
        self.expert_bias.add_(self.bias_update_rate * delta)
        if self.bias_clamp is not None:
            self.expert_bias.clamp_(-self.bias_clamp, self.bias_clamp)

    def forward(self, logits: torch.Tensor):
        orig_shape = logits.shape
        logits = logits.reshape(-1, orig_shape[-1])  # [T, E]
        E = logits.shape[-1]
        if E != self.n_experts:
            raise ValueError(f"Expected E={self.n_experts}, got {E}")

        scores = self._scores(logits)  # [T, E]
        routed_scores = scores + self.expert_bias.to(dtype=scores.dtype)

        topk_idx = torch.topk(routed_scores, k=self.top_k, dim=-1).indices  # [T, K]

        topk_w = scores.gather(dim=-1, index=topk_idx)                      # [T, K]
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + self.eps)     # [T, K]

        gate = torch.zeros_like(scores)                                     # [T, E]
        gate.scatter_(dim=-1, index=topk_idx, src=topk_w)

        if self.training and torch.is_grad_enabled():
            flat = topk_idx.reshape(-1)
            counts = torch.bincount(flat, minlength=E).to(torch.float32)
            self._update_bias_inplace(counts)

        return gate.reshape(orig_shape)