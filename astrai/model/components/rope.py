from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


def get_rotary_emb(
    dim: int,
    max_len: int,
    base: float = 10000,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Precompute cos/sin tables for rotary embedding.

    Returns:
        [max_len, dim/2, 2] (f32) — [cos, sin] pairs.
    """
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
    t = torch.arange(0, max_len, dtype=torch.float64, device=device)
    freqs = torch.outer(t, theta).float()
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.stack([cos, sin], dim=-1)


def ntk_base(base: float, dim: int, factor: float) -> float:
    return base * (factor ** (dim / (dim - 2)))


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_len: int,
        base: float = 10000,
        rope_scaling: Optional[Dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.rope_scaling = rope_scaling

        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type", "ntk")
            factor = rope_scaling.get("factor", 1.0)
            if scaling_type == "ntk":
                self.base = ntk_base(base, dim, factor)

        self._set_rotary_buffer(self.max_len)

    def _set_rotary_buffer(self, max_len: int):
        freqs_cis = get_rotary_emb(self.dim, max_len, self.base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """Lookup cos/sin for the given positions.

        Args:
            x: [batch, seq_len, ...] — only batch and seq_len are used.
            position_ids: [batch, seq_len] optional position indices.

        Returns:
            [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs.
        """
        if position_ids is None:
            if x.ndim == 2:
                position_ids = torch.arange(x.size(0), device=x.device)
            else:
                position_ids = (
                    torch.arange(x.size(1), device=x.device)
                    .unsqueeze(0)
                    .expand(x.size(0), -1)
                )
        return self.freqs_cis[position_ids].float()
