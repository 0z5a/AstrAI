from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def get_rotary_emb(
    dim: int,
    max_len: int,
    base: float = 10000,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for rotary embedding.

    Returns:
        (cos, sin) each of shape [max_len, dim/2] (f32)
    """
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)
    t = torch.arange(0, max_len, dtype=torch.float64, device=device)
    freqs = torch.outer(t, theta).float()
    return torch.cos(freqs), torch.sin(freqs)


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
        cos, sin = get_rotary_emb(self.dim, max_len, self.base)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(
        self, x: Tensor, position_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Lookup cos/sin for the given positions.

        Args:
            x: [batch, seq_len, ...] — only batch and seq_len are used.
            position_ids: [batch, seq_len] optional position indices.

        Returns:
            (cos, sin) each of shape [batch, seq_len, dim/2] (f32)
        """
        if position_ids is None:
            position_ids = (
                torch.arange(x.size(1), device=x.device)
                .unsqueeze(0)
                .expand(x.size(0), -1)
            )
        cos = self.cos_table[position_ids].float()
        sin = self.sin_table[position_ids].float()
        return cos, sin
