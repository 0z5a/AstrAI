from dataclasses import asdict
from typing import Optional

import torch.nn as nn
from torch import Tensor

from astrai.inference.core.cache import KVCache
from astrai.model.components.attention import AttnFactory
from astrai.model.components.mlp import FFNFactory
from astrai.model.components.norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        cfg = asdict(config)
        cfg.update(
            dim=config.hidden_size,
            dim_ffn=config.intermediate_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            norm_eps=config.rms_norm_eps,
            down_init_std=0.02 / (2 * config.num_hidden_layers) ** 0.5,
        )
        self.attention = AttnFactory.create(config.attn_type, **cfg, layer_id=layer_id)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        ffn_type = self._resolve_ffn_type(config, layer_id)
        self.mlp = FFNFactory.create(ffn_type, **cfg)

    @staticmethod
    def _resolve_ffn_type(config, layer_id: int) -> str:
        if config.ffn_type != "moe":
            return config.ffn_type
        mlp_only = config.mlp_only_layers or []
        if layer_id in mlp_only:
            return "mlp"
        if config.decoder_sparse_step > 1:
            if (layer_id + 1) % config.decoder_sparse_step != 0:
                return "mlp"
        return "moe"

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = False,
    ) -> Tensor:
        attn_output = self.attention(
            self.input_norm(x),
            rotary_emb,
            attention_mask,
            kv_cache,
            is_causal,
        )
        x = attn_output + x
        x = self.mlp(self.post_attention_norm(x)) + x

        return x
