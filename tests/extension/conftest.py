"""Shared fixtures for extension tests."""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM
from tests.conftest import skip_no_kernel  # noqa: F401 re-export for test modules

D = 64
CFG = dict(
    vocab_size=1000,
    hidden_size=128,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=256,
    max_position_embeddings=64,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
    attn_type="gqa",
    ffn_type="mlp",
)


@pytest.fixture
def cuda_model():
    config = AutoRegressiveLMConfig(**CFG)
    model = AutoRegressiveLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model, config
