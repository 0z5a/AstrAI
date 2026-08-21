"""Tests for HuggingFace checkpoint/config adaptation."""

import json

import pytest
import safetensors.torch as st
import torch

from astrai.config.model_config import ConfigFactory
from astrai.model import AutoModel, AutoRegressiveLM
from astrai.serialization import (
    adapt_config,
    convert_hf_config,
    convert_hf_weights,
    looks_like_hf_state_dict,
    save_model,
)
from tests.helpers import assert_state_dicts_equal, make_tiny_config

LLAMA_RAW = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0",
    "vocab_size": 1000,
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 16,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "mlp_bias": False,
    "head_dim": 4,
}

MOE_RAW = {
    **LLAMA_RAW,
    "model_type": "mixtral",
    "intermediate_size": 16,
    "num_local_experts": 2,
    "num_experts_per_tok": 1,
    "n_shared_experts": 1,
}


def to_hf_keys(state_dict):
    """Rename AstrAI state dict keys to HuggingFace LLaMA-style names."""
    out = {}
    for key, tensor in state_dict.items():
        if key == "embed_tokens.weight":
            out["model.embed_tokens.weight"] = tensor
        elif key == "norm.weight":
            out["model.norm.weight"] = tensor
        elif key.startswith("layers."):
            parts = key.split(".")
            layer = parts[1]
            if parts[2] == "attention":
                out[f"model.layers.{layer}.self_attn.{parts[3]}.{parts[4]}"] = tensor
            elif parts[2] == "input_norm":
                out[f"model.layers.{layer}.input_layernorm.weight"] = tensor
            elif parts[2] == "post_attention_norm":
                out[f"model.layers.{layer}.post_attention_layernorm.weight"] = tensor
            elif parts[2] == "mlp":
                if parts[3] in ("gate", "up", "down"):
                    out[f"model.layers.{layer}.mlp.{parts[3]}_proj.weight"] = tensor
                elif parts[3] == "router":
                    out[f"model.layers.{layer}.mlp.gate.weight"] = tensor
                elif parts[3] == "routed_experts":
                    sub, name = parts[4], parts[5]
                    out[
                        f"model.layers.{layer}.mlp.experts.{sub}.{name}_proj.weight"
                    ] = tensor
                elif parts[3] == "shared_experts":
                    sub, name = parts[4], parts[5]
                    out[
                        f"model.layers.{layer}.mlp.shared_experts.{sub}.{name}_proj.weight"
                    ] = tensor
        else:
            out[key] = tensor
    return out


def test_convert_hf_config_llama():
    cfg = convert_hf_config(LLAMA_RAW)
    assert cfg["model_type"] == "autoregressive_lm"
    assert cfg["hidden_size"] == 8
    assert cfg["num_key_value_heads"] == 1
    loaded = ConfigFactory.load(cfg)
    assert loaded.num_attention_heads == 2
    assert loaded.ffn_type == "mlp"


def test_convert_hf_config_defaults_kv_heads():
    raw = {k: v for k, v in LLAMA_RAW.items() if k != "num_key_value_heads"}
    cfg = ConfigFactory.load(convert_hf_config(raw))
    assert cfg.num_key_value_heads == 2


def test_convert_hf_config_mixtral_moe():
    cfg = convert_hf_config(MOE_RAW)
    assert cfg["ffn_type"] == "moe"
    assert cfg["n_routed_experts"] == 2
    assert cfg["n_activated_experts"] == 1
    assert cfg["n_shared_experts"] == 1
    assert cfg["moe_intermediate_size"] == 16
    loaded = ConfigFactory.load(cfg)
    assert loaded.ffn_type == "moe"


def test_convert_hf_config_mixtral_without_shared_experts():
    raw = {k: v for k, v in MOE_RAW.items() if k != "n_shared_experts"}
    cfg = ConfigFactory.load(convert_hf_config(raw))
    assert cfg.n_shared_experts == 0


def test_convert_hf_config_rejects_bias():
    with pytest.raises(NotImplementedError):
        convert_hf_config({**LLAMA_RAW, "attention_bias": True})


def test_convert_hf_config_rejects_mismatched_head_dim():
    with pytest.raises(NotImplementedError):
        convert_hf_config({**LLAMA_RAW, "head_dim": 8})


def test_looks_like_hf_state_dict():
    assert looks_like_hf_state_dict({"model.layers.0.self_attn.q_proj.weight": 1})
    assert looks_like_hf_state_dict({"model.embed_tokens.weight": 1})
    assert not looks_like_hf_state_dict({"layers.0.attention.q_proj.weight": 1})


def test_adapt_config_passthrough():
    raw = dict(LLAMA_RAW, model_type="autoregressive_lm")
    assert adapt_config(raw) is raw


def test_convert_hf_weights_dense_roundtrip():
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    converted = convert_hf_weights(to_hf_keys(model.state_dict()), cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_moe_roundtrip():
    cfg = make_tiny_config(
        ffn_type="moe",
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    model = AutoRegressiveLM(cfg)
    hf_raw = convert_hf_config(MOE_RAW)
    hf_cfg = ConfigFactory.load(hf_raw)
    converted = convert_hf_weights(to_hf_keys(model.state_dict()), hf_cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_keeps_astrai_keys():
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    converted = convert_hf_weights(dict(model.state_dict()), cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_skips_unmapped_keys():
    cfg = make_tiny_config()
    sd = {"model.rotary_emb.inv_freq": torch.zeros(4), "model.embed_tokens.weight": 1}
    converted = convert_hf_weights(sd, cfg)
    assert "embed_tokens.weight" in converted
    assert "model.rotary_emb.inv_freq" not in converted


def test_convert_hf_weights_rejects_mla():
    cfg = make_tiny_config(attn_type="mla", kv_lora_rank=2)
    sd = {"model.layers.0.self_attn.kv_a_proj_with_mqa.weight": 1}
    with pytest.raises(NotImplementedError):
        convert_hf_weights(sd, cfg)


def test_convert_hf_config_qwen2_moe_preserves_sparse_fields():
    raw = {
        **LLAMA_RAW,
        "model_type": "qwen2_moe",
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [0],
    }
    cfg = ConfigFactory.load(convert_hf_config(raw))
    assert cfg.decoder_sparse_step == 2
    assert cfg.mlp_only_layers == [0]


def test_convert_hf_config_gemma_enables_qk_norm():
    raw = {**LLAMA_RAW, "model_type": "gemma"}
    cfg = ConfigFactory.load(convert_hf_config(raw))
    assert cfg.use_qk_norm is True


def test_convert_hf_weights_moe_with_dense_layers_roundtrip():
    cfg = make_tiny_config(
        ffn_type="moe",
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        mlp_only_layers=[0],
        decoder_sparse_step=1,
    )
    model = AutoRegressiveLM(cfg)
    converted = convert_hf_weights(to_hf_keys(model.state_dict()), cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_qwen2_moe_singular_shared_expert_roundtrip():
    cfg = make_tiny_config(
        ffn_type="moe",
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    model = AutoRegressiveLM(cfg)
    hf_sd = to_hf_keys(model.state_dict())
    hf_sd = {
        k.replace("shared_experts.", "shared_expert.", 1): v for k, v in hf_sd.items()
    }
    converted = convert_hf_weights(hf_sd, cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_gemma_qk_norm_roundtrip():
    cfg = make_tiny_config(use_qk_norm=True)
    model = AutoRegressiveLM(cfg)
    converted = convert_hf_weights(to_hf_keys(model.state_dict()), cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_from_pretrained_hf_directory(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg).eval()
    save_model(
        config=LLAMA_RAW,
        state_dict=to_hf_keys(model.state_dict()),
        save_directory=str(tmp_path),
    )
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids)["logits"], model(input_ids)["logits"]
        )


def test_from_pretrained_astrai_directory(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg).eval()
    save_model(
        config=cfg.to_dict(),
        state_dict=model.state_dict(),
        save_directory=str(tmp_path),
    )
    loaded = AutoModel.from_pretrained(tmp_path, disable_random_init=False)
    assert_state_dicts_equal(loaded.state_dict(), model.state_dict())


def test_from_pretrained_weights_format_hf_on_astrai_dir(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    save_model(
        config=cfg.to_dict(),
        state_dict=model.state_dict(),
        save_directory=str(tmp_path),
    )
    loaded = AutoModel.from_pretrained(
        tmp_path, disable_random_init=False, weights_format="hf"
    )
    assert_state_dicts_equal(loaded.state_dict(), model.state_dict())


def test_from_pretrained_weights_format_astrai_rejects_hf(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    save_model(
        config=LLAMA_RAW,
        state_dict=to_hf_keys(model.state_dict()),
        save_directory=str(tmp_path),
    )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(tmp_path, weights_format="astrai")


def test_from_pretrained_invalid_weights_format(tmp_path):
    cfg = make_tiny_config()
    save_model(
        config=cfg.to_dict(),
        state_dict={},
        save_directory=str(tmp_path),
    )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(tmp_path, weights_format="llama")


def test_from_pretrained_hf_directory_sharded(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg).eval()
    hf_sd = to_hf_keys(model.state_dict())
    keys = sorted(hf_sd)
    split = len(keys) // 2
    shard_a = {k: hf_sd[k] for k in keys[:split]}
    shard_b = {k: hf_sd[k] for k in keys[split:]}
    st.save_file(shard_a, str(tmp_path / "model-00001-of-00002.safetensors"))
    st.save_file(shard_b, str(tmp_path / "model-00002-of-00002.safetensors"))
    index = {
        "metadata": {},
        "weight_map": {
            k: (
                "model-00001-of-00002.safetensors"
                if k in shard_a
                else "model-00002-of-00002.safetensors"
            )
            for k in keys
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps(LLAMA_RAW))

    loaded = AutoModel.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids)["logits"], model(input_ids)["logits"]
        )


def test_from_pretrained_hf_directory_with_moe(tmp_path):
    cfg = make_tiny_config(
        ffn_type="moe",
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    model = AutoRegressiveLM(cfg).eval()
    save_model(
        config=MOE_RAW,
        state_dict=to_hf_keys(model.state_dict()),
        save_directory=str(tmp_path),
    )
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids)["logits"], model(input_ids)["logits"]
        )
