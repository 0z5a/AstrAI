"""Shared test helpers for the AstrAI test suite."""

import json
import os

import torch
from torch.utils.data import Dataset

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM

TINY_CONFIG = dict(
    vocab_size=1000,
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=16,
    max_position_embeddings=64,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'user' %}USER: {{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
)


def make_tiny_config(**overrides):
    """Create a tiny ``AutoRegressiveLMConfig`` for tests.

    All keyword arguments override ``TINY_CONFIG`` defaults.
    """
    return AutoRegressiveLMConfig(**{**TINY_CONFIG, **overrides})


def make_rollout_config(vocab_size=200, max_position_embeddings=64, **kwargs):
    """Create a tiny config sized for rollout / strategy tests."""
    return make_tiny_config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=max_position_embeddings,
        **kwargs,
    )


def make_model(device, **cfg_overrides):
    """Create a tiny ``AutoRegressiveLM`` on *device* and return ``(model, config)``."""
    cfg = make_rollout_config(**cfg_overrides)
    model = AutoRegressiveLM(cfg).to(device=device)
    model.eval()
    return model, cfg


def make_frozen(model, device):
    """Create a frozen, eval-mode copy of *model* with identical weights."""
    cfg = make_rollout_config()
    copy = AutoRegressiveLM(cfg).to(device=device)
    copy.load_state_dict(model.state_dict())
    copy.requires_grad_(False)
    copy.eval()
    return copy


class RandomTokenDataset(Dataset):
    """Random token dataset combining all test dataset variants.

    Parameters
    ----------
    length : int or None
        Fixed length, or ``None`` for a random length in [100, 200).
    max_length : int
        Sequence length per sample.
    vocab_size : int
        Upper bound for random token ids.
    with_loss_mask : bool
        Include a ``loss_mask`` key in each sample.
    stop_after : int or None
        Raise ``RuntimeError`` after this many samples (for early-stopping tests).
    """

    def __init__(
        self,
        length=100,
        max_length=64,
        vocab_size=1000,
        *,
        with_loss_mask=False,
        stop_after=None,
    ):
        self.length = (
            length if length is not None else int(torch.randint(100, 200, (1,)).item())
        )
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.with_loss_mask = with_loss_mask
        self.stop_after = stop_after
        self._count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.stop_after is not None:
            self._count += 1
            if self._count == self.stop_after:
                raise RuntimeError("Simulated early stopping")

        item = {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "target_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
        }
        if self.with_loss_mask:
            item["loss_mask"] = torch.randint(0, 1, (self.max_length,))
        return item


class FakeTokenizer:
    """Minimal stub tokenizer with optional chat-template support."""

    stop_ids = [2]

    def __init__(self, *, with_chat_template=False):
        if with_chat_template:
            from astrai.tokenize.chat_template import ChatTemplate

            self._chat_template = ChatTemplate.from_string(CHAT_TEMPLATE)
        else:
            self._chat_template = None

    def encode(self, texts, **_):
        if isinstance(texts, str):
            texts = [texts]
        return [[b for b in t.encode("utf-8")] for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list):
            return bytes(b for b in ids if b > 2 or not skip_special_tokens).decode(
                "utf-8", errors="ignore"
            )
        return str(ids)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, **_
    ):
        if self._chat_template is None:
            raise RuntimeError("Chat template not configured")
        rendered = self._chat_template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        if tokenize:
            return (
                self.encode(rendered)[0]
                if isinstance(rendered, str)
                else [self.encode(t)[0] for t in rendered]
            )
        return rendered


class FakeExecutor:
    """Executor stub tracking ``sync_gradients`` and providing ``unwrap_model``."""

    use_distributed = False

    def __init__(self, sync_gradients=True):
        self._sync_gradients = sync_gradients

    @property
    def sync_gradients(self):
        return self._sync_gradients

    def unwrap_model(self, model):
        return model.state_dict()


def find_checkpoint_meta(ckpt_dir):
    """Walk *ckpt_dir* and return the path to the first ``meta.json`` found."""
    for root, _dirs, files in os.walk(ckpt_dir):
        if "meta.json" in files:
            return os.path.join(root, "meta.json")
    return None


def load_checkpoint_meta(ckpt_dir):
    """Find and load the first checkpoint ``meta.json`` under *ckpt_dir*."""
    meta_path = find_checkpoint_meta(ckpt_dir)
    assert meta_path is not None, f"No checkpoint meta.json found in {ckpt_dir}"
    with open(meta_path) as f:
        return json.load(f)


def load_shard_meta(out_dir):
    """Load ``meta.json`` from the default shard output directory."""
    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path), f"Shard meta.json not found at {meta_path}"
    with open(meta_path) as f:
        return json.load(f)


def assert_state_dicts_equal(a, b):
    """Assert two state dicts have identical keys and equal tensor values."""
    assert set(a.keys()) == set(b.keys()), f"Key mismatch: {set(a) ^ set(b)}"
    for key in a:
        assert torch.equal(a[key], b[key]), f"Tensor mismatch at key: {key}"
