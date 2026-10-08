"""End-to-end integration tests for online GRPO/DPO rollout."""

import os
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.trainer.rollout import BaseRewardModel
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer
from tests.support.tokenizers import CHAT_TEMPLATE
from tests.support.trainer import (
    make_online_lr_scheduler,
    make_online_model,
    make_online_optimizer,
)


class InstructionDataset(Dataset):
    """Toy instruction/input dataset for online RL rollout.

    Each sample has an ``instruction`` and an optional ``input``; the
    RolloutGenerator renders both through the tokenizer's chat template
    so the prompt matches the SFT-trained format.
    """

    _SAMPLES = (
        {"instruction": "Hello", "input": ""},
        {"instruction": "Tell me a story", "input": "about dragons"},
        {"instruction": "Summarize", "input": "the article"},
        {"instruction": "Translate", "input": "to French: hi"},
    )

    def __init__(self, repeats=1):
        self.repeats = repeats

    def __len__(self):
        return len(self._SAMPLES) * self.repeats

    def __getitem__(self, idx):
        return dict(self._SAMPLES[idx % len(self._SAMPLES)])


class LengthRewardModel(BaseRewardModel):
    """Rewards each response by its (non-pad) token count.

    Gives the group-normalized advantage a non-degenerate signal.
    """

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        rewards = torch.zeros(B, G)
        for i in range(B):
            for g in range(G):
                rewards[i, g] = float(len(responses[i][g]))
        return rewards


def instruction_collate_fn(batch):
    """Stack a list of instruction/input dicts into a batch dict of lists."""
    return {
        "instruction": [b["instruction"] for b in batch],
        "input": [b.get("input", "") for b in batch],
    }


_ONLINE_STRATEGIES = [
    pytest.param(
        "online_grpo",
        {"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        None,
        id="grpo",
    ),
    pytest.param("online_dpo", {"beta": 0.1, "group_size": 2}, None, id="dpo"),
    pytest.param(
        "online_ppo",
        {
            "clip_eps": 0.2,
            "kl_coef": 0.01,
            "group_size": 2,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
        },
        True,
        id="ppo",
    ),
]

_DDP_TEST_WORLD_SIZE = int(os.environ.get("ASTRAI_DDP_TEST_WORLD_SIZE", "2"))


def _minimal_online_config(**overrides):
    """A TrainConfig for online GRPO that only needs field overrides."""
    defaults = dict(
        strategy="online_grpo",
        model_fn=lambda: torch.nn.Linear(2, 2),
        dataset=InstructionDataset(),
        optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.0),
        scheduler_fn=lambda o: SchedulerFactory.create(
            "cosine", o, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
        ),
        reward_model_fn=LengthRewardModel,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


@pytest.mark.integration
@pytest.mark.skipif(
    torch.cuda.device_count() < _DDP_TEST_WORLD_SIZE,
    reason=f"{_DDP_TEST_WORLD_SIZE} CUDA devices are required",
)
def test_ddp_online_grpo_end_to_end(base_test_env):
    """Run real rollout and optimizer steps on the requested DDP replicas."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]

    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)

    dataset = InstructionDataset(repeats=10)
    train_config = TrainConfig(
        strategy="online_grpo",
        model_fn=partial(make_online_model, model_config),
        dataset=dataset,
        optimizer_fn=make_online_optimizer,
        scheduler_fn=make_online_lr_scheduler,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=1,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type="cuda",
        dp_size=_DDP_TEST_WORLD_SIZE,
        dp_mode="ddp",
        strategy_kwargs={"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        rollout_interval=1,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )

    Trainer(train_config).train(param_path=test_dir)

    expected_steps = (len(dataset) + _DDP_TEST_WORLD_SIZE - 1) // _DDP_TEST_WORLD_SIZE
    assert Path(test_dir, "ckpt", f"epoch_0_step_{expected_steps}").is_dir()
