import os

import pytest
import torch

from astrai.config import TrainConfig
from astrai.trainer.schedule import SchedulerFactory
from tests.helpers import RandomTokenDataset


def create_train_config(
    model_fn,
    dataset,
    test_dir: str,
    device: str,
    strategy: str = "seq",
    n_epoch: int = 1,
    batch_per_device: int = 2,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    ckpt_interval: int = 5,
    random_seed: int = 42,
    **kwargs,
):
    """Factory function to create common TrainConfig for tests."""

    def optimizer_fn(m):
        return torch.optim.AdamW(m.parameters(), lr=0.001)

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            "cosine", optim, warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    return TrainConfig(
        strategy=strategy,
        model_fn=model_fn,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=test_dir,
        log_dir=os.path.join(test_dir, "logs"),
        n_epoch=n_epoch,
        batch_per_device=batch_per_device,
        ckpt_interval=ckpt_interval,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        device_type=device,
        **kwargs,
    )


@pytest.fixture
def train_config_factory():
    """Fixture providing the ``create_train_config`` factory function."""
    return create_train_config


@pytest.fixture
def trainer_dataset():
    """Fixture providing a dataset for trainer tests."""
    return RandomTokenDataset()
