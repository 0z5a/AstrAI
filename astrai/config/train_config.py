from dataclasses import field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from astrai.config.base import BaseConfig
from astrai.model.components.lora import LoRAConfig


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrainConfig(BaseConfig):
    """Training configuration.

    Combines hyperparameters with runtime objects (model_fn, dataset, etc.).
    Only JSON-serializable fields are written to checkpoint meta via to_dict().

    Args:
        model_fn (Callable[[], nn.Module]): Model factory for training.
        strategy (str): Training strategy (seq, sft, dpo, grpo, online_*).
        dataset (Dataset): Dataset for training.
        optimizer_fn (Callable[[nn.Module], Optimizer]): Optimizer factory for training.
        scheduler_fn (Callable[[Optimizer], LRScheduler]): Scheduler factory for training.
        n_epoch (int): Number of epochs for training. Defaults to 1.
        batch_per_device (int): Batch size per device. Defaults to 4.
        grad_accum_steps (int): Number of iterations between optimizer steps. Defaults to 1.
        max_grad_norm (Optional[float]): Maximum gradient norm. None disables clipping. Defaults to 1.0.
        gradient_checkpointing_modules (List[type]): Module types to enable activation checkpointing for. Defaults to [].
        compile_mode (Optional[str]): torch.compile mode: 'default', 'reduce-overhead', 'max-autotune', or None. Defaults to None.
        start_epoch (int): Start epoch for training. Defaults to 0.
        start_samples (int): Start samples count (per rank). Superseded by checkpoint consumed_samples. Defaults to 0.
        ckpt_dir (str): Checkpoint directory. Defaults to "./checkpoint".
        ckpt_interval (int): Number of optimizer steps between checkpoints. Defaults to 5000.
        lora (Optional[LoRAConfig]): LoRA config. None means full fine-tuning. Defaults to None.
        metrics (List[str]): Metrics to record during training. Defaults to ["loss", "lr", "grad_norm"].
        random_seed (int): Random seed. Defaults to 3407.
        num_workers (int): Number of workers for dataloader. Defaults to 0.
        prefetch_factor (Optional[int]): Prefetch factor for dataloader. Defaults to None.
        pin_memory (bool): Pin memory for dataloader. Defaults to False.
        collate_fn (Optional[Callable[[List[Any]], Any]]): Collate function for dataloader (e.g. dpo_collate_fn). Defaults to None.
        nprocs (int): Number of processes for distributed training. Defaults to 1.
        backend (str): Distributed training backend. Defaults to "nccl".
        master_addr (str): Master address for distributed training. Defaults to "localhost".
        master_port (str): Master port for distributed training. Defaults to "29500".
        parallel_mode (str): Parallel strategy: none, ddp, fsdp. Defaults to "none".
        start_method (str): Multiprocessing start method: spawn/fork/forkserver. Defaults to "spawn".
        device_type (str): Device type for distributed training. Defaults to "cuda".
        val_dataset (Optional[Dataset]): Dataset for validation. Defaults to None.
        val_split (Optional[float]): Ratio to split from training dataset for validation, e.g. 0.05. Defaults to None.
        val_step (int): Number of optimizer steps between validation runs. Defaults to 1000.
        neftune_alpha (float): NEFTune noise alpha, 0=disabled, typical: 5.0. Defaults to 0.0.
        rollout_interval (int): Number of optimizer steps between online rollouts. Defaults to 512.
        rollout_temperature (float): Sampling temperature for online rollout. Defaults to 0.7.
        rollout_top_k (int): Top-k filtering for online rollout, 0=disable. Defaults to 0.
        rollout_top_p (float): Top-p (nucleus) filtering for online rollout. Defaults to 0.9.
        rollout_max_tokens (int): Maximum generated tokens per response in rollout. Defaults to 1024.
        reward_model_fn (Optional[Callable]): Factory for reward model, required for online RL strategies. Defaults to None.
        executor_kwargs (Dict[str, Any]): Extra kwargs passed to ExecutorFactory.create(). Defaults to {}.
        extra_kwargs (Dict[str, Any]): Other arguments. Defaults to {}.
    """

    model_fn: Callable[[], nn.Module]
    strategy: str
    dataset: Dataset
    optimizer_fn: Callable[[nn.Module], Optimizer]
    scheduler_fn: Callable[[Optimizer], LRScheduler]
    n_epoch: int = 1
    batch_per_device: int = 4
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    gradient_checkpointing_modules: List[type] = field(default_factory=list)
    compile_mode: Optional[str] = None

    start_epoch: int = 0
    start_samples: int = 0
    ckpt_dir: str = "./checkpoint"
    ckpt_interval: int = 5000

    lora: Optional[LoRAConfig] = None

    metrics: List[str] = field(default_factory=lambda: ["loss", "lr", "grad_norm"])

    random_seed: int = 3407
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    pin_memory: bool = False
    collate_fn: Optional[Callable[[List[Any]], Any]] = None

    nprocs: int = 1
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "29500"
    parallel_mode: str = "none"
    start_method: str = "spawn"

    device_type: str = "cuda"
    val_dataset: Optional[Dataset] = None
    val_split: Optional[float] = None
    val_step: int = 1000
    neftune_alpha: float = 0.0

    rollout_interval: int = 512
    rollout_temperature: float = 0.7
    rollout_top_k: int = 0
    rollout_top_p: float = 0.9
    rollout_max_tokens: int = 1024
    reward_model_fn: Optional[Callable] = None

    executor_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
