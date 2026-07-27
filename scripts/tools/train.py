import os
from collections.abc import Callable
from functools import partial
from typing import Any

import click
import torch
from torch import Tensor, nn, optim

from astrai import setup_logging
from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.dataset import DatasetFactory, dpo_collate_fn, grpo_collate_fn
from astrai.model import AutoRegressiveLM
from astrai.model.components.decoder_block import DecoderBlock
from astrai.trainer import SchedulerFactory, Trainer
from astrai.trainer.rollout import BaseRewardModel


class MuonMix(optim.Optimizer):
    """Combined Muon (matrix) + AdamW (non-matrix) optimizer."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, defaults)

        matrix_params: list[Tensor] = []
        other_params: list[Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.dim() >= 2
                and "norm" not in name
                and "bias" not in name
                and "embed" not in name
                and "lm_head" not in name
            ):
                matrix_params.append(param)
            else:
                other_params.append(param)

        self.muon = optim.Muon(
            matrix_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self.adamw = optim.AdamW(
            [{"params": other_params, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
            fused=True,
        )

        self.param_groups = [*self.muon.param_groups, *self.adamw.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = [*self.muon.param_groups, *self.adamw.param_groups]


def _merge_yaml_into_kwargs(config_path: str, passed_kwargs: dict) -> dict:
    """Load YAML config, then override with explicit CLI kwargs (None excluded)."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    merged = {}
    for section in ("model", "data", "parallel", "training", "ckpt", "log"):
        if section in cfg:
            merged.update(cfg[section])

    for key, value in passed_kwargs.items():
        if value is not None:
            merged[key] = value

    return merged


_TRAIN_TYPE = ["seq", "sft", "dpo", "grpo", "online_grpo", "online_dpo"]
_PARALLEL = ["none", "ddp", "fsdp", "fsdp2"]
_SCHEDULES = ["cosine", "sgdr", "wsd"]
_BACKENDS = ["nccl", "gloo"]
_START_METHODS = ["spawn", "fork", "forkserver"]


@click.command(
    name="train",
    help="Start model training (pretrain / SFT / DPO / GRPO).",
    context_settings={"show_default": True},
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    help="YAML config file. CLI flags override YAML values.",
)
@click.option(
    "--train_type",
    type=click.Choice(_TRAIN_TYPE),
    required=False,
    help="Training type.",
)
@click.option(
    "--data_root_path",
    type=click.Path(exists=True),
    help="Root directory of the dataset.",
)
@click.option(
    "--param_path",
    type=click.Path(exists=True),
    help="Path to model parameters or resume checkpoint.",
)
@click.option("--resume", is_flag=True, default=False, help="Resume from checkpoint.")
@click.option("--n_epoch", type=int, default=1, help="Number of epochs.")
@click.option("--batch_per_device", type=int, default=1, help="Batch size per GPU.")
@click.option(
    "--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps."
)
@click.option(
    "--warmup_ratio",
    type=float,
    default=0.05,
    help="Fraction of total steps for LR warmup.",
)
@click.option("--max_lr", type=float, default=3e-4, help="Max learning rate.")
@click.option(
    "--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping."
)
@click.option("--weight_decay", type=float, default=0.1, help="Weight decay.")
@click.option("--muon_momentum", type=float, default=0.95, help="Muon momentum factor.")
@click.option("--muon_nesterov/--no-muon_nesterov", default=True, help="Muon Nesterov.")
@click.option("--muon_ns_steps", type=int, default=5, help="Muon Newton-Schulz steps.")
@click.option(
    "--muon_adjust_lr",
    type=click.Choice(["original", "match_rms_adamw"]),
    default="match_rms_adamw",
    help="Muon LR adjustment strategy.",
)
@click.option("--random_seed", type=int, default=3407, help="Random seed.")
@click.option("--num_workers", type=int, default=4, help="DataLoader workers.")
@click.option("--pin_memory/--no-pin_memory", default=True, help="Pin memory.")
@click.option(
    "--window_size", type=int, default=None, help="Max input sequence length."
)
@click.option("--stride", type=int, default=None, help="Step size for sliding window.")
@click.option("--dpo_beta", type=float, default=0.1, help="DPO beta.")
@click.option("--group_size", type=int, default=4, help="GRPO group size.")
@click.option("--grpo_clip_eps", type=float, default=0.2, help="GRPO clip epsilon.")
@click.option(
    "--grpo_kl_coef", type=float, default=0.01, help="GRPO KL penalty coefficient."
)
@click.option("--label_smoothing", type=float, default=0.0, help="Label smoothing.")
@click.option(
    "--rollout_interval", type=int, default=512, help="Steps between rollouts."
)
@click.option(
    "--rollout_temperature", type=float, default=0.7, help="Rollout temperature."
)
@click.option("--rollout_top_k", type=int, default=0, help="Rollout top-k (0=disable).")
@click.option("--rollout_top_p", type=float, default=0.9, help="Rollout top-p.")
@click.option(
    "--rollout_max_tokens",
    type=int,
    default=1024,
    help="Max tokens per rollout response.",
)
@click.option(
    "--gradient_checkpointing/--no-gradient_checkpointing",
    default=False,
    help="Enable activation checkpointing.",
)
@click.option(
    "--ckpt_interval", type=int, default=5000, help="Steps between checkpoints."
)
@click.option(
    "--ckpt_dir", type=click.Path(), default="checkpoint", help="Checkpoint directory."
)
@click.option("--val_split", type=float, default=None, help="Validation split ratio.")
@click.option(
    "--val_step", type=int, default=1000, help="Steps between validation runs."
)
@click.option(
    "--metrics",
    multiple=True,
    default=("loss", "lr", "grad_norm"),
    help="Metrics to log (repeatable).",
)
@click.option("--start_epoch", type=int, default=0, help="Start epoch.")
@click.option("--start_samples", type=int, default=0, help="Start samples (per rank).")
@click.option(
    "--master_addr", type=str, default="localhost", help="Master node address."
)
@click.option("--master_port", type=str, default="29500", help="Master node port.")
@click.option(
    "--backend",
    type=click.Choice(_BACKENDS),
    default="nccl",
    help="Distributed backend.",
)
@click.option("--nprocs", type=int, default=1, help="Number of GPUs.")
@click.option(
    "--parallel_mode",
    type=click.Choice(_PARALLEL),
    default="none",
    help="Parallel strategy.",
)
@click.option("--device_type", type=str, default="cuda", help="Device type.")
@click.option(
    "--start_method",
    type=click.Choice(_START_METHODS),
    default="spawn",
    help="Multiprocessing start method.",
)
@click.option("--neftune_alpha", type=float, default=0.0, help="NEFTune noise alpha.")
@click.option(
    "--schedule_type",
    type=click.Choice(_SCHEDULES),
    default="cosine",
    help="LR scheduler.",
)
@click.option(
    "--min_rate", type=float, default=None, help="Minimum LR as fraction of base LR."
)
@click.option("--cycle_length", type=int, default=None, help="SGDR first cycle length.")
@click.option("--t_mult", type=int, default=2, help="SGDR cycle length multiplier.")
@click.option(
    "--stable_steps", type=int, default=None, help="WSD stable plateau steps."
)
@click.option("--decay_steps", type=int, default=None, help="WSD decay steps.")
@click.option("--tp_size", type=int, default=None, help="Tensor parallelism (future).")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config and print plan, do not train.",
)
@click.pass_context
def train_command(ctx, config_path, dry_run, metrics, **kwargs):
    """Start model training (pretrain / SFT / DPO / GRPO)."""
    if config_path:
        kwargs = _merge_yaml_into_kwargs(config_path, kwargs)

    required = ["train_type", "data_root_path", "param_path"]
    missing = [k for k in required if kwargs.get(k) is None]
    if missing:
        raise click.UsageError(
            f"Missing required options: {', '.join(missing)}. "
            f"Use --config YAML or provide them directly."
        )

    # Convert tuple back to list
    kwargs["metrics"] = list(metrics)
    # Remove tp_size (not yet wired)
    kwargs.pop("tp_size", None)

    if dry_run:
        _print_dry_run(kwargs)
        return

    train(**kwargs)


def _print_dry_run(kwargs: dict) -> None:
    """Print training plan summary."""
    rows = [
        ("Train type", kwargs.get("train_type")),
        ("Model path", kwargs.get("param_path")),
        ("Data path", kwargs.get("data_root_path")),
        ("Parallel mode", kwargs.get("parallel_mode", "none")),
        ("GPUs", str(kwargs.get("nprocs", 1))),
        ("Epochs", str(kwargs.get("n_epoch", 1))),
        ("Batch/device", str(kwargs.get("batch_per_device", 1))),
        ("Grad accum", str(kwargs.get("grad_accum_steps", 1))),
        ("Max LR", str(kwargs.get("max_lr", "?"))),
        ("Schedule", str(kwargs.get("schedule_type", "cosine"))),
        ("Warmup ratio", str(kwargs.get("warmup_ratio", 0.05))),
        ("Window size", str(kwargs.get("window_size", "config default"))),
        ("Checkpoint dir", str(kwargs.get("ckpt_dir", "checkpoint"))),
        ("Checkpoint interval", str(kwargs.get("ckpt_interval", 5000))),
        ("Resume", str(kwargs.get("resume", False))),
    ]
    max_len = max(len(k) for k, _ in rows)
    click.secho("\n=== Training Plan (dry-run) ===", fg="cyan", bold=True)
    for key, val in rows:
        click.echo(f"  {key:<{max_len}s} : {val}")
    click.secho("=" * 40, fg="cyan")


def create_model(config):
    return AutoRegressiveLM(config).to(dtype=torch.bfloat16)


def create_optimizer(model, **kwargs) -> MuonMix:
    return MuonMix(model, **kwargs)


def create_scheduler(
    optimizer: optim.Optimizer, **kwargs
) -> optim.lr_scheduler.LRScheduler:
    schedule_type = kwargs.pop("schedule_type")
    return SchedulerFactory.create(schedule_type, optimizer, **kwargs)


def compute_total_steps(
    dataset_len: int,
    n_epoch: int,
    batch_per_device: int,
    nprocs: int,
    grad_accum_steps: int,
) -> int:

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    samples_per_replica = ceil_div(dataset_len, nprocs)
    batches_per_replica = ceil_div(samples_per_replica, batch_per_device)
    total_steps = (batches_per_replica // grad_accum_steps) * n_epoch
    return total_steps


def train(
    train_type: str,
    param_path: str,
    data_root_path: str,
    resume: bool,
    n_epoch: int,
    batch_per_device: int,
    start_epoch: int,
    start_samples: int,
    grad_accum_steps: int,
    warmup_ratio: float,
    ckpt_interval: int,
    ckpt_dir: str,
    val_split: float,
    val_step: int,
    metrics: list[str],
    max_grad_norm: float,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
    gradient_checkpointing: bool,
    window_size: int,
    stride: int,
    nprocs: int,
    parallel_mode: str,
    device_type: str,
    backend: str,
    master_addr: str,
    master_port: str,
    start_method: str,
    neftune_alpha: float,
    schedule_type: str,
    min_rate: float,
    cycle_length: int,
    t_mult: int,
    stable_steps: int,
    decay_steps: int,
    **kwargs,
):
    if train_type not in [
        "seq",
        "sft",
        "dpo",
        "grpo",
        "online_grpo",
        "online_dpo",
    ]:
        raise ValueError(
            f"Invalid train_type '{train_type}'. "
            f"Must be one of: seq, sft, dpo, grpo, online_grpo, online_dpo"
        )
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Model directory not found: {param_path}")
    if nprocs > 1 and parallel_mode == "none":
        raise ValueError(
            "--nprocs > 1 requires --parallel_mode to be 'ddp', 'fsdp', or 'fsdp2'"
        )

    # Load config
    config_path = os.path.join(param_path, "config.json")
    config = AutoRegressiveLMConfig.from_file(config_path)
    config.neftune_alpha = neftune_alpha

    if window_size is None:
        window_size = config.max_position_embeddings

    strategy_kwargs = {
        "beta": kwargs.pop("dpo_beta"),
        "label_smoothing": kwargs.pop("label_smoothing"),
        "clip_eps": kwargs.pop("grpo_clip_eps"),
        "kl_coef": kwargs.pop("grpo_kl_coef"),
        "group_size": kwargs.pop("group_size"),
    }

    rollout_interval = kwargs.pop("rollout_interval", 512)
    rollout_temperature = kwargs.pop("rollout_temperature", 0.7)
    rollout_top_k = kwargs.pop("rollout_top_k", 0)
    rollout_top_p = kwargs.pop("rollout_top_p", 0.9)
    rollout_max_tokens = kwargs.pop("rollout_max_tokens", 1024)
    reward_model_fn: Callable[[], BaseRewardModel] | None = None

    executor_kwargs = {}
    if parallel_mode == "ddp":
        executor_kwargs.update(
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    model_fn = partial(create_model, config)
    dataset = DatasetFactory.load(
        train_type=train_type,
        load_path=data_root_path,
        window_size=window_size,
        stride=stride,
        tokenizer_path=param_path,
    )

    optimizer_fn = partial(
        create_optimizer,
        lr=kwargs.pop("max_lr"),
        weight_decay=kwargs.pop("weight_decay"),
        momentum=kwargs.pop("muon_momentum"),
        nesterov=kwargs.pop("muon_nesterov"),
        ns_steps=kwargs.pop("muon_ns_steps"),
        adjust_lr_fn=kwargs.pop("muon_adjust_lr"),
    )

    total_steps = compute_total_steps(
        len(dataset), n_epoch, batch_per_device, nprocs, grad_accum_steps
    )
    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, total_steps)

    scheduler_kwargs = {"warmup_steps": warmup_steps}

    if schedule_type == "cosine":
        scheduler_kwargs["lr_decay_steps"] = total_steps - warmup_steps
    elif schedule_type == "sgdr":
        scheduler_kwargs["cycle_length"] = cycle_length or (total_steps - warmup_steps)
        scheduler_kwargs["t_mult"] = t_mult
    elif schedule_type == "wsd":
        remaining = total_steps - warmup_steps
        stable_steps_ = stable_steps or max(1, int(remaining * 0.8))
        scheduler_kwargs["stable_steps"] = stable_steps_
        scheduler_kwargs["decay_steps"] = max(
            1, decay_steps or (remaining - stable_steps_)
        )

    if min_rate is not None:
        scheduler_kwargs["min_rate"] = min_rate

    scheduler_fn = partial(
        create_scheduler,
        schedule_type=schedule_type,
        **scheduler_kwargs,
    )

    grad_ckpt_modules = [DecoderBlock] if gradient_checkpointing else []

    collate_fn = None
    if train_type == "dpo":
        collate_fn = dpo_collate_fn
    elif train_type == "grpo":
        collate_fn = grpo_collate_fn
    elif train_type in ("online_grpo", "online_dpo"):
        collate_fn = None

    train_config = TrainConfig(
        model_fn=model_fn,
        strategy=train_type,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch,
        batch_per_device=batch_per_device,
        start_epoch=start_epoch,
        start_samples=start_samples,
        ckpt_interval=ckpt_interval,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        nprocs=nprocs,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
        parallel_mode=parallel_mode,
        device_type=device_type,
        start_method=start_method,
        val_split=val_split,
        val_step=val_step,
        metrics=metrics,
        gradient_checkpointing_modules=grad_ckpt_modules,
        executor_kwargs=executor_kwargs,
        extra_kwargs=strategy_kwargs,
        neftune_alpha=neftune_alpha,
        collate_fn=collate_fn,
        rollout_interval=rollout_interval,
        rollout_temperature=rollout_temperature,
        rollout_top_k=rollout_top_k,
        rollout_top_p=rollout_top_p,
        rollout_max_tokens=rollout_max_tokens,
        reward_model_fn=reward_model_fn,
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=param_path, resume=resume)


if __name__ == "__main__":
    setup_logging()
    train_command()
