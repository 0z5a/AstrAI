"""Optimizer implementations and factory registration."""

from torch.optim import Optimizer

from astrai.factory import BaseFactory


class OptimizerFactory(BaseFactory[Optimizer]):
    """Factory for built-in training optimizers."""


from astrai.optim.muon_mix import MuonMix
from astrai.optim.nora_nadamw import (
    NAdamW,
    Nora,
    NoraNAdamW,
    OptimizerParameterGroups,
    nora_direction,
    nora_lr_scale,
    partition_optimizer_parameters,
)

OptimizerFactory.register("nora_nadamw")(NoraNAdamW)
OptimizerFactory.register("muon_adamw")(MuonMix)

__all__ = [
    "MuonMix",
    "NAdamW",
    "Nora",
    "NoraNAdamW",
    "OptimizerFactory",
    "OptimizerParameterGroups",
    "nora_direction",
    "nora_lr_scale",
    "partition_optimizer_parameters",
]
