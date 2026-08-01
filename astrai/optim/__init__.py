"""Optimizer implementations and factory registration."""

from astrai.optim.composite import (
    OptimizerFactory,
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)
from astrai.optim.mano_adamw import Mano, ManoAdamW
from astrai.optim.muon_adamw import MuonAdamW
from astrai.optim.nora_nadamw import (
    NAdamW,
    Nora,
    NoraNAdamW,
    OptimizerParameterGroups,
    nora_direction,
    nora_lr_scale,
    partition_optimizer_parameters,
)

__all__ = [
    "Mano",
    "ManoAdamW",
    "MuonAdamW",
    "NAdamW",
    "Nora",
    "NoraNAdamW",
    "OptimizerFactory",
    "OptimizerParameterGroups",
    "composite_state_dict",
    "composite_step",
    "composite_zero_grad",
    "nora_direction",
    "nora_lr_scale",
    "partition_optimizer_parameters",
    "refresh_param_groups",
]
