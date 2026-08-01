"""Legacy Muon + AdamW combined optimizer."""

from typing import Any

import torch
from torch import Tensor, nn, optim

from astrai.optim.composite import (
    OptimizerFactory,
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)


@OptimizerFactory.register("muon_adamw")
class MuonAdamW(optim.Optimizer):
    """Combined Muon (matrix) + AdamW (non-matrix) optimizer."""

    optimizer_name = "muon_adamw"

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
        params = [param for param in model.parameters() if param.requires_grad]
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

        self.param_groups = refresh_param_groups([self.muon, self.adamw])

    @torch.no_grad()
    def step(self, closure=None):
        return composite_step([self.muon, self.adamw], closure)

    def zero_grad(self, set_to_none: bool = True):
        composite_zero_grad([self.muon, self.adamw], set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return composite_state_dict({"muon": self.muon, "adamw": self.adamw})

    def load_state_dict(self, state_dict: dict[str, Any]):
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError(
                "Checkpoint optimizer state is not compatible with muon_adamw"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = refresh_param_groups([self.muon, self.adamw])
