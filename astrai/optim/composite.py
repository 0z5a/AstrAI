"""Shared infrastructure for the optim package.

This module hosts two things:

* ``OptimizerFactory`` — the registry for built-in optimizers. Defining it
  here (rather than in ``__init__.py``) lets each optimizer module import it
  and register itself with a decorator, avoiding circular imports.
* Composite-optimizer helpers — ``step``/``zero_grad``/``state_dict``/
  ``param_groups`` delegation shared by every optimizer that routes different
  parameter groups through distinct sub-optimizers.
"""

from typing import Any

import torch
from torch.optim import Optimizer

from astrai.factory import BaseFactory


class OptimizerFactory(BaseFactory[Optimizer]):
    """Factory for built-in training optimizers."""


def composite_step(
    sub_optimizers: list[Optimizer],
    closure=None,
) -> torch.Tensor | None:
    """Run ``step`` on every sub-optimizer, invoking the closure once.

    The closure (if given) is executed inside ``torch.enable_grad`` exactly
    once before any sub-optimizer steps, matching the contract of a single
    ``Optimizer.step``. Sub-optimizers receive ``None`` so they do not
    re-execute it.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    for sub in sub_optimizers:
        sub.step()
    return loss


def composite_zero_grad(
    sub_optimizers: list[Optimizer],
    set_to_none: bool = True,
) -> None:
    for sub in sub_optimizers:
        sub.zero_grad(set_to_none=set_to_none)


def composite_state_dict(
    named_sub_optimizers: dict[str, Optimizer | None],
) -> dict[str, Any]:
    """Serialize sub-optimizers, preserving ``None`` slots."""
    return {
        name: sub.state_dict() if sub is not None else None
        for name, sub in named_sub_optimizers.items()
    }


def refresh_param_groups(
    sub_optimizers: list[Optimizer],
) -> list[dict]:
    """Concatenate param_groups from every non-None sub-optimizer."""
    groups: list[dict] = []
    for sub in sub_optimizers:
        if sub is not None:
            groups.extend(sub.param_groups)
    return groups
