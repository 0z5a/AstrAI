"""Utility functions for parallel training."""

from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from astrai.parallel.executor import BaseExecutor


def create_ref_model(
    model_fn: Callable[[], nn.Module],
    executor: Optional["BaseExecutor"] = None,
    model: Optional[nn.Module] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[str] = None,
) -> Optional[nn.Module]:
    """Create a frozen reference model from executor or state dict.

    On non-rank-0, returns None (executor.unwrap_model returns None).
    """
    if state_dict is None and executor is not None and model is not None:
        state_dict = executor.unwrap_model(model)
    if state_dict is None:
        return None

    ref_model = model_fn()
    ref_model.load_state_dict(state_dict)
    ref_model.requires_grad_(False)
    ref_model.eval()
    if device is not None:
        ref_model = ref_model.to(device=device)
    return ref_model
