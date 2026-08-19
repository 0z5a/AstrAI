"""
AutoModel base class for model loading and saving.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Union

import torch.nn as nn

from astrai.config.model_config import BaseModelConfig, ConfigFactory
from astrai.factory import BaseFactory
from astrai.serialization import load_model_config, load_model_weights, save_model


@contextmanager
def _disable_random_init(enable: bool = True):
    if not enable:
        yield
        return

    names = (
        "xavier_normal_",
        "xavier_uniform_",
        "kaiming_normal_",
        "kaiming_uniform_",
        "zeros_",
        "ones_",
        "constant_",
        "normal_",
        "uniform_",
    )
    orig = {n: getattr(nn.init, n) for n in names if hasattr(nn.init, n)}
    for n in orig:
        setattr(nn.init, n, lambda *a, **kw: None)
    try:
        yield
    finally:
        for n, fn in orig.items():
            setattr(nn.init, n, fn)


class ModelFactory(BaseFactory[nn.Module]):
    """Pure factory for model dispatch, separated from nn.Module state."""


class AutoModel(nn.Module):
    """Model base class with loading/saving and generation."""

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        disable_random_init: bool = True,
        strict: bool = True,
    ) -> nn.Module:

        model_path = Path(path)

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw = load_model_config(str(model_path))
        config = ConfigFactory.load(raw)
        model_type = config.model_type or "autoregressive_lm"

        actual_cls = ModelFactory.get_component_class(model_type)

        with _disable_random_init(enable=disable_random_init):
            model = actual_cls(config)

        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = load_model_weights(str(model_path))
            model.load_state_dict(state_dict, strict=strict)

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
    ):
        save_model(
            config=self.config.to_dict(),
            state_dict=self.state_dict(),
            save_directory=str(save_directory),
        )
