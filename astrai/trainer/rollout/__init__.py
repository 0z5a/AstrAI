"""Compatibility imports for online training rollout components.

Implementations live in :mod:`astrai.trainer.rollout`; existing import paths
remain stable for callers.
"""

from astrai.trainer.rollout.generator import RolloutGenerator
from astrai.trainer.rollout.runner import (
    RolloutEvaluator,
    RolloutRunner,
    _score_rewards,
)
from astrai.trainer.rollout.types import (
    _PAD,
    BaseRewardModel,
    RawRollout,
    RolloutResult,
    RolloutVersionError,
    SamplingParams,
)

__all__ = [
    "BaseRewardModel",
    "RawRollout",
    "RolloutEvaluator",
    "RolloutGenerator",
    "RolloutResult",
    "RolloutRunner",
    "RolloutVersionError",
    "SamplingParams",
    "_PAD",
    "_score_rewards",
]
