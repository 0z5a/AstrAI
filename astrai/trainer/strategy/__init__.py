"""Training strategy compatibility imports.

Implementations are grouped by objective in :mod:`astrai.trainer.strategy`.
"""

from astrai.trainer.strategy.base import BaseStrategy
from astrai.trainer.strategy.dpo import DPOStrategy
from astrai.trainer.strategy.factory import StrategyFactory
from astrai.trainer.strategy.grpo import GRPOStrategy
from astrai.trainer.strategy.ops import (
    _CHUNK_LOGIT_BYTES,
    ForwardResult,
    LogprobsOutput,
    LossOutput,
    _chunked_token_logprobs,
    _chunked_token_logprobs_grad,
    _collect_moe_diagnostics,
    _importance_ratio_metrics,
    _is_packed,
    _truncation_metric,
    _validate_behavior_logprobs,
    compute_gae,
    get_logprobs,
    make_doc_boundary_mask,
    move_to_device,
    rollout_sequences,
    rollout_token_logprobs,
    rollout_token_values,
)
from astrai.trainer.strategy.ppo import PPOStrategy
from astrai.trainer.strategy.supervised import SEQStrategy, SFTStrategy, _CEStrategy

__all__ = [
    "BaseStrategy",
    "DPOStrategy",
    "GRPOStrategy",
    "PPOStrategy",
    "SEQStrategy",
    "SFTStrategy",
    "StrategyFactory",
    "ForwardResult",
    "LossOutput",
    "LogprobsOutput",
    "get_logprobs",
    "move_to_device",
    "compute_gae",
    "_CEStrategy",
    "_CHUNK_LOGIT_BYTES",
    "_chunked_token_logprobs",
    "_chunked_token_logprobs_grad",
    "_collect_moe_diagnostics",
    "_importance_ratio_metrics",
    "_is_packed",
    "_truncation_metric",
    "_validate_behavior_logprobs",
    "make_doc_boundary_mask",
    "rollout_sequences",
    "rollout_token_logprobs",
    "rollout_token_values",
]
