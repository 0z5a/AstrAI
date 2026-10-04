"""Registry for training strategies."""

from astrai.factory import BaseFactory
from astrai.trainer.strategy.base import BaseStrategy


class StrategyFactory(BaseFactory[BaseStrategy]):
    """Factory class for creating training strategy instances.

    Supports decorator-based registration for extensible strategy types.
    All default strategies (seq, sft, dpo, grpo) are automatically registered.

    Example usage:
        @StrategyFactory.register("custom")
        class CustomStrategy(BaseStrategy):
            ...

        strategy = StrategyFactory.create("custom", model, device)
    """


# ============== Strategy Classes ==============
# All strategies are registered at class definition time using the decorator
