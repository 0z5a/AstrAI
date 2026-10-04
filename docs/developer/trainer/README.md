# Trainer module boundaries

The training package has three kinds of code: training run setup,
training objectives, and online rollout. Keep those responsibilities separate
when adding new features.

```text
astrai/trainer/
  trainer.py             training loop
  train_context.py       public context and builder entry point
  strategy/
    __init__.py          public strategy imports
    ops.py               tensor and loss helpers
    base.py              common strategy lifecycle
    factory.py           strategy registry
    supervised.py        sequence and SFT objectives
    dpo.py               DPO objective
    grpo.py              GRPO objective
    ppo.py               PPO objective
  rollout/
    __init__.py          public rollout imports
    types.py             rollout data and sampling contracts
    generator.py         inference-backed generation
    runner.py            reward scoring, replay cache and evaluation
    setup.py             connect model, backend, generator and evaluator
  backend.py             colocated and replica rollout backends
  schedule.py            learning-rate schedules
  train_callback.py      training callbacks
```

The context builder configures the strategy and rollout. Rollout generation
uses the existing in-process inference backend.
Objectives consume rollout values, not mutable inference requests. The online
generator remains colocated with the training model when that backend is
selected, and shares its policy-version boundary.

`strategy/__init__.py` and `rollout/__init__.py` preserve existing import
paths. New code should import the implementation module directly when it
needs to patch internal helpers.

Each objective owns its loss and algorithm-specific state. Shared tensor
operations belong in `strategy/ops.py`; model creation, checkpoint restore and
data loading remain in `train_context.py`, while rollout wiring belongs in
`rollout/setup.py`. Avoid adding another
conditional objective branch to the context builder when it can be expressed
as a registered strategy or a small assembly helper.
