import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.parallel import (
    DDPExecutor,
    FSDPExecutor,
    NoneExecutor,
    get_rank,
    only_on_rank,
    spawn_parallel_fn,
)


@only_on_rank(0)
def _test_only_on_rank_helper():
    return True


def only_on_rank():
    result = _test_only_on_rank_helper()
    if get_rank() == 0:
        assert result is True
    else:
        assert result is None


def all_reduce():
    x = torch.tensor([get_rank()], dtype=torch.int)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(dist.get_world_size()))
    assert x.item() == expected_sum


def ddp_model_views():
    executor = DDPExecutor()
    training_model, _, _ = executor.prepare(lambda: torch.nn.Linear(4, 3))

    assert isinstance(training_model, DDP)
    assert executor.model_for_training() is training_model
    assert executor.model_for_inference() is training_model.module
    assert executor.rollout_capabilities().supported

    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    training_choice = training_model(inputs).argmax(dim=-1)
    inference_choice = executor.model_for_inference()(inputs).argmax(dim=-1)
    assert torch.equal(training_choice, inference_choice)


def test_none_executor_exposes_same_training_and_inference_model():
    executor = NoneExecutor()
    model, _, _ = executor.prepare(lambda: torch.nn.Linear(2, 2))

    assert executor.model_for_training() is model
    assert executor.model_for_inference() is model


def test_fsdp_executor_rejects_online_rollout():
    executor = FSDPExecutor()
    assert not executor.rollout_capabilities().supported
    with pytest.raises(RuntimeError, match="does not support online rollout"):
        executor.model_for_inference()


def test_spawn_only_on_rank():
    spawn_parallel_fn(only_on_rank, world_size=2, backend="gloo")


def test_spawn_all_reduce():
    spawn_parallel_fn(all_reduce, world_size=2, backend="gloo")


def test_spawn_ddp_model_views():
    spawn_parallel_fn(
        ddp_model_views,
        world_size=2,
        backend="gloo",
        device_type="cpu",
    )
