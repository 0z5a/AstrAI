"""Streaming IterableDataset for pre-training with shard-level shuffle.

Unlike the map-style datasets, the streaming dataset yields windows
sequentially through each data shard — no random access, no sampler.
Each DataLoader worker independently streams its assigned shard subset,
giving better OS page-cache locality for large-scale (TB+) datasets.

Key properties:
- Implements ``torch.utils.data.IterableDataset``.
- ``__len__`` returns total window count so ``compute_total_steps`` works.
- Shard-level shuffle with deterministic seed (reproducible across runs).
- Distributed: each rank gets a disjoint subset of shards.
- Multi-worker: each worker within a rank gets a disjoint subset.
"""

import random
from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import IterableDataset

from astrai.dataset.storage import Store


def _resolve_rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _total_windows(token_count, window_size, stride):
    if token_count <= window_size:
        return 0
    return (token_count - 1 - window_size) // stride + 1


class StreamingSeqDataset(IterableDataset):
    """Streaming next-token prediction dataset.

    Yields ``{"input_ids": [L], "target_ids": [L]}`` dicts by sliding a
    window sequentially through each data shard.  Shards are shuffled
    deterministically.  Distributed and multi-worker DataLoader modes are
    supported: each consumer gets a disjoint shard subset.

    Args:
        store: Already-loaded Store with a ``"sequence"`` key.
        window_size: Context length per sample.
        stride: Step between consecutive windows (default: window_size).
        shuffle: Shuffle shard order.
        seed: Base seed for deterministic shard shuffle.
    """

    def __init__(
        self,
        store: Store,
        window_size: int,
        stride: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.store = store
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.shuffle = shuffle
        self.seed = seed
        self._rank, self._world_size = (
            rank,
            world_size if rank is not None else _resolve_rank_and_world_size(),
        )

        if "sequence" not in store.keys:
            raise KeyError(
                f"Store is missing required key 'sequence'; "
                f"available keys: {sorted(store.keys)}"
            )

    @property
    def num_samples(self) -> int:
        return _total_windows(self.store.token_count, self.window_size, self.stride)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        segments = self.store._data["sequence"]
        n_shards = len(segments)

        indices = list(range(n_shards))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_consumers = self._world_size
            consumer_id = self._rank
        else:
            num_consumers = self._world_size * worker_info.num_workers
            consumer_id = self._rank * worker_info.num_workers + worker_info.id

        my_shards = [
            i for idx, i in enumerate(indices) if idx % num_consumers == consumer_id
        ]

        for shard_idx in my_shards:
            segment = segments[shard_idx]
            seq_len = segment.shape[0]
            for begin in range(0, seq_len - self.window_size, self.stride):
                end = begin + self.window_size
                yield {
                    "input_ids": torch.as_tensor(segment[begin:end], dtype=torch.long),
                    "target_ids": torch.as_tensor(
                        segment[begin + 1 : end + 1], dtype=torch.long
                    ),
                }
