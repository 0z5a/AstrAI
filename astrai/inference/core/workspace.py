"""Pre-allocated buffers for the inference decode hot path.

Mirrors SGLang's pre-allocated input buffers (``input_buffers.py``): tensors
are sized once to the server's maximum dimensions and sliced to the live
batch each step, so the per-token decode loop never calls
``torch.empty``/``torch.zeros``/``torch.arange`` for the hot shapes.  Fills
go through ``out=`` variants (``torch.ge``) which write into the stable
buffers instead of allocating fresh results.

All buffers are allocated eagerly at init (nothing is lazy), so the
workspace is CUDA-graph-capture friendly: the decode step reads/writes
fixed-address tensors with no allocation during capture.
"""

import torch
from torch import Tensor


class InferenceWorkspace:
    """Reusable fixed-shape per-step buffers for decode.

    Families of buffers, all sized to ``max_batch_size`` / ``max_seq_len``
    and sliced via views each step:

    - ``decode_mask``: a ``[B, 1, total_len]`` validity mask, the RHS
      ``arange`` pre-computed so only a single ``torch.ge(out=)`` runs per
      step.
    - ``input_ids``: per-step token IDs filled from host (pinned, double-
      buffered so an in-flight async H2D copy never races the next fill).
    - KV-cache bind metadata (``req_pool_indices``, ``seq_lens``,
      ``kv_indptr``, ``inc``, ``out_cache_loc``), written by
      ``PagePool.bind_tasks`` when the Executor passes this workspace.

    No re-allocation while the server's bounds are respected.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # ``position_ids[:, None, None] >= arange`` RHS, reused every step.
        self.arange = torch.arange(max_seq_len, device=device)
        # Decode validity mask: [max_batch, 1, max_seq_len] bool.
        self.input_mask = torch.empty(
            (max_batch_size, 1, max_seq_len), dtype=torch.bool, device=device
        )

        # Per-step token IDs.  Values come from host Python lists every
        # step, so the device buffer is pre-allocated (stable address for
        # CUDA-graph capture) and filled via a host staging buffer.  A
        # double buffer keeps a copy in flight from being overwritten by
        # the next fill.
        self.input_ids = torch.empty((max_batch_size,), dtype=torch.long, device=device)
        self._pin = [
            torch.empty((max_batch_size,), dtype=torch.long),
            torch.empty((max_batch_size,), dtype=torch.long),
        ]
        self._pin_idx = 0

        # KV-cache bind metadata (fixed shape, written by ``PagePool.bind_tasks``
        # when the Executor passes this workspace).  Stable addresses make the
        # decode forward CUDA-graph capturable.
        self.req_pool_indices = torch.empty(
            (max_batch_size,), dtype=torch.long, device=device
        )
        self.seq_lens = torch.empty((max_batch_size,), dtype=torch.long, device=device)
        self.kv_indptr = torch.empty(
            (max_batch_size + 1,), dtype=torch.int32, device=device
        )
        self.qo_indptr = torch.empty(
            (max_batch_size + 1,), dtype=torch.int32, device=device
        )
        self.inc = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.out_cache_loc = torch.empty(
            (max_batch_size, 1), dtype=torch.long, device=device
        )

    def fill_input_ids(self, ids: "list[int]") -> Tensor:
        """Write ``ids`` into the device buffer and return ``[B]``.

        Host values are staged through the double buffer and copied into the
        stable device buffer (``copy_`` without pinning is synchronous, so
        the alternating buffers guard against an in-flight transfer).
        """
        b = len(ids)
        pin = self._pin[self._pin_idx]
        self._pin_idx ^= 1
        for i, v in enumerate(ids):
            pin[i] = v
        self.input_ids[:b].copy_(pin[:b])
        return self.input_ids[:b]

    def decode_mask(self, position_ids: Tensor, total_len: int) -> Tensor:
        """Return the ``[B, 1, total_len]`` validity mask for this step.

        Written into the pre-allocated buffer via ``torch.ge(out=)`` — no
        new tensor is allocated.  ``position_ids`` is the current step's
        ``[B]`` positions; ``total_len`` must not exceed ``max_seq_len``.
        """
        b = position_ids.size(0)
        out = self.input_mask[:b, :, :total_len]
        torch.ge(position_ids[:, None, None], self.arange[:total_len], out=out)
        return out
