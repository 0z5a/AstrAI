"""Physical KV cache buffers.

Layer 1 — ``KVStorage``:   flat token-level K/V GPU buffers [n_layers, size, n_kv_heads, head_dim]
Layer 2 — ``ReqToTokenPool``: index table [req_idx, pos] → physical token slot
Layer 3 — ``KVCache``:       pure dataclass passed to the model for direct buffer access

These classes have no knowledge of tasks, allocation policies, or scheduling.
They are the "dumb" physical storage layer.
"""

import threading
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor


class ReqToTokenPool:
    """Maps [req_idx, pos] → physical token slot in KV storage.

    Each row is one request; each column is a sequence position.  The value
    at [req_idx, pos] is the flat index into the KV storage buffers.
    """

    def __init__(self, size: int, max_context_len: int, device: torch.device):
        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.long, device=device
        )
        self.free_slots = list(range(size))
        self._lock = threading.Lock()

    def alloc(self, num_reqs: int) -> Optional[List[int]]:
        with self._lock:
            if num_reqs > len(self.free_slots):
                return None
            slots = self.free_slots[:num_reqs]
            self.free_slots = self.free_slots[num_reqs:]
            return slots

    def free(self, req_indices: List[int]):
        with self._lock:
            self.free_slots.extend(req_indices)

    def write(self, indices, values):
        self.req_to_token[indices] = values


class KVStorage:
    """Token-level KV cache storage.

    Buffers: ``[n_layers, size, n_kv_heads, head_dim]``.  Each token occupies
    one slot indexed by ``ReqToTokenPool``.
    """

    def __init__(
        self,
        size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.size = size
        self.k_buffer = torch.empty(
            (n_layers, size, n_kv_heads, head_dim), device=device, dtype=dtype
        )
        self.v_buffer = torch.empty(
            (n_layers, size, n_kv_heads, head_dim), device=device, dtype=dtype
        )

    def get_key_buffer(self, layer_id: int) -> Tensor:
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int) -> Tensor:
        return self.v_buffer[layer_id]

    def set_kv_buffer(self, layer_id: int, loc: Tensor, k: Tensor, v: Tensor) -> None:
        self.k_buffer[layer_id, loc] = k
        self.v_buffer[layer_id, loc] = v


@dataclass
class KVCache:
    """Pure data struct passed to model for KV cache I/O.

    The attention layer does raw buffer indexing — no methods, no abstraction.
    """

    k_buffer: Tensor
    v_buffer: Tensor
    req_to_token: Tensor
    req_pool_indices: Tensor
    seq_lens: Tensor
    out_cache_loc: Tensor
    max_len: int = 0
    kv_indptr: Optional[Tensor] = None
    qo_indptr: Optional[Tensor] = None
    decode_o_part: Optional[Tensor] = None
    decode_ml_part: Optional[Tensor] = None
    decode_out: Optional[Tensor] = None
