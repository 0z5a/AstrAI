"""KV cache architecture: three-layer separation (SGLang-inspired).

Layer 1 — KVStorage:      flat token-level K/V buffers [n_layers, size, H, D]
Layer 2 — ReqToTokenPool:  index table [req_idx, pos] → physical token slot
Layer 3 — Allocator:       slot/page allocation with ref-counting and LRU

PagePool orchestrates all three plus PrefixCache (content addressing).
KVCache is a pure dataclass passed to the model for direct buffer access.

Two modes:
  - contiguous (default): pre-allocated per-request blocks, no dynamic alloc
  - paged: shared pool with on-demand allocation, prefix caching support
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from astrai.inference.core.workspace import InferenceWorkspace


def page_hash(token_ids: List[int], page_idx: int, page_size: int) -> int:
    start = page_idx * page_size
    end = min(start + page_size, len(token_ids))
    h = 0
    for i in range(start, end):
        h = (h * 31 + token_ids[i]) & 0xFFFFFFFFFFFFFFFF
    return h


class Allocator:
    """Bitmask-based page allocator with ref-counting and LRU eviction."""

    def __init__(self, n_pages: int):
        self._free_mask = (1 << n_pages) - 1
        self._refs: List[int] = [0] * n_pages
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.on_evict: Optional[Callable[[int], None]] = None
        self._lock = threading.Lock()

    def alloc(self) -> int:
        with self._lock:
            if self._free_mask:
                lsb = self._free_mask & -self._free_mask
                idx = lsb.bit_length() - 1
                self._free_mask ^= lsb
                self._refs[idx] = 1
                return idx
            if self._lru:
                idx, _ = self._lru.popitem(last=False)
                if self.on_evict:
                    self.on_evict(idx)
                self._refs[idx] = 1
                self._free_mask &= ~(1 << idx)
                return idx
            return -1

    def free(self, idx: int, keep_cached: bool = False):
        with self._lock:
            self._refs[idx] -= 1
            if self._refs[idx] == 0:
                if keep_cached:
                    self._lru[idx] = None
                else:
                    self._free_mask |= 1 << idx

    def inc_ref(self, idx: int):
        with self._lock:
            self._refs[idx] += 1
            self._lru.pop(idx, None)

    def ref_count(self, idx: int) -> int:
        with self._lock:
            return self._refs[idx]

    def touch(self, idx: int):
        with self._lock:
            if idx in self._lru:
                self._lru.move_to_end(idx)


class PrefixCache:
    """Hash-based prefix matching: maps page hashes to physical page indices."""

    def __init__(self, page_size: int):
        self._page_size = page_size
        self._page_to_hash: Dict[int, int] = {}
        self._hash_to_page: Dict[int, int] = {}
        self._lock = threading.Lock()

    def evict(self, idx: int):
        with self._lock:
            h = self._page_to_hash.pop(idx, None)
            if h is not None:
                self._hash_to_page.pop(h, None)

    def has_page(self, idx: int) -> bool:
        with self._lock:
            return idx in self._page_to_hash

    def lookup(self, token_ids: List[int]) -> List[int]:
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            hits: List[int] = []
            for i in range(full_pages):
                h = page_hash(token_ids, i, self._page_size)
                p = self._hash_to_page.get(h)
                if p is None:
                    break
                hits.append(p)
            return hits

    def record(self, page_idx: int, token_ids: List[int], logical_page_idx: int):
        with self._lock:
            h = page_hash(token_ids, logical_page_idx, self._page_size)
            old_h = self._page_to_hash.pop(page_idx, None)
            if old_h is not None:
                self._hash_to_page.pop(old_h, None)
            self._page_to_hash[page_idx] = h
            self._hash_to_page[h] = page_idx


class ReqToTokenPool:
    """Maps [req_idx, pos] -> physical token slot in KV storage.

    Each row is one request; each column is a sequence position. The value
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

    Buffers: [n_layers, size, n_kv_heads, head_dim]. Each token occupies
    one slot indexed by ReqToTokenPool.
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

    Attributes:
        k_buffer: [n_layers, size, n_kv_heads, head_dim]
        v_buffer: [n_layers, size, n_kv_heads, head_dim]
        req_to_token: [num_reqs, max_ctx_len] — index table
        req_pool_indices: [batch_size] — row indices into req_to_token
        seq_lens: [batch_size] — per-request total sequence lengths
        out_cache_loc: [batch, new_seq_len] or [batch, 1] — write indices
        max_len: max(seq_lens) as Python int — avoids GPU sync in decode
        kv_indptr: [batch+1] int32 — prefix sum of seq_lens, precomputed once
            per step so the attention backend avoids rebuilding it per layer.
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


class PagePool:
    """Top-level KV cache manager.

    Combines KVStorage + ReqToTokenPool + Allocator + PrefixCache.

    Args:
        n_layers: Number of transformer layers.
        n_kv_heads: Number of KV attention heads.
        head_dim: Dimension per head.
        max_batch_size: Maximum concurrent requests.
        max_seq_len: Maximum sequence length per request.
        device, dtype: Tensor device and dtype.
        page_size: Page size for paged mode (1 = token-level).
        n_tokens: Total token slots for paged mode. None = contiguous mode
            (pre-allocates max_batch_size * max_seq_len).
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        page_size: int = 1,
        n_tokens: Optional[int] = None,
    ):
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.contiguous = n_tokens is None
        if self.contiguous:
            self.n_tokens = max_batch_size * max_seq_len
        else:
            self.n_tokens = n_tokens

        self._storage = KVStorage(
            self.n_tokens, n_layers, n_kv_heads, head_dim, device, dtype
        )
        self._req_pool = ReqToTokenPool(max_batch_size, max_seq_len, device)

        if self.contiguous:
            for i in range(max_batch_size):
                self._req_pool.req_to_token[i] = torch.arange(
                    i * max_seq_len, (i + 1) * max_seq_len, device=device
                )
            self._alloc: Optional[Allocator] = None
            self._prefix: Optional[PrefixCache] = None
        else:
            n_pages = self.n_tokens // page_size
            self._alloc = Allocator(n_pages)
            self._prefix = PrefixCache(page_size) if page_size > 1 else None
            if self._prefix is not None:
                self._alloc.on_evict = self._prefix.evict

        self._task_req: Dict[str, int] = {}
        self._task_len: Dict[int, int] = {}
        self._task_cached: Dict[str, int] = {}
        self._task_slots: Dict[str, List[int]] = {}
        self._task_pages: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

        # Steady-state decode validation state: the ordered task set and its
        # Python seq_lens mirror.  When the same set advances every sequence
        # by exactly one token per step, bind_tasks updates the stable
        # buffers in-place (+=1 / +=inc) instead of re-cumsumming.  Any
        # task-set change is a miss and rebuilds.
        self._bind_sig: Optional[tuple] = None
        self._bind_seq_lens: Optional[List[int]] = None

    # ---- task lifecycle ----

    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool:
        req_slots = self._req_pool.alloc(1)
        if req_slots is None:
            return False
        req_idx = req_slots[0]
        self._task_req[task_id] = req_idx

        if self.contiguous:
            self._task_len[req_idx] = len(prompt_ids)
            self._task_cached[task_id] = 0
            return True

        n_tokens_needed = len(prompt_ids)
        cached = 0

        if self._prefix is not None:
            hits = self._prefix.lookup(prompt_ids)
            cached = len(hits) * self.page_size
            for p in hits:
                self._alloc.inc_ref(p)
            self._task_pages[task_id] = list(hits)
            self._task_slots[task_id] = []
        else:
            self._task_pages[task_id] = []
            self._task_slots[task_id] = []

        remaining = n_tokens_needed - cached
        if remaining > 0:
            if self.page_size == 1:
                slots = self._alloc_tokens(remaining)
                if slots is None:
                    for p in self._task_pages[task_id]:
                        self._alloc.free(p)
                    self._req_pool.free([req_idx])
                    del self._task_req[task_id]
                    return False
                self._task_slots[task_id] = slots
            else:
                n_new_pages = (remaining + self.page_size - 1) // self.page_size
                new_pages = []
                for _ in range(n_new_pages):
                    p = self._alloc.alloc()
                    if p < 0:
                        for hp in self._task_pages[task_id]:
                            self._alloc.free(hp)
                        for np_ in new_pages:
                            self._alloc.free(np_)
                        self._req_pool.free([req_idx])
                        del self._task_req[task_id]
                        return False
                    new_pages.append(p)
                self._task_pages[task_id].extend(new_pages)

        self._write_req_to_token(task_id, prompt_ids, cached)
        self._task_len[req_idx] = len(prompt_ids)
        self._task_cached[task_id] = cached
        return True

    def task_free(self, task_id: str):
        req_idx = self._task_req.pop(task_id, None)
        if req_idx is None:
            return
        self._task_len.pop(req_idx, None)
        self._task_cached.pop(task_id, None)

        if not self.contiguous:
            if self._prefix is not None:
                for p in self._task_pages.get(task_id, []):
                    keep = self._prefix.has_page(p)
                    self._alloc.free(p, keep_cached=keep)
                    if not keep:
                        self._prefix.evict(p)
            else:
                for p in self._task_pages.get(task_id, []):
                    self._alloc.free(p)
            self._task_pages.pop(task_id, None)
            self._task_slots.pop(task_id, None)

        self._req_pool.free([req_idx])

    def task_extend(self, task_id: str, pos: int) -> bool:
        req_idx = self._task_req.get(task_id)
        if req_idx is None or pos >= self.max_seq_len:
            return False

        # Paged mode must also claim a physical slot for the new token;
        # contiguous mode's block is pre-allocated so this is a no-op.
        if not self.contiguous and not self._extend_slot(task_id, req_idx, pos):
            return False

        self._task_len[req_idx] = pos + 1
        return True

    def _extend_slot(self, task_id: str, req_idx: int, pos: int) -> bool:
        """Allocate the physical slot for one extended token (paged mode)."""
        if self.page_size == 1:
            slots = self._alloc_tokens(1)
            if slots is None:
                return False
            self._task_slots.setdefault(task_id, []).extend(slots)
            self._req_pool.req_to_token[req_idx, pos] = slots[0]
            return True

        page_idx = pos // self.page_size
        existing = self._task_pages.get(task_id, [])
        if page_idx >= len(existing):
            p = self._alloc.alloc()
            if p < 0:
                return False
            existing.append(p)
            self._task_pages[task_id] = existing
        page_offset = pos % self.page_size
        page = existing[page_idx]
        token_slot = page * self.page_size + page_offset
        self._req_pool.req_to_token[req_idx, pos] = token_slot
        return True

    def task_cached(self, task_id: str) -> int:
        return self._task_cached.get(task_id, 0)

    def task_record_hashes(
        self, task_id: str, prompt_ids: List[int], start_logical_page: int = 0
    ):
        if self._prefix is None or self.contiguous:
            return
        pages = self._task_pages.get(task_id, [])
        full_pages = len(prompt_ids) // self.page_size
        for i in range(start_logical_page, min(full_pages, len(pages))):
            self._prefix.record(pages[i], prompt_ids, i)

    # ---- bind for forward ----

    def bind_tasks(
        self,
        task_ids: List[str],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
    ) -> KVCache:
        if device is None:
            device = workspace.device
        req_indices = [self._task_req[tid] for tid in task_ids]
        # Per-request lengths come from the pool's own tracking (task_alloc
        # sets len(prompt_ids); task_extend sets pos+1), so callers need not
        # pass them.
        seq_lens = [self._task_len[req_idx] for req_idx in req_indices]
        b = len(task_ids)
        sig = tuple(task_ids)

        # Write into the caller's workspace buffers (fixed addresses, sized
        # to max_batch/max_seq at init) — the sole owner of the per-step
        # KV bind tensors.
        rpi_buf = workspace.req_pool_indices
        sl_buf = workspace.seq_lens
        kvp_buf = workspace.kv_indptr
        inc_buf = workspace.inc
        ocl_buf = workspace.out_cache_loc

        incremental = (
            start_pos is None
            and self._bind_sig is not None
            and self._bind_sig == sig
            and self._bind_seq_lens is not None
            and len(self._bind_seq_lens) == b
            and all(s == p + 1 for s, p in zip(seq_lens, self._bind_seq_lens))
        )
        if incremental:
            # Steady-state decode: advance the stable buffers in-place.
            # Normal-mode buffers keep ``+=`` legal regardless of whether
            # this runs inside ``torch.inference_mode()``.
            sl_buf[:b] += 1
            kvp_buf[: b + 1] += inc_buf[: b + 1]
            req_pool_indices = rpi_buf[:b]
            seq_lens_t = sl_buf[:b]
            kv_indptr = kvp_buf[: b + 1]
        else:
            # Cold path: fill the stable buffers from fresh host tensors.
            rpi_buf[:b].copy_(
                torch.tensor(req_indices, dtype=torch.long, device=device)
            )
            sl_buf[:b].copy_(torch.tensor(seq_lens, dtype=torch.long, device=device))
            kvp_buf[: b + 1].zero_()
            kvp_buf[1 : b + 1] = sl_buf[:b].cumsum(0).to(torch.int32)
            req_pool_indices = rpi_buf[:b]
            seq_lens_t = sl_buf[:b]
            kv_indptr = kvp_buf[: b + 1]
        self._bind_sig = sig
        self._bind_seq_lens = list(seq_lens)

        if start_pos is not None:
            seq_len = seq_lens[0]
            out_cache_loc = self._req_pool.req_to_token[
                req_pool_indices, start_pos:seq_len
            ]
            # Ragged query segmentation for the prefill kernel, computed once
            # (was rebuilt per layer in CudaBackend.fwd_prefill).
            q_len = seq_len - start_pos
            workspace.qo_indptr[: b + 1].copy_(
                torch.arange(b + 1, dtype=torch.int32, device=device) * q_len
            )
            qo_indptr = workspace.qo_indptr[: b + 1]
        else:
            write_pos = seq_lens_t - 1
            loc = self._req_pool.req_to_token[req_pool_indices, write_pos].unsqueeze(-1)
            ocl_buf[:b].copy_(loc)
            out_cache_loc = ocl_buf[:b]
            qo_indptr = None

        return KVCache(
            k_buffer=self._storage.k_buffer,
            v_buffer=self._storage.v_buffer,
            req_to_token=self._req_pool.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_t,
            out_cache_loc=out_cache_loc,
            max_len=max(seq_lens),
            kv_indptr=kv_indptr,
            qo_indptr=qo_indptr,
        )

    # ---- internals ----

    def _alloc_tokens(self, n: int) -> Optional[List[int]]:
        if self.page_size != 1:
            raise RuntimeError("_alloc_tokens is for page_size=1 only")
        slots = []
        for _ in range(n):
            p = self._alloc.alloc()
            if p < 0:
                for s in slots:
                    self._alloc.free(s)
                return None
            slots.append(p)
        return slots

    def _write_req_to_token(self, task_id: str, prompt_ids: List[int], cached: int):
        req_idx = self._task_req[task_id]
        total = len(prompt_ids)

        if self.contiguous:
            return

        if self.page_size == 1:
            slots = self._task_slots.get(task_id, [])
            all_slots = slots[: total - cached]
            if all_slots:
                self._req_pool.req_to_token[req_idx, cached:total] = torch.tensor(
                    all_slots, dtype=torch.long, device=self.device
                )
        else:
            pages = self._task_pages.get(task_id, [])
            for pos in range(cached, total):
                page_idx = pos // self.page_size
                page_offset = pos % self.page_size
                if page_idx < len(pages):
                    token_slot = pages[page_idx] * self.page_size + page_offset
                    self._req_pool.req_to_token[req_idx, pos] = token_slot
