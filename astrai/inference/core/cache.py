"""KV cache architecture: three-layer separation (SGLang-inspired).

Layer 1 — KVStorage:      flat token-level K/V buffers [n_layers, size, H, D]
Layer 2 — ReqToTokenPool:  index table [req_idx, pos] -> physical token slot
Layer 3 — AllocationStrategy: slot/page allocation with ref-counting and LRU

PagePool owns the physical buffers and bind (KVCache assembly); it does
not know about tasks.  TaskCacheManager owns task_id -> TaskCacheState
mapping and delegates physical slot allocation to the strategy.

KVCache is a pure dataclass passed to the model for direct buffer access.

Two strategies (selected once at construction):
  - ContiguousStrategy: pre-allocated per-request blocks, no dynamic alloc
  - PagedStrategy:      dynamic paged allocation; page_size is a parameter
                        (1 = token-level, >1 = page-level with radix prefix)
"""

import threading
from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, OrderedDict

import torch
from torch import Tensor

from astrai.inference.core.workspace import InferenceWorkspace


@dataclass
class _BindState:
    """Cached bind metadata for steady-state decode increment detection."""

    sig: tuple
    seq_lens: List[int]


@dataclass
class TaskCacheState:
    """Per-task cache allocation state.

    Co-locating all task-owned cache state in one object makes the
    alloc/free/extend lifecycle atomic.
    """

    req_idx: int
    length: int = 0
    cached: int = 0
    pages: List[int] = field(default_factory=list)


def _is_steady_increment(
    prev_sig: Optional[tuple],
    prev_vals: Optional[List[int]],
    cur_sig: tuple,
    cur_vals: List[int],
) -> bool:
    """True when the same ordered set has every value +1 from the previous step."""
    return (
        prev_sig is not None
        and prev_vals is not None
        and prev_sig == cur_sig
        and len(prev_vals) == len(cur_vals)
        and all(c == p + 1 for c, p in zip(cur_vals, prev_vals))
    )


def page_hash(
    token_ids: List[int], page_idx: int, page_size: int, parent_hash: int = 0
) -> int:
    start = page_idx * page_size
    end = min(start + page_size, len(token_ids))
    h = parent_hash
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


class RadixNode:
    """A page-aligned edge in the CPU-side prefix radix."""

    __slots__ = ("parent", "children", "page_idx", "tokens", "lock_ref")

    def __init__(self, parent=None, tokens=(), page_idx=None):
        self.parent = parent
        self.children: Dict[tuple, "RadixNode"] = {}
        self.page_idx = page_idx
        self.tokens = tuple(tokens)
        self.lock_ref = 0


class RadixCache:
    """Page-granular radix prefix index with exact token matching."""

    def __init__(self, page_size: int):
        self._page_size = page_size
        self._root = RadixNode()
        self._page_to_node: Dict[int, RadixNode] = {}
        self._lock = threading.Lock()

    def evict(self, idx: int):
        with self._lock:
            node = self._page_to_node.pop(idx, None)
            if node is None:
                return
            node.page_idx = None
            parent = node.parent
            if parent is not None:
                parent.children.pop(node.tokens, None)

    def has_page(self, idx: int) -> bool:
        with self._lock:
            return idx in self._page_to_node

    def lookup(self, token_ids: List[int]) -> List[int]:
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            hits: List[int] = []
            node = self._root
            for i in range(full_pages):
                start = i * self._page_size
                page_tokens = tuple(token_ids[start : start + self._page_size])
                child = node.children.get(page_tokens)
                if child is None or child.page_idx is None:
                    break
                hits.append(child.page_idx)
                node = child
            return hits

    def record(self, page_idx: int, token_ids: List[int], logical_page_idx: int):
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            if logical_page_idx >= full_pages:
                return
            old = self._page_to_node.pop(page_idx, None)
            if old is not None and old.parent is not None:
                old.parent.children.pop(old.tokens, None)

            node = self._root
            for i in range(logical_page_idx + 1):
                start = i * self._page_size
                page_tokens = tuple(token_ids[start : start + self._page_size])
                child = node.children.get(page_tokens)
                if child is None:
                    child = RadixNode(node, page_tokens)
                    node.children[page_tokens] = child
                node = child
            if node.page_idx is not None and node.page_idx != page_idx:
                replaced = node.page_idx
                self._page_to_node.pop(replaced, None)
            node.page_idx = page_idx
            self._page_to_node[page_idx] = node

    def release(self, pages: List[int]) -> None:
        with self._lock:
            for page_idx in pages:
                node = self._page_to_node.get(page_idx)
                if node is not None and node.lock_ref:
                    node.lock_ref -= 1


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


class AllocationStrategy(ABC):
    """Physical slot allocation policy.

    The base class provides contiguous-mode defaults (all no-ops): req_to_token
    is pre-filled at PagePool init, so no dynamic allocation is needed.
    PagedStrategy overrides every method to add dynamic allocation.
    """

    def alloc(self, state: TaskCacheState, prompt_ids: List[int]) -> bool:
        return True

    def free(self, state: TaskCacheState) -> None:
        pass

    def extend(self, state: TaskCacheState, pos: int) -> bool:
        return True

    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None:
        pass

    def record_hashes(
        self, state: TaskCacheState, prompt_ids: List[int], start: int
    ) -> None:
        pass


class PagedStrategy(AllocationStrategy):
    """Dynamic paged allocation from a shared bitmask pool.

    page_size is a parameter, not a separate strategy: at page_size=1 each
    allocated page *is* one token slot (``page * 1 + 0``), and prefix
    caching is simply disabled (``prefix=None``).  The unified page
    formula ``pages[page_idx] * page_size + offset`` holds for both.
    """

    def __init__(
        self,
        alloc: Allocator,
        prefix: Optional[RadixCache],
        page_size: int,
        req_pool: ReqToTokenPool,
        device,
    ):
        self._alloc = alloc
        self._prefix = prefix
        self._page_size = page_size
        self._req_pool = req_pool
        self._device = device

    def alloc(self, state: TaskCacheState, prompt_ids: List[int]) -> bool:
        if self._prefix is not None:
            hits = self._prefix.lookup(prompt_ids)
            state.cached = len(hits) * self._page_size
            for p in hits:
                self._alloc.inc_ref(p)
            state.pages = list(hits)

        remaining = len(prompt_ids) - state.cached
        if remaining <= 0:
            return True
        n_new = (remaining + self._page_size - 1) // self._page_size
        for _ in range(n_new):
            p = self._alloc.alloc()
            if p < 0:
                return False
            state.pages.append(p)
        return True

    def free(self, state: TaskCacheState) -> None:
        if self._prefix is not None:
            for p in state.pages:
                keep = self._prefix.has_page(p)
                self._alloc.free(p, keep_cached=keep)
                if not keep:
                    self._prefix.evict(p)
        else:
            for p in state.pages:
                self._alloc.free(p)

    def extend(self, state: TaskCacheState, pos: int) -> bool:
        page_idx = pos // self._page_size
        if page_idx >= len(state.pages):
            p = self._alloc.alloc()
            if p < 0:
                return False
            state.pages.append(p)
        offset = pos % self._page_size
        self._req_pool.req_to_token[state.req_idx, pos] = (
            state.pages[page_idx] * self._page_size + offset
        )
        return True

    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None:
        total = len(prompt_ids)
        for pos in range(state.cached, total):
            page_idx = pos // self._page_size
            offset = pos % self._page_size
            if page_idx < len(state.pages):
                self._req_pool.req_to_token[state.req_idx, pos] = (
                    state.pages[page_idx] * self._page_size + offset
                )

    def record_hashes(
        self, state: TaskCacheState, prompt_ids: List[int], start: int
    ) -> None:
        if self._prefix is None:
            return
        full = len(prompt_ids) // self._page_size
        for i in range(start, min(full, len(state.pages))):
            self._prefix.record(state.pages[i], prompt_ids, i)


class PagePool:
    """Physical KV cache: buffers + req-pool + allocation strategy + bind.

    Does not know about tasks — task lifecycle is managed by
    :class:`TaskCacheManager`, which holds a reference to this pool.
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
        self.n_tokens = max_batch_size * max_seq_len if self.contiguous else n_tokens

        self._storage = KVStorage(
            self.n_tokens, n_layers, n_kv_heads, head_dim, device, dtype
        )
        self._req_pool = ReqToTokenPool(max_batch_size, max_seq_len, device)

        if self.contiguous:
            for i in range(max_batch_size):
                self._req_pool.req_to_token[i] = torch.arange(
                    i * max_seq_len, (i + 1) * max_seq_len, device=device
                )
            self._strategy = AllocationStrategy()
        else:
            n_pages = self.n_tokens // page_size
            alloc = Allocator(n_pages)
            prefix = RadixCache(page_size) if page_size > 1 else None
            if prefix is not None:
                alloc.on_evict = prefix.evict
            self._strategy = PagedStrategy(
                alloc, prefix, page_size, self._req_pool, device
            )

    def bind_tasks(
        self,
        req_indices: List[int],
        seq_lens: List[int],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
        incremental: bool = False,
    ) -> KVCache:
        if device is None:
            device = workspace.device
        b = len(req_indices)

        rpi_buf = workspace.req_pool_indices
        sl_buf = workspace.seq_lens
        kvp_buf = workspace.kv_indptr
        inc_buf = workspace.inc
        ocl_buf = workspace.out_cache_loc

        if incremental:
            sl_buf[:b] += 1
            kvp_buf[: b + 1] += inc_buf[: b + 1]
        else:
            rpi_buf[:b].copy_(
                torch.tensor(req_indices, dtype=torch.long, device=device)
            )
            sl_buf[:b].copy_(torch.tensor(seq_lens, dtype=torch.long, device=device))
            kvp_buf[: b + 1].zero_()
            kvp_buf[1 : b + 1] = sl_buf[:b].cumsum(0).to(torch.int32)

        req_pool_indices = rpi_buf[:b]
        seq_lens_t = sl_buf[:b]
        kv_indptr = kvp_buf[: b + 1]

        if start_pos is not None:
            seq_len = seq_lens[0]
            out_cache_loc = self._req_pool.req_to_token[
                req_pool_indices, start_pos:seq_len
            ]
            q_len = seq_len - start_pos
            workspace.qo_indptr[: b + 1].copy_(
                torch.arange(b + 1, dtype=torch.int32, device=device) * q_len
            )
            qo_indptr = workspace.qo_indptr[: b + 1]
            decode_o_part = decode_ml_part = decode_out = None
        else:
            write_pos = seq_lens_t - 1
            loc = self._req_pool.req_to_token[req_pool_indices, write_pos].unsqueeze(-1)
            ocl_buf[:b].copy_(loc)
            out_cache_loc = ocl_buf[:b]
            qo_indptr = None
            decode_o_part = getattr(workspace, "decode_o_part", None)
            decode_ml_part = getattr(workspace, "decode_ml_part", None)
            decode_out = getattr(workspace, "decode_out", None)

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
            decode_o_part=decode_o_part,
            decode_ml_part=decode_ml_part,
            decode_out=decode_out,
        )


class TaskCacheManager:
    """Task <-> KV slot lifecycle manager.

    Sole owner of task state.  Owns the task_id -> TaskCacheState map and
    delegates physical slot allocation to the strategy, and KV bind to
    PagePool.  Held directly by the scheduler — not as a PagePool attribute.
    """

    def __init__(
        self,
        strategy: AllocationStrategy,
        req_pool: ReqToTokenPool,
        max_seq_len: int,
        pool: PagePool,
    ):
        self._strategy = strategy
        self._req_pool = req_pool
        self._max_seq_len = max_seq_len
        self._pool = pool
        self._states: Dict[str, TaskCacheState] = {}
        self._bind_state: Optional[_BindState] = None

    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool:
        self._bind_state = None
        req_slots = self._req_pool.alloc(1)
        if req_slots is None:
            return False
        state = TaskCacheState(req_idx=req_slots[0])
        self._states[task_id] = state
        if not self._strategy.alloc(state, prompt_ids):
            self._rollback(state, task_id)
            return False
        self._strategy.write_indices(state, prompt_ids)
        state.length = len(prompt_ids)
        return True

    def _rollback(self, state: TaskCacheState, task_id: str):
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])
        self._states.pop(task_id, None)

    def task_free(self, task_id: str):
        self._bind_state = None
        state = self._states.pop(task_id, None)
        if state is None:
            return
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])

    def task_extend(self, task_id: str, pos: int) -> bool:
        state = self._states.get(task_id)
        if state is None or pos >= self._max_seq_len:
            return False
        if not self._strategy.extend(state, pos):
            return False
        state.length = pos + 1
        return True

    def task_cached(self, task_id: str) -> int:
        state = self._states.get(task_id)
        return state.cached if state is not None else 0

    def task_record_hashes(
        self, task_id: str, prompt_ids: List[int], start_logical_page: int = 0
    ):
        state = self._states.get(task_id)
        if state is not None:
            self._strategy.record_hashes(state, prompt_ids, start_logical_page)

    @staticmethod
    def task_cacheable_ids(task_id: str, prompt_ids: List[int], output_ids: List[int]):
        """Return the sequence whose KV entries are already materialized.

        The first sampled output is produced by prompt prefill, and the last
        sampled output has not been decoded into KV yet.
        """
        return list(prompt_ids) + list(output_ids[:-1])

    def bind(
        self,
        task_ids: List[str],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
    ) -> KVCache:
        states = [self._states[tid] for tid in task_ids]
        req_indices = [s.req_idx for s in states]
        seq_lens = [s.length for s in states]
        sig = tuple(req_indices)

        prev = self._bind_state
        incremental = (
            start_pos is None
            and prev is not None
            and _is_steady_increment(prev.sig, prev.seq_lens, sig, seq_lens)
        )
        self._bind_state = _BindState(sig, list(seq_lens))
        self._bind_was_steady = incremental

        return self._pool.bind_tasks(
            req_indices,
            seq_lens,
            workspace,
            device=device,
            start_pos=start_pos,
            incremental=incremental,
        )

    @property
    def bind_was_steady(self) -> bool:
        """Whether the most recent ``bind()`` was a steady-state increment."""
        return self._bind_was_steady
