"""KV cache allocation layer.

Encapsulates the physical slot allocation policy, isolated from GPU buffers
and task lifecycle management.

- ``TaskCacheState``: data contract between strategy and manager (per-task slot state)
- ``Allocator``:       bitmask-based page allocator with LRU eviction
- ``RadixCache``:      page-granular prefix index (exact token match)
- ``AllocationStrategy``: ABC for physical slot allocation
- ``ContiguousStrategy``: statically partitioned, no dynamic allocation
- ``PagedStrategy``:    dynamic paged allocation from a shared pool
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, OrderedDict

import torch

from astrai.model.kv_cache import ReqToTokenPool

# ---- data contract: per-task slot state ----


@dataclass
class TaskCacheState:
    """Per-task cache allocation state.

    Co-locates all task-owned cache metadata so the alloc/free/extend
    lifecycle is atomic.  Owned by ``TaskCacheManager``, consumed by
    every ``AllocationStrategy`` method.

    ``slots`` mirrors the page-size-1 tail of ``req_to_token[req_idx]`` on
    the host: ``extend`` appends to it instead of issuing one tiny H2D copy
    per decoded token, and ``write_indices`` flushes the prefix it did not
    already cover in one bulk write.
    """

    req_idx: int
    length: int = 0
    cached: int = 0
    pages: List[int] = field(default_factory=list)
    _slots: List[int] = field(default_factory=list)
    _flushed: int = 0


# ---- allocation primitives ----


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

    def alloc_many(self, n: int) -> Optional[List[int]]:
        """Allocate up to ``n`` pages, or ``None`` if fewer are free.

        A whole prompt's pages in one lock acquisition and one mask scan:
        per-page ``alloc`` costs a lock round-trip and a bit extraction per
        token, which dominated host time at serving batch sizes (575 calls
        per 512-token prompt). Falls back to LRU eviction page-by-page when
        the free mask alone cannot satisfy the request.
        """
        if n <= 0:
            return []
        with self._lock:
            free = bin(self._free_mask).count("1")
            if free < n:
                promoted = 0
                while free + promoted < n and self._lru:
                    idx, _ = self._lru.popitem(last=False)
                    if self.on_evict:
                        self.on_evict(idx)
                    self._free_mask |= 1 << idx
                    promoted += 1
                if free + promoted < n:
                    return None
            out: List[int] = []
            mask = self._free_mask
            while len(out) < n:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                out.append(idx)
                mask ^= lsb
                self._refs[idx] = 1
            self._free_mask = mask
            return out

    def free(self, idx: int, keep_cached: bool = False):
        with self._lock:
            self._refs[idx] -= 1
            if self._refs[idx] == 0:
                if keep_cached:
                    self._lru[idx] = None
                else:
                    self._free_mask |= 1 << idx

    def free_many(self, idxs: List[int], keep_cached_for=None):
        """Release many pages under one lock acquisition.

        ``keep_cached_for`` is a predicate over the page index: pages it
        accepts stay in the LRU (prefix-cache hits), the rest return to the
        free mask.
        """
        if not idxs:
            return
        keep = keep_cached_for or (lambda idx: False)
        with self._lock:
            for idx in idxs:
                self._refs[idx] -= 1
                if self._refs[idx] == 0:
                    if keep(idx):
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

    def clear_cached(self) -> int:
        """Release every unreferenced LRU page back to the free pool."""
        with self._lock:
            cached = list(self._lru)
            self._lru.clear()
            for idx in cached:
                if self._refs[idx] != 0:
                    raise RuntimeError("Cannot invalidate a referenced cache page")
                if self.on_evict:
                    self.on_evict(idx)
                self._free_mask |= 1 << idx
            return len(cached)


class RadixNode:
    """A page-aligned edge in the CPU-side prefix radix trie."""

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


class AllocationStrategy(ABC):
    """Physical slot allocation policy.

    Subclasses implement the actual allocation semantics.  This ABC declares
    the contract; there are no default implementations.
    """

    @abstractmethod
    def alloc(self, state: TaskCacheState, prompt_ids: List[int]) -> bool: ...

    @abstractmethod
    def free(self, state: TaskCacheState) -> None: ...

    @abstractmethod
    def extend(self, state: TaskCacheState, pos: int) -> bool: ...

    @abstractmethod
    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None: ...

    @abstractmethod
    def record_hashes(
        self,
        state: TaskCacheState,
        prompt_ids: List[int],
        start: int,
    ) -> None: ...

    def flush_slots(self, states: List[TaskCacheState], device) -> None:
        """Push host-staged slot maps to the device row.

        Only the paged strategy stages slots on the host; the default is a
        no-op for strategies whose req_to_token rows are already current
        (contiguous rows are pre-filled and never rewritten).
        """

    def invalidate_cache(self) -> int:
        """Drop reusable KV entries after an inference weight update."""
        return 0


class ContiguousStrategy(AllocationStrategy):
    """Static contiguous allocation: slots are pre-assigned at pool init.

    No dynamic allocation or prefix caching.  All operations are no-ops
    because ``ReqToTokenPool`` is pre-filled with contiguous ranges.
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
        self,
        state: TaskCacheState,
        prompt_ids: List[int],
        start: int,
    ) -> None:
        pass


class PagedStrategy(AllocationStrategy):
    """Dynamic paged allocation from a shared bitmask pool.

    ``page_size`` is a parameter, not a separate strategy: at ``page_size=1``
    each allocated page *is* one token slot (``page * 1 + 0``), and prefix
    caching is simply disabled (``prefix=None``).  The unified page formula
    ``pages[page_idx] * page_size + offset`` holds for both.
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
        new_pages = self._alloc.alloc_many(n_new)
        if new_pages is None:
            return False
        state.pages.extend(new_pages)
        return True

    def free(self, state: TaskCacheState) -> None:
        if self._prefix is not None:
            self._alloc.free_many(state.pages, keep_cached_for=self._prefix.has_page)
            for p in state.pages:
                if not self._prefix.has_page(p):
                    self._prefix.evict(p)
        else:
            self._alloc.free_many(state.pages)

    def extend(self, state: TaskCacheState, pos: int) -> bool:
        page_idx = pos // self._page_size
        if page_idx >= len(state.pages):
            p = self._alloc.alloc()
            if p < 0:
                return False
            state.pages.append(p)
        slot = state.pages[page_idx] * self._page_size + pos % self._page_size
        if self._page_size == 1 and pos == len(state._slots):
            # page_size=1 fast path: positions arrive in order, so the slot
            # list extends by one element instead of each call issuing a
            # single-element H2D copy to req_to_token (128 such copies per
            # decode step at serving batch sizes). _slots is flushed to the
            # device in bulk by write_indices/bind.
            state._slots.append(slot)
            return True
        self._req_pool.req_to_token[state.req_idx, pos] = slot
        return True

    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None:
        total = min(len(prompt_ids), len(state.pages) * self._page_size)
        if total <= 0:
            return
        # One H2D write per row: every per-position assignment is a separate
        # tiny copy (~15us of driver overhead each), so a 512-token prompt
        # spends ~8ms in the loop while the same row written once as a tensor
        # costs ~60us. The position -> slot map is an affine function of the
        # page ids, built on the host in bulk either way.
        page_size = self._page_size
        pages = state.pages
        if page_size == 1:
            vals = pages[:total]
        else:
            vals = []
            for page_idx, p in enumerate(pages[: (total + page_size - 1) // page_size]):
                base = p * page_size
                n = min(page_size, total - page_idx * page_size)
                vals.extend(range(base, base + n))
        self._req_pool.req_to_token[state.req_idx, :total] = torch.tensor(
            vals, dtype=torch.int32, device=self._device
        )
        state._slots = list(vals)
        state._flushed = total

    def flush_slots(self, states: List[TaskCacheState], device) -> None:
        # One tiny H2D copy per dirty TAIL per step, all appended after the
        # bind's device-side work is enqueued in stream order — steady-state
        # decode also gathers req_to_token rows on-device (out_cache_loc), so
        # the new positions must land there even on the incremental path.
        # Writing only the unflushed tail keeps the per-step volume at one
        # int per decoded token instead of re-sending the whole row.
        for state in states:
            slots = state._slots
            n = len(slots)
            if n <= state._flushed:
                continue
            start = state._flushed
            state._flushed = n
            row = self._req_pool.req_to_token[state.req_idx]
            row[start:n] = torch.tensor(
                slots[start:], dtype=torch.int32, device=self._device
            )

    def record_hashes(
        self,
        state: TaskCacheState,
        prompt_ids: List[int],
        start: int,
    ) -> None:
        if self._prefix is None:
            return
        full = len(prompt_ids) // self._page_size
        for i in range(start, min(full, len(state.pages))):
            self._prefix.record(state.pages[i], prompt_ids, i)

    def invalidate_cache(self) -> int:
        if self._prefix is None:
            return 0
        return self._alloc.clear_cached()
