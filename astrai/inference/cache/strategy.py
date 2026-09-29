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
    """Bitmask-based page allocator with ref-counting and LRU eviction.

    The free set is a Python big-int bitmask (kept as the source of truth:
    ``_free_mask`` is read by tests and debug tooling). Scanning it
    directly costs a full limb-chain walk per extracted page on
    serving-scale pools, so the fast paths go through a shadow index of
    per-64-page words (``_words``) plus a cursor to the lowest non-empty
    word. Allocation only touches the handful of words it consumes;
    freeing sets one bit and pulls the cursor back. The two structures
    are updated together under the same lock, and ``_words`` can be
    rebuilt from the mask at any time (see ``_sync_words``).
    """

    _WORD_BITS = 64

    def __init__(self, n_pages: int):
        self._free_mask = (1 << n_pages) - 1
        self._n_pages = n_pages
        self._refs: List[int] = [0] * n_pages
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.on_evict: Optional[Callable[[int], None]] = None
        self._lock = threading.Lock()
        self._sync_words()

    def _sync_words(self):
        """Rebuild the word index from ``_free_mask`` (caller holds or
        owns exclusive access, e.g. at init or after mask surgery)."""
        n_words = (self._n_pages + self._WORD_BITS - 1) // self._WORD_BITS
        self._words: List[int] = [0] * n_words
        m = self._free_mask
        w = 0
        while m:
            self._words[w] = m & 0xFFFFFFFFFFFFFFFF
            m >>= self._WORD_BITS
            w += 1
        self._first_nonempty = 0
        while self._first_nonempty < n_words and self._words[self._first_nonempty] == 0:
            self._first_nonempty += 1

    def _mask_alloc_bits(self, pages: List[int]):
        # One aggregated mask subtraction for the whole batch: per-page
        # big-int clears cost a limb walk each, which is what the word
        # index exists to avoid.
        clear = 0
        for idx in pages:
            clear |= 1 << idx
            w = idx >> 6
            self._words[w] &= ~(1 << (idx & 63))
        self._free_mask &= ~clear
        # Advance the cursor past every word the batch emptied, in order.
        for w in range(self._first_nonempty, len(self._words)):
            if self._words[w] == 0:
                self._first_nonempty = w + 1
            else:
                break

    def _mask_free_bit(self, idx: int):
        bit = 1 << idx
        self._free_mask |= bit
        w = idx >> 6
        self._words[w] |= 1 << (idx & 63)
        if w < self._first_nonempty:
            self._first_nonempty = w

    def alloc(self) -> int:
        with self._lock:
            if self._free_mask:
                lsb = self._free_mask & -self._free_mask
                idx = lsb.bit_length() - 1
                self._mask_alloc_bits([idx])
                self._refs[idx] = 1
                return idx
            if self._lru:
                idx, _ = self._lru.popitem(last=False)
                if self.on_evict:
                    self.on_evict(idx)
                self._refs[idx] = 1
                # LRU promotion: the bit was set by the caller before or
                # during eviction; clear it in both structures now.
                self._free_mask &= ~(1 << idx)
                w = idx >> 6
                self._words[w] &= ~(1 << (idx & 63))
                return idx
            return -1

    def alloc_many(self, n: int) -> Optional[List[int]]:
        """Allocate exactly ``n`` pages, or ``None`` if fewer are free.

        One lock acquisition per prompt. Free pages come from the low
        words of the mask in bulk; only when the free set cannot satisfy
        the request does it promote LRU pages (page-by-page, evicting as
        it goes).
        """
        if n <= 0:
            return []
        with self._lock:
            free = self._free_mask.bit_count()
            if free < n:
                promoted = 0
                while free + promoted < n and self._lru:
                    idx, _ = self._lru.popitem(last=False)
                    if self.on_evict:
                        self.on_evict(idx)
                    self._mask_free_bit(idx)
                    promoted += 1
                if free + promoted < n:
                    return None
            # Word-index harvest: only the words actually consumed are
            # touched, so a fragmented pool costs the skipped empty words
            # (cursor advance) instead of a big-int limb walk per page.
            out: List[int] = []
            w = self._first_nonempty
            words = self._words
            while len(out) < n and w < len(words):
                word = words[w]
                if word == 0:
                    w += 1
                    continue
                base = w * self._WORD_BITS
                while word and len(out) < n:
                    lsb = word & -word
                    out.append(base + lsb.bit_length() - 1)
                    word ^= lsb
                words[w] = word
                w += 1
            self._mask_alloc_bits(out)
            for idx in out:
                self._refs[idx] = 1
            return out

    def free(self, idx: int, keep_cached: bool = False):
        with self._lock:
            self._refs[idx] -= 1
            if self._refs[idx] == 0:
                if keep_cached:
                    self._lru[idx] = None
                else:
                    self._mask_free_bit(idx)

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
            # Set-latch pass first (refs may drop to zero multiple times in
            # one batch; the freed bits are aggregated into ONE big-int
            # insertion and the word index updated per page — the index is
            # what the fast paths read, the mask follows in bulk).
            freed_bits = 0
            lowest_word = len(self._words)
            for idx in idxs:
                self._refs[idx] -= 1
                if self._refs[idx] == 0:
                    if keep(idx):
                        self._lru[idx] = None
                    else:
                        freed_bits |= 1 << idx
                        w = idx >> 6
                        self._words[w] |= 1 << (idx & 63)
                        if w < lowest_word:
                            lowest_word = w
            if freed_bits:
                self._free_mask |= freed_bits
                if lowest_word < self._first_nonempty:
                    self._first_nonempty = lowest_word

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
                self._mask_free_bit(idx)
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
        # Pinned staging for the per-step flush of host-staged slot tails:
        # synchronous ``torch.tensor(..., device=)`` construction costs a
        # blocking H2D per tensor per decode step; staging into pre-pinned
        # buffers keeps those transfers asynchronous.  The ring depth of 2
        # lets step t+1 write its staging while step t's transfer may still
        # be in flight (same stream, so ordering is preserved either way).
        # Pinned allocation needs a CUDA driver — CPU-only environments
        # (CI) stage through pageable memory instead (the flush then just
        # keeps its old blocking-copy behaviour).
        max_batch = req_pool.req_to_token.shape[0]
        pin = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self._flush_pin = [
            (
                torch.empty(max_batch, dtype=torch.int64, pin_memory=pin),
                torch.empty(max_batch, dtype=torch.int64, pin_memory=pin),
                torch.empty(max_batch, dtype=torch.int32, pin_memory=pin),
            )
            for _ in range(2)
        ]
        self._flush_ring = 0

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
        # req_to_token rows are gathered on-device by every decode bind
        # (out_cache_loc), so staged tails must land there before the
        # gather runs — steady incremental steps included. The staged tails
        # of the whole batch are concatenated into ONE tensor write (one
        # H2D per step instead of one per task; 128 per-task copies cost
        # ~5ms/step at serving batch sizes) using a built-once scatter
        # index (task row, in-row offset) pairs.
        flat_vals: List[int] = []
        row_idx: List[int] = []
        col_idx: List[int] = []
        for state in states:
            slots = state._slots
            n = len(slots)
            if n <= state._flushed:
                continue
            start = state._flushed
            state._flushed = n
            flat_vals.extend(slots[start:])
            req = state.req_idx
            row_idx.extend([req] * (n - start))
            col_idx.extend(range(start, n))
        if not flat_vals:
            return
        # Depth-2 pinned staging: writes go into the free ring slot, then
        # a single non_blocking H2D per tensor replaces three blocking
        # ``torch.tensor(..., device=)`` constructions.  Depth 2 (vs a
        # single buffer) keeps the host writable while the previous
        # transfer may still be queued — the caller (bind) runs strictly
        # step-serial, but the same stream executes the copies in order
        # regardless.
        pin_rows, pin_cols, pin_vals = self._flush_pin[self._flush_ring]
        self._flush_ring ^= 1
        n = len(flat_vals)
        pin_rows[:n] = torch.as_tensor(row_idx, dtype=torch.int64)
        pin_cols[:n] = torch.as_tensor(col_idx, dtype=torch.int64)
        pin_vals[:n] = torch.as_tensor(flat_vals, dtype=torch.int32)
        rows = pin_rows[:n].to(self._device, non_blocking=True)
        cols = pin_cols[:n].to(self._device, non_blocking=True)
        vals = pin_vals[:n].to(self._device, non_blocking=True)
        self._req_pool.req_to_token[rows, cols] = vals

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
