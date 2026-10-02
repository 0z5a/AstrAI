"""Unit tests for inference cache components."""

import random

import pytest
import torch

from astrai.inference.core.cache import (
    Allocator,
    BlockPool,
    KVCacheManager,
    KVStorage,
    RadixCache,
    ReqToTokenPool,
)
from astrai.inference.worker.workspace import InferenceWorkspace


def _ws(pool: BlockPool) -> InferenceWorkspace:
    """Workspace sized to the pool (bind_tasks requires it)."""
    return InferenceWorkspace(
        pool.max_batch_size,
        pool.max_seq_len,
        max_q_heads=2,
        head_dim=4,
        device=pool.device,
        dtype=pool.dtype,
    )


def _make_task_cache(pool: BlockPool) -> KVCacheManager:
    return KVCacheManager(pool)


@pytest.fixture(params=["cpu", "cuda"])
def kv_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _assert_prefix_reachable(prefix):
    reachable = {}
    pending = list(prefix._root.children.values())
    while pending:
        node = pending.pop()
        assert node.page_idx is not None
        assert node.parent.children[node.tokens] is node
        assert node.page_idx not in reachable
        reachable[node.page_idx] = node
        pending.extend(node.children.values())
    assert prefix._page_to_node == reachable


def _assert_allocator_cache_consistent(alloc, prefix):
    _assert_prefix_reachable(prefix)
    for page in range(alloc._n_pages):
        cached = prefix.has_page(page)
        free = bool(alloc._free_mask & (1 << page))
        assert free == (alloc._refs[page] == 0 and not cached)
        assert (page in alloc._lru) == (alloc._refs[page] == 0 and cached)
    assert alloc._free_mask == sum(
        word << (64 * i) for i, word in enumerate(alloc._words)
    )


# ---- Allocator ----


def test_allocator_alloc_free_cycle():
    alloc = Allocator(4)
    a = alloc.alloc()
    b = alloc.alloc()
    assert a != b
    alloc.free(a)
    alloc.free(b)
    c = alloc.alloc()
    assert c in (a, b)


def test_allocator_alloc_when_full():
    alloc = Allocator(2)
    alloc.alloc()
    alloc.alloc()
    assert alloc.alloc() == -1


def test_allocator_lru_eviction():
    alloc = Allocator(2)
    p0 = alloc.alloc()
    p1 = alloc.alloc()
    alloc.free(p0, keep_cached=True)
    alloc.free(p1, keep_cached=True)
    alloc.alloc()
    assert p0 in alloc._lru or p1 in alloc._lru


def test_allocator_inc_ref_and_free():
    alloc = Allocator(2)
    p = alloc.alloc()
    alloc.inc_ref(p)
    assert alloc._refs[p] == 2
    alloc.free(p)
    assert alloc._refs[p] == 1
    alloc.free(p)
    assert alloc._refs[p] == 0


# ---- RadixCache ----


def test_prefix_cache_lookup_returns_hits():
    token_ids = list(range(256))
    prefix = RadixCache(64)
    pages = [0, 1, 2, 3]
    for i, p in enumerate(pages):
        prefix.record(p, token_ids, i)
    hits = prefix.lookup(token_ids)
    assert hits == pages


def test_prefix_cache_lookup_stops_at_first_miss():
    token_ids = list(range(256))
    prefix = RadixCache(64)
    prefix.record(0, token_ids, 0)
    prefix.record(1, [99] * 64, 1)
    hits = prefix.lookup(token_ids)
    assert len(hits) == 1
    assert hits[0] == 0


def test_prefix_cache_ignores_partial_last_page():
    token_ids = list(range(100))
    prefix = RadixCache(64)
    prefix.record(0, token_ids, 0)
    hits = prefix.lookup(token_ids)
    assert len(hits) == 1


def test_prefix_cache_on_evict_clears_mappings():
    prefix = RadixCache(64)
    assert not prefix.has_page(0)
    prefix.record(0, list(range(64)), 0)
    assert prefix.has_page(0)
    prefix.evict(0)
    assert not prefix.has_page(0)


def test_prefix_cache_does_not_reuse_page_without_parent_prefix():
    prefix = RadixCache(2)
    prefix.record(0, [1, 2, 3, 4], 0)
    prefix.record(1, [1, 2, 3, 4, 5, 6], 1)
    prefix.record(2, [9, 10, 5, 6], 0)
    prefix.record(3, [9, 10, 5, 6, 7, 8], 1)
    assert prefix.lookup([1, 2, 3, 4, 5, 6]) == [0, 1]
    assert prefix.lookup([9, 10, 5, 6, 7, 8]) == [2, 3]


def test_prefix_cache_shares_branch_prefix():
    prefix = RadixCache(2)
    prefix.record(0, [1, 2, 3, 4], 0)
    prefix.record(1, [1, 2, 3, 4], 1)
    prefix.record(2, [1, 2, 7, 8], 1)
    assert prefix.lookup([1, 2, 3, 4]) == [0, 1]
    assert prefix.lookup([1, 2, 7, 8]) == [0, 2]
    prefix.evict(1)
    assert prefix.lookup([1, 2, 3, 4]) == [0]
    assert prefix.lookup([1, 2, 7, 8]) == [0, 2]


def test_prefix_cache_does_not_record_partial_page():
    prefix = RadixCache(4)
    prefix.record(0, [1, 2, 3, 4, 5, 6], 0)
    prefix.record(1, [1, 2, 3, 4, 5, 6], 1)
    assert prefix.lookup([1, 2, 3, 4, 5, 6]) == [0]

    prefix.record(1, [1, 2, 3, 4, 5, 6, 7, 8], 1)
    assert prefix.lookup([1, 2, 3, 4, 5, 6, 7, 8]) == [0, 1]


def test_prefix_cache_repeated_record_preserves_descendants_and_branches():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)
    branch = [1, 2, 7, 8]
    prefix.record(3, branch, 1)
    original_nodes = dict(prefix._page_to_node)

    for _ in range(3):
        for i in range(3):
            assert prefix.record(i, prompt, i) == []
        assert prefix.lookup(prompt) == [0, 1, 2]
        assert prefix.lookup(branch) == [0, 3]
        assert prefix._page_to_node == original_nodes
        _assert_prefix_reachable(prefix)


@pytest.mark.parametrize("victim, revoked", [(0, {0, 1, 2, 3}), (1, {1, 2})])
def test_prefix_cache_ancestor_eviction_revokes_only_its_subtree(victim, revoked):
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)
    prefix.record(3, [1, 2, 7, 8], 1)
    prefix.record(4, [9, 10], 0)

    assert set(prefix.evict(victim)) == revoked
    assert prefix.evict(victim) == []
    assert prefix.lookup(prompt) == ([] if victim == 0 else [0])
    assert prefix.lookup([1, 2, 7, 8]) == ([] if victim == 0 else [0, 3])
    assert prefix.lookup([9, 10]) == [4]
    for page in range(5):
        assert prefix.has_page(page) == (page not in revoked)
    _assert_prefix_reachable(prefix)


def test_prefix_cache_relocating_physical_page_revokes_old_descendants():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)

    assert set(prefix.record(0, [9, 10], 0)) == {1, 2}
    assert prefix.lookup(prompt) == []
    assert prefix.lookup([9, 10]) == [0]
    assert not prefix.has_page(1)
    assert not prefix.has_page(2)
    _assert_prefix_reachable(prefix)


def test_prefix_cache_replacing_same_prefix_preserves_descendants():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)

    assert prefix.record(3, prompt, 0) == [0]
    assert prefix.lookup(prompt) == [3, 1, 2]
    assert not prefix.has_page(0)
    _assert_prefix_reachable(prefix)
    assert set(prefix.evict(3)) == {1, 2, 3}
    assert not prefix._page_to_node


def test_prefix_cache_does_not_index_unreachable_descendant():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4]
    assert prefix.record(1, prompt, 1) == []
    assert not prefix.has_page(1)
    prefix.record(0, prompt, 0)
    assert prefix.lookup(prompt) == [0]
    prefix.record(1, prompt, 1)
    assert prefix.lookup(prompt) == [0, 1]
    _assert_prefix_reachable(prefix)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("hold_descendant", [False, True])
def test_allocator_ancestor_eviction_reclaims_only_unreferenced_pages(
    batched, hold_descendant
):
    alloc = Allocator(4)
    prefix = RadixCache(2)
    alloc.on_evict = prefix.evict
    assert alloc.alloc_many(4) == [0, 1, 2, 3]
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)
    prefix.record(3, [9, 10], 0)
    released = [0, 2, 3] if hold_descendant else [0, 1, 2, 3]
    alloc.free_many(released, keep_cached_for=prefix.has_page)
    _assert_allocator_cache_consistent(alloc, prefix)

    taken = alloc.alloc_many(2) if batched else [alloc.alloc()]
    expected = [0, 2 if hold_descendant else 1] if batched else [0]
    assert taken == expected
    assert prefix.lookup(prompt) == []
    assert prefix.lookup([9, 10]) == [3]
    assert list(alloc._lru) == [3]
    if hold_descendant:
        assert alloc.ref_count(1) == 1
        assert not alloc._free_mask & (1 << 1)
        alloc.free(1, keep_cached=prefix.has_page(1))
    _assert_allocator_cache_consistent(alloc, prefix)
    assert alloc.clear_cached() == 1
    alloc.free_many(taken)
    assert alloc._free_mask == (1 << 4) - 1
    _assert_allocator_cache_consistent(alloc, prefix)


def test_allocator_failed_bulk_allocation_does_not_revoke_prefixes():
    alloc = Allocator(2)
    prefix = RadixCache(2)
    alloc.on_evict = prefix.evict
    assert alloc.alloc_many(2) == [0, 1]
    prefix.record(0, [1, 2], 0)
    alloc.free(0, keep_cached=True)

    assert alloc.alloc_many(2) is None
    assert prefix.lookup([1, 2]) == [0]
    _assert_allocator_cache_consistent(alloc, prefix)


# ---- ReqToTokenPool ----


def test_req_to_token_pool_alloc_free():
    pool = ReqToTokenPool(4, 128, torch.device("cpu"))
    assert pool.req_to_token.dtype == torch.int32
    slots = pool.alloc(2)
    assert len(slots) == 2
    assert len(pool.free_slots) == 2
    pool.free(slots)
    assert len(pool.free_slots) == 4


def test_req_to_token_pool_alloc_when_full():
    pool = ReqToTokenPool(2, 128, torch.device("cpu"))
    pool.alloc(2)
    assert pool.alloc(1) is None


def test_req_to_token_pool_write():
    pool = ReqToTokenPool(4, 128, torch.device("cpu"))
    slots = pool.alloc(1)
    pool.write((slots[0], slice(0, 3)), torch.tensor([10, 20, 30]))
    assert pool.req_to_token[slots[0], 0].item() == 10
    assert pool.req_to_token[slots[0], 2].item() == 30


# ---- KVStorage ----


def test_kv_storage_buffer_shape():
    storage = KVStorage(
        size=32,
        n_layers=3,
        n_kv_heads=8,
        head_dim=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert storage.k_buffer.shape == (3, 32, 8, 16)
    assert storage.v_buffer.shape == (3, 32, 8, 16)


# ---- BlockPool (contiguous mode) ----


def _make_contiguous_pool(**kwargs):
    defaults = dict(
        n_layers=2,
        n_kv_heads=4,
        head_dim=8,
        max_batch_size=4,
        max_seq_len=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    defaults.update(kwargs)
    return BlockPool(**defaults)


def test_page_pool_contiguous_task_alloc_free():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    assert task_cache.alloc_slots("t1", [1, 2, 3])
    assert "t1" in task_cache._states
    task_cache.free_slots("t1")
    assert "t1" not in task_cache._states


def test_page_pool_contiguous_task_extend():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", [1, 2, 3])
    assert task_cache.extend_slots("t1", 3)
    assert task_cache.extend_slots("t1", 63)
    assert not task_cache.extend_slots("t1", 64)


def test_page_pool_contiguous_task_cached():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", [1, 2, 3])
    assert task_cache.cached_tokens("t1") == 0


def test_page_pool_contiguous_bind_tasks_prefill():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(10)))
    task_cache.alloc_slots("t2", list(range(10)))
    kv = task_cache.bind(["t1", "t2"], _ws(pool), start_pos=0)
    assert kv.out_cache_loc.shape == (20,)
    assert kv.out_cache_loc.dtype == torch.int32
    assert kv.seq_lens.tolist() == [10, 10]
    assert kv.req_pool_indices.shape == (2,)
    assert kv.req_pool_indices.dtype == torch.int32


def test_page_pool_bind_tasks_builds_compact_q_tile_mapping():
    pool = _make_contiguous_pool(max_batch_size=3, max_seq_len=256)

    kv = pool.bind_tasks([0, 1, 2], [70, 10, 130], _ws(pool), start_pos=0)

    assert kv.qo_indptr.tolist() == [0, 70, 80, 210]
    assert kv.q_tile_to_batch.tolist() == [0, 0, 1, 2, 2, 2]
    assert kv.q_tile_to_index.tolist() == [0, 1, 0, 0, 1, 2]


def test_page_pool_contiguous_bind_tasks_decode():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(10)))
    task_cache.alloc_slots("t2", list(range(8)))
    # Simulate one decode extension so seq_lens advance to 11 and 9.
    assert task_cache.extend_slots("t1", 10)
    assert task_cache.extend_slots("t2", 8)
    kv = task_cache.bind(["t1", "t2"], _ws(pool))
    assert kv.out_cache_loc.shape == (2,)
    assert kv.seq_lens.tolist() == [11, 9]


def test_page_pool_contiguous_bind_roundtrip():
    """Write KV via bind_tasks, then gather via req_to_token indexing."""
    pool = _make_contiguous_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(4)))

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0, indices]
    gathered_v = kv.v_buffer[0, indices]
    assert torch.allclose(gathered_k, k)
    assert torch.allclose(gathered_v, v)


# ---- BlockPool (paged mode, page_size=1) ----


def _make_paged_pool(**kwargs):
    defaults = dict(
        n_layers=1,
        n_kv_heads=2,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        page_size=1,
        n_tokens=128,
    )
    defaults.update(kwargs)
    return BlockPool(**defaults)


def test_page_pool_paged_task_alloc():
    pool = _make_paged_pool()
    task_cache = _make_task_cache(pool)
    assert task_cache.alloc_slots("t1", list(range(10)))
    state = task_cache._states["t1"]
    assert len(state.pages) == 10
    assert pool.req_pool.req_to_token[state.req_idx, 0].item() == state.pages[0]


def test_page_pool_paged_task_extend():
    pool = _make_paged_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(4)))
    assert task_cache.extend_slots("t1", 4)
    req_idx = task_cache._states["t1"].req_idx
    slot = pool.req_pool.req_to_token[req_idx, 4].item()
    assert slot >= 0


def test_page_pool_paged_task_free_releases_slots():
    pool = _make_paged_pool(n_tokens=16)
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(8)))
    task_cache.free_slots("t1")
    assert "t1" not in task_cache._states
    assert len(pool.req_pool.free_slots) == 4


def test_page_pool_paged_bind_roundtrip():
    pool = _make_paged_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(4)))

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0, indices]
    assert torch.allclose(gathered_k, k)


# ---- BlockPool (paged mode, page_size>1) ----


def _make_paged_pool_ps64(**kwargs):
    defaults = dict(
        n_layers=1,
        n_kv_heads=2,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=256,
        device=torch.device("cpu"),
        dtype=torch.float32,
        page_size=64,
        n_tokens=512,
    )
    defaults.update(kwargs)
    return BlockPool(**defaults)


def test_page_pool_paged_ps64_task_alloc():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    prompt = list(range(200))
    assert task_cache.alloc_slots("t1", prompt)
    assert task_cache.cached_tokens("t1") == 0
    n_pages = (200 + 63) // 64
    assert len(task_cache._states["t1"].pages) == n_pages


def test_page_pool_paged_ps64_task_extend_crosses_page():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(64)))
    assert task_cache.extend_slots("t1", 64)
    assert len(task_cache._states["t1"].pages) >= 2


def test_page_pool_prefix_hit_populates_request_mapping():
    pool = _make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = _make_task_cache(pool)
    prompt = [11, 12, 13, 14]

    assert task_cache.alloc_slots("first", prompt)
    task_cache.record_block_hashes("first", prompt, materialized_end=len(prompt))
    task_cache.free_slots("first")

    assert task_cache.alloc_slots("second", prompt)
    second_state = task_cache._states["second"]
    expected = [
        page * pool.page_size + offset
        for page in second_state.pages
        for offset in range(pool.page_size)
    ]

    assert second_state.cached == len(prompt)
    assert (
        pool.req_pool.req_to_token[second_state.req_idx, : len(prompt)].tolist()
        == expected
    )


def test_task_cache_invalidation_drops_cross_version_prefix_hits():
    pool = _make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = _make_task_cache(pool)
    prompt = [11, 12, 13, 14]

    assert task_cache.alloc_slots("first", prompt)
    task_cache.record_block_hashes("first", prompt, materialized_end=len(prompt))
    task_cache.free_slots("first")
    assert task_cache.alloc_slots("cached", prompt)
    assert task_cache.cached_tokens("cached") == len(prompt)

    with pytest.raises(RuntimeError, match="while requests are active"):
        task_cache.invalidate_cache()

    task_cache.free_slots("cached")
    assert task_cache.invalidate_cache() == 2
    assert task_cache.alloc_slots("after_update", prompt)
    assert task_cache.cached_tokens("after_update") == 0


def test_record_block_hashes_requires_explicit_materialized_end():
    pool = _make_paged_pool(page_size=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(8))
    assert task_cache.alloc_slots("writer", prompt)

    with pytest.raises(TypeError, match="materialized_end"):
        task_cache.record_block_hashes("writer", prompt)
    assert pool.strategy._prefix.lookup(prompt) == []


@pytest.mark.parametrize("allocated, ids_len, end", [(8, 6, 8), (4, 8, 8)])
def test_record_block_hashes_is_bounded_by_token_ids_and_allocated_pages(
    allocated, ids_len, end
):
    pool = _make_paged_pool(page_size=4)
    task_cache = _make_task_cache(pool)
    assert task_cache.alloc_slots("writer", list(range(allocated)))
    task_cache.record_block_hashes("writer", list(range(ids_len)), materialized_end=end)
    assert pool.strategy._prefix.lookup(list(range(8))) == [
        task_cache._states["writer"].pages[0]
    ]


@pytest.mark.parametrize("start, end", [(-1, 4), (0, -1)])
def test_record_block_hashes_rejects_negative_watermark_or_page(start, end):
    pool = _make_paged_pool(page_size=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(8))
    assert task_cache.alloc_slots("writer", prompt)
    with pytest.raises(ValueError, match="nonnegative"):
        task_cache.record_block_hashes("writer", prompt, start, materialized_end=end)
    assert pool.strategy._prefix.lookup(prompt) == []


def test_chunked_prefix_only_reuses_fully_written_kv(kv_device):
    pool = _make_paged_pool(page_size=4, max_seq_len=16, n_tokens=64, device=kv_device)
    task_cache = _make_task_cache(pool)
    ws = _ws(pool)
    prompt = list(range(8))
    assert task_cache.alloc_slots("writer", prompt)
    writer = task_cache._states["writer"]
    pool._storage.k_buffer.fill_(float("nan"))
    pool._storage.v_buffer.fill_(float("nan"))
    expected_k = torch.arange(64, dtype=pool.dtype, device=kv_device).reshape(8, 2, 4)
    expected_v = expected_k + 1000

    for start, end in [(0, 2), (2, 4), (4, 6), (6, 8)]:
        kv = task_cache.bind(["writer"], ws, start_pos=start, seq_ends=[end])
        assert kv.out_cache_loc.numel() == end - start
        kv.k_buffer[0, kv.out_cache_loc] = expected_k[start:end]
        kv.v_buffer[0, kv.out_cache_loc] = expected_v[start:end]
        task_cache.record_block_hashes(
            "writer", prompt, start // pool.page_size, materialized_end=end
        )

        assert task_cache.alloc_slots("reader", prompt)
        reader = task_cache._states["reader"]
        n_cached = end // pool.page_size * pool.page_size
        n_pages = n_cached // pool.page_size
        assert reader.cached == n_cached
        assert reader.pages[:n_pages] == writer.pages[:n_pages]
        assert set(reader.pages[n_pages:]).isdisjoint(writer.pages[n_pages:])
        cached_slots = pool.req_pool.req_to_token[reader.req_idx, :n_cached]
        torch.testing.assert_close(kv.k_buffer[0, cached_slots], expected_k[:n_cached])
        torch.testing.assert_close(kv.v_buffer[0, cached_slots], expected_v[:n_cached])
        unwritten = pool.req_pool.req_to_token[writer.req_idx, end : len(prompt)]
        assert torch.isnan(kv.k_buffer[0, unwritten]).all().item()
        assert torch.isnan(kv.v_buffer[0, unwritten]).all().item()
        task_cache.free_slots("reader")
        _assert_allocator_cache_consistent(pool.strategy._alloc, pool.strategy._prefix)

    task_cache.free_slots("writer")
    assert task_cache.invalidate_cache() == 2
    _assert_allocator_cache_consistent(pool.strategy._alloc, pool.strategy._prefix)


@pytest.mark.parametrize("release_first", [False, True])
def test_prefix_replacement_keeps_allocator_identity_consistent(release_first):
    pool = _make_paged_pool(page_size=4, max_seq_len=16, n_tokens=32)
    task_cache = _make_task_cache(pool)
    prompt = list(range(8))
    # Both prefills reserve private pages before either publishes its KV.
    assert task_cache.alloc_slots("first", prompt)
    assert task_cache.alloc_slots("second", prompt)
    first_pages = list(task_cache._states["first"].pages)
    second_pages = list(task_cache._states["second"].pages)
    task_cache.record_block_hashes("first", prompt, materialized_end=8)
    if release_first:
        task_cache.free_slots("first")

    # Replacing the first page leaves the already-materialized descendant
    # reusable, but drops the old first page's cached allocator identity.
    task_cache.record_block_hashes("second", prompt, materialized_end=4)
    prefix = pool.strategy._prefix
    alloc = pool.strategy._alloc
    assert prefix.lookup(prompt) == [second_pages[0], first_pages[1]]
    assert not prefix.has_page(first_pages[0])
    assert alloc.ref_count(first_pages[0]) == (0 if release_first else 1)
    _assert_allocator_cache_consistent(alloc, prefix)

    task_cache.record_block_hashes("second", prompt, materialized_end=8)
    assert prefix.lookup(prompt) == second_pages
    assert all(not prefix.has_page(page) for page in first_pages)
    task_cache.free_slots("first")
    assert all(alloc._free_mask & (1 << page) for page in first_pages)
    task_cache.free_slots("second")
    _assert_allocator_cache_consistent(alloc, prefix)
    assert task_cache.alloc_slots("reader", prompt)
    assert task_cache.cached_tokens("reader") == len(prompt)
    assert task_cache._states["reader"].pages == second_pages
    _assert_allocator_cache_consistent(alloc, prefix)


def test_page_pool_paged_ps64_bind_roundtrip():
    pool = _make_paged_pool_ps64(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(128))
    task_cache.alloc_slots("t1", prompt)

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 128, 2, 4)
    v = torch.randn(1, 128, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :128]
    gathered_k = kv.k_buffer[0, indices]
    assert torch.allclose(gathered_k, k)


def test_page_pool_paged_steady_decode_slots_reach_device():
    """The extend fast path stages slot ids on the host; every bind (the
    steady incremental decode path included) gathers req_to_token rows
    on-device, so the staged tails must land there before the gather."""
    pool = _make_paged_pool(n_tokens=64, max_seq_len=16)
    task_cache = _make_task_cache(pool)
    ws = _ws(pool)
    prompt = list(range(4))
    assert task_cache.alloc_slots("t1", prompt)

    # Prefill bind flushes the whole staged prefix.
    task_cache.bind(["t1"], ws, start_pos=0)
    state = task_cache._states["t1"]

    # Decode steps: extend stages one slot per token, bind (incremental or
    # not) must push it to the device row.
    for pos in range(4, 10):
        assert task_cache.extend_slots("t1", pos)
        task_cache.bind(["t1"], ws)
        device_row = pool.req_pool.req_to_token[state.req_idx, : pos + 1].tolist()
        expected = [p * pool.page_size for p in state.pages[: pos + 1]]
        assert device_row == expected


def test_allocator_alloc_many_matches_free_set_exactly():
    """Bulk allocation must yield exactly the pages the free set held.

    The word-window harvest once re-issued already-harvested pages (stale
    mask read across windows) and once left cleared pages marked free
    (mixed absolute/shifted bit coordinates); this randomized
    reference-model check pins the mask == free-set invariant.
    """
    rng = random.Random(7)
    alloc = Allocator(300)
    free = set(range(300))
    held = []
    for _ in range(400):
        if free and rng.random() < 0.55:
            n = rng.randint(1, 25)
            got = alloc.alloc_many(n)
            if got is None:
                assert len(free) < n
                continue
            assert len(got) == n
            assert len(set(got)) == n
            for p in got:
                assert p in free
                free.remove(p)
            held.append(got)
        elif held:
            g = held.pop(rng.randrange(len(held)))
            alloc.free_many(g)
            free.update(g)
        mask = alloc._free_mask
        for p in range(300):
            assert bool(mask >> p & 1) == (p in free)


def test_allocator_alloc_many_fragmentation_roundtrip():
    alloc = Allocator(10000)
    first = alloc.alloc_many(100)
    alloc.free_many(first[0::2])
    assert alloc.alloc_many(50) == sorted(first[0::2])


def test_extend_batch_matches_per_task_extend():
    """Batched decode-step extension is observably identical to per-request.

    Steady decode extends every request by exactly one position; the batch
    must produce the same pages, the same slot staging and the same
    length bookkeeping as the historical per-request loop, including the
    page ORDER (both harvest lowest-first).
    """

    def make_pool():
        return BlockPool(
            n_layers=2,
            n_kv_heads=1,
            head_dim=4,
            max_batch_size=8,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
            page_size=1,
            n_tokens=8 * 64,
        )

    pool_a, pool_b = make_pool(), make_pool()
    mgr_a = _make_task_cache(pool_a)
    mgr_b = _make_task_cache(pool_b)
    ws = _ws(pool_a)

    ids = [f"t{i}" for i in range(4)]
    for tid in ids:
        assert mgr_a.alloc_slots(tid, [10, 20, 30])
        assert mgr_b.alloc_slots(tid, [10, 20, 30])

    # Per-request reference: four extends at position 3.
    for tid in ids:
        assert mgr_a.extend_slots(tid, 3)
    # Batched: same positions through one strategy call.
    assert mgr_b.extend_slots_batch(list(ids), [3] * 4) == [True] * 4

    for tid in ids:
        sa = mgr_a._states[tid]
        sb = mgr_b._states[tid]
        assert sa.pages == sb.pages
        assert sa._slots == sb._slots
        assert sa.length == sb.length == 4

    # Next positions stay in lockstep, and bind flushes both equally.
    assert mgr_b.extend_slots_batch(list(ids), [4] * 4) == [True] * 4
    for tid in ids:
        assert mgr_a.extend_slots(tid, 4)
    mgr_a.bind(ids, ws)
    mgr_b.bind(ids, _ws(pool_b))
    for tid in ids:
        sa = mgr_a._states[tid]
        sb = mgr_b._states[tid]
        assert sa._slots == sb._slots
        assert sa.length == sb.length == 5

    # A missing request fails alone; the rest still extend.
    assert mgr_b.extend_slots_batch(["ghost"] + ids[1:], [5] * 4) == [
        False,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize("prompt_lens", [(5,), (5, 1, 2, 6), (4, 5, 8, 3)])
def test_paged_extend_batch_preserves_real_kv_across_mixed_positions(
    kv_device, prompt_lens
):
    pool = _make_paged_pool(page_size=4, max_seq_len=16, n_tokens=64, device=kv_device)
    task_cache = _make_task_cache(pool)
    ws = _ws(pool)
    ids = [f"request{i}" for i in range(len(prompt_lens))]
    for request_id, length in zip(ids, prompt_lens):
        assert task_cache.alloc_slots(request_id, list(range(length)))
        state = task_cache._states[request_id]
        # Unmapped positions must never be mistaken for physical slot zero.
        pool.req_pool.req_to_token[state.req_idx, length:].fill_(-1)

    pool._storage.k_buffer.fill_(float("nan"))
    pool._storage.v_buffer.fill_(float("nan"))
    initial = torch.arange(
        sum(prompt_lens) * 8, dtype=pool.dtype, device=kv_device
    ).reshape(-1, 2, 4)
    kv = task_cache.bind(ids, ws, start_pos=0)
    kv.k_buffer[0, kv.out_cache_loc] = initial
    kv.v_buffer[0, kv.out_cache_loc] = -initial - 1
    expected = list(initial.split(prompt_lens))

    # Includes all-in-page steps, page crossings and mixed in-page/crossing
    # requests. Single prompt=5 must map position 5 to slot 5, not slot 0.
    for step in range(5):
        positions = [length + step for length in prompt_lens]
        assert task_cache.extend_slots_batch(ids, positions) == [True] * len(ids)
        kv = task_cache.bind(ids, ws)
        expected_slots = [
            task_cache._states[request_id].pages[pos // pool.page_size] * pool.page_size
            + pos % pool.page_size
            for request_id, pos in zip(ids, positions)
        ]
        assert kv.out_cache_loc.tolist() == expected_slots
        assert kv.seq_lens.tolist() == [pos + 1 for pos in positions]
        assert len(set(expected_slots)) == len(ids)
        new_k = torch.arange(len(ids) * 8, dtype=pool.dtype, device=kv_device).reshape(
            -1, 2, 4
        ) + 1000 * (step + 1)
        kv.k_buffer[0, kv.out_cache_loc] = new_k
        kv.v_buffer[0, kv.out_cache_loc] = -new_k - 1
        for i, (request_id, pos) in enumerate(zip(ids, positions)):
            expected[i] = torch.cat([expected[i], new_k[i : i + 1]])
            row = task_cache._states[request_id].req_idx
            slots = pool.req_pool.req_to_token[row, : pos + 1]
            torch.testing.assert_close(kv.k_buffer[0, slots], expected[i])
            torch.testing.assert_close(kv.v_buffer[0, slots], -expected[i] - 1)


def test_paged_extend_batch_fallback_still_maps_in_page_positions():
    pool = _make_paged_pool(page_size=4, n_tokens=28, max_seq_len=16)
    task_cache = _make_task_cache(pool)
    lengths = [5, 4, 4, 4]
    ids = [f"request{i}" for i in range(len(lengths))]
    for request_id, length in zip(ids, lengths):
        assert task_cache.alloc_slots(request_id, list(range(length)))
        state = task_cache._states[request_id]
        pool.req_pool.req_to_token[state.req_idx, length:].fill_(-1)

    # Three page crossings but only two pages remain; the first request
    # needs no new page and must still have its intra-page write mapped.
    assert task_cache.extend_slots_batch(ids, lengths) == [True, True, True, False]
    kv = task_cache.bind(ids[:3], _ws(pool))
    assert kv.out_cache_loc.tolist() == [5, 20, 24]
    assert [task_cache._states[rid].length for rid in ids] == [6, 5, 5, 4]
    last = task_cache._states[ids[-1]]
    assert pool.req_pool.req_to_token[last.req_idx, 4].item() == -1


def test_extend_batch_falls_back_when_pool_runs_dry():
    """Pool exhaustion keeps per-request success ORDER via the fallback.

    The batch harvest cannot satisfy every state when the pool is
    nearly empty; the strategy then re-runs per-request extend so failure
    lands on the same requests the historical loop would have failed.
    """
    # 6 pages total: 4 consumed by prompts, 2 free.
    pool = BlockPool(
        n_layers=1,
        n_kv_heads=1,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        page_size=1,
        n_tokens=6,
    )
    mgr = _make_task_cache(pool)
    ids = [f"dry{i}" for i in range(4)]
    for tid in ids:
        # One page each leaves 2 free for decode extension.
        assert mgr.alloc_slots(tid, [7])
    results = mgr.extend_slots_batch(list(ids), [1] * 4)
    assert results == [True, True, False, False]
    for tid, ok in zip(ids, results):
        assert (mgr._states[tid].length == 2) == ok
