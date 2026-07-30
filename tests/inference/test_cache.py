"""Unit tests for inference cache components."""

import torch

from astrai.inference import (
    Allocator,
    KVStorage,
    PagePool,
    PrefixCache,
    ReqToTokenPool,
    page_hash,
)

# ---- page_hash ----


def test_page_hash_full_page():
    token_ids = list(range(256))
    h = page_hash(token_ids, 0, 64)
    assert isinstance(h, int)
    assert h >= 0


def test_page_hash_different_page_differs():
    token_ids = list(range(256))
    assert page_hash(token_ids, 0, 64) != page_hash(token_ids, 1, 64)


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


# ---- PrefixCache ----


def test_prefix_cache_lookup_returns_hits():
    token_ids = list(range(256))
    prefix = PrefixCache(64)
    pages = [0, 1, 2, 3]
    for i, p in enumerate(pages):
        prefix.record(p, token_ids, i)
    hits = prefix.lookup(token_ids)
    assert hits == pages


def test_prefix_cache_lookup_stops_at_first_miss():
    token_ids = list(range(256))
    prefix = PrefixCache(64)
    prefix.record(0, token_ids, 0)
    prefix.record(1, [99] * 64, 1)
    hits = prefix.lookup(token_ids)
    assert len(hits) == 1
    assert hits[0] == 0


def test_prefix_cache_ignores_partial_last_page():
    token_ids = list(range(100))
    prefix = PrefixCache(64)
    prefix.record(0, token_ids, 0)
    hits = prefix.lookup(token_ids)
    assert len(hits) == 1


def test_prefix_cache_on_evict_clears_mappings():
    prefix = PrefixCache(64)
    prefix.record(0, list(range(64)), 0)
    assert 0 in prefix._page_to_hash
    prefix.evict(0)
    assert 0 not in prefix._page_to_hash


def test_prefix_cache_has_page():
    prefix = PrefixCache(64)
    assert not prefix.has_page(0)
    prefix.record(0, list(range(64)), 0)
    assert prefix.has_page(0)


# ---- ReqToTokenPool ----


def test_req_to_token_pool_alloc_free():
    pool = ReqToTokenPool(4, 128, torch.device("cpu"))
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


def test_kv_storage_set_and_get():
    storage = KVStorage(
        size=16,
        n_layers=2,
        n_kv_heads=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    loc = torch.tensor([[0, 1]], dtype=torch.long)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    storage.set_kv_buffer(0, loc, k, v)
    assert torch.allclose(storage.get_key_buffer(0)[loc], k)
    assert torch.allclose(storage.get_value_buffer(0)[loc], v)


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


# ---- PagePool (contiguous mode) ----


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
    return PagePool(**defaults)


def test_page_pool_contiguous_task_alloc_free():
    pool = _make_contiguous_pool()
    assert pool.task_alloc("t1", [1, 2, 3])
    assert "t1" in pool._task_req
    pool.task_free("t1")
    assert "t1" not in pool._task_req


def test_page_pool_contiguous_task_extend():
    pool = _make_contiguous_pool()
    pool.task_alloc("t1", [1, 2, 3])
    assert pool.task_extend("t1", 3)
    assert pool.task_extend("t1", 63)
    assert not pool.task_extend("t1", 64)


def test_page_pool_contiguous_task_cached():
    pool = _make_contiguous_pool()
    pool.task_alloc("t1", [1, 2, 3])
    assert pool.task_cached("t1") == 0


def test_page_pool_contiguous_bind_tasks_prefill():
    pool = _make_contiguous_pool()
    pool.task_alloc("t1", list(range(10)))
    pool.task_alloc("t2", list(range(10)))
    kv = pool.bind_tasks(["t1", "t2"], [10, 10], torch.device("cpu"), start_pos=0)
    assert kv.out_cache_loc.shape == (2, 10)
    assert kv.seq_lens.tolist() == [10, 10]
    assert kv.req_pool_indices.shape == (2,)


def test_page_pool_contiguous_bind_tasks_decode():
    pool = _make_contiguous_pool()
    pool.task_alloc("t1", list(range(10)))
    pool.task_alloc("t2", list(range(8)))
    kv = pool.bind_tasks(["t1", "t2"], [11, 9], torch.device("cpu"))
    assert kv.out_cache_loc.shape == (2, 1)
    assert kv.seq_lens.tolist() == [11, 9]


def test_page_pool_contiguous_bind_roundtrip():
    """Write KV via bind_tasks, then gather via req_to_token indexing."""
    pool = _make_contiguous_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    pool.task_alloc("t1", list(range(4)))

    kv = pool.bind_tasks(["t1"], [4], torch.device("cpu"), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0][kv.out_cache_loc] = k
    kv.v_buffer[0][kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0][indices]
    gathered_v = kv.v_buffer[0][indices]
    assert torch.allclose(gathered_k, k)
    assert torch.allclose(gathered_v, v)


# ---- PagePool (paged mode, page_size=1) ----


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
    return PagePool(**defaults)


def test_page_pool_paged_task_alloc():
    pool = _make_paged_pool()
    assert pool.task_alloc("t1", list(range(10)))
    req_idx = pool._task_req["t1"]
    slots = pool._task_slots["t1"]
    assert len(slots) == 10
    assert pool._req_pool.req_to_token[req_idx, 0].item() == slots[0]


def test_page_pool_paged_task_extend():
    pool = _make_paged_pool()
    pool.task_alloc("t1", list(range(4)))
    assert pool.task_extend("t1", 4)
    req_idx = pool._task_req["t1"]
    slot = pool._req_pool.req_to_token[req_idx, 4].item()
    assert slot >= 0


def test_page_pool_paged_task_free_releases_slots():
    pool = _make_paged_pool(n_tokens=16)
    pool.task_alloc("t1", list(range(8)))
    pool.task_free("t1")
    assert "t1" not in pool._task_req
    assert len(pool._req_pool.free_slots) == 4


def test_page_pool_paged_bind_roundtrip():
    pool = _make_paged_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    pool.task_alloc("t1", list(range(4)))

    kv = pool.bind_tasks(["t1"], [4], torch.device("cpu"), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0][kv.out_cache_loc] = k
    kv.v_buffer[0][kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0][indices]
    assert torch.allclose(gathered_k, k)


# ---- PagePool (paged mode, page_size>1) ----


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
    return PagePool(**defaults)


def test_page_pool_paged_ps64_task_alloc():
    pool = _make_paged_pool_ps64()
    prompt = list(range(200))
    assert pool.task_alloc("t1", prompt)
    assert pool.task_cached("t1") == 0
    n_pages = (200 + 63) // 64
    assert len(pool._task_pages["t1"]) == n_pages


def test_page_pool_paged_ps64_task_extend_crosses_page():
    pool = _make_paged_pool_ps64()
    pool.task_alloc("t1", list(range(64)))
    assert pool.task_extend("t1", 64)
    assert len(pool._task_pages["t1"]) >= 2


def test_page_pool_paged_ps64_bind_roundtrip():
    pool = _make_paged_pool_ps64(n_layers=1, n_kv_heads=2, head_dim=4)
    prompt = list(range(128))
    pool.task_alloc("t1", prompt)

    kv = pool.bind_tasks(["t1"], [128], torch.device("cpu"), start_pos=0)
    k = torch.randn(1, 128, 2, 4)
    v = torch.randn(1, 128, 2, 4)
    kv.k_buffer[0][kv.out_cache_loc] = k
    kv.v_buffer[0][kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :128]
    gathered_k = kv.k_buffer[0][indices]
    assert torch.allclose(gathered_k, k)
