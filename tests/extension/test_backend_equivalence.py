"""Numerical equivalence between TorchNativeBackend and CudaBackend.

Covers training forward, inference prefill, inference decode (mixed
seq_lens with padding mask), and end-to-end scheduler.run_batch.
"""

import torch

from astrai.extension import ATTN_BACKEND, attn_backend
from astrai.inference.core.cache import PagePool
from tests.extension.conftest import D, skip_no_cuda


@skip_no_cuda
def test_training_forward_matches_torch(cuda_model):
    """Training forward (kv_cache=None) should produce identical logits."""
    model, _ = cuda_model
    input_ids = torch.randint(0, 1000, (2, 16), device="cuda")

    with torch.no_grad():
        out_torch = model(input_ids)
    with attn_backend(ATTN_BACKEND.CUDA):
        with torch.no_grad():
            out_cuda = model(input_ids)

    diff = (out_torch["logits"].float() - out_cuda["logits"].float()).abs().max().item()
    assert diff == 0.0, f"Training forward diff {diff} should be 0"


@skip_no_cuda
def test_prefill_with_kv_cache_matches_torch(cuda_model):
    """Inference prefill with KV cache should match torch backend."""
    model, _ = cuda_model
    prompt_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15]]
    max_len = max(len(p) for p in prompt_ids)
    batch = len(prompt_ids)

    device = "cuda"
    input_ids = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    input_mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    position_ids = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids):
        input_ids[i, : len(p)] = torch.tensor(p, device=device)
        input_mask[i, : len(p)] = True
        position_ids[i, : len(p)] = torch.arange(len(p), device=device)

    cache = PagePool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=4,
        max_seq_len=64,
        device=device,
        dtype=torch.bfloat16,
    )

    cache.task_alloc("t1", prompt_ids[0])
    cache.task_alloc("t2", prompt_ids[1])
    kv1 = cache.bind_tasks(
        ["t1", "t2"], [len(prompt_ids[0]), len(prompt_ids[1])], device, start_pos=0
    )
    with torch.inference_mode():
        out_torch = model(
            input_ids, input_mask=input_mask, kv_cache=kv1, position_ids=position_ids
        )

    cache.task_free("t1")
    cache.task_free("t2")
    cache.task_alloc("t1", prompt_ids[0])
    cache.task_alloc("t2", prompt_ids[1])
    kv2 = cache.bind_tasks(
        ["t1", "t2"], [len(prompt_ids[0]), len(prompt_ids[1])], device, start_pos=0
    )
    with attn_backend(ATTN_BACKEND.CUDA):
        with torch.inference_mode():
            out_cuda = model(
                input_ids,
                input_mask=input_mask,
                kv_cache=kv2,
                position_ids=position_ids,
            )

    for i, p in enumerate(prompt_ids):
        d = (
            (
                out_torch["logits"][i, : len(p)].float()
                - out_cuda["logits"][i, : len(p)].float()
            )
            .abs()
            .max()
            .item()
        )
        assert d == 0.0, f"Prefill diff for sample {i}: {d}"


@skip_no_cuda
def test_decode_mixed_seq_lens_matches_torch(cuda_model):
    """Decode with mixed seq_lens in batch — padding mask must produce correct output."""
    model, _ = cuda_model
    device = "cuda"

    prompt_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15]]
    cache = PagePool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=4,
        max_seq_len=64,
        device=device,
        dtype=torch.bfloat16,
    )

    # Prefill to populate cache
    max_len = max(len(p) for p in prompt_ids)
    batch = len(prompt_ids)
    input_ids = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    input_mask = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    position_ids = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    for i, p in enumerate(prompt_ids):
        input_ids[i, : len(p)] = torch.tensor(p, device=device)
        input_mask[i, : len(p)] = True
        position_ids[i, : len(p)] = torch.arange(len(p), device=device)

    cache.task_alloc("t1", prompt_ids[0])
    cache.task_alloc("t2", prompt_ids[1])
    kv = cache.bind_tasks(
        ["t1", "t2"], [len(prompt_ids[0]), len(prompt_ids[1])], device, start_pos=0
    )
    with torch.inference_mode():
        model(input_ids, input_mask=input_mask, kv_cache=kv, position_ids=position_ids)

    # Decode step — seq_lens are 9 and 7 (after extending)
    dec_ids = torch.tensor([[99], [98]], dtype=torch.long, device=device)
    dec_pos = torch.tensor([[8], [6]], dtype=torch.long, device=device)
    total_len = 9
    dec_mask = dec_pos[:, None, None] >= torch.arange(total_len, device=device)

    kv_t = cache.bind_tasks(["t1", "t2"], [9, 7], device)
    with torch.inference_mode():
        out_torch = model(
            dec_ids, input_mask=dec_mask, kv_cache=kv_t, position_ids=dec_pos
        )

    kv_c = cache.bind_tasks(["t1", "t2"], [9, 7], device)
    with attn_backend(ATTN_BACKEND.CUDA):
        with torch.inference_mode():
            out_cuda = model(
                dec_ids, input_mask=dec_mask, kv_cache=kv_c, position_ids=dec_pos
            )

    diff = (out_torch["logits"].float() - out_cuda["logits"].float()).abs().max().item()
    assert diff < 0.05, f"Decode diff (mixed seq_lens): {diff}"


@skip_no_cuda
def test_run_batch_cuda_matches_torch_greedy(cuda_model):
    """Greedy decode (temperature=0) should produce identical tokens."""
    from astrai.inference.core.scheduler import InferenceScheduler
    from tests.helpers import FakeTokenizer

    model, _ = cuda_model
    tokenizer = FakeTokenizer()

    prompts = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15, 16]]

    sched = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out_torch = sched.run_batch(prompts, max_tokens=5, temperature=0.0)
    sched.stop()

    cache_cuda = PagePool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sched2 = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
        cache=cache_cuda,
    )
    with attn_backend(ATTN_BACKEND.CUDA):
        out_cuda = sched2.run_batch(prompts, max_tokens=5, temperature=0.0)
    sched2.stop()

    assert out_torch == out_cuda, f"Torch={out_torch} != CUDA={out_cuda}"
