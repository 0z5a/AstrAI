"""Shared machinery for the scripts/eval benchmarks.

Generation-side (HumanEval / MBPP): engine batch loop, code-execution pool,
pass@k scoring, JSON I/O, result report.
Scoring-side (MMLU / HellaSwag): model+tokenizer loader and batched
(context, continuation) log-likelihood.

Protocol decisions stay in each benchmark script (prompt format, stop
sequences, extraction); only mechanism lives here.
"""

import json
import subprocess
import sys
from math import prod
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deduplicate(seq: Sequence[str]) -> List[str]:
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def generate_batch(
    engine,
    prompt: str,
    n: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    """Draw n completions for one prompt, deduplicated."""
    completions: List[str] = []
    remaining = n
    while remaining > 0:
        current = min(batch_size, remaining)
        outputs = engine.generate(
            prompt=[prompt] * current,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        completions.extend(outputs if isinstance(outputs, list) else [outputs])
        remaining -= current
    return deduplicate(completions)


def execute_one(args: tuple) -> bool:
    full_code, timeout = args
    try:
        r = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def test_all(
    items: Sequence[dict],
    codes_for,
    test_workers: int,
    desc: str = "Testing",
) -> List[Tuple[str, int, int]]:
    """Execute generated code for each item in a process pool.

    codes_for(item) -> (task_id, [(full_code, timeout), ...]); returns
    (task_id, n, passed) per item.
    """
    from concurrent.futures import ProcessPoolExecutor

    results: List[Tuple[str, int, int]] = []
    pool = ProcessPoolExecutor(max_workers=test_workers)
    try:
        for item in tqdm.tqdm(items, desc=desc, unit="problem"):
            task_id, codes = codes_for(item)
            passed = sum(1 for ok in pool.map(execute_one, codes) if ok)
            results.append((task_id, len(codes), passed))
    finally:
        pool.shutdown(wait=True)
    return results


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def score_results(
    results: Sequence[Tuple[str, int, int]],
    k_values: Tuple[int, ...],
) -> Dict:
    """Unbiased pass@k per problem; per-problem k entries are None when
    n < k (e.g. after deduplication); the summary averages only computed ks."""
    scores: Dict[int, List[float]] = {k: [] for k in k_values}
    output: Dict = {}
    for task_id, n, passed in results:
        entry = {"task_id": task_id, "n": n, "passed": passed}
        for k in k_values:
            if k <= n:
                pk = round(pass_at_k(n, passed, k), 4)
                entry[f"pass@{k}"] = pk
                scores[k].append(pk)
            else:
                entry[f"pass@{k}"] = None
        output[str(task_id)] = entry

    summary = {}
    for k in k_values:
        vals = scores[k]
        summary[f"pass@{k}"] = round(float(np.mean(vals)), 4) if vals else None
    output["_summary"] = summary
    return output


def report(scored: Dict):
    summary = scored.pop("_summary", {})
    print(f"\n{'=' * 60}")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: N/A")
    print(f"{'=' * 60}")
    scored["_summary"] = summary


def load_score_model(
    param_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device=device, dtype=getattr(torch, dtype))
    model.eval()
    return model, tokenizer


def loglikelihood_batched(
    model,
    tokenizer,
    requests: List[Tuple[List[int], List[int]]],
    device: str,
    max_model_len: int,
) -> List[float]:
    """Summed log-probability of each continuation given its context.

    requests: (ctx_ids, cont_ids) pairs; token sequences are concatenated as
    ``ctx_ids + cont_ids`` — callers must ensure the tokenization matches how
    the model saw such text in training (see lm-eval's _encode_pair caveat).
    """
    all_inputs = []
    for i, (ctx_ids, cont_ids) in enumerate(requests):
        input_ids = ctx_ids + cont_ids
        if len(input_ids) > max_model_len:
            overflow = len(input_ids) - max_model_len
            input_ids = input_ids[overflow:]
            ctx_len = len(input_ids) - len(cont_ids)
        else:
            ctx_len = len(ctx_ids)
        all_inputs.append((i, input_ids, ctx_len, cont_ids))

    n = len(all_inputs)
    max_len = max(len(x[1]) for x in all_inputs)
    padded = torch.zeros(n, max_len, dtype=torch.long, device=device)
    mask = torch.zeros(n, max_len, dtype=torch.bool, device=device)
    for i, (_, ids, _, _) in enumerate(all_inputs):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[i, : len(ids)] = True

    with torch.inference_mode():
        logits = model(padded, input_mask=mask)["logits"]

    scores = [0.0] * len(requests)
    for i, (ri, _, ctx_len, cont_ids) in enumerate(all_inputs):
        score = 0.0
        for j, tid in enumerate(cont_ids):
            pos = ctx_len - 1 + j
            if pos >= logits.size(1):
                break
            score += torch.nn.functional.log_softmax(logits[i, pos].float(), dim=-1)[
                tid
            ].item()
        scores[ri] = score
    return scores
