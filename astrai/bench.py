"""Shared machinery for the scripts/eval benchmarks.

Generation-side (HumanEval / MBPP): engine batch loop, code-execution pool,
pass@k scoring, JSON I/O, result report.
Scoring-side (MMLU / HellaSwag): model+tokenizer loader and batched
(context, continuation) log-likelihood.
Results-side: parse benchmark output JSONs into standardized metrics, collect
`results/<bench>_<tag>.json` files into comparison tables.
Suite-side: plan/launch the standard benchmark set for one checkpoint with
per-job retry (output-file + process-liveness polling).

Protocol decisions stay in each benchmark script (prompt format, stop
sequences, extraction); only mechanism lives here.
"""

import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Sequence, Tuple

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


def extract_metrics(payload: dict, bench_key: str) -> Dict[str, Optional[float]]:
    """Standardized metrics from one benchmark output JSON.

    Fraction-valued metrics are 0..1; counts are ints (see COUNT_METRICS).
    """
    if bench_key in ("humaneval", "mbpp", "mbpp2"):
        summary = payload.get("_summary", {})
        problems = [
            v
            for k, v in payload.items()
            if not k.startswith("_") and isinstance(v, dict)
        ]
        return {
            "pass@1": summary.get("pass@1"),
            "pass@10": summary.get("pass@10"),
            "zero_pass": sum(
                1
                for v in problems
                if v.get("pass@1", 0) == 0 and v.get("pass@10", 0) == 0
            ),
            "strong_pass": sum(1 for v in problems if (v.get("pass@1") or 0) >= 0.5),
        }
    if bench_key == "mmlu":
        return {"acc": payload.get("_overall", {}).get("accuracy")}
    if bench_key == "hellaswag":
        s = payload.get("_summary", {})
        return {"acc": s.get("acc"), "acc_norm": s.get("acc_norm")}
    if bench_key == "ifeval":
        sub: Dict[str, List[int]] = {}
        total_passed = total_constraints = 0
        for v in payload.values():
            if not isinstance(v, dict) or "constraints" not in v:
                continue
            total_passed += v.get("num_passed", 0)
            total_constraints += v.get("num_constraints", 0)
            for con in v["constraints"]:
                st = sub.setdefault(con.get("instruction_id"), [0, 0])
                st[1] += 1
                st[0] += 1 if con.get("passed") else 0
        # Prefer the script's own aggregate: its denominator excludes
        # unsupported constraints (e.g. 793 of 834), unlike the per-problem sum.
        summary = payload.get("_summary", {})
        if "overall_accuracy" in summary:
            acc: Optional[float] = summary["overall_accuracy"]
        else:
            acc = (total_passed / total_constraints) if total_constraints else None
        out: Dict[str, Optional[float]] = {"acc": acc}
        for iid, name in (
            ("detectable_format:json_format", "json_format"),
            ("detectable_format:number_bullet_lists", "num_bullet_lists"),
        ):
            p, t = sub.get(iid, [0, 0])
            out[name] = (p / t) if t else None
        return out
    raise KeyError(f"no extractor for benchmark {bench_key!r}")


# prefix -> (display label, bench_key); longer prefixes first when matching
RESULTS_REGISTRY = {
    "mbpp2": ("MBPP-sig", "mbpp2"),
    "mbpp": ("MBPP", "mbpp"),
    "hellaswag": ("HellaSwag", "hellaswag"),
    "humaneval": ("HumanEval", "humaneval"),
    "ifeval": ("IFEval", "ifeval"),
    "mmlu": ("MMLU", "mmlu"),
}

COUNT_METRICS = {"zero_pass", "strong_pass"}


def parse_results_name(filename: str) -> Optional[Tuple[str, str]]:
    """'mbpp2_mix3.json' -> ('mbpp2', 'mix3'); None for non-results files."""
    if not filename.endswith(".json") or filename.endswith("_completions.json"):
        return None
    stem = filename[: -len(".json")]
    for prefix in sorted(RESULTS_REGISTRY, key=len, reverse=True):
        if stem.startswith(prefix + "_"):
            return prefix, stem[len(prefix) + 1 :]
    return None


def collect_results(
    results_dir: str,
    benchmarks: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """{tag: {bench_key: metrics}} for every parseable file in results_dir."""
    collected: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for fn in sorted(os.listdir(results_dir)):
        parsed = parse_results_name(fn)
        if parsed is None:
            continue
        bench_key, tag = parsed
        if benchmarks and bench_key not in benchmarks:
            continue
        if tags and tag not in tags:
            continue
        with open(os.path.join(results_dir, fn), encoding="utf-8") as f:
            payload = json.load(f)
        collected.setdefault(tag, {})[bench_key] = extract_metrics(payload, bench_key)
    return collected


def render_table(collected: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> str:
    """Markdown comparison table: rows = benchmark metrics, columns = tags."""
    tags = sorted(collected)
    row_order: List[Tuple[str, str, str]] = []  # (display_label, metric, bench_key)
    cells: Dict[Tuple[str, str, str], Dict[str, Optional[float]]] = {}
    for tag in tags:
        for bench_key in RESULTS_REGISTRY:
            metrics = collected.get(tag, {}).get(bench_key)
            if not metrics:
                continue
            label = RESULTS_REGISTRY[bench_key][0]
            for metric, value in metrics.items():
                row = (label, metric, bench_key)
                if row not in cells:
                    cells[row] = {}
                    row_order.append(row)
                cells[row][tag] = value

    def fmt(metric: str, value: Optional[float]) -> str:
        if value is None:
            return "—"
        if metric in COUNT_METRICS:
            return str(int(value))
        return f"{value * 100:.2f}%"

    lines = ["| metric | " + " | ".join(tags) + " |", "|---" * (len(tags) + 1) + "|"]
    for label, metric, bench_key in row_order:
        row = [fmt(metric, cells[(label, metric, bench_key)].get(tag)) for tag in tags]
        lines.append(f"| {label} {metric} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def summarize_tag(tag: str, results_dir: str = "results") -> str:
    return render_table(collect_results(results_dir, tags=[tag]))


# ---------------------------------------------------------------------------
# Suite runner: the standard benchmark set for one checkpoint, with retries
# ---------------------------------------------------------------------------

STANDARD_PARAMS = {
    "mmlu": "--n_shot 5 --batch_size 4",
    "ifeval": "--num_samples 1 --batch_size 64 --temperature 0.1",
    "humaneval": "--num_samples 20 --batch_size 64",
    "mbpp": "--num_samples 20 --batch_size 64",
    "mbpp2": "--num_samples 20 --batch_size 64",
    "hellaswag": "--batch_size 16",
}

# Cheap invocations for a plumbing smoke test. IFEval has no cheap mode — it
# is skipped (None) and must run full or not at all.
SMOKE_PARAMS = {
    "mmlu": "--n_shot 5 --batch_size 4 --subjects abstract_algebra",
    "ifeval": None,
    "humaneval": "--num_samples 2 --batch_size 2 --problems 0 1 2",
    "mbpp": "--num_samples 2 --batch_size 2 --problems 0 1 2",
    "mbpp2": "--num_samples 2 --batch_size 2 --problems 0 1 2",
    "hellaswag": "--limit 64 --batch_size 8",
}

SUITE_SCRIPTS = {
    "mmlu": "scripts/eval/evaluate_mmlu.py",
    "ifeval": "scripts/eval/evaluate_ifeval.py",
    "humaneval": "scripts/eval/evaluate_humaneval.py",
    "mbpp": "scripts/eval/evaluate_mbpp.py",
    "mbpp2": "scripts/eval/evaluate_mbpp.py",
    "hellaswag": "scripts/eval/evaluate_hellaswag.py",
}

DEFAULT_GPU_ORDER = [4, 5, 6, 7, 0, 1, 2, 3]


@dataclass
class SuiteJob:
    bench: str
    cmd: str
    outfile: str
    logfile: str


def pick_free_gpus(n: int, preferred: Optional[Sequence[int]] = None) -> List[int]:
    """n GPUs with <100 MiB allocated, caller's order first, else 4-7 first."""
    q = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    free = set()
    for line in q.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) < 100:
            free.add(int(parts[0]))
    order = list(preferred or []) + [
        g for g in DEFAULT_GPU_ORDER if g not in set(preferred or [])
    ]
    picked = [g for g in order if g in free]
    return picked[:n]


def plan_jobs(
    ckpt: str,
    tag: str,
    benchmarks: Sequence[str],
    gpus: Optional[Sequence[int]] = None,
    smoke: bool = False,
    results_dir: str = "results",
    logs_dir: str = "logs",
) -> Tuple[List[SuiteJob], List[str]]:
    """One SuiteJob per benchmark (pinning one GPU each) + skipped-bench list.

    Command strings use repo-root-relative paths: run from the AstrAI root.
    """
    params = SMOKE_PARAMS if smoke else STANDARD_PARAMS
    runnable = [b for b in benchmarks if params.get(b) is not None]
    skipped = [b for b in benchmarks if params.get(b) is None]
    gpus = pick_free_gpus(len(runnable), preferred=gpus)
    if len(gpus) < len(runnable):
        raise RuntimeError(f"need {len(runnable)} free GPUs, found {len(gpus)}")
    jobs = []
    for bench, gpu in zip(runnable, gpus):
        outfile = f"{results_dir}/{bench}_{tag}.json"
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu} {sys.executable} -u {SUITE_SCRIPTS[bench]} "
            f"--param_path {ckpt} {params[bench]} --output {outfile}"
        )
        jobs.append(SuiteJob(bench, cmd, outfile, f"{logs_dir}/{bench}_{tag}.log"))
    return jobs, skipped


def run_suite(
    jobs: List[SuiteJob], attempts: int = 3, poll_s: int = 15
) -> Dict[str, str]:
    """Run jobs concurrently; retry a job when its process dies without
    writing its output file. Jobs whose output already exists are skipped
    (idempotent restarts). Returns per-bench final status strings."""
    status: Dict[str, str] = {}
    queue: List[SuiteJob] = []
    for job in jobs:
        if os.path.exists(job.outfile):
            status[job.bench] = "skipped (output exists)"
        else:
            queue.append(job)
    active: Dict[str, Tuple[SuiteJob, subprocess.Popen, int, float]] = {}
    launched: Dict[str, int] = collections.Counter()
    while queue or active:
        while queue:
            job = queue.pop(0)
            if launched[job.bench] >= attempts:
                status[job.bench] = f"failed after {attempts} attempts"
                continue
            launched[job.bench] += 1
            log = open(job.logfile, "a", encoding="utf-8")
            log.write(
                f"\n=== attempt {launched[job.bench]} ({time.strftime('%H:%M:%S')}) ===\n"
            )
            log.flush()
            proc = subprocess.Popen(
                job.cmd, shell=True, stdout=log, stderr=log, start_new_session=True
            )
            active[job.bench] = (job, proc, launched[job.bench], time.time())
        time.sleep(poll_s)
        for bench, (job, proc, n, t0) in list(active.items()):
            if os.path.exists(job.outfile):
                status[bench] = "ok"
                del active[bench]
            elif proc.poll() is not None:
                print(
                    f"[suite] {bench}: attempt {n} died (rc={proc.returncode}), retrying"
                )
                del active[bench]
                queue.append(job)
            elif time.time() - t0 > 3600:
                print(f"[suite] {bench}: attempt {n} timed out after 1h, killing")
                proc.kill()
                del active[bench]
                queue.append(job)
    return status
