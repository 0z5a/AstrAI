"""Generate AOT plan-table rows from measured shape sweeps.

Every dtype combo is swept at each (M, N, K, batch) grid point under every
candidate the host planner can pick — the degraded-bands fallback plus the six
forced recipes (big/narrow/small CTA s2 and s3 ring; feasibility mirror
drops the combos whose ring exceeds the smem opt-in ceiling, e.g. big s3
on the 2B x 2B pair) — and the measured winner per point becomes a
dispatch-table row (csrc/kernels/gemm/plan_table.h). The rows are keyed
(M, N) bands per dtype class: K is not a row key (the ring K is fixed at
64), so a conflict across K / batch at one (M, N) resolves to the recipe
with the best tflops.

The candidates are measured back-to-back at each shape in a single
process (shape outer loop, recipe inner loop): each forced recipe runs as
a one-row ASTR_GEMM_TABLE file toggled per launch (the row sources re-read
their env per call, so the comparison happens under the same GPU
clock/thermal state). A sweep that measured whole recipe batches in
separate processes compared the big CTA (measured first) against the
small CTA (measured 30 minutes later) under different boost states and
picked systematically wrong winners.

Usage (rows to a runtime-override file, no rebuild):
    python csrc/bench/gen_plan_table.py \
        --m-values 512,2048,4096 \
        --shapes "qkv:4096:4096,up_gate:14336:4096" \
        --batch 1 --combos w16a16 --output plan_table.txt

--save-results dumps the raw measurements to JSON so the grid can be
re-searched offline with --results-json (e.g. classic merge vs
--band-search cut optimization — the grid search over M band splits).

This script only measures and emits the row file: use it with
ASTR_GEMM_TABLE=plan_table.txt to serve the rows without a rebuild, or
paste them into the compiled-in GENERATED block of plan_table.h by hand.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import time
from pathlib import Path

import click
import torch

from astrai.extension import is_available
from astrai.extension.ops.gemm import quant_gemm

# Combo name -> (activation dtype, weight dtype). Scales: int8 operands
# require their dequant scale, fp8 accept one optionally (a [1] per-tensor
# scalar), bf16 rejects one. Matches the csrc dispatch (gemm.cu).
COMBOS: dict[str, tuple[torch.dtype, torch.dtype]] = {
    "w16a16": (torch.bfloat16, torch.bfloat16),
    "w8a16": (torch.bfloat16, torch.int8),
    "w8a16_f8e4m3": (torch.bfloat16, torch.float8_e4m3fn),
    "w8a16_f8e5m2": (torch.bfloat16, torch.float8_e5m2),
    "w8a8": (torch.int8, torch.int8),
    "f8a8_e4m3": (torch.float8_e4m3fn, torch.float8_e4m3fn),
    "f8a8_e5m2": (torch.float8_e5m2, torch.float8_e5m2),
}

# GemmPerfClass ids (gemm.cuh): W16A16 / W8A16 / W8A8 / F8A8.
PERF_CLASS: dict[str, int] = {
    "w16a16": 0,
    "w8a16": 1,
    "w8a16_f8e4m3": 1,
    "w8a16_f8e5m2": 1,
    "w8a8": 2,
    "f8a8_e4m3": 3,
    "f8a8_e5m2": 3,
}

# Candidate recipes -> (cta id, stages): 0 small / 1 narrow / 2 big.
# The C++ recipe knob (ASTR_GEMM_RECIPE) forces s2 rings only, so s3
# candidates are measured as one-row ASTR_GEMM_TABLE files instead (open
# bands, -1 keys) — plan_gemm smem-gates them exactly like real rows.
RECIPES: dict[str, tuple[int, int] | None] = {
    "model": None,
    "big": (2, 2),
    "big_s3": (2, 3),
    "narrow": (1, 2),
    "narrow_s3": (1, 3),
    "small": (0, 2),
    "small_s3": (0, 3),
}
# Tie preference: stable big > narrow > small, s2 before its s3 twin.
RECIPE_ORDER = (
    "big",
    "narrow",
    "small",
    "big_s3",
    "narrow_s3",
    "small_s3",
    "model",
)

# Ring feasibility mirror (policy.cuh ring_smem_bytes vs the device's
# smem opt-in ceiling): over-budget recipes cannot launch and would be
# silently measured as the model/degraded — filter them out per combo.
CTA_GEOM: dict[int, tuple[int, int]] = {2: (128, 128), 1: (128, 64), 0: (64, 64)}
SMEM_OPTIN = 101376  # sm_120 (RTX 5090) MaxSharedMemoryPerBlockOptin
_DTYPE_BYTES = {
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
}
BYTES: dict[str, tuple[int, int]] = {
    combo: (_DTYPE_BYTES[a], _DTYPE_BYTES[b]) for combo, (a, b) in COMBOS.items()
}


def ring_smem_bytes(stages: int, cta: int, ba: int, bb: int) -> int:
    bm, bn = CTA_GEOM[cta]
    return (stages + 1) * 64 * (bm * ba + bn * bb)


def candidate_recipes(combo: str) -> tuple[str, ...]:
    """Measurable candidates for one combo, in RECIPE_ORDER."""
    return tuple(
        recipe
        for recipe in RECIPE_ORDER
        if recipe == "model"
        or ring_smem_bytes(RECIPES[recipe][1], RECIPES[recipe][0], *BYTES[combo])
        <= SMEM_OPTIN
    )


def _candidate_row_files() -> dict[str, str]:
    """One synthetic open row per recipe — (cta, stages) on -1/-1 keys, so
    the row is the only plan source and plan_from_row smem-gates it."""
    directory = tempfile.mkdtemp(prefix="gemm_recipe_rows_")
    files: dict[str, str] = {}
    for recipe, spec in RECIPES.items():
        if spec is None:
            continue
        cta, stages = spec
        path = os.path.join(directory, f"row_{recipe}.txt")
        with open(path, "w") as f:
            f.write(f"0 0 0 0 -1 -1 {cta} {stages} 0\n")
        files[recipe] = path
    return files


def parse_positive_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def parse_shape(value: str) -> tuple[str, int, int]:
    name, n, k = value.split(":")
    return name, int(n), int(k)


def make_scale(operand_dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    # bf16 operands reject scales; int8 requires its dequant scale, fp8
    # takes one optionally — a [1] per-tensor float32 scalar covers both.
    if operand_dtype == torch.bfloat16:
        return None
    return torch.tensor([0.1], dtype=torch.float32, device=device)


def random_operand(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    base = torch.rand(shape, device=device, dtype=torch.float32) * 0.2 - 0.1
    return base.to(dtype)


def measure(fn, warmup: int, iterations: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / iterations)
    return best


def sweep(
    m_values: tuple[int, ...],
    shapes: list[tuple[str, int, int]],
    combos: tuple[str, ...],
    batch: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[dict]:
    """One process, shape outer loop — every candidate measured back-to-back."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("gemm"):
        raise click.ClickException(
            "the built gemm kernel is required (rebuild "
            "the extension with CSRC_KERNELS=true first)"
        )

    # Candidates are one-row table files toggled per launch: the row
    # sources re-read their env per call, so all candidates at a shape
    # share its GPU clock/thermal state. The "model" candidate measures
    # the degraded band rows (the cost model is deleted from the C++):
    # it is the "this band has no table row" reference of the min-gain
    # mode, not a planner.
    os.environ.pop("ASTR_GEMM_RECIPE", None)
    row_files = _candidate_row_files()

    device = torch.device(torch.cuda.current_device())
    torch.manual_seed(0)
    results = []
    for combo in combos:
        act_dtype, weight_dtype = COMBOS[combo]
        perf_class = PERF_CLASS[combo]
        a_scale = make_scale(act_dtype, device)
        b_scale = make_scale(weight_dtype, device)
        for _name, n, k in shapes:
            weight = random_operand((batch, n, k), weight_dtype, device)
            for m in m_values:
                acts = random_operand((batch, m, k), act_dtype, device)

                def run(acts=acts, weight=weight, a_scale=a_scale, b_scale=b_scale):
                    return quant_gemm(acts, weight, a_scale, b_scale)

                for recipe in candidate_recipes(combo):
                    if recipe == "model":
                        # "-" = AOT off: skip both override and builtin rows
                        # — the degraded-bands reference (the cost model is
                        # deleted from the C++).
                        os.environ["ASTR_GEMM_TABLE"] = "-"
                    else:
                        os.environ["ASTR_GEMM_TABLE"] = row_files[recipe]
                    run()  # steady state for this (shape, recipe)
                    torch.cuda.synchronize()
                    ms = measure(run, warmup, iterations, trials)
                    flops = 2.0 * batch * m * n * k
                    results.append(
                        {
                            "combo": combo,
                            "perf_class": perf_class,
                            "m": m,
                            "n": n,
                            "k": k,
                            "batch": batch,
                            "recipe": recipe,
                            "ms": ms,
                            "tflops": flops / ms / 1e12,
                        }
                    )
                    print(
                        f"{recipe:8s} {combo:14s} b{batch} m{m:5d} n{n:6d} "
                        f"k{k:5d} {ms * 1e3:8.3f} ms {flops / ms / 1e12:7.1f} "
                        f"TFLOPS",
                        flush=True,
                    )
    os.environ.pop("ASTR_GEMM_TABLE", None)
    return results


def band_edges(values: tuple[int, ...]) -> list[tuple[int, int]]:
    """(min, max] bands for sorted unique values; 0 = open upper."""
    uniq = sorted(set(values))
    if len(uniq) <= 1:
        return [(0, 0)]
    mid_pairs = itertools.pairwise(uniq)
    mids = [a + (b - a) // 2 for a, b in mid_pairs]
    edges = [(0, mids[0])]
    edges += [(mids[i], mids[i + 1]) for i in range(len(mids) - 1)]
    edges.append((mids[-1], 0))
    return edges


def _optimal_partition(
    aggregate: dict[tuple[int, int, int], dict[str, float]],
    points: list[tuple[int, int, int]],
    min_gain: float,
) -> list[tuple[int, int, str]]:
    """Best (start, end, recipe) partition of points — the band grid search:
    a segment's recipe minimizes its total time (1/tflops summed over the
    segment) and a cut pays a min_gain fee of the point median baseline,
    the same noise floor --min-gain enforces for winner rows. This beats
    merge-adjacent-equals (which only cuts where the winner stays
    constant) at the cost of rows whose recipe wins on the segment mean
    rather than at every sampled M."""
    n = len(points)
    recipes = [r for r in RECIPE_ORDER if r != "model"]
    cost = {
        r: [
            1.0 / aggregate[p][r] if r in aggregate.get(p, {}) else float("inf")
            for p in points
        ]
        for r in recipes
    }
    penalty = min_gain * sum(
        sorted(cost[r][i] for r in recipes)[len(recipes) // 2] for i in range(n)
    )
    seg_cost: dict[tuple[int, int], float] = {}
    seg_recipe: dict[tuple[int, int], str] = {}
    for i in range(n):
        for j in range(i, n):
            best = float("inf")
            best_recipe = recipes[0]
            for r in recipes:
                c = sum(cost[r][i : j + 1])
                if c < best:
                    best = c
                    best_recipe = r
            seg_cost[i, j] = best + penalty
            seg_recipe[i, j] = best_recipe
    dp = [float("inf")] * (n + 1)
    prev = [0] * (n + 1)
    dp[0] = 0.0
    for end in range(1, n + 1):
        for start in range(1, end + 1):
            cand = dp[start - 1] + seg_cost[start - 1, end - 1]
            if cand < dp[end]:  # strict < keeps the earlier, longer last
                dp[end] = cand  # segment — fewer rows on exact ties
                prev[end] = start - 1
    segments: list[tuple[int, int, str]] = []
    end = n
    while end > 0:
        start = prev[end]
        segments.append((start, end - 1, seg_recipe[start, end - 1]))
        end = start
    segments.reverse()
    return segments


def _winners_with_lead(
    aggregate: dict[tuple[int, int, int], dict[str, float]],
    points: list[tuple[int, int, int]],
) -> tuple[list[str], list[float]]:
    """Per-point (winner, lead fraction) over the forced recipes."""
    winners: list[str] = []
    leads: list[float] = []
    for p in points:
        over = {r: v for r, v in aggregate.get(p, {}).items() if r != "model"}
        if not over:
            winners.append("small")
            leads.append(0.0)
            continue
        values = sorted(over.values(), reverse=True)
        best = values[0]
        second = values[1] if len(values) > 1 else best
        leads.append(0.0 if best == 0 else (best - second) / best)
        for recipe in RECIPE_ORDER:
            if recipe != "model" and over.get(recipe, 0.0) == best:
                winners.append(recipe)
                break
    return winners, leads


def _smooth_winners(
    winners: list[str], leads: list[float], min_gain: float
) -> list[str]:
    """Noise merge: a point whose winner's lead is under the min-gain floor
    is absorbed into a neighbor's recipe when the neighbors agree (the
    flip is measurement noise, not a real crossover); run until stable.
    Same floor the min-gain mode uses, but with a recipe fallback instead
    of leaving the band unrowed, so full coverage still never misses."""
    out = list(winners)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            if leads[i] >= min_gain:
                continue
            left = out[i - 1] if i > 0 else None
            right = out[i + 1] if i + 1 < len(out) else None
            adopt = None
            if left is None and right is not None and right != out[i]:
                adopt = right  # run start: follow the run
            elif right is None and left is not None and left != out[i]:
                adopt = left  # run end: follow the run
            elif left is not None and left == right and left != out[i]:
                adopt = left  # sandwiched flip: join the neighbors
            if adopt is not None:
                out[i] = adopt
                changed = True
    return out


def build_rows(
    results: list[dict],
    min_gain: float = 0.0,
    full_coverage: bool = False,
    band_search: bool = False,
    smooth: bool = False,
) -> list[str]:
    # (perf_class, m, n) -> {recipe: best tflops across the k/batch grid}.
    aggregate: dict[tuple[int, int, int], dict[str, float]] = {}
    m_values: set[int] = set()
    n_values: set[int] = set()
    for point in results:
        key = (point["perf_class"], point["m"], point["n"])
        over = aggregate.setdefault(key, {})
        tflops = point["tflops"]
        over[point["recipe"]] = max(over.get(point["recipe"], 0.0), tflops)
        m_values.add(point["m"])
        n_values.add(point["n"])

    sorted_m = sorted(m_values)
    sorted_n = sorted(n_values)
    m_bands = band_edges(tuple(sorted_m))
    n_bands = band_edges(tuple(sorted_n))

    rows: list[str] = []
    for perf_class in sorted({pt[0] for pt in aggregate}):
        for n_idx, (n_min, n_max) in enumerate(n_bands):
            keys = [(perf_class, m, sorted_n[n_idx]) for m in sorted_m]
            if full_coverage and band_search:
                segments = _optimal_partition(aggregate, keys, min_gain)
                for m_start, m_end, recipe in segments:
                    m_min, m_max = m_bands[m_start][0], m_bands[m_end][1]
                    cta, stages = RECIPES[recipe]
                    rows.append(
                        f"{m_min} {m_max} {n_min} {n_max} {perf_class} 0 "
                        f"{cta} {stages} 0"
                    )
                continue
            # Winner per m band at this n band; merge adjacent m runs that
            # pick the same recipe into one row.
            if full_coverage and smooth:
                winners, leads = _winners_with_lead(aggregate, keys)
                recipes = _smooth_winners(winners, leads, min_gain)
            elif full_coverage:
                # The table is the only production dispatch: every band
                # gets a row (no min-gain gate, no model fallback), and
                # ties resolve to the stable big>narrow>small preference.
                recipes = [
                    _best_forced(aggregate, (perf_class, m, sorted_n[n_idx]))
                    for m in sorted_m
                ]
            else:
                recipes = [
                    _winner(aggregate, (perf_class, m, sorted_n[n_idx]), min_gain)
                    for m in sorted_m
                ]
            run_start = 0
            for i in range(1, len(recipes) + 1):
                if i == len(recipes) or recipes[i] != recipes[run_start]:
                    recipe = recipes[run_start]
                    if full_coverage or recipe != "model":
                        m_min, m_max = m_bands[run_start][0], m_bands[i - 1][1]
                        cta, stages = RECIPES[recipe]
                        rows.append(
                            f"{m_min} {m_max} {n_min} {n_max} {perf_class} 0 "
                            f"{cta} {stages} 0"
                        )
                    run_start = i
        if full_coverage:
            # Tail catch-all row per class: shapes outside the grid bands
            # still hit (the table never misses). The default recipe is
            # the one measured at the grid's top-right corner (largest M
            # and N) — shapes beyond the grid are big shapes, where the
            # corner's winner beats the grid's most common (small-shape
            # biased) winner.
            corner = _best_forced(aggregate, (perf_class, sorted_m[-1], sorted_n[-1]))
            cta, stages = RECIPES[corner]
            rows.append(f"0 0 0 0 {perf_class} 0 {cta} {stages} 0")
    return rows


def _best_forced(
    aggregate: dict[tuple[int, int, int], dict[str, float]],
    key: tuple[int, int, int],
) -> str:
    # Best measured forced recipe; ties resolve big > narrow > small (a
    # tie carries no information, and the model is retired so there is
    # no fallback).
    over = aggregate.get(key, {})
    best = max((v for r, v in over.items() if r != "model"), default=0.0)
    for recipe in RECIPE_ORDER:
        if recipe == "model":
            continue
        if over.get(recipe, 0.0) == best:
            return recipe
    return "small"


def _winner(
    aggregate: dict[tuple[int, int, int], dict[str, float]],
    key: tuple[int, int, int],
    min_gain: float = 0.0,
) -> str:
    # A forced recipe only wins when it beats every other option (model
    # included) by at least min_gain (a fraction of its throughput): the
    # sweep's near-ties are measurement noise, and a table row claimed on
    # a tie would flip on every re-run. Ties resolve to "model" (no row).
    over = aggregate.get(key, {})
    if not over:
        return "model"
    best = max(over.values())
    for recipe in RECIPE_ORDER:
        if over.get(recipe, 0.0) == best:
            winner = recipe
            break
    else:
        return "model"
    if winner == "model" or min_gain <= 0.0:
        return winner
    second = max((v for r, v in over.items() if r != winner), default=0.0)
    if (best - second) < min_gain * best:
        return "model"
    return winner


@click.command()
@click.option(
    "--m-values",
    default="512,2048,4096",
    show_default=True,
    callback=lambda _c, _p, v: parse_positive_ints(v),
)
@click.option(
    "--shapes",
    "shape_values",
    multiple=True,
    help="NAME:N:K — the sweep's weight shapes (N, K); a small probe grid "
    "is used when omitted.",
)
@click.option(
    "--n-values",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Explicit N grid (used when --shapes is empty).",
)
@click.option(
    "--k-values",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Explicit K grid (used when --shapes is empty).",
)
@click.option("--batch", default=1, show_default=True, help="Batch dim b.")
@click.option(
    "--combos",
    default=",".join(COMBOS),
    show_default=True,
    callback=lambda _c, _p, v: tuple(
        part.strip() for part in v.split(",") if part.strip()
    ),
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Plan-table row file (ASTR_GEMM_TABLE=/path/to/this).",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=3, show_default=True)
@click.option(
    "--min-gain",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Minimum relative gain (%) a forced recipe must show over the "
    "runner-up to claim a row; smaller leads are treated as ties and keep "
    "the degraded bands. Ignored with --full-coverage (every band gets a row, "
    "ties resolve to the big>narrow>small preference).",
)
@click.option(
    "--full-coverage",
    is_flag=True,
    default=False,
    help="Emit a row for every (class, M/N band): production dispatch "
    "becomes the table alone (dispatch is table-only — no row is ever "
    "skipped), and each class ends with a catch-all row so no shape can "
    "miss the table.",
)
@click.option(
    "--band-search",
    is_flag=True,
    default=False,
    help="With --full-coverage: grid-search the M band cuts — DP over all "
    "possible partitions, each segment taking the recipe with the lowest "
    "total time and each cut paying --min-gain of the segment baseline "
    "(the measurement noise floor) instead of merge-adjacent-equals. "
    "Not combinable with --noise-merge.",
)
@click.option(
    "--noise-merge",
    is_flag=True,
    default=False,
    help="With --full-coverage: merge noise-level band flips — a point whose "
    "winner leads the runner-up by less than --min-gain follows its "
    "neighbors' recipe instead (run-to-run flips are measurement noise, "
    "and per-point winners chase them).",
)
@click.option(
    "--save-results",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Dump the raw per-point measurements to JSON (reproducible search: "
    "rebuild rows from it with --results-json).",
)
@click.option(
    "--results-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Skip the sweep: build rows from a JSON saved by --save-results.",
)
def plan_table_command(
    m_values: tuple[int, ...],
    shape_values: tuple[str, ...],
    n_values: tuple[int, ...] | None,
    k_values: tuple[int, ...] | None,
    batch: int,
    combos: tuple[str, ...],
    output: Path,
    warmup: int,
    iterations: int,
    trials: int,
    min_gain: float,
    full_coverage: bool,
    band_search: bool,
    noise_merge: bool,
    save_results: Path | None,
    results_json: Path | None,
) -> None:
    """Sweep every candidate recipe (interleaved) at the M x shape grid."""
    unknown = [combo for combo in combos if combo not in COMBOS]
    if unknown:
        raise click.BadParameter(f"unknown combos: {', '.join(unknown)}")
    if noise_merge and not full_coverage:
        raise click.BadParameter("--noise-merge needs --full-coverage")
    if band_search and noise_merge:
        raise click.BadParameter(
            "--band-search and --noise-merge are mutually exclusive"
        )

    if results_json is not None:
        results = json.loads(results_json.read_text())
    else:
        if shape_values:
            shapes = [parse_shape(value) for value in shape_values]
        elif n_values and k_values:
            shapes = [(f"n{n}k{k}", n, k) for n in n_values for k in k_values]
        else:
            shapes = [("n4096k4096", 4096, 4096), ("n14336k4096", 14336, 4096)]

        click.echo(
            f"--- interleaved sweep ({len(combos)} combos x {len(shapes)} shapes "
            f"x {len(m_values)} M x b={batch}, recipes measured per shape)"
            + ("; full coverage" if full_coverage else "")
        )
        results = sweep(m_values, shapes, combos, batch, warmup, iterations, trials)
        if save_results is not None:
            save_results.write_text(json.dumps(results))
            click.echo(f"saved measurements to {save_results}")

    rows = build_rows(
        results,
        min_gain=min_gain / 100.0,
        full_coverage=full_coverage,
        band_search=band_search,
        smooth=noise_merge,
    )
    header = (
        "# AOT dispatch rows: m_min m_max n_min n_max perf_class crosswise "
        "cta stages raster\n"
        "# (min, max] bands, 0 = open; perf_class 0..3 (W16A16/W8A16/W8A8/"
        "F8A8); crosswise 0 = NT; cta 0 small / 1 narrow / 2 big; raster 0 "
        "= auto.\n"
        "# The sweep times quant_gemm's fused-linear (NT) layout, so every "
        "row carries\n"
        "# crosswise 0: TT/TN shapes miss this table and take the degraded "
        "bands in C++.\n"
        "# Generated by csrc/bench/gen_plan_table.py; tune the grid then "
        "re-run.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n".join(rows) + ("\n" if rows else ""))
    click.echo(f"wrote {len(rows)} rows to {output}")
    click.echo(f"use: ASTR_GEMM_TABLE={output}")


if __name__ == "__main__":
    plan_table_command()
