"""Map the dispatch LOGIC over a dense (m, n, k) grid — no kernel launches.

The planner chain is host-only by design (plan_probe in gemm.cuh), so
"which recipe serves this cell" is an analytic question: probing a full
stride grid costs seconds where measuring one costs ~10s/cell (and a
batch-ordered measurement of that size is the thermal-bias trap the
workspace AGENTS.md documents). This tool answers logic questions —
decision diversity, builtin-row coverage, where along m the pick flips,
whether k moves the pick at all — and emits a CSV when the per-cell
decisions are wanted (e.g. diffing two planner modes or two tables).

    python csrc/bench/dispatch_grid.py --step 256 --max 4096
    python csrc/bench/dispatch_grid.py --step 512 --max 4096 --combos w16a16 \
        --planner model --csv /tmp/logic_model.csv

The measurement side stays what it always was: sparse named shapes
(tune_plan_table.py sweep), then interleaved A/B for anything that ships
(diff_rows.py). A dense grid here nominates band boundaries; it proves
nothing about performance.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

import click
import torch

from astrai.extension import ops

COMBOS = {
    "w16a16": (torch.bfloat16, torch.bfloat16),
    "w8a16": (torch.bfloat16, torch.int8),
    "w8a8": (torch.int8, torch.int8),
    "f8a8": (torch.float8_e4m3fn, torch.float8_e4m3fn),
}


def grid(step: int, hi: int) -> list[int]:
    return list(range(step, hi + 1, step))


def recipe_name(probe: dict, names: list[str]) -> str:
    return f"{names[probe['cta']][1:]}_s{probe['stages']}_kk{probe['kk']}"


@click.command()
@click.option("--step", default=256, show_default=True, help="Grid stride.")
@click.option("--max", "hi", default=4096, show_default=True, help="Grid top.")
@click.option("--combos", default=",".join(COMBOS), show_default=True)
@click.option(
    "--planner",
    default="hybrid",
    show_default=True,
    type=click.Choice(("hybrid", "model", "table")),
    help="Which chain to map (the shipped default is hybrid).",
)
@click.option("--csv", "csv_path", default=None, type=click.Path(path_type=Path))
def main(step: int, hi: int, combos: str, planner: str, csv_path: Path | None):
    # A clean logic map: the shipped builtin rows, no runtime override or
    # injected tier shadowing them (model_capture's --check does the same).
    ops.gemm.set_table("")
    ops.gemm.inject_rows("")
    ops.gemm.set_planner(planner)
    names = ops.gemm.get_module("gemm").tile_class_names()

    rows_out: list[dict] | None = (
        [] if csv_path is not None else None
    )
    values = grid(step, hi)
    for combo in (c for c in combos.split(",") if c):
        act, weight = COMBOS[combo]
        decisions: Counter[tuple[str, str]] = Counter()
        # m boundaries where any (n, k) slice flips its pick, and whether k
        # moves the pick at all for a fixed (m, n).
        m_flips: Counter[int] = Counter()
        k_varies = 0
        prev_by_nk: dict[tuple[int, int], tuple[str, str]] = {}
        for m in values:
            for n in values:
                pick_by_k: dict[tuple[str, str], int] = {}
                for k in values:
                    d = ops.gemm.probe(m, n, k, act, weight)
                    dec = (d["source"], recipe_name(d, names))
                    decisions[dec] += 1
                    pick_by_k[dec] = k
                    if dec != prev_by_nk.get((n, k)):
                        if (n, k) in prev_by_nk:
                            m_flips[m - step // 2] += 1
                        prev_by_nk[(n, k)] = dec
                if len(pick_by_k) > 1:
                    k_varies += 1
        cells = len(values) ** 3
        print(f"=== {combo}  planner={planner}  cells={cells}")
        print(f"    distinct decisions: {len(decisions)}")
        for (source, recipe), count in decisions.most_common(8):
            print(f"    {count:6d} ({100 * count / cells:5.1f}%)  {source:8s} {recipe}")
        if len(decisions) > 8:
            print(f"    ... {len(decisions) - 8} more")
        flips = sorted(m_flips)
        if flips:
            per_slice = [m_flips[f] for f in flips]
            print(
                f"    m-flip edges (any n,k slice): {len(flips)} unique,"
                f" median slices/edge {statistics.median(per_slice):.0f},"
                f" max {max(per_slice)}"
            )
            print(f"      {flips}")
        print(
            f"    (m,n) cells whose pick varies with k: "
            f"{k_varies}/{len(values) ** 2}"
        )
        if rows_out is not None:
            for m in values:
                for n in values:
                    for k in values:
                        d = ops.gemm.probe(m, n, k, act, weight)
                        rows_out.append(
                            {
                                "combo": combo,
                                "m": m,
                                "n": n,
                                "k": k,
                                "source": d["source"],
                                "recipe": recipe_name(d, names),
                                "raster": d["raster"],
                            }
                        )
    ops.gemm.set_planner("")  # restore the shipped default
    if rows_out is not None and csv_path is not None:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        click.echo(f"wrote {len(rows_out)} rows to {csv_path}")


if __name__ == "__main__":
    main()
