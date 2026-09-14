"""Offline capture harness for planner candidates.

Scores a planner *rule* against saved measurements without launching a
kernel: for every measured point, the rule's pick is compared with the best
measured recipe, and the per-class geomean of that ratio is the capture.
This is the gate a model change must pass before it touches gemm.cuh — the
2026-09-13/14 lessons (tc_eff, wave_eff, thermal-ordered diff rows) were all
analytic terms shipped without one.

    python csrc/bench/model_capture.py /tmp/tuned_step0b.json
    python csrc/bench/model_capture.py /tmp/tuned_step0b.json --rule bytes

Rules are functions of (tile geometry, problem shape, device facts) only —
never of the measurement — so a rule that scores well here can be written in
C++ verbatim. `resident` mirrors policy.cuh's min_ctas_for_ring with the
ring the sweep computes, plus the load-thread cap; `wave_eff` is the same
fill fraction the sweep's wave terms use.

Measured points come from `tune_plan_table.py sweep --save-results`; those
numbers are phase-ordered (see the workspace AGENTS.md on thermal bias), so
use this harness for RULE selection and the interleaved A/B for shipping
decisions.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import click
import torch

_TUNE = Path(__file__).resolve().parent / "tune_plan_table.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("tune_plan_table", _TUNE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpt = _load_tune()


def device_facts() -> dict:
    """The queried device geometry, the same fields the C++ prices against
    (common/device.cuh DeviceFacts). smem is the PER-SM figure and smem_max
    the PER-BLOCK opt-in ceiling: residency prices from the former, ring
    feasibility from the latter. Both are read, never written down — a
    hard-coded 100KB made every rule below mis-price on a part with a
    different smem/SM (228KB on Hopper/Blackwell datacenter)."""
    props = torch.cuda.get_device_properties(0)
    return {
        "sms": props.multi_processor_count,
        "threads": props.max_threads_per_multi_processor,
        "smem": props.shared_memory_per_multiprocessor,
        "smem_max": props.shared_memory_per_block_optin,
        "regs": props.regs_per_multiprocessor,
    }


def geometry(name: str) -> dict:
    m = tpt._FACTS_RE.fullmatch(name)
    bm, bn, kk, wm, wn, stages, _fast = m.groups()
    bm, bn, kk, wm, wn, stages = map(int, (bm, bn, kk, wm, wn, stages))
    # The 16-warp widening (warp_widened_t) doubles the load threads of a
    # small kk=64 tile on any pair with a 2-byte operand; a byte pair keeps
    # the 8-warp form. The rules below must see the widened shape or they
    # mis-price every small 2-byte candidate (that mistake cost the first
    # bytes-rule prototype its class-1 capture).
    threads = (bm // wm) * (bn // wn) * 32
    return {"bm": bm, "bn": bn, "kk": kk, "stages": stages, "threads": threads,
            "wm": wm, "wn": wn}


def priced(name: str, m: int, n: int, k: int, ba: int, bb: int, dev: dict) -> dict:
    g = geometry(name)
    # widening: 2-byte operand, small CTA, kk=64 (policy.cuh warp_widened_t)
    widened = (ba + bb >= 3 and g["bm"] == 64 and g["bn"] == 64 and g["kk"] == 64)
    threads = 512 if widened else g["threads"]
    ring = (g["stages"] + 1) * g["kk"] * (g["bm"] * ba + g["bn"] * bb)
    # policy.cuh: min_ctas_for_ring is bytes <= 48KB ? 2 : 1 — residency is
    # smem-capped ONLY (no thread term). An earlier version of this file
    # added its own thread cap and simulated picks the kernel never makes,
    # which is how a rule that scored +2.6pp here regressed 8 cells
    # end-to-end. Fidelity is now checked against the binding (--check).
    resident = min(dev["smem"] // ring, 2 if ring <= 48 * 1024 else 1)
    # Feasibility is the ring's, not a floor on residency: the C++'s
    # plan_resident_ctas returns 0 (not 1) for a ring past the opt-in
    # ceiling, and a rule that prices it anyway simulates a launch the
    # binding never makes. The rule loop skips ok=false.
    ok = resident > 0 and ring <= dev["smem_max"]
    blocks = ((m + g["bm"] - 1) // g["bm"]) * ((n + g["bn"] - 1) // g["bn"])
    waves = max(1, (blocks + dev["sms"] * max(resident, 1) - 1) // (dev["sms"] * max(resident, 1)))
    wave_eff = blocks / (waves * dev["sms"] * max(resident, 1))
    return {
        **g, "ring": ring, "resident": resident, "blocks": blocks, "ok": ok,
        "waves": waves, "wave_eff": wave_eff, "widened": widened,
        "threads": threads,
        # the dispatch key's first column, for orderings that prefer a class
        "cta": tpt.tile_facts(name)[0],
        # per-block operand bytes through L2 (the tile's own traffic; the
        # problem's total is this times blocks)
        "b2": k * (g["bm"] * ba + g["bn"] * bb),
        "fat": g["bm"] * g["bn"],
        "kiters": k // g["kk"],
    }


def rules() -> dict:
    """Candidate orderings: (name, key fn) — larger key wins. Each key fn
    takes (features, problem) so a rule may consult the shape, like the
    model's own byte-pair floor does."""
    def shipped(p, q):
        # gemm.cuh ModelPlanner, mirrored term for term: cost_of() is
        # (operand bytes + output bytes) * waves * resident — DeepGEMM's
        # max(L1,L2)/wave_efficiency collapsed, see the C++ comment. Byte
        # pairs rank on the cost alone (their candidates all tie on
        # residency); every other pair ranks (resident, stages, cost).
        cost = (p["b2"] + 2 * p["fat"]) * p["waves"] * p["resident"]
        if q["ba"] == 1 and q["bb"] == 1:
            return (-cost,)
        return (p["resident"], p["stages"], -cost)

    return {
        "model_exact": shipped,
        # the pre-floor ranking, to size the floor rule's contribution
        "resource": lambda p, q: (p["resident"], p["stages"]),
        # fattest tile that still fills, fill-first when nothing fills
        "fat_fill": lambda p, q: (p["fat"] if p["wave_eff"] >= 0.5 else 0.0,
                               p["wave_eff"]),
        # fill-weighted harmonic tile size: bigger tiles amortise traffic,
        # fill decides how much of the tile's work overlaps
        "fill_harm": lambda p, q: p["wave_eff"] * p["fat"] / max(1, p["bm"] + p["bn"]),
        # traffic per unit fill: minimise bytes for the fill achieved
        "traffic_fill": lambda p, q: p["wave_eff"] / max(1, p["b2"]),
        # resident weighted by how compute-rich the tile's ring is
        # C++ mirrors this exactly: the product, then ring depth
        "density": lambda p, q: (p["resident"] * p["kk"], p["stages"]),
        # traffic alone, fill as a tie-break (no constants at all)
        "traffic": lambda p, q: (p["wave_eff"], -p["b2"]),
    }


@click.command()
@click.argument("results_json", type=click.Path(exists=True, path_type=Path))
@click.option("--rule", default=None, help="Only this rule (default: all).")
@click.option("--class-filter", default=None, type=int,
              help="Only this GemmPerfClass id.")
@click.option("--check", is_flag=True, default=False,
              help="Fidelity: compare each rule's pick with ops.gemm.probe on "
                   "the measured points (the model rule must match 100%).")
def main(results_json, rule, class_filter, check):
    dev = device_facts()
    by_point: dict[tuple, dict[str, float]] = defaultdict(dict)
    for point in json.loads(results_json.read_text()):
        key = (point["combo"], point["n"], point["k"], point["m"])
        by_point[key][point["recipe"]] = point["tflops"]

    table = rules()
    if rule:
        table = {rule: table[rule]}
    names = sorted(table)

    check_seen: dict[str, list[str]] = defaultdict(list)
    captures: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (combo, n, k, m), over in by_point.items():
        cands = [r for r in over if r != "model"]
        if not cands:
            continue
        perf_class = tpt.PERF_CLASS[combo]
        if class_filter is not None and perf_class != class_filter:
            continue
        best = max(over[r] for r in cands)
        ba, bb = tpt.BYTES[combo]
        ctx = {"m": m, "n": n, "k": k, "ba": ba, "bb": bb}
        feats = {r: priced(r, m, n, k, ba, bb, dev) for r in cands}
        # order_for mirrors dispatch_tile's first-match tie-break
        order = tpt.order_for(perf_class)[:-1]
        for name in names:
            key = table[name]
            pick, pick_key = None, None
            for r in order:
                # A candidate the device cannot launch is not a candidate
                # (C++'s price() gate): pricing it would simulate a pick the
                # binding never makes.
                if r not in feats or not feats[r]["ok"]:
                    continue
                v = key(feats[r], ctx)
                if pick is None or v > pick_key:
                    pick, pick_key = r, v
            if pick is None:
                continue
            captures[(name, perf_class)].append(over[pick] / best)
        captures[("model(measured)", perf_class)].append(over.get("model", 0.0) / best)

    if check:
        from astrai.extension import ops

        ops.gemm.set_table("")
        ops.gemm.inject_rows("")
        ops.gemm.set_planner("model")
        seen: set[tuple] = set()
        for (combo, n, k, m) in sorted(by_point):
            perf_class = tpt.PERF_CLASS[combo]
            if class_filter is not None and perf_class != class_filter:
                continue
            if (combo, n, k, m) in seen:
                continue
            seen.add((combo, n, k, m))
            act, weight = tpt.COMBOS[combo]
            info = ops.gemm.probe(m, n, k, act, weight)
            real = (info["cta"], info["stages"], info["kk"])
            ba, bb = tpt.BYTES[combo]
            ctx = {"m": m, "n": n, "k": k, "ba": ba, "bb": bb}
            # The binding chooses among the whole ladder, so the fidelity
            # comparison must too. Restricting to the recipes this dataset
            # measured reported every cell where the model picks a tile the
            # dataset predates as a mismatch (meas_w16a16 has no tall entry).
            cands = list(tpt.REACHABLE[tpt.ladder_for_widths(ba, bb)])
            feats = {r: priced(r, m, n, k, ba, bb, dev) for r in cands}
            order = tpt.order_for(perf_class)[:-1]
            for name in names:
                key = table[name]
                pick, pick_key = None, None
                for r in order:
                    if r not in feats or not feats[r]["ok"]:
                        continue
                    v = key(feats[r], ctx)
                    if pick is None or v > pick_key:
                        pick, pick_key = r, v
                if pick is None:
                    continue
                got = tpt.RECIPES[pick]
                if got != real and len(check_seen[name]) < 5:
                    check_seen[name].append(
                        f"combo={combo} m{m} n{n} k{k}: harness {got} vs binding {real}")
        ops.gemm.set_planner("")
        click.echo("fidelity vs the binding (mismatches, first 5 per rule):")
        for name in names + ["model(measured)"]:
            bad = check_seen.get(name, [])
            status = "n/a" if name == "model(measured)" else ("MATCH" if not bad else f"{len(bad)}+")
            click.echo(f"  {name:16s} {status}")
            for line in bad:
                click.echo(f"      {line}")
        return

    classes = sorted({c for _n, c in captures})
    header = "rule".ljust(16) + "".join(f"class{c:>8d}" for c in classes)
    click.echo(header)
    for name in names + ["model(measured)"]:
        cells = []
        for c in classes:
            vals = captures.get((name, c), [])
            if not vals:
                cells.append("       -")
                continue
            gm = math.exp(sum(math.log(max(v, 1e-3)) for v in vals) / len(vals))
            cells.append(f"{gm:8.3f}")
        click.echo(name.ljust(16) + "".join(cells))
    click.echo("\npoints per class: " + ", ".join(
        f"class{c}={len(captures[('model(measured)', c)])}" for c in classes))


if __name__ == "__main__":
    main()
