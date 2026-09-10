"""Compress an AOT plan-table row file without changing its decisions.

A row is a rectangle in the (M, N) plane — ``m > m_min`` and
``m <= m_max`` (0 = unbounded), the same on N, plus exact-match keys on
perf_class / crosswise. The sweep emits one row per (N band, M segment), so a
recipe that wins across several adjacent N bands is spelled once per band
even when its M extent is identical, and every class ends with a catch-all
row that an already-tiling set of bands shadows. Both are pure spelling.

Two edits, each accepted only with a proof of safety rather than a
resemblance check:

  merge(i, j)  same keys and same recipe, the two rectangles abutting along
               one axis (equal extent on the other axis, contiguous bands on
               the merging axis), and — the condition that makes a
               non-adjacent merge legal — no row between them overlapping
               the later rectangle. Then every point the merged rectangle
               captures was already resolved to that recipe: a point in the
               earlier/absent-interception later rectangle matched i or j,
               and a point matching an earlier row still matches it first
               because the rows before i keep their order.
  drop(i)      verified by probing, not by algebra: the row goes only if the
               table resolves every probe point identically without it.

The final table is checked against the original over the probe grid — band
edges and their neighbours, powers of two, a dense low range and the LLM
weight shapes, crossed with the perf classes and crosswise counts a caller
can present (TT/TN miss every row here and fall to the degraded bands, which
is itself a decision the probe records).

Usage:
    python csrc/bench/compress_plan_table.py \\
        --input plan_table.txt --output plan_table_compact.txt [--report]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

INF = float("inf")

# Queries the decision function is checked over: every dtype class, and the
# crosswise-operand counts a caller can present. Every row this tool sees is
# crosswise 0, so the two non-zero counts are interchangeable; probing 0 and
# 1 pins both the "row applies" and "no row applies" behaviours.
PERF_CLASSES = (0, 1, 2, 3)
CROSSWISE = (0, 1)


@dataclass(frozen=True)
class Row:
    """One dispatch row: (min, max] bands, 0 = open upper bound."""

    m_min: int
    m_max: int
    n_min: int
    n_max: int
    perf_class: int
    crosswise: int
    cta: int
    stages: int
    raster: int

    @property
    def key(self) -> tuple[int, ...]:
        """Fields that decide whether two rows mean the same thing."""
        return (self.perf_class, self.crosswise, self.cta, self.stages, self.raster)

    @property
    def recipe(self) -> tuple[int, int, int]:
        return (self.cta, self.stages, self.raster)

    @property
    def m_band(self) -> tuple[int, int]:
        return (self.m_min, self.m_max)

    @property
    def n_band(self) -> tuple[int, int]:
        return (self.n_min, self.n_max)

    def matches(self, m: int, n: int, perf_class: int, crosswise: int) -> bool:
        """plan_row_for's test, wildcard keys included (-1 matches any)."""
        if self.perf_class != -1 and self.perf_class != perf_class:
            return False
        if self.crosswise != -1 and self.crosswise != crosswise:
            return False
        if m <= self.m_min or (self.m_max and m > self.m_max):
            return False
        return n > self.n_min and (not self.n_max or n <= self.n_max)

    def render(self) -> str:
        return (
            f"{self.m_min} {self.m_max} {self.n_min} {self.n_max} "
            f"{self.perf_class} {self.crosswise} {self.cta} {self.stages} "
            f"{self.raster}"
        )


def parse_rows(text: str) -> list[Row]:
    rows: list[Row] = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"expected 9 fields, got {len(fields)}: {line!r}")
        rows.append(Row(*(int(f) for f in fields)))
    return rows


def band_bounds(band: tuple[int, int]) -> tuple[float, float]:
    """(lo, hi] as half-open floats; 0 in the upper slot means +inf."""
    lo, hi = band
    return float(lo), (INF if hi == 0 else float(hi))


def overlaps(a: Row, b: Row) -> bool:
    """Do the two rectangles share any (M, N) point?"""
    a_m, a_n = band_bounds(a.m_band), band_bounds(a.n_band)
    b_m, b_n = band_bounds(b.m_band), band_bounds(b.n_band)
    return min(a_m[1], b_m[1]) > max(a_m[0], b_m[0]) and min(a_n[1], b_n[1]) > max(
        a_n[0], b_n[0]
    )


def resolve(
    rows: list[Row], m: int, n: int, perf_class: int, crosswise: int
) -> tuple[int, int, int] | None:
    """The recipe (cta, stages, raster) the planner resolves, or None.

    The recipe is the whole decision — plan_from_row reads the row's cta,
    stages and raster and nothing else — so a probe that lands on a different
    row of the same recipe is not a changed decision, which is exactly what a
    widened band is.
    """
    for row in rows:
        if row.matches(m, n, perf_class, crosswise):
            return row.recipe
    return None


def probe_points(rows: list[Row]) -> list[tuple[int, int]]:
    """Band edges and neighbours, powers of two, dense low range, LLM N/K."""
    values: set[int] = set(range(1, 257))
    for row in rows:
        for edge in (row.m_min, row.m_max, row.n_min, row.n_max):
            if edge:
                values.update((max(1, edge - 1), edge, edge + 1))
    values.update(1 << p for p in range(20))
    values.update((384, 768, 1536, 3072, 6144, 8192, 11008, 14336, 28672, 65536, 10**7))
    ordered = sorted(values)
    return [(m, n) for m in ordered for n in ordered]


def same_decisions(a: list[Row], b: list[Row], probes: list[tuple[int, int]]) -> bool:
    return all(
        resolve(a, m, n, pc, cw) == resolve(b, m, n, pc, cw)
        for m, n in probes
        for pc in PERF_CLASSES
        for cw in CROSSWISE
    )


def try_merge(rows: list[Row], i: int, j: int) -> Row | None:
    """The merged row for the pair, or None when the merge is not provable."""
    a, b = rows[i], rows[j]
    if a.key != b.key:
        return None
    if a.m_band == b.m_band and a.n_max and a.n_max == b.n_min:
        cand = Row(*a.m_band, a.n_min, b.n_max, *a.key)
    elif a.n_band == b.n_band and a.m_max and a.m_max == b.m_min:
        cand = Row(a.m_min, b.m_max, *a.n_band, *a.key)
    else:
        return None
    # A row between the two would take precedence over b's rectangle after
    # the merge, changing its decision.
    if any(overlaps(rows[k], b) for k in range(i + 1, j)):
        return None
    return cand


def merge_rows(rows: list[Row]) -> list[Row]:
    out = list(rows)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                cand = try_merge(out, i, j)
                if cand is None:
                    continue
                out = out[:i] + [cand] + out[i + 1 : j] + out[j + 1 :]
                changed = True
                break
            if changed:
                break
    return out


def drop_shadowed(rows: list[Row], probes: list[tuple[int, int]]) -> list[Row]:
    out = list(rows)
    for i in range(len(out) - 1, -1, -1):
        trial = out[:i] + out[i + 1 :]
        if same_decisions(rows, trial, probes):
            out = trial
    return out


@click.command()
@click.option(
    "--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True)
)
@click.option(
    "--output", required=True, type=click.Path(path_type=Path, dir_okay=False)
)
@click.option("--report", is_flag=True, help="Print the per-class row counts.")
def compress_command(input_path: Path, output: Path, report: bool) -> None:
    rows = parse_rows(input_path.read_text())
    probes = probe_points(rows)
    if not same_decisions(rows, rows, probes):
        raise click.ClickException("probe set is degenerate")

    merged = merge_rows(rows)
    compressed = drop_shadowed(merged, probes)
    if not same_decisions(rows, compressed, probes):
        raise click.ClickException("compression changed a decision; refusing to write")

    header = (
        "# AOT dispatch rows: m_min m_max n_min n_max perf_class crosswise "
        "cta stages raster\n"
        "# (min, max] bands, 0 = open; perf_class 0..3 (W16A16/W8A16/W8A8/"
        "F8A8); crosswise 0 = NT; cta 0 small / 1 narrow / 2 big; raster 0 "
        "= auto.\n"
        f"# Compressed from {input_path.name} by "
        "csrc/bench/compress_plan_table.py:\n"
        f"# {len(rows)} -> {len(compressed)} rows, decision-identical over "
        f"{len(probes)} probe points x {len(PERF_CLASSES)} classes x "
        f"{len(CROSSWISE)} crosswise counts.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n".join(r.render() for r in compressed) + "\n")
    click.echo(
        f"{len(rows)} -> {len(compressed)} rows (decision-identical); wrote {output}"
    )
    if report:
        for pc in sorted({row.perf_class for row in rows}):
            before = sum(1 for row in rows if row.perf_class == pc)
            after = sum(1 for row in compressed if row.perf_class == pc)
            click.echo(f"  perf_class {pc}: {before} -> {after} rows")


if __name__ == "__main__":
    compress_command()
