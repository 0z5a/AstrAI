# InfraSWE Draft: fused BF16 SwiGLU

This directory binds the fused BF16 SwiGLU benchmark to AstrAI as an explicit
repository target. AstrAI is not one of InfraSWE v0.5's pinned built-in
projects, so using a default project would evaluate the change against the
wrong host contract. The Draft remains `D3-contract-proposed`; it does not
claim maintainer review, sealing, hidden-probe completion, or official
ProjectFit.

Before this PR was opened, the Draft was validated and resolved with InfraSWE
commit `811bc775ed5b3a6ec853219245f3469f78818020`:

```bash
PYTHONPATH=src .venv/bin/infraswe draft validate \
  /path/to/AstrAI/benchmarks/infraswe/astrai-swiglu-draft.json

PYTHONPATH=src .venv/bin/infraswe draft resolve \
  --local-draft \
  /path/to/AstrAI/benchmarks/infraswe/astrai-swiglu-draft.json \
  --output /tmp/astrai-swiglu-resolution.json
```

The candidate and contract digests bind the ordered source, tests,
documentation, operator results, engine log, and greedy checkpoint probes. The
required comparison cell is one NVIDIA L20 (`sm_89`); optional A100 and H100
cells are explicitly untested. Compilation happens before timed cases.

Applying InfraSWE's frozen `project-fit-kernel-v0.5` formula to the visible
evidence gives a diagnostic ProjectFit of **90.95/100** and BenchmarkTrust of
**95.87/100**. The rationale is machine-readable in
`benchmarks/results/swiglu_l20_sm89_infraswe_score.json`. Both scores are
non-official. Official ProjectFit remains unresolved until the Draft is
sealed, at least five fresh-process replays and system traces exist, hidden
probes are complete, and the evidence manifest is verified.
