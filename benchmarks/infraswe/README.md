# InfraSWE Draft: common-shape BF16 GEMV

This directory binds the common LLaMA/GPT-NeoX GEMV benchmark to AstrAI as an
explicit repository target. AstrAI is not one of InfraSWE v0.5's built-in
projects, so selecting a built-in default would score the change against the
wrong host contract. The Draft remains `D3-contract-proposed`; it does not
claim maintainer review, sealing, hidden-probe completion, or an official
ProjectFit.

The Draft was validated and resolved before opening the PR with InfraSWE commit
`811bc775ed5b3a6ec853219245f3469f78818020`:

```bash
PYTHONPATH=src .venv/bin/infraswe draft validate \
  /path/to/AstrAI/benchmarks/infraswe/astrai-gemv-common-draft.json

PYTHONPATH=src .venv/bin/infraswe draft resolve \
  --local-draft \
  /path/to/AstrAI/benchmarks/infraswe/astrai-gemv-common-draft.json \
  --output /tmp/astrai-gemv-common-draft-resolution.json
```

The candidate and contract digests bind the ordered source, tests,
documentation, and raw benchmark evidence. The required comparison cell is one
NVIDIA L20 (`sm_89`); optional A100 and H100 cells are explicitly untested.
Compilation happens before timed cases. The checked-in kernel suite uses 21
paired/interleaved samples, while the distinct-weight synthetic chains include
the guarded Python dispatcher. Neither is presented as whole-model throughput.

Applying InfraSWE's frozen `project-fit-kernel-v0.5` formula to the visible
evidence yields a diagnostic ProjectFit of **91.68/100** and BenchmarkTrust of
**95.87/100**. The machine-readable rationale is
`benchmarks/results/gemv_common_l20_sm89_infraswe_score.json`. Both numbers are
non-official: official ProjectFit remains unresolved until the Draft is sealed,
five or more fresh-process replays and system traces exist, hidden probes are
complete, and the evidence manifest is verified.
