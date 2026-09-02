# InfraSWE: multi-process DDP online rollout

This directory binds the DDP online-rollout change to AstrAI's repository
contract and the checked-in two-GPU NVIDIA L20 evidence. It uses InfraSWE's
`project-fit-system-path-v0.5.1` comparison and scoring models because this is a
training/inference lifecycle path, not a standalone kernel candidate.

InfraSWE v0.5's generic Draft document currently admits kernel and pure-Triton
formula identifiers only. Substituting a kernel formula would misclassify this
change, so the repository stores and validates the native
`ProjectComparisonCell` instead. The result remains explicitly diagnostic and
unsealed.

Before the PR was opened, commit
`f07ed0ae86cf1c09cf79499ce0d3cf59a77bca98` of InfraSWE was used to:

1. validate the comparison cell with the `ProjectComparisonCell` Pydantic model;
2. run the frozen system-path ProjectFit and BenchmarkTrust scoring functions;
3. run the Draft and system-path engine test subsets (41 tests); and
4. verify that official scoring stays unresolved without a seal, system-trace
   evidence, hidden probes, and a verified manifest.

The visible-evidence diagnostic score is **95.00/100** and BenchmarkTrust is
**97.40/100**. The evidence includes five fresh-process replays of the real
two-rank, 20-optimizer-step online GRPO test. Those timings measure replay
stability only; this correctness PR makes no throughput or latency improvement
claim. The complete inputs and rationale are stored in
`benchmarks/results/ddp_rollout_l20_sm89_infraswe_score.json`.
