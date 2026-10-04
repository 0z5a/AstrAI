# GEMM / Linear Kernel

> Part of the [operator docs](README.md); required reading before changing
> the tile vocabulary, the planners, or the fp8 recipes.

The `quantize` family accelerates bf16 linear layers by
quantizing to FP8 and running tensor-core GEMMs (**requires sm_89+**; fp8
`mma.sync.m16n8k32` only exists on Ada/Hopper). The GEMM device code is
split CUTLASS-style into one layered directory:

| File | Role |
|------|------|
| `api/quantize_common.h` | capability helpers (`sm_at_least`, `kMinSmForFp8`) + `QuantLayout` + `QuantParams` POD — raw `__nv_fp8_*` element types, no format enum, no torch |
| `kernel/quantize.cuh` | pure-CUDA device code: vectorized `fp8_quantize_kernel` + 64×32-tile transpose kernel (out_layout 0/1/2, Dual orientation a template param), `fp8_cvt_traits<Fp8T>` convert + `quant_in_traits<InT>` unpack (primary templates undefined — one specialization per dtype/format) — no torch |
| `datatype/dequant.cuh` | in-register dequantization functors (`DequantPair<SrcT, MmaT>`): the exact int8→bf16 expansion quantized-GEMM operands fold between the smem read and the mma |
| `api/gemm_common.h` | dtype-neutral GEMM family declarations: layout tags, `gemm_elem_traits<T>` (kBytes — the smem ring budgets; the MMA K extent rides `MmaShapeFor<MmaT>`), `gemm_mma_traits<ElemA, ElemB>` (MmaT promotion + per-operand kDequantA/B), `GemmParams` POD |
| `policy/traits.cuh` | Promoted MMA traits and shared-memory ring budget (`GemmTraits`, `GemmSmem`) |
| `policy/manifest.cuh` | Named tile recipes, CTA classes, and staging-specific manifests |
| `policy.cuh` | `GemmPolicy`: the kernel's composed dtype, layout, tile, staging, and output policy |
| `launcher/plan_types.h` | Runtime config, planner query, recipe, and dispatch decision shared by launch and planning code |
| `memory/load_async.cuh` / `load_crosswise.cuh` / `load_crosswise_packed.cuh` | Operand staging by access pattern: cp.async (congruous and 16-bit transposed) with `PrefetchCarry`; direct 8-bit crosswise LDG+PRMT with `CrosswiseCarry`; packed k-pair crosswise with `PairPackCarry` |
| `scheduler.cuh` | CTA id → (block_m, block_n) grouped/plain raster (runtime `raster` knob) |
| `kernel/gemm_mainloop.cuh` | `GemmCollectiveMainloop`: stage rings, stage loads, fragment addressing (ldmatrix + dequantized scalar paths), pipelined mma.sync loop |
| `epilogue/writer.cuh` | `GemmCollectiveEpilogue`: fused bias + per-row/per-channel scale folding + bf16/fp32 smem scatter + coalesced copy-out |
| `kernel/gemm.cuh` | Device entry kernels for cp.async and TMA staging; they compose the mainloop and epilogue |
| `launcher/gemm_launch.cuh` | Typed CUDA launch, TMA descriptor setup, and the `GemmParams` to `PlanQuery` conversion |
| `launcher/gemm_tiles.cuh` | Manifest tile selection, output-reclaim fallback, and TMA/cp.async policy resolution |
| `launcher/gemm_dispatch.cuh` | Layout canonicalization and tag routing shared by `gemm_dispatch` and `plan_probe_for`; the include used by dtype-pair instantiation units |
| `launcher/plan_row.h` / `plan_table_parse.h` / `plan_table_builtin.h` | Row vocabulary and matching; row-file and runtime-text parsing; measured device-specific rows and the degraded fallback ladder, respectively |
| `launcher/plan_table.h` | `RowSource` containers and runtime config seed; includes the row, parser, and builtin headers for existing callers. The planners (`RowSetPlanner`, `ModelPlanner`) live in `launcher/planning.h` |
| `launcher/planning.h` | The planner chain: `RowSetPlanner` (rows from one `RowSource`), `ModelPlanner` (the cost-ranked analytical planner), the rank-ordered chain assembly and the crosswise-ladder / raster L2-budget rules. Single-inclusion impl header (one TU per binary) |
| `api/gemm.h` | The family's C++ surface — declarations only, and template-free so including it instantiates no dtype-pair kernel: `quant_gemm_impl` (the one GEMM entry), the planner face (`PlanProbe` + `plan_probe`, `GemmConfigPatch` — `rows` + `tier` (`RowTier`) + `table_off` + the mode/log/staging knobs — with the re-installable `GemmConfigState`, through `configure` / `config_state`) and the vocabulary (`tile_vocabulary` / `tile_class_names`). No Python type in a signature — the composed fp8 linear and the bindings TU call the same functions |
| `gemm/gemm.cu` | The typed host layer: the dtype-pair registry (`ASTRAI_GEMM_PAIRS`, one entry feeding both the `gemm_dispatch` and the `plan_probe_for` lookup; one extern-template declaration per pair, which is what keeps this TU from re-instantiating them) plus the `api/gemm.h` implementations. Holds no `py::` type |
| `gemm/fp8_linear.cu` | Composed fp8 training linear (forward and backward) in one C++ `autograd::Function` |
| `gemm/fp8_runtime.cu` / `fp8_linear.h` | Single `State` owner, checkpoint/debug and pybind interface; internal declarations connecting the runtime to the autograd entry |
| `gemm/fp8_ring.h` / `fp8_cache.h` / `fp8_state.h` | Delayed-scaling ring and recipe; weight/activation cast caches; meta registry and restore helpers |
| `gemm/bindings.cu` | Quantized GEMM and planner pybind surface: None-tolerant argument marshalling, state/config dict contracts and `PYBIND11_MODULE`; calls `bind_fp8` from `fp8_runtime.cu` |
| `csrc/quantize/bindings.cu` | pybind surface only → module `quantize` |
| `csrc/quantize/entry.cu` | the entry implementation: run_quantize + ring binding + dtype dispatch (also compiled into the gemm module so fp8_linear shares the chain) |

Scale semantics: `quantize` takes the quantization *multiplier*; the
strategy layer passes `scale.reciprocal()` and the kernel multiplies by it.
`quant_gemm` takes per-operand dequant scales (`a_scale`, `b_scale`;
the fp8 training path passes `sa` / `sb` separately). `amax` is produced
only by the delayed-scaling ring fold (the in-kernel fused reduction) or
measured by the caller — the plain no-ring quantize runs a pure scale+cast
and returns `amax=None`.

Python layer (two levels): `astrai/extension/kernel/quantize.py` and
`kernel/gemm.py` are the stateless kernel adapters (plain `quantize` /
`quantize_dual` / `quant_gemm` wrappers, one adapter file per compiled kernel
module), and `astrai/extension/quantize.py` is the strategy layer (fp8
recipes, delayed / dynamic scaling, `fp8_autocast`, plus the int8
quantizers). The composed fp8 linear — quantize, ring fold/advance, the
weight-cast cache and all three GEMMs — lives in C++
(`csrc/gemm/fp8_linear.cu`, compiled into the `gemm` module
where the GEMM dispatch state lives) behind a single Python->C++ crossing;
the strategy layer routes `aten::linear` to it on CUDA. The aten override
installs lazily on the first fp8 activation (autocast enter or the global
enable), so importing the module is dispatcher-neutral.

The fp8 training math contract — the scaled-cast formula, the delayed vs
dynamic scaling recipes, `quantize_dual`, and the forward/backward dataflow
formulas that consume these operands — lives in
[quantize.md](quantize.md); the sm_120 block_scale mma cell is covered in
the design notes below.

## The entry ladder

`gemm/entry.h` is the op entry in one place, a fixed ladder:
classify the dtype pair → gate the device (`check_fp8_device`) →
resolve the dequant scales (`resolve_quant_scale`) → resolve the operand
layouts and leading dims → validate the geometry → allocate the output →
fill `GemmParams` → dispatch. `gemm.cu`'s `quant_gemm_impl` is the thin
wrapper that instantiates the ladder with the family's dtype-pair lookup
(passed in as a template parameter, so the header carries no include-order
contract). The pybind spelling is `bindings.cu`'s.

Layout resolution: the user flag names the math (0 = last two dims are
`[rows][contract]`, 1 = transposed); a col-major *view* of a contiguous
buffer folds into the returned dispatch flag at zero copy — the kernel's
`LayoutA`/`LayoutB` tags cover both storages, and m/n/k derive from the
user flag only.

Degenerate geometry: `k == 0` is the empty sum — zero mainloop iterations,
so the epilogue writes zero, plus bias when given. `m == 0` / `n == 0`
return an empty result without launching (a zero grid extent is an illegal
launch); the guard sits at the end of the entry ladder, so an empty call
still validates its configuration. Both are pinned in
`tests/extension/test_w8.py` (`TestQuantGemmDegenerate`).

## Device design notes

**Staging layouts and swizzle.** Staging layouts are CUTLASS-style *types*:
`utils/swizzle.cuh` provides the `Swizzle<Bits, Shift>` /
`Layout<Shape<Rows, Chunks>, Stride<Chunks, 1>>` /
`composition(Swizzle, Layout)` vocabulary (16B-chunk units,
dtype-independent `(Bits, Shift)` pairs), and each collective declares its
tile's layout once — `SmemLayoutA/B` and the trans mirrors in the mainloop,
`OutLayout` in the epilogue. The family instances: congruous 2B staging is
the TMA 128B mode `<3,3>`, 1B is `<2,3>`, 16-bit trans staging swizzles
chunks by the k-row bits (custom `<log2(min(chunks,8)), log2(chunks)>`),
and the epilogue output keeps its row-width mode. The tensor layer
dispatches to `ComposedLayout::operator()` (the closed two-coordinate
form) — the 16B chunk index XORed with row bits at
`[3, 3+log2(kChunks))` — so a warp's ldmatrix fragment load (8 consecutive
rows × 16B) hits all 32 banks exactly once (the unswizzled row word-stride
is `kK/4` words, so rows `r` and `r + 8/kChunks` collide mod 32). Chunks
stay contiguous, so cp.async staging is unaffected. The helper derives the
XOR term from the row coordinate alone (the layout's `kRowShift`/`kMask`)
rather than the linearized index — keep the two-coordinate form in the hot
paths.

**Tensor vocabulary** (`utils/tensor.cuh`, cute's `Tensor<Engine, Layout>`).
ONE tensor type — storage and addressing are its two template parameters,
and every operation dispatches to a layout op; use sites spell
`Tensor<...>` directly, with no second names. Engines: `PtrEngine<T>`
(smem/gmem) and `ArrayEngine<T, N>` (the mma fragment cells; `MmaOp` names
them `AFrag`/`BFrag`/`CFrag`, and the typed `fma`/`ldmatrix` overloads take
them by reference so the registers stay in place). Layouts: the
`ComposedLayout` instances of `utils/swizzle.cuh` (16B chunk grids,
dtype-blind; `chunk_of` is the swizzled-chunk op, and the tensor scales the
row/chunk terms separately in 32-bit), `RingLayout` (slot rotation over a
per-stage grid) and `CellLayout` (element-unit (m, n) grid). A staged tile
is `Tensor<PtrEngine<Elem>, ComposedLayout>`; the stage ring adds the slot
dimension (`make_ring` / `stage_of`); the warp's accumulator is
`Tensor<ArrayEngine<CFrag>, CellLayout>` — `*acc(mt, nt)` at the fma seam,
the kPairB x4 fold is `BFragPair::cell`. Every method folds away at -O3.

**Boundary predication.** Predicated staging rides cp.async's runtime
source size (CUTLASS 2.x's zfill iterators): `cp.async.cg [dst], [src],
16, src_size` with `src_size` derived per chunk from the remaining
contract extent — 0 reads nothing and the hardware zero-fills the 16B
chunk, a partial size covers the k tail, and only a misaligned base
(non-16B `ld`) keeps the scalar-copy fallback.

**Fragment addressing (base-pair scheme).** One base register per operand
per k_seg, every fragment offset an LDSM immediate: for A,
`addr(s, mt) = lane_base + mt*(16*kK) ^ (s<<5)`; for B,
`addr(s, nt) = lane_base + nt*(8*kK) ^ (s<<5)`. The closure holds because
the XOR swizzle's source bits come only from the lane's row-within-matrix
`r7` — the 8/16-row fragment steps never reach them. Steady-state read
pointers advance one stage per iteration with an equality wrap instead of a
per-k-tile `(tile % ring) * stage_bytes` recomputation.

**Pipeline and barriers (cp.async).** Every operand ring holds
`kStages+1` buffers: the load for tile `i+kStages` targets slot
`(i-1)%(kStages+1)`, which compute(i-1) finished reading before this
iteration's barrier — no post-compute barrier, one `__syncthreads` per
k-tile. Prologue and tail commits are unconditional so the group sequence
stays tile-indexed and the fixed `wait_group<kStages-1>` is
iteration-invariant. The final iteration carries no trailing barrier of its
own, so the epilogue transition pays one explicit drain
(`PipelineSync<Stages>::drain()`, which empties only the calling thread's
groups) followed by `__syncthreads()` — without it a thread racing into the
epilogue scatters the output tile over peers' still-in-flight staging
writes and final fragment reads. The ring discipline runs through the
`PipelineSync` stage-pipeline type (`memory/pipeline.cuh`), with the raw
mbarrier PTX sites (init / arrive_expect_tx / wait_parity) shared with the
TMA path below.

**TMA staging** (sm_90+, dual-congruous operands, `memory/tma.cuh`). The
staging swizzles ARE the TMA hardware modes (`<3,3>` = SWIZZLE_128B for
2-byte elements, `<2,3>` = SWIZZLE_64B for 1-byte), so fragment addressing,
ring slots and the epilogue reclaim are untouched — only the load/wait
discipline changes. `TmaSwizzleOf<Staged>` derives the swizzle width and
the box's inner extent from the declared `ComposedLayout` (the encoder
decodes the `CUtensorMap` swizzle enum from that width), and
`tma_spec<Elem, Staged, BoxRows>` fills only the runtime geometry — the map
cannot drift from what the fragments read. Each operand's rank is a
template bit on `GemmTmaContext` / `gemm_kernel_tma` (strided batch = 3D
emitter, broadcast = shared 2D map), so the per-stage 2D/3D issue pick
compiles away; the launcher dispatches the four rank combinations. One
elected thread arms a per-slot mbarrier with `arrive.expect_tx` and issues
the boxes; OOB coordinates zero-fill, which absorbs the k tail and edge
tiles the cp.async zfill iterators predicated per chunk. Two TMA-specific
invariants: the swizzle applies to the ABSOLUTE shared address, so the ring
base rounds up to 1024B (budgeted in `Policy::kSmemBytes` together with the
barrier array); and the pipeline is the CUTLASS `PipelineTmaAsync`
handshake — `full[slot]` (count 1, expect_tx) for arrival, `empty[slot]`
(count = CTA threads, every consumer arrives after its last fragment read)
for release — which replaces the per-k-tile `__syncthreads`: warps skew
freely across slots and the producer's overwrite gate is the empty phase
alone. Descriptors are host-encoded (`cuTensorMapEncodeTiled` via dlsym —
no link-line changes) and cached exact-match, so steady-state calls pay the
few-microsecond encode once. The planner gates TMA on the device's compute
capability, dual-congruous layouts, 1-/2-byte dtypes and descriptor
encodability (misaligned base/ld falls back to the cp.async twin, which
stays compiled); `set_staging(tma=False)` forces the fallback.

**Stage depth.** The manifest carries s3 deep-ring siblings of every class
(thinner operand pairs leave more smem headroom under the budget). A row
names its ring depth directly and is smem-gated like any row; candidates
are measured as one-row tables (`set_table`, see
`csrc/bench/tune_plan_table.py sweep`).

**MMA cells and dtype promotion.** The same mainloop serves every dtype
pairing through one promotion rule (`gemm_mma_traits`): the mma runs on the
**MmaT** — symmetric fp8 keeps its native `m16n8k32`, symmetric bf16
(W16A16) passes through untouched, symmetric int8 (W8A8) keeps its native
`m16n8k32.s8.s8.s32` (int32 accumulators; `.satfinite` clamps the wrap
all-max-magnitude K≈16k inputs could reach), and a *lone* int8 (W8A16
weight-only or the mirrored A8W16) promotes to bf16 `m16n8k16` with
per-operand in-register dequant (`kDequantA`/`kDequantB` — W8A16 only B).
Staging never changes: int8 operands ride the existing congruous cp.async /
crosswise PRMT paths into the canonical swizzled tiles, and `kMmaK` follows
the mma cell (the `MmaShapeFor` trait) so the tile geometry is shared — the
native s8 pair reuses the fp8 k32 fragment layouts verbatim (1-byte dtypes
share the packed two-per-b16-slot layout, so ldmatrix addressing is
identical).

**Mma trait layer** (`mma/mma.cuh`). The instruction vocabulary assembles
from two specializable traits: `MmaShapeFor<Dtype>` maps an input dtype to
its instruction `Shape<M, N, K>` (primary template undefined — a dtype with
no MMA is a compile error; K follows the 256-bit A-fragment invariant: bf16
k16, 1-byte dtypes k32; `kMinArch` encodes each instruction's hardware
floor as a build-time assert), and `MmaOp<A, B, Shape>` — one
specialization per instantiated pairing cell carrying the accumulator type,
register counts and the dedicated asm block. `mma_sync<InT>` remains as the
fp32-family convenience view (attention + tests); the gemm mainloop calls
`Traits::MmaOp::fma` directly so the accumulator type rides the cell (fp32
for float families, s32 for s8). `Shape` itself lives in `utils/shape.cuh`.

**Dequant** (`datatype/dequant.cuh`). Each fragment register pair costs one
`LDS.16` + four LOP3-class instructions, exact for the full int8 range
including −128. bf16 carries only 7 mantissa bits, so the naive
"OR the byte into a bf16 base" trick (0x6400-style, exact for fp16) breaks
linearity — bit 7 spills into the exponent. Instead the magnitude bits
(0-6) and the sign bit (7) take separate LOP3s:
`h = (u & 0x7F) | 0x4300` → exactly 128+u7; `s = (u & 0x80) | 0x4300` →
128 or 256 as the sign picks; `v = h - s` is the exact int8 value, and
every intermediate is bf16-exact.

**MMA cell (sm_120a block_scale).** The plain warp-level fp8 mma
(`m16n8k32.e4m3/e5m2`) decodes at HALF rate on sm_120. The
`kind::mxf8f6f4` block_scale variant
(`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X…ue8m0`) runs
at the full rate with an IDENTICAL A/B/C/D register contract, so the
mainloop/staging/epilogue layers are untouched: `MxMmaOp` (mma/mma.cuh)
swaps only the cell, carrying a constant unit scale (every ue8m0 scale byte
0x7f = 2^0, selectors inert — the scale-factored product IS the plain
product; quant_gemm_test output stays byte-identical). Warp-level
block_scale is an sm_120-FAMILY instruction (CUDA 13.0 ptxas: 120a/121a/
120f accepted, 100a/103a/110a rejected — datacenter Blackwell does MX
through tcgen05, which needs 13.1+), so the cell gates on the family pass
(`ASTRAI_ARCH_FAMILY >= 1200`, mma/mma.cuh's value-macro twin of
`ASTRAI_DEVICE_ARCH`) and the gemm fatbin carries the sm_120a image (the
arch token names it — `ASTRAI_CUDA_ARCH=120a`; CMake's native `a` grammar
emits one image per token, and a plain `120` list warns at configure time)
which the driver picks on sm_120; every other pass/device falls back to the
plain cell inside the same tree, so routing can only trade speed, never
correctness. `launch_plan` routes the symmetric-fp8 pair through the mx
tree on sm_120 unless `set_staging(mx=False)` knocks it out (the A/B knob;
read once per process — separate processes to compare).

**Crosswise loads.** Crosswise operands (A `[K][M]` / B `[N][K]` storage)
cannot cp.async into the canonical tile; they take the direct LDG.128×4 +
in-register PRMT transpose + STS.32 path. 8-bit crosswise operands with an
admissible tile take the k-pair packed grid instead: the operand staged as
16-bit (row, k-pair) units so the 16-bit reader's ldmatrix.trans contract
applies unchanged.

**Interior copy.** When both operands are congruous, the whole CTA is
interior, base|ld is 16B-aligned and K has no tail, the mainloop switches
to a predication-free copy with loop-carried prefetch state.

**NN swap.** The dual-N-contiguous problem runs as its transpose
`E = B^T @ A^T` over swapped operands with an out-transposed epilogue
scatter (CUTLASS-sm90 `is_swapAB`): one instantiation fewer per tile
config, at the cost of a scalar-store scatter on a path no LLM-linear
operand pair hits.

**Tile vocabulary (CUTLASS-style).** Tile geometry is expressed as types,
not positional ints: `Shape<M, N, K>` (CTA tile; K = the per-stage k-tile)
and `Shape<M, N>` (warp tile) compose into a `GemmTileConfig` — one named
recipe bundling shapes + stage depth + loop mode. `Shape` itself is the
shared vocabulary type of `utils/shape.cuh`: the same `Shape<...>` spells
both the CTA tile here and the staging layouts' chunk grids, so tile
geometry and smem layout read in one notation. The production manifest in
`policy/manifest.cuh` — the named `Tile_*` recipes; read the list there, not here,
because copied enumerations rot — is the `TileManifest` type list the
launch ladders dispatch over (CUTLASS builder-table style: `dispatch_tile`
in `gemm.cuh` indexes the manifest by the plan's `TileClass` and depth bit,
both ladder-agnostic); a new geometry is one alias plus one manifest entry
and one planner branch, never a re-spelled per-site ladder. Device
collectives only read the derived `Traits::kBlockM/kBlockN/...` constants,
so this is purely a configuration surface — the generated SASS is
unchanged.

## Planning

**The row table** (`launcher/plan_table.h`) is the production planner's
first resort — `plan_gemm` consults a measured row table: (M, N, K) bands
(min exclusive, max inclusive, 0 = open), keyed per dtype class / crosswise
count, each row naming a recipe (CTA class + ring depth; raster 0 =
`plan_raster` with the row's geometry). The compiled-in rows are one table
per dtype class (`kBuiltinPlanW16A16` / `W8A16` / `W8A8` / `F8A8`, selected
by `builtin_plan_table`): the class *is* the table — a row tuned for one
operand pair cannot fire on another (`gemm_perf_class` tests the int8 pair
before the "not bf16 → fp8 pair" arm). `plan_from_row` is the single
interpreter: geometry from the CTA class, ring depth from the row, raster 0
= `plan_raster`, and one smem gate — a row whose ring exceeds the smem
opt-in ceiling falls through to the degraded bands instead of a failed
launch. A miss falls to the degraded band rows (last-resort M-band
geometry: small ≤ 512, narrow ≤ 3072, big beyond), always matching, so
planning is a total function. Full-coverage tables end each class with a
catch-all row, so a miss means the table is empty or stale, not a shape
the planner should infer.

**Row format** — one row per line:
`m_min m_max n_min n_max perf_class crosswise cta stages raster [k [k_min k_max [min_ctas_per_sm [min_wave_permille]]]]`.
The optional `k` is the row's ring K (omitted keeps 64; a kK=32 row only
survives a dual-2-byte pair); the optional pair is the row's K band, same
(min, max] rule, omitted keeps it open. The trailing gates are wave
arithmetic priced against `DeviceFacts` at lookup time: `min_ctas_per_sm`
matches only while the row's own grid covers that many CTAs per SM, and
`min_wave_permille` is the same idea in per-mille of a full machine —
literal M bounds calibrated to one SM count and wave gates both exist
because measured latency crossovers sometimes keep literal bounds and
sometimes scale with occupancy. Compiled-in rows are pasted manually
between the GENERATED markers of `launcher/plan_table.h` (the measurement
script only emits the row file; a rebuild picks it up).

**The planner chain**: override rows, injected rows, the compiled-in table,
the analytical model, the degraded bands — first to answer wins, and the
mode only picks which chain runs: `"table"` (rows then degraded),
`"hybrid"` (+model between), `"model"` alone. `table_off=True` disables
every row tier at once; every chain ends in the degraded rows (open bands,
m=0 fallback), so dispatch is total. The shipped default is **hybrid with
the compiled-in tables empty**, so a fresh process answers with the
analytical model and takes measured recipes from the override tier or the
autotuner cache; a measured row is only shipped when a device-specific
build pastes one in. `configure(planner="")` restores the default instead
of pinning a mode. The `ASTR_GEMM_*` environment variables are read once
per process as a seed; explicit API calls win.

**The analytical model** (`ModelPlanner` in `launcher/planning.h`) is a
port of DeepGEMM's config search (`get_best_configs`) reduced to what the
recipe space needs. Wave count alone is not a valid ranking proxy here: a
64×64 CTA's wave carries a quarter of a 128×128's work, so counting waves
prefers the coarse tile regardless of fit. The model prices cost instead —

- TMA (dual-congruous): `cost = max(operand + output + issue, mma_arm) × W_eff`
  per CTA, with `operand = k·(bm·ba + bn·bb)` (bytes per operand, `ba`/`bb`
  the per-element byte widths), `output = out_elem_bytes·bm·bn`, `issue`
  the per-k-tile issue overhead priced only for byte pairs, and
  `mma_arm` the tensor-pipe arm. `W_eff = waves × resident` for dual-2-byte
  pairs, plain `ceil(blocks/sms)` otherwise. The arms overlap on
  independent hardware, so a non-binding arm must not tax the ranking.
- cp.async: `cost = (k-padded operand + output) × waves` with raw-floor
  residency in the waves denominator — the software ring is the only
  latency hiding, so residency divides the makespan instead of sharing
  bandwidth; the axis flips with staging.

Every width pair ranks on the cost alone; a tie keeps the candidate seen
first (the manifest's own order). `kK` is not an independent axis — it
falls out of residency: the kK=32 twin's smaller ring holds more CTAs per
SM. The model carries no per-architecture constants; what it cannot
express is the per-CTA efficiency that separates classes at a given
(M, N) — the measured axis the row tables own, and the reason the hybrid
chain keeps rows first.

The Python extension separates configuration from tuning: `plan.py` owns
the public config/probe/override API and the launch hook, while
`autotune.py` owns candidate measurement, cache rows, and persistence.
The FP8 path uses the same ownership rule: `fp8_slots.py` owns stable
module-to-slot identities, and `autocast.py` owns region policy and
`aten::linear` routing.

**Runtime plan surface** (`astrai.extension.policy.gemm.plan`, re-exported from
`astrai.extension`):

```python
from astrai.extension.policy.gemm import plan

plan.config                            # the whole configuration, as a value
plan.configure(planner="hybrid")       # "table" | "hybrid" | "model" | "" (unset)
plan.configure(rows="rows.txt", tier="override")   # a row file or inline text
plan.configure(rows="", tier="injected")           # clear that tier
plan.configure(table_off=True)         # every row tier off at once
plan.configure(log=True, tma=False)    # the decision log; the A/B staging switches
plan.probe(512, 11008, 4096)           # the decision + who made it
plan.facts                             # the DeviceFacts geometry
plan.tiles()                           # the recipe vocabulary, with class names

with plan.override(planner="model", rows="", tier="override"):
    ...                                # restored on exit — knobs *and* rows
```

`configure` leaves every argument it is not given alone and returns the
resulting value, so a saved `config` is re-installable: feeding its fields
back restores exactly that state (each row tier carries the source spec it
was installed from, and a plain `override(...)` block does this for you —
including when the block raises). The flat `set_table` / `set_planner` /
`set_log` / `set_staging` / `state` / `probe` / `facts` /
`tile_vocabulary` names remain as the same bindings in their raw
dict/list shapes — the `csrc/bench` tools parse those keys, so those
spellings are contract.

`plan.probe` returns the decision `gemm_dispatch` would make, with the
planner that made it: `"override"`, `"injected"`, `"builtin"`, `"model"`,
or `"degraded"`.

**Autotuning.** `kernel.gemm.enable()` installs the runtime autotuner: shapes
no row serves tune once (candidates from `tile_vocabulary` filtered to the
staging pair and smem ceiling, forced as one-row tables, interleaved
CUDA-event medians over the caller's own tensors; the winner persists under
`~/.astrai/cache/gemm_plans/<device-sig>.rows` so a new process or a
different part re-derives nothing measured). The hook costs one flag check
when disabled and idles while an override table owns the source. The
offline whole-table recalibration is `csrc/bench/tune_plan_table.py run`:
it sweeps (`sweep --full-coverage` — every combo × recipe over the M ×
shape grid, each candidate as a one-row table toggled per launch,
interleaved at each shape so the comparison shares one clock/thermal
state), gates on the holdout validator (a ≥2% per-shape regression
rejects), and installs under the same device signature. Sweep winners
become rows: adjacent M runs with the same winner band-merge at mid-point
edges; `--min-gain` (default 1%) keeps only rows above a minimum gain.
K is not a row key (the ring K is fixed at 64): a K/batch conflict at one
(M, N) resolves to the best-TFLOPS point. The sweep times the fused-linear
(NT) layout, so generated rows carry crosswise 0 — non-NT shapes (TT, TN,
the mixed dual-row-major NN case) miss into the degraded bands, where the
NN swap path covers the dual-row-major case.

## Not implemented

Grouped-along-K / 2-D block scales (GPTQ/AWQ import — needs mainloop scale
application; the epilogue cannot fold them, and the `mxf8f6f4.block_scale`
cell runs identity scales today), asymmetric quantization with zero-points
(offline folding at repack time), sub-int8 dtypes (int4 and 3/5/6/7-bit
need packed staging + a second dequant family), the offline weight
interleave (one `LDS.32` feeding both registers of a dequant pair),
stream-K (wave quantization), warp specialization / cluster / PDL (the TMA
producer is an elected thread of the math CTA, not a dedicated warp),
NVRTC JIT, and MoE gather/grouped GEMM. Supported beyond the base design:
strided-batch operands with broadcast, fp32 output, and the transposed
output orientation (out_layout Dual) produced from one read.
