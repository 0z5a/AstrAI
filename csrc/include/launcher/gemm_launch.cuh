#pragma once
/* Typed CUDA launch, TMA descriptor setup, and planner query. */
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <tuple>
#include <type_traits>
#include <utility>

#include <launcher/plan_types.h>
#include <kernel/gemm/kernel.cuh>
#include <utils/device.cuh>
#include <utils/launch.cuh>

namespace astrai {
namespace gemm {

/*
 * Launchers — pure CUDA, usable from the binding and the C tests. The
 * runtime knobs live in GemmConfig (launcher/plan_types.h), owned at runtime by
 * astrai.extension.policy.gemm.plan.
 */

// Grid for one Policy's tile: N x M blocks, batch on z.
template <typename Traits> dim3 gemm_grid(const GemmParams& p) {
    return dim3((p.n + Traits::kBlockN - 1) / Traits::kBlockN,
                (p.m + Traits::kBlockM - 1) / Traits::kBlockM, p.batch);
}

// One plan-log line per launch (" mx" marks the block_scale cell).
inline void log_gemm_plan(const GemmParams& p,
                          const dim3& grid,
                          int bm,
                          int bn,
                          int stages,
                          int smem,
                          bool tma,
                          bool mx = false) {
    if (!gemm_plan_log_enabled())
        return;
    std::fprintf(stderr,
                 "[gemm-plan] %lldx%lldx%lld b=%d -> tile %dx%d s%d%s%s "
                 "grid %dx%dx%d raster %d smem %d\n",
                 (long long)p.m, (long long)p.n, (long long)p.k, p.batch, bm, bn, stages,
                 tma ? " tma" : "", mx ? " mx" : "", grid.x, grid.y, grid.z, p.raster, smem);
}

/*
 * Launch with the smem budget: >48KB opt-ins once per instantiation via
 * cudaFuncSetAttribute. Templated on the kernel VALUE (auto NTTP) so
 * same-signature kernels never share the armed flag; a failed opt-in arms
 * nothing and the launch fails loudly.
 */
template <auto Kernel, typename... Args>
void launch_with_smem(int smem_bytes, dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
    if (smem_bytes > 48 * 1024) {
        static bool armed = false; // per instantiation
        if (!armed) {
            const cudaError_t err = cudaFuncSetAttribute(
                Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            armed = (err == cudaSuccess);
        }
    }
    Kernel<<<grid, block, smem_bytes, stream>>>(args...);
    ASTRAI_LAUNCH_CHECK();
}

/*
 * Dtype-class derivation (returns plan_types.h's GemmPerfClass): the mma
 * promotion rule plus operand widths (mixed bf16xfp8 lands with W8A16 —
 * same bytes, same promoted bf16 k16 mma). int8 is tested FIRST: it
 * promotes to an int8 mma, so the "not bf16 -> fp8" arm would swallow it
 * and every W8A8 row would be unreachable.
 */
template <typename ElemA, typename ElemB> constexpr GemmPerfClass gemm_perf_class() {
    using MmaT = typename gemm_mma_traits<ElemA, ElemB>::MmaT;
    if constexpr (std::is_same_v<ElemA, int8_t> && std::is_same_v<ElemB, int8_t>) {
        return GemmPerfClass::kW8A8;
    } else if constexpr (!std::is_same_v<MmaT, __nv_bfloat16>) {
        return GemmPerfClass::kF8A8; // native fp8 symmetric pair
    } else if constexpr (std::is_same_v<ElemA, __nv_bfloat16> &&
                         std::is_same_v<ElemB, __nv_bfloat16>) {
        return GemmPerfClass::kW16A16;
    } else {
        return GemmPerfClass::kW8A16;
    }
}

static_assert(gemm_perf_class<int8_t, int8_t>() == GemmPerfClass::kW8A8,
              "int8 x int8 is its own class (see the ordering note above)");
static_assert(gemm_perf_class<__nv_fp8_e4m3, __nv_fp8_e4m3>() == GemmPerfClass::kF8A8,
              "fp8 x fp8 keys the F8A8 table");
static_assert(gemm_perf_class<__nv_bfloat16, __nv_bfloat16>() == GemmPerfClass::kW16A16,
              "bf16 x bf16 keys the W16A16 table");
static_assert(gemm_perf_class<__nv_bfloat16, int8_t>() == GemmPerfClass::kW8A16,
              "a quantized weight against bf16 activations keys W8A16");

/*
 * The one place a GemmParams becomes planner input and the dispatch key is
 * derived: perf class, widths and crosswise are computed from the typed call,
 * so a caller cannot hand the planner a key that contradicts its own types.
 * OutT (default bf16) prices the model cost's output term at real size.
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync>
PlanQuery plan_query(const GemmParams& p, const DeviceFacts& dev) {
    PlanQuery q;
    q.m = p.m;
    q.n = p.n;
    q.k = p.k;
    q.batch = p.batch;
    q.perf_class = (int)gemm_perf_class<ElemA, ElemB>();
    q.crosswise = crosswise_of<LayoutA, LayoutB>();
    q.ba = (int)sizeof(ElemA);
    q.bb = (int)sizeof(ElemB);
    q.out_elem_bytes = (int)sizeof(OutT);
    /*
     * The staging the launch that follows will take — launch_plan's
     * predicate minus the descriptor-encodable runtime check: TMA only for
     * the dual-congruous layout pair with descriptor-encodable dtypes on
     * sm_90+ without the kill switch, cp.async otherwise.
     */
    q.tma = Schedule::kTma && crosswise_of<LayoutA, LayoutB>() == 0 &&
            sizeof(ElemA) <= 2 && sizeof(ElemB) <= 2 && dev.cc >= 90 &&
            !gemm_tma_staging_disabled();
    q.dev = dev;
    return q;
}

/*
 * Typed dispatch entry: the layout tags as types tie the decision to the
 * launch that follows it.
 */
template <typename ElemA,
          typename ElemB,
          typename LayoutA,
          typename LayoutB,
          typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync>
PlanDecision plan_dispatch_for(const GemmParams& p) {
    return plan_dispatch(plan_query<ElemA, ElemB, LayoutA, LayoutB, OutT, Schedule>(p, device_facts()));
}
template <typename Policy> void launch_policy(GemmParams p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages, Policy::kSmemBytes,
                  /*tma=*/false, Traits::kMxCell);
    launch_with_smem<gemm_kernel<Policy>>(Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads),
                                          stream, p);
}

/*
 * TMA staging (sm_90+): descriptor build + the TMA twin of launch_policy;
 * descriptors are cached exact-match (tma.cuh), the encode is paid once.
 */

/*
 * Output-reclaim feasibility: a fat output (fp32) can outgrow the ring the
 * planner priced — the wide CTA's 128x256 of fp32 is 131072B against the
 * 73728B a 1-byte pair leaves. Exactly the launch twins' reclaim
 * static_assert, so a new CTA class cannot drift from it.
 */
template <typename Tile, typename ElemA, typename ElemB, typename OutT>
constexpr bool reclaim_fits() {
    return Tile::CtaShape::kM * Tile::CtaShape::kN * sizeof(OutT) <=
           ring_smem_bytes(Tile::CtaShape::kM, Tile::CtaShape::kN, Tile::CtaShape::kK,
                           Tile::kStages, (int)sizeof(ElemA), (int)sizeof(ElemB));
}

/*
 * Both operand descriptors for one TMA Policy. Dim/stride in bytes along
 * the contract dim; batch is a third dim only when it strides. Swizzle mode,
 * box extents and byte scaling derive from the staging layout (tma_spec in
 * memory/tma.cuh) — the same instances the fragment readers consume.
 */
template <typename Policy>
bool tma_maps_for(const GemmParams& p, CUtensorMap* ma, CUtensorMap* mb) {
    using Mainloop = GemmCollectiveMainloop<Policy>;
    const auto a = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemA, typename Mainloop::SmemLayoutA,
                         Mainloop::kBlockM>(p.a_ptr, p.m, p.k, p.a_ld, p.batch, p.a_batch_stride));
    if (!a)
        return false;
    *ma = *a;
    const auto b = astrai::tma_map_cache().lookup(
        astrai::tma_spec<typename Mainloop::ElemB, typename Mainloop::SmemLayoutB,
                         Mainloop::kBlockN>(p.b_ptr, p.n, p.k, p.b_ld, p.batch, p.b_batch_stride));
    if (!b)
        return false;
    *mb = *b;
    return true;
}

/*
 * TMA launch for one Policy; false (nothing launched) on an undescribable
 * operand (misaligned base/ld) so the caller falls back to cp.async.
 */
template <typename Policy> bool launch_policy_tma(const GemmParams& p, cudaStream_t stream) {
    using Traits = typename Policy::Traits;
    /*
     * The planner prices rings only; TMA's pad + barriers can tip past
     * the opt-in ceiling on the fattest pair — fall back, not fail.
     */
    if (Policy::kSmemBytes > astrai::device_facts().smem_max)
        return false;
    CUtensorMap ma{}, mb{};
    if (!tma_maps_for<Policy>(p, &ma, &mb))
        return false;
    dim3 grid = gemm_grid<Traits>(p);
    log_gemm_plan(p, grid, Traits::kBlockM, Traits::kBlockN, Traits::kStages, Policy::kSmemBytes,
                  /*tma=*/true, Traits::kMxCell);
    /*
     * Rank bits pick the instantiation: strided batch rides the 3D
     * emitters, broadcast keeps the shared 2D map; the per-stage pick
     * compiles away.
     */
    auto launch_rank = [&](auto rank3a, auto rank3b) {
        launch_with_smem<gemm_kernel_tma<Policy, decltype(rank3a)::value, decltype(rank3b)::value>>(
            Policy::kSmemBytes, grid, dim3(Traits::kCtaThreads), stream, p, ma, mb);
    };
    if (p.batch > 1 && p.a_batch_stride > 0 && p.b_batch_stride > 0)
        launch_rank(std::true_type{}, std::true_type{});
    else if (p.batch > 1 && p.a_batch_stride > 0)
        launch_rank(std::true_type{}, std::false_type{});
    else if (p.batch > 1 && p.b_batch_stride > 0)
        launch_rank(std::false_type{}, std::true_type{});
    else
        launch_rank(std::false_type{}, std::false_type{});
    return true;
}


} // namespace gemm
} // namespace astrai
