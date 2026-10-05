#pragma once
/* Public typed GEMM dispatch and planner probe definitions. */
#include <type_traits>
#include <utility>

#include <launcher/gemm_tiles.cuh>

namespace astrai {
namespace gemm {

/*
 * Pure problem rewrite: dual-N-contiguous (NN) has no instantiation — it
 * runs as its transpose E = B^T A^T over swapped operands (CUTLASS-sm90's
 * is_swapAB) with a transposed-epilogue scatter into the [M][N] buffer.
 * The rare NN path pays a scalar-store scatter for it.
 */
inline void canonicalize_gemm(GemmParams& p, bool& trans_a, bool& trans_b) {
    if (!trans_a && !trans_b) {
        GemmParams s = p; // E = B^T * A^T: swap roles, M <-> N
        s.m = p.n;
        s.n = p.m;
        s.a_ptr = p.b_ptr;
        s.b_ptr = p.a_ptr;
        s.a_ld = p.b_ld;
        s.b_ld = p.a_ld;
        s.a_batch_stride = p.b_batch_stride;
        s.b_batch_stride = p.a_batch_stride;
        /*
         * The caller's [M][N] buffer read as E = B^T A^T: the epilogue
         * walks the caller's rows (kernel n) with the caller's N stride.
         */
        s.out_ld = p.n;
        p = s;
        trans_a = trans_b = true;
    }
}

/*
 * The (trans_a, trans_b) -> layout-tag ladder, ONE home for the branch both
 * the launch and the probe walk (a probe cannot answer for a layout the
 * launch does not take). fn gets three tags; the probe ignores the output
 * one. `swapped` marks the symmetric-NN rewrite's TT — the only arm whose
 * OUTPUT tag is transposed. A visitor, not a tag-returning function: a tag's
 * identity is its TYPE; every arm calls fn, keeping this total.
 */
template <typename ElemA, typename ElemB, typename F>
auto with_layout_tags(bool trans_a, bool trans_b, bool swapped, F&& fn) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    if (trans_a && trans_b) {
        if constexpr (kSymmetric) {
            if (swapped)
                return fn(ColMajor{}, ColMajor{}, ColMajor{});
            return fn(ColMajor{}, ColMajor{}, RowMajor{});
        }
        return fn(ColMajor{}, ColMajor{}, RowMajor{});
    }
    if (trans_b) {
        // NT (the fused-linear shape), the production nn.Linear route.
        return fn(RowMajor{}, ColMajor{}, RowMajor{});
    }
    if (trans_a)
        return fn(ColMajor{}, RowMajor{}, RowMajor{});
    if constexpr (kSymmetric) {
        /*
         * Unreachable: canonicalize_gemm turns a symmetric NN into the TT arm
         * above, so both callers arrive here for a mixed pair only. The arm
         * stays total anyway — the probe returns fn's value and needs no dead
         * fallback — and names the instantiation the rewrite's own TT branch
         * already takes.
         */
        return fn(ColMajor{}, ColMajor{}, RowMajor{});
    } else {
        /*
         * Dual row-major: mixed only — symmetric NN was rewritten above
         * into the transposed TT kernel (if constexpr keeps this
         * instantiation out of symmetric builds).
         */
        return fn(RowMajor{}, RowMajor{}, RowMajor{});
    }
}

/*
 * Dtype-generic entry: canonicalize, plan, wire the tags. ElemA/ElemB/OutT
 * are independent knobs. The one asymmetry is NN: the swap rewrite assumes
 * a single element type, so symmetric NN rewrites to TT while mixed NN
 * instantiates the dual-row-major shape directly (A congruous, B
 * crosswise).
 */
template <typename ElemA, typename ElemB = ElemA, typename OutT = __nv_bfloat16,
          typename Schedule = MmaSync>
void gemm_dispatch(GemmParams p, cudaStream_t stream, bool trans_a, bool trans_b) {
    constexpr bool kSymmetric = std::is_same_v<ElemA, ElemB>;
    bool swapped = false;
    if constexpr (kSymmetric) {
        swapped = !trans_a && !trans_b; // canonicalize rewrites NN
        canonicalize_gemm(p, trans_a, trans_b);
    }
    /*
     * Each branch plans from its OWN layout tags, so the plan and the launch
     * below cannot disagree about the crosswise count, widths or perf class.
     * The tags ride empty tag instances; decltype recovers the types.
     */
    const auto launch = [&](auto la, auto lb, auto lout) {
        launch_plan<ElemA, ElemB, decltype(la), decltype(lb), decltype(lout), OutT, Schedule>(
            p, plan_dispatch_for<ElemA, ElemB, decltype(la), decltype(lb), OutT, Schedule>(p), stream);
    };
    with_layout_tags<ElemA, ElemB>(trans_a, trans_b, swapped, launch);
}

/*
 * Explicit-instantiation spelling shared by the per-pair TUs (bare) and
 * gemm.cu's extern block: one place names the signature.
 */
#ifndef ASTRAI_GEMM_SCHEDULE
#define ASTRAI_GEMM_SCHEDULE MmaSync
#endif

#define ASTRAI_GEMM_INSTANTIATE(W, A)                                                           \
    template void gemm_dispatch<W, A, __nv_bfloat16, ASTRAI_GEMM_SCHEDULE>(GemmParams, cudaStream_t, bool, bool)

/*
 * Host-only planner probe (the autotuner's coverage check): the decision
 * gemm_dispatch would make, without a launch — the planner is GPU-free.
 * The shared tag ladder (NN rewrite included) keeps a probe from
 * disagreeing with the real call's branch. Returns the decision plus the
 * query it answered.
 */
template <typename ElemA, typename ElemB, typename Schedule = MmaSync>
std::pair<PlanDecision, PlanQuery> plan_probe_for(int64_t m,
                                                  int64_t n,
                                                  int64_t k,
                                                  int64_t batch,
                                                  bool trans_a,
                                                  bool trans_b,
                                                  const DeviceFacts& dev) {
    GemmParams p{}; // the planner reads m/n/k/batch only
    p.m = static_cast<int>(m);
    p.n = static_cast<int>(n);
    p.k = static_cast<int>(k);
    p.batch = static_cast<int>(batch);
    bool swapped = false;
    if constexpr (std::is_same_v<ElemA, ElemB>) {
        swapped = !trans_a && !trans_b;
        canonicalize_gemm(p, trans_a, trans_b); // symmetric NN -> transposed TT
    }
    return with_layout_tags<ElemA, ElemB>(trans_a, trans_b, swapped, [&](auto la, auto lb, auto) {
        PlanQuery q = plan_query<ElemA, ElemB, decltype(la), decltype(lb), __nv_bfloat16, Schedule>(p, dev);
        return std::make_pair(plan_dispatch(q), std::move(q));
    });
}

} // namespace gemm
} // namespace astrai
