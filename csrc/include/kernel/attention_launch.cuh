// Shared attention launch vocabulary — split-KV heuristic, head_dim guard,
// causal×mask dispatch ladder, kernel-family launchers with their tile-config
// maps, and the four family dispatchers. Pure CUDA, no torch: the standalone
// harnesses compile the exact code the production dispatch runs, and each
// production .cu is then exactly one torch-facing function. Kernel bodies
// live in kernel/attention_split_{q,kv}.cuh.

#pragma once

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <cuda_runtime.h>

#include <api/attention_common.h>
#include <kernel/attention_split_kv.cuh>
#include <kernel/attention_split_q.cuh>
#include <memory/layout_policies.cuh>
#include <utils/launch.cuh>

namespace astrai {
namespace attention {

// The one list of instantiated head dims — the dispatch switch, the fatal
// message and (drift-asserted) the Python backend's HEAD_DIMS all read it.
#define ASTRAI_ATTN_HEAD_DIMS(X) X(32) X(64) X(128) X(256)

// A head dim no kernel was instantiated for — the launch discipline: never
// run a kernel that was not built for the shape.
[[noreturn]] inline void head_dim_fatal(int head_dim) {
    std::fprintf(
        stderr,
        "ASTRAI: attention: head_dim %d has no kernel instantiation (instantiated:", head_dim);
#define ASTRAI_HEAD_DIM_ROW(D) std::fprintf(stderr, " %d", D);
    ASTRAI_ATTN_HEAD_DIMS(ASTRAI_HEAD_DIM_ROW)
#undef ASTRAI_HEAD_DIM_ROW
    std::fprintf(stderr, ")\n");
    std::exit(EXIT_FAILURE);
}

// Split-KV count: fill exactly one wave of blocks.
//
// The GPU runs blocks in waves of (SM count x resident blocks per SM). A
// grid smaller than a wave leaves SMs idle; a grid that crosses into a
// second wave pays a full extra wave of latency for the few straggler
// blocks. So the split count is chosen to bring the grid as close to one
// full wave as possible without crossing it:
//   grid = base_blocks * splits <= wave_capacity
//   =>   splits = floor(wave_capacity / base_blocks)
// The work caps still apply: never more splits than the tile count allows
// (each split needs at least min_tiles_per_split tiles to not be pure
// combine overhead) and never more than MAX_SPLITS.
inline int compute_num_splits(int base_blocks,
                              int tiles_total,
                              int wave_capacity,
                              int min_tiles_per_split = 1) {
    int cap = std::min(tiles_total / std::max(min_tiles_per_split, 1), MAX_SPLITS);
    if (cap <= 1)
        return 1;
    return std::max(1, std::min(wave_capacity / std::max(base_blocks, 1), cap));
}

// Wave capacity = SM count x blocks resident per SM, for one decode kernel
// instantiation. Residency depends on the kernel's shared memory and
// register footprint, so it is queried from the occupancy API rather than
// assumed. The query and the device-property reads are not free and decode
// launches per token, so the answer is memoized per instantiation (the
// lambda runs once).
template <typename Kernel>
inline int decode_wave_capacity(Kernel kernel, int threads) {
    static int capacity = [kernel, threads] {
        int per_sm = 0, device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, 0) !=
                cudaSuccess ||
            per_sm < 1)
            per_sm = 1;
        int c = prop.multiProcessorCount * per_sm;
        return c > 0 ? c : 1;
    }();
    return capacity;
}

// Dispatch IsCausal × HasMask. FN is a function template
// <int HEAD_DIM, bool IsCausal, bool HasMask>; HEAD_DIM forwards first so
// callers spell it once:
//   DISPATCH_CAUSAL_MASK(is_causal, has_mask,
//                        launcher<KV>::template launch, HEAD_DIM, p, stream);
#define DISPATCH_CAUSAL_MASK(is_causal, has_mask, FN, HEAD_DIM, ...)                               \
    do {                                                                                           \
        if (is_causal) {                                                                           \
            if (has_mask)                                                                          \
                FN<HEAD_DIM, true, true>(__VA_ARGS__);                                             \
            else                                                                                   \
                FN<HEAD_DIM, true, false>(__VA_ARGS__);                                            \
        } else {                                                                                   \
            if (has_mask)                                                                          \
                FN<HEAD_DIM, false, true>(__VA_ARGS__);                                            \
            else                                                                                   \
                FN<HEAD_DIM, false, false>(__VA_ARGS__);                                           \
        }                                                                                          \
    } while (0)

// Kernel-family launchers. Every supported target is sm_80+, so the
// tensor-core kernels are the only implementations; both families expose
// the same static launch<HEAD_DIM, IsCausal, HasMask> interface.

template <int BC_> struct PrefillKernelConfig {
    static constexpr int BC = BC_;
    static constexpr int WARPS = 4;
    static constexpr int STAGES = 2;
};

// Prefill tile-config map (BC by head_dim × causal), shared by the
// contiguous and paged entries. Unsupported head dims have no mapping.
template <int HEAD_DIM, bool IsCausal> struct PrefillConfigMap;

template <> struct PrefillConfigMap<32, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<32, true> : PrefillKernelConfig<64> {};
template <> struct PrefillConfigMap<64, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<64, true> : PrefillKernelConfig<64> {};
template <> struct PrefillConfigMap<128, false> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<128, true> : PrefillKernelConfig<32> {};
template <> struct PrefillConfigMap<256, false> : PrefillKernelConfig<16> {};
template <> struct PrefillConfigMap<256, true> : PrefillKernelConfig<16> {};

template <typename QSchedule, typename KV> struct PrefillLauncher {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        using Config = PrefillConfigMap<HEAD_DIM, IsCausal>;
        using Traits =
            KernelTraits<HEAD_DIM, Config::BC, Config::WARPS, Config::STAGES, typename KV::Elem>;
        // PackGQA-folded grid: each block owns BLOCK_M packed (head,row) rows
        // of the request's packed space; grid.y is the kv head (see the
        // prefill kernel header for the fold math).
        constexpr int BLOCK_M = Traits::BR * Config::WARPS;
        dim3 grid(QSchedule::packed_grid_x(p, BLOCK_M, BLOCK_M), p.kv_head,
                  QSchedule::host_grid_batch(p));
        dim3 block(Traits::NUM_THREADS);
        attn_prefill_split_q_mma_kernel<Traits, QSchedule, KV, IsCausal, HasMask>
            <<<grid, block, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

// BC=16: halves smem (16KB vs 32KB) → doubles occupancy; for D=256 it also
// cuts register pressure enough for STAGES=2 within the 32KB budget,
// eliminating the 176-byte spill of STAGES=1+BC=32.
template <typename KV> struct DecodeLauncher {
    template <int HEAD_DIM, bool IsCausal, bool HasMask>
    static void launch(AttentionParams& p, cudaStream_t stream) {
        int G = p.q_head / p.kv_head;
        constexpr int MAX_G = 16;
        int num_passes = (G + MAX_G - 1) / MAX_G;
        constexpr int BC = 16;
        int kv_len = KV::host_kv_len(p);
        int tiles_total = (kv_len + BC - 1) / BC;
        using Traits = KernelTraits<HEAD_DIM, BC, 1, 2, typename KV::Elem>;
        p.num_splits = compute_num_splits(
            p.batch * p.kv_head * num_passes, tiles_total,
            decode_wave_capacity(attn_decode_split_kv_mma_kernel<Traits, KV, IsCausal, HasMask>,
                                 32),
            2);
        dim3 grid(p.kv_head * num_passes, p.batch, p.num_splits);
        attn_decode_split_kv_mma_kernel<Traits, KV, IsCausal, HasMask><<<grid, 32, 0, stream>>>(p);
        ASTRAI_LAUNCH_CHECK();
    }
};

// Family dispatchers — shared between the production .cu entries and the
// standalone torch-free harnesses, which compile them directly (a .o link
// would drag the pybind module and torch in). T is the element type; the
// head dim switches here because it selects tile configs, not storage. The
// four entries differ only in the policy pair, so they funnel into one impl
// per family.

template <typename QSchedule, typename KV, int HEAD_DIM>
static inline void dispatch_prefill_impl(AttentionParams& p, cudaStream_t stream) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

    using Launcher = PrefillLauncher<QSchedule, KV>;
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, Launcher::template launch, HEAD_DIM, p, stream);
}

template <typename QSchedule, typename KV, int HEAD_DIM>
static inline void dispatch_paged_prefill_impl(AttentionParams& p, cudaStream_t stream) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

    using Launcher = PrefillLauncher<QSchedule, KV>;
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, Launcher::template launch, HEAD_DIM, p, stream);
}

// Decode funnel: the causal/mask ladder plus the combine pass reducing the
// split partials.
template <typename KV, int HEAD_DIM>
static inline void dispatch_decode_impl(AttentionParams& p, cudaStream_t stream) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

    using Launcher = DecodeLauncher<KV>;
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, Launcher::template launch, HEAD_DIM, p, stream);

    attn_decode_combine_kernel<KV><<<p.batch * p.q_head, p.head_dim, 0, stream>>>(p);
    ASTRAI_LAUNCH_CHECK();
}

// One table-driven head-dim dispatch per family: the caller passes a
// Fn object exposing template <int HEAD_DIM> operator()(AttentionParams&,
// cudaStream_t); the switch stamps one call per list row. The four family
// entries below differ only in the policy pair they bind.
template <typename Fn>
static inline void dispatch_head_dim(AttentionParams& p, cudaStream_t stream) {
    switch (p.head_dim) {
#define ASTRAI_HEAD_DIM_CASE(D)                                                                    \
    case D:                                                                                        \
        Fn::template run<D>(p, stream);                                                            \
        break;
        ASTRAI_ATTN_HEAD_DIMS(ASTRAI_HEAD_DIM_CASE)
#undef ASTRAI_HEAD_DIM_CASE
    default:
        head_dim_fatal(p.head_dim);
    }
}

// Family bindings: one thin Fn per entry, naming its policy pair.
template <typename T> struct DispatchPrefill {
    template <int HEAD_DIM> static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_prefill_impl<DenseQSchedule, ContigKV<T>, HEAD_DIM>(p, stream);
    }
};
template <typename T> struct DispatchPagedPrefill {
    template <int HEAD_DIM> static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_prefill_impl<PackedQSchedule, PagedKV<T>, HEAD_DIM>(p, stream);
    }
};
template <typename T> struct DispatchDecode {
    template <int HEAD_DIM> static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_decode_impl<ContigKV<T>, HEAD_DIM>(p, stream);
    }
};
template <typename T> struct DispatchPagedDecode {
    template <int HEAD_DIM> static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_decode_impl<PagedKV<T>, HEAD_DIM>(p, stream);
    }
};

template <typename T> static inline void dispatch_prefill(AttentionParams& p, cudaStream_t stream) {
    dispatch_head_dim<DispatchPrefill<T>>(p, stream);
}
template <typename T>
static inline void dispatch_paged_prefill(AttentionParams& p, cudaStream_t stream) {
    dispatch_head_dim<DispatchPagedPrefill<T>>(p, stream);
}
template <typename T> static inline void dispatch_decode(AttentionParams& p, cudaStream_t stream) {
    dispatch_head_dim<DispatchDecode<T>>(p, stream);
}
template <typename T>
static inline void dispatch_paged_decode(AttentionParams& p, cudaStream_t stream) {
    dispatch_head_dim<DispatchPagedDecode<T>>(p, stream);
}

// Per-family wrappers: a class template per entry (function templates
// cannot be template-template arguments), each forwarding to the free
// dispatch function above.
template <typename T> struct AttnDispatchDecode {
    static void run(AttentionParams& p, cudaStream_t stream) { dispatch_decode<T>(p, stream); }
};
template <typename T> struct AttnDispatchPrefill {
    static void run(AttentionParams& p, cudaStream_t stream) { dispatch_prefill<T>(p, stream); }
};
template <typename T> struct AttnDispatchPagedDecode {
    static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_paged_decode<T>(p, stream);
    }
};
template <typename T> struct AttnDispatchPagedPrefill {
    static void run(AttentionParams& p, cudaStream_t stream) {
        dispatch_paged_prefill<T>(p, stream);
    }
};

} // namespace attention
} // namespace astrai
