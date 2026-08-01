#pragma once
// Shared attention dispatchers — used by both production .cu and test .cu.
// No torch dependency; pure CUDA.

#include <cuda_runtime.h>
#include <algorithm>
#include "attn_warp_utils.cuh"
#include "attn_prefill_split_q.cuh"
#include "attn_decode_split_kv.cuh"
#include "attn_paged_decode_split_kv.cuh"
#ifndef ASTRAI_NO_MMA
#include "attn_prefill_split_q_mma.cuh"
#include "attn_decode_split_kv_mma.cuh"
#include "attn_paged_decode_split_kv_mma.cuh"
#include "attn_paged_prefill_split_q_mma.cuh"
#endif

// Cached SM count — cudaDeviceGetAttribute is a host-side call that was
// invoked on every decode/paged-decode launch.  Cache per-device so multi-GPU
// setups with heterogeneous GPUs still get the right count, while the common
// single-GPU path hits the cache after the first call.
inline int get_sm_count() {
    int dev = 0;
    cudaGetDevice(&dev);
    static int cached_dev = -1;
    static int cached_count = 0;
    if (dev != cached_dev) {
        cudaDeviceGetAttribute(&cached_count, cudaDevAttrMultiProcessorCount, dev);
        cached_dev = dev;
    }
    return cached_count;
}

// Split-KV: compute number of splits to fill all SMs for small-batch decode.
// Caps splits so each split processes at least `min_tiles_per_split` tiles,
// avoiding excessive loop/prologue overhead when tiles are small.
inline int compute_num_splits(int base_blocks, int tiles_total,
                               int min_tiles_per_split = 1) {
    int sm_count = get_sm_count();
    int n = (2 * sm_count + base_blocks - 1) / base_blocks;
    int max_by_work = tiles_total / min_tiles_per_split;
    return std::max(1, std::min(n, std::min(max_by_work, MAX_SPLITS)));
}

// Dispatch IsCausal × HasMask — eliminates the duplicated 4-way if/else
// ladder that appeared in each dispatch_* function.  FN must be a function
// template <int HEAD_DIM, bool IsCausal, bool HasMask>; HEAD_DIM is forwarded
// as the first template argument so callers only spell it once.
//
// Usage:  DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_decode_mma, HEAD_DIM, p, group_size);
#define DISPATCH_CAUSAL_MASK(is_causal, has_mask, FN, HEAD_DIM, ...) \
    do { \
        if (is_causal) { \
            if (has_mask) FN<HEAD_DIM, true,  true>(__VA_ARGS__); \
            else          FN<HEAD_DIM, true,  false>(__VA_ARGS__); \
        } else { \
            if (has_mask) FN<HEAD_DIM, false, true>(__VA_ARGS__); \
            else          FN<HEAD_DIM, false, false>(__VA_ARGS__); \
        } \
    } while (0)

// ======================================================================
// Prefill
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_prefill_mma(AttentionParams<bf16>& p) {
    constexpr int WARPS = 4;
    constexpr int BC = (HEAD_DIM <= 128) ? 32 : 16;
    using Traits = KernelTraits<HEAD_DIM, BC, WARPS, 2>;
    dim3 grid((p.q_len + Traits::BR * WARPS - 1) / (Traits::BR * WARPS), p.q_head, p.batch);
    dim3 block(Traits::NUM_THREADS);
    attn_prefill_split_q_mma_kernel<Traits, IsCausal, HasMask><<<grid, block>>>(p);
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_prefill_scalar(AttentionParams<bf16>& p) {
    constexpr int G = 8, ROWS = 32, P_BC = 32;
    dim3 grid((p.q_len + ROWS - 1) / ROWS, p.q_head, p.batch);
    dim3 block(G, ROWS);
    attn_prefill_split_q_kernel_t<HEAD_DIM, G, ROWS, P_BC, IsCausal, HasMask><<<grid, block>>>(p);
}

template <int HEAD_DIM>
static inline void dispatch_prefill(AttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

#ifndef ASTRAI_NO_MMA
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_prefill_mma, HEAD_DIM, p);
#else
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_prefill_scalar, HEAD_DIM, p);
#endif
}

// ======================================================================
// Decode
// ======================================================================

#ifndef ASTRAI_NO_MMA
// BC=16: halves smem (16KB vs 32KB) → doubles occupancy (6 vs 3 blocks/SM).
// For D=256, BC=16 also reduces register pressure (fewer Sacc/PV frags),
// enabling STAGES=2 (double-buffer) within the 32KB smem budget — eliminates
// the 176-byte spill that STAGES=1+BC=32 suffered.
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_decode_mma(AttentionParams<bf16>& p, int group_size) {
    int G = p.q_head / p.kv_head;
    constexpr int MAX_G = 16;
    int num_passes = (G + MAX_G - 1) / MAX_G;
    constexpr int BC = 16;
    int tiles_total = (p.kv_len + BC - 1) / BC;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, tiles_total, 2);
    constexpr int STAGES = 2;
    using Traits = KernelTraits<HEAD_DIM, BC, 1, STAGES>;
    dim3 grid(p.kv_head * num_passes, p.batch, p.num_splits);
    attn_decode_split_kv_mma_kernel<Traits, IsCausal, HasMask><<<grid, 32>>>(p);
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_decode_scalar(AttentionParams<bf16>& p, int group_size) {
    int chunks_total = (p.kv_len + DC_CHUNK - 1) / DC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    size_t smem = DC_CHUNK * p.head_dim * sizeof(bf16);
    int g = min(group_size, 32);  // cap at 32 to respect 1024-thread limit
    dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
    dim3 block(32, g);
    attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
}

template <int HEAD_DIM>
static inline void dispatch_decode(AttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);
    int group_size = p.q_head / p.kv_head;

#ifndef ASTRAI_NO_MMA
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_decode_mma, HEAD_DIM, p, group_size);
#else
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_decode_scalar, HEAD_DIM, p, group_size);
#endif

    attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

// ======================================================================
// Paged Decode (SGLang-style: flat pool + req_to_token + kv_indptr)
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_paged_decode_mma(PagedAttentionParams<bf16>& p, int) {
    int G = p.q_head / p.kv_head;
    constexpr int MAX_G = 16;
    constexpr int BC = 16;
    int num_passes = (G + MAX_G - 1) / MAX_G;
    int tiles_total = (p.max_seq_len + BC - 1) / BC;
    p.num_splits = compute_num_splits(p.batch * p.kv_head * num_passes, tiles_total, 2);
    constexpr int STAGES = 2;
    using Traits = KernelTraits<HEAD_DIM, BC, 1, STAGES>;
    dim3 grid(p.kv_head * num_passes, p.batch, p.num_splits);
    paged_attn_decode_split_kv_mma_kernel<Traits, IsCausal, HasMask> <<<grid, 32>>>(p);
}
#endif

template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_paged_decode_scalar(PagedAttentionParams<bf16>& p, int group_size) {
    int chunks_total = (p.max_seq_len + PDC_CHUNK - 1) / PDC_CHUNK;
    p.num_splits = compute_num_splits(p.batch * p.kv_head, chunks_total);
    size_t smem = PDC_CHUNK * p.head_dim * sizeof(bf16);
    int g = min(group_size, 32);
    dim3 grid(p.batch * p.kv_head, 1, p.num_splits);
    dim3 block(32, g);
    paged_attn_decode_split_kv_kernel<HEAD_DIM, IsCausal, HasMask><<<grid, block, smem>>>(p);
}

template <int HEAD_DIM>
static inline void dispatch_paged_decode(PagedAttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);
    int group_size = p.q_head / p.kv_head;

#ifndef ASTRAI_NO_MMA
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_paged_decode_mma, HEAD_DIM, p, 0);
#else
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_paged_decode_scalar, HEAD_DIM, p, group_size);
#endif

    paged_attn_decode_combine_kernel<<<p.batch * p.q_head, p.head_dim>>>(p);
}

// ======================================================================
// Paged Prefill (SGLang-style: flat pool + ragged batch)
// ======================================================================

#ifndef ASTRAI_NO_MMA
template <int HEAD_DIM, bool IsCausal, bool HasMask>
static inline void launch_paged_prefill_mma(PagedAttentionParams<bf16>& p) {
    constexpr int WARPS = 4;
    constexpr int BC = (HEAD_DIM <= 128) ? 32 : 16;
    using Traits = KernelTraits<HEAD_DIM, BC, WARPS, 2>;
    int max_q_tiles = (p.max_q_len + Traits::BR * WARPS - 1) / (Traits::BR * WARPS);
    dim3 grid(max_q_tiles, p.q_head, p.batch);
    dim3 block(Traits::NUM_THREADS);
    paged_attn_prefill_split_q_mma_kernel<Traits, IsCausal, HasMask><<<grid, block>>>(p);
}
#endif

template <int HEAD_DIM>
static inline void dispatch_paged_prefill(PagedAttentionParams<bf16>& p) {
    bool is_causal = (p.causal_offset >= 0);
    bool has_mask = (p.use_mask && p.mask);

#ifndef ASTRAI_NO_MMA
    DISPATCH_CAUSAL_MASK(is_causal, has_mask, launch_paged_prefill_mma, HEAD_DIM, p);
#endif
}
