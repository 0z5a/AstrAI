#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_mma_utils.cuh"
#include "attn_warp_utils.cuh"

// SGLang-style split-KV tensor-core decode.
//
// Reads K/V directly from a flat pool [size, kv_head, head_dim] via
// req_to_token indexing — no gather, no page-table dimension.
// Each batch element has its own seq_len (from kv_indptr), eliminating
// padding waste: short sequences only process the tiles they own.
//
// For decode (q_len=1), causal masking is implicit — each request attends
// to [0, seq_len) which is exactly its valid range.  The IsCausal flag
// is accepted for dispatch uniformity but does not change maxc.
template <typename Traits, bool IsCausal, bool HasMask>
__global__ void paged_attn_decode_split_kv_mma_kernel(PagedAttentionParams<bf16> p) {
    const int lane = threadIdx.x;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;

    const int pass = blockIdx.x / p.kv_head;
    const int kv_head = blockIdx.x % p.kv_head;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;

    // Per-request seq_len from device-side kv_indptr — no padding.
    const int seq_len = p.kv_indptr[batch + 1] - p.kv_indptr[batch];
    const int64_t req_idx = p.req_pool_indices[batch];

    constexpr int MAX_G = 16;
    const int G_total = p.q_head / p.kv_head;
    const int g_begin = pass * MAX_G;
    const int G = min(MAX_G, G_total - g_begin);
    const int q_head0 = kv_head * G_total + g_begin;

    __shared__ __align__(16) bf16 sK[Traits::STAGES * Traits::BC * Traits::LD];
    __shared__ __align__(16) bf16 sV[Traits::STAGES * Traits::BC * Traits::LD];

    const int q_base = batch * p.q_stride_l + q_head0 * p.q_stride_h;
    const int qra = gid;
    const int qrb = gid + 8;
    const bool va = qra < G, vb = qrb < G;
    unsigned Qa[Traits::KD][4];
    load_q_mma_frags<Traits::KD>(p.q + q_base,
                                   p.q_stride_h, p.q_stride_d,
                                   qra, qrb, va, vb, tid4, Qa);

    float Oacc[Traits::DN8][4];
    #pragma unroll
    for (int j = 0; j < Traits::DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int tiles_total = (seq_len + Traits::BC - 1) / Traits::BC;
    const int tiles_per_split = (tiles_total + p.num_splits - 1) / p.num_splits;
    const int ti_begin = split * tiles_per_split;
    const int ti_end = min(tiles_total, ti_begin + tiles_per_split);

    // Flat pool stride: [size, kv_head, head_dim] — contiguous.
    const int64_t pool_stride = (int64_t)p.kv_head * Traits::HEAD_DIM;
    const int64_t head_off    = (int64_t)kv_head * Traits::HEAD_DIM;
    const int64_t rtt_stride  = (int64_t)p.max_context_len;

    // ---- Load tile lambda: SGLang addressing ----
    // slot = req_to_token[req_idx * max_context_len + kc]
    // gmem = k_cache[slot * pool_stride + head_off + d]
    auto load_tile = [&](int ti, int buf) {
        int kv0 = ti * Traits::BC;
        bf16* dK = sK + buf * Traits::BC * Traits::LD;
        bf16* dV = sV + buf * Traits::BC * Traits::LD;
        #pragma unroll
        for (int i = lane * Traits::VEC; i < Traits::TOTAL;
             i += Traits::NUM_THREADS * Traits::VEC) {
            int r = i / Traits::HEAD_DIM, d = i % Traits::HEAD_DIM;
            int kc = kv0 + r;
            bool valid = (kc < seq_len);
            if constexpr (HasMask) {
                valid = valid && p.mask[batch * p.mask_b_stride + kc];
            }
            int64_t slot = valid ? p.req_to_token[req_idx * rtt_stride + kc] : 0;
            valid = valid && (slot >= 0);
            int64_t gmem_base = slot * pool_stride + head_off;
            int off = r * Traits::LD + swiz_col(d, r, Traits::SWIZ_MASK);
            cp_async_16_pred(&dK[off], &p.k_cache[gmem_base + d], valid);
            cp_async_16_pred(&dV[off], &p.v_cache[gmem_base + d], valid);
        }
        cp_async_commit();
    };

    constexpr int STAGES = Traits::STAGES;
    const int ntiles = ti_end - ti_begin;

    auto process_tile = [&](int it, int buf) {
        const bf16* bK = sK + buf * Traits::BC * Traits::LD;
        const bf16* bV = sV + buf * Traits::BC * Traits::LD;
        int kv0 = (ti_begin + it) * Traits::BC;

        float Sacc[Traits::NC8][4];
        mma_compute_scores<Traits>(Qa, bK, lane, Sacc);

        #pragma unroll
        for (int n8 = 0; n8 < Traits::NC8; n8++)
            Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
            Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

        // For decode, maxc = seq_len regardless of IsCausal — the valid
        // range [0, seq_len) IS the causal range (query is the last token).
        mma_softmax_tile<Traits, HasMask>(kv0, seq_len, seq_len,
                                           0, 0,
                                           p.mask_b_stride, 0, 0,
                                           batch, 0,
                                           p.mask,
                                           Sacc, Oacc, m0, m1, l0, l1, lane);

        mma_pv_accumulate<Traits>(Sacc, bV, lane, Oacc);
    };

    if (ntiles >= STAGES) {
        #pragma unroll
        for (int i = 0; i < STAGES; i++)
            load_tile(ti_begin + i, i);

        for (int it = 0; it < ntiles; it++) {
            cp_async_wait_group<STAGES - 1>();
            __syncwarp();
            process_tile(it, it & (STAGES - 1));
            __syncwarp();
            if (it + STAGES < ntiles)
                load_tile(ti_begin + it + STAGES, (it + STAGES) & (STAGES - 1));
        }
    } else {
        for (int i = 0; i < ntiles; i++)
            load_tile(ti_begin + i, i);
        cp_async_wait_group<0>();
        __syncwarp();
        for (int it = 0; it < ntiles; it++)
            process_tile(it, it);
    }

    // ---- write partials ----
    auto split_slot = [&](int h) -> size_t {
        size_t bh = (size_t)batch * p.q_head + h;
        return bh * MAX_SPLITS + split;
    };
    #pragma unroll
    for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int h = q_head0 + r0;
            float* op = p.o_part + split_slot(h) * Traits::HEAD_DIM;
            op[d] = Oacc[dn8][0];
            op[d + 1] = Oacc[dn8][1];
        }
        if (r1 < G) {
            int h = q_head0 + r1;
            float* op = p.o_part + split_slot(h) * Traits::HEAD_DIM;
            op[d] = Oacc[dn8][2];
            op[d + 1] = Oacc[dn8][3];
        }
    }
    if (tid4 == 0) {
        int r0 = gid, r1 = gid + 8;
        if (r0 < G) {
            int h = q_head0 + r0;
            float* mp = p.ml_part + split_slot(h) * 2;
            mp[0] = m0; mp[1] = l0;
        }
        if (r1 < G) {
            int h = q_head0 + r1;
            float* mp = p.ml_part + split_slot(h) * 2;
            mp[0] = m1; mp[1] = l1;
        }
    }
}
