#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_mma_utils.cuh"

// SGLang-style split-Q tensor-core prefill.
//
// Reads K/V directly from a flat pool [size, kv_head, head_dim] via
// req_to_token — no gather, no temporary tensor.  Supports ragged batches:
// each request has its own q_len and kv_len, addressed via qo_indptr and
// kv_indptr.
//
// Grid: (max_q_tiles, q_head, batch) — one batch element per blockIdx.z.
// Blocks beyond a request's q_len exit early after writing sentinel-free
// no-ops.  This avoids the binary-search approach and guarantees every Q
// token is covered, even when q_len < BR*WARPS (e.g. decode-like prefill).
//
// Q layout: [total_q, q_head, head_dim] (3D, flattened across requests).
// O layout: same as Q.
//
// IsCausal is a compile-time bool.  When true, each Q row qi (within its
// request) attends to [0, causal_offset_b + qi + 1) where
// causal_offset_b = kv_len_b - q_len_b (position of first Q token).
template <typename Traits, bool IsCausal, bool HasMask>
__global__ void paged_attn_prefill_split_q_mma_kernel(PagedAttentionParams<bf16> p) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;

    const int q_head = blockIdx.y;
    const int req_b  = blockIdx.z;
    const int qrow0  = (blockIdx.x * Traits::WARPS + warp) * Traits::BR;

    const int seq_len    = p.kv_indptr[req_b + 1] - p.kv_indptr[req_b];
    const int q_len      = p.qo_indptr[req_b + 1] - p.qo_indptr[req_b];
    const int causal_off = seq_len - q_len;
    const int64_t req_idx = p.req_pool_indices[req_b];

    // No per-warp early exit — all warps must participate in __syncthreads.
    // Warps beyond q_len get zero-filled Q frags (va=vb=false) and skip output.
    const int kv_head = q_head / (p.q_head / p.kv_head);

    __shared__ __align__(16) bf16 sK[Traits::STAGES * Traits::BC * Traits::LD];
    __shared__ __align__(16) bf16 sV[Traits::STAGES * Traits::BC * Traits::LD];

    // Q base: offset by qo_indptr[req_b] to get absolute token address.
    const int q_base = p.qo_indptr[req_b] * p.q_stride_l + q_head * p.q_stride_h;
    const int qra = qrow0 + gid;
    const int qrb = qrow0 + gid + 8;
    const bool va = qra < q_len, vb = qrb < q_len;
    unsigned Qa[Traits::KD][4];
    load_q_mma_frags<Traits::KD>(p.q + q_base, p.q_stride_l, p.q_stride_d,
                                   qra, qrb, va, vb, tid4, Qa);

    float Oacc[Traits::DN8][4];
    #pragma unroll
    for (int j = 0; j < Traits::DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int64_t pool_stride = (int64_t)p.kv_head * Traits::HEAD_DIM;
    const int64_t head_off    = (int64_t)kv_head * Traits::HEAD_DIM;
    const int64_t rtt_stride  = (int64_t)p.max_context_len;

    const int tiles = (seq_len + Traits::BC - 1) / Traits::BC;
    const int qr0 = qrow0 + gid;
    const int qr1 = qrow0 + gid + 8;

    // Causal tile-skip (dead code when IsCausal == false)
    const int max_kv = qrow0 + Traits::BR - 1 + causal_off;
    const int block_max_kv =
        blockIdx.x * Traits::WARPS * Traits::BR + Traits::WARPS * Traits::BR - 1
        + causal_off;

    int t_end = tiles - 1;
    if constexpr (IsCausal) {
        int bt = block_max_kv / Traits::BC;
        if (bt < t_end) t_end = bt;
    }

    // ---- Load tile lambda: SGLang addressing ----
    auto load_tile = [&](int ti, int buf) {
        int kv0 = ti * Traits::BC;
        bf16* dK = sK + buf * Traits::BC * Traits::LD;
        bf16* dV = sV + buf * Traits::BC * Traits::LD;
        #pragma unroll
        for (int i = threadIdx.x * Traits::VEC; i < Traits::TOTAL;
             i += Traits::NUM_THREADS * Traits::VEC) {
            int r = i / Traits::HEAD_DIM, d = i % Traits::HEAD_DIM;
            int kc = kv0 + r;
            bool valid = kc < seq_len;
            int64_t slot = valid ? p.req_to_token[req_idx * rtt_stride + kc] : 0;
            valid = valid && (slot >= 0);
            int64_t gmem_base = slot * pool_stride + head_off;
            int off = r * Traits::LD + swiz_col(d, r, Traits::SWIZ_MASK);
            cp_async_16_pred(&dK[off], &p.k_cache[gmem_base + d], valid);
            cp_async_16_pred(&dV[off], &p.v_cache[gmem_base + d], valid);
        }
        cp_async_commit();
    };

    // ---- Prologue + main loop (FA2-style double-buffer) ----
    load_tile(0, 0);

    for (int ti = 0; ti <= t_end; ti++) {
        int buf = ti & 1;

        cp_async_wait_group<0>();
        __syncthreads();
        if (ti < t_end) load_tile(ti + 1, (ti + 1) & 1);

        const bf16* bK = sK + buf * Traits::BC * Traits::LD;
        const bf16* bV = sV + buf * Traits::BC * Traits::LD;
        int kv0 = ti * Traits::BC;

        if (!IsCausal || kv0 <= max_kv) {
            float Sacc[Traits::NC8][4];
            mma_compute_scores<Traits>(Qa, bK, lane, Sacc);

            #pragma unroll
            for (int n8 = 0; n8 < Traits::NC8; n8++)
                Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
                Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

            int maxc0 = IsCausal ? min(seq_len, causal_off + qr0 + 1)
                                 : seq_len;
            int maxc1 = IsCausal ? min(seq_len, causal_off + qr1 + 1)
                                 : seq_len;
            // HasMask: mask[batch, q_head, qi, kc] — kc is request-local.
            mma_softmax_tile<Traits, HasMask>(kv0, maxc0, maxc1,
                                              qr0, qr1,
                                              p.mask_b_stride, p.mask_h_stride,
                                              p.mask_q_stride,
                                              req_b, q_head,
                                              p.mask,
                                              Sacc, Oacc, m0, m1, l0, l1, lane);

            mma_pv_accumulate<Traits>(Sacc, bV, lane, Oacc);
        }
    }

    // ---- write output: packed bf16x2 stores ----
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
    const int o_base = p.qo_indptr[req_b] * p.q_stride_l + q_head * p.q_stride_h;
    #pragma unroll
    for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        if (qr0 < q_len) {
            __nv_bfloat162 v = __floats2bfloat162_rn(Oacc[dn8][0] * rl0,
                                                       Oacc[dn8][1] * rl0);
            *reinterpret_cast<__nv_bfloat162*>(
                &p.o[o_base + qr0 * p.q_stride_l + d * p.q_stride_d]) = v;
        }
        if (qr1 < q_len) {
            __nv_bfloat162 v = __floats2bfloat162_rn(Oacc[dn8][2] * rl1,
                                                       Oacc[dn8][3] * rl1);
            *reinterpret_cast<__nv_bfloat162*>(
                &p.o[o_base + qr1 * p.q_stride_l + d * p.q_stride_d]) = v;
        }
    }
}
