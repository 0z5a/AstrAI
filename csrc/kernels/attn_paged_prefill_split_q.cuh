#pragma once
#include <cuda_bf16.h>
#include <float.h>
#include "attn_common.h"

using bf16 = __nv_bfloat16;

// Scalar paged prefill (fallback for sm < 80, no tensor cores).
// Reads K/V from a flat pool via req_to_token, supports ragged batches
// via qo_indptr + kv_indptr.  Mirrors the split-Q MMA kernel's indexing:
// grid (max_q_tiles, q_head, batch), block (G, ROWS).
//
// HasMask: 4D mask [batch, 1, q_len, kv_len] (True=keep), columns are
// request-local kv positions.  q_head is the q-index (mask_h broadcast).
//
// group_reduce_sum<G> is provided by attn_prefill_split_q.cuh (already
// included via the dispatcher).
template <int HEAD_DIM, int G, int ROWS, int P_BC, bool IsCausal, bool HasMask>
__global__ void paged_attn_prefill_split_q_kernel(PagedAttentionParams<bf16> p) {
    constexpr int DPT = HEAD_DIM / G;

    const int q_tile = blockIdx.x;
    const int q_head = blockIdx.y;
    const int req_b  = blockIdx.z;
    const int gpos   = threadIdx.x;  // 0..G-1 (d-chunk)
    const int row    = threadIdx.y;  // 0..ROWS-1 (q row within tile)
    const int q_row  = q_tile * ROWS + row;

    const int seq_len    = p.kv_indptr[req_b + 1] - p.kv_indptr[req_b];
    const int q_len      = p.qo_indptr[req_b + 1] - p.qo_indptr[req_b];
    const int causal_off = seq_len - q_len;
    const int64_t req_idx = p.req_pool_indices[req_b];
    const int kv_head = q_head / (p.q_head / p.kv_head);

    __shared__ __align__(16) bf16 sK[P_BC * HEAD_DIM];
    __shared__ __align__(16) bf16 sV[P_BC * HEAD_DIM];

    // Q base: absolute token = qo_indptr[req_b] + q_row.
    float qreg[DPT];
    if (q_row < q_len) {
        int q_off = (p.qo_indptr[req_b] + q_row) * p.q_stride_l
                  + q_head * p.q_stride_h + gpos * DPT * p.q_stride_d;
        #pragma unroll
        for (int i = 0; i < DPT; i++)
            qreg[i] = __bfloat162float(p.q[q_off + i * p.q_stride_d]);
    }

    float m = -FLT_MAX, l = 0.0f, acc[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; i++) acc[i] = 0.0f;

    const int64_t pool_stride = (int64_t)p.kv_head * p.head_dim;
    const int64_t head_off    = (int64_t)kv_head * p.head_dim;
    const int64_t rtt_stride  = (int64_t)p.max_context_len;
    const int mask_base = req_b * p.mask_b_stride + q_head * p.mask_h_stride
                        + q_row * p.mask_q_stride;

    int tiles = (seq_len + P_BC - 1) / P_BC;
    int tt = G * ROWS;
    int lid = row * G + gpos;

    // Each warp holds (32/G) q-rows; reduce only within this row's G lanes.
    int lane_in_warp = lid & 31;
    unsigned gmask = (G == 32) ? 0xFFFFFFFFu
                               : (((1u << G) - 1u) << (lane_in_warp & ~(G - 1)));

    for (int ti = 0; ti < tiles; ti++) {
        int kv0 = ti * P_BC;
        int tlen = min(P_BC, seq_len - kv0);

        // Load K/V tile into shared memory via req_to_token (request-local pos).
        for (int i = lid; i < tlen * HEAD_DIM; i += tt) {
            int s = i / HEAD_DIM, d_dim = i % HEAD_DIM;
            int pos = kv0 + s;
            int64_t slot = p.req_to_token[req_idx * rtt_stride + pos];
            int64_t off = slot * pool_stride + head_off + d_dim;
            sK[i] = (slot >= 0) ? p.k_cache[off] : __float2bfloat16(0.0f);
            sV[i] = (slot >= 0) ? p.v_cache[off] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        int lim = tlen;
        if constexpr (IsCausal) {
            if (q_row < q_len) {
                int ep = causal_off + q_row + 1;
                if (kv0 >= ep)
                    lim = 0;
                else if (kv0 + tlen > ep)
                    lim = ep - kv0;
            }
        }

        for (int s = 0; s < lim; s++) {
            bool keep = true;
            if constexpr (HasMask) {
                if (q_row < q_len && !p.mask[mask_base + kv0 + s])
                    keep = false;
            }
            float w = 0.0f;
            #pragma unroll
            for (int i = 0; i < DPT; i++)
                w += qreg[i] * __bfloat162float(sK[s * HEAD_DIM + gpos * DPT + i]);
            w = group_reduce_sum<G>(w, gmask) * p.scale;
            if (!keep) w = -FLT_MAX;

            float nm = fmaxf(m, w);
            float alpha = __expf(m - nm);
            float beta = __expf(w - nm);
            l = l * alpha + beta;
            #pragma unroll
            for (int i = 0; i < DPT; i++)
                acc[i] = acc[i] * alpha
                       + __bfloat162float(sV[s * HEAD_DIM + gpos * DPT + i]) * beta;
            m = nm;
        }
        __syncthreads();
    }

    if (q_row >= q_len) return;
    float inv = (l > 1e-20f) ? (1.0f / l) : 0.0f;
    int o_off = (p.qo_indptr[req_b] + q_row) * p.q_stride_l
              + q_head * p.q_stride_h + gpos * DPT * p.q_stride_d;
    #pragma unroll
    for (int i = 0; i < DPT; i++)
        p.o[o_off + i * p.q_stride_d] = __float2bfloat16(acc[i] * inv);
}
