#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_kv_source.cuh"

using bf16 = __nv_bfloat16;

// v9: group-split register blocking. G threads cooperate on one query row,
// each owning HEAD_DIM/G dims of qreg[]/acc[]. IsCausal and HasMask are
// compile-time bools — the compiler eliminates dead branches.
// Unified across contiguous and paged (SGLang flat-pool) K/V via KV.
// Templated on <HEAD_DIM, KV, G, ROWS, P_BC, IsCausal, HasMask>.

template <int G>
__device__ __forceinline__ float group_reduce_sum(float v, unsigned mask) {
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1)
        v += __shfl_xor_sync(mask, v, o);
    return v;
}

// load 8 contiguous bf16 from (16-byte aligned) smem as one float4
__device__ __forceinline__ void ld8(const bf16* p, float* o) {
    float4 raw = *reinterpret_cast<const float4*>(p);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float2 f = __bfloat1622float2(h[j]);
        o[2 * j] = f.x;
        o[2 * j + 1] = f.y;
    }
}

template <int HEAD_DIM, typename KV, int G, int ROWS, int P_BC, bool IsCausal, bool HasMask>
__global__ void attn_prefill_split_q_kernel_t(AttentionParams<bf16> p) {
    constexpr int DPT = HEAD_DIM / G;

    int batch, q_tile;
    if (!map_q_block<ROWS, KV>(p, batch, q_tile))
        return;

    int q_head = blockIdx.y;
    int gpos   = threadIdx.x;  // 0..G-1  (which d-chunk)
    int row    = threadIdx.y;  // 0..ROWS-1
    int q_row  = q_tile * ROWS + row;

    // Per-request dims (from KV policy — paged reads kv_indptr/qo_indptr).
    const int seq_len    = KV::kv_len(p, batch);
    const int q_len      = KV::q_len(p, batch);
    const int causal_off = KV::causal_offset(p, batch);
    const int kv_head = q_head / (p.q_head / p.kv_head);
    const KVContext kctx = KV::template make_ctx<HEAD_DIM>(p, batch, kv_head);

    __shared__ __align__(16) bf16 sK[P_BC * HEAD_DIM];
    __shared__ __align__(16) bf16 sV[P_BC * HEAD_DIM];

    // Q: stride-based load [batch, q_head, q_len, head_dim]
    const int q_base = KV::q_base(p, batch, q_head);
    float qreg[DPT];
    if (q_row < q_len) {
        int q_off = q_base + q_row * p.q_l_stride + gpos * DPT * p.q_d_stride;
#pragma unroll
        for (int i = 0; i < DPT; i++)
            qreg[i] = __bfloat162float(p.q_ptr[q_off + i * p.q_d_stride]);
    }

    float m = -FLT_MAX, l = 0.0f;
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; i++)
        acc[i] = 0.0f;

    int mask_batch_base = batch * p.mask_b_stride + q_head * p.mask_h_stride;
    int tiles   = (seq_len + P_BC - 1) / P_BC;
    int tt      = G * ROWS;
    int lid     = row * G + gpos;

    int lane_in_warp = lid & 31;
    unsigned gmask = (G == 32) ? 0xFFFFFFFFu
                               : (((1u << G) - 1u) << (lane_in_warp & ~(G - 1)));

    for (int ti = 0; ti < tiles; ti++) {
        int kv0  = ti * P_BC;
        int tlen = min(P_BC, seq_len - kv0);

        // Load K/V into shared memory (addressing via KV policy; paged
        // guards empty slots with zero-fill).
        for (int i = lid; i < tlen * HEAD_DIM; i += tt) {
            int s = i / HEAD_DIM;
            int d_dim = i % HEAD_DIM;
            int kc = kv0 + s;
            int token = KV::resolve_token(p, kctx, kc, true);
            KVAddr a = KV::kv_addr_from_token(p, kctx, token, d_dim);
            sK[i] = a.valid ? *reinterpret_cast<const bf16*>(a.k) : (bf16)0.f;
            sV[i] = a.valid ? *reinterpret_cast<const bf16*>(a.v) : (bf16)0.f;
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

        int mask_row_base = mask_batch_base + q_row * p.mask_l_stride;
        for (int s = 0; s < lim; s++) {
            const bf16* kr = sK + s * HEAD_DIM + gpos * DPT;
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < DPT; i += 8) {
                float k8[8];
                ld8(kr + i, k8);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    part = fmaf(qreg[i + j], k8[j], part);
            }
            float dot = group_reduce_sum<G>(part, gmask) * p.scale;

            int kv_idx = kv0 + s;
            if constexpr (HasMask) {
                if (!p.mask[mask_row_base + kv_idx])
                    dot = -FLT_MAX;
            }

            float nm = fmaxf(m, dot);
            float al = __expf(m - nm);
            float be = __expf(dot - nm);
            l = l * al + be;

            const bf16* vr = sV + s * HEAD_DIM + gpos * DPT;
#pragma unroll
            for (int i = 0; i < DPT; i += 8) {
                float v8[8];
                ld8(vr + i, v8);
#pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[i + j] = fmaf(v8[j], be, acc[i + j] * al);
            }
            m = nm;
        }
        __syncthreads();
    }

    if (q_row < q_len) {
        int o_off = q_base + q_row * p.q_l_stride + gpos * DPT * p.q_d_stride;
        float rl = (l > 1e-20f) ? (1.0f / l) : 0.0f;
#pragma unroll
        for (int i = 0; i < DPT; i++)
            p.o_ptr[o_off + i * p.q_d_stride] = __float2bfloat16(acc[i] * rl);
    }
}
