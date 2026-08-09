#pragma once
#include <cfloat>
#include <cuda_bf16.h>
#include "attn_common.h"
#include "attn_kv_source.cuh"
#include "attn_mma_utils.cuh"

// Tensor-core prefill flash attention (raw mma.sync PTX), unified across
// contiguous and paged (SGLang flat-pool) K/V via the KV template parameter.
// One warp owns BR=16 query rows. S = Q@K^T and O = P@V run on bf16 tensor
// cores via mma.sync.m16n8k16 (f32 accumulate).
//
// KV = ContigKV (dense [batch, kv_head, kv_len, head_dim]) or PagedKV
//      (flat pool + req_to_token, ragged batches via qo_indptr/kv_indptr).
// IsCausal and HasMask are compile-time bools — the compiler eliminates all
// dead branches in the inner compute loop (FA2-style).
//
// Traits = KernelTraits<HEAD_DIM, BC, WARPS=4, STAGES=2>.
template <typename Traits, typename KV, bool IsCausal, bool HasMask>
__global__ void attn_prefill_split_q_mma_kernel(AttentionParams<bf16> p) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid = lane >> 2;   // 0..7
    const int tid4 = lane & 3;   // 0..3

    const int q_head = blockIdx.y;
    __shared__ QTileMapper<Traits::BR * Traits::WARPS, KV> qmap;
    if (!qmap.init(p, blockIdx.x, blockIdx.z))
        return;
    const int batch = qmap.batch;
    const int q_tile = qmap.q_tile;
    const int kv_head = q_head / (p.q_head / p.kv_head);
    const int qrow0 = (q_tile * Traits::WARPS + warp) * Traits::BR;

    // Per-request dims (from KV policy — paged reads kv_indptr/qo_indptr).
    const int seq_len    = KV::kv_len(p, batch);
    const int q_len      = KV::q_len(p, batch);
    const int causal_off = KV::causal_offset(p, batch);
    const KVContext kctx = KV::template make_ctx<Traits::HEAD_DIM>(p, batch, kv_head);

    // Static shared memory: double-buffered K/V (no sQ — Q goes direct
    // to registers in mma A-operand layout).
    __shared__ __align__(16) bf16 sK[Traits::STAGES * Traits::BC * Traits::LD];
    __shared__ __align__(16) bf16 sV[Traits::STAGES * Traits::BC * Traits::LD];

    // Load Q fragments straight from global into mma A-operand layout.
    const int q_base = KV::q_base(p, batch, q_head);
    const int qra = qrow0 + gid;
    const int qrb = qrow0 + gid + 8;
    const bool va = qra < q_len, vb = qrb < q_len;
    unsigned Qa[Traits::KD][4];
    load_q_mma_frags<Traits::KD>(p.q_ptr + q_base, p.q_l_stride, p.q_d_stride,
                                  qra, qrb, va, vb, tid4, Qa);

    float Oacc[Traits::DN8][4];
    #pragma unroll
    for (int j = 0; j < Traits::DN8; j++)
        Oacc[j][0] = Oacc[j][1] = Oacc[j][2] = Oacc[j][3] = 0.0f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.0f, l1 = 0.0f;

    const int tiles = (seq_len + Traits::BC - 1) / Traits::BC;
    const int qr0 = qrow0 + gid;
    const int qr1 = qrow0 + gid + 8;

    // Causal tile-skip bounds (dead code when IsCausal == false)
    const int max_kv = qrow0 + Traits::BR - 1 + causal_off;
    const int block_max_kv =
        q_tile * Traits::WARPS * Traits::BR + Traits::WARPS * Traits::BR - 1
        + causal_off;

    int t_end = tiles - 1;
    if constexpr (IsCausal) {
        int bt = block_max_kv / Traits::BC;
        if (bt < t_end) t_end = bt;
    }

    // ---- Load tile lambda: predicated cp.async (addressing via KV policy) ----
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
            KVAddr a = KV::kv_addr(p, kctx, kc, d, valid);
            int off = r * Traits::LD + swiz_col(d, r, Traits::SWIZ_MASK);
            cp_async_16_pred(&dK[off], a.k, a.valid);
            cp_async_16_pred(&dV[off], a.v, a.valid);
        }
        cp_async_commit();
    };

    // ---- Prologue: issue first tile load ----
    load_tile(0, 0);

    for (int ti = 0; ti <= t_end; ti++) {
        int buf = ti & 1;

        // Wait for current tile, then publish cross-warp + guard buffer reuse.
        cp_async_wait_group<0>();
        __syncthreads();
        if (ti < t_end) load_tile(ti + 1, (ti + 1) & 1);

        const bf16* bK = sK + buf * Traits::BC * Traits::LD;
        const bf16* bV = sV + buf * Traits::BC * Traits::LD;
        int kv0 = ti * Traits::BC;

        // Warp-level causal skip (dead branch eliminated when IsCausal == false)
        if (!IsCausal || kv0 <= max_kv) {

            float Sacc[Traits::NC8][4];
            mma_compute_scores<Traits>(Qa, bK, lane, Sacc);

            // Post-multiply scale in float (no bf16 precision loss)
            #pragma unroll
            for (int n8 = 0; n8 < Traits::NC8; n8++)
                Sacc[n8][0] *= p.scale, Sacc[n8][1] *= p.scale,
                Sacc[n8][2] *= p.scale, Sacc[n8][3] *= p.scale;

            int maxc0 = IsCausal ? min(seq_len, causal_off + qr0 + 1)
                                 : seq_len;
            int maxc1 = IsCausal ? min(seq_len, causal_off + qr1 + 1)
                                 : seq_len;
            mma_softmax_tile<Traits, HasMask>(kv0, maxc0, maxc1,
                                               qr0, qr1,
                                               p.mask_b_stride, p.mask_h_stride, p.mask_l_stride,
                                               batch, q_head, q_head,
                                               p.mask,
                                               va, vb,
                                               Sacc, Oacc, m0, m1, l0, l1, lane);

            mma_pv_accumulate<Traits>(Sacc, bV, lane, Oacc);
        }
    }

    // ---- write output: packed bf16x2 stores ----
    float rl0 = (l0 > 1e-20f) ? (1.0f / l0) : 0.0f;
    float rl1 = (l1 > 1e-20f) ? (1.0f / l1) : 0.0f;
    const int o_base = KV::q_base(p, batch, q_head);
    #pragma unroll
    for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
        int d = dn8 * 8 + 2 * tid4;
        if (qr0 < q_len) {
            __nv_bfloat162 v = __floats2bfloat162_rn(Oacc[dn8][0] * rl0,
                                                      Oacc[dn8][1] * rl0);
            *reinterpret_cast<__nv_bfloat162*>(
                &p.o_ptr[o_base + qr0 * p.q_l_stride + d * p.q_d_stride]) = v;
        }
        if (qr1 < q_len) {
            __nv_bfloat162 v = __floats2bfloat162_rn(Oacc[dn8][2] * rl1,
                                                      Oacc[dn8][3] * rl1);
            *reinterpret_cast<__nv_bfloat162*>(
                &p.o_ptr[o_base + qr1 * p.q_l_stride + d * p.q_d_stride]) = v;
        }
    }
}
