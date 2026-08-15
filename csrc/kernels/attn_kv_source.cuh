#pragma once
#include <cuda_bf16.h>
#include "attn_common.h"

// ============================================================================
// KVSource policies — the single dimension along which the paged and
// non-paged attention kernels differ.  Each kernel is templated on one of
// these (ContigKV / PagedKV) and stays fully generic: the policy owns every
// place where "where does K/V live" and "what is this request's seq_len"
// are answered.  All methods are __host__ __device__ so the same policy
// serves both the device kernels (addressing, seq_len) and the host-side
// launchers (grid / split computation).
//
//   ContigKV:  K/V are dense [batch, kv_head, kv_len, head_dim] tensors.
//              Params fields used: k, v, kv_stride_*, kv_len, q_len,
//              q_b_stride, causal_offset.
//   PagedKV:   K/V live in a flat pool [size, kv_head, head_dim] indexed via
//              req_to_token.  Params fields used: k_cache, v_cache,
//              req_to_token, req_pool_indices, kv_indptr, qo_indptr,
//              max_context_len, q_l_stride.
//
// Addressing state that is constant across a whole kernel invocation for one
// (batch, kv_head) pair is captured once by make_ctx<HEAD_DIM>() and passed
// to kv_addr, so the load loops never redo the hoistable base computation
// (e.g. the req_pool_indices global read) element-by-element.
// ============================================================================

// Every policy method is static + callable from both host and device code.
#define HOST_DEV_FORCEINLINE static __host__ __device__ __forceinline__

using bf16 = __nv_bfloat16;

// Hoisted per-(batch, kv_head) addressing context.
struct KVContext {
    int kv_base;          // contig: batch*kv_b_stride + kv_head*kv_h_stride
    int req_idx;          // paged: req_pool_indices[batch]
    int64_t rtt_stride;   // paged: max_context_len
    int64_t pool_stride;  // paged: kv_head * HEAD_DIM
    int64_t head_off;     // paged: kv_head * HEAD_DIM
};

// Per-element K/V global addresses for one (kc, d) position of a K/V tile.
// The pointers are ALWAYS the computed addresses (never nullptr) — callers
// gate on `valid` (cp.async src_size=0, or a guarded scalar deref).  `valid`
// starts as "within the request's seq_len"; the paged policy further degrades
// it when req_to_token maps the position to a negative slot (empty padding).
// This matches the original hand-rolled load loops, where the address was
// always formed and the predicate decided whether anything was read.
struct KVAddr {
    const void* k;
    const void* v;
    bool valid;
};

// ---- Contiguous K/V ----
struct ContigKV {
    static constexpr bool kPaged = false;

    // host-side length hooks (grid + split computation in the launchers)
    HOST_DEV_FORCEINLINE int host_q_blocks(const AttentionParams<bf16>& p, int rows) {
        return (p.q_len + rows - 1) / rows;
    }
    template <int ROWS>
    HOST_DEV_FORCEINLINE bool map_q_tile(const AttentionParams<bf16>&,
                                         int flat_tile, int grid_batch,
                                         int& batch, int& q_tile) {
        batch = grid_batch;
        q_tile = flat_tile;
        return true;
    }
    HOST_DEV_FORCEINLINE int host_kv_len(const AttentionParams<bf16>& p) {
        return p.kv_len;
    }

    // prefill: element offset of the request's Q rows (kernel adds qrow*q_l_stride)
    HOST_DEV_FORCEINLINE int q_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }
    // decode: same offset (q_len == 1, so there is no row stride component)
    HOST_DEV_FORCEINLINE int q_decode_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return batch * p.q_b_stride + q_head * p.q_h_stride;
    }

    HOST_DEV_FORCEINLINE int kv_len(const AttentionParams<bf16>& p, int batch) {
        return p.kv_len;
    }
    HOST_DEV_FORCEINLINE int q_len(const AttentionParams<bf16>& p, int batch) {
        return p.q_len;
    }
    HOST_DEV_FORCEINLINE int causal_offset(const AttentionParams<bf16>& p, int batch) {
        return p.causal_offset;
    }
    // decode: exclusive bound of the single query's attend range
    HOST_DEV_FORCEINLINE int decode_attend_len(const AttentionParams<bf16>& p, int batch) {
        return (p.kv_len < p.causal_offset + 1) ? p.kv_len : (p.causal_offset + 1);
    }

    template <int HEAD_DIM>
    HOST_DEV_FORCEINLINE KVContext make_ctx(
        const AttentionParams<bf16>& p, int batch, int kv_head) {
        KVContext c = {};
        c.kv_base = batch * p.kv_b_stride + kv_head * p.kv_h_stride;
        return c;
    }
    HOST_DEV_FORCEINLINE KVAddr kv_addr(
        const AttentionParams<bf16>& p, const KVContext& c, int kc, int d, bool valid) {
        const int g_off = c.kv_base + kc * p.kv_l_stride + d * p.kv_d_stride;
        return {&p.k_ptr[g_off], &p.v_ptr[g_off], valid};
    }
};

// ---- Paged (SGLang-style flat pool) K/V ----
struct PagedKV {
    static constexpr bool kPaged = true;

    HOST_DEV_FORCEINLINE int host_q_blocks(const AttentionParams<bf16>& p, int rows) {
        // sum(ceil(q_len[b] / rows)) <= ceil(total_q / rows) + batch - 1.
        return (p.q_len + rows - 1) / rows + p.batch - 1;
    }
    template <int ROWS>
    HOST_DEV_FORCEINLINE bool map_q_tile(const AttentionParams<bf16>& p,
                                         int flat_tile, int,
                                         int& batch, int& q_tile) {
        int tile_base = 0;
        for (int b = 0; b < p.batch; ++b) {
            int len = p.qo_indptr[b + 1] - p.qo_indptr[b];
            int tiles = (len + ROWS - 1) / ROWS;
            if (flat_tile < tile_base + tiles) {
                batch = b;
                q_tile = flat_tile - tile_base;
                return true;
            }
            tile_base += tiles;
        }
        return false;
    }
    HOST_DEV_FORCEINLINE int host_kv_len(const AttentionParams<bf16>& p) {
        return p.max_context_len;
    }

    // prefill: Q rows start at qo_indptr[batch] (ragged batch base)
    HOST_DEV_FORCEINLINE int q_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return p.qo_indptr[batch] * p.q_l_stride + q_head * p.q_h_stride;
    }
    // decode: Q is [batch, q_head, head_dim], so batch is the outer row
    HOST_DEV_FORCEINLINE int q_decode_base(
        const AttentionParams<bf16>& p, int batch, int q_head) {
        return batch * p.q_l_stride + q_head * p.q_h_stride;
    }

    HOST_DEV_FORCEINLINE int kv_len(const AttentionParams<bf16>& p, int batch) {
        return p.kv_indptr[batch + 1] - p.kv_indptr[batch];
    }
    HOST_DEV_FORCEINLINE int q_len(const AttentionParams<bf16>& p, int batch) {
        return p.qo_indptr[batch + 1] - p.qo_indptr[batch];
    }
    HOST_DEV_FORCEINLINE int causal_offset(const AttentionParams<bf16>& p, int batch) {
        return kv_len(p, batch) - q_len(p, batch);
    }
    // decode: the query is the last token, so [0, seq_len) IS its causal range
    HOST_DEV_FORCEINLINE int decode_attend_len(const AttentionParams<bf16>& p, int batch) {
        return kv_len(p, batch);
    }

    template <int HEAD_DIM>
    HOST_DEV_FORCEINLINE KVContext make_ctx(
        const AttentionParams<bf16>& p, int batch, int kv_head) {
        KVContext c = {};
        c.req_idx = p.req_pool_indices[batch];
        c.rtt_stride = (int64_t)p.max_context_len;
        c.pool_stride = (int64_t)p.kv_head * HEAD_DIM;
        c.head_off = (int64_t)kv_head * HEAD_DIM;
        return c;
    }
    HOST_DEV_FORCEINLINE KVAddr kv_addr(
        const AttentionParams<bf16>& p, const KVContext& c, int kc, int d, bool valid) {
        const int slot = valid ? p.req_to_token[c.req_idx * c.rtt_stride + kc] : 0;
        const bool ok = valid && (slot >= 0);
        const int64_t gmem_off = (int64_t)slot * c.pool_stride + c.head_off + d;
        return {&p.k_ptr[gmem_off], &p.v_ptr[gmem_off], ok};
    }
};

// ---- Q-block mapping ----
// Contiguous grids map directly to (batch, q_tile). Paged grids flatten the
// ragged Q tiles, so one thread resolves the request and broadcasts it.
template <int ROWS, typename KV>
__device__ __forceinline__ bool map_q_block(
    const AttentionParams<bf16>& p, int& batch, int& q_tile) {
    if constexpr (!KV::kPaged) {
        batch = blockIdx.z;
        q_tile = blockIdx.x;
        return true;
    } else {
        __shared__ int mapped_batch;
        __shared__ int mapped_q_tile;

        if ((threadIdx.x | threadIdx.y) == 0) {
            mapped_batch = -1;
            KV::template map_q_tile<ROWS>(
                p, blockIdx.x, blockIdx.z, mapped_batch, mapped_q_tile);
        }
        __syncthreads();

        batch = mapped_batch;
        q_tile = mapped_q_tile;
        return batch >= 0;
    }
}
