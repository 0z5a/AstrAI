#pragma once

// Tensor layout for Q/K/V tensors passed to attention kernels.
// Internally, kernels always operate on BHLD [batch, n_heads, seq_len, head_dim].
// When the caller passes BLHD, dims 1 and 2 are transposed at entry.
enum TensorLayout : int {
    BHLD = 0,  // [batch, n_heads, seq_len, head_dim]
    BLHD = 1,  // [batch, seq_len, n_heads, head_dim]
};


template<typename T, typename AT = float>
struct AttentionParams {
    int batch;
    int q_head;
    int kv_head;
    int q_len;
    int kv_len;
    int head_dim;
    int use_mask;
    int causal_offset;   // -1 = non-causal; >=0 = absolute position of first Q token
    int num_splits;
    float scale;

    // Q strides (element offsets for each dim — layout-agnostic)
    int q_stride_b, q_stride_h, q_stride_l, q_stride_d;
    // KV strides (K and V share the same layout — only base pointers differ)
    int kv_stride_b, kv_stride_h, kv_stride_l, kv_stride_d;

    // Mask: 2D [batch, kv_len], 3D [batch, q_len, kv_len],
    // or 4D [batch, n_heads, q_len, kv_len] (head dim broadcasts when stride=0)
    int mask_b_stride;   // batch stride
    int mask_h_stride;   // head stride (0 = broadcast across heads)
    int mask_q_stride;   // q stride (0 = all q rows share)

    const T* __restrict__ q;
    const T* __restrict__ k;
    const T* __restrict__ v;
    const bool* __restrict__ mask;

    T* __restrict__ o;
    AT* __restrict__ o_part;
    AT* __restrict__ ml_part;
};

// ---- PagedAttentionParams ----
// SGLang-style indirect params over a shared KV pool.
// k_cache/v_cache: [size, kv_head, head_dim] (bare buffers, no gather).
// req_to_token:    [num_reqs, max_context_len] token -> slot.
// req_pool_indices:[batch] rows of the current batch into req_to_token.
// kv_indptr:       [batch+1] prefix sum of per-request seq_lens (device).
// qo_indptr:       [batch+1] prefix sum of per-request q_len (prefill) or
//                  nullptr for decode (q_len == 1 everywhere).
template<typename T, typename AT = float>
struct PagedAttentionParams {
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int num_splits;
    int use_mask;
    int causal_offset;   // -1 = non-causal; >=0 = causal (per-request offset
                         //   computed inside kernel from kv_indptr/qo_indptr)
    float scale;

    // Q: [total_q, q_head, head_dim] (3D flattened — no batch dim).
    // For decode total_q == batch (q_len=1 per request).
    // For prefill total_q == qo_indptr[batch].
    int q_stride_l, q_stride_h, q_stride_d;

    // Q: [total_q, q_head, head_dim]
    const T* __restrict__ q;

    // Flat KV pool: [size, kv_head, head_dim]
    const T* __restrict__ k_cache;
    const T* __restrict__ v_cache;

    // Indexing
    const int64_t* __restrict__ req_to_token;      // [num_reqs, max_context_len]
    const int64_t* __restrict__ req_pool_indices;   // [batch]
    const int* __restrict__ kv_indptr;               // [batch+1]
    const int* __restrict__ qo_indptr;               // [batch+1] or nullptr (decode)
    int max_context_len;  // req_to_token stride (dim 1)
    int max_seq_len;      // max per-request seq_len (host-side, for split computation)
    int total_q;          // total Q tokens across all requests (host-side, for grid)
    int max_q_len;        // max per-request q_len (host-side, for prefill grid)

    // Mask: [batch, max_seq_len] (decode) or [batch, 1, q_len, kv_len]
    // (prefill, optional).  mask_h_stride/mask_q_stride are 0 when those
    // dims are size 1 (broadcast).
    int mask_b_stride;
    int mask_h_stride;
    int mask_q_stride;
    const bool* __restrict__ mask;

    T* __restrict__ o;
    AT* __restrict__ o_part;
    AT* __restrict__ ml_part;
};
