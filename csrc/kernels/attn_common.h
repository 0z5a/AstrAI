#pragma once

// Tensor layout for Q/K/V tensors passed to attention kernels.
// Internally, kernels always operate on BHLD [batch, n_heads, seq_len, head_dim].
// When the caller passes BLHD, dims 1 and 2 are transposed at entry.
enum TensorLayout : int {
    BHLD = 0,  // [batch, n_heads, seq_len, head_dim]
    BLHD = 1,  // [batch, seq_len, n_heads, head_dim]
};


// Unified attention params covering BOTH addressing modes:
//   - Contiguous K/V: dense [batch, kv_head, kv_len, head_dim] tensors (k/v).
//   - Paged (SGLang-style): flat pool [size, kv_head, head_dim] + req_to_token.
// Each kernel selects the addressing via a KVSource policy (see
// attn_kv_source.cuh); a given call only touches the fields of one mode, so
// this is a POD shared by both paths rather than two parallel structs that
// drift out of sync.
template<typename T, typename AT = float>
struct AttentionParams {
    // ---- shared across all paths ----
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int use_mask;
    int causal_offset;   // -1 = non-causal; >=0 = absolute position of first Q token
    int num_splits;
    float scale;

    // Q strides (element offsets for each dim — layout-agnostic)
    int q_stride_b, q_stride_h, q_stride_l, q_stride_d;

    // Mask: 2D [batch, kv_len], 3D [batch, q_len, kv_len],
    // or 4D [batch, n_heads, q_len, kv_len] (head dim broadcasts when stride=0)
    int mask_b_stride;   // batch stride
    int mask_h_stride;   // head stride (0 = broadcast across heads)
    int mask_q_stride;   // q stride (0 = all q rows share)
    const bool* __restrict__ mask;

    const T* __restrict__ q;
    T* __restrict__ o;
    AT* __restrict__ o_part;
    AT* __restrict__ ml_part;

    // ---- contiguous K/V mode ----
    int q_len;
    int kv_len;
    int kv_stride_b, kv_stride_h, kv_stride_l, kv_stride_d;
    const T* __restrict__ k;
    const T* __restrict__ v;

    // ---- paged (SGLang flat pool) mode ----
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
};
