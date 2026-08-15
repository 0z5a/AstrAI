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
    // Shape
    int batch;
    int q_head;
    int kv_head;
    int head_dim;
    int q_len;   // Per-request in contiguous mode; total_q in paged mode.
    int kv_len;  // Contiguous mode; paged mode uses kv_indptr.

    // Attention behavior
    float scale;
    // -1 = non-causal; >=0 = absolute position of first Q token
    int causal_offset;
    int use_mask;

    // pointers
    const T* __restrict__ q_ptr;
    const T* __restrict__ k_ptr;
    const T* __restrict__ v_ptr;
    T* __restrict__ o_ptr;
    const bool* __restrict__ mask;

    // strides
    int q_b_stride;
    int q_h_stride;
    int q_l_stride;
    int q_d_stride;

    int kv_b_stride;
    int kv_h_stride;
    int kv_l_stride;
    int kv_d_stride;

    int mask_b_stride;
    int mask_h_stride;
    int mask_l_stride;

    // Paged K/V addressing
    const int* __restrict__ req_to_token;         // [num_reqs, max_context_len]
    const int* __restrict__ req_pool_indices;     // [batch]
    const int* __restrict__ kv_indptr;             // [batch + 1]
    const int* __restrict__ qo_indptr;             // [batch + 1] or nullptr for decode
    int max_context_len; // req_to_token stride (dim 1)

    // Decode split-KV workspace
    int num_splits;
    AT* __restrict__ o_part;
    AT* __restrict__ ml_part;

};
