#pragma once


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

template<typename T, typename AT = float>
struct PagedAttentionParams {
    int batch;
    int q_head;
    int kv_head;
    int q_len;
    int kv_len;
    int head_dim;
    int use_mask;
    int causal_offset;
    float scale;

    int num_splits;
    int page_size;
    int max_pages;

    // Q strides (layout-agnostic)
    int q_stride_b, q_stride_h, q_stride_l, q_stride_d;

    // Mask strides (2D, 3D, or 4D)
    int mask_b_stride;
    int mask_h_stride;
    int mask_q_stride;

    const T* __restrict__ q;
    const T* __restrict__ k_cache;
    const T* __restrict__ v_cache;
    const bool* __restrict__ mask;
    const int64_t* __restrict__ page_table;

    T* __restrict__ o;
    AT* __restrict__ o_part;
    AT* __restrict__ ml_part;
};
