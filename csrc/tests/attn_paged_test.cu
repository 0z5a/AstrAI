// Compile:
//   nvcc -I csrc -arch=sm_89 -O3 --use_fast_math --ptxas-options=-O3 \
//        --extra-device-vectorization -Xcompiler -fopenmp \
//        csrc/tests/attn_paged_test.cu \
//        -o /tmp/test_paged && /tmp/test_paged

#include <cstring>
#include <vector>
#include "test_utils.cuh"
#include "../kernels/attention/dispatchers.cuh"

struct PagedDecodeDispatch { AttentionParams<bf16>& p; template<int H> void operator()() { dispatch_paged_decode<H>(p, 0); } };
struct PagedPrefillDispatch { AttentionParams<bf16>& p; template<int H> void operator()() { dispatch_paged_prefill<H>(p, 0); } };

static int make_q_tile_mapping(const std::vector<int>& q_lens,
                               int** d_batch, int** d_tile) {
    constexpr int ROWS = 64;
    std::vector<int> h_batch;
    std::vector<int> h_tile;
    for (int b = 0; b < (int)q_lens.size(); ++b) {
        int n_tiles = (q_lens[b] + ROWS - 1) / ROWS;
        for (int tile = 0; tile < n_tiles; ++tile) {
            h_batch.push_back(b);
            h_tile.push_back(tile);
        }
    }
    size_t bytes = h_batch.size() * sizeof(int);
    cudaMalloc(d_batch, bytes);
    cudaMalloc(d_tile, bytes);
    cudaMemcpy(*d_batch, h_batch.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_tile, h_tile.data(), bytes, cudaMemcpyHostToDevice);
    return (int)h_batch.size();
}

// ---- CPU reference: paged decode with variable seq_lens ----
// Q: [B, Hq, D], K/V pool: [pool_size, Hkv, D]
// req_to_token: [num_reqs, max_ctx_len], req_pool_indices: [B]
// kv_indptr: [B+1].  mask: [B, max_seq_len] bool (True=keep) or NULL.
static void cpu_paged_decode_ref(
    const float* Q, const float* K_pool, const float* V_pool,
    const int* req_to_token, const int* req_pool_indices,
    const int* kv_indptr, const bool* mask, int mask_b_stride,
    int B, int Hq, int Hkv, int D, int max_ctx_len,
    float* O)
{
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hkv;
    for (int b = 0; b < B; b++) {
        int seq_len = kv_indptr[b + 1] - kv_indptr[b];
        int req_idx = req_pool_indices[b];
        #pragma omp parallel for schedule(dynamic)
        for (int h = 0; h < Hq; h++) {
            int kv_h = h / n_rep;
            float mv = -INFINITY, sv = 0.0f;
            float accum[256] = {0.0f};
            for (int kj = 0; kj < seq_len; kj++) {
                if (mask && !mask[b * mask_b_stride + kj]) continue;
                int slot = req_to_token[req_idx * max_ctx_len + kj];
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q[(b * Hq + h) * D + d] *
                           K_pool[slot * Hkv * D + kv_h * D + d];
                dot *= scale;
                float nm = fmaxf(mv, dot);
                float a = expf(mv - nm);
                float be = expf(dot - nm);
                sv = sv * a + be;
                for (int d = 0; d < D; d++)
                    accum[d] = accum[d] * a +
                               V_pool[slot * Hkv * D + kv_h * D + d] * be;
                mv = nm;
            }
            float inv = 1.0f / sv;
            for (int d = 0; d < D; d++)
                O[(b * Hq + h) * D + d] = accum[d] * inv;
        }
    }
}

// ---- CPU reference: paged prefill with ragged batch ----
// Q: [total_q, Hq, D], K/V pool: [pool_size, Hkv, D]
// req_to_token: [num_reqs, max_ctx_len], req_pool_indices: [B]
// kv_indptr: [B+1], qo_indptr: [B+1].
// mask: [B, max_q_len, max_seq_len] bool (True=keep, q-local + kv-local
//   positions) or NULL.  Used only when causal==0 to apply an arbitrary
//   attention mask on top of the (unused) causal logic.
static void cpu_paged_prefill_ref(
    const float* Q, const float* K_pool, const float* V_pool,
    const int* req_to_token, const int* req_pool_indices,
    const int* kv_indptr, const int* qo_indptr,
    const bool* mask, int mask_l_stride, int mask_kv_stride,
    int B, int Hq, int Hkv, int D, int max_ctx_len, int causal,
    float* O)
{
    float scale = 1.0f / sqrtf((float)D);
    int n_rep = Hq / Hkv;
    for (int b = 0; b < B; b++) {
        int seq_len = kv_indptr[b + 1] - kv_indptr[b];
        int q_len = qo_indptr[b + 1] - qo_indptr[b];
        int causal_off = seq_len - q_len;
        int req_idx = req_pool_indices[b];
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int h = 0; h < Hq; h++) {
            for (int qi = 0; qi < q_len; qi++) {
                int kv_h = h / n_rep;
                float mv = -INFINITY, sv = 0.0f;
                float accum[256] = {0.0f};
                int lim = causal ? min(seq_len, causal_off + qi + 1) : seq_len;
                for (int kj = 0; kj < lim; kj++) {
                    if (mask && !mask[b * mask_l_stride * mask_kv_stride
                                     + qi * mask_kv_stride + kj]) continue;
                    int slot = req_to_token[req_idx * max_ctx_len + kj];
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++)
                        dot += Q[(qo_indptr[b] + qi) * Hq * D + h * D + d] *
                               K_pool[slot * Hkv * D + kv_h * D + d];
                    dot *= scale;
                    float nm = fmaxf(mv, dot);
                    float a = expf(mv - nm);
                    float be = expf(dot - nm);
                    sv = sv * a + be;
                    for (int d = 0; d < D; d++)
                        accum[d] = accum[d] * a +
                                   V_pool[slot * Hkv * D + kv_h * D + d] * be;
                    mv = nm;
                }
                float inv = 1.0f / sv;
                for (int d = 0; d < D; d++)
                    O[(qo_indptr[b] + qi) * Hq * D + h * D + d] = accum[d] * inv;
            }
        }
    }
}

// ---- paged validation table (kernel vs CPU ref, abs error only) ----
inline void print_paged_header() {
    printf("%-58s | %11s | %6s\n",
           "config", "max_err", "result");
    printf("----------------------------------------------------------------"
           "--------------------------------\n");
}

inline void print_paged_row(const char* cfg, float max_err, bool pass) {
    printf("%-58s | %11.3e | %s\n",
           cfg, max_err, pass ? "PASS" : "FAIL");
}

// ======================================================================
// DECODE TEST
// ======================================================================
template <int HEAD_DIM>
static int run_decode_test(int B, int Hq, int Hkv, int max_seq,
                            int causal, int seed, int context_capacity = 0,
                            int fixed_seq_len = 0) {
    // Variable seq_lens per request
    srand(seed);
    std::vector<int> seq_lens(B);
    for (int b = 0; b < B; b++)
        seq_lens[b] = fixed_seq_len ? fixed_seq_len : 8 + rand() % (max_seq - 8);
    int max_sl = *std::max_element(seq_lens.begin(), seq_lens.end());
    int max_ctx = context_capacity ? context_capacity : max_sl + 16;

    int pool_size = B * max_ctx;
    int num_reqs = B + 4;

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "DECODE B=%d Hq=%d Hkv=%d D=%d max_sl=%d causal=%d",
             B, Hq, Hkv, HEAD_DIM, max_sl, causal);

    size_t sz_q  = (size_t)B * Hq * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)pool_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_rtt = (size_t)num_reqs * max_ctx * sizeof(int);
    size_t sz_rpi = (size_t)B * sizeof(int);
    size_t sz_kvi = (size_t)(B + 1) * sizeof(int);
    size_t sz_op = (size_t)B * Hq * MAX_SPLITS * HEAD_DIM * sizeof(float);
    size_t sz_ml = (size_t)B * Hq * MAX_SPLITS * 2 * sizeof(float);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int *d_rtt, *d_rpi;
    int *d_kvi;
    float *d_op, *d_ml;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_rtt, sz_rtt); cudaMalloc(&d_rpi, sz_rpi);
    cudaMalloc(&d_kvi, sz_kvi);
    cudaMalloc(&d_op, sz_op); cudaMalloc(&d_ml, sz_ml);

    auto rnd = [&]() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };

    bf16* h_q = (bf16*)malloc(sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) h_q[i] = f2bf(rnd());
    cudaMemcpy(d_q, h_q, sz_q, cudaMemcpyHostToDevice);

    bf16* h_k_pool = (bf16*)malloc(sz_kv);
    bf16* h_v_pool = (bf16*)malloc(sz_kv);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) {
        h_k_pool[i] = f2bf(rnd());
        h_v_pool[i] = f2bf(rnd());
    }
    cudaMemcpy(d_k_pool, h_k_pool, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, h_v_pool, sz_kv, cudaMemcpyHostToDevice);

    // req_to_token: assign unique slots per request (scattered, not contiguous)
    int* h_rtt = (int*)malloc(sz_rtt);
    int next_slot = 0;
    for (int r = 0; r < num_reqs; r++)
        for (int p = 0; p < max_ctx; p++) {
            h_rtt[r * max_ctx + p] = next_slot % pool_size;
            next_slot++;
        }
    cudaMemcpy(d_rtt, h_rtt, sz_rtt, cudaMemcpyHostToDevice);

    // req_pool_indices: pick B random request rows
    int* h_rpi = (int*)malloc(sz_rpi);
    for (int b = 0; b < B; b++) h_rpi[b] = b;
    cudaMemcpy(d_rpi, h_rpi, sz_rpi, cudaMemcpyHostToDevice);

    // kv_indptr: prefix sum of seq_lens
    int* h_kvi = (int*)malloc(sz_kvi);
    h_kvi[0] = 0;
    for (int b = 0; b < B; b++) h_kvi[b + 1] = h_kvi[b] + seq_lens[b];
    cudaMemcpy(d_kvi, h_kvi, sz_kvi, cudaMemcpyHostToDevice);

    // CPU reference
    float* h_q_f = (float*)malloc(B * Hq * HEAD_DIM * sizeof(float));
    float* h_k_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    float* h_v_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) h_q_f[i] = bf2f(h_q[i]);
    for (int i = 0; i < pool_size * Hkv * HEAD_DIM; i++) {
        h_k_f[i] = bf2f(h_k_pool[i]);
        h_v_f[i] = bf2f(h_v_pool[i]);
    }
    float* h_o_ref = (float*)calloc(B * Hq * HEAD_DIM, sizeof(float));
    cpu_paged_decode_ref(h_q_f, h_k_f, h_v_f, h_rtt, h_rpi, h_kvi,
                            nullptr, 0,
                            B, Hq, Hkv, HEAD_DIM, max_ctx, h_o_ref);

    // Kernel launch
    AttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
    p.head_dim = HEAD_DIM;
    p.q_l_stride = Hq * HEAD_DIM; p.q_h_stride = HEAD_DIM; p.q_d_stride = 1;
    p.max_context_len = max_ctx;
    p.causal_offset = causal ? 0 : -1; p.use_mask = 0;
    p.mask = nullptr; p.mask_b_stride = 0;
    p.mask_h_stride = 0; p.mask_l_stride = 0;
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.q_ptr = d_q; p.k_ptr = d_k_pool; p.v_ptr = d_v_pool;
    p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
    p.kv_indptr = d_kvi; p.qo_indptr = nullptr;
    p.o_ptr = d_o; p.o_part = d_op; p.ml_part = d_ml;

    dispatch_by_head_dim(HEAD_DIM, PagedDecodeDispatch{p});
    cudaDeviceSynchronize();

    bf16* h_o_bf = (bf16*)malloc(sz_q);
    cudaMemcpy(h_o_bf, d_o, sz_q, cudaMemcpyDeviceToHost);
    float* h_o_got = (float*)malloc(B * Hq * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) h_o_got[i] = bf2f(h_o_bf[i]);

    const float atol = 0.01f, rtol = 0.01f;
    bool pass = true;
    float max_err = 0.0f;
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) {
        float e = fabsf(h_o_got[i] - h_o_ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * fabsf(h_o_ref[i])) { pass = false; break; }
    }

    print_paged_row(cfg, max_err, pass);

    free(h_q); free(h_k_pool); free(h_v_pool); free(h_rtt); free(h_rpi);
    free(h_kvi); free(h_q_f); free(h_k_f); free(h_v_f);
    free(h_o_ref); free(h_o_bf); free(h_o_got);
    cudaFree(d_q); cudaFree(d_o); cudaFree(d_k_pool); cudaFree(d_v_pool);
    cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi); cudaFree(d_op); cudaFree(d_ml);
    return pass ? 0 : 1;
}

// ======================================================================
// DECODE WITH MASK TEST (regression: 2D mask on mixed seq_lens)
// ======================================================================
template <int HEAD_DIM>
static int run_decode_mask_test(int B, int Hq, int Hkv, int max_seq,
                                int seed) {
    srand(seed);
    std::vector<int> seq_lens(B);
    for (int b = 0; b < B; b++)
        seq_lens[b] = 8 + rand() % (max_seq - 8);
    int max_sl = *std::max_element(seq_lens.begin(), seq_lens.end());
    int max_ctx = max_sl + 16;
    int pool_size = B * max_ctx;
    int num_reqs = B + 4;

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "DECODE-MASK B=%d Hq=%d Hkv=%d D=%d max_sl=%d",
             B, Hq, Hkv, HEAD_DIM, max_sl);

    size_t sz_q  = (size_t)B * Hq * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)pool_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_rtt = (size_t)num_reqs * max_ctx * sizeof(int);
    size_t sz_rpi = (size_t)B * sizeof(int);
    size_t sz_kvi = (size_t)(B + 1) * sizeof(int);
    size_t sz_mask = (size_t)B * max_sl * sizeof(bool);
    size_t sz_op = (size_t)B * Hq * MAX_SPLITS * HEAD_DIM * sizeof(float);
    size_t sz_ml = (size_t)B * Hq * MAX_SPLITS * 2 * sizeof(float);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int *d_rtt, *d_rpi;
    int *d_kvi;
    bool *d_mask;
    float *d_op, *d_ml;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_rtt, sz_rtt); cudaMalloc(&d_rpi, sz_rpi);
    cudaMalloc(&d_kvi, sz_kvi);
    cudaMalloc(&d_mask, sz_mask);
    cudaMalloc(&d_op, sz_op); cudaMalloc(&d_ml, sz_ml);

    auto rnd = [&]() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };

    bf16* h_q = (bf16*)malloc(sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) h_q[i] = f2bf(rnd());
    cudaMemcpy(d_q, h_q, sz_q, cudaMemcpyHostToDevice);

    bf16* h_k_pool = (bf16*)malloc(sz_kv);
    bf16* h_v_pool = (bf16*)malloc(sz_kv);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) {
        h_k_pool[i] = f2bf(rnd());
        h_v_pool[i] = f2bf(rnd());
    }
    cudaMemcpy(d_k_pool, h_k_pool, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, h_v_pool, sz_kv, cudaMemcpyHostToDevice);

    int* h_rtt = (int*)malloc(sz_rtt);
    int next_slot = 0;
    for (int r = 0; r < num_reqs; r++)
        for (int p = 0; p < max_ctx; p++) {
            h_rtt[r * max_ctx + p] = next_slot % pool_size;
            next_slot++;
        }
    cudaMemcpy(d_rtt, h_rtt, sz_rtt, cudaMemcpyHostToDevice);

    int* h_rpi = (int*)malloc(sz_rpi);
    for (int b = 0; b < B; b++) h_rpi[b] = b;
    cudaMemcpy(d_rpi, h_rpi, sz_rpi, cudaMemcpyHostToDevice);

    int* h_kvi = (int*)malloc(sz_kvi);
    h_kvi[0] = 0;
    for (int b = 0; b < B; b++) h_kvi[b + 1] = h_kvi[b] + seq_lens[b];
    cudaMemcpy(d_kvi, h_kvi, sz_kvi, cudaMemcpyHostToDevice);

    // Mask: keep first half of each request's kv range, drop the rest —
    // exercises the HasMask path with per-request seq_len.
    bool* h_mask = (bool*)malloc(sz_mask);
    for (int b = 0; b < B; b++)
        for (int k = 0; k < max_sl; k++)
            h_mask[b * max_sl + k] = (k < seq_lens[b]) && (k % 2 == 0);
    cudaMemcpy(d_mask, h_mask, sz_mask, cudaMemcpyHostToDevice);

    float* h_q_f = (float*)malloc(B * Hq * HEAD_DIM * sizeof(float));
    float* h_k_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    float* h_v_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) h_q_f[i] = bf2f(h_q[i]);
    for (int i = 0; i < pool_size * Hkv * HEAD_DIM; i++) {
        h_k_f[i] = bf2f(h_k_pool[i]);
        h_v_f[i] = bf2f(h_v_pool[i]);
    }
    float* h_o_ref = (float*)calloc(B * Hq * HEAD_DIM, sizeof(float));
    cpu_paged_decode_ref(h_q_f, h_k_f, h_v_f, h_rtt, h_rpi, h_kvi,
                            h_mask, max_sl,
                            B, Hq, Hkv, HEAD_DIM, max_ctx, h_o_ref);

    AttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
    p.head_dim = HEAD_DIM;
    p.q_l_stride = Hq * HEAD_DIM; p.q_h_stride = HEAD_DIM; p.q_d_stride = 1;
    p.max_context_len = max_ctx;
    p.causal_offset = -1; p.use_mask = 1;
    p.mask = d_mask; p.mask_b_stride = max_sl;
    p.mask_h_stride = 0; p.mask_l_stride = 0;
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.q_ptr = d_q; p.k_ptr = d_k_pool; p.v_ptr = d_v_pool;
    p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
    p.kv_indptr = d_kvi; p.qo_indptr = nullptr;
    p.o_ptr = d_o; p.o_part = d_op; p.ml_part = d_ml;

    dispatch_by_head_dim(HEAD_DIM, PagedDecodeDispatch{p});
    cudaDeviceSynchronize();

    bf16* h_o_bf = (bf16*)malloc(sz_q);
    cudaMemcpy(h_o_bf, d_o, sz_q, cudaMemcpyDeviceToHost);
    float* h_o_got = (float*)malloc(B * Hq * HEAD_DIM * sizeof(float));
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) h_o_got[i] = bf2f(h_o_bf[i]);

    const float atol = 0.01f, rtol = 0.01f;
    bool pass = true;
    float max_err = 0.0f;
    for (int i = 0; i < B * Hq * HEAD_DIM; i++) {
        float e = fabsf(h_o_got[i] - h_o_ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * fabsf(h_o_ref[i])) { pass = false; break; }
    }

    print_paged_row(cfg, max_err, pass);

    free(h_q); free(h_k_pool); free(h_v_pool); free(h_rtt); free(h_rpi);
    free(h_kvi); free(h_mask); free(h_q_f); free(h_k_f); free(h_v_f);
    free(h_o_ref); free(h_o_bf); free(h_o_got);
    cudaFree(d_q); cudaFree(d_o); cudaFree(d_k_pool); cudaFree(d_v_pool);
    cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi); cudaFree(d_mask);
    cudaFree(d_op); cudaFree(d_ml);
    return pass ? 0 : 1;
}

// ======================================================================
// PREFILL TEST
// ======================================================================
template <int HEAD_DIM>
static int run_prefill_test(int B, int Hq, int Hkv,
                             std::vector<int>& q_lens,
                             std::vector<int>& kv_lens,
                             int causal, int seed) {
    int total_q = 0;
    int max_sl = 0;
    for (int b = 0; b < B; b++) {
        total_q += q_lens[b];
        max_sl = max(max_sl, kv_lens[b]);
    }
    int max_ctx = max_sl + 16;
    int pool_size = B * max_ctx;
    int num_reqs = B + 4;

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "PREFILL B=%d Hq=%d Hkv=%d D=%d max_sl=%d causal=%d",
             B, Hq, Hkv, HEAD_DIM, max_sl, causal);

    size_t sz_q  = (size_t)total_q * Hq * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)pool_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_rtt = (size_t)num_reqs * max_ctx * sizeof(int);
    size_t sz_rpi = (size_t)B * sizeof(int);
    size_t sz_kvi = (size_t)(B + 1) * sizeof(int);
    size_t sz_qoi = (size_t)(B + 1) * sizeof(int);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int *d_rtt, *d_rpi;
    int *d_kvi, *d_qoi;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_rtt, sz_rtt); cudaMalloc(&d_rpi, sz_rpi);
    cudaMalloc(&d_kvi, sz_kvi); cudaMalloc(&d_qoi, sz_qoi);

    srand(seed);
    auto rnd = [&]() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };

    bf16* h_q = (bf16*)malloc(sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) h_q[i] = f2bf(rnd());
    cudaMemcpy(d_q, h_q, sz_q, cudaMemcpyHostToDevice);

    bf16* h_k_pool = (bf16*)malloc(sz_kv);
    bf16* h_v_pool = (bf16*)malloc(sz_kv);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) {
        h_k_pool[i] = f2bf(rnd());
        h_v_pool[i] = f2bf(rnd());
    }
    cudaMemcpy(d_k_pool, h_k_pool, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, h_v_pool, sz_kv, cudaMemcpyHostToDevice);

    int* h_rtt = (int*)malloc(sz_rtt);
    int next_slot = 0;
    for (int r = 0; r < num_reqs; r++)
        for (int p = 0; p < max_ctx; p++) {
            h_rtt[r * max_ctx + p] = next_slot % pool_size;
            next_slot++;
        }
    cudaMemcpy(d_rtt, h_rtt, sz_rtt, cudaMemcpyHostToDevice);

    int* h_rpi = (int*)malloc(sz_rpi);
    for (int b = 0; b < B; b++) h_rpi[b] = b;
    cudaMemcpy(d_rpi, h_rpi, sz_rpi, cudaMemcpyHostToDevice);

    int* h_kvi = (int*)malloc(sz_kvi);
    h_kvi[0] = 0;
    for (int b = 0; b < B; b++) h_kvi[b + 1] = h_kvi[b] + kv_lens[b];
    cudaMemcpy(d_kvi, h_kvi, sz_kvi, cudaMemcpyHostToDevice);

    int* h_qoi = (int*)malloc(sz_qoi);
    h_qoi[0] = 0;
    for (int b = 0; b < B; b++) h_qoi[b + 1] = h_qoi[b] + q_lens[b];
    cudaMemcpy(d_qoi, h_qoi, sz_qoi, cudaMemcpyHostToDevice);

    // CPU reference
    float* h_q_f = (float*)malloc(total_q * Hq * HEAD_DIM * sizeof(float));
    float* h_k_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    float* h_v_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    for (int i = 0; i < total_q * Hq * HEAD_DIM; i++) h_q_f[i] = bf2f(h_q[i]);
    for (int i = 0; i < pool_size * Hkv * HEAD_DIM; i++) {
        h_k_f[i] = bf2f(h_k_pool[i]);
        h_v_f[i] = bf2f(h_v_pool[i]);
    }
    float* h_o_ref = (float*)calloc(total_q * Hq * HEAD_DIM, sizeof(float));
    cpu_paged_prefill_ref(h_q_f, h_k_f, h_v_f, h_rtt, h_rpi, h_kvi, h_qoi,
                             nullptr, 0, 0,
                             B, Hq, Hkv, HEAD_DIM, max_ctx, causal, h_o_ref);

    int *d_qtb, *d_qti;
    int num_q_tiles = make_q_tile_mapping(q_lens, &d_qtb, &d_qti);

    // Kernel launch
    AttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
    p.head_dim = HEAD_DIM;
    p.q_l_stride = Hq * HEAD_DIM; p.q_h_stride = HEAD_DIM; p.q_d_stride = 1;
    p.max_context_len = max_ctx;
    p.q_len = total_q;
    p.causal_offset = causal ? 0 : -1; p.use_mask = 0;
    p.mask = nullptr; p.mask_b_stride = 0;
    p.mask_h_stride = 0; p.mask_l_stride = 0;
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.q_ptr = d_q; p.k_ptr = d_k_pool; p.v_ptr = d_v_pool;
    p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
    p.kv_indptr = d_kvi; p.qo_indptr = d_qoi;
    p.q_tile_to_batch = d_qtb; p.q_tile_to_index = d_qti;
    p.num_q_tiles = num_q_tiles;
    p.o_ptr = d_o; p.o_part = nullptr; p.ml_part = nullptr;

    dispatch_by_head_dim(HEAD_DIM, PagedPrefillDispatch{p});
    cudaDeviceSynchronize();

    bf16* h_o_bf = (bf16*)malloc(sz_q);
    cudaMemcpy(h_o_bf, d_o, sz_q, cudaMemcpyDeviceToHost);
    float* h_o_got = (float*)malloc(total_q * Hq * HEAD_DIM * sizeof(float));
    for (int i = 0; i < total_q * Hq * HEAD_DIM; i++) h_o_got[i] = bf2f(h_o_bf[i]);

    const float atol = 0.01f, rtol = 0.01f;
    bool pass = true;
    float max_err = 0.0f;
    for (int i = 0; i < total_q * Hq * HEAD_DIM; i++) {
        float e = fabsf(h_o_got[i] - h_o_ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * fabsf(h_o_ref[i])) { pass = false; break; }
    }

    print_paged_row(cfg, max_err, pass);

    free(h_q); free(h_k_pool); free(h_v_pool); free(h_rtt); free(h_rpi);
    free(h_kvi); free(h_qoi); free(h_q_f); free(h_k_f); free(h_v_f);
    free(h_o_ref); free(h_o_bf); free(h_o_got);
    cudaFree(d_q); cudaFree(d_o); cudaFree(d_k_pool); cudaFree(d_v_pool);
    cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi); cudaFree(d_qoi);
    cudaFree(d_qtb); cudaFree(d_qti);
    return pass ? 0 : 1;
}

// ======================================================================
// PREFILL WITH MASK TEST (regression: 4D causal mask on single request)
// ======================================================================
template <int HEAD_DIM>
static int run_prefill_mask_test(int Hq, int Hkv, int q_len, int seed) {
    srand(seed);
    int B = 1;
    int total_q = q_len;
    int seq_len = q_len;  // pure prefill: kv_len == q_len
    int max_ctx = seq_len + 16;
    int pool_size = B * max_ctx;
    int num_reqs = B + 4;

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "PREFILL-MASK Hq=%d Hkv=%d D=%d q_len=%d",
             Hq, Hkv, HEAD_DIM, q_len);
    fflush(stdout);

    size_t sz_q  = (size_t)total_q * Hq * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)pool_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_rtt = (size_t)num_reqs * max_ctx * sizeof(int);
    size_t sz_rpi = (size_t)B * sizeof(int);
    size_t sz_kvi = (size_t)(B + 1) * sizeof(int);
    size_t sz_qoi = (size_t)(B + 1) * sizeof(int);
    size_t sz_mask = (size_t)B * q_len * q_len * sizeof(bool);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int *d_rtt, *d_rpi;
    int *d_kvi, *d_qoi;
    bool *d_mask;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_rtt, sz_rtt); cudaMalloc(&d_rpi, sz_rpi);
    cudaMalloc(&d_kvi, sz_kvi); cudaMalloc(&d_qoi, sz_qoi);
    cudaMalloc(&d_mask, sz_mask);

    auto rnd = [&]() { return (rand() / (float)RAND_MAX) * 2.0f - 1.0f; };

    bf16* h_q = (bf16*)malloc(sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) h_q[i] = f2bf(rnd());
    cudaMemcpy(d_q, h_q, sz_q, cudaMemcpyHostToDevice);

    bf16* h_k_pool = (bf16*)malloc(sz_kv);
    bf16* h_v_pool = (bf16*)malloc(sz_kv);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) {
        h_k_pool[i] = f2bf(rnd());
        h_v_pool[i] = f2bf(rnd());
    }
    cudaMemcpy(d_k_pool, h_k_pool, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, h_v_pool, sz_kv, cudaMemcpyHostToDevice);

    int* h_rtt = (int*)malloc(sz_rtt);
    int next_slot = 0;
    for (int r = 0; r < num_reqs; r++)
        for (int p = 0; p < max_ctx; p++) {
            h_rtt[r * max_ctx + p] = next_slot % pool_size;
            next_slot++;
        }
    cudaMemcpy(d_rtt, h_rtt, sz_rtt, cudaMemcpyHostToDevice);

    int* h_rpi = (int*)malloc(sz_rpi);
    h_rpi[0] = 0;
    cudaMemcpy(d_rpi, h_rpi, sz_rpi, cudaMemcpyHostToDevice);

    int* h_kvi = (int*)malloc(sz_kvi);
    h_kvi[0] = 0; h_kvi[1] = seq_len;
    cudaMemcpy(d_kvi, h_kvi, sz_kvi, cudaMemcpyHostToDevice);

    int* h_qoi = (int*)malloc(sz_qoi);
    h_qoi[0] = 0; h_qoi[1] = q_len;
    cudaMemcpy(d_qoi, h_qoi, sz_qoi, cudaMemcpyHostToDevice);

    // 4D causal mask [B, 1, q_len, q_len], True=keep.
    bool* h_mask = (bool*)malloc(sz_mask);
    for (int qi = 0; qi < q_len; qi++)
        for (int kj = 0; kj < q_len; kj++)
            h_mask[qi * q_len + kj] = (kj <= qi);
    cudaMemcpy(d_mask, h_mask, sz_mask, cudaMemcpyHostToDevice);

    float* h_q_f = (float*)malloc(total_q * Hq * HEAD_DIM * sizeof(float));
    float* h_k_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    float* h_v_f = (float*)malloc(pool_size * Hkv * HEAD_DIM * sizeof(float));
    for (int i = 0; i < total_q * Hq * HEAD_DIM; i++) h_q_f[i] = bf2f(h_q[i]);
    for (int i = 0; i < pool_size * Hkv * HEAD_DIM; i++) {
        h_k_f[i] = bf2f(h_k_pool[i]);
        h_v_f[i] = bf2f(h_v_pool[i]);
    }
    float* h_o_ref = (float*)calloc(total_q * Hq * HEAD_DIM, sizeof(float));
    // CPU ref with causal=0 so it consults the mask (not the causal flag).
    cpu_paged_prefill_ref(h_q_f, h_k_f, h_v_f, h_rtt, h_rpi, h_kvi, h_qoi,
                             h_mask, q_len, q_len,
                             B, Hq, Hkv, HEAD_DIM, max_ctx, 0, h_o_ref);

    std::vector<int> q_lens(B, q_len);
    int *d_qtb, *d_qti;
    int num_q_tiles = make_q_tile_mapping(q_lens, &d_qtb, &d_qti);

    AttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
    p.head_dim = HEAD_DIM;
    p.q_l_stride = Hq * HEAD_DIM; p.q_h_stride = HEAD_DIM; p.q_d_stride = 1;
    p.max_context_len = max_ctx;
    p.q_len = B * q_len;
    p.causal_offset = -1; p.use_mask = 1;
    p.mask = d_mask; p.mask_b_stride = q_len * q_len;
    p.mask_h_stride = 0; p.mask_l_stride = q_len;
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.q_ptr = d_q; p.k_ptr = d_k_pool; p.v_ptr = d_v_pool;
    p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
    p.kv_indptr = d_kvi; p.qo_indptr = d_qoi;
    p.q_tile_to_batch = d_qtb; p.q_tile_to_index = d_qti;
    p.num_q_tiles = num_q_tiles;
    p.o_ptr = d_o; p.o_part = nullptr; p.ml_part = nullptr;

    dispatch_by_head_dim(HEAD_DIM, PagedPrefillDispatch{p});
    cudaDeviceSynchronize();

    bf16* h_o_bf = (bf16*)malloc(sz_q);
    cudaMemcpy(h_o_bf, d_o, sz_q, cudaMemcpyDeviceToHost);
    float* h_o_got = (float*)malloc(total_q * Hq * HEAD_DIM * sizeof(float));
    for (int i = 0; i < total_q * Hq * HEAD_DIM; i++) h_o_got[i] = bf2f(h_o_bf[i]);

    const float atol = 0.01f, rtol = 0.01f;
    bool pass = true;
    float max_err = 0.0f;
    for (int i = 0; i < total_q * Hq * HEAD_DIM; i++) {
        float e = fabsf(h_o_got[i] - h_o_ref[i]);
        if (e > max_err) max_err = e;
        if (e > atol + rtol * fabsf(h_o_ref[i])) { pass = false; break; }
    }

    print_paged_row(cfg, max_err, pass);

    free(h_q); free(h_k_pool); free(h_v_pool); free(h_rtt); free(h_rpi);
    free(h_kvi); free(h_qoi); free(h_mask); free(h_q_f); free(h_k_f); free(h_v_f);
    free(h_o_ref); free(h_o_bf); free(h_o_got);
    cudaFree(d_q); cudaFree(d_o); cudaFree(d_k_pool); cudaFree(d_v_pool);
    cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi); cudaFree(d_qoi);
    cudaFree(d_mask);
    cudaFree(d_qtb); cudaFree(d_qti);
    return pass ? 0 : 1;
}

// ======================================================================
// BENCH
// ======================================================================
template <int HEAD_DIM>
static void bench_decode(int B, int Hq, int Hkv, int seq_len) {
    int max_ctx = max(16384, seq_len + 16);
    int pool_size = B * (seq_len + 16);
    int num_reqs = B;

    size_t sz_q  = (size_t)B * Hq * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)pool_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_rtt = (size_t)num_reqs * max_ctx * sizeof(int);
    size_t sz_rpi = (size_t)B * sizeof(int);
    size_t sz_kvi = (size_t)(B + 1) * sizeof(int);
    size_t sz_op = (size_t)B * Hq * MAX_SPLITS * HEAD_DIM * sizeof(float);
    size_t sz_ml = (size_t)B * Hq * MAX_SPLITS * 2 * sizeof(float);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int *d_rtt, *d_rpi;
    int *d_kvi;
    float *d_op, *d_ml;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_rtt, sz_rtt); cudaMalloc(&d_rpi, sz_rpi);
    cudaMalloc(&d_kvi, sz_kvi);
    cudaMalloc(&d_op, sz_op); cudaMalloc(&d_ml, sz_ml);

    bf16* tmp = (bf16*)malloc(sz_kv > sz_q ? sz_kv : sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) tmp[i] = f2bf(randf());
    cudaMemcpy(d_q, tmp, sz_q, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) tmp[i] = f2bf(randf());
    cudaMemcpy(d_k_pool, tmp, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, tmp, sz_kv, cudaMemcpyHostToDevice);

    int* h_rtt = (int*)malloc(sz_rtt);
    for (int r = 0; r < num_reqs; r++)
        for (int p = 0; p < max_ctx; p++)
            h_rtt[r * max_ctx + p] = (r * max_ctx + p) % pool_size;
    cudaMemcpy(d_rtt, h_rtt, sz_rtt, cudaMemcpyHostToDevice);
    int* h_rpi = (int*)malloc(sz_rpi);
    for (int b = 0; b < B; b++) h_rpi[b] = b;
    cudaMemcpy(d_rpi, h_rpi, sz_rpi, cudaMemcpyHostToDevice);
    int* h_kvi = (int*)malloc(sz_kvi);
    h_kvi[0] = 0;
    for (int b = 0; b < B; b++) h_kvi[b + 1] = h_kvi[b] + seq_len;
    cudaMemcpy(d_kvi, h_kvi, sz_kvi, cudaMemcpyHostToDevice);

    AttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
    p.head_dim = HEAD_DIM;
    p.q_l_stride = Hq * HEAD_DIM; p.q_h_stride = HEAD_DIM; p.q_d_stride = 1;
    p.max_context_len = max_ctx;
    p.causal_offset = 0; p.use_mask = 0;
    p.mask = nullptr; p.mask_b_stride = 0;
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.q_ptr = d_q; p.k_ptr = d_k_pool; p.v_ptr = d_v_pool;
    p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
    p.kv_indptr = d_kvi; p.qo_indptr = nullptr;
    p.o_ptr = d_o; p.o_part = d_op; p.ml_part = d_ml;

    auto launch = [&]() {
        dispatch_by_head_dim(HEAD_DIM, PagedDecodeDispatch{p});
    };
    // Decode: q_len=1, query is the last token → attends to all [0, seq_len).
    // FLOPs = 2 * (QK^T + PV) = 4 * B * Hq * seq_len * D.
    double flops = 4.0 * B * Hq * (double)seq_len * HEAD_DIM;
    BenchResult r = bench_kernel(launch, 3, 10, flops);

    char cfg[64];
    snprintf(cfg, sizeof(cfg), "DEC B=%2d Hq=%2d Hk=%d kv=%4d D=%3d",
             B, Hq, Hkv, seq_len, HEAD_DIM);
    print_bench_row(cfg, r);

    free(tmp); free(h_rtt); free(h_rpi); free(h_kvi);
    cudaFree(d_q); cudaFree(d_o); cudaFree(d_k_pool); cudaFree(d_v_pool);
    cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi); cudaFree(d_op); cudaFree(d_ml);
}

template <int HEAD_DIM>
static void bench_prefill(int B, int Hq, int Hkv, int q_len, int kv_len, int causal) {
    int total_q = B * q_len;
    int max_ctx = kv_len + 16;
    int pool_size = B * max_ctx;
    int num_reqs = B;

    size_t sz_q  = (size_t)total_q * Hq * HEAD_DIM * sizeof(bf16);
    size_t sz_kv = (size_t)pool_size * Hkv * HEAD_DIM * sizeof(bf16);
    size_t sz_rtt = (size_t)num_reqs * max_ctx * sizeof(int);
    size_t sz_rpi = (size_t)B * sizeof(int);
    size_t sz_kvi = (size_t)(B + 1) * sizeof(int);
    size_t sz_qoi = (size_t)(B + 1) * sizeof(int);

    bf16 *d_q, *d_o, *d_k_pool, *d_v_pool;
    int *d_rtt, *d_rpi;
    int *d_kvi, *d_qoi;
    cudaMalloc(&d_q, sz_q); cudaMalloc(&d_o, sz_q);
    cudaMalloc(&d_k_pool, sz_kv); cudaMalloc(&d_v_pool, sz_kv);
    cudaMalloc(&d_rtt, sz_rtt); cudaMalloc(&d_rpi, sz_rpi);
    cudaMalloc(&d_kvi, sz_kvi); cudaMalloc(&d_qoi, sz_qoi);

    bf16* tmp = (bf16*)malloc(sz_kv > sz_q ? sz_kv : sz_q);
    for (size_t i = 0; i < sz_q / sizeof(bf16); i++) tmp[i] = f2bf(randf());
    cudaMemcpy(d_q, tmp, sz_q, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < sz_kv / sizeof(bf16); i++) tmp[i] = f2bf(randf());
    cudaMemcpy(d_k_pool, tmp, sz_kv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_pool, tmp, sz_kv, cudaMemcpyHostToDevice);

    int* h_rtt = (int*)malloc(sz_rtt);
    for (int r = 0; r < num_reqs; r++)
        for (int p = 0; p < max_ctx; p++)
            h_rtt[r * max_ctx + p] = (r * max_ctx + p) % pool_size;
    cudaMemcpy(d_rtt, h_rtt, sz_rtt, cudaMemcpyHostToDevice);
    int* h_rpi = (int*)malloc(sz_rpi);
    for (int b = 0; b < B; b++) h_rpi[b] = b;
    cudaMemcpy(d_rpi, h_rpi, sz_rpi, cudaMemcpyHostToDevice);
    int* h_kvi = (int*)malloc(sz_kvi);
    h_kvi[0] = 0;
    for (int b = 0; b < B; b++) h_kvi[b + 1] = h_kvi[b] + kv_len;
    cudaMemcpy(d_kvi, h_kvi, sz_kvi, cudaMemcpyHostToDevice);
    int* h_qoi = (int*)malloc(sz_qoi);
    h_qoi[0] = 0;
    for (int b = 0; b < B; b++) h_qoi[b + 1] = h_qoi[b] + q_len;
    cudaMemcpy(d_qoi, h_qoi, sz_qoi, cudaMemcpyHostToDevice);

    std::vector<int> q_lens(B, q_len);
    int *d_qtb, *d_qti;
    int num_q_tiles = make_q_tile_mapping(q_lens, &d_qtb, &d_qti);

    AttentionParams<bf16> p;
    p.batch = B; p.q_head = Hq; p.kv_head = Hkv;
    p.head_dim = HEAD_DIM;
    p.q_l_stride = Hq * HEAD_DIM; p.q_h_stride = HEAD_DIM; p.q_d_stride = 1;
    p.max_context_len = max_ctx;
    p.q_len = B * q_len;
    p.causal_offset = causal ? 0 : -1; p.use_mask = 0;
    p.mask = nullptr; p.mask_b_stride = 0;
    p.scale = 1.0f / sqrtf((float)HEAD_DIM);
    p.q_ptr = d_q; p.k_ptr = d_k_pool; p.v_ptr = d_v_pool;
    p.req_to_token = d_rtt; p.req_pool_indices = d_rpi;
    p.kv_indptr = d_kvi; p.qo_indptr = d_qoi;
    p.q_tile_to_batch = d_qtb; p.q_tile_to_index = d_qti;
    p.num_q_tiles = num_q_tiles;
    p.o_ptr = d_o; p.o_part = nullptr; p.ml_part = nullptr;

    auto launch = [&]() {
        dispatch_by_head_dim(HEAD_DIM, PagedPrefillDispatch{p});
    };
    // FLOPs = 2 * (QK^T + PV) = 4 * effective_qk_pairs * Hq * D.
    // Non-causal: effective = q_len * kv_len.
    // Causal: Q row qi attends to [0, causal_off + qi + 1) where
    //   causal_off = kv_len - q_len.  Total KV accesses per request:
    //   sum_{qi=0}^{q_len-1} (kv_len - q_len + qi + 1)
    //   = q_len * (kv_len - q_len) + q_len * (q_len + 1) / 2.
    double eff_kv;
    if (causal) {
        eff_kv = (double)q_len * (kv_len - q_len)
               + (double)q_len * (q_len + 1) / 2.0;
    } else {
        eff_kv = (double)q_len * kv_len;
    }
    double flops = 4.0 * B * Hq * eff_kv * HEAD_DIM;
    BenchResult r = bench_kernel(launch, 3, 10, flops);

    char cfg[80];
    snprintf(cfg, sizeof(cfg), "PRE B=%d Hq=%2d Hk=%d q=%4d kv=%4d D=%3d c=%d",
             B, Hq, Hkv, q_len, kv_len, HEAD_DIM, causal);
    print_bench_row(cfg, r);

    free(tmp); free(h_rtt); free(h_rpi); free(h_kvi); free(h_qoi);
    cudaFree(d_q); cudaFree(d_o); cudaFree(d_k_pool); cudaFree(d_v_pool);
    cudaFree(d_rtt); cudaFree(d_rpi); cudaFree(d_kvi); cudaFree(d_qoi);
    cudaFree(d_qtb); cudaFree(d_qti);
}

int main() {
    int fail = 0;

    // ===== DECODE TESTS =====
    printf("=== Paged Decode Tests ===\n");
    print_paged_header();
    fail += run_decode_test<128>(1, 32, 4, 512, 0, 1);
    fail += run_decode_test<128>(1, 32, 4, 1024, 0, 2);
    fail += run_decode_test<128>(4, 32, 4, 512, 0, 3);
    fail += run_decode_test<128>(8, 32, 4, 1024, 0, 4);
    fail += run_decode_test<128>(4, 32, 8, 2048, 0, 5);
    fail += run_decode_test<128>(1, 16, 1, 256, 0, 6);
    fail += run_decode_test<128>(2, 8, 2, 512, 1, 7);
    fail += run_decode_test<64>(1, 4, 2, 256, 0, 8);
    fail += run_decode_test<256>(1, 2, 1, 256, 0, 9);
    fail += run_decode_test<128>(16, 32, 4, 2048, 0, 10);
    fail += run_decode_test<128>(32, 32, 4, 1024, 0, 11);
    // Production keeps a fixed 32768-wide request table.  This forces 32
    // splits, so seq_len > 512 gives each split multiple cp.async tiles.
    fail += run_decode_test<64>(1, 24, 4, 1100, 0, 12, 32768, 1100);

    // Decode with 2D mask (regression: mixed seq_lens + HasMask)
    fail += run_decode_mask_test<128>(2, 8, 2, 256, 30);
    fail += run_decode_mask_test<128>(4, 32, 4, 512, 31);
    fail += run_decode_mask_test<64>(2, 4, 2, 128, 32);

    if (fail) { printf("\nFAILED decode tests\n"); return fail; }

    // ===== PREFILL TESTS =====
    printf("\n=== Paged Prefill Tests ===\n");
    print_paged_header();
    // Single request, pure prefill (q_len == kv_len)
    {
        std::vector<int> ql = {512};
        std::vector<int> kl = {512};
        fail += run_prefill_test<128>(1, 32, 4, ql, kl, 1, 20);
    }
    {
        std::vector<int> ql = {1024};
        std::vector<int> kl = {1024};
        fail += run_prefill_test<128>(1, 32, 4, ql, kl, 1, 21);
    }
    {
        std::vector<int> ql = {2048};
        std::vector<int> kl = {2048};
        fail += run_prefill_test<128>(1, 32, 4, ql, kl, 1, 22);
    }
    // Ragged batch: different q_lens and kv_lens
    {
        std::vector<int> ql = {128, 256, 64};
        std::vector<int> kl = {128, 256, 64};
        fail += run_prefill_test<128>(3, 32, 4, ql, kl, 1, 23);
    }
    {
        std::vector<int> ql = {64, 128, 256, 32};
        std::vector<int> kl = {64, 128, 256, 32};
        fail += run_prefill_test<128>(4, 32, 4, ql, kl, 1, 24);
    }
    // Extend: kv_len > q_len (append to existing cache)
    {
        std::vector<int> ql = {64, 128};
        std::vector<int> kl = {256, 512};
        fail += run_prefill_test<128>(2, 32, 4, ql, kl, 1, 25);
    }
    // Non-causal
    {
        std::vector<int> ql = {256, 128};
        std::vector<int> kl = {256, 128};
        fail += run_prefill_test<128>(2, 32, 4, ql, kl, 0, 26);
    }
    // Single token (q_len=1 per request, like decode but via prefill path)
    {
        std::vector<int> ql = {1, 1, 1, 1};
        std::vector<int> kl = {128, 256, 64, 512};
        fail += run_prefill_test<128>(4, 32, 4, ql, kl, 1, 27);
    }
    // D=64
    {
        std::vector<int> ql = {128, 64};
        std::vector<int> kl = {128, 64};
        fail += run_prefill_test<64>(2, 4, 2, ql, kl, 1, 28);
    }
    // D=256
    {
        std::vector<int> ql = {128, 64};
        std::vector<int> kl = {128, 64};
        fail += run_prefill_test<256>(2, 2, 1, ql, kl, 1, 29);
    }

    // Prefill with 4D causal mask (regression: single-request mask path)
    fail += run_prefill_mask_test<128>(32, 4, 512, 40);
    fail += run_prefill_mask_test<128>(32, 4, 1024, 41);
    fail += run_prefill_mask_test<64>(4, 2, 256, 42);

    if (fail) { printf("\nFAILED prefill tests\n"); return fail; }
    printf("\nAll tests passed!\n");

    // ===== BENCH =====
    printf("\n===== PAGED DECODE BENCH =====\n");
    print_bench_header();
    bench_decode<128>(1, 32, 4, 512);
    bench_decode<128>(1, 32, 4, 1024);
    bench_decode<128>(1, 32, 4, 2048);
    bench_decode<128>(1, 32, 4, 4096);
    bench_decode<128>(1, 32, 4, 16384);
    bench_decode<128>(4, 32, 4, 2048);
    bench_decode<128>(16, 32, 4, 2048);

    printf("\n===== PAGED PREFILL BENCH =====\n");
    print_bench_header();
    bench_prefill<128>(1, 32, 4, 512, 512, 0);
    bench_prefill<128>(1, 32, 4, 1024, 1024, 0);
    bench_prefill<128>(1, 32, 4, 2048, 2048, 0);
    bench_prefill<128>(1, 32, 4, 2048, 2048, 1);
    bench_prefill<128>(4, 32, 4, 2048, 2048, 1);
    bench_prefill<128>(1, 32, 4, 4096, 4096, 1);

    return 0;
}
