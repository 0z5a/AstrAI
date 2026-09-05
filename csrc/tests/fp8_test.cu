/*
FP8 family tests: single-warp MMA demo + full GEMM correctness.

Part 1 exercises one bf16 -> fp8 -> mma.sync m16n8k32 instruction pair
(sanity for astrai::mma_sync + the fragment layout contract).
Part 2 checks launch_fp8_gemm across all four operand layouts, both K
tiles, and ragged shapes against an fp32 CPU reference.

nvcc -I csrc/kernels -arch=sm_89 -std=c++17 -O3 csrc/tests/fp8_test.cu -o /tmp/fp8_test \
    && /tmp/fp8_test
*/

#include "test_utils.cuh"

#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

#include "common/launch.cuh"
#include "common/mma.cuh"
#include "gemm/gemm.cuh"

using namespace astrai::quant;
using namespace astrai::gemm;

// ---------------------------------------------------------------------------
// Part 1: single-kernel BF16 -> FP8 MMA -> BF16 demo (m16n8k32)
// ---------------------------------------------------------------------------

namespace {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 32;

__device__ __forceinline__ unsigned pack_fp8x4(float x0, float x1, float x2,
                                                float x3) {
    __nv_fp8_e4m3 q0(x0);
    __nv_fp8_e4m3 q1(x1);
    __nv_fp8_e4m3 q2(x2);
    __nv_fp8_e4m3 q3(x3);
    return static_cast<unsigned>(q0.__x) |
           (static_cast<unsigned>(q1.__x) << 8) |
           (static_cast<unsigned>(q2.__x) << 16) |
           (static_cast<unsigned>(q3.__x) << 24);
}

__device__ __forceinline__ unsigned load_quantize_fp8x4(
    const bf16* src, float scale_inv) {
    return pack_fp8x4(__bfloat162float(src[0]) * scale_inv,
                      __bfloat162float(src[1]) * scale_inv,
                      __bfloat162float(src[2]) * scale_inv,
                      __bfloat162float(src[3]) * scale_inv);
}

__global__ void fused_bf16_fp8_mma_kernel(
    const bf16* __restrict__ a, const bf16* __restrict__ b,
    bf16* __restrict__ out, float scale_a, float scale_b) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    const int k0 = thread_in_group * 4;

    // PTX m16n8k32 A fragment: two rows, two 16-column K partitions.
    unsigned a_frag[4];
    a_frag[0] = load_quantize_fp8x4(&a[group * kMmaK + k0], 1.0f / scale_a);
    a_frag[1] =
        load_quantize_fp8x4(&a[(group + 8) * kMmaK + k0], 1.0f / scale_a);
    a_frag[2] =
        load_quantize_fp8x4(&a[group * kMmaK + k0 + 16], 1.0f / scale_a);
    a_frag[3] = load_quantize_fp8x4(&a[(group + 8) * kMmaK + k0 + 16],
                                    1.0f / scale_a);

    // B is supplied as row-major [N,K], equivalent to the col-major [K,N]
    // operand required by the MMA instruction.
    unsigned b_frag[2];
    b_frag[0] = load_quantize_fp8x4(&b[group * kMmaK + k0], 1.0f / scale_b);
    b_frag[1] =
        load_quantize_fp8x4(&b[group * kMmaK + k0 + 16], 1.0f / scale_b);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    astrai::mma_sync<__nv_fp8_e4m3>(acc, a_frag, b_frag, acc);

    const int col = thread_in_group * 2;
    const float output_scale = scale_a * scale_b;
    *reinterpret_cast<__nv_bfloat162*>(&out[group * kMmaN + col]) =
        __floats2bfloat162_rn(acc[0] * output_scale, acc[1] * output_scale);
    *reinterpret_cast<__nv_bfloat162*>(&out[(group + 8) * kMmaN + col]) =
        __floats2bfloat162_rn(acc[2] * output_scale, acc[3] * output_scale);
}

static float quantize_e4m3(float value) {
    return static_cast<float>(__nv_fp8_e4m3(value));
}

static bool test_single_mma() {
    srand(0);
    std::vector<float> a(kMmaM * kMmaK), b(kMmaN * kMmaK),
        reference(kMmaM * kMmaN, 0.0f);
    std::vector<bf16> a_bf16(kMmaM * kMmaK), b_bf16(kMmaN * kMmaK),
        output(kMmaM * kMmaN);
    for (float& value : a) value = randf() * 4.0f;
    for (float& value : b) value = randf() * 4.0f;
    for (int i = 0; i < kMmaM * kMmaK; ++i) {
        a_bf16[i] = f2bf(a[i]);
        a[i] = bf2f(a_bf16[i]);
    }
    for (int i = 0; i < kMmaN * kMmaK; ++i) {
        b_bf16[i] = f2bf(b[i]);
        b[i] = bf2f(b_bf16[i]);
    }

    const float amax = *std::max_element(
        a.begin(), a.end(),
        [](float x, float y) { return fabsf(x) < fabsf(y); });
    const float bmax = *std::max_element(
        b.begin(), b.end(),
        [](float x, float y) { return fabsf(x) < fabsf(y); });
    const float scale_a = fabsf(amax) / 448.0f;
    const float scale_b = fabsf(bmax) / 448.0f;

    for (int row = 0; row < kMmaM; ++row) {
        for (int col = 0; col < kMmaN; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < kMmaK; ++k) {
                float qa = quantize_e4m3(a[row * kMmaK + k] / scale_a);
                float qb = quantize_e4m3(b[col * kMmaK + k] / scale_b);
                sum = fmaf(qa, qb, sum);
            }
            reference[row * kMmaN + col] = sum * scale_a * scale_b;
        }
    }

    bf16 *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, a_bf16.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_b, b_bf16.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_out, output.size() * sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(d_a, a_bf16.data(), a_bf16.size() * sizeof(bf16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b_bf16.data(), b_bf16.size() * sizeof(bf16),
                          cudaMemcpyHostToDevice));

    fused_bf16_fp8_mma_kernel<<<1, 32>>>(d_a, d_b, d_out, scale_a, scale_b);
    ASTRAI_LAUNCH_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), d_out, output.size() * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (int i = 0; i < kMmaM * kMmaN; ++i) {
        float error = fabsf(bf2f(output[i]) - reference[i]);
        max_abs_error = fmaxf(max_abs_error, error);
        max_rel_error = fmaxf(
            max_rel_error, error / fmaxf(fabsf(reference[i]), 1e-4f));
    }
    const bool pass = max_abs_error < 0.05f;
    print_test_row("M=16 N=8 K=32 fused BF16->E4M3 MMA", max_abs_error,
                   max_rel_error, pass);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return pass;
}

// ---------------------------------------------------------------------------
// Part 2: GEMM correctness — layouts x K-tiles vs fp32 CPU reference
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared GEMM correctness harness — upload / reference / launch / compare
// ---------------------------------------------------------------------------

// Element conversions: host fp32 -> ElemT for the upload, ElemT -> fp32 in
// the reference kernel, OutT -> fp32 after the download. fp8 goes through
// the type constructors; bf16 keeps the rounding intrinsics.
template <typename ElemT>
static inline ElemT to_elem(float x) {
    return ElemT(x);
}
template <>
inline __nv_bfloat16 to_elem<__nv_bfloat16>(float x) {
    return f2bf(x);
}
template <typename ElemT>
__device__ __forceinline__ float elem2f(ElemT x) {
    if constexpr (std::is_same_v<ElemT, __nv_bfloat16>)
        return __bfloat162float(x);
    else
        return (float)x;
}
static inline float out2f(__nv_bfloat16 x) { return __bfloat162float(x); }
static inline float out2f(float x) { return x; }

// Naive fp32 reference on the GPU (O(m*n) to check instead of O(m*n*k) to
// compute on the host). a_rm/b_rm select each operand's storage: 1 = row
// major along the output dim, 0 = contract-contiguous (the "colmajor" of
// the canonical [K][N] view).
template <typename ElemT>
__global__ static void naive_gemm_ref(const ElemT* a, const ElemT* b,
                                      float* out, int m, int n, int k, int a_ld,
                                      int b_ld, int a_rm, int b_rm) {
    const int i = blockIdx.y * 32 + threadIdx.y;
    const int j = blockIdx.x * 32 + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.f;
    for (int kk = 0; kk < k; ++kk) {
        float av = a_rm ? elem2f(a[i * a_ld + kk]) : elem2f(a[kk * a_ld + i]);
        float bv = b_rm ? elem2f(b[kk * b_ld + j]) : elem2f(b[j * b_ld + kk]);
        acc += av * bv;
    }
    out[i * n + j] = acc;
}

// One correctness case: convert the fp32 host operands to ElemT, upload,
// run the naive reference, launch through `dispatch(GemmParams&)`, compare
// with tolerance `tol * max(|ref|, 1.0)` (the printed max_rel normalizes
// by max(|ref|, 0.5)). ElemT and OutT are independent knobs.
template <typename ElemT, typename OutT = __nv_bfloat16, typename Fn>
static bool check_gemm(const float* ha, const float* hb, int m, int n, int k,
                       int a_ld, int b_ld, int a_rm, int b_rm, const char* tag,
                       float tol, Fn&& dispatch) {
    const size_t na = (size_t)m * k, nb = (size_t)n * k, nout = (size_t)m * n;
    std::vector<ElemT> qa(na), qb(nb);
    for (size_t i = 0; i < na; ++i) qa[i] = to_elem<ElemT>(ha[i]);
    for (size_t i = 0; i < nb; ++i) qb[i] = to_elem<ElemT>(hb[i]);
    ElemT *da, *db;
    OutT* dout;
    float *d_ref, *d_scale;
    cudaMalloc(&da, na * sizeof(ElemT));
    cudaMalloc(&db, nb * sizeof(ElemT));
    cudaMalloc(&dout, nout * sizeof(OutT));
    cudaMalloc(&d_ref, nout * 4);
    cudaMalloc(&d_scale, 4);
    cudaMemcpy(da, qa.data(), na * sizeof(ElemT), cudaMemcpyHostToDevice);
    cudaMemcpy(db, qb.data(), nb * sizeof(ElemT), cudaMemcpyHostToDevice);
    const float one = 1.0f;  // fp8 epilogues read *scale; bf16 ignores it
    cudaMemcpy(d_scale, &one, 4, cudaMemcpyHostToDevice);
    naive_gemm_ref<ElemT><<<dim3((n + 31) / 32, (m + 31) / 32), dim3(32, 32)>>>(
        da, db, d_ref, m, n, k, a_ld, b_ld, a_rm, b_rm);
    ASTRAI_LAUNCH_CHECK();

    GemmParams p = {};
    p.a_ptr = da;
    p.b_ptr = db;
    p.out_ptr = dout;
    p.scale = d_scale;
    p.m = m;
    p.n = n;
    p.k = k;
    p.a_ld = a_ld;
    p.b_ld = b_ld;
    p.out_ld = n;
    dispatch(p);

    const cudaError_t e = cudaDeviceSynchronize();
    bool ok = e == cudaSuccess;
    double max_rel = 0;
    if (!ok) {
        printf("  %-18s CUDA err: %s\n", tag, cudaGetErrorString(e));
    } else {
        std::vector<OutT> hout(nout);
        std::vector<float> href(nout);
        cudaMemcpy(hout.data(), dout, nout * sizeof(OutT),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(href.data(), d_ref, nout * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nout && ok; ++i) {
            const float err = fabsf(out2f(hout[i]) - href[i]);
            max_rel = fmax(max_rel,
                           (double)(err / fmaxf(fabsf(href[i]), 0.5f)));
            if (err > tol * fmaxf(fabsf(href[i]), 1.0f)) ok = false;
        }
        printf("  %-18s max_rel=%.4f %s\n", tag, max_rel, ok ? "PASS" : "FAIL");
    }
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(d_ref);
    cudaFree(d_scale);
    return ok;
}

// ---------------------------------------------------------------------------
// Part 2: fp8 GEMM — all four operand layouts x K-tiles x production routes
// ---------------------------------------------------------------------------

// Big-CTA policies for the direct-layout cases: kK/Stages vary per case;
// the fast interior loop follows the dual-congruous rule, grouped raster 8
// matches the production dispatch.
template <typename LA, typename LB>
constexpr bool kCaseFast =
    !std::is_same_v<LA, ColMajor> && !std::is_same_v<LB, RowMajor>;
template <typename LA, typename LB, int kK, int Stages>
using CasePolicy =
    Fp8GemmPolicy<FP8Format::E4M3, LA, LB, RowMajor, __nv_bfloat16, 128, 128,
                  64, 32, kK, Stages, 8, false, kCaseFast<LA, LB>>;

// fp8 e4m3 layout case: direct big-CTA policy (dispatch=0), the production
// NN-swap route (1) or the production NT route (2).
template <typename LA, typename LB, int kK, int Stages>
static bool run_gemm_case(const float* ha, const float* hb, int m, int n,
                          int k, int a_ld, int b_ld, const char* tag,
                          int dispatch = 0) {
    auto launch = [&](GemmParams& p) {
        if (dispatch == 1)
            // Production route, NN: the dual-N-contiguous problem has no
            // dedicated instantiation — canonicalize_gemm swaps to the
            // transposed <ColMajor, ColMajor> kernel with its out-transposed
            // epilogue (see gemm.cuh).
            gemm<FP8Format::E4M3>(p, 0, false, false);
        else if (dispatch == 2)
            // Production route, NT: exercises plan_gemm's small/narrow/big
            // selection for this shape.
            gemm<FP8Format::E4M3>(p, 0, false, true);
        else
            launch_policy<CasePolicy<LA, LB, kK, Stages>>(p, 0);
    };
    return check_gemm<__nv_fp8_e4m3>(
        ha, hb, m, n, k, a_ld, b_ld,
        !std::is_same_v<LA, ColMajor>, !std::is_same_v<LB, ColMajor>, tag,
        0.06f, launch);
}

static bool test_gemm() {
    struct {
        int m, n, k;
    } cfgs[] = {
        {128, 128, 128}, {256, 128, 256}, {128, 256, 64},
        {100, 130, 96},  {64, 64, 160},   {300, 200, 320},
        {2048, 256, 512}, {1024, 1024, 512},
    };
    bool all = true;
    for (auto& c : cfgs) {
        float* ha = new float[c.m * c.k];
        float* hb_rowmajor = new float[c.k * c.n];  // [K][N] for B RowMajor
        float* hb_colmajor = new float[c.n * c.k];  // [N][K] for B ColMajor
        for (int i = 0; i < c.m * c.k; ++i) ha[i] = randf();
        for (int i = 0; i < c.k * c.n; ++i) hb_rowmajor[i] = randf();
        for (int i = 0; i < c.k * c.n; ++i)
            hb_colmajor[i / c.k * c.k + i % c.k] = hb_rowmajor[i];
        float* ha_t = new float[c.k * c.m];  // [K][M] for A ColMajor
        for (int i = 0; i < c.m; ++i)
            for (int p = 0; p < c.k; ++p) ha_t[p * c.m + i] = ha[i * c.k + p];
        printf("%dx%dx%d:\n", c.m, c.n, c.k);
        all &= run_gemm_case<RowMajor, ColMajor, 32, 3>(ha, hb_colmajor, c.m,
                                                        c.n, c.k, c.k, c.k,
                                                        "NT K32");
        all &= run_gemm_case<RowMajor, ColMajor, 64, 2>(ha, hb_colmajor, c.m,
                                                        c.n, c.k, c.k, c.k,
                                                        "NT K64");
        all &= run_gemm_case<RowMajor, RowMajor, 64, 2>(
            ha, hb_rowmajor, c.m, c.n, c.k, c.k, c.n, "NN swap", 1);
        all &= run_gemm_case<RowMajor, ColMajor, 64, 2>(
            ha, hb_colmajor, c.m, c.n, c.k, c.k, c.k, "NT disp", 2);
        all &= run_gemm_case<ColMajor, ColMajor, 32, 3>(ha_t, hb_colmajor, c.m,
                                                        c.n, c.k, c.m, c.k,
                                                        "TN K32");
        all &= run_gemm_case<ColMajor, ColMajor, 64, 2>(ha_t, hb_colmajor, c.m,
                                                        c.n, c.k, c.m, c.k,
                                                        "TN K64");
        all &= run_gemm_case<ColMajor, RowMajor, 64, 2>(ha_t, hb_rowmajor, c.m,
                                                        c.n, c.k, c.m, c.n,
                                                        "TT K64");
        delete[] ha;
        delete[] hb_rowmajor;
        delete[] hb_colmajor;
        delete[] ha_t;
    }
    return all;
}

// ---------------------------------------------------------------------------
// Part 3: dtype coverage — bf16 operands (congruous NT) and fp32 output
// ---------------------------------------------------------------------------

static bool test_gemm_dtypes() {
    bool all = true;
    auto prep = [](std::vector<float>& a, std::vector<float>& b, int m, int n,
                   int k, int seed) {
        srand(seed);
        a.resize((size_t)m * k);
        b.resize((size_t)n * k);
        for (float& v : a) v = randf();
        for (float& v : b) v = randf();
    };

    // bf16 operands through the dtype-generic GemmPolicy: big CTA (fast
    // loop, dual-congruous) and small CTA s3.
    using Bf16Big = GemmPolicy<__nv_bfloat16, RowMajor, ColMajor, RowMajor,
                               __nv_bfloat16, 128, 128, 64, 32, 64, 2, 8,
                               false, true>;
    using Bf16Small = GemmPolicy<__nv_bfloat16, RowMajor, ColMajor, RowMajor,
                                 __nv_bfloat16, 64, 64, 32, 32, 64, 3, 8,
                                 false, true>;
    printf("bf16 operands (NT congruous):\n");
    for (int k : {64, 128, 320, 512}) {
        std::vector<float> ha, hb;
        prep(ha, hb, 256, 256, k, 1234 + k);
        printf(" 256x256x%d:\n", k);
        all &= check_gemm<__nv_bfloat16>(
            ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
            "bf16 big 128x128", 0.02f,
            [&](GemmParams& p) { launch_policy<Bf16Big>(p, 0); });
        all &= check_gemm<__nv_bfloat16>(
            ha.data(), hb.data(), 256, 256, k, k, k, 1, 0,
            "bf16 small 64x64", 0.02f,
            [&](GemmParams& p) { launch_policy<Bf16Small>(p, 0); });
    }

    // fp32 output (OutT = float): one fixed narrow-CTA policy and one
    // production-planned route through the dtype-generic dispatch. The
    // narrow tile's 32KB output fits the 36KB reclaimed operand rings
    // (launch_plan compile-time-reroutes the 128x128 CTA for fat outputs).
    using Fp8F32Out = GemmPolicy<__nv_fp8_e4m3, RowMajor, ColMajor, RowMajor,
                                 float, 128, 64, 32, 32, 64, 2, 8, false,
                                 true>;
    printf("fp8 operands, fp32 output:\n");
    for (int k : {64, 320, 512}) {
        std::vector<float> ha, hb;
        prep(ha, hb, 300, 200, k, 4321 + k);
        char tag[24];
        snprintf(tag, sizeof(tag), "f32-out K%d", k);
        all &= check_gemm<__nv_fp8_e4m3, float>(
            ha.data(), hb.data(), 300, 200, k, k, k, 1, 0, tag, 0.02f,
            [&](GemmParams& p) { launch_policy<Fp8F32Out>(p, 0); });
        snprintf(tag, sizeof(tag), "f32-out disp K%d", k);
        all &= check_gemm<__nv_fp8_e4m3, float>(
            ha.data(), hb.data(), 300, 200, k, k, k, 1, 0, tag, 0.02f,
            [&](GemmParams& p) {
                gemm_dispatch<__nv_fp8_e4m3, float>(p, 0, false, true);
            });
    }
    return all;
}

}  // namespace

int main() {
    print_test_header();
    bool ok = test_single_mma();
    ok &= test_gemm();
    ok &= test_gemm_dtypes();
    printf(ok ? "All PASS\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
