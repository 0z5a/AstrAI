// Per-recipe GEMM tile benchmark: times every manifest tile config
// directly (launch_policy<GemmPolicy<..., TileXxx, ...>>) across the
// llama-linear shape grid, bypassing plan_gemm entirely. This is the
// calibration harness for the device-parameterized launch planner: the
// per-recipe columns give the eff throughput scalars and the smem-driven
// residency, and the old/new pick columns compare the retired threshold
// ladder against the wave-count model on the same data.
//
// Build (RTX 5090 / sm_120):
//   nvcc -I csrc/kernels -arch=sm_120 -std=c++17 -O3 --use_fast_math \
//       csrc/tests/gemm_tile_bench.cu -o /tmp/gemm_tile_bench \
//   && /tmp/gemm_tile_bench
//
// Optional argv[1] filters cases by substring (dtype or shape name).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gemm/gemm.cuh"

namespace {

using astrai::gemm::ColMajor;
using astrai::gemm::GemmParams;
using astrai::gemm::RowMajor;
using BF16 = __nv_bfloat16;

// Deterministic small-magnitude fill (device-side; values in [-8, 8] so
// every dtype — int8 included — carries them exactly).
template <typename T>
__global__ void init_vals(T* p, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = T((float)(((int)(i * 131 + 7) % 17) - 8));
}

template <typename T>
void fill(T* dst, int64_t n) {
    init_vals<T><<<(unsigned)((n + 255) / 256), 256>>>(dst, n);
}

struct ShapeSpec {
    const char* name;
    int n, k;
};

const ShapeSpec kShapes[] = {
    {"qkv", 4096, 4096},
    {"up_gate", 11008, 4096},
    {"down", 4096, 11008},
    {"llama3_up_gate", 28672, 8192},
};
const int kMs[] = {1, 4, 16, 64, 256, 512, 1024, 2048, 4096};

// The retired threshold ladder (L20-measured), NT path only, kept here as
// the comparison baseline for the calibration table.
const char* old_pick(int64_t m, int64_t n, int64_t sms) {
    if (astrai::gemm::small_cta_padding(m, n)) return "small64-s2";
    const int64_t tiles_128 = ((m + 127) / 128) * ((n + 127) / 128);
    if (tiles_128 >= sms) {
        const int64_t tiles_narrow = ((m + 127) / 128) * ((n + 63) / 64);
        const auto waves = [sms](int64_t t) { return (t + sms - 1) / sms; };
        if (waves(tiles_narrow) * 53 < waves(tiles_128) * 100)
            return "narrow128x64";
        return "big128x128";
    }
    if (tiles_128 >= sms * 5 / 8) return "big128x128";
    const int64_t tiles_narrow = ((m + 127) / 128) * ((n + 63) / 64);
    if (tiles_narrow >= sms * 3 / 8) return "narrow128x64";
    const int64_t tiles_64 = ((m + 63) / 64) * ((n + 63) / 64);
    return tiles_64 > sms * 3 ? "small64-s3" : "small64-s2";
}

const char* plan_name(const astrai::gemm::GemmPlan& plan) {
    switch (plan.cta) {
    case astrai::gemm::GemmPlan::Cta::kBig128: return "big128x128";
    case astrai::gemm::GemmPlan::Cta::kNarrow128x64: return "narrow128x64";
    case astrai::gemm::GemmPlan::Cta::kSmall64:
        return plan.small_s3 ? "small64-s3" : "small64-s2";
    }
    return "?";
}

template <typename Policy>
float time_policy(GemmParams p, int warmup, int iters) {
    using astrai::gemm::launch_policy;
    cudaStream_t s = (cudaStream_t)0;
    for (int i = 0; i < warmup; ++i) launch_policy<Policy>(p, s);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch_policy<Policy>(p, s);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms / iters;
}

// One dtype case: instantiate the four manifest recipes for the NT linear
// orientation and time each with its production raster. Scales mirror the
// production epilogue load pattern (per-channel b_scale for W8A16, both
// sides for W8A8).
template <typename ElemA, typename ElemB>
void run_dtype(const char* dtype, bool with_a_scale, bool with_b_scale,
               int m, const ShapeSpec& s, const astrai::DeviceFacts& dev,
               const char* filter) {
    char case_tag[96];
    std::snprintf(case_tag, sizeof case_tag, "%s_%s", dtype, s.name);
    if (filter && !std::strstr(case_tag, filter)) return;

    const int64_t n = s.n, k = s.k;
    ElemA* a;
    ElemB* b;
    BF16* out;
    float* a_scale = nullptr;
    float* b_scale = nullptr;
    cudaMalloc(&a, m * k * sizeof(ElemA));
    cudaMalloc(&b, n * k * sizeof(ElemB));
    cudaMalloc(&out, m * n * sizeof(BF16));
    if (with_a_scale) cudaMalloc(&a_scale, m * sizeof(float));
    if (with_b_scale) cudaMalloc(&b_scale, n * sizeof(float));
    fill(a, (int64_t)m * k);
    fill(b, n * k);
    if (a_scale) fill(a_scale, m);
    if (b_scale) fill(b_scale, n);

    GemmParams p;
    p.a_ptr = a;
    p.b_ptr = b;
    p.out_ptr = out;
    p.a_scale = a_scale;
    p.b_scale = b_scale;
    p.a_scale_m = with_a_scale ? m : 0;
    p.b_scale_n = with_b_scale ? (int)n : 0;
    p.m = m;
    p.n = (int)n;
    p.k = (int)k;
    p.a_ld = (int)k;
    p.b_ld = (int)k;
    p.out_ld = (int)n;

    using astrai::gemm::TileBigFast;
    using astrai::gemm::TileNarrow128x64;
    using astrai::gemm::TileSmall64s2;
    using astrai::gemm::TileSmall64s3;

    struct Entry {
        const char* name;
        int raster;
        float ms;
    };
    Entry entries[] = {
        {"big128x128", 0, 0.f},
        {"narrow128x64", 0, 0.f},
        {"small64-s3", 0, 0.f},
        {"small64-s2", 0, 0.f},
    };
    // iters picked so a cell takes ~0.3 s at the ~10 ms upper end.
    const int iters = m >= 2048 ? 24 : 48;

    GemmParams pb = p;
    pb.raster = astrai::gemm::plan_raster(p, 128, 128, sizeof(ElemA),
                                          sizeof(ElemB), dev);
    entries[0].raster = pb.raster;
    entries[0].ms = time_policy<
        astrai::gemm::GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileBigFast,
                                RowMajor, BF16>>(pb, 5, iters);
    GemmParams pn = p;
    pn.raster = astrai::gemm::plan_raster(p, 128, 64, sizeof(ElemA),
                                          sizeof(ElemB), dev);
    entries[1].raster = pn.raster;
    entries[1].ms = time_policy<
        astrai::gemm::GemmPolicy<ElemA, ElemB, RowMajor, ColMajor,
                                TileNarrow128x64, RowMajor, BF16>>(pn, 5, iters);
    GemmParams ps3 = p;
    ps3.raster = astrai::gemm::plan_raster(p, 64, 64, sizeof(ElemA),
                                           sizeof(ElemB), dev);
    entries[2].raster = ps3.raster;
    entries[2].ms = time_policy<
        astrai::gemm::GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileSmall64s3,
                                RowMajor, BF16>>(ps3, 5, iters);
    GemmParams ps2 = p;
    ps2.raster = astrai::gemm::plan_raster(p, 64, 64, sizeof(ElemA),
                                           sizeof(ElemB), dev);
    entries[3].raster = ps2.raster;
    entries[3].ms = time_policy<
        astrai::gemm::GemmPolicy<ElemA, ElemB, RowMajor, ColMajor, TileSmall64s2,
                                RowMajor, BF16>>(ps2, 5, iters);

    const char* oldp = old_pick(m, n, dev.sms);
    const char* newp = plan_name(astrai::gemm::plan_gemm(
        p, (int)sizeof(ElemA), (int)sizeof(ElemB),
        astrai::gemm::gemm_perf_class<ElemA, ElemB>(), 0));
    const double flops = 2.0 * m * (double)n * k;
    for (const Entry& e : entries) {
        std::printf("%s,%s,%d,%lld,%lld,%s,%d,%.4f,%.1f,%s,%s\n", dtype,
                    s.name, m, (long long)n, (long long)k, e.name, e.raster,
                    e.ms, flops / (e.ms * 1e-3) / 1e12, oldp, newp);
    }
    std::fflush(stdout);

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
    if (a_scale) cudaFree(a_scale);
    if (b_scale) cudaFree(b_scale);
}

}  // namespace

int main(int argc, char** argv) {
    const astrai::DeviceFacts dev = astrai::device_facts();
    int dev_idx = 0;
    cudaGetDevice(&dev_idx);
    cudaDeviceProp prop = {};
    cudaGetDeviceProperties(&prop, dev_idx);
    std::printf("# %s sms=%d l2=%lld (sm_%d%d)\n", prop.name, dev.sms,
                (long long)dev.l2_bytes, prop.major, prop.minor);
    std::printf("dtype,shape,m,n,k,recipe,raster,ms,tflops,old_pick,new_pick\n");

    const char* filter = argc > 1 ? argv[1] : nullptr;
    for (int mi = 0; mi < (int)(sizeof kMs / sizeof kMs[0]); ++mi) {
        for (const ShapeSpec& s : kShapes) {
            const int m = kMs[mi];
            run_dtype<BF16, BF16>("w16a16", false, false, m, s, dev, filter);
            run_dtype<BF16, int8_t>("w8a16", false, true, m, s, dev, filter);
            run_dtype<int8_t, int8_t>("w8a8", true, true, m, s, dev, filter);
            run_dtype<__nv_fp8_e4m3, __nv_fp8_e4m3>("f8a8", false, false, m, s,
                                                    dev, filter);
        }
    }
    return 0;
}
