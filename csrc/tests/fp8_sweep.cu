/*
FP8 GEMM config sweep — pure C, no torch. Times (BM, BN, WarpM, WarpN, kK,
Stages, raster) tile configurations across the production square shapes so
the launcher's shape dispatch table is grounded in measurements.

    nvcc -I csrc -arch=sm_89 -std=c++17 -O3 --use_fast_math \
        csrc/tests/fp8_sweep.cu -o /tmp/fp8_sweep && /tmp/fp8_sweep [iters] [sizes...]
*/

#include "test_utils.cuh"

#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#include "../kernels/fp8/gemm.cuh"

using namespace astrai::fp8;

namespace {

struct BenchData {
    __nv_fp8_e4m3 *da, *db;
    __nv_bfloat16* dout;
    float* dscale;
};

template <int BM, int BN, int WM, int WN, int kK, int Stages, bool GroupRaster,
          bool LeanRing = false>
float bench_config(BenchData& d, int m, int n, int k, int iters) {
    FP8Params p = {};
    p.a_ptr = d.da;
    p.b_ptr = d.db;
    p.out_ptr = d.dout;
    p.scale = d.dscale;
    p.m = m;
    p.n = n;
    p.k = k;
    p.a_ld = k;
    p.b_ld = k;

    using Traits = Fp8GemmTraits<FP8Format::E4M3, BM, BN, kK, Stages, WM, WN>;
    using Smem = Fp8GemmSmem<Traits, RowMajor, ColMajor, false, LeanRing>;
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    dim3 block(Traits::kCtaThreads);
    const int smem = Smem::kBytes;

    auto launch = [&] {
        launch_with_smem<fp8_gemm_kernel<Traits, RowMajor, ColMajor, GroupRaster,
                                         false, LeanRing>>(
            smem, grid, block, 0, p);
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for (int i = 0; i < 3; ++i) launch();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms / iters;
}

// One named config column.
struct Col {
    const char* name;
    float (*fn)(BenchData&, int, int, int, int);
};

template <int BM, int BN, int WM, int WN, int kK, int Stages, bool R,
          bool Lean = false>
float run(BenchData& d, int m, int n, int k, int iters) {
    return bench_config<BM, BN, WM, WN, kK, Stages, R, Lean>(d, m, n, k, iters);
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 50;
    int sizes[] = {512, 1024, 2048, 4096, 8192, 0, 0, 0};
    for (int i = 2; i < argc && i < 10; ++i) sizes[i - 2] = atoi(argv[i]);

    Col cols[] = {
        {"128ring3", &run<128, 128, 64, 32, 64, 2, true>},
        {"128ring3s3", &run<128, 128, 64, 32, 64, 3, true>},
        {"128lean", &run<128, 128, 64, 32, 64, 2, true, true>},
        {"64r5s4", &run<64, 64, 32, 32, 64, 4, false>},
        {"64lean4", &run<64, 64, 32, 32, 64, 4, false, true>},
        {"64lean5", &run<64, 64, 32, 32, 64, 5, false, true>},
        {"64lean3", &run<64, 64, 32, 32, 64, 3, false, true>},
    };
    const int ncols = sizeof(cols) / sizeof(cols[0]);

    printf("%6s |", "shape");
    for (auto& c : cols) printf(" %11s |", c.name);
    printf("\n");

    for (int s : sizes) {
        if (s <= 0) continue;
        const int m = s, n = s, k = s;
        BenchData d;
        std::vector<__nv_fp8_e4m3> a((size_t)m * k), b((size_t)n * k);
        for (auto& v : a) v = __nv_fp8_e4m3(randf());
        for (auto& v : b) v = __nv_fp8_e4m3(randf());
        cudaMalloc(&d.da, a.size());
        cudaMalloc(&d.db, b.size());
        cudaMalloc(&d.dout, (size_t)m * n * 2);
        cudaMalloc(&d.dscale, 4);
        const float one = 1.0f;
        cudaMemcpy(d.dscale, &one, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d.da, a.data(), a.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d.db, b.data(), b.size(), cudaMemcpyHostToDevice);

        const double flops = 2.0 * m * n * k;
        printf("%6d |", s);
        for (auto& c : cols) {
            const float ms = c.fn(d, m, n, k, iters);
            printf(" %5.2fus %4.1fT |", ms * 1000,
                   flops / (ms * 1e-3) / 1e12);
        }
        printf("\n");
        cudaFree(d.da);
        cudaFree(d.db);
        cudaFree(d.dout);
        cudaFree(d.dscale);
    }
    return 0;
}
