// tcgen05 smoke test (sm_100a-class): one CTA computes C[64][64] +=
// A[64][64] x B[64][64] (bf16 operands, fp32 accumulate) through the
// full Blackwell path — swizzled smem staging -> shared-memory
// descriptors -> 4 chained tcgen05.mma issues (K=16 each) -> mbarrier
// completion -> tcgen05.ld readback — against a CPU reference.
//
// Toolkit matrix: CUDA 13.0's ptxas accepts tcgen05 only for
// compute_100a (datacenter Blackwell). Consumer sm_120a (RTX 5090) has
// the hardware but needs CUDA 13.1+ to assemble it (humming's
// patch_cubin notes were established on CUDA 13.3 for exactly this
// reason, and humming itself runs mma.sync on sm_120). On a device the
// toolchain cannot target, this test reports SKIP.
//
// Build (sm_100a-class GPU):
//   nvcc -I csrc/kernels -gencode arch=compute_100a,code=sm_100a \
//       -std=c++17 -O3 csrc/tests/tcgen05_test.cu -o /tmp/tcgen05_test \
//   && /tmp/tcgen05_test

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/pipeline.cuh"
#include "common/swizzle.cuh"
#include "common/mma.cuh"

namespace {

constexpr int kM = 64, kN = 64, kK = 64;

struct Smem {
    __nv_bfloat16 a[kM * kK];  // 128B-swizzled staging (chunk ^ row&7)
    __nv_bfloat16 b[kN * kK];
    uint64_t mbar;
    uint32_t tmem_base;
};

__global__ void tcgen05_smoke(const __nv_bfloat16* a, const __nv_bfloat16* b,
                              float* c) {
    __shared__ Smem sm;
    const int tid = threadIdx.x;

    // Stage both operands with the 128B-swizzle pattern (Swizzle<3,3> over
    // the 16B-chunk index; 8 chunks per 64-element bf16 row).
    for (int i = tid; i < kM * kK; i += kN * 2) {  // 128 threads
        const int row = i / kK, col = i % kK;
        const unsigned lin =
            (unsigned)(row * 8 + (col >> 3));  // row * chunks + chunk
        const unsigned swz = astrai::Swizzle<3, 3>{}(lin);
        const int off =
            (int)((swz >> 3) * kK + ((swz & 7) << 3) + (col & 7));
        if (row < kM) sm.a[off] = a[i];
        if (row < kN) sm.b[off] = b[i];
    }

    if (tid == 0) astrai::mbarrier_init(&sm.mbar, 1);
    __syncthreads();
    if (tid < 32) astrai::tcgen05_alloc<64>(
        (uint32_t)__cvta_generic_to_shared(&sm.tmem_base));
    __syncthreads();
    const uint32_t tmem = sm.tmem_base;

    if (tid == 0) {
        astrai::fence_proxy_async_shared();
        // 4 chained MMAs, K=16 each: advance the descriptors by one row of
        // the K stride (16 bf16 = 32B = 2 in the descriptor's 16B units).
        const uint64_t a0 = astrai::tcgen05_smem_desc_bf16_k128(sm.a);
        const uint64_t b0 = astrai::tcgen05_smem_desc_bf16_k128(sm.b);
#pragma unroll
        for (int k = 0; k < kK / 16; ++k)
            astrai::tcgen05_mma_bf16<kM, kN>(
                tmem, a0 + 2 * (unsigned)k, b0 + 2 * (unsigned)k, k > 0);
        astrai::tcgen05_commit((uint32_t)__cvta_generic_to_shared(&sm.mbar));
    }

    astrai::mbarrier_wait_parity(&sm.mbar, 0);
    astrai::tcgen05_fence_after_thread_sync();

    // Read C back: warp w covers TMEM lanes [32w, 32w+32); 8 column groups
    // of 8 columns each.
    const int warp = tid >> 5, lane = tid & 31;
    for (int cg = 0; cg < kN / 8; ++cg) {
        uint32_t v[8];
        astrai::tcgen05_ld_32x32b_x8(tmem + (uint32_t)(cg * 8), v);
        astrai::tcgen05_wait_ld();
#pragma unroll
        for (int i = 0; i < 8; ++i)
            c[(size_t)(warp * 32 + lane) * kN + cg * 8 + i] =
                __uint_as_float(v[i]);
    }

    __syncthreads();
    if (tid < 32) astrai::tcgen05_dealloc<64>(tmem);
}

}  // namespace

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // The asm sites assemble to tcgen05 only where this build's toolchain
    // supports them (CUDA 13.0: sm_100a only — consumer sm_120a needs
    // CUDA 13.1+). Compute-capability majors: 10 = datacenter Blackwell,
    // 12 = consumer Blackwell. Running the inert stubs would deadlock in
    // mbarrier_wait_parity, so gate strictly on major == 10.
    if (prop.major != 10) {
        printf("tcgen05 smoke: SKIP (device sm_%d%d lacks/needs newer "
               "toolchain for tcgen05)\n",
               prop.major, prop.minor);
        return 0;
    }
    std::vector<__nv_bfloat16> ha(kM * kK), hb(kN * kK);
    srand(42);
    auto rnd = []() {
        return __float2bfloat16((rand() / (float)RAND_MAX - 0.5f) * 2.f);
    };
    for (auto& v : ha) v = rnd();
    for (auto& v : hb) v = rnd();

    std::vector<float> ref(kM * kN, 0.f);
    for (int i = 0; i < kM; ++i)
        for (int j = 0; j < kN; ++j) {
            float acc = 0;
            for (int k = 0; k < kK; ++k)
                acc += __bfloat162float(ha[i * kK + k]) *
                       __bfloat162float(hb[j * kK + k]);
            ref[i * kN + j] = acc;
        }

    __nv_bfloat16 *da, *db;
    float* dc;
    cudaMalloc(&da, ha.size() * 2);
    cudaMalloc(&db, hb.size() * 2);
    cudaMalloc(&dc, ref.size() * 4);
    cudaMemcpy(da, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dc, 0, ref.size() * 4);

    tcgen05_smoke<<<1, 128>>>(da, db, dc);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::vector<float> out(ref.size());
    cudaMemcpy(out.data(), dc, ref.size() * 4, cudaMemcpyDeviceToHost);

    double max_rel = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double rel =
            fabs(out[i] - ref[i]) / fmax(fabs(ref[i]), 1.0);
        if (rel > max_rel) max_rel = rel;
    }
    printf("tcgen05 64x64x64 bf16: max_rel=%.4f %s\n", max_rel,
           max_rel < 0.02 ? "PASS" : "FAIL");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return max_rel < 0.02 ? 0 : 1;
}
