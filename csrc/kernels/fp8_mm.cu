// Fused BF16 -> E4M3 MMA -> BF16 matrix multiplication

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 32;
constexpr int kBlockM = 32;
constexpr int kBlockN = 32;
constexpr int kWarps = 8;
constexpr int kForwardBlockM = 64;
constexpr int kForwardBlockN = 64;

__device__ __forceinline__ unsigned pack_fp8x4_scalar(float x0, float x1,
                                                       float x2, float x3) {
    __nv_fp8_e4m3 q0(x0);
    __nv_fp8_e4m3 q1(x1);
    __nv_fp8_e4m3 q2(x2);
    __nv_fp8_e4m3 q3(x3);
    return static_cast<unsigned>(q0.__x) |
           (static_cast<unsigned>(q1.__x) << 8) |
           (static_cast<unsigned>(q2.__x) << 16) |
           (static_cast<unsigned>(q3.__x) << 24);
}

__device__ __forceinline__ unsigned pack_fp8x4_vector(float x0, float x1,
                                                       float x2, float x3) {
    const auto low = __nv_cvt_float2_to_fp8x2(
        make_float2(x0, x1), __NV_SATFINITE, __NV_E4M3);
    const auto high = __nv_cvt_float2_to_fp8x2(
        make_float2(x2, x3), __NV_SATFINITE, __NV_E4M3);
    return static_cast<unsigned>(low) | (static_cast<unsigned>(high) << 16);
}

__device__ __forceinline__ void mma_fp8_16832(float d[4],
                                               const unsigned a[4],
                                               const unsigned b[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#endif
}

template <bool Transpose>
__device__ __forceinline__ float load_bf16(
    const __nv_bfloat16* src, int64_t row, int64_t col,
    int64_t rows, int64_t cols, float& amax) {
    if (row >= rows || col >= cols) return 0.0f;
    int64_t index = Transpose ? col * rows + row : row * cols + col;
    float value = __bfloat162float(src[index]);
    amax = fmaxf(amax, fabsf(value));
    return value;
}

__device__ __forceinline__ void atomic_max_float(float* destination,
                                                  float value) {
    if (destination)
        atomicMax(reinterpret_cast<unsigned*>(destination), __float_as_uint(value));
}

// One thread moves eight BF16 values (16 bytes). The async copy is issued
// through a uint4-shaped pointer so the source and destination are both
// naturally 128-bit aligned for contiguous forward GEMMs.
__device__ __forceinline__ void cp_async_bf16_8(
    __nv_bfloat16* destination, const __nv_bfloat16* source, bool valid) {
    const unsigned shared_address = __cvta_generic_to_shared(destination);
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_address), "l"(source_vec),
                    "r"(valid ? 16 : 0));
}

template <bool TrackAmax = true, bool VectorPack = true>
__device__ __forceinline__ unsigned load_fp8x4_from_bf16(
    const __nv_bfloat16* source, float scale_inv, float& amax,
    bool track_amax = true) {
    float x0 = __bfloat162float(source[0]);
    float x1 = __bfloat162float(source[1]);
    float x2 = __bfloat162float(source[2]);
    float x3 = __bfloat162float(source[3]);
    if constexpr (TrackAmax) {
        if (track_amax) {
        amax = fmaxf(amax, fmaxf(fabsf(x0), fmaxf(fabsf(x1),
                             fmaxf(fabsf(x2), fabsf(x3)))));
        }
    }
    if constexpr (VectorPack) {
        return pack_fp8x4_vector(x0 * scale_inv, x1 * scale_inv,
                                 x2 * scale_inv, x3 * scale_inv);
    } else {
        return pack_fp8x4_scalar(x0 * scale_inv, x1 * scale_inv,
                                 x2 * scale_inv, x3 * scale_inv);
    }
}

template <bool TransposeA, bool TransposeB>
__device__ __forceinline__ void load_direct_fragments(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    int64_t row0, int64_t row1, int b_row, int64_t k0,
    int64_t m, int64_t n, int64_t k, float inv_a, float inv_b,
    float& amax_a, float& amax_b, unsigned a_frag[4], unsigned b_frag[2]) {
    auto load_a = [&](int64_t row, int64_t col) {
        return load_bf16<TransposeA>(a, row, col, m, k, amax_a) * inv_a;
    };
    auto load_b = [&](int64_t col) {
        return load_bf16<TransposeB>(b, b_row, col, n, k, amax_b) * inv_b;
    };
    a_frag[0] = pack_fp8x4_scalar(load_a(row0, k0), load_a(row0, k0 + 1),
                                  load_a(row0, k0 + 2), load_a(row0, k0 + 3));
    a_frag[1] = pack_fp8x4_scalar(load_a(row1, k0), load_a(row1, k0 + 1),
                                  load_a(row1, k0 + 2), load_a(row1, k0 + 3));
    a_frag[2] = pack_fp8x4_scalar(
        load_a(row0, k0 + 16), load_a(row0, k0 + 17),
        load_a(row0, k0 + 18), load_a(row0, k0 + 19));
    a_frag[3] = pack_fp8x4_scalar(
        load_a(row1, k0 + 16), load_a(row1, k0 + 17),
        load_a(row1, k0 + 18), load_a(row1, k0 + 19));
    b_frag[0] = pack_fp8x4_scalar(load_b(k0), load_b(k0 + 1), load_b(k0 + 2),
                                  load_b(k0 + 3));
    b_frag[1] = pack_fp8x4_scalar(load_b(k0 + 16), load_b(k0 + 17),
                                  load_b(k0 + 18), load_b(k0 + 19));
}

template <bool TransposeA, bool TransposeB, bool AddBias, int BlockM = kBlockM,
          int BlockN = kBlockN, bool TrackAmax = true>
__global__ void fused_fp8_gemm_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ bias,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    float* __restrict__ amax_a,
    float* __restrict__ amax_b,
    int64_t m, int64_t n, int64_t k) {
    __shared__ __align__(16) __nv_fp8_e4m3 a_tile[2][BlockM][kMmaK];
    __shared__ __align__(16) __nv_fp8_e4m3 b_tile[2][BlockN][kMmaK];
    __shared__ __align__(16) __nv_bfloat16 a_bf16[2][BlockM][kMmaK];
    __shared__ __align__(16) __nv_bfloat16 b_bf16[2][BlockN][kMmaK];
    __shared__ float warp_amax_a[kWarps];
    __shared__ float warp_amax_b[kWarps];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int thread_in_group = lane & 3;
    constexpr int warps_n = kBlockN / kMmaN;
    const int warp_m = warp / warps_n;
    const int warp_n = warp % warps_n;
    const int64_t row_base = blockIdx.y * BlockM + warp_m * kMmaM + group;
    const int64_t output_col = blockIdx.x * BlockN + warp_n * kMmaN +
                               thread_in_group * 2;
    const float sa = *scale_a;
    const float sb = *scale_b;
    const float inv_a = 1.0f / sa;
    const float inv_b = 1.0f / sb;
    float local_amax_a = 0.0f;
    float local_amax_b = 0.0f;
    float acc[4 * (BlockM / kMmaM) * (BlockN / kBlockN)] = {};

    constexpr bool AsyncContiguous = !TransposeA && !TransposeB;
    const bool track_amax_a =
        TrackAmax && (!AsyncContiguous || blockIdx.x == 0);
    const bool track_amax_b =
        TrackAmax && (!AsyncContiguous || blockIdx.y == 0);
    auto load_bf16_tile = [&](int buffer, int64_t k_base) {
        if constexpr (AsyncContiguous) {
            const bool active_loader = true;
            const int async_row = tid >> 2;
            const int async_col = (tid & 3) * 8;
            if (active_loader) {
                 const int64_t a_row = blockIdx.y * BlockM + async_row;
                 const int64_t b_row = blockIdx.x * BlockN + async_row;
                const auto* a_source = a + a_row * k + k_base + async_col;
                const auto* b_source = b + b_row * k + k_base + async_col;
                 const bool full_chunk = k_base + async_col + 7 < k;
                const bool aligned_a =
                    (reinterpret_cast<uintptr_t>(a_source) & 15) == 0;
                const bool aligned_b =
                    (reinterpret_cast<uintptr_t>(b_source) & 15) == 0;
                if (a_row < m && full_chunk && aligned_a) {
                    cp_async_bf16_8(
                        &a_bf16[buffer][async_row][async_col], a_source, true);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        a_bf16[buffer][async_row][async_col + i] =
                            a_row < m && k_base + async_col + i < k
                            ? a_source[i]
                            : __float2bfloat16(0.0f);
                    }
                }
                if (b_row < n && full_chunk && aligned_b) {
                    cp_async_bf16_8(
                        &b_bf16[buffer][async_row][async_col], b_source, true);
                } else if (async_row < BlockN) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        b_bf16[buffer][async_row][async_col + i] =
                            b_row < n && k_base + async_col + i < k
                            ? b_source[i]
                            : __float2bfloat16(0.0f);
                    }
                }
            }
        }
    };

    if constexpr (AsyncContiguous) {
        load_bf16_tile(0, 0);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }

    const int64_t tile_count = (k + kMmaK - 1) / kMmaK;
    for (int64_t tile_index = 0; tile_index < tile_count; ++tile_index) {
        const int buffer = tile_index & 1;
        const int64_t k_base = tile_index * kMmaK;
        if constexpr (AsyncContiguous) {
            if (tile_index + 1 < tile_count) {
                load_bf16_tile(buffer ^ 1, k_base + kMmaK);
                asm volatile("cp.async.commit_group;");
            }
            // Quantization is performed from the prefetched BF16 tile while
            // the next tile is in flight. No FP8 global temporary is used.
            const int quant_row = tid >> 2;
            const int quant_col = (tid & 3) * 8;
            const __nv_bfloat16* a_source =
                &a_bf16[buffer][quant_row][quant_col];
            *reinterpret_cast<unsigned*>(&a_tile[buffer][quant_row][quant_col]) =
                load_fp8x4_from_bf16<TrackAmax, !TransposeA && !TransposeB>(
                    a_source, inv_a, local_amax_a, track_amax_a);
            *reinterpret_cast<unsigned*>(&a_tile[buffer][quant_row][quant_col + 4]) =
                load_fp8x4_from_bf16<TrackAmax, !TransposeA && !TransposeB>(
                    a_source + 4, inv_a, local_amax_a, track_amax_a);
            if (quant_row < BlockN) {
                const __nv_bfloat16* b_source =
                    &b_bf16[buffer][quant_row][quant_col];
                *reinterpret_cast<unsigned*>(
                    &b_tile[buffer][quant_row][quant_col]) =
                    load_fp8x4_from_bf16<TrackAmax, !TransposeA && !TransposeB>(
                        b_source, inv_b, local_amax_b, track_amax_b);
                *reinterpret_cast<unsigned*>(
                    &b_tile[buffer][quant_row][quant_col + 4]) =
                    load_fp8x4_from_bf16<TrackAmax, !TransposeA && !TransposeB>(
                        b_source + 4, inv_b, local_amax_b, track_amax_b);
            }
        } else {
            const int64_t k0 = k_base + thread_in_group * 4;
            const int b_row = blockIdx.x * BlockN + warp_n * kMmaN + group;
            unsigned a_direct[4];
            unsigned b_direct[2];
            load_direct_fragments<TransposeA, TransposeB>(
                a, b, row_base, row_base + 8, b_row, k0, m, n, k, inv_a, inv_b,
                local_amax_a, local_amax_b, a_direct, b_direct);
            mma_fp8_16832(acc, a_direct, b_direct);
            continue;
        }
        __syncthreads();

        const int fragment_col = thread_in_group * 4;
        const int a_row0 = warp_m * kMmaM + group;
        const int a_row1 = a_row0 + 8;
#pragma unroll
        for (int n_tile = 0; n_tile < BlockN / kBlockN; ++n_tile) {
            const int b_row = warp_n * kMmaN + group + n_tile * kBlockN;
            unsigned b_frag[2];
            b_frag[0] = *reinterpret_cast<unsigned*>(
                &b_tile[buffer][b_row][fragment_col]);
            b_frag[1] = *reinterpret_cast<unsigned*>(
                &b_tile[buffer][b_row][fragment_col + 16]);
#pragma unroll
            for (int m_tile = 0; m_tile < BlockM / kBlockM; ++m_tile) {
                const int m_offset = m_tile * kBlockM;
                unsigned a_frag[4];
                a_frag[0] = *reinterpret_cast<unsigned*>(
                    &a_tile[buffer][a_row0 + m_offset][fragment_col]);
                a_frag[1] = *reinterpret_cast<unsigned*>(
                    &a_tile[buffer][a_row1 + m_offset][fragment_col]);
                a_frag[2] = *reinterpret_cast<unsigned*>(
                    &a_tile[buffer][a_row0 + m_offset][fragment_col + 16]);
                a_frag[3] = *reinterpret_cast<unsigned*>(
                    &a_tile[buffer][a_row1 + m_offset][fragment_col + 16]);
                mma_fp8_16832(
                    acc + (n_tile * (BlockM / kBlockM) + m_tile) * 4,
                    a_frag, b_frag);
            }
        }
        if constexpr (AsyncContiguous) {
            if (tile_index + 1 < tile_count) {
                asm volatile("cp.async.wait_group 0;");
            }
        }
        __syncthreads();
    }

    if constexpr (TrackAmax) {
    for (int offset = 16; offset; offset >>= 1) {
        local_amax_a = fmaxf(local_amax_a,
                             __shfl_xor_sync(0xffffffffu, local_amax_a, offset));
        local_amax_b = fmaxf(local_amax_b,
                             __shfl_xor_sync(0xffffffffu, local_amax_b, offset));
    }
    if (lane == 0) {
        warp_amax_a[warp] = local_amax_a;
        warp_amax_b[warp] = local_amax_b;
    }
    __syncthreads();
    if (warp == 0) {
        float block_amax_a = lane < kWarps ? warp_amax_a[lane] : 0.0f;
        float block_amax_b = lane < kWarps ? warp_amax_b[lane] : 0.0f;
        for (int offset = 16; offset; offset >>= 1) {
            block_amax_a = fmaxf(
                block_amax_a,
                __shfl_xor_sync(0xffffffffu, block_amax_a, offset));
            block_amax_b = fmaxf(
                block_amax_b,
                __shfl_xor_sync(0xffffffffu, block_amax_b, offset));
        }
        if (lane == 0) {
            if (track_amax_a) atomic_max_float(amax_a, block_amax_a);
            if (track_amax_b) atomic_max_float(amax_b, block_amax_b);
        }
    }
    }

    const float output_scale = sa * sb;
#pragma unroll
    for (int n_tile = 0; n_tile < BlockN / kBlockN; ++n_tile) {
        const int64_t col = output_col + n_tile * kBlockN;
#pragma unroll
        for (int m_tile = 0; m_tile < BlockM / kBlockM; ++m_tile) {
            const int64_t row0 = row_base + m_tile * kBlockM;
            const int64_t row1 = row0 + 8;
            float* tile_acc =
                acc + (n_tile * (BlockM / kBlockM) + m_tile) * 4;
            if (col < n) {
                float bias0 = 0.0f;
                float bias1 = 0.0f;
                if constexpr (AddBias) {
                    bias0 = __bfloat162float(bias[col]);
                    if (col + 1 < n)
                        bias1 = __bfloat162float(bias[col + 1]);
                }
                if (row0 < m) {
                    out[row0 * n + col] =
                        __float2bfloat16(tile_acc[0] * output_scale + bias0);
                    if (col + 1 < n)
                        out[row0 * n + col + 1] = __float2bfloat16(
                            tile_acc[1] * output_scale + bias1);
                }
                if (row1 < m) {
                    out[row1 * n + col] =
                        __float2bfloat16(tile_acc[2] * output_scale + bias0);
                    if (col + 1 < n)
                        out[row1 * n + col + 1] = __float2bfloat16(
                            tile_acc[3] * output_scale + bias1);
                }
            }
        }
    }
}

template <bool TransposeA, bool TransposeB, bool AddBias = false,
          int BlockM = kBlockM, int BlockN = kBlockN, bool TrackAmax = true>
void launch_fused_fp8_gemm(
    const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& out,
    const torch::Tensor& bias, const torch::Tensor& scale_a,
    const torch::Tensor& scale_b, torch::Tensor* amax_a,
    torch::Tensor* amax_b, int64_t m, int64_t n, int64_t k,
    cudaStream_t stream) {
    dim3 grid((n + BlockN - 1) / BlockN,
              (m + BlockM - 1) / BlockM);
    const auto* bias_ptr = AddBias
        ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr())
        : nullptr;
    fused_fp8_gemm_kernel<TransposeA, TransposeB, AddBias, BlockM, BlockN,
                          TrackAmax>
        <<<grid, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), bias_ptr,
        scale_a.data_ptr<float>(), scale_b.data_ptr<float>(),
        amax_a ? amax_a->data_ptr<float>() : nullptr,
        amax_b ? amax_b->data_ptr<float>() : nullptr, m, n, k);
}

void check_fp8_device(const torch::Tensor& tensor) {
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported;
    const int device = tensor.device().index();
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto cached = supported.find(device);
        if (cached != supported.end()) {
            TORCH_CHECK(cached->second,
                        "fused FP8 MMA requires compute capability 8.9 or newer");
            return;
        }
    }

    const auto* properties = at::cuda::getDeviceProperties(device);
    const bool is_supported = properties->major > 8 ||
                              (properties->major == 8 && properties->minor >= 9);
    {
        std::lock_guard<std::mutex> lock(mutex);
        supported.emplace(device, is_supported);
    }
    TORCH_CHECK(is_supported,
                "fused FP8 MMA requires compute capability 8.9 or newer");
}

void check_scale(const torch::Tensor& scale, const torch::Tensor& input,
                 const char* name) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                name, " must be a CUDA float32 scalar on the input device");
}

}  // namespace

torch::Tensor fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor sx,
                     torch::Tensor sw) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 &&
                    b.scalar_type() == torch::kBFloat16,
                "a and b must be bf16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.size(1) == b.size(1), "inner dim mismatch");
    check_scale(sx, a, "sx");
    check_scale(sw, a, "sw");
    check_fp8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = torch::empty({a_c.size(0), b_c.size(0)}, a_c.options());
    torch::Tensor no_bias;
    launch_fused_fp8_gemm<false, false, false, kForwardBlockM,
                          kForwardBlockN, false>(
        a_c, b_c, out, no_bias, sx, sw, nullptr, nullptr,
        a_c.size(0), b_c.size(0), a_c.size(1), stream.stream());
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor fp8_linear_forward_scaled(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor sx,
    torch::Tensor sw, torch::Tensor sx_inv, torch::Tensor sw_inv,
    torch::Tensor amax_x, torch::Tensor amax_w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                    w.scalar_type() == torch::kBFloat16,
                "x and w must be bf16");
    TORCH_CHECK(x.device() == w.device(), "x and w must be on the same device");
    check_scale(sx, x, "sx");
    check_scale(sw, x, "sw");
    check_fp8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c = x.reshape({-1, w.size(1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = x_c.size(0), k = x_c.size(1), n = w_c.size(0);
    TORCH_CHECK(w_c.dim() == 2 && w_c.size(1) == k, "inner dim mismatch");
    C10_CUDA_CHECK(cudaMemsetAsync(amax_x.data_ptr<float>(), 0, sizeof(float),
                                   stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(amax_w.data_ptr<float>(), 0, sizeof(float),
                                   stream.stream()));
    auto out = torch::empty({m, n}, x_c.options());
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.device() == x.device() &&
                        bias.scalar_type() == torch::kBFloat16 &&
                        bias.numel() == n,
                    "bias must be CUDA bf16 with shape [N]");
        launch_fused_fp8_gemm<false, false, true, kForwardBlockM,
                              kForwardBlockN>(
            x_c, w_c, out, bias, sx, sw, &amax_x, &amax_w,
            m, n, k, stream.stream());
    } else {
        launch_fused_fp8_gemm<false, false, false, kForwardBlockM,
                              kForwardBlockN>(
            x_c, w_c, out, bias, sx, sw, &amax_x, &amax_w,
            m, n, k, stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());

    (void)sx_inv;
    (void)sw_inv;
    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 1);
    shape.push_back(n);
    return out.reshape(shape);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fp8_linear_backward_scaled(
    torch::Tensor g, torch::Tensor x, torch::Tensor w,
    std::vector<int64_t> masks, torch::Tensor sg, torch::Tensor sw,
    torch::Tensor sx, torch::Tensor sg_inv, torch::Tensor sw_inv,
    torch::Tensor sx_inv, torch::Tensor amax_g) {
    TORCH_CHECK(g.is_cuda() && x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 &&
                    x.scalar_type() == torch::kBFloat16 &&
                    w.scalar_type() == torch::kBFloat16,
                "g, x, and w must be bf16");
    TORCH_CHECK(g.device() == x.device() && g.device() == w.device(),
                "g, x, and w must be on the same device");
    TORCH_CHECK(masks.size() == 3, "masks must contain three values");
    check_fp8_device(g);
    const at::cuda::OptionalCUDAGuard guard(g.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto g_c = g.reshape({-1, w.size(0)}).contiguous();
    auto x_c = x.reshape({-1, x.size(-1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = g_c.size(0), n = w_c.size(0), k = w_c.size(1);
    TORCH_CHECK(x_c.size(0) == m && x_c.size(1) == k && g_c.size(1) == n,
                "backward shape mismatch");

    auto grad_input = torch::empty_like(x);
    auto grad_weight = torch::empty_like(w);
    auto grad_bias = torch::empty({0}, g.options());
    C10_CUDA_CHECK(cudaMemsetAsync(amax_g.data_ptr<float>(), 0, sizeof(float),
                                   stream.stream()));
    torch::Tensor no_bias;
    bool recorded_amax = false;
    if (masks[0]) {
        auto grad_input_2d = grad_input.reshape({m, k});
        launch_fused_fp8_gemm<false, true>(
            g_c, w_c, grad_input_2d, no_bias, sg, sw, &amax_g, nullptr,
            m, k, n, stream.stream());
        recorded_amax = true;
    }
    if (masks[1]) {
        launch_fused_fp8_gemm<true, true>(
            g_c, x_c, grad_weight, no_bias, sg, sx,
            recorded_amax ? nullptr : &amax_g, nullptr,
            n, k, m, stream.stream());
        recorded_amax = true;
    }
    if (!recorded_amax) {
        amax_g.copy_(g_c.abs().amax().to(torch::kFloat32));
    }
    C10_CUDA_CHECK(cudaGetLastError());
    if (masks[2]) grad_bias = g_c.sum(0).to(g.scalar_type());

    (void)sg_inv;
    (void)sw_inv;
    (void)sx_inv;
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_mm", &fp8_mm, py::arg("a"), py::arg("b"), py::arg("sx"),
          py::arg("sw"),
          "Fused BF16 input, E4M3 MMA, FP32 accumulation, BF16 output GEMM");
    m.def("fp8_linear_forward_scaled", &fp8_linear_forward_scaled,
          py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("sx"),
          py::arg("sw"), py::arg("sx_inv"), py::arg("sw_inv"),
          py::arg("amax_x"), py::arg("amax_w"),
          "Fused BF16-to-FP8 linear forward with FP32 accumulation");
    m.def("fp8_linear_backward_scaled", &fp8_linear_backward_scaled,
          py::arg("g"), py::arg("x"), py::arg("w"), py::arg("masks"),
          py::arg("sg"), py::arg("sw"), py::arg("sx"), py::arg("sg_inv"),
          py::arg("sw_inv"), py::arg("sx_inv"), py::arg("amax_g"),
          "Fused BF16-to-FP8 linear backward with FP32 accumulation");
}
