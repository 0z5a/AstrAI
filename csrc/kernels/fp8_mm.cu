// FP8 e4m3 matrix multiply via cuBLASLt (sm89 TN layout).
//
// cuBLASLt exposes fp8 kernels only for op(A)=T, op(B)=N on Ada; we exploit
// the identity: row-major a[M,K] == A^T as col-major [K,M] (zero copy), and
// row-major wT[N,K] == B as col-major [K,N] (zero copy). The col-major
// result D[M,N] is C^T in row-major terms, so we transpose the output once.
//
// Inputs arrive pre-scaled fp8 e4m3 tensors; output is unscaled fp32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

static std::recursive_mutex g_mutex;

static cublasLtHandle_t g_handle = nullptr;
static cublasLtMatmulDesc_t g_desc = nullptr;
static cublasLtMatrixLayout_t g_layout_a = nullptr;
static cublasLtMatrixLayout_t g_layout_b = nullptr;
static cublasLtMatrixLayout_t g_layout_c = nullptr;
static cublasLtMatmulPreference_t g_pref = nullptr;
static void* g_workspace = nullptr;
static size_t g_ws_size = 0;

struct ShapeKey {
    int64_t m;
    int64_t k;
    int64_t n;
    bool operator==(const ShapeKey& other) const {
        return m == other.m && k == other.k && n == other.n;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& s) const {
        size_t h = std::hash<int64_t>()(s.m);
        h ^= std::hash<int64_t>()(s.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(s.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

using AlgoCache = std::unordered_map<ShapeKey, cublasLtMatmulAlgo_t, ShapeKeyHash>;

static void create_matmul_config(cublasLtMatmulDesc_t* desc,
                                 cublasLtMatrixLayout_t* layout_a,
                                 cublasLtMatrixLayout_t* layout_b,
                                 cublasLtMatrixLayout_t* layout_c) {
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    TORCH_CHECK(cublasLtMatmulDescCreate(desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
                    *desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
                    *desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(layout_a, CUDA_R_8F_E4M3, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(layout_b, CUDA_R_8F_E4M3, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(layout_c, CUDA_R_16BF, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
}

static void ensure_cublas_lt() {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_handle) {
        return;
    }
    TORCH_CHECK(cublasLtCreate(&g_handle) == CUBLAS_STATUS_SUCCESS);
    create_matmul_config(&g_desc, &g_layout_a, &g_layout_b, &g_layout_c);
    TORCH_CHECK(cublasLtMatmulPreferenceCreate(&g_pref) == CUBLAS_STATUS_SUCCESS);
    size_t ws = 16 * 1024 * 1024;
    TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(
                   g_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) ==
               CUBLAS_STATUS_SUCCESS);
}

static cublasStatus_t get_algo_cached(int64_t m, int64_t k, int64_t n,
                                      AlgoCache* cache,
                                      cublasLtMatmulAlgo_t* algo);

static void fp8_gemm_into(torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out,
                          int64_t m, int64_t k, int64_t n,
                          const float* a_scale, const float* b_scale,
                          cudaStream_t stream);

static const float k_scale_one = 1.0f;

static void set_layout(cublasLtMatrixLayout_t layout, int64_t rows, int64_t cols,
                       int64_t ld) {
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ROWS,
                                                 &rows, sizeof(rows)) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_COLS,
                                                 &cols, sizeof(cols)) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_LD,
                                                 &ld, sizeof(ld)) ==
                CUBLAS_STATUS_SUCCESS);
}

torch::Tensor fp8_mm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn, "a must be float8_e4m3fn");
    TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "2D tensors required");
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    int64_t m = a_c.size(0), k = a_c.size(1), n = b_c.size(0);
    TORCH_CHECK(b_c.size(1) == k, "inner dim mismatch");

    auto buf = torch::empty({m, n}, a_c.options().dtype(torch::kBFloat16));
    ensure_cublas_lt();
    fp8_gemm_into(a_c, b_c, buf, m, k, n, &k_scale_one, &k_scale_one,
                  stream.stream());
    return buf;
}


// ---------------------------------------------------------------------------
// Quantize: bf16 * scale_inv -> fp8, one atomicMax amax per kernel call.
// amax_ptr must be zeroed before launch; float-bits atomicMax works because
// |v| >= 0 has a monotonic IEEE bit pattern.
// ---------------------------------------------------------------------------

template <typename T8>
__device__ __forceinline__ T8 cast_fp8(float v);

template <>
__device__ __forceinline__ __nv_fp8_e4m3 cast_fp8<__nv_fp8_e4m3>(float v) {
    return __nv_fp8_e4m3(v);
}

template <>
__device__ __forceinline__ __nv_fp8_e5m2 cast_fp8<__nv_fp8_e5m2>(float v) {
    return __nv_fp8_e5m2(v);
}

template <typename T8>
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ src,
                                const float* __restrict__ scale_inv,
                                T8* __restrict__ dst,
                                float* __restrict__ amax_ptr, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    float amax = 0.f;
    if (i < n) {
        float raw = __bfloat162float(src[i]);
        dst[i] = cast_fp8<T8>(raw * *scale_inv);
        amax = fabsf(raw);
    }
    for (int off = 16; off; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.f;
        for (int w = 0; w < blockDim.x / 32; ++w) m = fmaxf(m, sm[w]);
        atomicMax(reinterpret_cast<unsigned*>(amax_ptr), __float_as_uint(m));
    }
}

// Same but with a transpose (rows x cols bf16 row-major -> fp8 [cols, rows]).
template <typename T8>
__global__ void transpose_quantize_kernel(
    const __nv_bfloat16* __restrict__ src, const float* __restrict__ scale_inv,
    T8* __restrict__ dst, float* __restrict__ amax_ptr, int64_t rows,
    int64_t cols) {
    __shared__ T8 tile[32][33];
    int64_t x = blockIdx.x * 32 + threadIdx.x;
    int64_t y = blockIdx.y * 32 + threadIdx.y;
    float amax = 0.f;
    for (int j = 0; j < 32; j += 8) {
        if (x < cols && y + j < rows) {
            float raw = __bfloat162float(src[(y + j) * cols + x]);
            tile[threadIdx.y + j][threadIdx.x] = cast_fp8<T8>(raw * *scale_inv);
            amax = fmaxf(amax, fabsf(raw));
        }
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int j = 0; j < 32; j += 8) {
        if (x < rows && y + j < cols) {
            dst[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
    for (int off = 16; off; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.f;
        for (int w = 0; w < blockDim.x / 32; ++w) m = fmaxf(m, sm[w]);
        atomicMax(reinterpret_cast<unsigned*>(amax_ptr), __float_as_uint(m));
    }
}

__global__ void bias_add_bf16_kernel(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ bias,
    int64_t total, int64_t n) {
    // GEMM and output use the same row-major [M,N] layout.
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = __bfloat162float(dst[idx]);
    dst[idx] = __float2bfloat16(v + __bfloat162float(bias[idx % n]));
}

static cublasStatus_t get_algo_cached(int64_t m, int64_t k, int64_t n,
                                      AlgoCache* cache,
                                      cublasLtMatmulAlgo_t* algo) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    ShapeKey key{m, k, n};
    auto it = cache->find(key);
    if (it != cache->end()) {
        *algo = it->second;
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        g_handle, g_desc, g_layout_a, g_layout_b, g_layout_c, g_layout_c, g_pref, 1,
        &heur, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0)
        return CUBLAS_STATUS_NOT_SUPPORTED;
    if (heur.workspaceSize > g_ws_size) {
        if (g_workspace) cudaFree(g_workspace);
        TORCH_CHECK(cudaMalloc(&g_workspace, heur.workspaceSize) == cudaSuccess);
        g_ws_size = heur.workspaceSize;
    }
    cache->emplace(key, heur.algo);
    *algo = heur.algo;
    return CUBLAS_STATUS_SUCCESS;
}

static void fp8_gemm_into(torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out,
                          int64_t m, int64_t k, int64_t n,
                          const float* a_scale, const float* b_scale,
                          cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    set_layout(g_layout_a, k, n, k);  // param A = rhs (op=T -> [N,K])
    set_layout(g_layout_b, k, m, k);  // param B = lhs (op=N -> [K,M])
    set_layout(g_layout_c, n, m, n);  // col-major [N,M] == row-major [M,N]
    // Per-tensor FP32 scales applied inside the GEMM:
    // D = alpha * A_SCALE * B_SCALE * A * B (alpha = 1).
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
                    g_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale,
                    sizeof(a_scale)) == CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulDescSetAttribute(
                    g_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale,
                    sizeof(b_scale)) == CUBLAS_STATUS_SUCCESS);
    float alpha = 1.0f, beta = 0.0f;
    static AlgoCache cache;
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t st = get_algo_cached(m, k, n, &cache, &algo);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmulAlgoGetHeuristic failed: ", cublasLtGetStatusName(st));
    st = cublasLtMatmul(g_handle, g_desc, &alpha, rhs.data_ptr(), g_layout_a,
                        lhs.data_ptr(), g_layout_b, &beta, out.data_ptr(), g_layout_c,
                        out.data_ptr(), g_layout_c, &algo, g_workspace, g_ws_size,
                        stream);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmul failed: ", cublasLtGetStatusName(st));
}

// ---------------------------------------------------------------------------
// Scaled FP8 linear forward: quantize x/w with per-tensor scales -> cublasLt
// GEMM (scales applied inside) -> bias in-place -> bf16 [..., N].
// sx/sw: f32 scale tensors (device scalars); sx_inv/sw_inv: 1/scale.
// amax_x/amax_w: f32 buffers receiving max-abs of the quantized tensors.
// ---------------------------------------------------------------------------

torch::Tensor fp8_linear_forward_scaled(torch::Tensor x, torch::Tensor w,
                                        torch::Tensor bias, torch::Tensor sx,
                                        torch::Tensor sw, torch::Tensor sx_inv,
                                        torch::Tensor sw_inv,
                                        torch::Tensor amax_x,
                                        torch::Tensor amax_w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16,
                "x and w must be bf16");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c = x.reshape({-1, w.size(1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = x_c.size(0), k = x_c.size(1), n = w_c.size(0);
    TORCH_CHECK(w_c.size(1) == k, "inner dim mismatch");
    ensure_cublas_lt();

    const float* sx_ptr = sx.data_ptr<float>();
    const float* sw_ptr = sw.data_ptr<float>();
    const float* sxi_ptr = sx_inv.data_ptr<float>();
    const float* swi_ptr = sw_inv.data_ptr<float>();
    float* amax_x_ptr = amax_x.data_ptr<float>();
    float* amax_w_ptr = amax_w.data_ptr<float>();
    C10_CUDA_CHECK(cudaMemsetAsync(amax_x_ptr, 0, sizeof(float), stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(amax_w_ptr, 0, sizeof(float), stream.stream()));

    auto x8 = torch::empty({m, k}, x_c.options().dtype(torch::kFloat8_e4m3fn));
    auto w8 = torch::empty({n, k}, w_c.options().dtype(torch::kFloat8_e4m3fn));
    int64_t block = 256;
    quantize_kernel<__nv_fp8_e4m3>
        <<<(unsigned)((m * k + block - 1) / block), block, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr()), sxi_ptr,
            reinterpret_cast<__nv_fp8_e4m3*>(x8.data_ptr()), amax_x_ptr, m * k);
    quantize_kernel<__nv_fp8_e4m3>
        <<<(unsigned)((n * k + block - 1) / block), block, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()), swi_ptr,
            reinterpret_cast<__nv_fp8_e4m3*>(w8.data_ptr()), amax_w_ptr, n * k);
    C10_CUDA_CHECK(cudaGetLastError());

    auto out = torch::empty({m, n}, x_c.options());
    fp8_gemm_into(x8, w8, out, m, k, n, sw_ptr, sx_ptr, stream.stream());

    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.scalar_type() == torch::kBFloat16 && bias.numel() == n,
                    "bias must be bf16 with shape [N]");
        bias_add_bf16_kernel<<<(unsigned)((m * n + block - 1) / block), block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()), m * n, n);
        C10_CUDA_CHECK(cudaGetLastError());
    }

    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 1);
    shape.push_back(n);
    return out.reshape(shape);
}

// ---------------------------------------------------------------------------
// Scaled FP8 linear backward: dX = g @ W, dW = g^T @ X, dB = sum(g).
// Scales: g uses sg (immediate), w/x reuse the forward scales.
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fp8_linear_backward_scaled(
    torch::Tensor g, torch::Tensor x, torch::Tensor w,
    std::vector<int64_t> masks, torch::Tensor sg, torch::Tensor sw,
    torch::Tensor sx, torch::Tensor sg_inv, torch::Tensor sw_inv,
    torch::Tensor sx_inv, torch::Tensor amax_g) {
    const at::cuda::OptionalCUDAGuard guard(g.device());
    TORCH_CHECK(g.dtype() == torch::kBFloat16 && x.dtype() == torch::kBFloat16 &&
                    w.dtype() == torch::kBFloat16,
                "g, x, and w must be bf16");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto g_c = g.reshape({-1, w.size(0)}).contiguous();
    auto x_c = x.reshape({-1, x.size(-1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = g_c.size(0);
    int64_t n = w.size(0);
    int64_t k = w.size(1);
    TORCH_CHECK(x_c.size(0) == m && x_c.size(1) == k && g_c.size(1) == n,
                "backward shape mismatch");

    auto grad_input = torch::empty_like(x);
    auto grad_weight = torch::empty_like(w);
    auto grad_bias = torch::empty({0}, g_c.options().dtype(g.dtype()));
    ensure_cublas_lt();

    const float* sg_ptr = sg.data_ptr<float>();
    const float* sw_ptr = sw.data_ptr<float>();
    const float* sx_ptr = sx.data_ptr<float>();
    const float* sgi_ptr = sg_inv.data_ptr<float>();
    const float* swi_ptr = sw_inv.data_ptr<float>();
    const float* sxi_ptr = sx_inv.data_ptr<float>();
    float* amax_g_ptr = amax_g.data_ptr<float>();
    C10_CUDA_CHECK(cudaMemsetAsync(amax_g_ptr, 0, sizeof(float), stream.stream()));

    auto fp8_options = g_c.options().dtype(torch::kFloat8_e4m3fn);
    auto g8 = torch::empty({m, n}, fp8_options);
    auto gt8 = masks[1] ? torch::empty({n, m}, fp8_options) : torch::Tensor();
    auto wt8 = masks[0] ? torch::empty({k, n}, fp8_options) : torch::Tensor();
    auto xt8 = masks[1] ? torch::empty({k, m}, fp8_options) : torch::Tensor();
    // w/x transpose-quantize amax goes to a scratch buffer, NOT amax_g: the
    // gradient scale must only see the gradient's own max-abs.
    auto amax_t = torch::zeros({1}, g_c.options().dtype(torch::kFloat32));

    int64_t block = 256;
    quantize_kernel<__nv_fp8_e4m3>
        <<<(unsigned)((m * n + block - 1) / block), block, 0, stream.stream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(g_c.data_ptr()), sgi_ptr,
            reinterpret_cast<__nv_fp8_e4m3*>(g8.data_ptr()), amax_g_ptr, m * n);
    dim3 threads(32, 8);
    if (masks[0]) {
        dim3 blocks((k + 31) / 32, (n + 31) / 32);
        transpose_quantize_kernel<__nv_fp8_e4m3>
            <<<blocks, threads, 0, stream.stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()), swi_ptr,
                reinterpret_cast<__nv_fp8_e4m3*>(wt8.data_ptr()),
                amax_t.data_ptr<float>(), n, k);
        fp8_gemm_into(g8, wt8, grad_input.reshape({m, k}), m, n, k, sg_ptr,
                      sw_ptr, stream.stream());
    }
    if (masks[1]) {
        dim3 g_blocks((n + 31) / 32, (m + 31) / 32);
        dim3 x_blocks((k + 31) / 32, (m + 31) / 32);
        transpose_quantize_kernel<__nv_fp8_e4m3>
            <<<g_blocks, threads, 0, stream.stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(g_c.data_ptr()), sgi_ptr,
                reinterpret_cast<__nv_fp8_e4m3*>(gt8.data_ptr()),
                amax_t.data_ptr<float>(), m, n);
        transpose_quantize_kernel<__nv_fp8_e4m3>
            <<<x_blocks, threads, 0, stream.stream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr()), sxi_ptr,
                reinterpret_cast<__nv_fp8_e4m3*>(xt8.data_ptr()),
                amax_t.data_ptr<float>(), m, k);
        fp8_gemm_into(gt8, xt8, grad_weight, n, m, k, sg_ptr, sx_ptr,
                      stream.stream());
    }
    C10_CUDA_CHECK(cudaGetLastError());
    if (masks[2]) {
        grad_bias = g_c.sum(0).to(g.dtype());
    }
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
        grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_mm", &fp8_mm, py::arg("a"), py::arg("b"),
          "FP8 e4m3 GEMM: a[M,K] x b[N,K] -> bf16[M,N] (pre-scaled inputs)");
    m.def("fp8_linear_forward_scaled", &fp8_linear_forward_scaled,
          py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("sx"),
          py::arg("sw"), py::arg("sx_inv"), py::arg("sw_inv"),
          py::arg("amax_x"), py::arg("amax_w"),
          "Scaled FP8 linear forward: quantize with per-tensor scales + "
          "cublasLt GEMM (scales applied inside) + bias -> bf16");
    m.def("fp8_linear_backward_scaled", &fp8_linear_backward_scaled,
          py::arg("g"), py::arg("x"), py::arg("w"), py::arg("masks"),
          py::arg("sg"), py::arg("sw"), py::arg("sx"), py::arg("sg_inv"),
          py::arg("sw_inv"), py::arg("sx_inv"), py::arg("amax_g"),
          "Scaled FP8 linear backward: dX = g*sw @ W, dW = (g*sx)^T @ X, "
          "dB = sum(g)");
}
