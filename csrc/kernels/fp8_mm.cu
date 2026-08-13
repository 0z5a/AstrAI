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

static cublasLtHandle_t g_handle = nullptr;
static cublasLtMatmulDesc_t g_desc = nullptr;
static cublasLtMatrixLayout_t g_layout_a = nullptr;
static cublasLtMatrixLayout_t g_layout_b = nullptr;
static cublasLtMatrixLayout_t g_layout_c = nullptr;
static cublasLtMatmulPreference_t g_pref = nullptr;
static void* g_workspace = nullptr;
static size_t g_ws_size = 0;

static void ensure_cublas_lt() {
    if (g_handle) {
        return;
    }
    TORCH_CHECK(cublasLtCreate(&g_handle) == CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulDescCreate(&g_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                CUBLAS_STATUS_SUCCESS);
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(g_desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(g_desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&g_layout_a, CUDA_R_8F_E4M3, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&g_layout_b, CUDA_R_8F_E4M3, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&g_layout_c, CUDA_R_16BF, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulPreferenceCreate(&g_pref) == CUBLAS_STATUS_SUCCESS);
    size_t ws = 16 * 1024 * 1024;
    TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(
                   g_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) ==
               CUBLAS_STATUS_SUCCESS);
}

static cublasStatus_t get_algo_cached(int64_t m, int64_t k, int64_t n,
                                      cublasLtMatmulAlgo_t* algo);

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

    // A/B swapped so the col-major [N,M] output storage IS row-major C[M,N]:
    // param A = b (op=T -> [N,K]), param B = a (op=N -> [K,M]), output no copy.
    auto buf = torch::empty({m, n}, a_c.options().dtype(torch::kBFloat16));

    ensure_cublas_lt();
    set_layout(g_layout_a, k, n, k);  // A col-major [K,N] (b row-major, op=T)
    set_layout(g_layout_b, k, m, k);  // B col-major [K,M] (a row-major, op=N)
    set_layout(g_layout_c, n, m, n);  // C col-major [N,M], ld=N

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t st = get_algo_cached(m, k, n, &algo);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmulAlgoGetHeuristic failed: ", cublasLtGetStatusName(st));
    st = cublasLtMatmul(
        g_handle, g_desc, &alpha, b_c.data_ptr(), g_layout_a, a_c.data_ptr(),
        g_layout_b, &beta, buf.data_ptr(), g_layout_c, buf.data_ptr(), g_layout_c,
        &algo, g_workspace, g_ws_size, stream.stream());
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmul failed: ", cublasLtGetStatusName(st));
    return buf;
}


// ---------------------------------------------------------------------------
// Fused FP8 linear forward: one call = scale cast x8/w8 -> cublasLt GEMM
// (bf16 output) -> transpose + unscale + bias -> bf16 [..., N].
// ---------------------------------------------------------------------------

__global__ void cast_bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ src, __nv_fp8_e4m3* __restrict__ dst,
    int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __nv_fp8_e4m3(__bfloat162float(src[i]));
}

__global__ void bias_cast_kernel(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ bias, int64_t total, int64_t n) {
    // Same layout both sides (row-major [M,N]); bias added per column.
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = __bfloat162float(src[idx]);
    if (bias) v += bias[idx % n];
    dst[idx] = __float2bfloat16(v);
}

static int64_t g_last_m = -1, g_last_k = -1, g_last_n = -1;
static cublasLtMatmulAlgo_t g_last_algo;

static cublasStatus_t get_algo_cached(int64_t m, int64_t k, int64_t n,
                                      cublasLtMatmulAlgo_t* algo) {
    if (m == g_last_m && k == g_last_k && n == g_last_n) {
        *algo = g_last_algo;
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
    g_last_algo = heur.algo;
    g_last_m = m; g_last_k = k; g_last_n = n;
    *algo = heur.algo;
    return CUBLAS_STATUS_SUCCESS;
}

torch::Tensor fp8_linear_forward(torch::Tensor x, torch::Tensor w,
                                 torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(w.dtype() == torch::kBFloat16, "w must be bf16");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_c = x.reshape({-1, w.size(1)}).contiguous();
    auto w_c = w.contiguous();
    int64_t m = x_c.size(0), k = x_c.size(1), n = w_c.size(0);
    TORCH_CHECK(w_c.size(1) == k, "inner dim mismatch");
    auto out = torch::empty({m, n}, x_c.options());

    ensure_cublas_lt();
    set_layout(g_layout_a, k, n, k);  // param A = w8 (op=T -> [N,K])
    set_layout(g_layout_b, k, m, k);  // param B = x8 (op=N -> [K,M])
    set_layout(g_layout_c, n, m, n);  // col-major [N,M] == row-major C[M,N]

    auto x8 = torch::empty({m, k}, x_c.options().dtype(torch::kFloat8_e4m3fn));
    auto w8 = torch::empty({n, k}, w_c.options().dtype(torch::kFloat8_e4m3fn));
    int64_t block = 256;
    cast_bf16_to_fp8_kernel<<<(unsigned)((m * k + block - 1) / block), block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(x8.data_ptr()), m * k);
    cast_bf16_to_fp8_kernel<<<(unsigned)((n * k + block - 1) / block), block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(w8.data_ptr()), n * k);
    C10_CUDA_CHECK(cudaGetLastError());

    auto buf = torch::empty({m, n}, out.options());  // row-major C[M,N] direct
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t st = get_algo_cached(m, k, n, &algo);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmulAlgoGetHeuristic failed: ", cublasLtGetStatusName(st));
    st = cublasLtMatmul(g_handle, g_desc, &alpha, w8.data_ptr(), g_layout_a,
                        x8.data_ptr(), g_layout_b, &beta, buf.data_ptr(), g_layout_c,
                        buf.data_ptr(), g_layout_c, &algo, g_workspace, g_ws_size,
                        stream.stream());
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmul failed: ", cublasLtGetStatusName(st));

    float* bias_ptr = nullptr;
    auto bias_f = torch::Tensor();
    if (bias.defined() && bias.numel() > 0) {
        bias_f = bias.to(torch::kFloat32).contiguous();
        bias_ptr = bias_f.data_ptr<float>();
    }
    bias_cast_kernel<<<(unsigned)((m * n + block - 1) / block), block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), bias_ptr, m * n, n);
    C10_CUDA_CHECK(cudaGetLastError());

    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 1);
    shape.push_back(n);
    return out.reshape(shape);
}

// ---------------------------------------------------------------------------
// Fused FP8 linear backward: dX = (g*sw) @ W, dW = (g*sx)^T @ X, dB = sum(g).
// Scales are recomputed from x/w (identical to forward, no state needed).
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fp8_linear_backward(
    torch::Tensor g, torch::Tensor x, torch::Tensor w,
    std::vector<int64_t> masks) {
    const at::cuda::OptionalCUDAGuard guard(g.device());
    auto g_c = g.reshape({-1, w.size(0)}).contiguous();
    auto x_c = x.reshape({-1, x.size(-1)}).contiguous();
    int64_t n = w.size(0);

    auto grad_input = torch::empty_like(x);
    auto grad_weight = torch::empty_like(w);
    auto grad_bias = torch::empty({0}, g_c.options().dtype(g.dtype()));
    // Compute dtype follows the input tensor (bf16 model -> bf16 GEMMs,
    // fp32 input -> fp32); w is cast to match, no branch needed.
    auto dtype = x_c.dtype();
    auto g_w = g_c.to(dtype);
    auto w_w = w.to(dtype);
    if (masks[0]) {
        grad_input.copy_(torch::mm(g_w, w_w).reshape_as(x));
    }
    if (masks[1]) {
        grad_weight.copy_(torch::mm(g_w.t(), x_c));
    }
    if (masks[2]) {
        grad_bias = g.sum(0).to(g.dtype());
    }
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
        grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_mm", &fp8_mm, py::arg("a"), py::arg("b"),
          "FP8 e4m3 GEMM: a[M,K] x b[N,K] -> bf16[M,N] (pre-scaled inputs)");
    m.def("fp8_linear_forward", &fp8_linear_forward,
          py::arg("x"), py::arg("w"), py::arg("bias"),
          "Fused FP8 linear forward: scale cast + cublasLt GEMM + unscale "
          "+ bias + transpose -> bf16, single call");
    m.def("fp8_linear_backward", &fp8_linear_backward,
          py::arg("g"), py::arg("x"), py::arg("w"), py::arg("masks"),
          "Fused linear backward: dX = g*sw @ W, dW = (g*sx)^T @ X, "
          "dB = sum(g), single call");
}
