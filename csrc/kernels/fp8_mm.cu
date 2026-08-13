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
    TORCH_CHECK(cublasLtMatrixLayoutCreate(&g_layout_c, CUDA_R_32F, 1, 1, 1) ==
                CUBLAS_STATUS_SUCCESS);
    TORCH_CHECK(cublasLtMatmulPreferenceCreate(&g_pref) == CUBLAS_STATUS_SUCCESS);
    size_t ws = 16 * 1024 * 1024;
    TORCH_CHECK(cublasLtMatmulPreferenceSetAttribute(
                   g_pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) ==
               CUBLAS_STATUS_SUCCESS);
}

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

    auto buf = torch::empty({n, m}, a_c.options().dtype(torch::kFloat32));

    ensure_cublas_lt();
    set_layout(g_layout_a, k, m, k);  // A col-major [K,M] (a row-major, op=T)
    set_layout(g_layout_b, k, n, k);  // B col-major [K,N] (wT row-major, op=N)
    set_layout(g_layout_c, m, n, m);  // C col-major [M,N]

    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        g_handle, g_desc, g_layout_a, g_layout_b, g_layout_c, g_layout_c, g_pref, 1,
        &heur, &returned);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmulAlgoGetHeuristic failed: ", cublasLtGetStatusName(st));
    if (heur.workspaceSize > g_ws_size) {
        if (g_workspace) {
            cudaFree(g_workspace);
        }
        TORCH_CHECK(cudaMalloc(&g_workspace, heur.workspaceSize) == cudaSuccess);
        g_ws_size = heur.workspaceSize;
    }
    st = cublasLtMatmul(
        g_handle, g_desc, &alpha, a_c.data_ptr(), g_layout_a, b_c.data_ptr(),
        g_layout_b, &beta, buf.data_ptr(), g_layout_c, buf.data_ptr(), g_layout_c,
        &heur.algo, g_workspace, g_ws_size, stream.stream());
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmul failed: ", cublasLtGetStatusName(st));
    return buf.transpose(0, 1).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_mm", &fp8_mm, py::arg("a"), py::arg("b"),
          "FP8 e4m3 GEMM: a[M,K] x b[N,K] -> fp32[M,N] (pre-scaled inputs)");
}
