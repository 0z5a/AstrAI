#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

__global__ void rotary_emb_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ freqs_cis,
    __nv_bfloat16* __restrict__ out,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int half_dim = head_dim >> 1;
    const int total = n_tokens * n_heads * half_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {

        int pair = idx % half_dim;
        int tmp = idx / half_dim;
        int head = tmp % n_heads;
        tmp /= n_heads;
        int token = tmp;

        int x_offset = (token * n_heads + head) * head_dim + (pair << 1);
        int cs_offset = (token * half_dim + pair) * 2;

        __nv_bfloat162 x_pair = *reinterpret_cast<const __nv_bfloat162*>(x + x_offset);
        float x_even = __bfloat162float(__low2bfloat16(x_pair));
        float x_odd  = __bfloat162float(__high2bfloat16(x_pair));

        float c = freqs_cis[cs_offset];
        float s = freqs_cis[cs_offset + 1];

        float out_even = x_even * c - x_odd * s;
        float out_odd  = x_even * s + x_odd * c;

        __nv_bfloat162 out_pair = __floats2bfloat162_rn(out_even, out_odd);
        *reinterpret_cast<__nv_bfloat162*>(out + x_offset) = out_pair;
    }
}

torch::Tensor rotary_emb(
    torch::Tensor x,
    torch::Tensor freqs_cis
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(freqs_cis.is_cuda(), "freqs_cis must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
                "x must be [tokens, n_heads, head_dim] or "
                "[batch, seq_len, n_heads, head_dim]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(freqs_cis.dim() == x.dim(), "freqs_cis rank must match x rank");
    TORCH_CHECK(freqs_cis.is_contiguous(), "freqs_cis must be contiguous");
    TORCH_CHECK(freqs_cis.scalar_type() == torch::kFloat32, "freqs_cis must be f32");

    int n_tokens = x.dim() == 3 ? x.size(0) : x.size(0) * x.size(1);
    int n_heads = x.size(x.dim() - 2);
    int head_dim = x.size(x.dim() - 1);

    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
    TORCH_CHECK(freqs_cis.numel() == (int64_t)n_tokens * head_dim,
                "freqs_cis token or rotary dimension mismatch");
    TORCH_CHECK(freqs_cis.size(-2) == head_dim / 2, "freqs_cis dim/2 mismatch");
    TORCH_CHECK(freqs_cis.size(-1) == 2, "freqs_cis last dim must be 2 [cos, sin]");

    auto out = torch::empty_like(x);

    int half_dim = head_dim / 2;
    int total = n_tokens * n_heads * half_dim;
    int block = 256;
    int grid = std::min((total + block - 1) / block, 1024);

    rotary_emb_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        freqs_cis.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        n_tokens, n_heads, head_dim
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_emb", &rotary_emb,
        py::arg("x"),
        py::arg("freqs_cis"),
        "Fused rotary embedding for packed 3D or dense 4D tensors"
    );
}
