#include <torch/extension.h>
#include <cuda_bf16.h>

__global__ void rotary_emb_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int batch,
    int seq_len,
    int n_heads,
    int head_dim
) {
    const int half_dim = head_dim >> 1;
    const int total = batch * seq_len * n_heads * half_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {

        int pair = idx % half_dim;
        int tmp = idx / half_dim;
        int head = tmp % n_heads;
        tmp /= n_heads;
        int seq = tmp % seq_len;
        int b = tmp / seq_len;

        int x_offset = ((b * seq_len + seq) * n_heads + head) * head_dim + (pair << 1);
        int cs_offset = (b * seq_len + seq) * half_dim + pair;

        __nv_bfloat162 x_pair = *reinterpret_cast<const __nv_bfloat162*>(x + x_offset);
        float x_even = __bfloat162float(__low2bfloat16(x_pair));
        float x_odd  = __bfloat162float(__high2bfloat16(x_pair));

        float c = cos[cs_offset];
        float s = sin[cs_offset];

        float out_even = x_even * c - x_odd * s;
        float out_odd  = x_even * s + x_odd * c;

        __nv_bfloat162 out_pair = __floats2bfloat162_rn(out_even, out_odd);
        *reinterpret_cast<__nv_bfloat162*>(out + x_offset) = out_pair;
    }
}

torch::Tensor rotary_emb(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin
) {

    int batch = x.size(0);
    int seq_len = x.size(1);
    int n_heads = x.size(2);
    int head_dim = x.size(3);

    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(cos.is_cuda(), "cos must be on CUDA");
    TORCH_CHECK(sin.is_cuda(), "sin must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, seq_len, n_heads, head_dim]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(cos.dim() == 3, "cos must be 3D [batch, seq_len, head_dim/2]");
    TORCH_CHECK(sin.dim() == 3, "sin must be 3D [batch, seq_len, head_dim/2]");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");

    auto out = torch::empty_like(x);

    int half_dim = head_dim / 2;
    int total = batch * seq_len * n_heads * half_dim;
    int block = 256;
    int grid = std::min((total + block - 1) / block, 1024);

    rotary_emb_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        batch, seq_len, n_heads, head_dim
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_emb", &rotary_emb,
        py::arg("x"),
        py::arg("cos"),
        py::arg("sin"),
        "Fused rotary embedding (bf16 x, f32 cos/sin, bf16 out)"
    );
}
