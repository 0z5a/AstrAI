#pragma once
// C++ entry into the quantized GEMM family, for composed callers *inside this
// module* (the fp8 linear op compiles into the gemm module so the dispatch
// state — plan table, planner mode, staging switches — stays single-source).
// Deliberately template-free: including this header must not instantiate the
// dtype-pair kernels, which are explicitly instantiated in their own TUs.

#include <c10/util/Optional.h>
#include <torch/extension.h>

namespace astrai {
namespace gemm {

// ``quant_gemm`` with C++-typed operands. Semantics, validation and the
// dispatch are documented at the definition in gemm.cu; the pybind
// ``quant_gemm`` is a thin None-tolerant wrapper over this.
torch::Tensor quant_gemm_impl(torch::Tensor a, torch::Tensor b,
                              c10::optional<torch::Tensor> a_scale,
                              c10::optional<torch::Tensor> b_scale,
                              bool trans_a, bool trans_b,
                              c10::optional<torch::Tensor> bias);

}  // namespace gemm
}  // namespace astrai
