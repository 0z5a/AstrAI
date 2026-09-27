#pragma once
// Quantize's declaration surface: the composed one-call pass shared by the
// standalone quantize bindings (quantize/bindings.cu) and the fp8-linear
// composition (gemm/fp8_linear.cu, compiled into the gemm module so the
// GEMM dispatch state stays single-source). The implementation is
// quantize/entry.cu, listed in both modules' CMake source lists — a caller
// that needs the quantize chain from C++ links the body via its own module
// instead of re-deriving it. Torch-tensor level, no pybind (the api.h
// rules): declarations only, struct definitions included.

#include <c10/util/Optional.h>
#include <cstdint>
#include <torch/extension.h>

#include <utils/quantize_common.h>

namespace astrai {
namespace quant {

struct QuantizeOutputs {
    torch::Tensor out;   // row-major orientation (undefined if not asked)
    torch::Tensor out_t; // [cols][rows] transpose (undefined if not asked)
    torch::Tensor amax;  // the fold's raw-domain amax of the round (undefined
                         // without a ring — nothing measures one)
};

// One quantize pass, end to end: validation, output allocation, launch.
// ``layout`` picks which orientations are produced; ``transposed_dtype``
// (default: the row-major dtype) casts the transposed orientation in a
// different fp8 format — the hybrid training pair casts the forward format
// on one side and the backward format on the other from a single read. A
// ring switches on the in-kernel delayed-scaling fold; without one the
// kernel runs a pure scale+cast. ``pub_scale``/``pub_recip`` redirect where
// the fold publishes (default: the ring's own slots). A ring also requires
// ``hist_len`` (see RingLayout in utils/quantize_common.h); one without is
// rejected rather than guessed. Semantics are documented at the definition
// (quantize/entry.cu).
QuantizeOutputs run_quantize(torch::Tensor x,
                             torch::Tensor scale,
                             QuantLayout layout,
                             at::ScalarType dtype_a,
                             c10::optional<at::ScalarType> dtype_b,
                             c10::optional<torch::Tensor> ring,
                             int64_t hist_idx,
                             double fp8_max,
                             double pow2_margin,
                             c10::optional<torch::Tensor> pub_scale = c10::nullopt,
                             c10::optional<torch::Tensor> pub_recip = c10::nullopt,
                             c10::optional<int64_t> hist_len = c10::nullopt);

} // namespace quant
} // namespace astrai
