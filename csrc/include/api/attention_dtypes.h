#pragma once
// Attention's instantiation list — the one place that names which precisions
// have kernels, in gemm's pair-table dialect: X(torch ScalarType, element
// type). The entries switch on q.scalar_type() over this list and the refusal
// below is generated from the same rows, so the supported set cannot drift.
// Adding a precision = a row here + its ElemTrait (utils/dtype.cuh) and, for
// tensor-core dtypes, the MmaShapeFor/MmaOp cell (mma/mma.cuh).

#include <string>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <utils/dtype.cuh>

namespace astrai {
namespace attention {

// Rows expand at the entries' file scope, so element types are spelled
// namespace-qualified.
#define ASTRAI_ATTN_DTYPE_LIST(X) X(at::kBFloat16, astrai::bf16)

// A scalar type attention has no kernel for: say which ones it does have.
inline void attn_dtype_unsupported(at::ScalarType st) {
    std::string instantiated;
#define ASTRAI_ATTN_DTYPE_NAME_ROW(tag, type)                                                      \
    instantiated += std::string(instantiated.empty() ? "" : ", ") + c10::toString(tag);
    ASTRAI_ATTN_DTYPE_LIST(ASTRAI_ATTN_DTYPE_NAME_ROW)
#undef ASTRAI_ATTN_DTYPE_NAME_ROW
    TORCH_CHECK(false, "attention has no kernel for ", c10::toString(st),
                " (instantiated: ", instantiated, ")");
}

} // namespace attention
} // namespace astrai
