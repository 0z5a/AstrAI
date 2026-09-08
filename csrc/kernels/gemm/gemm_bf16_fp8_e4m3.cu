// W-F8A16 instantiation unit, e4m3 weights: bf16 x fp8 e4m3 (in-register
// hardware widen). One gemm_dispatch specialization per unit — see
// gemm_cases.h.
#include "gemm_cases.h"

namespace astrai {
namespace gemm {

void gemm_bf16_fp8_e4m3(GemmParams p, cudaStream_t s, bool ta, bool tb) {
    gemm_dispatch<__nv_bfloat16, __nv_fp8_e4m3>(p, s, ta, tb);
}

}  // namespace gemm
}  // namespace astrai
