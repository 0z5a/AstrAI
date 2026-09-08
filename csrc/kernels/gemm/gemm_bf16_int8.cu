// W8A16 instantiation unit: bf16 x int8 (b_scale required). One
// gemm_dispatch specialization per unit — see gemm_cases.h.
#include "gemm_cases.h"

namespace astrai {
namespace gemm {

void gemm_bf16_int8(GemmParams p, cudaStream_t s, bool ta, bool tb) {
    gemm_dispatch<__nv_bfloat16, int8_t>(p, s, ta, tb);
}

}  // namespace gemm
}  // namespace astrai
