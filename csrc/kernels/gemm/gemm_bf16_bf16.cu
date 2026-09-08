// W16A16 instantiation unit: bf16 x bf16 (no scales). One gemm_dispatch
// specialization per unit — see gemm_cases.h.
#include "gemm_cases.h"

namespace astrai {
namespace gemm {

void gemm_bf16_bf16(GemmParams p, cudaStream_t s, bool ta, bool tb) {
    gemm_dispatch<__nv_bfloat16, __nv_bfloat16>(p, s, ta, tb);
}

}  // namespace gemm
}  // namespace astrai
