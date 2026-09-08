// W8A8 instantiation unit: int8 x int8 (both scales; native s8 mma, s32
// accumulators). One gemm_dispatch specialization per unit — see
// gemm_cases.h.
#include "gemm_cases.h"

namespace astrai {
namespace gemm {

void gemm_int8_int8(GemmParams p, cudaStream_t s, bool ta, bool tb) {
    gemm_dispatch<int8_t, int8_t>(p, s, ta, tb);
}

}  // namespace gemm
}  // namespace astrai
