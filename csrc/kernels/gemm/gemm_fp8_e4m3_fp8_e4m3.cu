// fp8 x fp8 e4m3 instantiation unit. One gemm_dispatch specialization per
// unit — see gemm_cases.h. (The C tests instantiate gemm_dispatch from the
// headers directly, so no entry beyond the pair specialization is needed.)
#include "gemm_cases.h"

namespace astrai {
namespace gemm {

void gemm_fp8_e4m3_fp8_e4m3(GemmParams p, cudaStream_t s, bool ta, bool tb) {
    gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3>(p, s, ta, tb);
}

}  // namespace gemm
}  // namespace astrai
