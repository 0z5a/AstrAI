// fp8 x fp8 e4m3 instantiation unit. One gemm_dispatch specialization per
// unit — see gemm_cases.h. (The FP8Format entry gemm<FP8Format::E4M3> in
// the headers has no caller in the production module — the C tests
// instantiate it from the headers directly — so it is not pinned here.)
#include "gemm_cases.h"

namespace astrai {
namespace gemm {

void gemm_fp8_e4m3_fp8_e4m3(GemmParams p, cudaStream_t s, bool ta, bool tb) {
    gemm_dispatch<__nv_fp8_e4m3, __nv_fp8_e4m3>(p, s, ta, tb);
}

}  // namespace gemm
}  // namespace astrai
