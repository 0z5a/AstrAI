// Cross-TU instantiation points of the quant-GEMM dispatch: one unit per
// supported dtype pair (one gemm_dispatch specialization per unit, named
// cuBLAS-style after the operand pair — gemm_bf16_bf16, gemm_int8_int8,
// ...). The heavy template work parallelizes across nvcc jobs instead of
// serializing inside the single binding TU; the switch in gemm.cu calls
// through these externs and pays for jump-table pointers only. The C tests
// instantiate straight from the headers instead.
#pragma once

#include "gemm.cuh"

namespace astrai {
namespace gemm {

void gemm_bf16_bf16(GemmParams, cudaStream_t, bool, bool);            // bf16 x bf16
void gemm_bf16_int8(GemmParams, cudaStream_t, bool, bool);            // bf16 x int8
void gemm_bf16_fp8_e4m3(GemmParams, cudaStream_t, bool, bool);        // bf16 x fp8 e4m3
void gemm_bf16_fp8_e5m2(GemmParams, cudaStream_t, bool, bool);        // bf16 x fp8 e5m2
void gemm_int8_int8(GemmParams, cudaStream_t, bool, bool);            // int8 x int8
void gemm_fp8_e4m3_fp8_e4m3(GemmParams, cudaStream_t, bool, bool);    // fp8 x fp8 e4m3
void gemm_fp8_e5m2_fp8_e5m2(GemmParams, cudaStream_t, bool, bool);    // fp8 x fp8 e5m2

}  // namespace gemm
}  // namespace astrai
