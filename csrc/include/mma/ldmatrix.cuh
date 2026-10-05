#pragma once
/* ldmatrix loads: lane i supplies row i % 8 of matrix i / 8.
 * Callers compute per-lane addresses for their shared-memory layout. */
#include <cuda_runtime.h>
#include <utils/define.cuh>
#include <utils/tensor.cuh>

namespace astrai {

template <bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x2_lane(unsigned r[2], unsigned addr) {
    if constexpr (Trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(addr));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(addr));
    }
}

template <bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x4_lane(unsigned r[4], unsigned addr) {
    if constexpr (Trans) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(addr));
    }
}

/* Typed fragment adapters. */
template <bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x2_lane(ArrayEngine<unsigned, 2>& f, unsigned addr) {
    ldmatrix_x2_lane<Trans>(f.storage, addr);
}

template <bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x4_lane(ArrayEngine<unsigned, 4>& f, unsigned addr) {
    ldmatrix_x4_lane<Trans>(f.storage, addr);
}

/* Shared-pointer adapter for attention. */
template <typename T, bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x2(unsigned r[2], const T* p) {
    ldmatrix_x2_lane<Trans>(r, __cvta_generic_to_shared(p));
}

} // namespace astrai
