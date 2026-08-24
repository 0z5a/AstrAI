// Shared cp.async primitives — pure CUDA, no torch.
//
// One header for the async-copy pipeline used by both the attention kernels
// (predicated 16-byte K/V tile staging) and the fp8 GEMM (predicated operand
// staging + wait_group dispatch). PTX requires wait_group's operand to be an
// immediate, hence the template forms.

#pragma once

#include <cuda_runtime.h>

namespace astrai {

// Predicated cp.async: copy 16 bytes when `pred`, otherwise zero-fill.
// src_size=0 means no bytes are read, so an out-of-bounds address is safe.
// BypassL1 defaults to .cg (L2 only); false selects .ca (L1 + L2).
// `T` is the smem element type; only the destination pointer's type matters.
template <typename T, bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(T* smem_ptr, const void* gmem_ptr,
                                            bool pred) {
    const unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    const int src_size = pred ? 16 : 0;
    if constexpr (BypassL1) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                     :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
    }
}

// Commit all outstanding cp.async ops of this thread as one group.
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

// Wait for every committed group (pipeline drain).
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// Wait until at most KeepGroups committed groups are still in flight.
// PTX requires an immediate operand; keep it as a template argument so the
// stage policy stays compile-time configurable.
template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7,
                  "cp.async.wait_group supports immediates in [0, 7]");
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
}

// Runtime dispatch over cp_async_wait_group<N>: unrolls into a compare
// ladder over [0, MaxKeepGroups] so the immediate-only PTX constraint is
// hidden behind a runtime `keep_groups` (used by the fp8 GEMM pipeline,
// whose remaining-tile count is dynamic).
template <int MaxKeepGroups>
__device__ __forceinline__ void cp_async_wait_group_dispatch(int keep_groups) {
    static_assert(MaxKeepGroups >= 0 && MaxKeepGroups <= 7,
                  "cp.async.wait_group supports immediates in [0, 7]");
    if (keep_groups == MaxKeepGroups) {
        cp_async_wait_group<MaxKeepGroups>();
    } else if constexpr (MaxKeepGroups > 0) {
        cp_async_wait_group_dispatch<MaxKeepGroups - 1>(keep_groups);
    } else {
        cp_async_wait_group<0>();
    }
}

}  // namespace astrai
