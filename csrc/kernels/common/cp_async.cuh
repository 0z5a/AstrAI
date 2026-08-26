// Shared cp.async primitives — pure CUDA, no torch.
//
// One header for the async-copy pipeline used by both the attention kernels
// (predicated 16-byte K/V tile staging) and the fp8 GEMM (predicated operand
// staging + the fixed-depth wait_group). The emitter is split from its
// policies: cp_async_16_raw owns the single PTX site, and each wrapper states
// one destination contract (generic pointer vs loop-carried shared offset)
// and one predication contract (unconditional vs zero-fill-when-false), so
// call sites never pass a dead `true` predicate or re-convert a carried
// offset. PTX requires wait_group's operand to be an immediate, hence the
// template form below.

#pragma once

#include <cuda_runtime.h>

namespace astrai {

// Raw emitter: read src_size bytes (<= 16) from gmem into the shared
// offset. src_size = 0 reads nothing, so a predicated-off call zero-fills
// its destination without touching the (possibly out-of-range) source.
// BypassL1 selects .cg (L2 only, default) vs .ca (L1 + L2).
template <bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16_raw(unsigned smem_addr,
                                                const void* gmem_ptr,
                                                int src_size) {
    if constexpr (BypassL1) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                     :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                     :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_size));
    }
}

// Unconditional 16-byte copy to a generic shared pointer.
// `T` is the smem element type; only the destination pointer's type matters.
template <typename T, bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(T* smem_ptr,
                                            const void* gmem_ptr) {
    cp_async_16_raw<BypassL1>(__cvta_generic_to_shared(smem_ptr), gmem_ptr,
                              16);
}

// Predicated: full copy when `pred`, zero-fill otherwise.
template <typename T, bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(T* smem_ptr, const void* gmem_ptr,
                                            bool pred) {
    cp_async_16_raw<BypassL1>(__cvta_generic_to_shared(smem_ptr), gmem_ptr,
                              pred ? 16 : 0);
}

// Predicated raw-offset form: the destination is an already-converted
// shared-memory offset (e.g. a loop-carried swizzled stage address), so
// steady-state prefetch sites issue one LDGSTS straight from the register.
template <bool BypassL1 = true>
__device__ __forceinline__ void cp_async_16(unsigned smem_addr,
                                            const void* gmem_ptr, bool pred) {
    cp_async_16_raw<BypassL1>(smem_addr, gmem_ptr, pred ? 16 : 0);
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

}  // namespace astrai
