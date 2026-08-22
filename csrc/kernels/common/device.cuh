// Pure-CUDA device helpers shared across kernel families (no torch).
//
// Family-local headers under kernels/<family>/ own their POD params and
// strategy traits; anything cross-cutting (compute-capability checks, device
// constants) lives here.

#pragma once

namespace astrai {

// Compute-capability comparison: is the device at least (major, minor)?
inline bool sm_at_least(int device_major, int device_minor, int major,
                        int minor) {
    return device_major > major ||
           (device_major == major && device_minor >= minor);
}

// FP8 tensor-core MMA (`mma.sync.aligned.m16n8k32` with fp8 inputs) exists on
// Ada (sm_89) and Hopper (sm_90+); sm_80 has no fp8 instructions.
inline constexpr int kMinSmForFp8Major = 8;
inline constexpr int kMinSmForFp8Minor = 9;

}  // namespace astrai
