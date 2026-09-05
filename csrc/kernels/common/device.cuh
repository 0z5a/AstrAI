// Cross-family device helpers shared by the torch bindings.
//
// Family-local headers under kernels/<family>/ own their POD params and
// strategy traits; anything cross-cutting (compute-capability checks, device
// constants, shared binding-side tensor validation) lives here. Binding
// helpers — hence torch; the pure-CUDA kernel headers never include this.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <mutex>
#include <unordered_map>

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

inline void check_fp8_device(const torch::Tensor& tensor) {
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported;
    const int device = tensor.device().index();
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = supported.find(device);
        if (it != supported.end()) {
            TORCH_CHECK(it->second, "FP8 MMA requires compute capability 8.9+");
            return;
        }
    }
    const auto* properties = at::cuda::getDeviceProperties(device);
    const bool ok = sm_at_least(properties->major, properties->minor,
                                kMinSmForFp8Major, kMinSmForFp8Minor);
    {
        std::lock_guard<std::mutex> lock(mutex);
        supported.emplace(device, ok);
    }
    TORCH_CHECK(ok, "FP8 MMA requires compute capability 8.9+");
}

inline void check_scale(const torch::Tensor& scale, const torch::Tensor& input) {
    TORCH_CHECK(scale.is_cuda() && scale.device() == input.device() &&
                    scale.scalar_type() == torch::kFloat32 && scale.numel() == 1,
                "scale must be a CUDA float32 scalar on the input device");
}

}  // namespace astrai
