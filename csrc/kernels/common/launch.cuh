// Launch-and-check macros — pure CUDA, no torch, so out-of-tree harnesses
// (tile sweeps, csrc/tests) share the exact production launch discipline.
//
// Include order matters for overrides: define ASTRAI_LAUNCH_FAIL before
// including this header (directly or via another kernel header) to swap
// print+abort for a throwing check, as the torch entry units do with
// C10_CUDA_CHECK.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define ASTRAI_LAUNCH_FAIL(err, what)                                        \
    do {                                                                     \
        std::fprintf(                                                        \
            stderr, "ASTRAI: %s failed: %s (%s:%d)\n", what,                 \
            cudaGetErrorString(err), __FILE__, __LINE__                      \
        );                                                                   \
        std::abort();                                                        \
    } while (0)

#define ASTRAI_CUDA_CHECK(expr)                                              \
    do {                                                                     \
        cudaError_t astrai_err_ = (expr);                                    \
        if (astrai_err_ != cudaSuccess) {                                    \
            ASTRAI_LAUNCH_FAIL(astrai_err_, #expr);                          \
        }                                                                    \
    } while (0)

// Check a kernel launch. Wrap the raw <<<>>> with this on the next line;
// a rejected configuration must fail loudly instead of silently measuring
// as a constant ~3us no-op (the tile-sweep lesson).
#define ASTRAI_LAUNCH_CHECK() ASTRAI_CUDA_CHECK(cudaGetLastError())
