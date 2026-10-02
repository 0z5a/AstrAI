/*
 * Shared online-softmax recurrence for split-KV and MMA paths. Scores and
 * running maxima stay pre-scale; scale*log2(e) is folded into exp2.
 *
 * Clamp the empty-state scaled max to 0 so masked -FLT_MAX scores weigh 0,
 * not exp2(0)=1. Fully masked rows therefore keep l=0 and normalize to 0.
 */

#pragma once

#include <cfloat>
#include <cuda_runtime.h>
#include <utils/define.cuh>

namespace astrai {
namespace attention {

// Flash-attention style running state: rescale-on-max (m, l) pair.
struct SoftmaxState {
    float m = -FLT_MAX;
    float l = 0.0f;
};

/* Scaled running max; clamp the empty-state sentinel (see above). */
DEVICE_FORCEINLINE float softmax_scaled_max(float m, float s2) {
    return (m == -FLT_MAX) ? 0.0f : m * s2;
}

/* Add the term with beta after rescaling by alpha; w=1 for keys, li for splits. */
DEVICE_FORCEINLINE void
softmax_step(SoftmaxState& s, float score, float w, float& alpha, float& beta, float s2) {
    float nm = fmaxf(s.m, score);
    alpha = exp2f((s.m - nm) * s2);
    beta = exp2f(score * s2 - softmax_scaled_max(nm, s2));
    s.l = s.l * alpha + w * beta;
    s.m = nm;
}

/* Subtract-first keeps corr=1 exactly when the raw max is unchanged. */
DEVICE_FORCEINLINE float softmax_remax(float& m, float cand, float& corr, float s2) {
    float nm = fmaxf(m, cand);
    corr = exp2f((m - nm) * s2);
    m = nm;
    return nm;
}

} // namespace attention
} // namespace astrai
