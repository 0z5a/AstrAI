/*
 * Shared online-softmax recurrence — pure CUDA, no torch.
 *
 * One sentinel policy for every attention consumer: the split-KV combine and
 * the MMA tile softmax (mma_utils.cuh). Scores and the running max live in
 * the RAW (pre-scale) space throughout; softmax_scale_log2 = scale * log2(e)
 * is folded into the exp2 argument so each weight is a single FFMA feeding
 * MUFU.EX2 (FlashAttention's base change — see the softmax.cuh note in
 * mma/utils.cuh).
 *
 * Sentinel policy: while the running max is still -FLT_MAX (no valid key seen
 * yet), the max's OWN exp2 argument would be (-FLT_MAX - -FLT_MAX) * s2 = 0,
 * making every masked-out term weigh 1. Clamping the scaled max to 0 for the
 * still-empty state keeps exp2((-FLT_MAX) * s2 - 0) == 0, so a fully-masked
 * row/split stays l == 0 and normalises to 0 instead of mean(V).
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

/*
 * The scaled-max argument every exp2 uses: max * s2, with the still-empty
 * state clamped to 0 (see the header comment). Reading `m` through this
 * keeps the sentinel at ONE call site instead of leaking into every consumer.
 */
DEVICE_FORCEINLINE float softmax_scaled_max(float m, float s2) {
    return (m == -FLT_MAX) ? 0.0f : m * s2;
}

/*
 * Advance the state with one scored term of weight `w` — a plain key uses
 * w = 1; the split-KV combine merges a (mi, li) partial with w = li.
 * `alpha` rescales the caller's accumulator carrying the OLD max, `beta`
 * weights the new term:  acc = acc * alpha + x * beta;  l likewise.
 */
DEVICE_FORCEINLINE void
softmax_step(SoftmaxState& s, float score, float w, float& alpha, float& beta, float s2) {
    float nm = fmaxf(s.m, score);
    alpha = exp2f((s.m - nm) * s2);
    beta = exp2f(score * s2 - softmax_scaled_max(nm, s2));
    s.l = s.l * alpha + w * beta;
    s.m = nm;
}

/*
 * Running-max advance for consumers that reduce a whole tile before taking
 * the exp (the MMA path): returns the new raw-space max, writes the
 * old-state rescale factor. The subtract-first form keeps corr == 1.0f
 * exactly when the max did not move, preserving the Oacc skip-rescale gate.
 */
DEVICE_FORCEINLINE float softmax_remax(float& m, float cand, float& corr, float s2) {
    float nm = fmaxf(m, cand);
    corr = exp2f((m - nm) * s2);
    m = nm;
    return nm;
}

} // namespace attention
} // namespace astrai
