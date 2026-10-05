#pragma once
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <arith/softmax.cuh>
#include <memory/pipeline.cuh>
#include <mma/ldmatrix.cuh>
#include <mma/mma.cuh>
#include <utils/define.cuh>
#include <datatype/element.cuh>

/* Predicated cp.async requires CUDA 11.2+; tensor-core mma.sync requires
 * sm_80+. Each element type defines its architecture floor in MmaShapeFor.
 */
#if CUDART_VERSION < 11020
#error "AstrAI CUDA kernels require CUDA 11.2 or later (CUDART_VERSION >= 11020)."
#endif

namespace astrai {
namespace attention {

/* KernelTraits groups compile-time constants shared by the attention kernels.
 * Device functions take one Traits parameter instead of separate values such
 * as KD, NC8, and KT2. Elem selects the MMA cell and storage width, allowing
 * each traits family to support every instantiated precision.
 */
template <int HEAD_DIM_, int BC_, int WARPS_, int STAGES_, typename T_ = bf16> struct KernelTraits {
    using Elem = T_;

    static constexpr int HEAD_DIM = HEAD_DIM_;
    static constexpr int BC = BC_;         // K/V tile size along seq dim
    static constexpr int WARPS = WARPS_;   // warps per block
    static constexpr int STAGES = STAGES_; // double-buffer stages (1 or 2)

    static constexpr int BR = 16; // Q rows per warp (mma M=16)

    /* Derived MMA tile counts use the shared mma_shape. Unsupported element
     * types fail at MmaShapeFor because they have no tensor-core cell.
     */
    static constexpr int KD = HEAD_DIM / astrai::mma_shape<Elem>::k; // Q/K k-slides
    static constexpr int NC8 = BC / 8;                               // S n-tiles (N=8)
    static constexpr int KT2 = BC / astrai::mma_shape<Elem>::k;      // P k-tiles (K=16)
    static constexpr int DN8 = HEAD_DIM / 8;                         // O n-tiles (N=8)

    static constexpr int LD = HEAD_DIM; // smem leading dim

    /* XOR swizzle mask uses log2(LD/8) bits for ldmatrix bank-conflict
     * avoidance, clamped to the LD range.
     */
    static constexpr int SWIZ_MASK = (HEAD_DIM >= 64) ? 7 : (HEAD_DIM / 8 - 1);

    static constexpr int NUM_THREADS = WARPS * 32;
    static constexpr int VEC = 16 / (int)sizeof(Elem); // elements per cp.async unit
    static constexpr int TOTAL = BC * HEAD_DIM;        // total elements per tile
};

/* PTX wrappers: mma.sync is defined in mma/mma.cuh; element packing
 * operations (pack2/unpack2/...) are defined in datatype/element.cuh.
 */

// pack two floats into one 16-bit-pair register as .b32 (mma A/B operand cell)
template <typename T> DEVICE_FORCEINLINE unsigned pk2(float a, float b) {
    return ElemTrait<T>::pack2(a, b);
}

/* ldmatrix is defined in mma/mma.cuh. Its normal and transposed template
 * forms load K/V fragments in the register layout expected by mma.sync.
 */

// XOR swizzle for shared-memory column at 8-element chunk granularity.
DEVICE_FORCEINLINE int swiz_col(int d, int r, int mask = 7) {
    return ((d >> 3) ^ (r & mask)) << 3 | (d & 7);
}

/* cp.async primitives are defined in common/pipeline.cuh. They stage K/V
 * tiles with predicated copies, group commits, and configurable waits.
 */

/* Load query rows from global memory into the MMA A-operand register layout.
 * off_a/off_b are row offsets in Q; they may refer to different heads after
 * PackGQA folding. Decode uses one base because each row represents a head.
 */
template <int KD, typename T>
__device__ inline void load_q_mma_frags(const T* __restrict__ qa,
                                        const T* __restrict__ qb,
                                        int stride_d,
                                        int off_a,
                                        int off_b,
                                        bool va,
                                        bool vb,
                                        int tid4,
                                        unsigned Qa[KD][4]) {
#pragma unroll
    for (int kt = 0; kt < KD; kt++) {
        int c = kt * 16 + tid4 * 2;
        const unsigned* pau = reinterpret_cast<const unsigned*>(&qa[off_a + c * stride_d]);
        const unsigned* pbu = reinterpret_cast<const unsigned*>(&qb[off_b + c * stride_d]);
        Qa[kt][0] = va ? pau[0] : 0u;
        Qa[kt][1] = vb ? pbu[0] : 0u;
        Qa[kt][2] = va ? pau[4] : 0u;
        Qa[kt][3] = vb ? pbu[4] : 0u;
    }
}

/* Stage one BC×HEAD_DIM K/V tile in shared memory using predicated cp.async
 * and XOR swizzling. AddrFn maps (kc, d, valid) to a KVAddr. Decode and
 * prefill provide different address policies; decode may also persist new K/V.
 */
template <typename Traits, typename AddrFn>
__device__ inline void load_kv_tile(typename Traits::Elem* sK, // ring bases (STAGES * BC * LD each)
                                    typename Traits::Elem* sV,
                                    int ti,
                                    int buf, // tile index, ring slot
                                    int seq_len,
                                    const AddrFn& addr) {
    int kv0 = ti * Traits::BC;
    typename Traits::Elem* dK = sK + buf * Traits::BC * Traits::LD;
    typename Traits::Elem* dV = sV + buf * Traits::BC * Traits::LD;
#pragma unroll
    for (int i = threadIdx.x * Traits::VEC; i < Traits::TOTAL;
         i += Traits::NUM_THREADS * Traits::VEC) {
        int r = i / Traits::HEAD_DIM, d = i % Traits::HEAD_DIM;
        int kc = kv0 + r;
        bool valid = kc < seq_len;
        auto a = addr(kc, d, valid);
        int off = r * Traits::LD + swiz_col(d, r, Traits::SWIZ_MASK);
        astrai::cp_async_16(&dK[off], a.k, a.valid);
        astrai::cp_async_16(&dV[off], a.v, a.valid);
    }
    astrai::cp_async_commit_group();
}

/* Compute raw scores S = Q @ K^T from the preloaded Qa fragments. Softmax
 * applies scale * log2(e) later, avoiding a separate per-element multiply.
 * Traits provides Elem, KD, NC8, LD, and SWIZ_MASK.
 */
template <typename Traits>
__device__ inline void mma_compute_scores(const unsigned Qa[Traits::KD][4],
                                          const typename Traits::Elem* __restrict__ sK,
                                          int lane,
                                          float Sacc[Traits::NC8][4]) {
#pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        Sacc[n8][0] = Sacc[n8][1] = Sacc[n8][2] = Sacc[n8][3] = 0.0f;
        int krow_l = n8 * 8 + (lane & 7);
        int kcol_h = (lane & 8) ? 8 : 0;
#pragma unroll
        for (int kt = 0; kt < Traits::KD; kt++) {
            unsigned b[2];
            astrai::ldmatrix_x2<typename Traits::Elem>(
                b,
                &sK[krow_l * Traits::LD + swiz_col(kt * 16 + kcol_h, krow_l, Traits::SWIZ_MASK)]);
            astrai::mma_sync<typename Traits::Elem>(Sacc[n8], Qa[kt], b, Sacc[n8]);
        }
    }
}

/* Update online softmax and rescale Oacc for one K/V tile. MaskView groups
 * mask addressing with both rows' causal/valid predicates. HasMask is a
 * compile-time flag, so the mask path is removed when disabled.
 */
struct MaskView {
    const bool* __restrict__ mask;
    int b_stride, h_stride, l_stride;
    int batch, head0, head1;
    int qrow0, qrow1;
};

template <typename Traits, bool HasMask>
__device__ inline void mma_softmax_tile(int kv0,
                                        int maxc0,
                                        int maxc1,
                                        MaskView mv,
                                        bool valid0,
                                        bool valid1,
                                        float scale_log2,
                                        float Sacc[Traits::NC8][4],
                                        float Oacc[Traits::DN8][4],
                                        float& m0,
                                        float& m1,
                                        float& l0,
                                        float& l1,
                                        int lane) {
    int tid4 = lane & 3;

    float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
    int mask_base0 = mv.batch * mv.b_stride + mv.head0 * mv.h_stride + mv.qrow0 * mv.l_stride;
    int mask_base1 = mv.batch * mv.b_stride + mv.head1 * mv.h_stride + mv.qrow1 * mv.l_stride;
#pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        int cc = kv0 + n8 * 8 + 2 * tid4;
        int c1 = cc + 1;
        bool b0 = !valid0 || (cc >= maxc0) || (HasMask && !mv.mask[mask_base0 + cc]);
        bool b1 = !valid0 || (c1 >= maxc0) || (HasMask && !mv.mask[mask_base0 + c1]);
        bool b2 = !valid1 || (cc >= maxc1) || (HasMask && !mv.mask[mask_base1 + cc]);
        bool b3 = !valid1 || (c1 >= maxc1) || (HasMask && !mv.mask[mask_base1 + c1]);
        float s0 = b0 ? -FLT_MAX : Sacc[n8][0];
        float s1 = b1 ? -FLT_MAX : Sacc[n8][1];
        float s2 = b2 ? -FLT_MAX : Sacc[n8][2];
        float s3 = b3 ? -FLT_MAX : Sacc[n8][3];
        Sacc[n8][0] = s0;
        Sacc[n8][1] = s1;
        Sacc[n8][2] = s2;
        Sacc[n8][3] = s3;
        rmax0 = fmaxf(rmax0, fmaxf(s0, s1));
        rmax1 = fmaxf(rmax1, fmaxf(s2, s3));
    }
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
    rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
    rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

    float corr0, corr1;
    float nm0 = softmax_remax(m0, rmax0, corr0, scale_log2);
    float nm1 = softmax_remax(m1, rmax1, corr1, scale_log2);

    /* Clamp the empty state's scaled max to 0. This keeps masked scores at
     * -FLT_MAX, so exp2 yields 0 rather than 1 for the softmax sentinel.
     */
    float nm2_0 = softmax_scaled_max(nm0, scale_log2);
    float nm2_1 = softmax_scaled_max(nm1, scale_log2);

    float rsum0 = 0.0f, rsum1 = 0.0f;
#pragma unroll
    for (int n8 = 0; n8 < Traits::NC8; n8++) {
        /* The compiler contracts x*s2 - nm2 into an FFMA feeding MUFU.EX2,
         * avoiding a separate scale multiply for each score.
         */
        float p0 = exp2f(Sacc[n8][0] * scale_log2 - nm2_0);
        float p1 = exp2f(Sacc[n8][1] * scale_log2 - nm2_0);
        float p2 = exp2f(Sacc[n8][2] * scale_log2 - nm2_1);
        float p3 = exp2f(Sacc[n8][3] * scale_log2 - nm2_1);
        Sacc[n8][0] = p0;
        Sacc[n8][1] = p1;
        Sacc[n8][2] = p2;
        Sacc[n8][3] = p3;
        rsum0 += p0 + p1;
        rsum1 += p2 + p3;
    }
    rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 1);
    rsum0 += __shfl_xor_sync(0xFFFFFFFF, rsum0, 2);
    rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 1);
    rsum1 += __shfl_xor_sync(0xFFFFFFFF, rsum1, 2);
    l0 = l0 * corr0 + rsum0;
    l1 = l1 * corr1 + rsum1;

    /* Skip O rescaling when the max is unchanged: corr is exactly 1.0f, so
     * multiplying by it would leave each value bit-identical.
     */
    if (corr0 != 1.0f) {
#pragma unroll
        for (int j = 0; j < Traits::DN8; j++) {
            Oacc[j][0] *= corr0;
            Oacc[j][1] *= corr0;
        }
    }
    if (corr1 != 1.0f) {
#pragma unroll
        for (int j = 0; j < Traits::DN8; j++) {
            Oacc[j][2] *= corr1;
            Oacc[j][3] *= corr1;
        }
    }
}

/* Accumulate O += P @ V, where Sacc contains the post-softmax weights.
 * Traits provides Elem, DN8, KT2, LD, and SWIZ_MASK.
 */
template <typename Traits>
__device__ inline void mma_pv_accumulate(float Sacc[][4],
                                         const typename Traits::Elem* __restrict__ sV,
                                         int lane,
                                         float Oacc[Traits::DN8][4]) {
#pragma unroll
    for (int kt2 = 0; kt2 < Traits::KT2; kt2++) {
        unsigned Pa[4];
        Pa[0] = pk2<typename Traits::Elem>(Sacc[kt2 * 2][0], Sacc[kt2 * 2][1]);
        Pa[1] = pk2<typename Traits::Elem>(Sacc[kt2 * 2][2], Sacc[kt2 * 2][3]);
        Pa[2] = pk2<typename Traits::Elem>(Sacc[kt2 * 2 + 1][0], Sacc[kt2 * 2 + 1][1]);
        Pa[3] = pk2<typename Traits::Elem>(Sacc[kt2 * 2 + 1][2], Sacc[kt2 * 2 + 1][3]);
        int vrow_l = kt2 * 16 + (lane & 15);
#pragma unroll
        for (int dn8 = 0; dn8 < Traits::DN8; dn8++) {
            unsigned b[2];
            astrai::ldmatrix_x2<typename Traits::Elem, true>(
                b, &sV[vrow_l * Traits::LD + swiz_col(dn8 * 8, vrow_l, Traits::SWIZ_MASK)]);
            astrai::mma_sync<typename Traits::Elem>(Oacc[dn8], Pa, b, Oacc[dn8]);
        }
    }
}

} // namespace attention
} // namespace astrai
