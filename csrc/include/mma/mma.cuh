/*
 * Shared mma.sync wrappers — pure CUDA, no torch.
 *
 * The instruction vocabulary is assembled from two trait layers (the
 * humming codegen's compile-time format, hand-written):
 *
 *   MmaShapeFor<Dtype> — instruction shape per input dtype. The primary
 *      template is UNDEFINED: a dtype with no tensor-core MMA is a compile
 *      error at the use site, not a silent fallback. The K extent follows
 *      the 256-bit A-fragment invariant (16B per lane row): bf16 k16, the
 *      1-byte dtypes k32. kMinArch encodes each instruction's hardware
 *      floor — the one place the requirement lives. (humming also keys on
 *      an Arch tag for its sm_75 Turing thin-instruction correction; every
 *      AstrAI target is sm_80+, where one shape serves all archs, so the
 *      arch dimension collapses into kMinArch's build-time assert.)
 *
 *   MmaOp<A, B, Shape> — one specialization per instantiated
 *      <dtype-pair, shape> cell: accumulator type, register counts and the
 *      dedicated asm block. Mixed dtype pairs never reach the tensor core
 *      (the gemm promotes both sides to MmaT first), so every cell is
 *      symmetric; the A/B parameters keep the pairing explicit.
 *
 * All floating-point variants accumulate into fp32; the s8 pair accumulates
 * into s32 (satfinite clamps the wrap that all-max-magnitude K~16k inputs
 * could reach — the standard production int8-GEMM semantics). `d` may alias
 * `c` (in-place accumulate, as the FP8 GEMM does).
 */

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cuda_runtime.h>
#include <type_traits>

#include <utils/define.cuh>
#include <utils/shape.cuh>
#include <utils/tensor.cuh>

namespace astrai {

/*
 * Compute capability of the current compilation pass: 0 in the host pass,
 * the numeric CC (e.g. 890) in device passes where __CUDA_ARCH__ is defined.
 * Defined() cannot appear in expressions, so this macro lets the mma ops use
 * the arch in a static_assert instead of per-branch #if guards.
 */
#ifndef __CUDA_ARCH__
#define ASTRAI_DEVICE_ARCH 0
#else
#define ASTRAI_DEVICE_ARCH __CUDA_ARCH__
#endif

/*
 * Family (sm_100f/sm_120f/...) arch of the current pass, same trick: 0 in
 * the host pass and on plain (non-'f', non-'a') targets, else the family
 * CC (e.g. 1200). The value is usable in device expressions — the mx cell
 * keys its asm on it (#if) and any cell can static_assert it.
 */
#ifndef __CUDA_ARCH_FAMILY_SPECIFIC__
#define ASTRAI_ARCH_FAMILY 0
#else
#define ASTRAI_ARCH_FAMILY __CUDA_ARCH_FAMILY_SPECIFIC__
#endif

/*
 * --- <Dtype> -> instruction shape ------------------------------------------
 * Primary template undefined: illegal <Dtype> combinations fail at compile
 * time. Specializations spell one Shape<M, N, K> each.
 */
template <typename Dtype> struct MmaShapeFor;

template <> struct MmaShapeFor<__nv_bfloat16> {
    using type = Shape<16, 8, 16>; // f16/bf16 family: 256b / 16 bits
    static constexpr int kMinArch = 800;
};

template <> struct MmaShapeFor<int8_t> {
    using type = Shape<16, 8, 32>; // s8: 256b / 8 bits (sm_80 wide form)
    static constexpr int kMinArch = 800;
};

template <> struct MmaShapeFor<__nv_fp8_e4m3> {
    using type = Shape<16, 8, 32>;
    static constexpr int kMinArch = 890; // fp8 mma.sync, sm_89+ (Ada/Hopper)
};

template <> struct MmaShapeFor<__nv_fp8_e5m2> {
    using type = Shape<16, 8, 32>;
    static constexpr int kMinArch = 890;
};

/*
 * --- <A, B, Shape> -> the mma op -------------------------------------------
 * Primary template undefined: only the instantiated cells below exist.
 *
 * Each cell names its register cells as types (humming's ARegisters /
 * BRegisters / CRegisters role): AFrag/BFrag/CFrag over utils/tensor.cuh's
 * ArrayEngine (cute's Array). The typed fma overload takes fragments BY
 * REFERENCE, so
 * fragment tensors index by semantic coordinates and no pointer arithmetic
 * survives at the mma seam; the raw-array fma stays the core (the
 * attention kernels' mma_sync and the C tests build on it).
 */
template <typename A, typename B, typename ShapeT> struct MmaOp;

/*
 * Every cell shares one register contract (4x b32 A, 2x b32 B, 4-acc C/D)
 * and one operand list, so the CELL is one template skeleton whose typed
 * facts (accumulator, arch floor) derive from traits — the single source —
 * and whose only free variable is the asm statement itself, injected as a
 * macro-stamped tag type (asm operands and constraints must be literals
 * inside the asm expression, so they cannot ride a template parameter;
 * everything else can and does). The tag's asm_fma carries the constraint
 * list that differs per cell: float families feed C as "f" inputs, the s8
 * cell accumulates in place ("+r" outputs, no C input). fma is a template
 * keyed on the pass arch so the static_assert fires only when the cell is
 * CALLED: member bodies of full specializations are checked in every
 * including TU, and a bare assert would trip uncalled on older passes
 * (attention includes this header for the bf16 cell and ldmatrix).
 */
#define ASTRAI_MMA_ASM(TAG, ACC, ASM, OUT_C, ...)                                                  \
    struct TAG {                                                                                   \
        static DEVICE_FORCEINLINE void                                                             \
        asm_fma(ACC d[4], const unsigned a[4], const unsigned b[2], const ACC c[4]) {              \
            asm volatile(ASM                                                                       \
                         : OUT_C(d[0]), OUT_C(d[1]), OUT_C(d[2]), OUT_C(d[3])                      \
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),                  \
                           "r"(b[1])__VA_ARGS__);                                                  \
        }                                                                                          \
    };

ASTRAI_MMA_ASM(MmaAsmBf16,
               float,
               "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
               "=f",
               ,
               "f"(c[0]),
               "f"(c[1]),
               "f"(c[2]),
               "f"(c[3]))
ASTRAI_MMA_ASM(MmaAsmE4m3,
               float,
               "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
               "=f",
               ,
               "f"(c[0]),
               "f"(c[1]),
               "f"(c[2]),
               "f"(c[3]))
ASTRAI_MMA_ASM(MmaAsmE5m2,
               float,
               "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
               "=f",
               ,
               "f"(c[0]),
               "f"(c[1]),
               "f"(c[2]),
               "f"(c[3]))
ASTRAI_MMA_ASM(MmaAsmS8,
               int32_t,
               "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};",
               "+r", )

#undef ASTRAI_MMA_ASM

/*
 * The accumulator the cell's C/D registers carry: float for the float
 * families, the int32 the s8 pair accumulates into.
 */
template <typename DT> struct MmaAccFor;
template <> struct MmaAccFor<__nv_bfloat16> {
    using type = float;
};
template <> struct MmaAccFor<__nv_fp8_e4m3> {
    using type = float;
};
template <> struct MmaAccFor<__nv_fp8_e5m2> {
    using type = float;
};
template <> struct MmaAccFor<int8_t> {
    using type = int32_t;
};

/*
 * The cell skeleton: accumulator from MmaAccFor, arch floor from
 * MmaShapeFor, asm from the tag. Mixed dtype pairs never reach the tensor
 * core (the gemm promotes both sides to MmaT first), so every cell is
 * symmetric; the MmaOp<A, B, Shape> specializations below keep the pairing
 * explicit.
 */
template <typename DT, typename AsmT> struct MmaOpImpl {
    using AccT = typename MmaAccFor<DT>::type;
    static constexpr int kARegs = 4;
    static constexpr int kBRegs = 2;
    static constexpr int kCRegs = 4;
    using AFrag = ArrayEngine<unsigned, kARegs>;
    using BFrag = ArrayEngine<unsigned, kBRegs>;
    using CFrag = ArrayEngine<AccT, kCRegs>;
    static DEVICE_FORCEINLINE void fma(CFrag& d, const AFrag& a, const BFrag& b, const CFrag& c) {
        fma<ASTRAI_DEVICE_ARCH>(d.storage, a.storage, b.storage, c.storage);
    }
    template <int Arch = ASTRAI_DEVICE_ARCH>
    static DEVICE_FORCEINLINE void
    fma(AccT d[4], const unsigned a[4], const unsigned b[2], const AccT c[4]) {
        static_assert(Arch == 0 || Arch >= MmaShapeFor<DT>::kMinArch,
                      "mma.sync requires the cell's kMinArch");
        AsmT::asm_fma(d, a, b, c);
    }
};

template <>
struct MmaOp<__nv_bfloat16, __nv_bfloat16, Shape<16, 8, 16>>
    : MmaOpImpl<__nv_bfloat16, MmaAsmBf16> {};
template <>
struct MmaOp<__nv_fp8_e4m3, __nv_fp8_e4m3, Shape<16, 8, 32>>
    : MmaOpImpl<__nv_fp8_e4m3, MmaAsmE4m3> {};
template <>
struct MmaOp<__nv_fp8_e5m2, __nv_fp8_e5m2, Shape<16, 8, 32>>
    : MmaOpImpl<__nv_fp8_e5m2, MmaAsmE5m2> {};
template <> struct MmaOp<int8_t, int8_t, Shape<16, 8, 32>> : MmaOpImpl<int8_t, MmaAsmS8> {};

/*
 * --- full-rate fp8 via block_scale (sm_120 family) -------------------------
 * The plain fp8 mma.sync decodes at half rate on sm_120 (measured issue
 * rate 506 vs 1011 TFLOPS, RTX 5090); kind::mxf8f6f4 block_scale runs full
 * rate with an identical A/B/C/D register contract and constant unit scales
 * (every ue8m0 byte 0x7f = 2^0, selectors inert — the scale-factored
 * product IS the plain product). Warp-level block_scale is sm_120-family
 * only (100a/103a/110a reject it; datacenter Blackwell does MX through
 * tcgen05), so fma dispatches on the family pass (the same call-time
 * template trick as the plain fp8 cells) and falls back to the plain cell.
 */
template <typename InT> struct MxMmaOp;

/*
 * Same stamping as the plain fp8 cells: the two formats differ only in the
 * format token and the plain cell they fall back to off the sm_120 family.
 */
#define ASTRAI_MX_MMA_OP(FP8T, FMT)                                                                \
    template <> struct MxMmaOp<FP8T> {                                                             \
        using AccT = float;                                                                        \
        static constexpr int kARegs = 4;                                                           \
        static constexpr int kBRegs = 2;                                                           \
        static constexpr int kCRegs = 4;                                                           \
        static constexpr int kMinArch = 1200; /* block_scale mxf8f6f4 */                           \
        using AFrag = ArrayEngine<unsigned, kARegs>;                                               \
        using BFrag = ArrayEngine<unsigned, kBRegs>;                                               \
        using CFrag = ArrayEngine<AccT, kCRegs>;                                                   \
        template <int Family = ASTRAI_ARCH_FAMILY>                                                 \
        static DEVICE_FORCEINLINE void                                                             \
        fma(CFrag& d, const AFrag& a, const BFrag& b, const CFrag& c) {                            \
            fma<Family>(d.storage, a.storage, b.storage, c.storage);                               \
        }                                                                                          \
        template <int Family = ASTRAI_ARCH_FAMILY>                                                 \
        static DEVICE_FORCEINLINE void                                                             \
        fma(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {              \
            if constexpr (Family >= 1200) {                                                        \
                constexpr uint32_t sf_one = 0x7f7f7f7fu; /* ue8m0 1.0 x4 */                        \
                const uint16_t sel = 0;                                                            \
                asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale."                        \
                             "scale_vec::1X.m16n8k32.row.col.f32." FMT "." FMT                     \
                             ".f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "                  \
                             "{%10,%11,%12,%13}, {%14}, {%15,%16}, {%17}, "                        \
                             "{%18,%19};"                                                          \
                             : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])                      \
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),   \
                               "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(sf_one), "h"(sel),  \
                               "h"(sel), "r"(sf_one), "h"(sel), "h"(sel));                         \
            } else {                                                                               \
                MmaOp<FP8T, FP8T, Shape<16, 8, 32>>::fma(d, a, b, c);                              \
            }                                                                                      \
        }                                                                                          \
    };

ASTRAI_MX_MMA_OP(__nv_fp8_e4m3, "e4m3")
ASTRAI_MX_MMA_OP(__nv_fp8_e5m2, "e5m2")

// --- convenience views over the trait layers --------------------------------

/*
 * Compile-time facts of an input type's MMA, flattened for the layers that
 * want plain ints (the attention kernels' KD/KT2 math). Derives entirely
 * from the two traits above.
 */
template <typename InT> struct mma_shape {
    static constexpr int k = MmaShapeFor<InT>::type::kK;
};

/*
 * d[4] = a[4] x b[2] + c[4], row-major A, col-major B — the fp32-accumulating
 * family (bf16 / fp8 pairs). The s8 cell keeps its int32 accumulators and is
 * reached through MmaOp directly. Building for a compute capability below
 * MmaShapeFor<InT>::kMinArch is a **compile error** — the instruction does
 * not exist there, and a silent no-op would produce wrong results.
 */
template <typename InT>
static DEVICE_FORCEINLINE void
mma_sync(float d[4], const unsigned a[4], const unsigned b[2], const float c[4]) {
    static_assert(ASTRAI_DEVICE_ARCH == 0 || ASTRAI_DEVICE_ARCH >= MmaShapeFor<InT>::kMinArch,
                  "mma_sync: this MMA shape requires a newer compute "
                  "capability than the build target");
    MmaOp<InT, InT, typename MmaShapeFor<InT>::type>::fma(d, a, b, c);
}

#undef ASTRAI_DEVICE_ARCH

/*
 * ldmatrix — cooperatively load 8x8 b16 matrices from smem into registers.
 *
 * The instruction is identical for every 16-bit-storage element type: bf16
 * maps 1:1 onto b16 slots; fp8 and s8 are stored packed two-per-slot (see
 * gemm/mainloop.cuh), so one b16 slot holds two 1-byte values. `T` is the element
 * type and only serves as a semantic tag.
 *
 *   x2 (single address): matrix0 = p (8 rows), matrix1 = p + 8*16 bytes
 *   x4:                  four matrices at p, +128, +256, +384 bytes
 *   Trans:               transpose variant (V fragments of attention)
 *
 * ldmatrix takes a *single* smem address per thread, but the addresses of
 * the 32 lanes are *not* all the same: lane i supplies the start address of
 * matrix-row i (modulo 8) for matrix (i/8) — lanes 0-7 feed matrix 0's rows,
 * lanes 8-15 matrix 1's rows (x2/x4), lanes 16-23 / 24-31 matrix 2 / 3's rows
 * (x4 only; their addresses are ignored by x2). Each matrix is 8 rows x 16
 * bytes, and consecutive matrices of one instruction are contiguous at
 * 128-byte strides. 1-byte fragment layouts in gemm/mainloop.cuh are arranged
 * around this constraint.
 */

/*
 * Per-lane-address cores: the caller supplies a raw shared-memory address
 * per lane instead of one common pointer. Use when the fragment tiles are
 * XOR-swizzled per 16B chunk so each lane must compute its own row and
 * chunk address (see gemm/mainloop.cuh's a_lane_off / b_lane_off and the
 * trans selectors for the m16n8k32 operand layouts). Trans selects the
 * transposed load — the gemm's crosswise 16-bit staging ([K][rows] tiles)
 * reads its fragments through it. (ldmatrix is a b16-only instruction:
 * 8-bit crosswise operands keep the PRPT staging + plain loads.)
 */
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

/*
 * Array-typed cores: the register cell IS the destination — fragment
 * tensors hand their cells straight to the instruction, no decayed pointers
 * at the seam. Same instructions, forwarding wrappers.
 */
template <bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x2_lane(ArrayEngine<unsigned, 2>& f, unsigned addr) {
    ldmatrix_x2_lane<Trans>(f.storage, addr);
}

template <bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x4_lane(ArrayEngine<unsigned, 4>& f, unsigned addr) {
    ldmatrix_x4_lane<Trans>(f.storage, addr);
}

/*
 * Common-pointer wrapper over the per-lane core (see the matrix layout notes
 * above).
 */
template <typename T, bool Trans = false>
static DEVICE_FORCEINLINE void ldmatrix_x2(unsigned r[2], const T* p) {
    ldmatrix_x2_lane<Trans>(r, __cvta_generic_to_shared(p));
}

} // namespace astrai
