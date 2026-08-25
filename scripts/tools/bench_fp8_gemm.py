#!/usr/bin/env python3
"""FP8 GEMM micro-benchmark: AstrAI kernel vs cuBLAS (torch._scaled_mm).

Sizes 512-8192, forward (NT) layout: x8[M,K] @ w8[N,K]^T -> bf16.
cuBLAS reference uses the same pre-quantized fp8 operands and the same
combined scale, so the comparison isolates the GEMM loop itself.

    python scripts/tools/bench_fp8_gemm.py --sizes 512 1024 2048
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")

from astrai.extension import loader


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    parser.add_argument(
        "--rect", action="store_true", help="also bench M=4096xN=1024 style rectangles"
    )
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    assert loader.is_available("fp8_ops"), "fp8_ops extension not built"
    from astrai.extension.ops.fp8 import mm_fp8

    dev = torch.device("cuda")
    torch.manual_seed(0)

    shapes = [(s, s, s) for s in args.sizes]
    if args.rect:
        shapes += [(4096, 1024, 4096), (8192, 4096, 8192), (2048, 8192, 2048)]

    print(
        f"{'M':>6} {'N':>6} {'K':>6} | {'ours(ms)':>9} {'TFLOPs':>7} | "
        f"{'cublas(ms)':>10} {'TFLOPs':>7} | {'ratio':>6}"
    )
    print("-" * 72)
    for m, n, k in shapes:
        x8 = (torch.randn(m, k, device=dev) * 0.05).to(torch.float8_e4m3fn)
        w8 = (torch.randn(n, k, device=dev) * 0.05).to(torch.float8_e4m3fn)
        scale = torch.ones(1, device=dev, dtype=torch.float32)

        # ours: NT (x8 @ w8^T, LayoutB=ColMajor = weight layout)
        t_ours = bench(lambda: mm_fp8(x8, w8, scale, trans_b=True), iters=args.iters)
        # cuBLAS: _scaled_mm needs A row-major, B column-major (= w8.t())
        wt = w8.t()
        sa = torch.ones(1, device=dev)
        sb = torch.ones(1, device=dev)

        def cublas():
            return torch._scaled_mm(x8, wt, sa, sb, out_dtype=torch.bfloat16)

        t_cublas = bench(cublas, iters=args.iters)
        flops = 2.0 * m * n * k
        tf_ours = flops / (t_ours * 1e-3) / 1e12
        tf_cublas = flops / (t_cublas * 1e-3) / 1e12
        print(
            f"{m:>6} {n:>6} {k:>6} | {t_ours:>9.3f} {tf_ours:>7.1f} | "
            f"{t_cublas:>10.3f} {tf_cublas:>7.1f} | "
            f"{tf_ours / tf_cublas:>5.0%}"
        )


if __name__ == "__main__":
    main()
