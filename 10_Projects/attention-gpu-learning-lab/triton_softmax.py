#!/usr/bin/env python3
"""A small Attention experiment: fuse score scaling and row-wise softmax.

Requirements: NVIDIA CUDA GPU, a compatible PyTorch build, and Triton.
Examples:
    python triton_softmax.py
    python triton_softmax.py --n 1031 --warps 8
    python triton_softmax.py --d 80  # non-power-of-two scale exposes rounding

QK scores are computed before timing. This is an educational kernel for rows
that fit on chip, not a replacement for a tuned general-purpose softmax.
"""

import argparse
import math
import statistics

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_softmax(S, P, N: tl.constexpr, SCALE: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(S + row * N + cols, mask=valid, other=-float("inf"))
    x = x.to(tl.float32) * SCALE
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    p = numerator / tl.sum(numerator, axis=0)
    tl.store(P + row * N + cols, p, mask=valid)


def fused(scores, scale, warps):
    """Contiguous [rows, columns] FP16 input; output storage is also FP16."""
    rows, columns = scores.shape
    out = torch.empty_like(scores)
    _scale_softmax[(rows,)](
        scores, out, columns, scale,
        BLOCK=triton.next_power_of_2(columns), num_warps=warps,
    )
    return out


def errors(name, actual, expected):
    delta = actual.float() - expected.float()
    abs_max = delta.abs().max().item()
    relative_l2 = (
        torch.linalg.vector_norm(delta)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    ).item()
    print(f"{name}: max_abs={abs_max:.6g}, relative_L2={relative_l2:.6g}")


def benchmark_ms(fn, warmup, repetitions, rounds):
    # Compile and warm up before creating the measured interval.
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            result = fn()
            del result  # do not retain previous outputs across calls
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repetitions)
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--h", type=int, default=8)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This experiment requires a CUDA-enabled PyTorch and NVIDIA GPU.")
    if not 1 <= args.n <= 8192:
        parser.error("This small row-kernel experiment supports 1 <= N <= 8192.")
    if min(args.h, args.d, args.warmup, args.repetitions, args.rounds) < 1:
        parser.error("H, D, warmup, repetitions, and rounds must be positive.")
    if args.h * args.n * args.n >= 2**31:
        parser.error("This teaching kernel uses int32 score offsets; require H*N*N < 2**31.")

    print(f"PyTorch={torch.__version__}, CUDA build={torch.version.cuda}, "
          f"Triton={triton.__version__}")
    print(f"GPU={torch.cuda.get_device_name()}, "
          f"compute capability={torch.cuda.get_device_capability()}")
    print(f"B=1, H={args.h}, N={args.n}, D={args.d}, dtype=float16, "
          f"num_warps={args.warps}")
    torch.manual_seed(0)
    q = torch.randn(args.h, args.n, args.d, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    scores = (q @ k.transpose(-2, -1)).reshape(args.h * args.n, args.n)
    del q, k
    assert scores.is_contiguous()
    scale = args.d ** -0.5

    # Reference uses the SAME already-computed FP16 S; it does not test QK.
    reference = torch.softmax(scores.float() * scale, dim=-1).to(scores.dtype)
    eager = torch.softmax(scores * scale, dim=-1)
    actual = fused(scores, scale, args.warps)
    errors("Triton vs FP32-intermediate reference", actual, reference)
    errors("eager vs FP32-intermediate reference", eager, reference)
    errors("Triton vs eager", actual, eager)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-5)
    # Eager materializes the scaled scores in FP16. Fused keeps them in FP32.
    # These are smoke-check tolerances for the standard-normal inputs above,
    # not a numerical accuracy guarantee for all possible inputs.
    torch.testing.assert_close(actual, eager, rtol=1e-2, atol=3e-5)
    print("Correctness checks passed for this input; bitwise equality is not required.")
    print("Eager rounds scaled S to FP16 before softmax; fused keeps scaled S in FP32.")
    if math.frexp(scale)[0] == 0.5:
        print("Here scale is a power of two (D=64 gives 1/8), so scaling can be exact "
              "apart from range/underflow effects. Try --d 80 to expose rounding.")
    del reference, eager, actual

    eager_ms = benchmark_ms(
        lambda: torch.softmax(scores * scale, dim=-1),
        args.warmup, args.repetitions, args.rounds,
    )
    fused_ms = benchmark_ms(
        lambda: fused(scores, scale, args.warps),
        args.warmup, args.repetitions, args.rounds,
    )
    print(f"CUDA Events median: eager={eager_ms:.6f} ms, Triton={fused_ms:.6f} ms")
    print(f"Measured eager/Triton time ratio={eager_ms / fused_ms:.3f}")
    tensor_bytes = scores.numel() * scores.element_size()
    print(f"S storage={tensor_bytes:,} bytes ({tensor_bytes / 2**20:.3f} MiB)")
    print(f"Ideal logical traffic: eager=4*size(S)={4*tensor_bytes:,} bytes; "
          f"fused=2*size(S)={2*tensor_bytes:,} bytes")
    print("Eager accounting: read S, write scaled S, read scaled S, write P. "
          "Fused accounting: read S, write P.")
    print("These are ideal tensor-I/O counts, NOT measured DRAM/HBM traffic. "
          "Caches, repeated loads, spills, and launch overhead affect runtime; "
          "this kernel is not guaranteed to be faster.")
    print("Timing excludes QK, correctness checks, and first-time Triton compilation. "
          "Very small kernels may still include CPU launch gaps in event intervals.")


if __name__ == "__main__":
    main()
