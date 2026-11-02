"""Tiled QK^T only: FP16 input/output, FP32 accumulation; no attention scale.

Examples: python triton_qk.py --n 1024
          python triton_qk.py --n 1031 --block-m 64
Requires an existing CUDA PyTorch + Triton environment; installs nothing.
tl.arange describes logical tile indices, not an explicit lane assignment.
Confirm actual Tensor Core instructions with Nsight Compute or disassembly.
"""

import argparse
import statistics

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit("Use an environment with CUDA PyTorch and Triton installed.") from exc


@triton.jit
def qk_kernel(Q, K, S, N: tl.constexpr, D: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for block_k in range(tl.cdiv(D, BK)):
        dims = block_k * BK + tl.arange(0, BK)
        q = tl.load(Q + head * N * D + rows[:, None] * D + dims[None, :],
                    mask=(rows[:, None] < N) & (dims[None, :] < D), other=0.0)
        k_t = tl.load(K + head * N * D + cols[None, :] * D + dims[:, None],
                      mask=(cols[None, :] < N) & (dims[:, None] < D), other=0.0)
        acc += tl.dot(q, k_t)
    tl.store(S + head * N * N + rows[:, None] * N + cols[None, :],
             acc.to(tl.float16), mask=(rows[:, None] < N) & (cols[None, :] < N))


def benchmark(launch, warmup, repeats, iterations):
    # Compilation, initial allocation, and first-use initialization precede timing.
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_ms = []
    for _ in range(repeats):
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples_ms), min(samples_ms), max(samples_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--block-m", type=int, choices=(32, 64, 128),
                        help="Omit to sweep 32, 64, and 128; BN=64, BK=32.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.n, args.heads, args.d, args.warmup, args.repeats, args.iterations) < 1:
        parser.error("Shapes and timing counts must all be positive.")
    if max(args.heads * args.n * args.n, args.heads * args.n * args.d) >= 2**31:
        parser.error("This teaching kernel uses int32 offsets; require H*N*N and H*N*D < 2**31.")
    if not torch.cuda.is_available():
        raise SystemExit("This experiment requires an NVIDIA CUDA GPU.")

    device = torch.cuda.current_device()
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, triton={triton.__version__}")
    print(f"GPU={torch.cuda.get_device_name(device)}, CC={torch.cuda.get_device_capability(device)}")
    print(f"Q/K shape=(1,{args.heads},{args.n},{args.d}), dtype=float16, contiguous")
    print("Dot accumulates in float32; score storage is float16. Timings exclude JIT.")
    torch.manual_seed(7)
    shape = (1, args.heads, args.n, args.d)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.25
    # Reuse one output allocation across launches and tile configurations.
    out = torch.empty((1, args.heads, args.n, args.n), device="cuda", dtype=q.dtype)
    with torch.inference_mode():
        ref = q @ k.transpose(-1, -2)

    blocks_m = (args.block_m,) if args.block_m is not None else (32, 64, 128)
    for bm in blocks_m:
        bn, bk, warps = 64, 32, 4
        grid = (triton.cdiv(args.n, bm), triton.cdiv(args.n, bn), args.heads)

        def launch():
            qk_kernel[grid](q, k, out, args.n, args.d, bm, bn, bk, num_warps=warps)

        launch()
        torch.cuda.synchronize()
        diff = out.float() - ref.float()
        max_abs = diff.abs().max().item()
        rmse = diff.square().mean().sqrt().item()
        rel_l2 = (diff.norm() / ref.float().norm().clamp_min(1e-12)).item()
        del diff
        # A demonstration tolerance for these scaled random FP16 inputs only.
        # Different input ranges, shapes, and accumulation modes require validation.
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-3)
        median_ms, min_ms, max_ms = benchmark(
            launch, args.warmup, args.repeats, args.iterations
        )
        tflops = (2 * args.heads * args.n * args.n * args.d) / (median_ms * 1e9)
        print(f"BM={bm}, BN={bn}, BK={bk}, warps={warps}, grid={grid}")
        print(f"  max_abs={max_abs:.6g}, RMSE={rmse:.6g}, relative_L2={rel_l2:.6g}")
        print(f"  median={median_ms:.6f} ms, range=[{min_ms:.6f},{max_ms:.6f}] ms, "
              f"effective_GEMM={tflops:.3f} TFLOP/s")
    print("Larger tiles need not be faster: reuse competes with register/shared-memory pressure.")
    print("Repeated launches use warm buffers; these timings are not a cold-DRAM measurement.")


if __name__ == "__main__":
    main()
