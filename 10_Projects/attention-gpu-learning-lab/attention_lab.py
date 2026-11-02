#!/usr/bin/env python3
"""A small, explicit attention lab. Requires an existing compatible PyTorch install.

No autograd, dropout, mask, torch.compile, or external flash-attn package.
Run --help for modes. CUDA modes fail explicitly when CUDA is unavailable.
"""
import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def attention_eager(q, k, v, annotate=False):
    def region(name):
        return torch.profiler.record_function(name) if annotate else contextlib.nullcontext()
    with region("attention/QK"):
        scores = q @ k.transpose(-2, -1)
    with region("attention/scale"):
        scores = scores * (q.shape[-1] ** -0.5)
    with region("attention/softmax"):
        probs = torch.softmax(scores, dim=-1)
    del scores
    with region("attention/PV"):
        return probs @ v


def attention_online(q, k, v, block_q=32, block_k=64):
    """Numerical reference, NOT a fused or fast GPU kernel.

    Q/K/V should be float64 for the intended reference check. This Python loop
    repeatedly launches ops on a GPU; use it to understand the recurrence only.
    """
    out = torch.empty(q.shape[:-1] + (v.shape[-1],), device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[-2], block_q):
        qi = q[..., i:i + block_q, :]
        m = torch.full(qi.shape[:-1] + (1,), -float("inf"), device=q.device, dtype=q.dtype)
        ell = torch.zeros_like(m)
        u = torch.zeros(qi.shape[:-1] + (v.shape[-1],), device=q.device, dtype=q.dtype)
        for j in range(0, k.shape[-2], block_k):
            z = (qi @ k[..., j:j + block_k, :].transpose(-2, -1)) * (q.shape[-1] ** -0.5)
            new_m = torch.maximum(m, z.amax(dim=-1, keepdim=True))
            alpha = torch.exp(m - new_m)
            e = torch.exp(z - new_m)
            ell = alpha * ell + e.sum(dim=-1, keepdim=True)
            u = alpha * u + e @ v[..., j:j + block_k, :]
            m = new_m
        out[..., i:i + block_q, :] = u / ell
    return out


def metrics(actual, expected):
    a, b = actual.detach().double().cpu(), expected.detach().double().cpu()
    diff = a - b
    return {
        "max_abs": diff.abs().max().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "relative_l2": (diff.norm() / b.norm().clamp_min(1e-30)).item(),
        "finite": bool(torch.isfinite(a).all()),
    }


def cuda_time(fn, warmup=10, iterations=20, repeats=5):
    # Do not retain the previous output while computing the next iteration.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}


def peak_extra(fn):
    # Measure allocated tensor memory, not allocator reserved memory or DRAM IO.
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak / (1024 ** 2)


def describe(name, t):
    print(json.dumps({"tensor": name, "shape": list(t.shape), "stride": list(t.stride()),
                      "storage_offset": t.storage_offset(), "dtype": str(t.dtype),
                      "contiguous": t.is_contiguous(), "logical_MiB": t.numel() * t.element_size() / 2**20,
                      "storage_pointer": t.untyped_storage().data_ptr()}))


def make_inputs(args, n=None, device=None, dtype=None):
    shape = (args.batch, args.heads, args.seq if n is None else n, args.dim)
    return tuple(torch.randn(shape, device=device or args.device, dtype=dtype or args.dtype) for _ in range(3))


def tensors(args):
    q, k, v = make_inputs(args)
    kt = k.transpose(-2, -1)
    s = q @ kt
    z = s * (args.dim ** -0.5)
    p = z.softmax(dim=-1)
    o = p @ v
    for name, tensor in [("Q", q), ("K", k), ("K_transpose", kt), ("V", v), ("S", s), ("Z", z), ("P", p), ("O", o)]:
        describe(name, tensor)
    print("K and K_transpose share storage:", k.untyped_storage().data_ptr() == kt.untyped_storage().data_ptr())
    kt_copy = kt.contiguous()
    describe("K_transpose_contiguous_copy", kt_copy)
    print("P row-sum max deviation:", (p.float().sum(-1) - 1).abs().max().item())
    print("This inspect mode retains S/Z/P deliberately; do not use its peak memory as the eager lifetime model.")


def check(args):
    if args.seq > 512:
        raise ValueError("Use seq <= 512 for the small CPU float64 correctness reference.")
    q, k, v = make_inputs(args)
    ref_inputs = tuple(x.double().cpu() for x in (q, k, v))
    ref = attention_eager(*ref_inputs)
    out = attention_eager(q, k, v)
    print("eager vs CPU FP64:", json.dumps(metrics(out, ref)))
    # These are teaching thresholds for random normal inputs, not universal tolerances.
    tolerance = 2e-2 if args.dtype in (torch.float16, torch.bfloat16) else 1e-3
    atol = 2e-3 if args.dtype in (torch.float16, torch.bfloat16) else 1e-4
    torch.testing.assert_close(out.double().cpu(), ref, rtol=tolerance, atol=atol)
    print("correctness check: PASS (documented random-input tolerance)")


def online(args):
    if args.seq > 512:
        raise ValueError("Use seq <= 512 for the online recurrence exercise.")
    q, k, v = make_inputs(args, device="cpu", dtype=torch.float64)
    ref = attention_eager(q, k, v)
    for bk in (1, 7, 32, 64):
        out = attention_online(q, k, v, block_q=17, block_k=bk)
        print("block_k", bk, json.dumps(metrics(out, ref)))
        torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-11)
    print("online recurrence: PASS; these Python loops are not a speed implementation")


def profile(args):
    q, k, v = make_inputs(args)
    for _ in range(10):
        attention_eager(q, k, v)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as prof:
        attention_eager(q, k, v, annotate=True)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(args.trace)
    print("Trace:", args.trace)
    print("Timing / peak allocation must be measured separately: profiling may retain tensors and perturb execution.")


def bench(args):
    q, k, v = make_inputs(args)
    fn = lambda: attention_eager(q, k, v)
    print("event timing:", json.dumps(cuda_time(fn, iterations=args.iters)))
    print("peak extra allocated MiB:", peak_extra(fn))
    # A short demonstration of asynchronous host launch vs completed work.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(json.dumps({"host_enqueue_loop_ms_per_call": (t1-t0)*1e3/args.iters,
                      "synchronized_wall_ms_per_call": (t2-t0)*1e3/args.iters}))


def sweep(args):
    for n in (512, 1024, 2048, 4096):
        q, k, v = make_inputs(args, n=n)
        fn = lambda: attention_eager(q, k, v)
        result = cuda_time(fn, iterations=args.iters)
        peak = peak_extra(fn)
        x = args.batch * args.heads * n * args.dim * q.element_size()
        a = args.batch * args.heads * n * n * q.element_size()
        flops = 4 * args.batch * args.heads * n * n * args.dim
        result.update(seq=n, peak_extra_allocated_MiB=peak,
                      ideal_logical_traffic_bytes=6*a+4*x,
                      gemm_GFLOPs=flops/1e9,
                      effective_GEMM_TFLOPs=flops/(result["median_ms"]*1e-3)/1e12)
        print(json.dumps(result))
        del fn, q, k, v


def sdpa(args):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q, k, v = make_inputs(args)
    reference = attention_eager(q, k, v)
    print("eager", json.dumps(cuda_time(lambda: attention_eager(q,k,v), iterations=args.iters)))
    print("eager peak extra MiB", peak_extra(lambda: attention_eager(q,k,v)))
    for backend in (SDPBackend.MATH, SDPBackend.FLASH_ATTENTION):
        # Context forces a backend; an unsupported configuration is reported, never silently relabeled.
        try:
            with sdpa_kernel(backends=[backend]):
                fn = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
                out = fn()
                print(str(backend), "vs eager:", json.dumps(metrics(out, reference)))
                del out
                print(str(backend), json.dumps(cuda_time(fn, iterations=args.iters)))
                print(str(backend), "peak extra allocated MiB:", peak_extra(fn))
        except (RuntimeError, NotImplementedError) as exc:
            print(str(backend), "UNAVAILABLE:", str(exc))
    print("SDPA math may retain FP32 intermediates for FP16/BF16 inputs; its memory model differs from eager_fp16.")
    print("FLASH_ATTENTION is a PyTorch backend label, not evidence of an external flash-attn package version.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tensors", "check", "online", "profile", "bench", "sweep", "sdpa"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32", "fp64"], default="fp32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trace", default="attention_trace.json")
    args = parser.parse_args()
    if min(args.batch, args.heads, args.seq, args.dim, args.iters) < 1:
        parser.error("all sizes and iters must be positive")
    args.dtype = {"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32,"fp64":torch.float64}[args.dtype]
    if args.mode in ("profile", "bench", "sweep", "sdpa") and (not args.device.startswith("cuda") or not torch.cuda.is_available()):
        parser.error("This mode needs a working CUDA PyTorch installation and NVIDIA GPU.")
    if args.device.startswith("cuda"):
        requested = torch.device(args.device)
        torch.cuda.set_device(requested.index if requested.index is not None else torch.cuda.current_device())
    torch.manual_seed(0)
    print("torch", torch.__version__, "build CUDA", torch.version.cuda, "git", torch.version.git_version)
    print("device", args.device, "dtype", args.dtype)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print("GPU", torch.cuda.get_device_name(), "CC", torch.cuda.get_device_capability())
        print("float32_matmul_precision", torch.get_float32_matmul_precision())
    with torch.inference_mode():
        globals()[args.mode](args)


if __name__ == "__main__":
    main()
