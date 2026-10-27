#!/usr/bin/env python3
"""Profiling 入门实验：时间线、CUDA 显存历史、GEMM 形状对比。

依赖：PyTorch。CPU 模式只支持 timeline，不模拟 GPU 性能。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule


@contextmanager
def region(name: str, nvtx: bool = False):
    with record_function(name):
        with torch.cuda.nvtx.range(name) if nvtx else nullcontext():
            yield


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timeline(args: argparse.Namespace, device: torch.device) -> None:
    if args.capture == "nsys" and device.type != "cuda":
        raise ValueError("nsys 实验需要 CUDA；CPU 模式请使用 --capture torch 或 none。")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    x = torch.randn(args.dim, args.dim, device=device, dtype=dtype)
    w = torch.randn_like(x)
    with torch.inference_mode():
        for _ in range(8):
            _ = (x @ w).relu()
    sync(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def save_trace(p) -> None:
        p.export_chrome_trace(str(args.output / "timeline.json"))
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        (args.output / "operators.txt").write_text(
            p.key_averages().table(sort_by=sort_key, row_limit=20), encoding="utf-8"
        )

    profiler = profile(
        activities=activities,
        schedule=schedule(wait=2, warmup=2, active=4, repeat=1),
        on_trace_ready=save_trace,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) if args.capture == "torch" else nullcontext()

    nvtx = args.capture == "nsys"
    if nvtx:
        torch.cuda.profiler.start()
    start = time.perf_counter()
    try:
        with profiler as p, torch.inference_mode():
            for step in range(args.steps):
                with region(f"step_{step}", nvtx):
                    with region("host_prepare", nvtx):
                        if args.delay_ms:
                            time.sleep(args.delay_ms / 1000)
                    with region("gemm_and_relu", nvtx):
                        y = (x @ w).relu()
                if args.capture == "torch":
                    p.step()
            sync(device)
    finally:
        if nvtx:
            torch.cuda.profiler.stop()
    elapsed = time.perf_counter() - start
    result = {
        "device": str(device), "capture": args.capture, "steps": args.steps,
        "delay_ms_per_step": args.delay_ms, "window_wall_ms": elapsed * 1000,
        "average_step_wall_ms": elapsed * 1000 / args.steps,
        "notice": "带 profiler 的时间包含采集及导出扰动；性能比较请另跑 --capture none。",
    }
    (args.output / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def memory(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("memory 实验需要 CUDA；不会用 CPU 分配冒充 CUDA 显存。")
    if not hasattr(torch.cuda.memory, "_record_memory_history"):
        raise RuntimeError("本版 PyTorch 缺少 memory history API，请核对对应版本的官方文档。")
    # 总张量峰值约 384 MiB；保留额外余量，不主动制造 OOM。
    mib = 1024**2
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < 1024 * mib:
        raise RuntimeError("建议至少有 1 GiB 空闲显存再运行本教学实验。")
    torch.cuda.reset_peak_memory_stats(device)
    rows = []

    def sample(phase: str) -> None:
        sync(device)
        rows.append({
            "phase": phase,
            "allocated_MiB": torch.cuda.memory_allocated(device) / mib,
            "reserved_MiB": torch.cuda.memory_reserved(device) / mib,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / mib,
        })

    # 这两个下划线 API 来自官方内存调试文档，接口仍需按安装版本核对。
    torch.cuda.memory._record_memory_history(max_entries=10000)
    try:
        sample("initial")
        weights = torch.zeros(128 * mib, device=device, dtype=torch.uint8)
        sample("weights")
        activation = torch.zeros(192 * mib, device=device, dtype=torch.uint8)
        sample("activation")
        temporary = torch.zeros(64 * mib, device=device, dtype=torch.uint8)
        sample("temporary_peak")
        del temporary
        sample("temporary_freed")
        del activation
        sample("activation_freed")
        del weights
        sample("all_references_released")
    finally:
        try:
            sync(device)
            torch.cuda.memory._dump_snapshot(str(args.output / "memory.pickle"))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            if rows:
                write_csv(args.output / "memory_samples.csv", rows)
    print(f"已保存 {args.output / 'memory.pickle'} 和 memory_samples.csv")
    print("样本点调用 synchronize 是为获得清晰的教学状态；不是生产重叠执行的基准。")


def gemm(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("gemm 实验使用 CUDA Events；CPU 模式仅支持 timeline。")
    rows = []
    k = n = args.dim
    dtype = torch.float16
    with torch.inference_mode():
        for m in args.m_values:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
            c = torch.empty(m, n, device=device, dtype=dtype)
            for _ in range(8):
                torch.mm(a, b, out=c)
            sync(device)
            times_ms = []
            for _ in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.inner):
                    torch.mm(a, b, out=c)
                end.record()
                end.synchronize()
                times_ms.append(start.elapsed_time(end) / args.inner)
            median_ms = statistics.median(times_ms)
            operations = 2 * m * n * k
            ideal_bytes = a.element_size() * (m*k + k*n + m*n)
            rows.append({
                "M": m, "N": n, "K": k, "dtype": str(dtype),
                "median_batch_average_ms": median_ms,
                "effective_TFLOPs": operations / (median_ms / 1000) / 1e12,
                "ideal_tensor_bytes_NOT_measured_HBM": ideal_bytes,
                "ideal_intensity_FLOP_per_byte": operations / ideal_bytes,
            })
    write_csv(args.output / "gemm.csv", rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print("Events 计时可能包含 CPU 提交间隙；小 GEMM 请结合 trace 验证。理想字节数不是实测 HBM 流量。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["timeline", "memory", "gemm"], default="timeline")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--capture", choices=["torch", "none", "nsys"], default="torch", help="只用于 timeline 模式")
    parser.add_argument("--output", type=Path, default=Path("profile_output"))
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 32, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner", type=int, default=30)
    args = parser.parse_args()
    if args.dim <= 0 or args.steps < 8 or args.delay_ms < 0:
        parser.error("dim 必须为正，steps 至少为 8，delay-ms 不得为负。")
    if args.repeats < 1 or args.inner < 1 or any(m < 1 for m in args.m_values):
        parser.error("repeats、inner、m-values 必须为正。")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("未检测到 CUDA。只体验 CPU trace 可指定 --device cpu --mode timeline。")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    environment = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "device": str(device), "seed": 0}
    if device.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(device)
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    {"timeline": timeline, "memory": memory, "gemm": gemm}[args.mode](args, device)


if __name__ == "__main__":
    main()
