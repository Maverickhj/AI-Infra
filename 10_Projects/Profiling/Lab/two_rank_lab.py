#!/usr/bin/env python3
"""两 rank 迟到实验。默认 NCCL/CUDA；--cpu 使用 Gloo，仅验证 CPU 分布式逻辑。

本实验刻意加每步 barrier 和一次 host delay，以展示等待传播。
这些同步是教学条件，不应复制到生产训练循环作为计时方法。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--slow-step", type=int, default=2)
    p.add_argument("--delay-ms", type=float, default=50)
    p.add_argument("--numel", type=int, default=1024 * 1024)
    p.add_argument("--output", type=Path, default=Path("rank_traces"))
    a = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("请使用 torchrun --standalone --nproc-per-node=2 启动。")
    if a.steps < 1 or not 0 <= a.slow_step < a.steps or a.delay_ms < 0 or a.numel < 1:
        p.error("请检查 steps、slow-step、delay-ms 和 numel。")
    if not a.cpu and not torch.cuda.is_available():
        p.error("未发现 CUDA；--cpu 仅用于 Gloo/CPU 教学验证。")
    torch.set_num_threads(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    if a.cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo" if a.cpu else "nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    a.output.mkdir(parents=True, exist_ok=True)

    def sync() -> None:
        if not a.cpu:
            torch.cuda.synchronize(device)

    try:
        x = torch.empty(a.numel, device=device)
        for _ in range(3):
            x.fill_(rank + 1)
            dist.all_reduce(x)
        sync()
        dist.barrier()
        activities = [ProfilerActivity.CPU]
        if not a.cpu:
            activities.append(ProfilerActivity.CUDA)
        rows = []
        with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
            for step in range(a.steps):
                with record_function(f"rank_{rank}/step_{step}"):
                    with record_function("teaching_step_alignment"):
                        dist.barrier()
                        sync()
                    start = time.perf_counter()
                    with record_function("before_collective"):
                        if rank == 1 and step == a.slow_step:
                            with record_function("injected_host_delay"):
                                time.sleep(a.delay_ms / 1000)
                        x.fill_(rank + 1)
                    with record_function("all_reduce_and_completion"):
                        dist.all_reduce(x)
                        sync()
                    rows.append({"step": step, "local_wall_ms": (time.perf_counter()-start)*1000})
                prof.step()
        prof.export_chrome_trace(str(a.output / f"rank_{rank}.json"))
        expected = 3.0  # 两个 rank 的输入分别是 1 和 2。
        if not torch.all(x == expected).item():
            raise RuntimeError("AllReduce 结果校验失败。")
        metadata = {"rank": rank, "device": str(device), "backend": dist.get_backend(),
                    "torch": torch.__version__, "measurements": rows,
                    "note": "带 profiler、人工 barrier 和人工 delay 的教学数据，不是通信带宽基准。"}
        (a.output / f"rank_{rank}_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"rank {rank}: trace 已保存，AllReduce 结果校验通过。")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
