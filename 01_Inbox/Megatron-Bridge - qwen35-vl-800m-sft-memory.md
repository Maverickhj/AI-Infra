---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains: []
aliases: []
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/codex/dev-v0.5.1-minimind/docs/debug/qwen35-vl-800m-sft-memory.md
sync_source: docs/debug/qwen35-vl-800m-sft-memory.md
ai_generated: false
reviewed: false
---

> [!todo]
> 自动剪藏自 `Maverickhj/Megatron-Bridge`，待整理到 `10_Projects/Megatron-Bridge/Experiments/`。
> 来源：https://github.com/Maverickhj/Megatron-Bridge/blob/codex/dev-v0.5.1-minimind/docs/debug/qwen35-vl-800m-sft-memory.md
> 同步脚本：`scripts/sync_ai_infra_inbox.py`；如要手工编辑，请先把本笔记移出 `01_Inbox/`。

## 原文元数据

```yaml
type: how-to
status: growing
created: 2026-08-14
updated: 2026-08-14
domains:
  - megatron
  - distributed-training
  - multimodal-training
  - memory-planning
aliases:
  - Qwen3.5-VL 0.8B SFT 显存估算
related:
  - "[[qwen35-vl-800m-sft-smoke]]"
  - "[[qwen3.5 vl 链路观测指南]]"
  - "[[megatron-lm 链路可观测性 roadmap]]"
  - "[[megatron-bridge-estimator-count-parameters]]"
```

## 原文

# Qwen3.5-VL 0.8B SFT 显存估算

本文整理 `qwen35_vl_800m_sft_smoke_config()` 在当前 devcontainer 中的显存估算、
计算过程、实测现象和容量建议。目标是先回答“大概需要多少显存才能跑”，再解释
为什么不能把 host memory 当成 GPU 显存。

## 1. 当前配置

取自 `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py` 的 smoke recipe：

- `TP=1`，`PP=1`，`DP=1`，`CP=1`
- `global_batch_size=1`
- `micro_batch_size=1`
- `seq_length=256`
- `train_iters=2`，`eval_iters=0`
- full SFT，不冻结语言模型、视觉塔或视觉 projection
- `bf16` 混合精度
- `grad_reduce_in_fp32=True`
- `main_grads_dtype=float32`
- `main_params_dtype=float32`
- `use_distributed_optimizer=True`，但 `DP=1`，所以 optimizer state 没有跨卡分摊
- `recompute_granularity=None`
- `sequence_parallel=False`
- `cuda_graph_impl=none`
- smoke recipe 默认 `checkpoint.save=None`

## 2. 参数量来源

训练日志中 DDP/optimizer bucket 的实际元素数为：

```text
873,438,784 parameters ≈ 0.87B
```

这个数比 Bridge 公式型 estimator 打印的 `0.66B` 更完整，因为它包含了
Qwen3.5-VL 的语言模型、视觉塔和 MTP 部分。做显存规划时应使用
`873,438,784`，而不是理论报告里的 `660,707,328`。

## 3. 模型与优化器状态计算

当前是 full SFT + Adam，单卡且没有 optimizer 切分。每个参数按 18 bytes 估算：

```text
18 B/param =
    BF16 weight           2 B
  + FP32 master weight    4 B
  + FP32 gradient         4 B
  + Adam m                4 B
  + Adam v                4 B
```

计算：

```text
873,438,784 params × 18 bytes
= 15,721,898,112 bytes
≈ 15.72 GB   （十进制）
≈ 14.64 GiB  （二进制）
```

## 4. 激活、workspace 与实测峰值

当前 `seq_length=256`、`micro_batch_size=1`，激活不会特别大；但由于
`sequence_parallel=False`、`recompute_granularity=None`，没有 activation
recompute 节省。

训练日志中的实际 CUDA allocator 峰值：

```text
mem-max-allocated-gigabytes: 18.157
mem-max-reserved-gigabytes:  19.105
```

用实际峰值倒推非稳态部分：

```text
18.16 GB - 15.72 GB ≈ 2.4 GB
```

这部分主要包括：

- 激活和临时 tensor；
- 视觉塔、MTP 的额外激活；
- Transformer Engine / attention / cross-entropy 等算子的 workspace；
- CUDA context 和 PyTorch caching allocator 碎片；
- 没有 sequence parallel 和 selective recompute 时的额外中间结果。

## 5. 内置 estimator 为什么偏低

训练日志中的：

```text
Theoretical memory footprints: weight and optimizer=11341.79 MB
```

来自 Bridge 的公式型 estimator。它按 text/decoder config 估算约 `0.66B`
参数，没有完整计入视觉塔和 MTP，因此会低估。实际应以 DDP bucket 的
`873,438,784` 参数为基准。

## 6. 当前 GPU 与 host memory 边界

当前 devcontainer 是 Linux，不是 Windows WDDM 环境。

`nvidia-smi -q -d MEMORY`：

```text
FB Memory Total: 16376 MiB
BAR1 Memory Total: 16384 MiB
```

没有 Windows 常见的 `Shared GPU Memory` 字段。

PyTorch 看到的总显存：

```text
torch.cuda.get_device_properties(0).total_memory
= 17,170,956,288 bytes
≈ 15.992 GiB
= 16376 MiB
```

host memory：

```text
RAM:  15 GiB
Swap:  4 GiB
```

因此：

- GPU 显存硬上限是 `15.992 GiB`。
- host RAM `15 GiB` 是另一块内存，不能自动变成 CUDA 显存。
- 可以显式使用 pinned memory、CPU offload 或 unified/managed memory 使用 host
  memory，但这不是“共享显存”，训练性能通常会明显下降。

## 7. 容量结论

- 理论模型+优化器下限：约 `15.7 GB`。
- 当前 smoke 配置的实测训练峰值：约 `18.2 GB` allocated，`19.1 GB` reserved。
- 建议：
  - `20 GB` 是最低安全线；
  - `24 GB` 比较稳妥；
  - 当前 `16 GB` 卡是临界状态，只适合短序列、单样本、关闭 checkpoint 的 smoke。

## 8. 降显存方向

如果只能在 16 GB 卡上跑：

- 保持 `seq_length=256`、`micro_batch_size=1`；
- 开启 selective recompute；
- 开启 `sequence_parallel`；
- 关闭 checkpoint；
- 使用 LoRA/PEFT，只训练 adapter；
- 关闭视觉塔或冻结视觉部分；
- 使用 CPU offload 或 fractional optimizer state offload。

## 9. 验证命令

查看 GPU 总显存：

```bash
nvidia-smi -q -d MEMORY
```

用 PyTorch 查看设备显存：

```python
import torch

props = torch.cuda.get_device_properties(0)
print(props.name)
print(props.total_memory, props.total_memory / 1024**3)

free, total = torch.cuda.mem_get_info(0)
print(free, total)
```

查看 host memory：

```bash
free -h
cat /sys/fs/cgroup/memory.max
```

运行 smoke：

```bash
FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-smoke-test uv run --no-sync \
  python -m torch.distributed.run --nproc_per_node=1 \
  scripts/training/run_recipe.py \
  --recipe qwen35_vl_800m_sft_smoke_config \
  --step_func qwen3_vl_step \
  train.train_iters=1
```

## 10. 关联文档

- 最小训练链路：[[qwen35-vl-800m-sft-smoke]]
- 观测测试方案：[[qwen3.5 vl 链路观测指南]]
- 通用观测 roadmap：[[megatron-lm 链路可观测性 roadmap]]
- `_count_parameters` 计算逻辑：[[megatron-bridge-estimator-count-parameters]]
