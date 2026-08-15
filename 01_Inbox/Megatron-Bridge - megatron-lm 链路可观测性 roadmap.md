---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains: []
aliases: []
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/codex/dev-v0.5.1-minimind/docs/debug/megatron-lm%20%E9%93%BE%E8%B7%AF%E5%8F%AF%E8%A7%82%E6%B5%8B%E6%80%A7%20roadmap.md
sync_source: docs/debug/megatron-lm 链路可观测性 roadmap.md
ai_generated: false
reviewed: false
---

> [!todo]
> 自动剪藏自 `Maverickhj/Megatron-Bridge`，待整理到 `10_Projects/Megatron-Bridge/Notes/`。
> 来源：https://github.com/Maverickhj/Megatron-Bridge/blob/codex/dev-v0.5.1-minimind/docs/debug/megatron-lm%20%E9%93%BE%E8%B7%AF%E5%8F%AF%E8%A7%82%E6%B5%8B%E6%80%A7%20roadmap.md
> 同步脚本：`scripts/sync_ai_infra_inbox.py`；如要手工编辑，请先把本笔记移出 `01_Inbox/`。

## 原文元数据

```yaml
type: roadmap
status: growing
created: 2026-08-14
updated: 2026-08-14
domains:
  - megatron
  - distributed-training
  - multimodal-training
  - observability
aliases:
  - Megatron-LM 链路可观测性
  - Megatron 观测分层
related:
  - "[[qwen3.5 vl 链路观测指南]]"
```

## 原文

# Megatron-LM / Megatron-Bridge 链路可观测性 Roadmap

> [!warning]
> 本文是模型无关的通用观测路线图，部分能力仍处于设计阶段，不代表当前仓库已经实现。

## 1. 目标

建立一套不绑定具体模型的 Megatron 训练链路观测能力，用于回答：

- 当前 rank 处于哪些并行/通信 group，实际使用了哪个 schedule；
- 一个 iteration 的 Python/module、autograd/operator、GPU/NCCL 三层链路分别是什么；
- 关键 module 与算子的输入输出 shape、dtype、device 是什么；
- backward、重计算、MTP、optimizer、参数/梯度同步分别发生在哪里；
- 修改 TP/PP/DP/CP/EP/SP/FP8/recompute/CUDA graph 等配置后，链路发生了什么变化。

具体模型只作为**测试场景**，例如 [[qwen3.5 vl 链路观测指南]] 的 Qwen3.5-VL SFT。

## 2. 工作原则

- 优先复用已有 timer、PyTorch profiler、Nsight Systems/Nsight Compute、NVTX 和
  `torch.distributed` / MCore `ProcessGroupCollection` 能力。
- 不在每个 CUDA 算子前后 `print()` 或 `torch.cuda.synchronize()`，避免破坏异步执行与通信重叠。
- 观测数据必须能按 `step` 和 `rank` 对齐；不同来源的数据只有对齐后才能用于结论。
- 各级别独立可落地，不要求一次性实现全部层级。
- 先只改 Bridge；只有当 trace 明确证明存在盲区时，再在上游 Megatron-LM 增加能力。

## 3. 观测对象

| 对象 | 要回答的问题 | 主要来源 |
| --- | --- | --- |
| topology/config/schedule | 我在哪些 group，选择哪个 schedule | dist API、`ProcessGroupCollection`、`ConfigContainer` |
| module call order/shape | 调用了哪些 Python module，输入输出 shape | module hooks 或轻量 shape probe |
| operator/autograd | 调用了哪些 ATen/TE op，shape 与调用栈 | PyTorch profiler |
| GPU/NCCL timeline | 哪些 kernel/collective 实际发生，重叠如何 | Nsight Systems、NVTX |
| communication matrix | 每类 parallelism 引入什么通信 | 多卡单变量对照实验 |

## 4. 分级设计

### L0：拓扑、配置与 schedule 快照

初始化完成后每 rank 写一份 `topology-rank<N>.json`，只写一次，包含：

- global/local rank、world size、hostname、device；
- TP/PP/DP/CP/EP/ETP/VPP size 与当前 rank；
- 每个 process group 的 global ranks；
- `num_microbatches` 与 schedule 名称；
- 本 rank 的 layer range / model chunks；
- model wrapper、DDP、optimizer 的实际类名；
- sequence parallel、recompute、FP8、MTP、packing、CUDA graph、overlap 开关；
- 参数总量、trainable 参数量和每类 module 数量。

优先复用 `src/megatron/bridge/training/utils/pg_utils.py::get_pg_collection()` 和
`torch.distributed.get_process_group_ranks()`，不引入新依赖。

### L1：已有 timer

复用 MCore timers，观察每 iteration 的阶段耗时。这里沿用 MCore 的 timer log level 术语：

- timer log level 1：`forward-backward`、梯度同步、参数 all-gather、optimizer 各子阶段；
- timer log level 2：`batch-generator`、`forward-compute`、`backward-compute`、PP send/recv。

timer 是阶段聚合，不能列出每个 layer/operator；`barrier_with_L1_time` 会改变性能，
读链路时可以开，做性能对比时应关闭。

### L2（可延后）：module 调用顺序 JSONL

目标是生成 per-sequence forward/backward 调用链。实现方式为
`register_forward_pre_hook` / `register_forward_hook` / `register_full_backward_*_hook`，
只记录 tensor metadata，不做 synchronize。

> [!note] 当前决策
> 如果首要目标是 input/output shape，优先使用 L3 PyTorch profiler 的 shape 能力，
> 不先实现 `code_path_observer.py` / `code_path_viz.py`。L2 留到需要解释
> backward/recompute 重复 forward、MTP 独立路径或 PP interleaving 时再补。

### L3：PyTorch profiler operator/autograd trace

观察 ATen、Transformer Engine custom ops、autograd nodes、shape 和调用栈。

```text
profiling.use_pytorch_profiler=true
profiling.profile_step_start=2
profiling.profile_step_end=4
profiling.profile_ranks=[0]
profiling.pytorch_profiler_collect_shapes=true
profiling.pytorch_profiler_collect_callstack=true
```

shape 和 callstack 同时开启开销很大；读链路时可同时开，性能测量时全部关闭。

### L4：Nsys + NVTX + NCCL/CUDA timeline

确认 GPU kernel、stream、NCCL collective 和通信计算重叠。Bridge 的 Nsys 路径已使用
`torch.autograd.profiler.emit_nvtx()`；MCore 自定义 NVTX range 默认关闭，需要显式启用。

```bash
nsys profile \
--sample=none \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
--force-overwrite=true \
--output=results/megatron-code-path-rank0 \
python -m torch.distributed.run --nproc_per_node=1 <training_script> \
profiling.use_nsys_profiler=true \
profiling.profile_step_start=2 \
profiling.profile_step_end=3 \
profiling.profile_ranks=[0] \
profiling.nvtx_ranges=true
```

Nsys 是判断“通信是否实际发生”的权威来源：size=1 process group 上，即使 Python 调用了
reduce-scatter，也可能没有可见的 NCCL data-transfer kernel。

### L5：分布式通信对照

按单变量方式实验，不要第一次同时开启多个并行维度：

1. 单卡：TP=PP=CP=EP=1；
2. 2 卡 DP=2；
3. 2 卡 TP=2 + SP；
4. 2 卡 PP=2；
5. 支持时再做 CP=2；
6. MoE 模型再做 EP=2。

每次保存 L0 topology JSON 与一个 Nsys trace，方便 diff。

## 5. 实施 Roadmap

### Phase 0：现状盘点

当前已可直接使用：

- L0 所需原语：`torch.distributed`、`ProcessGroupCollection`、`ConfigContainer`；
- L1：MCore timers；
- L3：PyTorch profiler，含 shape/callstack 采集；
- L4：Nsys、NVTX、`ncu`；
- 显存：`report_memory()`。

### Phase 1：低风险，直接解决“读懂链路”

1. 新增 topology/config/schedule JSON snapshot；
2. 优先复用 PyTorch profiler 的 shape/autograd 数据，暂不新增 module JSONL observer；
3. profile step start/end 同步启用和关闭 MCore custom NVTX；
4. 修正 PyTorch profiler plugin 的 shape 配置映射；
5. 为上述纯 Python 逻辑增加 unit tests。

只修改 Bridge，不修改 `3rdparty/Megatron-LM`。

### Phase 2：按需补 module 级 trace

当 PyTorch profiler 无法回答“为什么某 Python module 被重复调用”时，再逐步引入：

1. 最小 `shape_probe`：只记录关键 module 的 input/output shape；
2. 完整 `code_path_observer.py`：记录 forward/backward 顺序与阶段边界；
3. `code_path_viz.py`：将 JSONL 渲染成时间线、调用树、拓扑视图。

### Phase 3：上游 Megatron-LM 增强

仅在 trace 证明存在盲区时推动上游增强：

- 给 TP collectives 增加语义化 NVTX；
- 给 DDP bucket grad reduce 和 dist-opt param gather 增加 bucket id/元素数；
- 给 PP P2P send/recv 增加 microbatch/stage 标签；
- 统一 optimizer main-grad copy、inner step、param gather 的 range 命名。

这部分必须落在独立 Megatron-LM 仓库，不应修改 Bridge 的 submodule。

## 6. 验收标准

完成 Phase 1 后应能回答：

1. 任意 rank 实际属于哪些 process group，group 成员是谁？
2. 当前 step 选中了哪个 forward/backward schedule，为什么？
3. 一个 microbatch 的 operator/autograd 链是什么，关键 module 的输入输出 shape 是什么？
4. backward 为什么出现某些 forward module 的重复调用？
5. 哪些 collective 只被 Python 调用，哪些实际产生了 NCCL kernel？
6. optimizer step 中 grad copy、clip、Adam、param copy/all-gather 各自耗时多少？
7. 修改一个 parallelism/config 开关后，拓扑、operator trace 和通信 timeline 分别发生了什么变化？

当前阶段先以 timer 与 profiler timeline 能按 step/rank 对齐为基线；需要解释
backward/recompute 或 Python module 边界时再补充 module JSONL，三者对齐后再视为完整。

## 7. 测试场景

- **首选测试场景**：Qwen3.5-VL SFT，具体单卡/多卡、实验集和预期观察见
  [[qwen3.5 vl 链路观测指南]]。
- **最小冒烟入口**：[[qwen35-vl-800m-sft-smoke]]，用于先验证 SFT forward/backward 链路，
  再叠加 profiler/nsys。
- 其他模型只需替换测试场景，不改变本文的分层设计和 roadmap。
