---
type: knowledge
status: seed
created: 2026-08-19
updated: 2026-08-19
domains:
  - post-training
  - reinforcement-learning
  - async-rollout
  - data-loading
aliases:
  - MultiTask DataLoader
  - Task-Quota Rollout Scheduler
source:
  - https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md
  - https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/async.md
  - https://raw.githubusercontent.com/areal-project/AReaL/main/areal/api/cli_args.py
  - https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py
  - https://github.com/redai-infra/Relax/blob/main/docs/en/guide/fully-async-training.md
ai_generated: true
reviewed: false
---

# 多数据集混训中的异步 Rollout 配额与陈旧度控制

> [!warning]
> 本文包含 AI 生成或改写的实质内容，尚未完成人工核验。

## 优化后的核心问题

在一个支持 partial rollout 或其他异步范式的多任务 RL 系统中，如何让每个 RL train step 按指定任务配额消费**版本仍有效**的 rollout，同时只为当前缺口继续调度 rollout？

这里的任务至少包括 language RLVR、multimodal RLVR、language agentic 和 multimodal agentic。它们的 rollout 时延可相差很大：单步 RLVR 往往较快，agentic RL 往往较慢。系统需要同时满足两类约束：

- **训练批次约束**：每个 train step 的输入要满足按任务定义的配额；
- **异步有效性约束**：只消费符合 staleness 规则的 rollout，且不让快任务的富余结果挤占慢任务的配额或无限堆积。

关键不是“让 DataLoader 按比例取数”本身，而是把**任务配额、已完成 backlog、pending / in-flight rollout 与版本有效性**一起纳入下一轮 rollout 的提交决策。

## 可判定的输入、决策与输出

设当前训练 step 为 $k$，训练批次包含 $B$ 个不可拆分的训练单元。训练单元应先明确为样本、GRPO group、trajectory，或某种 token-budget chunk；若 GRPO group 需要原子性，不能把“单样本数”当作配额单位。

| 类别 | 内容 |
| --- | --- |
| 输入 | 任务配比 $r_t$、当前 step $k$、每个 rollout 的 `task`、生成版本、完成状态、训练单元数，以及全局 staleness / queue-capacity 规则。 |
| 有效库存 | $I_t(k)$：任务 $t$ 中通过 `eligible(rollout, k)` 检查的已完成结果。`eligible` 的版本方向、阈值与过期策略必须由 staleness 组件唯一解释。 |
| 训练配额 | $q_t(k)$：把 $r_t \times B$ 用确定的整数分配规则（例如最大余数法）转为每个任务的目标训练单元数，且 $\sum_t q_t(k)=B$。 |
| 缺口 | $d_t(k)=\max(0, q_t(k)-\text{已接纳的有效单元数})$。 |
| 决策 | 先从有效 backlog 按配额取数；再在并发和 staleness 容量允许时，仅为 $d_t(k)>0$ 的任务提交或恢复 rollout。 |
| 输出 | 一个配额满足的 train batch，以及可观测的 staged、pending、in-flight、过期和被丢弃数据统计。 |

`[step 0, step 1]` 不能只作为一个模糊标签：rollout 的**生成版本**与它被哪个 **consumer step** 消费必须分别记录。只有在 `eligible` 为真时，`step 0` 的富余结果才可作为 `step 1` 的 backlog。

## 推荐的职责边界与命名

用户最初的两个组件划分是对的，但 staleness 需要成为共享的准入边界，而不是被隐藏在 ratio 逻辑中。

| 层 | 推荐职责 | 建议名称 |
| --- | --- | --- |
| 数据供给 | 按任务读取 dataset，保留 `task`、模态和 group 边界，并支持 checkpoint 恢复；不决定 runtime 配额。 | `MultiTaskDataLoader` |
| 任务配额与调度 | 维护每批 target、已接纳、staged、pending 与 in-flight 的任务级状态；从缺口生成提交计划，并对快任务施加 backpressure。 | `Task-Quota Rollout Scheduler`；内部类可简写为 `TaskQuotaController` |
| 有效性与容量 | 维护 rollout 生成版本、消费版本、staleness、队列容量、过期和恢复语义；向调度器暴露 `eligible` 与可用 capacity。 | `Staleness Admission Controller` 或 `Rollout Data Plane` |

`MultiTask Task Ratio Controller` 容易让人误以为它只做静态 sampling ratio。若保留现有术语，建议配置仍叫 `task_ratio`，但组件命名为 `Task-Quota Rollout Scheduler`：它控制的是**每个训练批次的离散配额和异步库存调度**，而非只控制数据集抽样概率。

## 典型时序

以 $B=8$、单步 RLVR : agentic RL = 1 : 1 为例：

1. 在 `step 0`，已完成且有效的单步 RLVR 有 8 个，agentic RL 有 4 个。训练批次接纳 4 个 RLVR 和 4 个 agentic；其余 4 个 RLVR 进入 staged backlog。
2. 进入 `step 1`，先检查这 4 个 staged RLVR 仍满足 `eligible`。若满足，它们先占满 RLVR 配额，此时 agentic 的缺口为 4、RLVR 的缺口为 0。
3. 调度器只为 agentic RL 提交或恢复 4 个训练单元，不再为 RLVR 增加新的 rollout。若 staged RLVR 已过期，则它们必须被丢弃或重新生成，不能为了凑比例继续消费。

这正是“快任务先生产 `step 1` 可用结果，下一步优先补慢任务”的可验证表达。

## 验收条件与待定语义

- 每个训练批次都满足目标配额；比例无法整除时，采用确定、可复现的 rounding 规则。
- 不会消费超过 staleness 边界的 rollout；版本、任务和 group 边界随结果传播。
- 快任务的 staged + pending + in-flight 数量有上界；慢任务有可见的 deficit 与提交优先级。
- checkpoint / recovery 要么持久化调度状态，要么显式失效未完成与 staged 结果，不能静默复用来源不明的数据。
- 至少记录 `task × version` 的 accepted、staged、pending、in-flight、expired、dropped 和等待时间。
- [待核验] 配额单位最终应按样本、GRPO group、trajectory 还是 token / packed-token 定义；不同模态是否需要成本归一化。

## 简要调研：已有方案与边界

| 系统 | 已有的相近机制 | 与本问题的差距或适用边界 |
| --- | --- | --- |
| [AReaL（官方 GitHub `main`）](https://github.com/areal-project/AReaL) | 官方文档明确支持 rollout 与训练重叠、partial rollout、`max_head_offpolicyness` 的版本差控制；公开配置还提供 `max_concurrent_rollouts`、`queue_size` 和 `consumer_batch_size`。 | 这是 staleness 与异步容量控制的直接参考。对当前公开 CLI 文档和 `main` 的配置源码进行检索时，未找到任务配额或多任务 DataLoader 的专用配置项；这不是对所有自定义 workflow 的否定，只表示本轮 GitHub 公开材料不能作为该能力已实现的证据。 |
| [verl Fully Async Policy Trainer](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md) | `MessageQueue`、`staleness_threshold`、streaming producer / consumer，以及 `partial_rollout=True` 时中断后继续未完成 rollout；其 Mode 3/4 直接对应“快 rollout 提前生成、后续 step 消费仍新鲜的数据”。 | 本轮文档与源码快扫只确认了全局 freshness / sample 流控制，未发现按异构任务配额补齐与抑制快任务的公开实现证据。 |
| [slime fully_async_rollout](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py) | 常驻异步 worker 保持 in-flight pool，完成 group 放入 output queue；被 abort 的 group 回到 `data_buffer`，避免直接进入训练。标准 partial-rollout 路径也能回收未完成样本。 | 当前 fully-async 示例明确仍把 partial-resume 列为 TODO；[公开 Issue #1800](https://github.com/THUDM/slime/issues/1800) 也将版本跟踪、staleness 预算和训练更新 hook 列为待补能力。因此不能把它当作本问题的完整 staleness + quota 方案。 |
| [Relax Fully Async Training](https://github.com/redai-infra/Relax/blob/main/docs/en/guide/fully-async-training.md) | 用 TransferQueue 将 rollout / actor / actor-fwd / reference / advantage 解耦，以 partition 和 consumer `task_name` 记录消费进度，并以 `max_staleness` 对未消费 partition 施加 backpressure。 | 这是很好的 rollout data plane 参考；其 `task_name` 指的是下游消费者，不等同于数据集 / rollout 任务类别。本轮没有找到它对“每个 train step 的跨任务配额”提供的公开证据。 |
| [Libra](https://github.com/NetX-lab/Libra) | 基于 queue pressure、rollout 状态和 staleness 做资源重规划与请求路由，适合参考慢 agentic 任务的资源调度。 | 重点是资源池与请求调度，而不是训练 batch 的语义任务配额。 |

### AReaL（GitHub `main`）复核结论

- [异步训练文档](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/async.md) 明确将 `max_head_offpolicyness > 0` 定义为异步模式，允许 rollout 相对训练策略落后若干版本，并说明 partial rollout 可跨多个策略版本分段完成一条 trajectory。
- 官方 [CLI reference](https://github.com/areal-project/AReaL/blob/main/docs/en/cli_reference.md) 与对应 [配置源码](https://raw.githubusercontent.com/areal-project/AReaL/main/areal/api/cli_args.py) 公开了 `max_concurrent_rollouts`、`queue_size`、`consumer_batch_size` 和 `max_head_offpolicyness`；它们足以表达异步队列、消费批大小和版本准入。
- 本轮对上述公开配置材料的精确检索没有命中 `task_ratio`、`MultiTaskDataLoader` 或 `tasks`。因此，AReaL 是本问题中 **staleness / partial-rollout 层** 的可靠参考；“异构任务逐 step 配额与 backlog 补齐”仍应被视为待设计能力，而不是已由公开 AReaL 证实的能力。

### 当前结论

业界已经分别具备了 async / partial rollout、staleness-bound buffer、显式 data plane 和资源调度等积木。**但在本轮公开资料与 GitHub 源码快扫中，尚未确认有一个公开方案同时明确保证“异构任务每个 train step 的离散配额”与“版本受限的异步 backlog 调度”。**因此，这仍是一个合理且有实现价值的系统问题，而不是单纯配置现有 DataLoader 即可解决的问题。

## 下一轮最小调研动作

- 沿官方 GitHub AReaL 的 RolloutController v1 / v2 与示例配置继续核对：公开配置中未出现任务配额字段，是文档覆盖范围所限，还是当前主线确实未提供通用控制器。
- 对 verl 的 `fully_async_policy` 验证其 MessageQueue 中是否可安全携带 task / modality 元数据，并定位插入 task-quota scheduler 的最小 hook。
- 将一个两任务、两时延的 toy trace 写成测试：验证 `step 0` 的快任务富余在未过期时被 `step 1` 消费，而提交计划只补慢任务。

## 已核验来源（访问日期：2026-08-19）

- verl 项目文档：[Fully Async Policy Trainer](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)。
- AReaL 官方 GitHub 材料：[Asynchronous RL](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/async.md)、[CLI reference](https://github.com/areal-project/AReaL/blob/main/docs/en/cli_reference.md)、[当前 `main` 配置源码](https://raw.githubusercontent.com/areal-project/AReaL/main/areal/api/cli_args.py)。AReaL 小节的事实仅依据上述公开 GitHub 来源。
- slime 项目源码与公开讨论：[fully_async_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py)、[Issue #1800](https://github.com/THUDM/slime/issues/1800)。
- Relax 项目文档：[Fully Async Training](https://github.com/redai-infra/Relax/blob/main/docs/en/guide/fully-async-training.md)。
- Libra 项目仓库：[Efficient Resource Management for Agentic RL Post-Training](https://github.com/NetX-lab/Libra)。

## 后续处理

<!-- 状态可选：seed、exploring、promoted、dropped。升格后在此链接正式项目。 -->

- **处理决定**：已完成第一轮资料与源码快扫；AReaL 部分已改为 GitHub `main` 证据。
- **正式项目**：
- **决定日期**：
- **原因**：现有系统已有可复用积木，但通用的跨任务配额与 staleness backlog 组合仍待验证。
