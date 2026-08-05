---
type: moc
status: active
created: 2026-08-05
updated: 2026-08-05
aliases: [AI-Infra Index]
ai_generated: true
reviewed: false
---

# AI-Infra 总索引

> [!info]
> 这是内容索引，不是时间线。每次 ingest 或沉淀可复用问答时更新；操作历史见 [[70_Timeline/log|演化日志]]。

## 导航

| 页面 | 用途 |
| --- | --- |
| [[00_Home/首页|首页]] | 进入当前重点、常用 Runbook 与待整理区 |
| [[80_MOCs/大模型工程知识地图|大模型工程知识地图]] | 按主题、学习路径和问题类型导航 |
| [[80_MOCs/Transformer 系统地图|Transformer 系统地图]] | 从张量计算连接到训练、推理和 RL 系统 |
| [[91_AI/Wiki 维护工作流|Wiki 维护工作流]] | Ingest、Query、Lint 的执行约定 |

## 核心知识

| 页面 | 一句话摘要 | 状态 |
| --- | --- | --- |
| [[20_Knowledge/Transformer/Transformer 工程总览|Transformer 工程总览]] | 用统一的数据流理解 attention、MLP、残差与执行边界 | seed |
| [[20_Knowledge/Transformer/张量形状与计算约定|张量形状与计算约定]] | 形状、dtype、布局与单位的统一分析协议 | seed |
| [[20_Knowledge/Distributed/大模型并行训练|大模型并行训练]] | DP/TP/PP/CP/EP/SP 的切分对象、通信与约束 | seed |
| [[20_Knowledge/Memory/显存分析方法|显存分析方法]] | 从静态预算到运行时峰值的显存账本 | seed |
| [[20_Knowledge/Inference/推理系统与训练对齐|推理系统与训练对齐]] | 比较训练与推理 runtime 时建立同一性证据链 | seed |
| [[20_Knowledge/RL/LLM 强化学习训练链路|LLM 强化学习训练链路]] | rollout、reward、log-prob、advantage、update 的闭环 | seed |
| [[20_Knowledge/Multimodal/多模态模型数据流|多模态模型数据流]] | 从媒体处理到特征注入与 loss mask 的端到端路径 | seed |
| [[20_Knowledge/Systems/跨仓库调用链追踪|跨仓库调用链追踪]] | 在多仓库分布式系统中定位真实所有权和消费点 | seed |

## Runbooks

| 页面 | 适用场景 |
| --- | --- |
| [[40_Runbooks/分布式训练故障排查|分布式训练故障排查]] | 失败、挂起、重启、POD_FAILED、NCCL/CUDA 异常 |
| [[40_Runbooks/OOM 排查|OOM 排查]] | 训练或推理显存不足、碎片化、泄漏与容量回归 |
| [[40_Runbooks/跨 Runtime 数值对齐|跨 Runtime 数值对齐]] | Megatron、Transformers、SGLang 等实现输出不一致 |

## 来源、实验与决策

| 页面 | 类型 |
| --- | --- |
| [[30_Sources/Notes/LLM Wiki - Persistent Knowledge Base|LLM Wiki - Persistent Knowledge Base]] | 建库模式来源笔记 |
| [[50_Experiments/实验索引|实验索引]] | 验证记录入口 |
| [[60_Decisions/ADR-0001-采用可持续演化的工程 Wiki|ADR-0001：采用可持续演化的工程 Wiki]] | 知识库架构决策 |
| [[70_Timeline/log|演化日志]] | Append-only 操作时间线 |

## 模板

- [[90_Templates/Knowledge Note|Knowledge Note]]
- [[90_Templates/Source Note|Source Note]]
- [[90_Templates/Experiment|Experiment]]
- [[90_Templates/Runbook|Runbook]]
