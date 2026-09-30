---
type: project
status: active
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# megatron-learning-lab

## 目标与状态

面向有一定基础的初学者，以 Qwen2.5、Qwen3、DeepSeek 完整 Transformer 为学习对象，将模型计算、源码、SFT/RL 数据与更新、分布式执行连接起来。

当前已完成首轮 E0 学习前端，进入人工 review；尚未执行真实 HF/SFT/RL 训练。本轮 Review 已纳入两项要求：基础概括、核心精讲；实际环境最小接口兼容，而非复制 Megatron-Bridge 全量版本锁。

## 工作目录与阅读入口

工作目录：`AI-Infra/10_Projects/megatron-learning-lab/`。本项目有独立目标与交付物，按仓库约定放在 `10_Projects`，不混入 Inbox，也不修改其他学习项目。

- [实施方案](PLAN.md)：范围、页面、实验阶段与验收。
- [学习深度与课程编排](CURRICULUM.md)：基础速览与核心精讲的边界。
- [最小兼容策略](profiles/minimum-compatibility.md)：环境复用、接口/语义合约与探针要求。
- [Codex 首轮执行任务](CODEX_START.md)：直接用于下一轮实现。
- [源码调研](research/FINDINGS.md) 与 [参考文献](research/REFERENCES.md)：此前固定版本的证据和仍需验证项。
- [Trace 合约](profiles/trace-contract.md) 与 [SFT runbook](profiles/bridge_sft_runbook.md)：实际执行时的数据和配置边界。

## 核心课程

[Qwen3 整模与 SFT](content/cases/01_qwen3_complete_sft.md)、[DeepSeek MLA 与 MoE](content/cases/02_deepseek_complete_model.md)、[完整 RL 更新](content/cases/03_complete_rl_update.md)。

基础只做一页可跳过的定位，默认进入整模核心。源码与实验仍须精读/验证；“有基础”不表示已经熟悉 Megatron 的分布式语义。

## 本轮提交范围

本项目目录包含方案、课程、源码证据与摘录、E0 前端、最小兼容合约和离线/浏览器测试；不迁移旧笔记，不改变仓库级设置。大型派生 JSON 和运行产物不入库，使用生成器重建。

离线检查见 [validation-report.json](validation-report.json)。它们不证明 Bridge/HF/RL 数值正确或真实训练可用；GPU 与真实接口行为探针尚未执行；首轮浏览器验收和后续源码摘录记录见 [implementation-report.json](implementation-report.json)。

## 关联项目与已有资料

[[MiniMind-Megatron-Bridge/_Project - MiniMind Megatron Bridge|MiniMind Megatron Bridge]] 提供另一条项目实践背景；[[Transformer Architecture Causal Chain]] 可作为按需基础参考。本轮不修改这两个已有条目。

## 后续深入研读 TODO

- [ ] **掌握 Megatron 基本概念后，沿 NeMo RL、Megatron-Bridge、Megatron-LM 的 roadmap 与近两年关键 feature 深入研读。**
  - 启动条件：能解释整模 forward/backward、SFT mask/shift/归一化、主要并行维度，以及 RL rollout→update→refit 的职责边界。
  - 启动时固定调研截止日期，并按此前 24 个月界定“近两年”；区分 roadmap 规划、已合并实现和已发布能力。
  - 以各仓库官方 roadmap、release notes、设计文档、关键 PR/commit 为入口，筛选与当前案例相关的 feature，记录其跨仓依赖与版本边界。
  - 每个专题沿“解决什么问题 → 原理与数据合约 → 关键源码/演进 PR → 最小实验 → 收益、代价和限制”形成研读笔记。
  - 先建立 feature 清单和优先级，再逐项深入；本条仅记录后续任务，不表示已经完成 roadmap/feature 调研，也不默认启动 GPU 实验。
