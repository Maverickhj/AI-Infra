---
type: decision
status: active
created: 2026-08-05
updated: 2026-08-05
domains: [knowledge-management]
aliases: [ADR-0001]
source: ["[[30_Sources/Notes/LLM Wiki - Persistent Knowledge Base]]"]
ai_generated: true
reviewed: false
---

# ADR-0001：采用可持续演化的工程 Wiki

## 状态

提议采用，待人工审核。

## 背景

大模型工程问题跨 Transformer 数学、多个代码仓库、分布式进程、训练/推理 runtime 与集群运维。若答案只保留在聊天中，后续问题会重复发现同一调用链和边界条件，且容易复用过期版本结论。

## 决策

采用三层知识库：

- `30_Sources/Raw/` 保存不可变原始输入，`30_Sources/Notes/` 保存来源摘要。
- Wiki 层按生命周期保存项目、通用知识、Runbook、实验和决策。
- `AGENTS.md` 与 [[91_AI/Wiki 维护工作流|Wiki 维护工作流]] 约束 AI 的 ingest/query/lint 行为。

使用 [[index|总索引]] 做内容目录，使用 [[70_Timeline/log|演化日志]] 记录操作时间线，使用 MOC 维护学习路径与高价值连接。

## 备选方案

### 仅保存聊天记录

成本最低，但难以版本化、交叉链接和主动发现冲突。

### 只对原始资料做 RAG

适合大规模检索，但每次仍需重新综合；不能自然保存已经形成的工程判断和排障流程。

### 一开始引入向量数据库

扩展性强，但当前知识规模不足以抵消运维和 schema 复杂度。先使用索引、Wikilinks 与全文搜索，达到实际瓶颈后再评估。

## 后果

正面：问题与资料会复利；证据、版本和边界更可见；Runbook 与实验可直接复用。

负面：每次高价值交互需要更新多个链接与日志；AI 生成页面需要人工审核；若缺少 lint，仍可能出现重复、孤儿和陈旧结论。

## 复审条件

- 页面达到数百且全文搜索与索引明显不足。
- 同一概念出现持续冲突或重复，现有 schema 无法控制。
- 团队协作需要审批、权限或自动化 ingest pipeline。
