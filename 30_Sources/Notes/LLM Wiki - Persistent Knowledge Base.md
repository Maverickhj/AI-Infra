---
type: source
status: active
created: 2026-08-05
updated: 2026-08-05
aliases: [LLM Wiki]
source: ["[[llm-wiki]]"]
ai_generated: true
reviewed: false
---

# LLM Wiki - Persistent Knowledge Base

## 来源

本地原始文档：[[llm-wiki|llm-wiki.md]]。原文描述一种由 LLM 持续维护个人 Wiki 的模式；本页是面向 AI-Infra 的来源摘要，不替代原文。

## 核心主张

传统 RAG 在每次问题中从原始资料重新检索与拼接；persistent wiki 则把已经完成的总结、交叉引用、矛盾与综合认知保存为长期工件。每个新来源和高价值问题都会更新已有页面，使知识产生复利。

架构分为：

1. 不可变的 raw sources。
2. 由 LLM 维护、用户阅读和审核的 wiki。
3. 约束结构、元数据和操作流程的 schema。

核心操作是 `ingest`、`query` 和 `lint`；`index.md` 负责内容导航，`log.md` 负责时间线。

## 对 AI-Infra 的落地

| 原始模式 | 本库实现 |
| --- | --- |
| Raw sources | `30_Sources/Raw/` 与本地原始文档 |
| Wiki | `20_Knowledge/`、`40_Runbooks/`、`50_Experiments/`、`60_Decisions/` |
| Schema | `AGENTS.md`、`90_Templates/`、`91_AI/` |
| Content index | [[index|总索引]] 与 `80_MOCs/` |
| Chronological log | [[70_Timeline/log|演化日志]] |
| Lint | 链接、孤儿、冲突、证据、成熟度检查 |

## 采用边界

- 第一阶段用 Markdown、Wikilinks 和全文搜索，不提前引入 embedding/RAG 基础设施。
- LLM 拥有写入流程，但不拥有“审核通过”的权力。
- 原始资料不可修改；综合结论可以在新证据下修订。
- 具体项目事实优先放 `10_Projects/`，成熟的通用结论再提炼到 `20_Knowledge/`。

## 派生页面

- [[60_Decisions/ADR-0001-采用可持续演化的工程 Wiki|ADR-0001：采用可持续演化的工程 Wiki]]
- [[91_AI/Wiki 维护工作流|Wiki 维护工作流]]
- [[80_MOCs/大模型工程知识地图|大模型工程知识地图]]
