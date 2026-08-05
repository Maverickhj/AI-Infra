---
type: knowledge
status: active
created: 2026-08-05
updated: 2026-08-05
domains: [knowledge-management]
aliases: [Wiki Maintenance Workflow]
tags: [ai, ingest, query, lint]
source: ["[[30_Sources/Notes/LLM Wiki - Persistent Knowledge Base]]"]
ai_generated: true
reviewed: false
---

# Wiki 维护工作流

## Ingest

适用于“把这篇文章/论文/日志/代码分析加入知识库”。

1. 确认输入身份、日期、版本、来源 URL 或本地路径。
2. 将原始资料保存在 `30_Sources/Raw/`；若它已由 Git 或外部系统可靠保存，则创建链接，不复制大文件。
3. 使用 [[90_Templates/Source Note|Source Note]] 记录主张、证据强度、适用范围、冲突与待核验项。
4. 先搜索 [[index|总索引]] 与相关 MOC，更新已有知识页；只有主题具有独立检索价值时才新建页面。
5. 将操作性知识写入 Runbook，测量结果写入 Experiment，关键取舍写入 ADR。
6. 更新所有受影响页面的 `updated`、`source`、review state 和必要交叉链接。
7. 更新 `index.md`，追加 [[70_Timeline/log|演化日志]]。

## Query

适用于日常工程问题。

1. 从 `index.md` 选择相关 MOC 和知识页。
2. 若问题依赖当前代码或配置，读取真实 checkout；Wiki 只提供假设和搜索路线。
3. 构建“语义—张量—执行—传输—证据”五层答案。
4. 明确事实、推导、推断、建议与待核验项。
5. 若答案形成可复用链路、公式、决策或 Runbook 更新，将其写回合适页面，而非创建聊天摘要堆积。

## Lint

建议在批量 ingest 或每月维护时执行。

### 结构

- 是否存在没有 MOC/index 入链的 orphan note？
- 是否有相同概念的重复页面？
- 项目专属事实是否错误进入通用知识层？
- Raw source 是否被修改？

### 链接与元数据

- Wikilink 目标是否存在？
- `type/status/date/reviewed` 是否合法且必要？
- source note 与派生知识页是否双向可导航？

### 内容

- 新来源是否与旧结论冲突？
- 版本敏感结论是否写明 commit/config/date？
- 静态推断是否被误写为运行时验证？
- 公式是否定义符号、单位、作用范围和排除项？
- Runbook 是否有验证、回滚与结束条件？

### 输出

生成一个短报告：阻断问题、建议修复、planned pages、建议补充的来源或实验。未经授权不批量改名、移动或删除页面。

## 将日常问题沉淀到哪里

| 产物 | 位置 |
| --- | --- |
| 某仓库当前实现与架构地图 | `10_Projects/<Project>/` |
| 可跨项目复用的原理/分析框架 | `20_Knowledge/` |
| 外部论文、文档、代码来源摘要 | `30_Sources/Notes/` |
| 可重复排障步骤 | `40_Runbooks/` |
| 有环境、输入和测量的验证 | `50_Experiments/` |
| 做出的架构选择 | `60_Decisions/` |
| 尚不清楚或未分类 | `01_Inbox/` |

## 可直接使用的请求

```text
请 ingest 这份资料：保留原始来源，更新相关知识页和 MOC，列出冲突与待核验项，并追加 log。
```

```text
请基于 Wiki 和当前代码回答这个问题。先给结论，再给 producer-to-consumer 链，区分代码证明与推断；把可复用部分写回知识库。
```

```text
请 lint AI-Infra：检查 broken links、orphans、重复概念、陈旧/冲突结论和缺失证据，只输出报告，不自动移动或删除文件。
```
