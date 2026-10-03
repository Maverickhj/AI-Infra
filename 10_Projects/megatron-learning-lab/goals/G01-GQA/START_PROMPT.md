---
type: project
status: draft
created: 2026-10-02
updated: 2026-10-02
ai_generated: true
reviewed: false
---

# G01 启动 prompt

在以本项目为工作目录的 Codex **输入框**发送以下内容，不是在 shell 执行。文件存在本身不会开启 Goal。无需再先要求一轮方案；当前目标已经完成界定。

```text
/goal 在当前 AI-Infra/10_Projects/megatron-learning-lab 中完成 goals/G01-GQA/GOAL.md 定义的 G01：把现有整模导览补成可演算、可定位源码、可验证的 GQA 精讲单元。先读仓库与项目 AGENTS.md、GOAL.md、acceptance.json、PROGRESS.md 和 REVIEW-ed606ee.md；本次阶段由 G01 决定，不重复旧 CODEX_START.md 的 P0。

先运行 python tools/goal_gate.py baseline，确认当前提交的修复和现有功能没有回归，再依次完成独立 CPU 参考、共享 TS 计算、子步骤 UI、源码映射、课程与浏览器验证。基础保持简短，核心精讲；教学小张量明确标 reference，真实 HF/Bridge/RL 仍为 not_run。优先复用现有依赖，只处理当前目标必需接口，不重装训练栈。

允许 GOAL.md 列明的项目内多文件修改，保护无关改动；不降低验收标准，不删除或跳过失败测试，不改仓库级设置，不下载模型权重，不启动 GPU/集群或公开部署，不 commit/push。

每个 checkpoint 更新进度与真实证据。只有 acceptance.json 全部满足、evidence.json 证据齐全、python tools/goal_gate.py final 输出 ready_for_review 且 verify 通过后，才能报告 G01 完成并等待用户审阅；完成后停止，不自动进入 SFT/RL 后续目标。必要检查因权限、依赖、网络或预算不可执行时记录 blocked、已尝试路径和最小解阻条件；同一失败没有新证据不无限重试。不要停在重新写计划或只做页面壳。
```

客户端没有 `/goal` 时按 goals/README.md 的官方入口启用。保留当前 sandbox 和审批策略；本包不指定绕过审批参数，也不写不存在的预算配置键。

若恢复的是同一个有目标的线程，先 `/goal` 查看状态，再按需 `/goal resume`；新线程重新读本文件和进度，不把 PROGRESS.md 当作官方 Goal 的持久化数据库。