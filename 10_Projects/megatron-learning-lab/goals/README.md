---
type: project
status: draft
created: 2026-10-02
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# Codex Goal 工作入口

## 当前入口：连续多阶段 PROGRAM-V1

当前使用 [PROGRAM-V1 总目标](PROGRAM-V1/MASTER_GOAL.md)、[阶段定义](PROGRAM-V1/STAGES.md) 与 [启动 prompt](PROGRAM-V1/START_PROMPT.md)。在一个原生 Goal 中依次完成 M00、G02–G09，以及资源授权后的 R01/R02；通过一阶段就继续下一阶段，不再逐项等待用户发送“继续”。

`538afeb` 已提交 G01 实现，M00 先重验当前基线。原 [G01-GQA](G01-GQA/GOAL.md) 的文件和证据保留为历史及回归依据；只有单独重新启动 G01 时才适用它的单目标停止条件。PROGRAM-V1 不降低原有验收，也不自动授权 GPU、集群或发布。

```bash
python tools/program_plan.py check
python tools/program_plan.py next
```

上述工具只做计划/状态和证据文件检查，不启动 Codex、不执行训练、不替代真实测试。新阶段/资源阻塞/最终完成规则见 PROGRAM-V1，不使用下面旧 G01 的 final 作为整个项目的终点。

## G01 历史入口

以下安装和门槛说明保留给历史 G01 使用；当前多阶段运行从上方 PROGRAM-V1 进入。

## 安装与启动

本包安装工具只新增文件，不覆盖 AGENTS.md、CODEX_START.md、package.json 或用户配置，也不 commit/push。脚手架本身不会启动 Codex Goal。

在 `AI-Infra/10_Projects/megatron-learning-lab/` 作为工作目录打开 Codex，先阅读仓库与项目 AGENTS.md，再使用 [启动 prompt](G01-GQA/START_PROMPT.md)。本次阶段范围由 G01 的 GOAL.md 决定；旧 CODEX_START.md 保留为历史首轮范围，不重复实施 P0。

终端中先执行 `codex --version`。已有客户端显示 `/goal` 时直接使用；没有时按官方说明执行 `codex features enable goals`，重新打开会话。此命令会修改 Codex 功能设置，本包不会代为执行。现有 sandbox/审批策略保持不变，禁止用 bypass/yolo 代替环境修复。

## 文件分工

- GOAL.md：目标、范围、停止条件与迭代规则。
- acceptance.json：固定验收项、实际命令、期望产物和浏览器测试标识。
- PROGRESS.md：短进度记录，不是 Codex 自身的 Goal 状态数据库。
- evidence.json：逐项填写证据路径及 SHA256，初始全部 pending。
- tools/goal_gate.py：运行质量检查、保存新日志、核对源码指纹与证据。它不调度 Codex，不控制 Goal 生命周期。

不需要额外 AGENTS.override.md、MCP、插件、数据库、循环 shell 或复制整套训练环境。

## 三个检查入口

```bash
python tools/goal_gate.py preflight
python tools/goal_gate.py baseline
python tools/goal_gate.py final
```

preflight 只检查项目结构和必需工具。baseline 重跑当前 Python、源码完整性、构建、浏览器测试；只能得出 baseline_passed，不能代表 G01 完成。final 另外执行 GQA 数值测试、上游缓存比对、验收文件与证据检查，满足后输出 ready_for_review。

日志保存在被现有 .gitignore 排除的 `runs/goal-g01/`。每次报告都绑定当前源码指纹，不能拿旧 implementation-report.json 作为本次通过证据。`verify <report>` 会检查报告、日志和当前源码是否仍对应。

所有 checks 通过仍不自动证明教学解释正确。Codex 应逐项人工式审读并在 G01 的 REVIEW.md 中记录数学、源码和界面核对；reviewed 仍保持 false，等待用户复核。

## 官方资料

核验于 2026-10-02。Goal 是线程上的持久目标，文件只是项目内的工作约定，不是官方专用的 GOAL.md 自动发现机制。

[Follow a goal](https://learn.chatgpt.com/use-cases/follow-goals)；[Using Goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)；[Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)；[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

官方命令为 `/goal <目标>`、`/goal`、`/goal pause`、`/goal resume`、`/goal clear`。features.goals 当前在配置参考中标为稳定、默认开启。这里不臆造 token_budget 配置字段或无人值守承诺。预算与暂停通过当前客户端支持的控件处理；达到限制不等于目标完成。
