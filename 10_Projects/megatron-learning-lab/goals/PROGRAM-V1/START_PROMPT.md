---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# PROGRAM-V1 启动与恢复

在项目根 `AI-Infra/10_Projects/megatron-learning-lab/` 打开 Codex。先确认没有另一个会话写同一工作区；原 G01 Goal 仍活动时先暂停/清除或通过当前客户端编辑为下面的总 Goal。以下是发到 Codex 输入框的文本，不是 shell 命令。

```text
/goal 连续完成当前 megatron-learning-lab 的 goals/PROGRAM-V1/MASTER_GOAL.md 与 plan.json 定义的 PROGRAM-V1。先读仓库/项目 AGENTS.md、总目标、STATE.json、PROGRESS.md，按需读当前阶段 STAGES.md 和 VALIDATION.md。以当前最新工作区接续已提交 G01，不重建网站、不重复 P0。

先执行 M00 复验，再按依赖顺序完成 G02–G09；R01/R02 在资源已明确授权且就绪时也继续执行。阶段通过并保存新证据后立即选下一阶段，不停下来等我说继续。旧 G01 “完成后停止”的范围限制不适用于本总 Goal，但质量与安全约束仍有效。

每阶段完成内容/计算/源码/交互/反例/实际检查，更新 STATE 与进度，使用 python tools/program_plan.py next 选后续任务。基础简短，核心精讲；完整 Qwen/DeepSeek 和 SFT/RL 始终是主线。版本按最小接口与语义兼容，不全量重装训练栈，不把模拟或参考标成实测。

一个外部阻塞只阻塞相关阶段，继续其余依赖已满足的工作；必要测试失败不准跳过或减弱。G09 必须在最终当前源码上重跑全部软件阶段。只有软件和真实运行阶段均验证才可称 full_project_ready_for_review；软件完成但缺真实运行资源时记录 software_ready_runtime_blocked，不将总项目标完成，也不无限等待。

允许总目标列明的项目内多文件修改；保护无关改动，不 commit/push、不改仓库级设置、不公开部署。未获资源授权不启动 GPU/集群、不下载模型大文件。阶段报告必须是本轮真实日志，不能复用旧绿色记录。全部完成、无可执行工作、用户暂停或预算限制时保存 checkpoint 后停止，恢复时从账本接续。
```

`/goal` 查看原生目标；`/goal pause` 和 `/goal resume` 控制同线程。文件不会自动启动或保证后台执行；保留现有审批，不配置 bypass。恢复时不复制一个“忽略之前约束”的 prompt；重新核验当前 HEAD 和缺失证据后继续。
