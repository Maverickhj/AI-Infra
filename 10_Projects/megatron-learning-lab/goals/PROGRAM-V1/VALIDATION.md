---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# 连续执行的验收与报告协议

## 共用回归

M00 复用现有 `goal_gate.py baseline` 和 `npm run test:gqa`。后续各阶段至少运行：

```bash
python tools/build_case_data.py
python -m unittest discover -s tests -v
python tools/verify_handoff.py
python tools/verify_source_snippets.py
npm run build
npm run test:e2e -- --reporter=json
git diff --check
```

沿用项目所需 Python 解释器/uv 与浏览器缓存，命令中记录真正调用的 argv 和环境差异。以上列表不是安装命令。新增数值测试必须包含计划中的阶段入口；通过计划检查工具不是执行上述测试。

构建和浏览器不能复用不明来源的旧 dev server；执行 JSON 测试报告必须来自当前子进程。检查 expected>0、unexpected/skipped/flaky=0，且当前阶段 marker 与旧回归 marker 真实出现。Python/数值测试也不接受零用例、全部 skip 或只验证输出文件存在。

每个新增核心算法至少一个独立参考、一个可手算例子、一个错误实现反例。双实现共享同一错误 helper 不是独立对照。容差在运行前固定，BF16/GPU 容差不能照搬 CPU float64；失败后修改容差必须解释数学/误差依据并保留原结果，不能只求通过。

## 阶段报告

文件建议放入 `runs/program-v1/<stage>/<run-id>/report.json`，只在阶段完成后更新 STATE.json 的 report 路径与 SHA256。每阶段自审摘要放到 PROGRESS.md 或小型项目内报告，原始 tensor、模型、长日志不入 Git。

报告至少包含下面结构（这是结构示意，不是通过记录）：

```json
{
  "program_id": "PROGRAM-V1",
  "stage_id": "G02",
  "status": "passed",
  "source_fingerprint": "<运行时计算的 SHA256>",
  "git_head": "<实际 HEAD>",
  "started_at": "<ISO 时间>",
  "completed_at": "<ISO 时间>",
  "checks": [{
    "id": "stage-specific", "status": "passed",
    "command": ["npm", "run", "test:sft-data"],
    "returncode": 0,
    "log": "runs/program-v1/G02/<run-id>/numeric.log",
    "log_sha256": "<实际 SHA256>"
  }],
  "artifacts": [{"path": "<实际产物>", "sha256": "<实际 SHA256>"}],
  "limitations": [], "reviewed": false
}
```

`checks` 必须覆盖 plan.json 中的所有 required_checks；每条日志与 artifact 必须存在且 hash 匹配。`source_fingerprint` 为本次实际实现、课程、测试、source lock、有效 plan 的文件清单内容 hash；可复用旧 snapshot 思路，但排除生成产物、进度和原始 runs。报告要另存参与指纹的文件清单，不凭手写一串 hash 声称可复现。

`baseline`：现有完整回归；`stage-specific`：阶段数值/语义检查；`browser`：相关用户路线；`source`：新增/变更固定原文及语义路由；`review`：数学/源码/前端自审的检查日志；R 阶段增加 runtime-sft/runtime-rl；G09 增加 integrated-regression、production-smoke。

review 日志由真实审读形成，记录具体文件/步骤/已查问题；机器只能检查其存在，不能代替审读。报告中被引用的代码产物后来改变，应重验相关阶段。历史报告仍可留存，G09 不使用历史报告代替最终代码回归。

## STATE 与证据保全

阶段状态：pending、in_progress、validated、revalidate、failed、blocked_external。validated 必须有报告/hash；blocked_external 必须有具体 blocker 和解除条件。没有 `skipped == done` 或仅靠 status 字段成为完成的路径。

`tools/program_plan.py check/status/next` 是只读计划与证据结构检查。它校验 DAG、状态、依赖和引用文件 hash，不运行报告 command、不证明日志不是伪造，也不审查教学正确性。报告缺失/内容变化时 next 优先要求重验该阶段，不将其成功继承。

资源门槛不是质量门槛的豁免。对尚无资源的 R01/R02，账本记录 blocked_external，不能通过空壳报告满足依赖。

## 完成条件与恢复

G09 最终报告须同时列出 validated、blocked、未执行、来源类型与源码快照。R01/R02 没跑时只输出 software_ready_runtime_blocked，不将 PROGRAM 原生 Goal 标 achieved。剩余没有可执行工作时停止并保存最小解阻信息，不循环消耗预算。

在同一线程恢复时，先检查真实 `/goal` 状态、最新 HEAD 和账本，再重试已满足解阻条件的阶段。新工作区没有 ignored runs 证据时必须补取或重跑；不要为了通过检查把缺失的证据路径删掉。
