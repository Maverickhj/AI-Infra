---
type: experiment
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# M00 接续自审

当前 HEAD 为 8a61e3b，进入阶段前工作区干净。远端快进只引入 PROGRAM-V1 计划和对应工具/测试，未重建页面，未修改 G01 历史 evidence。

本阶段行为验收与文件范围：保留 G01 的 13 子步骤、公式块与语义源码路由；重跑全部既有 Python/Chromium 与 GQA 反例；对完整缓存核验逐字源码；记录实际环境；建立本轮报告。仅新增 tools/program_gate.py 与本阶段记录，更新 PROGRAM-V1/STATE.json、PROGRESS.md。后续阶段各自记录范围。

已审读 web/data.ts 的状态归一化/语义片段路由、web/main.tsx 的整模入口与 SampleInspector、tests/browser/routes.spec.ts 和 G01 数值检查结果。head 非整数/越界、MLA 路径、公式及历史 P0 回归均继续由原断言执行，不修改数量来替代结果。

首次独立基线报告：runs/goal-g01/20261004T150509257648Z/report.json；全部七条命令通过。当前固定缓存 29 文件/54 片段完整核验通过。GQA 1872 个有限值、48 个 masked 位置对照通过，最大绝对误差 3.552713678800501e-15；四个故障变体产生非零差异。

运行环境是 minimind-megatron-bridge-dev-1，Node v22.22.0；已有系统 Python 3.12.3 / Torch 2.12.0a0+0291f960b6.nv26.04.48445190。Torch 仅导入读取版本，未触发 GPU 训练。项目测试使用 uv 的独立 Python -S，PATH 指向既有 Node，PLAYWRIGHT_BROWSERS_PATH 指向既有 Chromium 缓存。接下来的 program gate 会再次执行并保存命令/源码清单；此自审不代替运行报告。

限制：Final norm 专属片段仍待 G03；SampleInspector 当前仍是角色跨度示意，G02 必须补真实可演算的 authored token fixture；未得到 R01/R02 的一次性资源授权，未下载模型、未启动 GPU/rollout。
