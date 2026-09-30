---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Codex 首轮执行任务

读取 `AGENTS.md`、`PLAN.md`、`CURRICULUM.md`、`profiles/minimum-compatibility.md` 与三篇 `content/cases/*.md`，按需查 `research/FINDINGS.md` 和证据条目。本轮完成 P0/P1 的用户体验，不重新输出一份方案。

## 先检查，再复用

检查当前已有 lab 和旧实现，保留可用的构建、组件、Markdown/公式能力。不要重置用户工作区。没有旧工程时，在本目录建立最小静态前端。`source.lock.json` 是参考源码清单而非运行安装锁；本轮只读学习页不要求安装 Bridge/CUDA。已有环境不同于参考版不构成阻塞：后续训练按所选 profile 的最小接口/语义合约验证并记录实际来源。未找到源码时，可保留明确的远程固定源码链接，不因此退回 mock-only 页面。

## 学习深度要求

读者是“有一定基础的初学者”。将 tensor、Linear、autograd 和 Transformer 基本说明合并为一页可跳过速览，基础篇幅目标 10%–15%。默认直接进入完整 Qwen3 的核心计算与实现，不安排长篇基础前置章节。

Qwen/DeepSeek 结构差异、SFT mask/shift/归一化、RL 四类 logprob/更新/refit、并行张量布局仍须精讲。不得用“精简”删除关键公式、符号表、配置条件和反例。

## 本轮必须交付

1. 首页默认 Qwen3-0.6B 完整模型，而不是 Linear/MLP 首页。用户能走通 messages→tokens/targets/mask→embedding→28层 decoder→final norm/head→loss→backward/update 的完整学习流程；不可观测步骤明确标注。
2. Model Atlas 的 Qwen2.5、Qwen3 Dense、Qwen3 MoE、DeepSeek-V3 切换真正改变结构与字段。DeepSeek 有 dense/MoE 层分布、MLA down/up、shared experts；R1-Distill 不能作为其代用品。
3. SFT sample inspector 支持 assistant/last_turn/full 语义说明与 mask 对照。没有 tokenizer trace 时展示角色跨度示意，不编造 token IDs。
4. RL Cycle 是完整可读课程，显示 rollout/reward/logprobs/advantage/update/refit；四类 logprob 独立，真实运行状态初始为 not_run。
5. 当前 step 可打开对应固定源码和审阅说明。源码不可用时显示原因，不显示编造代码。全局 model/scenario/layer/step/sourceLane 状态一致。
6. 至少三条浏览器用户路线测试：Qwen3完整流程；Qwen2.5/Qwen3对比；DeepSeek切换及RL四类logprob。检查窄屏、键盘、公式、源码侧栏和直接访问路由；补验基础速览可跳过且核心正文仍完整。

复用 `content/models.json` 与 `tools/build_case_data.py` 的推导数据，不能把它们标成真实模型执行。先运行 `python tools/build_case_data.py` 与本包 tests，再实现前端。派生大文件在本地生成，不提交到知识库。

## 非本轮目标

不训练大模型；不构建远程GPU服务；不新增账号/数据库/AI助教；不做通用全仓调用图；不默认运行FP8/128K/异步RL。SFT/RL真实采集分别按 tasks/02、03 后续执行，页面先如实显示未运行。

## 完成报告

给出真实本地启动命令、完成的用户路线、截图/交互测试结果、source/runtime状态和剩余限制。只做了网站壳或只有MLP不能标记完成。已有实现的具体问题只有看过代码/日志后才能归因。
