---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# megatron-learning-lab

**整模驱动、源码可追溯、连接真实 SFT/RL 的 Megatron 学习项目。**

方案版本：2.1，Review 修订日期：2026-09-30。源码研究沿用此前 2026-09-30 的记录；本轮不重新声称核验上游最新版本。替代旧方案的 MLP-first 路线，不替换或删除用户已有工程。

## 这份交付包含什么

这是新的研究与实施交付包，不是已实现的网站，也不是完成 GPU 验证的训练工程。它包含保留的参考源码快照、27 个源码证据条目、6 个模型档案、三篇整模/训练课程正文、页面验收要求、分阶段 Codex 指令，以及可运行的元数据生成/校验工具。没有附带模型权重或第三方源码。

首先阅读 [实施方案](PLAN.md)、[学习深度](CURRICULUM.md) 与 [最小兼容策略](profiles/minimum-compatibility.md)，然后执行 [Codex 首轮任务](CODEX_START.md)。源码问题按需查 [研究结论](research/FINDINGS.md) 与 [REFERENCES](research/REFERENCES.md)。`source.lock.json` 固定阅读证据，不是运行环境的安装锁。

## 本轮 Review 决策

目标读者为“有一定基础的初学者”：基础速览一页可跳过，核心实现与 SFT/RL/并行语义精讲。压缩的是前期铺垫和重复，保留独立符号表、必要推导与反例。

环境优先复用、按场景最小满足：只要求 learning-lab 所需接口和数据/计算语义一致，不强制逐包复制上游 lock。版本不同但合约测试通过可接受；版本相同但语义不符应拒绝。真实行为测试尚未执行的能力保持未核验。

## 最重要的路线变化

旧方案从 Linear/MLP 与通用可视化组件起步，将完整模型、SFT 和 RL 推迟到后续。这会让“工具是否建好”挤占“用户是否学会整模训练”的目标。

新版从一个真实 checkpoint、一条训练样本、一次模型计算/参数更新进入：

`Qwen3-0.6B 整模 → Qwen2.5 对照 → SFT 样本/更新 → 同步 RL 闭环 → Qwen3 MoE / DeepSeek MLA+MoE → 并行与性能深挖`。

MLP、collective、rank 图是整模流程中的局部解释，不再单独成为第一版终点。

## 先完成的用户体验

用户打开首页选择 Qwen3-0.6B，可从一条 messages 样本一直走到 embedding、全部 decoder 层、logits、loss、backward 和 optimizer。切换 Qwen2.5 时，QKV bias、QK norm、head shape 真正改变。切换 DeepSeek 时，attention 变为 MLA，层内出现 dense/MoE 与 shared expert 的真实差别。

所有数值和图形都显示来源类型：`derived`、`reference` 或 `observed`。没有真实运行时，不显示 GPU 耗时、loss 曲线、热力图或“训练成功”。

## 运行本包的本地检查

```bash
python tools/build_case_data.py
python -m unittest discover -s tests -v
python tools/verify_handoff.py
```

这些命令只校验课程元数据和工具，不训练模型、不访问网络、不检查 GPU。结果见 `validation-report.json`。真正的 Bridge/NeMo RL 测试仍须在对应环境执行。

## 给 Codex

本项目工作目录为 `AI-Infra/10_Projects/megatron-learning-lab/`；在该目录执行。它独立于已有 MiniMind-Megatron-Bridge 项目，不搬移其他笔记。入口为 [_Project - megatron-learning-lab.md](_Project%20-%20megatron-learning-lab.md)。

在已有工程中复用时发送：

> 读取 AGENTS.md、PLAN.md、CURRICULUM.md、profiles/minimum-compatibility.md 和 CODEX_START.md，按需查研究证据。只执行 CODEX_START.md 中的首轮范围。优先做完整模型学习体验，复用已有可用代码，不重新搭建通用课程平台。区分已查证源码、推导数据与实际运行。不得以单独 MLP、占位页面或 mock 训练曲线交付。

未确认的 HF revision、实际使用的运行镜像 digest、硬件能力和 GPU 实验状态必须保留为未验证，不填造数据。未使用的可选组件标记不适用，不成为环境门槛。生成的 model-cases、运行日志、权重、trace 和 caches 不入库，保留生成工具与小型摘要。

## 源码锚点与实验结果格式

`tools/audit_sources.py` 可只读检查显式指定的本地仓库，不访问网络或修改worktree。示例和实验数据字段见 [Trace 合约](profiles/trace-contract.md)。本次只对临时Git测试仓库验证了该工具行为，未在本地完整Bridge/Core仓库执行该脚本；源码调研证据来自已固定版本的远程读取。
