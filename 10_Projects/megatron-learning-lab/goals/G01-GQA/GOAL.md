---
type: project
status: draft
created: 2026-10-02
updated: 2026-10-02
ai_generated: true
reviewed: false
---

# G01：从整模导览到可验证的 GQA 源码精讲

审查基线：`ed606ee657097431e674785653da77dc6de19306`。此 SHA 用于记录起点，不要求后续 HEAD 永远相等。若已有后续提交，先检查变化并保留已完成内容。

## 1. 完成后的用户体验

在现有 Qwen3-0.6B 整模漫游里选中一层，用户能按顺序查看 attention 输入、RMSNorm、QKV 与分组拆分、Q/K norm、RoPE、QK 打分、缩放、causal mask、softmax、加权 V、合并 heads、输出投影及残差。选择 query token/head 后，图、公式、维度、小张量演算与源码入口同步变化。

完整模型与当前层始终可见。Qwen2.5 对照明确 QKV bias / QK norm 差别；不把局部 GQA 演算重新变成孤立 MLP 首页。

## 2. 读者与内容

面向“有一定基础的初学者”。不重复讲 Python、矩阵乘法、autograd 基础。每个核心子步骤必须说明输入输出 shape、数学操作、配置条件、参考源码或后端边界、一个验证办法。保留独立数学符号表。

新增或补充 `content/cases/04_gqa_source_walkthrough.md`，接入实际页面，而不是写出无人引用的文档。原三篇课程不机械复制。源码阅读原文和教学演算分开；不得把本地朴素 attention 当成实际 TE/FlashAttention kernel。

## 3. 两个明确区分的视图

**真实配置视图**：读取已有 Qwen 档案。Qwen3-0.6B 的 H=1024、n_q=16、n_kv=8、d=128 以及 Q 宽2048保留；不下载 checkpoint、不创建真实权重结果。

**教学数值视图**：使用自造、确定性的缩小 fixture，`B=1,S=4,H=8,n_q=4,n_kv=2,d=4`。它有意保留 `H != n_q*d`，输出投影为16→8；不称为真实 Qwen3 激活、checkpoint 或 SFT。fixture、说明和导出均标 `reference`、`architecture_origin=authored_scaled_gqa`、`weights_origin=authored_fixture`；生产模型执行状态仍是 not_run。

教学 fixture 明确每个权重、epsilon、RoPE theta、position IDs、旋转布局、mask 约定。Qwen3-style 的 head norm 在 RoPE 前；Qwen2.5-style 对照关闭 QK norm并显式使用QKV bias，仍是缩小教学结构。RoPE 布局须与选定参考源码一致，不默认为相邻维旋转。不要用公式故事代替实际数组计算。

## 4. 数值验证

使用独立的 Python 标准库参考实现与 TypeScript 生产计算模块。Python 不调用前端输出或复制其结果；前端页面必须调用被数值测试覆盖的同一模块。无需安装 Torch、CUDA 或 Bridge。新增 `npm run test:gqa`，只运行本地数值测试，不隐式安装依赖。

固定容差：本微型 float64 教学用例 `abs_error <= 1e-10 + 1e-10*abs(reference)`。比较 Q/K/V、norm、RoPE、score、softmax、head output、projection/residual 等有限中间值；mask 的无穷值单独按位置判断。不能将该容差用于未运行的 BF16/GPU parity，也不能在失败后直接放宽。

必须测试：因果边界（改变未来 token 不影响更早输出）；允许位置的 softmax 行和为1且屏蔽位置为0；每个 query head使用正确 KV group；故意漏缩放和选错 KV group 的反例确实与正确结果不同；输出投影及残差维度正确；非整数/越界head或token被拒绝；NaN/空输入不能默默产生成功结果。全 mask 行在本教学实现中报清晰错误，不能声称所有 Megatron backend 都如此。

## 5. 源码定位

复用 `sourceExcerptId` 与稳定 excerpt ID，扩展 operator/substep 维度。切换 model/layer/head/token/substep 后不会显示旧默认片段。真实实现、调用入口、数学等价参考、缺少匹配片段四种状态分开。

QKV/grouped split、QK norm、RoPE 调用、core attention 后端入口、output projection和残差应有相关固定源码证据。必要时只补少量相关文件到现有证据清单，按固定 commit 校验 Git blob 与片段；不要抓取全仓/全量模型。对 TE 内部无可读摘录的部分定位到真实调用边界，并解释概念图不意味着物化完整 score 矩阵。

Final norm 暂缺专属片段可保持明确缺口；但本次新增 GQA 关键步骤不能全部用“待补”通过验收。

## 6. 执行范围与权限

允许对本项目 `web/`、`content/`、`tests/`、`tools/`、必要 `research/`、本目标文件和必要 package script 做有明确目的的多文件修改；保留无关工作区改动。先记录准确文件范围，不因根 AGENTS.md 的默认五文件限制而拆出不一致交付，也不扩散到其他项目。

禁止修改仓库级 AGENTS/设置、批量升级依赖、重新搭脚手架、删除现有测试或弱化断言、修改教学目标以求绿色；不创建远程服务、不发起 GPU/集群任务、不下载权重或大型数据、不公开 Sites、不 commit/push。发现本地依赖缺失，先报告必要项；联网安装、系统包或配置修改沿当前审批策略进行，不能绕过。

公开源码的少量固定版本读取及显式 cache 校验在本目标内。网络不可用时可用已有缓存；确无合法源码证据则记录阻塞，不伪造原文。未来的 SFT/RL/OPD、MoE/MLA 新课程不属于本目标。

## 7. 分段推进

A. 重跑当前基线：python tests、摘录校验、build、Chromium。确认四条新增回归测试真实执行。不要引用2026-09-30报告替代本次结果。

B. 先实现独立参考和反例，再实现可复用 TS 计算；确认数值之后接 UI。

C. 完成 GQA 子步骤、联动源码、数学符号、课程挂接及窄屏/深链接。

D. 跑完整门槛、自审、记录新证据。每轮更新 PROGRESS.md：本轮变化、实际验证、下一项、阻塞，不记隐私或完整思维链。

同一失败允许至多连续三次没有新增证据的重试；重复前说明新的可检验假设。超出时记录 blocked，停止该路径。可继续独立可行项，但不能将阻塞测试标通过，也不能开启下一目标转移注意力。

## 8. 完成与停止

以 acceptance.json 为准，不用章节数/代码量/截图存在代替验收。先填 evidence.json 的每项真实路径与 SHA256，再运行 `python tools/goal_gate.py final`。必须产生当前源码指纹的 `ready_for_review` 报告，并用 `verify` 复验该报告仍有效。REVIEW.md 要逐项说明数学、源码、界面核对与残余限制，ai_generated=true、reviewed=false。

所有门槛满足后，当前 Goal 完成为“G01 已实现并等待用户审阅”，停在此处；不自动做P2/P3。依赖、网络、权限、测试或预算阻塞时记录进度、已尝试路径、日志、最小解阻条件，报告 blocked 而不是 complete。不能只输出一个计划后结束目标，也不能无限循环。