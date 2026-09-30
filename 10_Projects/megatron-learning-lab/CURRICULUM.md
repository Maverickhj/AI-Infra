---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# 学习深度与课程编排

## 1. 目标读者：有一定基础的初学者

默认读者会基本 Python/PyTorch，能读懂 tensor shape、矩阵乘法、forward/backward、loss/optimizer，知道 Transformer 和 SFT/RL 的基本用途；尚未系统掌握 Megatron 的整模实现、分布式张量布局、源码分支与训练数据合约。

不假设读者已经懂得 GQA/MLA 的全部计算、MoE 路由、梯度归约或 RL 训推对齐。遇到这些核心内容仍须逐步解释，不能用“你应该知道”跳过关键推导。

## 2. 深度分层

| 层次 | 内容 | 编写要求 |
|---|---|---|
| 基础速览 | tensor/Linear/autograd/embedding 的基本用途；Transformer、SFT、RL 总览 | 一句定义、必要 shape、整模位置；不另开长篇教程 |
| 核心精讲 | Qwen/DeepSeek 实际计算与参数映射；SFT mask/shift/归一化/更新；RL logprob/advantage/refit；并行布局与通信 | 问题 → 数学与 shape → 源码与分支 → 样本/张量演算 → 反例 → 验证 |
| 按需查阅 | 数学基础补充、API 参数大全、安装细节、长上下文/FP8/kernel 优化 | 折叠说明或附录，不阻挡默认学习路线 |

“压缩基础”不是压缩所有细节。next-token 的概念可以概括，但标签到底在哪一层移位、mask 如何随目标对齐，必须精讲。backward 的定义可以概括，但 residual/共享权重/分布式归约造成的梯度路径必须展开。

## 3. 默认学习顺序

| 单元 | 重点 | 深度与出口 |
|---|---|---|
| 00 定位与约定 | 一张整模图、Bridge/Core/backend 分工、符号与来源标签 | 一页速览；最多两个基础小节，可直接跳过 |
| 01 Qwen 整模 | Qwen3-0.6B attention、GQA/QK norm/RoPE、SwiGLU、head；对照 Qwen2.5 | 核心精讲，能解释每步 shape 与真实实现 |
| 02 SFT 全链路 | checkpoint、template、shift、loss mask、sum/count、backward/update、resume | 核心精讲，以同一条训练样本串联 |
| 03 RL 全链路 | 固定 trajectory、四类 logprob、advantage/ratio/clip、权重版本与 refit；PPO value 分支 | 核心精讲，不重讲泛化的 RL 入门史 |
| 04 并行执行 | 在同一 Qwen/SFT/RL 案例中引入 TP/DP/PP/SP/CP | 核心精讲，不重新从脱离模型的 MLP demo 起步 |
| 05 MoE 与 MLA | Qwen3 MoE、DeepSeek-V2-Lite/V3、EP、shared expert、训练/decode 区分 | 核心精讲，按差异深入，不重复公共骨架 |
| 06 进阶观察 | LoRA、packing/长序列、overlap、精度与性能 | 按当前实验需要展开 |

这是阅读顺序，不是导航隔离：Model Atlas 从首版即含 DeepSeek；SFT/RL 全周期从首页可见。现有三篇 case 可继续作为内容源，不要求为了编号机械拆出六套重复正文。

## 4. 篇幅与页面约束

基础铺垫在默认主线正文中的篇幅目标为 10%–15%，不是新增内容的配额。基础单元最多一页；同一个概念只定义一次，后续链接或一句提醒。符号表、源码原文、可折叠参考资料不计入这一篇幅目标。

首屏默认进入 Qwen3 的整模图及当前步骤，不强制先完成基础练习、安装训练环境或阅读整篇研究报告。每个核心步骤通常围绕一个具体问题展开，避免同时摊开全部配置开关。

基础速览可跳过且不影响导航、进度或验收；关键假设、符号定义、shape、loss 语义、适用分支和错误示例不能折叠到找不到的位置。

## 5. 核心精讲的完成标准

每个核心单元至少能回答：输入与输出是什么；为何这样计算；固定参考源码在哪里；当前运行实现有何差别；改变一项配置会怎样；通过哪个实验区分正确和错误解释。

实现报告分别说明“课程可读、交互可用、源码可追溯、数值已验证”的状态。不能用篇幅、动画数量、测试总数替代核心解释质量。
