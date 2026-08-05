---
type: knowledge
status: seed
created: 2026-08-05
updated: 2026-08-05
domains: [transformer]
aliases: [Transformer Engineering Overview]
tags: [model, architecture]
source: []
ai_generated: true
reviewed: false
---

# Transformer 工程总览

> [!warning]
> 本文是未审核的工程框架。具体实现、公式变体和数值行为应以目标模型及当前代码版本为准。

## 核心心智模型

Decoder-only Transformer 重复执行“通信 token 信息”和“逐 token 变换”两类操作：

- Attention 在序列维上混合信息。
- MLP/MoE 在每个 token 内变换通道。
- Residual 保持信息与梯度的高速通路。
- Normalization 控制每个子层看到的数值尺度。

工程分析时，不应只停留在模块名；至少同时回答：输入输出 shape、参数布局、计算 dtype、mask/position 语义、并行切分和运行阶段。

## 一层 Decoder Block

以前置归一化结构为例：

$$
X_1 = X + \operatorname{Attention}(\operatorname{Norm}(X)),
$$

$$
Y = X_1 + \operatorname{MLP}(\operatorname{Norm}(X_1)).
$$

若 $X \in \mathbb{R}^{B \times S \times H}$：

- $B$：micro-batch 中的序列数；packing 后它不一定等于原始样本数。
- $S$：本次 forward 的 token 轴；CP 或 sequence packing 会改变它的局部含义。
- $H$：hidden size。

### Self-Attention

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
$$

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

$M$ 不只是“causal mask”：真实系统还可能叠加 padding、packed-sequence 边界、sliding window 或 prefix 语义。两个 runtime 的权重相同但 mask 不同，输出仍会从第一层开始分叉。

### MLP 与 MoE

Dense MLP 对每个 token 使用同一组参数。MoE 先由 router 选择 expert，再发生 token dispatch、expert compute 和 combine。分析 MoE 时应额外记录：top-k、capacity/drop 策略、router dtype、负载均衡 loss、EP group 与 all-to-all 边界。

## 从数学到 Kernel

同一个数学模块可能经过以下实现层：

```text
model definition
  -> framework module
  -> parallel wrapper
  -> fused op / custom autograd
  -> compiler or kernel dispatch
  -> CUDA kernel and collective
```

“功能层相同”不等于“数值算法等价”。融合 kernel 可能改变归约顺序、累积 dtype、epsilon 位置、padding 方式或近似函数。跨实现对齐时应逐层验证，而不是根据模块名直接认定等价。

## 训练与推理的主要差异

| 维度 | 训练 | 推理 |
| --- | --- | --- |
| 序列执行 | 通常一次处理完整训练片段 | prefill 后逐步 decode |
| 状态 | 保存激活以 backward，或重计算 | 保存 KV cache |
| 批处理 | 数据批、micro-batch、pipeline schedule | continuous batching、请求动态加入退出 |
| 输出 | loss、梯度、统计量 | token、logits/log-prob、finish reason |
| 数值约束 | 关注梯度稳定和吞吐 | 关注采样语义、cache 与延迟 |

训练和推理实现可能使用不同 processor、position IDs、kernel 与 dtype。对齐方法见 [[20_Knowledge/Inference/推理系统与训练对齐|推理系统与训练对齐]]。

## 分析一个模块的检查表

1. 输入从哪里产生，是否经过 padding、packing 或重排？
2. 每个张量的 global/local shape、dtype 和 device 是什么？
3. 参数属于哪个模型实例、rank、stage 和版本？
4. 是否有 autocast、融合、编译或量化分支？其优先级是什么？
5. 模块内部是否发起 collective 或跨进程传输？
6. 输出在哪里首次被消费，是否改变语义或精度？
7. 现有证据是静态代码、日志、dump，还是端到端实验？

## 相关笔记

- [[20_Knowledge/Transformer/张量形状与计算约定|张量形状与计算约定]]
- [[20_Knowledge/Distributed/大模型并行训练|大模型并行训练]]
- [[20_Knowledge/Memory/显存分析方法|显存分析方法]]
- [[80_MOCs/Transformer 系统地图|Transformer 系统地图]]

## 待核验与扩展

- 按具体实现补齐 GQA/MLA、RoPE/M-RoPE、RMSNorm 和 SwiGLU 页面。
- 用实际 profiler trace 标注一层 Transformer 的 kernel 与 collective 时间线。
