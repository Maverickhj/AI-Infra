---
type: knowledge
status: seed
created: 2026-08-25
updated: 2026-08-25
domains:
  - llm
  - transformer
aliases:
  - Transformer causal chain
source:
  - "[[2017 - Attention Is All You Need]]"
ai_generated: true
reviewed: false
---

# Transformer Architecture Causal Chain

> [!warning]
> 本文包含 AI 整理的可迁移结论，尚未完成人工核验。默认讨论原始 Encoder–Decoder、Post-LN Transformer；现代 variants 需要单独注明。

## Causal chain

### 1. Remove the sequence-internal recurrence

RNN 的 state transition 使 $h_t$ 等待 $h_{t-1}$。Self-Attention 改为让同一 layer 的 positions 通过 batched pairwise scores 交换信息，从而移除 layer 内 recurrence dependency。

Convolution 也可并行，但固定 kernel width 使远距离信息通常需要更多 layers；full self-attention 在一个 layer 内建立任意 position pair 的直接路径。

### 2. Restore capacities that recurrence or convolution implicitly carried

| Mechanism | Primary responsibility | Contract / boundary |
|---|---|---|
| Scaled Dot-Product Attention | 跨 positions 的 content-dependent aggregation | 不自行提供 order；full attention 的 score matrix 为 quadratic size |
| Multi-Head Attention | 在多组 learned projections 中同时学习 matching / aggregation patterns | Heads 的 weights 不同，但 subspaces 不保证正交或不重合；head count 与 quality 不存在论文支持的单调关系 |
| Position-wise FFN | 为每个 token 增加 nonlinear representation capacity | 每个 position 使用同一组 parameters；同一 FFN 内不进行跨 position 通信 |
| Residual Connection | 提供 additive input path，让 sub-layer 学习 update | Residual addition 要求 input/output shapes 一致；“统一维度”是 constraint 与 architecture consequence，不是 residual 的独立计算功能 |
| LayerNorm | 控制每个 token feature vector 的 statistics，并保留 learned gain / bias | 原始 Transformer 使用 Post-LN；$F(x)=0$ 时 output 仍是 $\operatorname{LayerNorm}(x)$ |
| Positional Encoding | 向 representation 注入 order signal | 不负责 causal visibility；mask 决定哪些 keys 对 query 可见 |

## Encoder–Decoder task closure

Conditional generation 要建模：

$$
p(y_t\mid y_{<t},x).
$$

- Masked Decoder self-attention 提供 target prefix $y_{<t}$。
- Encoder output 提供 source memory $x$ 的 contextual representation。
- Cross-Attention 用当前 Decoder state 产生 queries，以 Encoder memory 产生 keys 与 values，把 target-side need 对齐到 source information。

Source positions 彼此不是 future labels，因此 cross-attention 不需要 source causal mask；variable-length batches 仍需要 padding mask，避免读取 padding positions。

> [!note]
> 上述 Cross-Attention 事实有原论文与先前课程记录支持，但 Lesson 0012 的 final synthesis retrieval 被跳过，学习闭环状态仍标记为 `[待核验]`。

## Stackability is a system property

一个 Transformer layer 的 closure 不是单一组件完成的：

$$
\text{communication}
+
\text{position-wise transformation}
+
\text{shape-compatible residual path}
+
\text{normalization}
+
\text{order signal}.
$$

把某个组件的 responsibility 扩大到整个 block，会导致常见混淆，例如让 FFN 承担 token mixing、让 Positional Encoding 承担 causal masking，或把 Residual Connection 误写成保证 strict identity mapping。

## Related notes

- [[Transformer Parallelism Boundaries]]
- [[Adam Optimizer - Moments, Bias Correction, and Evidence Boundaries]]
- [[Reading ML Paper Results and Ablations]]
