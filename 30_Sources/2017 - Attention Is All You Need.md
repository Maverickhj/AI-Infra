---
type: paper
status: seed
created: 2026-08-25
updated: 2026-08-25
domains:
  - llm
  - transformer
  - sequence-modeling
aliases:
  - Attention Is All You Need
source:
  - https://arxiv.org/abs/1706.03762
ai_generated: true
reviewed: false
---

# 2017 - Attention Is All You Need

> [!warning]
> 本文是基于原论文与课程检索记录整理的 AI-assisted source note，尚未完成人工核验。论文事实以原文为准；推论与可迁移结论分别放入 linked knowledge notes。

## Source identity

- Authors: Ashish Vaswani et al.
- Venue: NeurIPS 2017
- Paper scope: sequence transduction 的 Encoder–Decoder Transformer，而不是现代 decoder-only LLM 的完整设计。
- Primary source: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## Thesis

论文的核心不是首次引入 attention，而是用 attention-based layers 替代 recurrent 与 convolutional sequence-processing layers。在同一 layer 内，所有 positions 的 representation 可以批量计算，任意两个 positions 之间的 maximum path length 降为 $O(1)$。

这个替换没有消除所有 sequential dependency：autoregressive inference 仍按

$$
p(y\mid x)=\prod_{t=1}^{m}p(y_t\mid y_{<t},x)
$$

逐 token 生成。

## Causal chain

### Previous constraint

- RNN 的 $h_t=f(h_{t-1},x_t)$ 在一个 sequence 内形成 recurrence dependency。
- Convolution 依靠 fixed receptive field；远距离信息通常需要多层传播。

### Core replacement

Self-Attention 通过 pairwise query–key scores 在一层内组织跨 position 信息传递：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

代价是在 full self-attention 下，per-layer complexity 为 $O(n^2d)$，attention-score compute 与 storage 随 sequence length 呈 quadratic growth。

### New gaps and architectural closure

| Gap | Mechanism | Responsibility | Boundary |
|---|---|---|---|
| Single weighted average 容易把多种 relations 压入同一 distribution | Multi-Head Attention | 学习多组 projections、matching 与 aggregation patterns | 不保证 heads 正交、互不重叠或具有固定语义；论文不支持“heads 越多越好” |
| Attention 主要负责跨 positions 通信 | Position-wise FFN | 对每个 position 独立执行 nonlinear transformation | 不负责 token-to-token communication |
| Deep stacking 需要 shape-compatible interface 和 input path | Residual Connection | 保留 additive input path，让 sub-layer 学习 update | $d_{\mathrm{model}}$ 一致是 addition 的 shape constraint；在 Post-LN 中不形成严格 identity output |
| Feature statistics 需要受控 | LayerNorm | 沿每个 token 的 feature axis 归一化 | 原论文 block 是 $\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$ |
| Attention 本身对 input permutation equivariant | Positional Encoding | 向 token representation 注入 order information | 不控制 information visibility；causal mask 承担该职责 |

### Task closure

- Encoder self-attention 构造 source memory。
- Decoder masked self-attention 表示已知 target prefix。
- Encoder–Decoder Attention 用 Decoder state 产生 queries，并以 Encoder output 作为 keys 与 values，使 prediction 同时依赖 $y_{<t}$ 与 $x$。
- Output projection 与 vocabulary-axis softmax 产生 next-token distribution。

### Training closure

- Adam optimizer：$\beta_1=0.9$、$\beta_2=0.98$、$\epsilon=10^{-9}$。
- Learning-rate schedule：

  $$
  \operatorname{lrate}
  =d_{\mathrm{model}}^{-1/2}
  \min\left(
  \operatorname{step}^{-1/2},
  \operatorname{step}\cdot\operatorname{warmup\_steps}^{-3/2}
  \right),
  $$

  其中 $\operatorname{warmup\_steps}=4000$。
- Approximate-length batching 与 token budget 减少 padding waste。
- Dropout 与 label smoothing 用于 regularization。

## Evidence map

| Evidence | Direct support | Boundary |
|---|---|---|
| Table 1 | 给定 algebraic cost model 下的 per-layer complexity、minimum sequential operations 与 maximum path length | 不是现代 kernel runtime、memory traffic 或 serving latency measurement |
| Table 2 | 完整 translation system 在 WMT test sets 上的 reported result | 不能隔离 architecture-only effect；score 还依赖 checkpoint averaging、beam search 与 metric setup |
| Table 3 | EN–DE newstest2013 dev 上 tested configurations 的 local comparisons | 不是 universal hyperparameter rule；多变量 row 不能做 single-factor attribution |

Source inconsistency 必须保留：Abstract 与 Table 2 报告 Transformer big 在 EN–FR 上为 $41.8$ BLEU，而 §6.1 narrative 写为 $41.0$ BLEU。

## Adam branch: causal link, not evidence substitution

Transformer 的 architecture 替换了 sequence computation graph，但仍需要 optimizer 与 schedule 把 gradient 转化为 parameter update。论文因此**采用** Adam，并改变了 Adam paper 的 default $\beta_2$ 与 $\epsilon$，再叠加自定义 warmup / inverse-square-root schedule。

这条链应读作：

$$
\text{Transformer architecture}
\rightarrow
\text{training dynamics problem}
\rightarrow
\text{Adam moments and adaptive scaling}
\rightarrow
\text{Transformer-specific schedule}.
$$

它不表示 Transformer 发明了 Adam，也不表示 Adam 原论文的 experiments 或 online-convex regret result 证明了 Transformer training 的收敛性。Adam 的机制与理论边界见 [[2015 - Adam - A Method for Stochastic Optimization]] 和 [[Adam Optimizer - Moments, Bias Correction, and Evidence Boundaries]]。

## Derived notes

- [[Transformer Architecture Causal Chain]]
- [[Transformer Parallelism Boundaries]]
- [[Reading ML Paper Results and Ablations]]
