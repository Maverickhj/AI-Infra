# 已建立：Transformer 全文因果链、Parallelism Boundary 与沉淀决策

## 1. Paper-level causal chain

用户已用自己的话建立以下路径：

- Previous constraint：RNN 的 sequence-internal recurrence 使当前 token information 依赖此前 state；convolution 通过 fixed receptive field 或多层路径传播远距离信息。
- Core replacement：Transformer Self-Attention 让同一 layer 内所有 positions 批量计算，并把任意 positions 之间的 maximum path length 降为 $O(1)$。
- New gaps：Single-head attention 会把多种 relations 压缩进同一 distribution；Multi-Head Attention 允许不同 heads 学习不同 matching / aggregation patterns。论文没有直接证据支持“head 数越多效果越好”。
- Position-wise capacity：Attention 主要负责跨 positions 信息传递；FFN 提供每个 position 上的 nonlinear transformation 与更丰富的 representation capacity。
- Architectural closure：Attention、FFN、Residual Connection、LayerNorm 与 Positional Encoding 共同构成可堆叠 architecture。
- Training closure：Adam / warmup schedule、length-aware token batching、dropout 与 label smoothing 共同构成可训练 system 的 recipe。

需要保留两点精化：

1. Residual Connection 的直接功能是提供 additive input path，使 sub-layer 可以学习 update；所有 sub-layer 保持 $d_{\mathrm{model}}$ 是 residual addition 的 shape constraint 与 architecture consequence。
2. “LayerNorm 稳定数值”是有用直觉，但正式表述仍要保留 normalization axis 与原论文 Post-LN boundary：

   $$
   \operatorname{LayerNorm}\left(x+\operatorname{Sublayer}(x)\right).
   $$

## 2. Cross-Attention final retrieval

用户选择跳过 Lesson 0012 的 Cross-Attention final retrieval。此前 Lesson 0003 已形成相关理解与记录，但本次没有完成 paper-level synthesis 复述，因此这一项的**最终检索状态**保留为 `[待核验]`，不把“跳过”写成“已答对”。

## 3. Parallelism boundary

用户已正确区分：Transformer 实现的是同一 layer 内不同 positions 之间 information transfer 的 parallel computation，但没有消除 autoregressive dependency。未知 target token 必须先生成，才能作为 prefix 的一部分参与下一 token 的计算。

因此：

$$
\text{layer-level position parallelism}
\not\Rightarrow
\text{parallel generation of unknown autoregressive tokens}.
$$

## 4. Knowledge promotion decision

用户确认需要全部 artifacts，并特别要求 Adam paper branch 与 *Attention Is All You Need* 建立 Obsidian 因果链。实际沉淀为：

- [[2017 - Attention Is All You Need]]
- [[2015 - Adam - A Method for Stochastic Optimization]]
- [[Transformer Architecture Causal Chain]]
- [[Transformer Parallelism Boundaries]]
- [[Adam Optimizer - Moments, Bias Correction, and Evidence Boundaries]]
- [[Reading ML Paper Results and Ablations]]

Adam 与 Transformer 的关系被记录为 training dependency：Transformer paper 采用 Adam moments / adaptive scaling，并组合自己的 hyperparameters 与 warmup schedule。两篇论文保持独立 evidence scope；Adam 的 theory / experiments 不被改写为 Transformer convergence proof。

## 5. Review state

课程第一轮 paper-level causal review 已完成；Cross-Attention final synthesis retrieval 仍为 `[待核验]`。新建的 formal source / knowledge notes 均标记为 `ai_generated: true`、`reviewed: false`，等待用户在 Vault 中继续审阅，而不自动视为 evergreen knowledge。
