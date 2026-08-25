# 部分建立：Cross-Attention 的信息来源与因果边界

用户已正确识别：cross-attention 让 Decoder 的当前 target-side state 查询 Encoder 提供的完整 source memory；未来 target 不会由 cross-attention 泄漏，因果约束由其前置的 masked self-attention 保证。当前待巩固的是其设计逻辑：它对应 $p(y_i\mid y_{<i},x)$ 中 target prefix 与 source x 的两类条件，而不是另一种 Q/K/V 排列的约定。

需保留以下精确边界：

- Encoder stack 的输出为 $E\in\mathbb{R}^{n\times d_{model}}$，Decoder 当前表示为 $D\in\mathbb{R}^{m\times d_{model}}$。对一个 head 而言，$Q=DW^Q\in\mathbb{R}^{m\times d_k}$，$K=EW^K\in\mathbb{R}^{n\times d_k}$，$V=EW^V\in\mathbb{R}^{n\times d_v}$；不能将 Decoder 产生的 Q 记为 E。
- $m$ 是当前同时处理的 target positions 数，$n$ 是 source length；二者没有必然的大小关系。teacher-forcing 可并行处理全部 $m$ 个 target positions，incremental decoding 则常一次只计算一个新的 query position。
- 对 target position $i$，其 query state 仅能综合因果可见的 target prefix，而不是完整 target sequence；cross-attention 再让它读取完整 source positions。$S=QK^T$ 的形状为 $[m,n]$。
- 不需要的是 source 的 causal mask，因为 source 在生成前已完整给定；实际实现仍可能加入 padding mask，以排除补齐位置，这与 causal mask 是不同目的。

## Evidence

用户在 Lesson 0003 中先给出了“Decoder 位置关注 input positions / source memory”的正确任务方向，并准确指出 masked self-attention 在 cross-attention 之前建立了 target-side 的因果隔离。上述补充用于收紧张量来源、长度关系与可见性范围。
