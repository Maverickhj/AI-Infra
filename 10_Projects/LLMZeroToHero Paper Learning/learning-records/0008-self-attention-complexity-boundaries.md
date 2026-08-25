# 已建立：Self-Attention 的三项评价指标与局部化 Trade-off

用户已正确建立以下理解：

- 论文使用 per-layer complexity、minimum sequential operations 与 maximum path length 三个不同维度比较 sequence layers。
- full self-attention 的 per-layer complexity 为 $O(n^2d)$，recurrent layer 为 $O(nd^2)$；比较二者并约去共同因子 $nd$ 后，得到 self-attention 主导项更小的条件是 $n<d$。
- self-attention layer 的 $O(1)$ minimum sequential operations 不会消除 autoregressive inference 的 token-to-token condition：只有生成 $y_t$ 后，才能把它加入 prefix 并计算依赖它的 $y_{t+1}$。
- restricted self-attention 让每个 query 只连接 neighborhood 中的 $r$ 个 positions，把 per-layer complexity 降为 $O(rnd)$，代价是 maximum path length 增加为 $O(n/r)$。

需保留以下精确边界：

- 三项指标不是“计算量”的三个名称：
  - per-layer complexity 衡量 asymptotic arithmetic work；
  - minimum sequential operations 衡量 layer computation graph 的可并行化边界；
  - maximum path length 衡量 information / gradient 在 positions 之间传播所需的最长 computational-graph path。
- $n<d$ 来自论文 Table 1 主导项的 algebraic comparison：

$$
\begin{aligned}
n^2d&<nd^2\\
n&<d.
\end{aligned}
$$

  它不是对实际 wall-clock performance 的无条件保证，也不包含 constants、kernel efficiency、memory traffic、batching、communication 或 quadratic attention-score memory。
- “未出现的位置不能加入并行 attention”只适用于 autoregressive inference 中尚未生成的 token value。Training 时，完整 target sequence 已作为数据给定；通过 shifted target inputs 与 causal mask，所有 target positions 仍可并行执行 layer computation，同时保持每个 position 的因果可见性。
- 更准确的 inference data dependency 是：

$$
y_t
\longrightarrow
E(y_t)
\longrightarrow
\text{prefix state for predicting }y_{t+1}.
$$

  在生成 $y_t$ 前，模型缺少下一步所需的实际 token choice 及其 embedding，而不只是缺少一个可以预先分配的 position index。
- restricted self-attention 中，$r$ 是每个 query 可见的 neighborhood size。它减少的是每个 query 参与计算的 key/value positions 数，而 sequence 仍包含 $n$ 个 query positions。论文将这种 restriction 提为 future direction，原始 Transformer 实验使用 full attention。

## Evidence

用户在 Lesson 0008 的思考题中准确推导了 $n<d$ 条件，区分了 layer-level parallelism 与 autoregressive generation，并说明 restricted attention 通过增加 maximum path length 换取 $O(rnd)$ per-layer complexity。上述补充用于区分三项 metric 的语义，并收紧 training / inference 与 neighborhood size 的边界。

Primary source: Vaswani et al. (2017), §4、Table 1。
