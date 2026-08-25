# 已建立：Output Softmax、Shifted Targets 与 Weight Tying

用户已正确建立以下理解：

- Decoder hidden states 与 output projection matrix 相乘，为每个 target position 产生 vocabulary logits；沿 vocabulary axis 做 softmax 后，得到总和为 $1$ 的 next-token probability distribution。
- attention softmax 固定一个 query，沿 key-position axis 归一化；output softmax 固定一个 target position，沿 vocabulary axis 归一化。两者使用相同函数，但 tensor axis 与输出语义不同。
- shifted targets 使 training 时所有 target positions 能够并行进入 Decoder，causal mask 则阻止每个 position 读取尚未因果可见的 target tokens。
- weight tying 共享的是 learned parameters，不是三处完全相同的 computation。

需保留以下精确边界：

- 对

$$
\begin{aligned}
Z&\in\mathbb{R}^{B\times T\times d_{\text{model}}},\\
W_{\text{out}}&\in\mathbb{R}^{|\mathcal{V}|\times d_{\text{model}}},
\end{aligned}
$$

  output projection 为：

$$
L=ZW_{\text{out}}^\top
\in\mathbb{R}^{B\times T\times|\mathcal{V}|}.
$$

  softmax 只沿最后的 vocabulary axis 归一化，不合并 batch 或 target positions。
- 使用从 $1$ 开始的 target indexing 时：

$$
\begin{aligned}
\text{Decoder input}
&=[\mathrm{BOS},y_1,\ldots,y_{T-1}],\\
\text{Prediction target}
&=[y_1,y_2,\ldots,y_T].
\end{aligned}
$$

  因此 Decoder input slot $t$ 包含 $y_{t-1}$，对应的 output slot $t$ 预测 $y_t$。若 $y_t$ 出现在 Decoder input 中，它位于下一格 $t+1$；causal mask 阻止 slot $t$ 读取这个 future input slot。这比“position $t$ 不能读取 position $t$”更准确，因为 masked self-attention 通常允许读取当前 input slot。
- 原始 Transformer 的三向 weight tying 是：

$$
W_{\mathrm{src}}
=W_{\mathrm{tgt}}
=W_{\mathrm{out}}
=\mathcal{E}.
$$

  source / target embedding 通过 token id 查找 $\mathcal{E}$ 的 row，并在 embedding use 中乘以 $\sqrt{d_{\text{model}}}$；output projection 则让 Decoder hidden state 与所有 rows 做 dot products。共享 parameter identity 不等于共享 lookup、scaling 与 projection 的执行方式。
- 准确术语是 causal mask（因果掩码），不是 casual mask。

## Evidence

用户在 Lesson 0007 的思考题中准确解释了 hidden-state-to-probability data flow、两个 softmax 的 axis、shifted targets 与 causal mask 的协作，以及 weight tying 的 parameter / computation 边界。上述补充用于钉牢 shifted input 与 prediction target 之间的 off-by-one indexing。

Primary source: Vaswani et al. (2017), §§3.1、3.4。
