# 已建立：Positional Encoding 的设计动机与职责边界

用户已正确建立以下因果链：

- RNN 通过 recurrence 顺序传播 state，convolution 通过带有固定相对 offset 的局部 kernel 携带位置结构；移除 recurrence 与 convolution 后，不含 position signal 和 position-dependent mask 的 self-attention 对 sequence permutation 是 equivariant 的，因此需要额外注入 sequence order。
- 原始 Transformer 在 Encoder 与 Decoder stack 的底部，把 token embedding 与同为 $d_{\text{model}}$ 维的 Positional Encoding 相加，使后续 Q/K/V projections 可以同时读取 content 与 position information。
- Sinusoidal Positional Encoding 使固定 offset $k$ 对应一个只依赖 $k$ 的 linear transform。论文将此作为设计动机，假设它会让模型更容易按 relative positions 进行 attention。
- PE 提供 position information；causal mask 控制 target-side information visibility。二者职责不同，不能互相替代。

需保留以下精确边界：

- 原始 Transformer 的输入是

$$
X_{\mathrm{pos}}
=
\sqrt{d_{\text{model}}}\,
E(\mathrm{token}_{\mathrm{pos}})
+PE_{\mathrm{pos}},
$$

  并在 training 时对这个和应用 dropout。PE 加在 Encoder 与 Decoder stack 的 bottom，而不是在每个 Attention sub-layer 前重新加入。
- 在

$$
\begin{aligned}
PE_{(\mathrm{pos},2i)}
&=\sin\left(
\frac{\mathrm{pos}}
{10000^{2i/d_{\text{model}}}}
\right),\\
PE_{(\mathrm{pos},2i+1)}
&=\cos\left(
\frac{\mathrm{pos}}
{10000^{2i/d_{\text{model}}}}
\right)
\end{aligned}
$$

  中，$\mathrm{pos}$ 是 token position；$i$ 更准确地说是 sinusoidal frequency / feature-pair index，对应 feature coordinates $2i$ 与 $2i+1$。它不是 attention head index，原公式中也没有 head index。
- “$PE_{\mathrm{pos}+k}$ 可由 $PE_{\mathrm{pos}}$ 线性变换得到”只说明这种表示具有可供模型利用的 relative-offset structure。原文使用的是 “we hypothesized”，因此不能升级为模型必然学到或精确恢复 relative positions 的保证。
- 准确术语是 causal mask（因果掩码），不是 casual mask。

## Evidence

用户在 Lesson 0006 的思考题中准确说明了顺序信息缺口、在 embedding 中加入 PE、sinusoidal offset 的线性关系，以及 PE 与 causal mask 的职责分工。上述补充用于收紧 embedding scaling、注入位置、公式索引和论文 claim strength。

Primary source: Vaswani et al. (2017), §§3.4–3.5。
