# 已建立：Transformer Batching、Dropout 与 Label Smoothing

用户已建立 variable-length batching 的核心因果链：

- Approximate-length grouping 把 lengths 相近的 sentence pairs 放入同一 batch；结合 padded tensor shapes，可以推断这会减少 padding waste。
- Token budget 用 token count 而不是固定 sentence-pair count 描述 batch capacity，因此 batch 中 sentence pairs 的数量会随平均 sequence length 改变。
- 原论文报告的是两个 separate approximate budgets：每批约 $25{,}000$ source tokens 与 $25{,}000$ target tokens，不能合并改写成“总共 $25{,}000$ tokens”。

用户已正确定位原始 Post-LN Transformer 中的 dropout：

$$
y
=
\operatorname{LayerNorm}
\left(
x+\operatorname{Dropout}(F(x))
\right).
$$

Dropout 作用于 sub-layer output，再进行 residual addition 与 LayerNorm。Stack 入口还对 embedding 与 positional encoding 的和使用 dropout：

$$
X_0
=
\operatorname{Dropout}(E+PE).
$$

这里应使用 positional encoding；原论文 base model 使用 sinusoidal positional encoding，不能在该语境中默认改写成 learned positional embedding。

用户已建立 label smoothing 的主要机制：

- Hard-label cross-entropy 对一个有效 target position 为 $-\log p_y$，只直接提高正确 token probability。
- Label smoothing 把 one-hot target 改为非 one-hot target distribution $q$，使 non-target coordinates 也获得 target probability mass。
- Smoothed cross-entropy

  $$
  \mathcal{L}_{\mathrm{smooth}}
  =
  -\sum_{k=1}^{V}q_k\log p_k
  $$

  使用完整 target distribution，而不是只读取正确 coordinate。
- 论文观察到 label smoothing 使 perplexity 变差，但 accuracy 与 BLEU 改善；这些 metrics 衡量的对象不同，因此不要求同方向变化。

需要保留的边界：BLEU 不是 token accuracy。某个 position 的正确 token 仍为 argmax，只能直接关联该位置的 token prediction；BLEU 取决于完整 decoded sequence 的 n-gram overlap，还受 autoregressive history 与 decoding procedure 影响。因此不能由单点 argmax 推出 BLEU 必然不下降。

## Paper fact、systems inference 与 implementation convention

### Paper facts

- Sentence pairs 按 approximate sequence length batched together。
- 每批约含 $25{,}000$ source tokens 与 $25{,}000$ target tokens。
- Dropout 作用于每个 sub-layer output，并位于 residual addition 与 LayerNorm 之前。
- Dropout 还作用于 embeddings 与 positional encodings 的和。
- Base model 使用 $P_{\mathrm{drop}}=0.1$。
- Label smoothing 使用 $\epsilon_{\mathrm{ls}}=0.1$。
- 论文报告 label smoothing hurts perplexity，但 improves accuracy and BLEU。

### Systems inferences

- Length grouping 减小同一 padded batch 中 $L_{\max}$ 与多数 sequence lengths 的差距，因此减少 padding slots。
- Token budget 相比 fixed example count 更接近控制每步 workload；但 self-attention cost 还受 individual sequence lengths 的 quadratic term 影响，所以固定 token count 不等于 compute 完全恒定。

### Implementation conventions

- Bernoulli mask、除以 $1-p$ 的 inverted-dropout scaling，以及不同 coordinates 是否独立采样，需要以 framework 和 dropout variant 为准。
- 原论文只给出 $\epsilon_{\mathrm{ls}}=0.1$，没有写出 coordinate-level smoothed distribution。Uniform mixture 与只向 $V-1$ 个 non-target tokens 分配 mass 是不同 conventions。
- Padding positions 是否排除、sequence/batch loss 使用 sum 还是 mean，以及具体 reduction denominator，都需要检查实现。

Primary source: Vaswani et al., *Attention Is All You Need*, §§5.1、5.4、6.2。
