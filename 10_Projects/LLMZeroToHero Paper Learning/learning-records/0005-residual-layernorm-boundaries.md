# 部分建立：Residual Connection 与 LayerNorm 的边界

用户已正确建立以下理解：

- Residual Connection 让 sub-layer 学习 update / residual，而不是每次从头重建输入 $x$；当 $F(x)$ 在某些方向上接近 $0$ 时，可保留对应的输入信息。
- 原始 Transformer 为满足 residual addition，使所有 sub-layers 与 embedding layers 的输出维度保持为 $d_{\text{model}}=512$。
- 原始 Post-LN 的输出为 $\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$；当 $\operatorname{Sublayer}(x)=0$ 时，输出为 $\operatorname{LayerNorm}(x)$，不严格等于 $x$。

当前待巩固的是 LayerNorm 的 reduce axis：对 $X\in\mathbb{R}^{B\times N\times H}$，它固定 batch index $b$ 与 token position $i$，沿 feature index $k\in\{1,\ldots,H\}$ 求 $\mu_{b,i}$ 与 $\sigma_{b,i}^2$。因此准确说法不是“沿 token dimension 归一化”，而是“对每个 token independently，沿 $d_{\text{model}}$ features 归一化”。$\gamma,\beta\in\mathbb{R}^{H}$ 是按 feature 学习、跨 batch 与 token positions 共享的参数。

另一个精确边界：若 $\operatorname{Sublayer}(x)=0$，training-time 的 $\operatorname{Dropout}(0)=0$，所以 dropout rate 不影响“输出为 $\operatorname{LayerNorm}(x)$”这一结论；非 identity 的原因是 Post-LN 本身。

## Evidence

用户在 Lesson 0005 后准确解释了 residual 的保留输入作用、统一 $d_{\text{model}}$ interface 与 Post-LN 非严格 identity。后续应通过单 token 的 feature-axis 例子，巩固 LayerNorm 的归一化范围与参数共享。
