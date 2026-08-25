# 已建立：Adam Moments 与 Per-Coordinate Adaptive Scale

用户已正确建立以下理解：

- SGD baseline 没有在 optimizer state 中显式保存近期 gradient direction 是否一致，以及每个 coordinate 的 gradient squared magnitude 通常有多大；Adam 分别用 $m_t$ 与 $v_t$ 保存这两类 exponentially weighted history。
- Adam update 的 denominator 随 coordinate $j$ 的 squared-gradient history 改变，因此 global stepsize $\alpha$ 之外还存在 per-coordinate adaptive scale。
- Adam 没有移除 global stepsize；parameter update 仍显式乘以 $\alpha$。

用户已进一步说明：恒定 gradient 的 centered variance 为 $0$，不代表其 squared expectation 为 $0$；second raw moment 估计的是 gradient 平方的 expectation。比较二者不是因为公式应该相同，而是为了防止把 $v_t$ 误解成“gradient 波动程度”。

考虑两个 idealized long-run gradient distributions：

$$
\begin{array}{c|ccc}
\text{pattern}
&\mathbb{E}[g]
&\mathbb{E}[g^2]
&\operatorname{Var}(g)\\
\hline
g=10
&10
&100
&0\\
g\in\{-10,+10\}\text{ equiprobably}
&0
&100
&100
\end{array}
$$

第一种 gradient 没有任何 fluctuation，但仍有很大的 squared magnitude。因此：

- $v_t$ 近似追踪 $\mathbb{E}[g^2]$，对两种 pattern 都会记录相近的 scale；
- $m_t$ 近似追踪 signed mean，区分稳定同向 gradient 与正负抵消；
- Adam 使用 $\widehat{m}_t/\sqrt{\widehat{v}_t}$ 组合 direction consistency 与 total squared scale。

若把 $v_t$ 错称为 centered variance，就会错误预测恒定 gradient 的 denominator 应接近 $0$。准确边界是：$v_t$ 不只在 gradient 波动时才非零；只要 gradient magnitude 非零，$g_t^2$ 就会贡献到 $v_t$。

## Evidence

用户在 Adam Lesson 0009A 的思考题中准确说明了 $m_t/v_t$ 对 SGD state 的补充、denominator 的 per-coordinate adaptation，以及 global stepsize $\alpha$ 的必要性。后续检索中又准确说明恒定 nonzero gradient 虽然 centered variance 为 $0$，但 second raw moment 仍由 squared magnitude 决定，因此本课边界已建立。

Primary source: Kingma & Ba, *Adam: A Method for Stochastic Optimization*, §2、Algorithm 1。
