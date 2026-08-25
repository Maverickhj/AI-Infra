---
type: knowledge
status: seed
created: 2026-08-25
updated: 2026-08-25
domains:
  - optimization
  - deep-learning
aliases:
  - Adam optimizer
source:
  - "[[2015 - Adam - A Method for Stochastic Optimization]]"
ai_generated: true
reviewed: false
---

# Adam Optimizer - Moments, Bias Correction, and Evidence Boundaries

> [!warning]
> 本文包含 AI 整理的可迁移结论，尚未完成人工核验。实现细节与 convergence claims 需要结合具体 optimizer variant、schedule 和 source version 判断。

## State semantics

Adam 在每个 parameter coordinate 上维护：

$$
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2.
\end{aligned}
$$

- $m_t$ 是 first-moment EMA，保留 signed direction history。
- $v_t$ 是 second-raw-moment EMA，保留 squared magnitude history。
- $v_t$ 不是 centered variance。恒定 nonzero gradient 的 variance 可以为 $0$，但 $\mathbb E[g^2]$ 仍非零。

因此，稳定同向 gradients 使 normalized ratio 保持明确方向；正负频繁抵消时，$m_t$ 变小，而 $v_t$ 仍记录 magnitude，update 会受到抑制。

## Zero-initialization bias

对 $u_0=0$ 的 EMA：

$$
u_t=(1-\beta)\sum_{i=1}^{t}\beta^{t-i}x_i.
$$

真实 observations 的 coefficient sum 是

$$
(1-\beta)\sum_{i=1}^{t}\beta^{t-i}=1-\beta^t.
$$

缺少的 $\beta^t$ 是 initialization 留下的 weight mass。Bias correction 使用：

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
$$

它主要校正 zero-initialization shrinkage；不能消除 stochastic noise，也不能使 non-stationary history 变成 exact expectation。

## Effective update

$$
\Delta\theta_{t,j}
=
-\alpha_t
\frac{\widehat m_{t,j}}{\sqrt{\widehat v_{t,j}}+\epsilon}.
$$

应区分：

- Global learning rate $\alpha_t$：optimizer 的共享 base scale。
- Per-coordinate preconditioner $(\sqrt{\widehat v_{t,j}}+\epsilon)^{-1}$：基于 coordinate history 的 scale。
- Realized update $\Delta\theta_{t,j}$：global scale、signed signal、preconditioner 与 stabilization 共同决定的 movement。

## Convergence claim boundary

“Adam converges”必须展开为：在哪个 problem class、哪些 assumptions、使用什么 metric，以及得到何种 bound。

原论文给出的核心 theoretical statement 位于 online convex optimization：面对事先未知的 convex costs $f_1,\ldots,f_T$，以 regret

$$
R(T)=\sum_{t=1}^{T}f_t(\theta_t)
-\min_{\theta\in\mathcal X}\sum_{t=1}^{T}f_t(\theta)
$$

比较 online decisions 与同一 feasible set 中的 best fixed comparator。其 conclusion、assumptions 与后续 convergence correction 都不能直接替换为“Transformer loss 会收敛到一个 optimum”。

## Transformer dependency chain

[[2017 - Attention Is All You Need]] 把 Adam 放入自己的 training recipe：

$$
\text{Adam state dynamics}
+
\text{Transformer-specific hyperparameters}
+
\text{warmup / decay schedule}
\rightarrow
\text{realized Transformer updates}.
$$

因此排查 Transformer optimization 时，不能只写“使用 Adam”；还需要同时记录 $\beta_1$、$\beta_2$、$\epsilon$、learning-rate schedule、step definition、gradient scaling 与 optimizer implementation。

## Related sources

- [[2015 - Adam - A Method for Stochastic Optimization]]
- [[2017 - Attention Is All You Need]]
