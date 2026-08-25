---
type: paper
status: seed
created: 2026-08-25
updated: 2026-08-25
domains:
  - optimization
  - deep-learning
aliases:
  - Adam - A Method for Stochastic Optimization
  - Adam paper
source:
  - https://arxiv.org/abs/1412.6980
ai_generated: true
reviewed: false
---

# 2015 - Adam - A Method for Stochastic Optimization

> [!warning]
> 本文是基于原论文与课程检索记录整理的 AI-assisted source note，尚未完成人工核验。理论结论必须连同原论文 assumptions 与后续修正一起阅读。

## Source identity

- Authors: Diederik P. Kingma and Jimmy Ba
- Venue: ICLR 2015; first arXiv submission in 2014
- Primary source: [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

## Optimization problem addressed

Adam 为 stochastic first-order optimization 显式维护两类 gradient history：

- First moment $m_t$ 近似记录 signed mean，即近期 direction consistency。
- Second raw moment $v_t$ 近似记录 squared magnitude，而不是 centered variance。

这使 optimizer 在 global stepsize 之外拥有 per-coordinate adaptive scale。

## Algorithm

对 stochastic gradient $g_t=\nabla_\theta f_t(\theta_{t-1})$：

$$
\begin{aligned}
m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\widehat m_t &= \frac{m_t}{1-\beta_1^t},\\
\widehat v_t &= \frac{v_t}{1-\beta_2^t},\\
\theta_t &= \theta_{t-1}
-\alpha\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\end{aligned}
$$

所有 vector operations 都是 element-wise。原论文给出的 tested default settings 是 $\alpha=0.001$、$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$。

## Why bias correction exists

EMA state 从 $m_0=v_0=0$ 开始。将一般 EMA

$$
u_t=\beta u_{t-1}+(1-\beta)x_t
$$

展开后，真实 observations 的 coefficient sum 只有 $1-\beta^t$；缺少的 $\beta^t$ 是 zero initialization 留下的 weight mass。在 stationarity approximation 下除以 $1-\beta^t$，可以校正 early-step shrinkage。

First- and second-moment EMA 使用不同的 $\beta_1$ 与 $\beta_2$，所以两类 shrinkage 不能假定会在 ratio 中自动抵消。

## Effective update decomposition

对 coordinate $j$：

$$
\Delta\theta_{t,j}
=
-\underbrace{\alpha_t}_{\text{global scale}}
\underbrace{\frac{1}{\sqrt{\widehat v_{t,j}}+\epsilon}}_{\text{per-coordinate preconditioner}}
\underbrace{\widehat m_{t,j}}_{\text{signed signal}}.
$$

Realized update 由三者共同决定，不能把最终 movement 只归因于 learning rate 或 denominator。

## Evidence and theory boundaries

- 原论文的 convergence analysis 位于 online convex optimization，并以 regret 衡量 online decisions 相对 best fixed comparator 的累计差距。
- Theorem 使用 convex cost functions、bounded gradients、bounded feasible-set diameter、$\alpha_t=\alpha/\sqrt{t}$ 与 exponentially decaying $\beta_{1,t}$ 等 assumptions。
- $R(T)/T=O(1/\sqrt{T})$ 表示 average regret 的 asymptotic upper bound，不表示 observed regret 每一步单调下降，也不推出 non-convex neural-network parameters 收敛到某个 fixed point。
- Neural-network experiments 是具体 empirical observations；论文自己明确指出 convex convergence analysis 不适用于这些 non-convex objectives。
- Reddi、Kale 与 Kumar 后续在 *On the Convergence of Adam and Beyond* 中给出 Adam 不能收敛到 optimal solution 的 simple convex setting，并提出 AMSGrad。这修正 theoretical guarantee，但不抹除原论文已报告的实验观察。

## Link to the Transformer paper

[[2017 - Attention Is All You Need]] 使用 Adam，但采用 $\beta_2=0.98$、$\epsilon=10^{-9}$，并把 optimizer 放入 warmup / inverse-square-root learning-rate schedule。两篇论文的证据职责不同：

- Adam paper 解释 moment estimates、bias correction、adaptive scaling，并给出自身的 theory / experiments。
- Transformer paper 记录该 optimizer recipe 在其 translation system 中的使用与 system-level results。

因此，Adam 是 Transformer training closure 的一个 dependency，不是 Transformer architecture claim 的证据。

## Derived note

- [[Adam Optimizer - Moments, Bias Correction, and Evidence Boundaries]]
