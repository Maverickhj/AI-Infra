# 已建立：Adam Effective Step 与 Evidence Boundaries

用户已正确区分 Adam update 中的三类 quantities：

- Global learning rate $\eta_t$ 是所有 coordinates 通常共享的基础尺度。
- Per-coordinate preconditioner

  $$
  \rho_{t,j}
  =
  \frac{1}{\sqrt{\widehat{v}_{t,j}}+\epsilon}
  $$

  根据 coordinate $j$ 的 squared-gradient history 调整 scale。
- Realized update

  $$
  \Delta\theta_{t,j}
  =
  -\eta_t\rho_{t,j}\widehat{m}_{t,j}
  $$

  是 global scale、signed first-moment signal、preconditioner 与稳定项共同作用后的实际 parameter movement；不能只归因于 $\eta_t$ 与 $\rho_{t,j}$。

用户也已建立 normalized moment ratio 的核心直觉：

- 稳定同向 gradients 使 signed first moment 保持较大，normalized ratio 保留明确方向。
- 正负频繁抵消时，$\widehat{m}_{t,j}$ 变小，而 $\widehat{v}_{t,j}$ 仍记录 squared magnitude，因此 update magnitude 受到抑制。
- Gradient descent 的 parameter update 方向带有负号，即沿 $-\widehat{m}_{t,j}$ 方向移动。

## Experimental evidence boundary

用户正确识别 Adam §6 的 two-hidden-layer ReLU network 是与 neural-network training 较接近的 experiment。该结果支持 Adam 在该具体 model、dataset、hyperparameter search 与 metric 下的 iteration 和 wall-clock progress，但不能推广为对所有 neural networks、Transformer-based LLM 或未来 architectures 的 universal effectiveness guarantee。

## Convergence evidence boundary

用户已识别原论文 theorem 与 Transformer training 之间的主要 scope mismatch：

- theorem 位于 online convex optimization，而 Transformer loss 对 parameters 是 non-convex；
- bounded gradients 不能对实际 training trajectory 自动成立；
- bounded iterate distance 指任意 iterates 之间满足 $\lVert\theta_n-\theta_m\rVert\leq D$，即 parameter trajectory 具有 bounded diameter，不是“迭代步数需要确定”；
- theorem 使用 $\alpha_t=\alpha/\sqrt{t}$，不能静默替换成 constant LR、warmup 或任意 schedule；
- theorem 使用 $\beta_{1,t}=\beta_1\lambda^{t-1}$，即 first-moment coefficient 随 step exponentially decay，而不是始终固定的 $\beta_{1,t}$；
- $R(T)/T=\mathcal{O}(1/\sqrt{T})$ 是 asymptotic bound，不保证 observed average regret 每一步单调下降，也不能推出 $\theta_t$ 收敛到固定 point。

用户同时识别了后续 primary research 的修正边界：Reddi、Kale 与 Kumar 给出 simple convex setting，使 Adam 不收敛到 optimal solution，并指出原 convergence analysis 的问题。这修正了 theoretical guarantee，不能反向抹除 Adam 原论文特定 experiments 中的 empirical observations。

Primary sources:

- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, §§2.1、4、6。
- Reddi, Kale & Kumar, *On the Convergence of Adam and Beyond*, ICLR 2018。
