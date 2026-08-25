# 已建立：Adam Zero-Initialization Bias Correction

用户已正确建立以下核心因果链：

- Adam 把 EMA state 初始化为 $u_0=0$。Training 开始前不存在 historical observations，因此这个 zero state 表示 empty history，而不是一个真实的 zero-valued observation。
- 将 recurrence 展开后，前 $t$ 个真实 observations 的 coefficient sum 为 $1-\beta^t$；缺失的 $\beta^t$ 是 zero initialization 留下的 weight mass。
- First-moment EMA 与 second-moment EMA 分别使用 $\beta_1$ 和 $\beta_2$，因此 initialization shrinkage 分别为 $1-\beta_1^t$ 与 $1-\beta_2^t$。
- 两种 shrinkage 通常不同，不能假设它们会在 $m_t/\sqrt{v_t}$ 中自动抵消；不做 correction 会留下额外 factor。

需要保留的表述边界：在 $t=1$ 时，不是 $u_{t-1}$ 不存在，而是 $u_0$ 被明确定义为 $0$，且它尚未汇总任何 pre-training observations。在 later early steps，$u_{t-1}$ 已经存在，但它仍继承此前未积累完整 history weight 的 initialization effect。

## Evidence boundary

用户表示 expectation-bias derivation 已经熟悉，因此本记录不重复要求公式推导。用户未对 bias correction 不能消除 stochastic noise 与 non-stationary tracking error 的边界做独立复述，因此不把该边界记为本轮独立检索证据。

Primary source: Kingma & Ba, *Adam: A Method for Stochastic Optimization*, §3、Algorithm 1。
