# 已建立：AdamW 的 Decoupled Weight Decay 边界

## 1. L2 regularization 与 weight decay 的等价层级

用户已正确区分两种机制的定义：

- L2 regularization 修改 objective，在 data loss 上加入 $\frac{\lambda'}{2}\lVert\theta\rVert_2^2$。
- Weight decay 修改 parameter update，直接施加乘性收缩 $(1-\lambda)\theta_t$。
- 两者不冲突，也不是同一个定义；Proposition 1 的结论只是在标准 SGD（无 momentum）下经过系数换算后 parameter iterate 相同。

标准 SGD 中的两个收缩因子分别为：

$$
1-\alpha\lambda'
\qquad\text{与}\qquad
1-\lambda.
$$

因此等价条件是：

$$
\lambda=\alpha\lambda',
\qquad
\lambda'=\frac{\lambda}{\alpha}.
$$

用户在第一次复述中误把“L2 penalty coefficient 除以 learning rate”说成“每步 decay rate 设为 $1/\alpha$”。经定点修正后，用户使用 $\alpha=0.01$、$\lambda=0.001$ 正确得到：

$$
\lambda'
=\frac{0.001}{0.01}
=0.1,
$$

并验证：

$$
1-\alpha\lambda'
=1-0.01\times0.1
=0.999
=1-0.001
=1-\lambda.
$$

这表明系数换算与 parameter-iterate equivalence 已通过 retrieval。

## 2. Adam 下等价性为什么失效

用户已建立 coupled L2 regularization 的数据流：

$$
\lambda'\theta_{t-1}
\longrightarrow
g_t
\longrightarrow
m_t,v_t
\longrightarrow
M_t
\longrightarrow
\theta_t.
$$

L2 penalty 不仅参与当前 gradient update，还进入 first moment $m_t$ 与 second raw moment $v_t$，从而改变后续的 per-coordinate preconditioner。若要把这种更新与直接 weight decay 写成同一形式，需要 $M_t$ 是常数倍单位矩阵 $kI$；Adam 的 $M_t$ 随 coordinate 与 step 变化，通常不满足这一条件。

## 3. L2-Adam 与 AdamW 的路径边界

用户已正确识别：

- `L2-Adam` 是本课程用于表示“Adam 优化 L2-regularized loss”的简称，不是独立的正式 optimizer 名称。
- L2-Adam 的 $g_t$ 包含 $\lambda'\theta_{t-1}$，因此 $m_t$、$v_t$ 通常也与 AdamW 不同。
- AdamW 的 $g_t$ 只包含 data gradient；moments 不聚合 decay term。
- AdamW 的 weight decay 直接作用于 parameters，与 moments / preconditioner 解耦。

需要保留的表述边界：L2-Adam 没有 **decoupled weight decay**，但它仍有 coupled L2 regularization；标准 AdamW 没有把 L2 penalty 加入 loss，但 decoupled weight decay 仍提供 regularization effect。二者同时启用会叠加两条正则路径，不是标准 AdamW 的含义。

## Evidence

用户完成了 Lesson 0013 的三段式机制复述，并通过手写数值检索修正 $\lambda'$、$\lambda$ 与 $\alpha$ 的换算关系。Recall gate 已通过。

Primary source: Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, §2、Appendix A、Algorithm 2。
