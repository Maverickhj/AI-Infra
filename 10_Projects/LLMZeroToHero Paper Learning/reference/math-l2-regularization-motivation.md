---
type: knowledge
status: growing
created: 2026-08-26
updated: 2026-08-26
domains: []
aliases: []
source: [1711.05101 AdamW (Loshchilov & Hutter), Hanson & Pratt 1988, Hoerl & Kennard 1970]
ai_generated: true
reviewed: false
---

# L2 Regularization 的引入动机（0013 前置阅读）

> [!warning]
> 本文内容由 AI 总结生成，尚未人工核验。本文由 `tmp` topic draft 升格为正式版本，是唯一维护版本；讨论记录已并入文末。

**定位**：`0013 AdamW` 的前置阅读。回答一个问题——给损失函数（loss）加入 L2 正则化的最初动机是什么——并按因果递进展开。L2 正则化是在原损失上增加与参数平方和成正比的惩罚项。本文是模型无关（model-agnostic）的数学底座，AdamW 特有的结论只出现在最后一步的衔接里。

## 一句话版本

正则化是给优化目标注入不来自数据的偏好；L2 偏好「小权重」。在线性模型中，这条偏好可以被完整推导：它能通过岭回归（ridge regression）的谱抬升封顶病态方向上的估计方差，可解释为高斯先验（Gaussian prior）下的最大后验估计（maximum a posteriori, MAP），并且与每步权重衰减（weight decay）逐步等价，但这一等价仅限标准随机梯度下降（stochastic gradient descent, SGD）。进入深度网络后，这套解释不再严格，更多是历史动机与经验依据。

## Step 0 · 问题：经验风险最小化没有偏好

监督学习的理想目标是最小化期望风险（expected risk）$R(\theta)=\mathbb{E}[\ell(f_\theta(x),y)]$，其中 $\ell$ 是单个样本的损失函数；实际只能最小化有限训练样本上的经验风险（empirical risk）：

$$
\hat R(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(f_\theta(x_i),y_i).
$$

经验风险最小化（empirical risk minimization, ERM）求解 $\operatorname*{arg\,min}_{\theta}\hat R(\theta)$。当模型容量大过有效数据量时，所得模型可能同时记住噪声。加入正则项，就是注入一个不来自训练样本的偏好，声明哪类 $\theta$ 在先验上更可信；L2 只是其中一种选择。

## Step 1 · 度量：偏差–方差分解

偏差–方差分解（bias–variance decomposition）把预测误差拆成系统性偏差、对训练集扰动的敏感程度和不可约噪声。设 $y=g(x)+\epsilon$，其中 $g(x)$ 是真实回归函数，噪声 $\epsilon$ 的均值为 $0$、方差为 $\sigma^2$。在平方损失下：

$$
\mathbb{E}\big[(y-\hat f(x))^2\big]
=\underbrace{\big(\mathbb{E}[\hat f(x)]-g(x)\big)^2}_{\text{偏差平方（squared bias）}}
+\underbrace{\mathbb{E}\!\left[\big(\hat f(x)-\mathbb{E}[\hat f(x)]\big)^2\right]}_{\text{方差（variance）}}
+\underbrace{\sigma^2}_{\text{不可约噪声}}.
$$

正则化是用偏差换方差的旋钮。问题变为：L2 在哪些方向上、以什么速率完成这种交换。

## Step 2 · 失败模式：最小二乘的方差爆点

考虑线性模型 $f_\theta(x)=x^{\top}\theta$，并令 $y=X\theta^\ast+\epsilon$。这里，$X$ 是设计矩阵，$\theta^\ast$ 是生成数据的真实参数，$I$ 是单位矩阵，且噪声协方差 $\operatorname{Cov}(\epsilon)=\sigma^2I$。普通最小二乘法（ordinary least squares, OLS）的解为：

$$
\hat\theta_{\mathrm{OLS}}=(X^{\top}X)^{-1}X^{\top}y,
\qquad
\operatorname{Cov}(\hat\theta_{\mathrm{OLS}})=\sigma^2(X^{\top}X)^{-1}.
$$

对称矩阵 $X^{\top}X$ 可作特征分解 $X^{\top}X=Q\Lambda Q^{\top}$，其中 $Q$ 的第 $j$ 列 $q_j$ 是特征方向，$\Lambda=\operatorname{diag}(s_1,\dots,s_d)$ 收集对应特征值。第 $j$ 个特征方向的方差是 $\sigma^2/s_j$。当特征近似共线时，某个 $s_j\to 0$，方差趋于无界；这就是病态方向上 ERM 失败的具体形态。

## Step 3 · 推导：惩罚项等价于谱抬升

$$
\hat\theta_{\lambda}
=\operatorname*{arg\,min}_{\theta}
\left\{\|y-X\theta\|_2^2+\lambda\|\theta\|_2^2\right\}.
$$

这里，$\|\theta\|_2$ 表示参数向量的 L2 范数，$\lambda\geq 0$ 是正则化强度。令目标函数对 $\theta$ 的梯度为零，得到正规方程（normal equations）$(X^{\top}X+\lambda I)\hat\theta_{\lambda}=X^{\top}y$。与 OLS 相比，它把每个特征值从 $s_j$ 抬升到 $s_j+\lambda$，即「谱抬升」。每个方向上估计系数的方差变为：

$$
\operatorname{Var}\big(q_j^{\top}\hat\theta_{\lambda}\big)
=\frac{\sigma^2 s_j}{(s_j+\lambda)^2}
\;\le\;\frac{\sigma^2}{4\lambda},
$$

上式的最大值在 $s_j=\lambda$ 处取得，因此方差放大因子在每个方向上都有上界。代价是信号被压缩为 $\frac{s_j}{s_j+\lambda}q_j^{\top}\theta^\ast$，弱方向几乎被丢弃。这就是 Hoerl 与 Kennard（1970）提出的岭回归：惩罚项不是任意附加项，而是正规方程中的谱抬升。

## Step 4 · 语义：高斯先验下的最大后验估计

令观测噪声 $\epsilon\sim\mathcal N(0,\sigma^2I)$，并为参数指定零均值的各向同性高斯先验 $\theta\sim\mathcal N(0,\tau^2I)$；「各向同性」表示先验在所有参数方向上使用相同方差 $\tau^2$。最大化后验概率等价于最小化负对数后验。去掉与 $\theta$ 无关的常数后：

$$
-\log p(\theta\mid y)
=\frac{1}{2\sigma^2}\|y-X\theta\|_2^2
+\frac{1}{2\tau^2}\|\theta\|_2^2.
$$

乘以 $\sigma^2$ 后即 Step 3 的目标，且 $\lambda=\sigma^2/\tau^2$：$\lambda$ 是噪声方差与先验方差之比，先验越紧收缩越强，收缩方向是 0。隐含假设是各向同性——所有方向的小权重同等可信；这条假设正是后面被 Adam 破坏的对象。

## Step 5 · 解的结构：谱收缩

谱收缩（spectral shrinkage）是指在 $X^{\top}X$ 的不同特征方向上，按对应特征值对估计系数进行不同强度的压缩：

$$
q_j^{\top}\hat\theta_{\lambda}
=\frac{s_j}{s_j+\lambda}\,\big(q_j^{\top}\hat\theta_{\mathrm{OLS}}\big),
\qquad
\frac{s_j}{s_j+\lambda}\in(0,1).
$$

- 收缩因子随 $s_j$ 单调递增：强信号方向几乎不动，弱方向几乎清零——收缩是谱自适应的，不是均匀的。
- 整体 $\|\hat\theta_{\lambda}\|_2\le\|\hat\theta_{\mathrm{OLS}}\|_2$，解被拉向原点。
- 伏笔：谱方向、参数坐标和 Adam 的逐坐标预条件矩阵（per-coordinate preconditioner）$M_t$ 是三个不同概念。坐标层面的均匀收缩与谱自适应收缩并不是一回事。

## Step 6 · 算法落地：梯度项等同于每步收缩

$$
\nabla_\theta\!\left(\frac{\lambda'}{2}\|\theta\|_2^2\right)=\lambda'\theta
\;\Longrightarrow\;
\theta_{t+1}=\theta_t-\alpha\big(\nabla f_t(\theta_t)+\lambda'\theta_t\big)
=(1-\alpha\lambda')\theta_t-\alpha\nabla f_t(\theta_t).
$$

其中，$f_t$ 是第 $t$ 步的数据损失，$\alpha$ 是学习率。梯度更新中的 $\lambda'\theta_t$ 与参数的逐步乘法收缩完全等价，这种收缩称为权重衰减。

**记号桥（接 0013）**：本步使用的 $\lambda'$ 是 L2 惩罚系数（penalty coefficient）；每步等价的衰减率为 $\lambda=\alpha\lambda'$（原文命题 1：$\lambda'=\lambda/\alpha$）。闭环：统计层（Step 1–3）→ 概率层（Step 4）→ 解结构（Step 5）→ 算法层（Step 6，Hanson 与 Pratt 的 weight decay，仅限标准 SGD）。

## Step 7 · 边界

- 成立域：平方损失 + 线性模型 + 可对角化 $X^{\top}X$；bias–variance 分解在非线性模型没有同样干净的逐点形式。
- ridge 尺度敏感，依赖特征标准化。
- 深网里断链：通常不存在一个固定的 $X^{\top}X$；参数空间中的高斯先验也不能直接等同于函数空间（function space，即模型所表示函数的空间）中的约束。因此，「L2 有效」主要是经验事实。Van Laarhoven（2017）与 Zhang 等（2019）认为，部分作用来自有效学习率（effective learning rate，即参数尺度变化后实际产生的相对更新幅度）渠道，而非单纯「向 0 收缩」。
- 在插值区间（interpolation regime，即模型能够把训练误差降至接近零的容量范围）可能出现双重下降（double descent）：测试误差随模型容量增大先降、再升、再降。此时经典偏差–方差图景不再完整；本链主要描述低到中等容量区的原初动机。
- 对 AdamW：链内没有任何一步依赖自适应预条件器（adaptive preconditioner，即按参数坐标缩放梯度的机制）。Adam 的 $M_t$ 恰在「乘以 $\alpha$」与「谱自适应」之间插入第三种逐坐标缩放，把原本均匀的偏好扭曲为坐标依赖偏好——这正是 `0013` 的瓶颈起点。

## 递进总览

目标定义（0）→ 交换度量（1）→ 失败定位（2）→ 修复推导（3）→ 语义赋值（4）→ 解结构（5）→ 算法落地（6）→ 成立边界（7）→ 接回 AdamW 因果链头。

## 阅读建议

- 读 `0013` 之前先读 Step 0–6；Step 7 可与 `0013` 的「作用边界」对照。

## 讨论记录

- `2026-08-26` 建立 topic draft：应要求把「给 loss 加 L2 正则的最初动机」的数学原理按递进关系展开，并在内容完整前将讨论记录于 draft。
- `2026-08-26` 术语规范化定稿（见 `0013` 与 `CURRICULUM.md`）：decay 率 $\lambda$（Eq. 1），L2 penalty coefficient $\lambda'$，等价条件 $\lambda'=\lambda/\alpha$；停用 `\lambda_{wd}` 复合记号、"ridge prior"、"拉力" 等非规范表述。
- `2026-08-26` 待解释点（已答复、按相关性证据边界处理）：AdamW 追平 SGD+momentum 泛化的机制是「uniform decay 恢复」；后续文献（Van Laarhoven 2017、Zhang et al. 2019）经 effective learning rate 渠道补充解释。
- `2026-08-26` 升格：用户确认 topic 讨论完全，本文件由 draft 升格为正式 reference 文档，tmp 草稿删除，不再双维护。
- 待办：`0013` 的 recall gate 未过，learning-record 未创建；lesson meta 与 `CURRICULUM.md` 已同步本前置阅读链接。
