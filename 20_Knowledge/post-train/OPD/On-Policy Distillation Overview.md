---
type: knowledge
status: growing
created: 2026-08-06
updated: 2026-08-10
domains:
  - post-training
  - knowledge-distillation
  - reinforcement-learning
aliases:
  - On-Policy Distillation
  - OPD
ai_generated: true
reviewed: false
---

# On-Policy Distillation：数学原理、方法分类与工程实现

> [!warning]
> 本文包含 AI 整理的实质技术内容，尚未完成人工核验；2026 年预印本及相关实现可能继续更新。

> 本文系统整理大语言模型中的 On-Policy Distillation（OPD）及其自蒸馏变体。重点区分四个经常被混在一起的维度：**采样分布、教师来源、分布散度、梯度估计方式**。
>
> 更新日期：2026-08-06

---

## 摘要

On-Policy Distillation 不是一个固定的损失函数，而是一类训练框架：学生模型首先生成自己可能在推理阶段访问的轨迹，教师随后在这些学生前缀上提供 token 级分布监督。它将强化学习中的 **on-policy 状态分布** 与知识蒸馏中的 **dense token-level supervision** 结合起来，用于缓解传统监督蒸馏的 exposure bias 和序列生成中的误差累积。

最一般的 OPD 可以从四个正交维度理解：

1. **状态从哪里来**：固定数据、教师轨迹、当前学生轨迹、旧学生轨迹，或师生混合轨迹；
2. **教师是谁**：外部大模型、同模型的特权上下文分支、上下文增强教师、历史 checkpoint、教师相对 base model 的增量；
3. **对齐什么分布**：Forward KL、Reverse KL、Generalized JSD、混合或自适应散度；
4. **如何计算梯度**：全词表精确散度、采样 token 的 policy gradient、token-local credit、sequence-level return-to-go。

本文首先在第 0 章统一全部数学符号和基础推导，再分别讨论 GKD、MiniLLM、OPSD、Entropy-Aware OPD、On-Policy Context Distillation、On-Policy Delta Distillation、Trust-Region OPD 与 $\beta$-OPSD。

---

## 目录

- [[#第 0 章：统一数学基础|第 0 章：统一数学基础]]
- [[#第 1 章：OPD 的核心定义|第 1 章：OPD 的核心定义]]
- [[#第 2 章：为什么需要 On-Policy 数据|第 2 章：为什么需要 On-Policy 数据]]
- [[#第 3 章：按散度类型分类|第 3 章：按散度类型分类]]
- [[#第 4 章：按梯度计算方式分类|第 4 章：按梯度计算方式分类]]
- [[#第 5 章：按教师来源分类|第 5 章：按教师来源分类]]
- [[#第 6 章：代表性方法|第 6 章：代表性方法]]
- [[#第 7 章：OPD、SFT、离线 KD 与 RLVR 的关系|第 7 章：OPD、SFT、离线 KD 与 RLVR 的关系]]
- [[#第 8 章：工程实现|第 8 章：工程实现]]
- [[#第 9 章：训练诊断与方法选择|第 9 章：训练诊断与方法选择]]
- [[#第 10 章：总结|第 10 章：总结]]
- [[#参考文献|参考文献]]

---

## 第 0 章：统一数学基础

### 0.1 基本对象与符号

设训练数据中的输入 prompt 为

$$
x\sim \mathcal D_x.
$$

模型生成长度为 $T$ 的输出序列：

$$
y=(y_1,y_2,\ldots,y_T),\qquad y_t\in\mathcal V,
$$

其中 $\mathcal V$ 是词表，词表大小记为 $|\mathcal V|=V$。

在自回归生成中，第 $t$ 步的状态或前缀定义为：

$$
s_t=(x,y_{<t}),
\qquad
 y_{<t}=(y_1,\ldots,y_{t-1}).
$$

将当前生成 token 视为动作：

$$
a_t=y_t.
$$

本文统一使用以下策略符号：

| 符号 | 含义 |
|---|---|
| $\pi_\theta$ | 当前可训练学生策略 |
| $q_\phi$ 或 $q$ | 教师策略；通常冻结参数或 stop-gradient |
| $\mu$ | 实际执行 rollout 的 behavior policy |
| $\pi_{\mathrm{old}}$ | 生成当前 batch 时使用的旧学生策略 |
| $\pi_{\mathrm{ref}}$ | KL 正则使用的 reference policy |
| $c$ | 教师可见、学生不可见的特权上下文 |
| $y^\star$ | 标准答案、参考解法或 verified reasoning trace |

在同一个状态 $s_t$ 上，学生和教师的下一 token 分布分别为：

$$
\pi_t(v)
:=\pi_\theta(v\mid s_t),
\qquad
q_t(v)
:=q_\phi(v\mid s_t,c),
\qquad v\in\mathcal V.
$$

如果教师没有额外上下文，则省略 $c$。

---

### 0.2 自回归序列概率

学生的完整序列概率可分解为：

$$
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
$$

取对数后：

$$
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T}
\log \pi_\theta(y_t\mid s_t).
$$

教师同理：

$$
q_\phi(y\mid x,c)
=
\prod_{t=1}^{T}
q_\phi(y_t\mid s_t,c).
$$

这两个分解是后面 sequence-level KL、token reward 和 return-to-go 推导的基础。

---

### 0.3 Softmax、温度与蒸馏分布

设模型在状态 $s_t$ 上输出 logits $z_t(v)$，温度为 $\tau>0$，则：

$$
p_\tau(v\mid s_t)
=
\frac{\exp(z_t(v)/\tau)}
{\sum_{u\in\mathcal V}\exp(z_t(u)/\tau)}.
$$

- $\tau<1$：分布更尖锐；
- $\tau=1$：标准 softmax；
- $\tau>1$：分布更平滑，尾部 token 获得更多概率质量。

经典知识蒸馏中常使用相同温度计算教师与学生分布，并将 loss 乘以 $\tau^2$ 补偿 logits 梯度缩放：

$$
\mathcal L_{\mathrm{KD},\tau}
=
\tau^2 D_{\mathrm{KL}}
\left(q_\tau\|\pi_{\theta,\tau}\right).
$$

OPD 中 rollout 温度和 distillation temperature 是两个不同超参数：

- rollout temperature 决定学生访问哪些前缀；
- distillation temperature 决定教师 soft target 的平滑程度。

把二者混为一个温度，会让采样分布变化和监督分布变化同时发生，调参时相当不体贴。

---

### 0.4 熵、交叉熵与 KL 散度

对离散分布 $P,Q$，熵定义为：

$$
H(P)
=
-\sum_v P(v)\log P(v).
$$

交叉熵定义为：

$$
H(P,Q)
=
-\sum_v P(v)\log Q(v).
$$

KL 散度定义为：

$$
D_{\mathrm{KL}}(P\|Q)
=
\sum_v P(v)
\log\frac{P(v)}{Q(v)}.
$$

三者满足：

$$
D_{\mathrm{KL}}(P\|Q)
=
H(P,Q)-H(P).
$$

当 $P$ 固定时，最小化 $D_{\mathrm{KL}}(P\|Q)$ 等价于最小化交叉熵 $H(P,Q)$。

KL 不对称：

$$
D_{\mathrm{KL}}(P\|Q)
\neq
D_{\mathrm{KL}}(Q\|P).
$$

本文统一规定：

- 教师分布是 $q$；
- 学生分布是 $\pi$；
- **Forward KL**：$D_{\mathrm{KL}}(q\|\pi)$；
- **Reverse KL**：$D_{\mathrm{KL}}(\pi\|q)$。

这个方向约定必须先写死，否则不同论文各自交换字母以后，读者会被迫从上下文猜作者到底在“forward”什么。

---

### 0.5 Token-level Forward KL

在状态 $s_t$ 上：

$$
D_{\mathrm{FKL},t}
:=
D_{\mathrm{KL}}(q_t\|\pi_t)
=
\sum_{v\in\mathcal V}
q_t(v)
\log\frac{q_t(v)}{\pi_t(v)}.
$$

由于教师分布固定：

$$
D_{\mathrm{FKL},t}
=
-\sum_v q_t(v)\log\pi_t(v)
+
\underbrace{\sum_v q_t(v)\log q_t(v)}_{\text{与学生参数无关}}.
$$

因此 Forward KL 等价于教师 soft label 上的交叉熵。

设学生 logits 为 $z_t(j)$，则精确梯度为：

$$
\frac{\partial D_{\mathrm{FKL},t}}
{\partial z_t(j)}
=
\pi_t(j)-q_t(j).
$$

这个梯度具有两个重要特点：

1. 对整个词表给出 dense supervision；
2. 梯度不需要采样当前 token，因此方差较低。

---

### 0.6 Token-level Reverse KL

在状态 $s_t$ 上：

$$
D_{\mathrm{RKL},t}
:=
D_{\mathrm{KL}}(\pi_t\|q_t)
=
\sum_{v\in\mathcal V}
\pi_t(v)
\log\frac{\pi_t(v)}{q_t(v)}.
$$

它也可以写成：

$$
D_{\mathrm{RKL},t}
=
-H(\pi_t)
-
\mathbb E_{v\sim\pi_t}\left[\log q_t(v)\right].
$$

因此，最小化 Reverse KL 同时包含：

- 提高教师认为高概率的 token；
- 增大学生熵，避免策略过早退化为确定性分布。

对学生 logits 的精确梯度为：

$$
\frac{\partial D_{\mathrm{RKL},t}}
{\partial z_t(j)}
=
\pi_t(j)
\left[
\log\frac{\pi_t(j)}{q_t(j)}
-
D_{\mathrm{KL}}(\pi_t\|q_t)
\right].
$$

推导使用 softmax Jacobian：

$$
\frac{\partial \pi_i}{\partial z_j}
=
\pi_i(\mathbf 1[i=j]-\pi_j).
$$

从梯度形式可以看到：Reverse KL 的每个词表项由学生概率 $\pi_t(j)$ 加权；学生几乎不分配概率的 token，对当前更新的影响也很小。

#### 关于“mode-covering”和“mode-seeking”的限定

这里的 **mode** 指概率分布中的一个高概率区域，而不是 model（模型）。例如，教师认为两种不同的回答路径都合理时，教师的序列分布中就可能存在两个 mode。

假设教师分布 $q$ 有两个相互分离的峰，而容量受限的学生分布 $\pi$ 只能表达一个较简单的峰：

- **mode-covering（模态覆盖）**：学生尽量覆盖教师的多个高概率区域，即使因此需要在这些区域之间也分配一些概率；
- **mode-seeking（模态寻找）**：学生选择其中一个高概率区域集中拟合，而放弃其他难以同时表示的峰。

这种差异可以从 KL 散度的加权方向理解。对于 Forward KL：

$$
D_{\mathrm{KL}}(q\|\pi)
=
\sum_v q(v)\log\frac{q(v)}{\pi(v)},
$$

各位置由教师概率 $q(v)$ 加权。只要教师在某处具有明显概率，而学生给出的 $\pi(v)$ 接近零，损失就会很大。因此学生倾向于照顾教师分布中的各个高概率区域，即“教师认为可能的区域，我最好都覆盖”。当受限学生只能使用单峰分布逼近多峰教师时，折中结果有时会落在多个峰之间，因此 Forward KL 也常被称为 mean-seeking。

对于 Reverse KL：

$$
D_{\mathrm{KL}}(\pi\|q)
=
\sum_v \pi(v)\log\frac{\pi(v)}{q(v)},
$$

各位置由学生概率 $\pi(v)$ 加权。学生没有覆盖的区域因为 $\pi(v)$ 很小，对当前损失和梯度的影响也很小；但如果学生把概率放在教师低概率区域，则会受到明显惩罚。因此学生可以选择教师的某一个峰集中拟合，即“只选择一个教师认可度高的区域，暂时不覆盖其他区域”。

在语言模型中，可以将上述直觉分成两个尺度理解：

- 在 token level，Forward KL 更倾向于保留教师给出的多个候选 token，包括具有非零概率的尾部 token；Reverse KL 更关注学生当前已经分配较高概率的候选；
- 在 sequence level，不同的语义回答或推理路径可以形成不同 mode，mode-covering 对应保留多种合理生成路径，mode-seeking 对应集中于其中一种高概率路径。

因此，在经典变分推断图景中，当学生分布族无法完整表达教师多峰分布时：

- Forward KL 常表现为 mode-covering 或 mean-seeking；
- Reverse KL 常表现为 mode-seeking。

但这一描述依赖于学生表达能力受限或优化尚未收敛等条件，不是所有 LLM token-level KD 训练动态的完整定理。如果学生能够精确表示教师分布，并且训练到达全局最优点，那么 FKL 与 RKL 都在 $\pi=q$ 时取到最小值。Wu 等人在 2024 年进一步指出，有限训练阶段更显著的差异可能是二者对 head/tail token 的梯度关注不同：FKL 的梯度 $\pi(j)-q(j)$ 会直接响应教师分配的概率，而 RKL 的梯度带有学生概率 $\pi(j)$ 的权重，学生当前几乎不选择的 token 很难得到显著更新。因此，本文将“mode-covering/mode-seeking”作为经典直觉，而不是万能解释。

---

### 0.7 Generalized Jensen-Shannon Divergence

> 主要来源：[GKD / On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)。

给定权重 $0<\beta<1$，先定义混合分布：

$$
m_\beta
=
\beta q+(1-\beta)\pi.
$$

Generalized JSD 定义为：

$$
D_{\mathrm{JSD}_\beta}(q\|\pi)
=
\beta D_{\mathrm{KL}}(q\|m_\beta)
+
(1-\beta)D_{\mathrm{KL}}(\pi\|m_\beta).
$$

它具有以下性质：

- 对相互不重叠的支持集仍然有界；
- 同时包含 teacher-weighted 与 student-weighted 项；
- 在 GKD 的参数化下，$\beta$ 接近 0 时，其适当归一化后的梯度接近 Forward KL；$\beta$ 接近 1 时，梯度行为接近 Reverse KL。

需要注意，未经归一化的 $D_{\mathrm{JSD}_\beta}$ 在边界处会趋近于 0；论文中所谓“插值”主要指梯度方向和归一化极限，而不是数值本身直接收敛到两个 KL。

---

### 0.8 状态访问分布

第 $t$ 步时，策略 $\pi$ 诱导的状态访问分布定义为：

$$
d_\pi^t(s\mid x)
=
\Pr_{y_{<t}\sim\pi(\cdot\mid x)}
\left[(x,y_{<t})=s\right].
$$

可进一步定义时间平均状态分布：

$$
d_\pi(s\mid x)
=
\frac{1}{T}
\sum_{t=1}^{T}d_\pi^t(s\mid x).
$$

OPD 中“on-policy”描述的核心是：训练监督施加在学生自己诱导的状态分布上，即

$$
s_t\sim d_{\pi_\theta}^t.
$$

它不决定内部使用 Forward KL 还是 Reverse KL。

因此下面的目标完全可以同时是 on-policy 和 Forward KL：

$$
\mathbb E_{s_t\sim d_{\pi_\theta}^t}
\left[
D_{\mathrm{KL}}(q_t\|\pi_t)
\right].
$$

---

### 0.9 Token-level KL 与 Sequence-level KL 的链式分解

这是理解 OPD 时最容易被省略、也最重要的一组公式。

#### 0.9.1 Sequence-level Reverse KL

完整序列 Reverse KL 为：

$$
D_{\mathrm{KL}}
\left(
\pi_\theta(y\mid x)
\middle\|
q(y\mid x)
\right)
=
\mathbb E_{y\sim\pi_\theta}
\left[
\log\frac{\pi_\theta(y\mid x)}{q(y\mid x)}
\right].
$$

利用自回归分解：

$$
\log\frac{\pi_\theta(y\mid x)}{q(y\mid x)}
=
\sum_{t=1}^{T}
\log\frac{\pi_\theta(y_t\mid s_t)}{q(y_t\mid s_t)}.
$$

于是：

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_\theta\|q)
&=
\sum_{t=1}^{T}
\mathbb E_{s_t\sim d_{\pi_\theta}^t}
\mathbb E_{y_t\sim\pi_\theta(\cdot\mid s_t)}
\left[
\log\frac{\pi_\theta(y_t\mid s_t)}{q(y_t\mid s_t)}
\right]
\\
&=
\sum_{t=1}^{T}
\mathbb E_{s_t\sim d_{\pi_\theta}^t}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s_t)
\middle\|
q(\cdot\mid s_t)
\right)
\right].
\end{aligned}
$$

因此：

> **Sequence-level Reverse KL 可以精确分解成学生状态分布上的 token-level Reverse KL。**

这里“精确”指目标值的链式分解；真正求梯度时，还要考虑状态分布 $d_{\pi_\theta}$ 随参数变化。

#### 0.9.2 Sequence-level Forward KL

完整序列 Forward KL 为：

$$
D_{\mathrm{KL}}(q\|\pi_\theta)
=
\mathbb E_{y\sim q}
\left[
\log\frac{q(y\mid x)}{\pi_\theta(y\mid x)}
\right].
$$

类似地：

$$
D_{\mathrm{KL}}(q\|\pi_\theta)
=
\sum_{t=1}^{T}
\mathbb E_{s_t\sim d_q^t}
\left[
D_{\mathrm{KL}}
\left(
q(\cdot\mid s_t)
\middle\|
\pi_\theta(\cdot\mid s_t)
\right)
\right].
$$

注意状态分布是 $d_q^t$，不是 $d_{\pi_\theta}^t$。

#### 0.9.3 关键区别

GKD/OPSD 常见的 on-policy Forward-KL surrogate 是：

$$
\sum_t
\mathbb E_{s_t\sim d_{\pi_\theta}^t}
\left[
D_{\mathrm{KL}}(q_t\|\pi_t)
\right].
$$

它与 sequence-level Forward KL：

$$
\sum_t
\mathbb E_{s_t\sim d_q^t}
\left[
D_{\mathrm{KL}}(q_t\|\pi_t)
\right]
$$

并不相同。

前者的意义是：

> 在学生真正访问到的状态上，用教师完整分布训练学生。

它更接近 DAgger 式 on-policy imitation learning，而不是简单地在序列层面最小化 $D_{\mathrm{KL}}(q\|\pi)$。

---

### 0.10 统一 OPD 目标

令 rollout 由 behavior policy $\mu$ 产生：

$$
y\sim\mu(\cdot\mid x).
$$

一般化 token-level OPD 目标写为：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=
\mathbb E_{x\sim\mathcal D_x}
\mathbb E_{y\sim\mu(\cdot\mid x)}
\left[
\frac{1}{T}
\sum_{t=1}^{T}
w_t\,
D_t(q_t,\pi_{\theta,t})
\right],
$$

其中：

- $D_t$ 可以是 FKL、RKL、JSD 或其他 $f$-divergence；
- $w_t$ 可以是 mask、置信度权重、长度权重、重要性权重或 credit assignment 权重；
- $\mu=\pi_\theta$ 时是严格 on-policy；
- $\mu=\pi_{\mathrm{old}}\approx\pi_\theta$ 时通常称为 lagged on-policy 或 near-on-policy；
- $\mu=q$ 或固定数据分布时是 off-policy distillation。

---

### 0.11 Stop-gradient 与完整梯度

令轨迹级目标为：

$$
J(\theta)
=
\mathbb E_{y\sim\pi_\theta}
\left[F_\theta(y)\right].
$$

本节符号说明如下：

| 符号 | 含义 |
|---|---|
| $\theta$ | 当前学生模型的可训练参数 |
| $\pi_\theta$ | 由参数 $\theta$ 决定的学生策略 |
| $y=(y_1,\ldots,y_T)$ | 学生采样得到的完整输出轨迹，$y_t$ 是第 $t$ 个 token |
| $y\sim\pi_\theta$ | 使用学生策略自回归采样轨迹 $y$ |
| $F_\theta(y)$ | 在给定轨迹 $y$ 上计算的可微损失或 cost；在 OPD 中通常是沿轨迹累积的 token-level divergence |
| $J(\theta)$ | 同时考虑轨迹采样分布和轨迹内损失的期望目标 |
| $\mathbb E_{y\sim\pi_\theta}[\cdot]$ | 对学生可能采样到的轨迹求期望 |
| $\nabla_\theta$ | 对学生参数 $\theta$ 求梯度 |
| $\log\pi_\theta(y)$ | 学生生成整条轨迹的对数概率，等于 $\sum_t\log\pi_\theta(y_t\mid y_{<t},x)$ |
| $\operatorname{stopgrad}(\cdot)$ 或 $\operatorname{sg}[\cdot]$ | 前向计算保持输入值不变，反向传播时将经过该算子的梯度置为零 |

其完整梯度由 log-derivative trick 给出：

$$
\nabla_\theta J
=
\mathbb E_{y\sim\pi_\theta}
\left[
F_\theta(y)
\nabla_\theta\log\pi_\theta(y)
+
\nabla_\theta F_\theta(y)
\right].
$$

这里有两部分：

1. **sampling-distribution gradient**：
   $$
   F_\theta(y)\nabla_\theta\log\pi_\theta(y),
   $$
   表示参数变化会改变未来访问哪些状态；
2. **direct loss gradient**：
   $$
   \nabla_\theta F_\theta(y),
   $$
   表示在当前固定轨迹上直接调整学生 logits。

一般地，stop-gradient 算子满足：

$$
\operatorname{sg}[u]=u,
\qquad
\frac{\partial\operatorname{sg}[u]}{\partial u}=0.
$$

它只改变反向传播，不改变前向值。GKD 和标准 full-logit OPSD 通常对 rollout 执行 stop-gradient：

$$
y\sim\operatorname{stopgrad}(\pi_\theta).
$$

这里的记号表示：轨迹仍由当前学生 $\pi_\theta$ 生成，但采样完成后，将离散 token $y$ 及其形成的前缀状态视为固定训练数据，不计算“改变 $\theta$ 会改变轨迹采样概率”的 score-function 梯度。它不表示冻结学生模型；在固定轨迹上重新计算 $F_\theta(y)$ 时，学生 logits 到 $\theta$ 的梯度仍然保留。

因此，实际优化丢弃 sampling-distribution gradient，只保留 direct loss gradient：

$$
\nabla_\theta J_{\mathrm{stop}}
\approx
\mathbb E
\left[
\nabla_\theta F_\theta(y)
\right].
$$

其中，$J_{\mathrm{stop}}$ 表示将 rollout 视为常量后实际优化的 surrogate objective；$\approx$ 强调该梯度通常不等于原始 $J(\theta)$ 的完整梯度。工程上常见的过程是：先在无梯度环境中调用学生生成 $y$，再把固定的 $(x,y_{<t})$ 输入学生和教师，计算 full-vocabulary divergence，最后只对学生 logits 反向传播。

这不是完整序列目标的无偏梯度，因为它忽略了参数变化对未来状态访问分布的影响；但它具有低方差、稳定、易实现的优点。

---

### 0.12 Sequence-level Reverse KL 的 Policy-Gradient 推导

> 相关原始推导：[MiniLLM](https://arxiv.org/abs/2306.08543)。本文采用统一的教师 $q$、学生 $\pi_\theta$ 记号重新推导。

定义单步 cost：

$$
c_t
=
\log\frac{\pi_\theta(y_t\mid s_t)}{q(y_t\mid s_t)}.
$$

完整 sequence-level Reverse KL 为：

$$
\mathcal L_{\mathrm{seq-RKL}}
=
\mathbb E_{y\sim\pi_\theta}
\left[
\sum_{t=1}^{T}c_t
\right].
$$

定义 cost-to-go：

$$
C_t
=
\sum_{k=t}^{T}c_k.
$$

利用因果性，完整梯度可写成：

$$
\nabla_\theta \mathcal L_{\mathrm{seq-RKL}}
=
\mathbb E_{y\sim\pi_\theta}
\left[
\sum_{t=1}^{T}
C_t\,
\nabla_\theta
\log\pi_\theta(y_t\mid s_t)
\right].
$$

若定义教师相对学生的 reward：

$$
r_t
=
\log\frac{q(y_t\mid s_t)}{\pi_\theta(y_t\mid s_t)}
=-c_t,
$$

以及 return-to-go：

$$
R_t
=
\sum_{k=t}^{T}r_k,
$$

则：

$$
\nabla_\theta \mathcal L_{\mathrm{seq-RKL}}
=
-
\mathbb E
\left[
\sum_{t=1}^{T}
R_t\,
\nabla_\theta
\log\pi_\theta(y_t\mid s_t)
\right].
$$

加入任意不依赖当前动作的 baseline $b_t(s_t)$，期望梯度不变：

$$
\nabla_\theta \mathcal L
=
-
\mathbb E
\left[
\sum_t
(R_t-b_t)
\nabla_\theta\log\pi_\theta(y_t\mid s_t)
\right].
$$

MiniLLM 的核心推导与此一致，并进一步把单步项做全词表精确求和，把未来影响保留为 policy-gradient 项，以降低方差。

---

### 0.13 为什么 sampled-token loss 不能直接反向传播

对单步 Reverse KL：

$$
D_{\mathrm{KL}}(\pi_t\|q_t)
=
\mathbb E_{a_t\sim\pi_t}
\left[
\log\pi_t(a_t)-\log q_t(a_t)
\right].
$$

假设先采样 $a_t$，随后将采样过程 detach，并直接定义：

$$
\widehat{\mathcal L}_t
=
\log\pi_t(a_t)-\log q_t(a_t).
$$

直接自动微分只得到：

$$
\nabla_\theta\widehat{\mathcal L}_t
=
\nabla_\theta\log\pi_t(a_t).
$$

而其期望为：

$$
\begin{aligned}
\mathbb E_{a_t\sim\pi_t}
\left[
\nabla_\theta\log\pi_t(a_t)
\right]
&=
\sum_a\pi_t(a)\nabla_\theta\log\pi_t(a)
\\
&=
\sum_a\pi_t(a)
\frac{\nabla_\theta\pi_t(a)}{\pi_t(a)}
\\
&=
\nabla_\theta\sum_a\pi_t(a)
\\
&=
\nabla_\theta 1
=0.
\end{aligned}
$$

这称为 score function 的零均值性质。单个样本上的 $\nabla_\theta\log\pi_t(a_t)$ 通常不为零；但动作按当前策略自身的概率反复采样后，这些更新会在期望上相互抵消。

真正的 Reverse-KL 梯度并不为零。令：

$$
c_t(a)
=
\log\frac{\pi_t(a)}{q_t(a)},
$$

则 sampling distribution 和 $c_t(a)$ 都依赖于学生参数，完整梯度为：

$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(\pi_t\|q_t)
&=
\mathbb E_{a\sim\pi_t}
\left[
c_t(a)\nabla_\theta\log\pi_t(a)
+
\nabla_\theta c_t(a)
\right]
\\
&=
\mathbb E_{a\sim\pi_t}
\left[
c_t(a)\nabla_\theta\log\pi_t(a)
\right].
\end{aligned}
$$

其中第二个等号使用了：

$$
\mathbb E_{a\sim\pi_t}
\left[\nabla_\theta c_t(a)\right]
=
\mathbb E_{a\sim\pi_t}
\left[\nabla_\theta\log\pi_t(a)\right]
=0.
$$

因此，真正携带学习信号的是由 sampled cost 调制的 score-function 项：

$$
\log\frac{\pi_t(a)}{q_t(a)}
\nabla_\theta\log\pi_t(a).
$$

直接反向传播 sampled log-ratio 只得到期望为零的 direct loss gradient，却遗漏了上述由采样分布产生的梯度。因此，下面这种代码并不是 sampled Reverse-KL 的正确无偏优化：

```python
loss = student_logp_sampled - teacher_logp_sampled
loss.backward()
```

正确做法至少需要以下一种：

1. 全词表计算 $D_{\mathrm{KL}}(\pi_t\|q_t)$；
2. 使用 REINFORCE / policy-gradient estimator；
3. 使用 PPO-style importance ratio 和 advantage；
4. 采用 MiniLLM 式 single-step exact term + future return estimator。

---

### 0.14 Off-policy 与重要性采样

如果 rollout 来自旧策略 $\mu=\pi_{\mathrm{old}}$，而训练目标针对当前策略 $\pi_\theta$，轨迹级重要性权重为：

$$
w(y)
=
\frac{\pi_\theta(y\mid x)}{\mu(y\mid x)}
=
\prod_{t=1}^{T}
\frac{\pi_\theta(y_t\mid s_t)}{\mu(y_t\mid s_t)}.
$$

理论上：

$$
\mathbb E_{y\sim\pi_\theta}[F(y)]
=
\mathbb E_{y\sim\mu}[w(y)F(y)].
$$

但长序列上乘积权重方差极大，因此工程中更常使用单步 ratio：

$$
\rho_t(\theta)
=
\frac{\pi_\theta(y_t\mid s_t)}
{\pi_{\mathrm{old}}(y_t\mid s_t)}.
$$

为了最小化 Reverse KL，可以根据生成 token 在教师与旧学生下的相对概率定义固定 reward：

$$
r_t
=
\log q(y_t\mid s_t)
-
\log\pi_{\mathrm{old}}(y_t\mid s_t).
$$

如果 $r_t>0$，说明教师比旧学生更认可当前 token；如果 $r_t<0$，说明旧学生对该 token 分配了过高概率。最简单的 token-local advantage 为：

$$
A_t
=
r_t-b_t(s_t),
$$

其中 $b_t(s_t)$ 是不依赖当前动作的 baseline，用于降低方差。如果需要把后续偏差归因给当前 token，也可以使用：

$$
R_t
=
\sum_{k=t}^{T}r_k,
\qquad
A_t
=
R_t-b_t(s_t).
$$

在优化 PPO surrogate 时，$\pi_{\mathrm{old}}$、教师概率、$r_t$ 和 $A_t$ 都作为固定量执行 stop-gradient。当前学生只通过 importance ratio $\rho_t(\theta)$ 接收梯度。加入 PPO clipping 后：

$$
\mathcal L_{\mathrm{clip}}
=
-
\mathbb E_t
\left[
\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
$$

在一次更新开始时，即使 $\pi_\theta=\pi_{\mathrm{old}}$ 使得 $\rho_t=1$，ratio 的梯度仍不为零：

$$
\nabla_\theta\rho_t
=
\rho_t\nabla_\theta
\log\pi_\theta(y_t\mid s_t).
$$

未触发 clipping 时：

$$
\nabla_\theta(-\rho_t A_t)
=
-A_t\rho_t
\nabla_\theta\log\pi_\theta(y_t\mid s_t).
$$

因此在 $\rho_t=1$ 处，它就是 $-A_t\nabla_\theta\log\pi_\theta(y_t\mid s_t)$ 形式的 policy gradient。与裸的 $\nabla_\theta\log\pi_t(a_t)$ 不同，动作相关的 $A_t$ 打破了零均值抵消：正 advantage token 的概率被提高，负 advantage token 的概率被降低。importance ratio 用旧策略样本近似当前策略更新，clipping 则限制同一批 rollout 上的策略变化幅度。

对于单步目标，在 $\pi_\theta=\pi_{\mathrm{old}}$ 附近且 $A_t=r_t-b_t$ 时，这个更新与 Reverse-KL policy gradient 的局部方向一致；经过多轮更新并触发 clipping 后，它不再是原始完整目标的严格无偏梯度，而是一种低方差、有偏的局部近似。

对应的简化伪代码为：

```python
# rollout 阶段：以下量全部 detach
old_logp = log_pi_old.gather(dim=-1, index=sampled_token).detach()
teacher_logp = log_q.gather(dim=-1, index=sampled_token).detach()
reward = teacher_logp - old_logp
advantage = (reward - baseline).detach()

# update 阶段：只让 current_logp 接收梯度
current_logp = log_pi_theta.gather(dim=-1, index=sampled_token)
ratio = torch.exp(current_logp - old_logp)
ratio_clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
loss = -torch.minimum(
    ratio * advantage,
    ratio_clipped * advantage,
).mean()
loss.backward()
```

---

### 0.15 Full-vocabulary 与 Sampled-token 的计算复杂度

设 batch 中有效 token 总数为 $N$，词表大小为 $V$。

#### Full-vocabulary distillation

需要教师和学生在每个位置提供全词表概率：

$$
O(NV).
$$

优点：

- 低方差；
- 可精确计算 FKL、RKL、JSD；
- 每个位置包含完整教师信息。

代价：

- teacher logits 的显存、通信和存储成本高；
- 大词表与长 reasoning 轨迹下尤其昂贵。

#### Sampled-token distillation

只需要学生实际生成 token 的：

$$
\log\pi_\theta(y_t\mid s_t),
\qquad
\log q(y_t\mid s_t).
$$

输出规模约为：

$$
O(N).
$$

优点是通信成本低、易复用 RL pipeline；缺点是梯度方差高且必须正确处理采样分布梯度。

---

## 第 1 章：OPD 的核心定义

### 1.1 一句话定义

On-Policy Distillation 的核心是：

$$
\boxed{
\text{学生生成前缀}
+
\text{教师在学生前缀上给 dense token supervision}
}
$$

一个典型目标为：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=
\mathbb E_{x\sim\mathcal D_x}
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\frac{1}{T}
\sum_{t=1}^{T}
D\left(q_t\|\pi_{\theta,t}\right)
\right].
$$

实际 GKD/OPSD 通常不通过离散 rollout 反向传播，而把 rollout 看成当前模型产生的新训练数据。

---

### 1.2 “On-policy”不等于“Reverse KL”

“On-policy”只回答：

> 监督施加在哪些状态上？

散度方向回答：

> 在一个固定状态上，学生如何逼近教师？

因此以下组合都成立：

| 状态分布 | Token-level 散度 | 是否合理 |
|---|---|---|
| 学生状态 $d_\pi$ | Forward KL $q\|\pi$ | 是，GKD 常见形式 |
| 学生状态 $d_\pi$ | Reverse KL $\pi\|q$ | 是，MiniLLM/许多 OPD 形式 |
| 学生状态 $d_\pi$ | JSD | 是，GKD/OPSD 使用 |
| 教师或固定数据状态 | Forward KL | 是，传统 KD |
| 教师或固定数据状态 | Reverse KL | 也是可定义的，但不是 on-policy |

---

### 1.3 OPD 的四维分类框架

建议用以下四维描述任何一个具体 OPD 方法，而不是只报一个含糊的名字：

1. **Rollout policy**：$\mu=\pi_\theta$、$\pi_{\mathrm{old}}$、教师或混合策略；
2. **Teacher construction**：外部 teacher、privileged self-teacher、context teacher、delta teacher；
3. **Divergence**：FKL、RKL、JSD、adaptive KL；
4. **Estimator**：full-vocab stop-gradient、sampled-token PG、return-to-go、PPO clipped。

例如，标准 GKD 可以描述为：

> student rollout + external teacher + full-vocab JSD/FKL/RKL + stop-gradient trajectory。

MiniLLM 可以描述为：

> student/mixed rollout + external teacher + sequence Reverse KL + single-step exact gradient 与 future policy gradient。

---

## 第 2 章：为什么需要 On-Policy 数据

### 2.1 传统离线蒸馏的状态分布错位

传统监督蒸馏一般在固定数据前缀上训练：

$$
\mathcal L_{\mathrm{off}}
=
\mathbb E_{s\sim d_{\mathrm{data}}}
\left[D(q_s\|\pi_s)\right].
$$

推理时，学生访问的是：

$$
s\sim d_{\pi_\theta}.
$$

如果：

$$
d_{\mathrm{data}}\neq d_{\pi_\theta},
$$

那么模型训练时没有学过自己犯错后形成的前缀。

例如教师轨迹是：

```text
问题 -> 正确步骤 A -> 正确步骤 B -> 正确答案
```

学生推理轨迹可能是：

```text
问题 -> 错误步骤 A' -> 模糊步骤 B' -> 后续崩坏
```

离线 KD 只教过模型在“正确步骤 A”之后如何继续，却没有教过它在“A'”之后如何纠偏。

---

### 2.2 误差累积

> 模仿学习背景：[DAgger](https://arxiv.org/abs/1011.0686)。

假设每个时间步学生在训练分布上的错误率为 $\epsilon$。在序列决策中，一个早期错误会改变后续状态，因此总损失可能按比 $O(T\epsilon)$ 更坏的方式累积。DAgger 的经典思想是：

1. 执行当前学习者；
2. 收集学习者实际访问的状态；
3. 请求专家在这些状态上给正确动作；
4. 将数据聚合后继续训练。

OPD 可以视为语言模型中的 soft-label DAgger：

- 学生生成前缀；
- 教师不是只给一个正确 token，而是给整个 next-token distribution。

---

### 2.3 OPD 如何缓解 exposure bias

OPD 目标：

$$
\mathcal L_{\mathrm{on}}
=
\mathbb E_{s\sim d_{\pi_\theta}}
\left[D(q_s\|\pi_s)\right].
$$

随着学生参数更新：

$$
\pi_{\theta_0}
\rightarrow
\pi_{\theta_1}
\rightarrow
\pi_{\theta_2},
$$

训练状态分布也同步演化：

$$
d_{\pi_{\theta_0}}
\rightarrow
d_{\pi_{\theta_1}}
\rightarrow
d_{\pi_{\theta_2}}.
$$

这构成一个反馈循环：学生在哪里容易出错，教师就在哪里提供监督；学生改善后，后续 rollout 又进入更高质量的新状态。

---

## 第 3 章：按散度类型分类

### 3.1 On-policy Forward KL

目标：

$$
\mathcal L_{\mathrm{on-FKL}}
=
\mathbb E_{s_t\sim d_{\pi_\theta}^t}
\left[
D_{\mathrm{KL}}(q_t\|\pi_t)
\right].
$$

特点：

- 完整覆盖教师给出的高概率候选；
- 梯度为 $\pi-q$，稳定且低方差；
- 对教师分布 head 部分通常有较强监督；
- 当学生容量不足时，可能难以同时表达教师所有合理 mode。

注意：它不是 sequence-level Forward KL，因为外层状态来自学生。

---

### 3.2 On-policy Reverse KL

目标：

$$
\mathcal L_{\mathrm{on-RKL}}
=
\mathbb E_{s_t\sim d_{\pi_\theta}^t}
\left[
D_{\mathrm{KL}}(\pi_t\|q_t)
\right].
$$

特点：

- 主要修正学生已经分配概率的区域；
- 倾向于压制教师低概率而学生高概率的 token；
- 与最大熵 RL 和 teacher-likelihood reward 有自然联系；
- 若师生差异过大，学生可能难以发现教师的优质 mode；
- 高熵位置上纯 RKL 可能损失多样性。

---

### 3.3 JSD 与混合散度

Generalized JSD：

$$
D_{\mathrm{JSD}_\beta}(q\|\pi)
=
\beta D_{\mathrm{KL}}(q\|m)
+
(1-\beta)D_{\mathrm{KL}}(\pi\|m),
$$

其中：

$$
m=\beta q+(1-\beta)\pi.
$$

一个更直接的 FKL/RKL 线性混合为：

$$
\mathcal L_{\mathrm{mix}}
=
\lambda D_{\mathrm{KL}}(q\|\pi)
+
(1-\lambda)D_{\mathrm{KL}}(\pi\|q).
$$

JSD 的优点是有界并通过共同混合分布缓和支持集错位；线性混合更容易解释，但两个 KL 都可能在极小概率处产生较大数值。

---

### 3.4 Entropy-Aware OPD

> 主要来源：[Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079)。

教师 token 熵为：

$$
H(q_t)
=
-\sum_v q_t(v)\log q_t(v).
$$

Entropy-Aware OPD 的基本思想是：

- 教师低熵时，教师已有明确偏好，RKL 足以进行精确追随；
- 教师高熵时，存在多个合理 token，应加入 FKL 保留覆盖范围。

一种概念化写法是：

$$
\mathcal L_t
=
D_{\mathrm{KL}}(\pi_t\|q_t)
+
\lambda(H(q_t))
D_{\mathrm{KL}}(q_t\|\pi_t),
$$

其中 $\lambda(H)$ 随教师熵增大而增大。最简单的 hard gate 是：

$$
\lambda(H(q_t))
=
\lambda_0\mathbf 1[H(q_t)>\tau_H].
$$

这类方法试图在 mode-seeking 的精度与 mode-covering 的鲁棒性之间动态折中。

---

### 3.5 Adaptive KL

Adaptive KL 不再固定 FKL/RKL 权重，而是根据 token 分布、训练阶段或局部误差调整：

$$
\mathcal L_t
=
\alpha_t D_{\mathrm{KL}}(q_t\|\pi_t)
+
(1-\alpha_t)D_{\mathrm{KL}}(\pi_t\|q_t).
$$

其中 $\alpha_t$ 可依赖：

- 教师熵；
- 学生与教师的 top-1 是否一致；
- head/tail probability mass；
- pointwise log-ratio；
- 训练步数。

这种方法的本质是：不同 token 位置的分布结构不同，不应强迫所有位置使用同一个散度偏好。

---

## 第 4 章：按梯度计算方式分类

### 4.1 Full-vocabulary Stop-gradient Distillation

流程：

1. 学生 rollout 得到 $y$；
2. 将 $y$ 作为固定序列；
3. 教师和学生在相同前缀上 teacher-forcing；
4. 对每个位置计算全词表散度；
5. 只对学生分支反向传播。

伪代码：

```python
with torch.no_grad():
    rollout = student.generate(prompts)
    teacher_logits = teacher(prompts, rollout, privileged_context)

student_logits = student(prompts, rollout)

teacher_probs = softmax(teacher_logits, dim=-1)
student_log_probs = log_softmax(student_logits, dim=-1)

loss = forward_kl(teacher_probs, student_log_probs, token_mask)
loss.backward()
```

优点：

- 最像普通监督学习；
- 稳定、低方差；
- 支持任意 differentiable token divergence。

限制：

- 忽略 rollout 分布对 $\theta$ 的导数；
- teacher logits 成本高；
- 它优化的是 DAgger-style local surrogate，而不一定是完整 sequence objective。

---

### 4.2 Sampled-token Policy Gradient

学生采样 token 后，只计算：

$$
\log\pi_\theta(y_t\mid s_t),
\qquad
\log q(y_t\mid s_t).
$$

定义即时教师相对学生 reward：

$$
r_t
=
\log q(y_t\mid s_t)
-
\log\pi_{\mathrm{old}}(y_t\mid s_t).
$$

如果采用 token-local advantage：

$$
A_t=r_t-b_t,
$$

则 PPO-style 目标为：

$$
\mathcal L_t
=
-
\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right).
$$

优点：

- 教师只需返回 sampled token logprob；
- 通信规模从 $O(NV)$ 降为 $O(N)$；
- 可直接复用 PPO/GRPO rollout infra。

限制：

- 方差较大；
- token-local credit 不能把后续崩坏归因给前面的错误 token；
- 需要 behavior logprob 与严格的 mask 对齐。

---

### 4.3 Return-to-go Credit Assignment

定义：

$$
r_t
=
\log q(y_t\mid s_t)
-
\log\pi_\theta(y_t\mid s_t),
$$

$$
R_t
=
\sum_{k=t}^{T}r_k.
$$

更新：

$$
\mathcal L_{\mathrm{PG}}
=
-
\sum_t
\operatorname{stopgrad}(R_t-b_t)
\log\pi_\theta(y_t\mid s_t).
$$

它允许早期 token 接收后续影响。例如第 5 个 token 导致第 30 到 100 个 token 全部偏离教师，则 $R_5$ 会累积这些后续偏差。

常见变体包括：

- undiscounted return：$R_t=\sum_{k=t}^T r_k$；
- discounted return：$R_t=\sum_{k=t}^T\gamma^{k-t}r_k$；
- truncated return：只累计未来 $K$ 步；
- GAE-like estimator；
- sequence normalization 或 length normalization。

---

### 4.4 MiniLLM 的 Single-step Decomposition

MiniLLM 将 sequence Reverse-KL 梯度拆成：

$$
\nabla\mathcal L
=
(\nabla\mathcal L)_{\mathrm{Single}}
+
(\nabla\mathcal L)_{\mathrm{Long}}.
$$

其中：

- 单步项在词表上精确求和，降低 Monte Carlo 方差；
- 长期项使用 future return 的 policy gradient，保留序列 credit assignment。

概念化写法为：

$$
(\nabla\mathcal L)_{\mathrm{Single}}
=
-\sum_t
\nabla_\theta
\mathbb E_{a_t\sim\pi_t}
\left[
\log\frac{q_t(a_t)}{\pi_t(a_t)}
\right],
$$

$$
(\nabla\mathcal L)_{\mathrm{Long}}
=
-\mathbb E
\left[
\sum_t R_{t+1}
\nabla_\theta\log\pi_t(y_t)
\right].
$$

这比纯 sampled-token REINFORCE 更精确，也比完全忽略状态分布梯度的 stop-gradient GKD 更接近 sequence-level objective。

---

### 4.5 Pointwise Divergence Clipping

全词表散度可以分解成每个词表项的贡献。对一般 $f$-divergence：

$$
D_f(q\|\pi)
=
\sum_v q(v)
 f\left(\frac{\pi(v)}{q(v)}\right).
$$

定义 pointwise contribution：

$$
\ell_{t,v}^{(f)}
=
q_t(v)
 f\left(\frac{\pi_t(v)}{q_t(v)}\right).
$$

OPSD 提出对每个贡献裁剪：

$$
D_{\mathrm{clip}}^{(f)}
=
\frac1T
\sum_t\sum_v
\min(\ell_{t,v}^{(f)},\tau).
$$

动机是避免少数风格 token 或极端概率比主导整个训练信号。

---

## 第 5 章：按教师来源分类

### 5.1 外部教师 OPD

教师是独立模型：

$$
q_\phi(\cdot\mid x,y_{<t}).
$$

典型设置：

- teacher 更大；
- teacher 经过更强的 SFT/RL；
- student rollout；
- teacher 对学生轨迹做 teacher-forcing evaluation。

这是 GKD、MiniLLM 和大多数模型压缩型 OPD 的基础形式。

---

### 5.2 On-Policy Self-Distillation（OPSD）

> 主要来源：[Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734)。

同一个模型参数 $\theta$ 构造两个条件分布。

学生只看问题：

$$
p_S(\cdot\mid x)
:=p_\theta(\cdot\mid x).
$$

教师看到特权信息：

$$
p_T(\cdot\mid x,y^\star)
:=p_\theta(\cdot\mid x,y^\star).
$$

学生生成：

$$
\hat y\sim p_S(\cdot\mid x).
$$

两者在相同学生前缀上评估：

$$
p_S(\cdot\mid x,\hat y_{<t}),
\qquad
p_T(\cdot\mid x,y^\star,\hat y_{<t}).
$$

目标：

$$
\mathcal L_{\mathrm{OPSD}}
=
\mathbb E_{(x,y^\star)}
\mathbb E_{\hat y\sim p_S}
\left[
\frac1T\sum_t
D\left(
\operatorname{sg}[p_T]\|p_S
\right)
\right].
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。

本质上这是 privileged-information distillation：

$$
p_\theta(a\mid s,\text{solution})
\longrightarrow
p_\theta(a\mid s).
$$

模型不是从真空里创造知识，新信息来自 $y^\star$、工具反馈或环境信息。

---

### 5.3 On-Policy Context Distillation

> 主要来源：[On-Policy Context Distillation for Language Models](https://arxiv.org/abs/2602.12275)。

教师额外看到上下文 $c$：

$$
q(\cdot\mid x,c,y_{<t}),
$$

学生只看到：

$$
\pi_\theta(\cdot\mid x,y_{<t}).
$$

上下文可以是：

- 检索文档；
- system prompt；
- 完整长上下文；
- 工具调用轨迹；
- 历史经验；
- 参考解法。

目标是将上下文带来的行为差异压入参数：

$$
q(\cdot\mid s,c)
\longrightarrow
\pi_\theta(\cdot\mid s).
$$

On-Policy Context Distillation（OPCD）使用学生自己的轨迹，并常以 Reverse KL 对齐 context-conditioned teacher。

---

### 5.4 Historical / EMA Teacher

教师可以是：

- 当前学生的 EMA copy；
- 固定 reference checkpoint；
- 前一训练阶段 checkpoint；
- 更强 RL checkpoint。

例如：

$$
q_{\bar\theta},
\qquad
\bar\theta
\leftarrow
\alpha\bar\theta+(1-\alpha)\theta.
$$

这类教师能降低 target drift，但也可能使监督滞后。

---

### 5.5 Delta Teacher / On-Policy Delta Distillation

> 主要来源：[On-Policy Delta Distillation](https://arxiv.org/abs/2607.15161)。

标准 OPD reward 常写为：

$$
R_t^{\mathrm{OPD}}(v)
=
\log q_{\mathrm{teacher},t}(v)
-
\log\pi_{\theta,t}(v).
$$

On-Policy Delta Distillation（OPD$^2$）引入 teacher base model：

$$
q_{\mathrm{base}}.
$$

定义 delta signal：

$$
R_t^\Delta(v)
=
\log q_{\mathrm{teacher},t}(v)
-
\log q_{\mathrm{base},t}(v).
$$

它试图提取教师在 reasoning post-training 中相对 base model 学到的增量，而不是复制教师的全部语言风格与基础知识。

如果需要 zero-centered advantage，可写为：

$$
A_t^\Delta(v)
=
R_t^\Delta(v)
-
\mathbb E_{u\sim\pi_t}
\left[R_t^\Delta(u)\right].
$$

其直觉是：

> 不问教师“喜欢什么”，而问 reasoning tuning 让教师“比原来更喜欢什么”。

---

## 第 6 章：代表性方法

### 6.1 GKD：Generalized Knowledge Distillation

> 原始论文：[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)。

GKD 引入两个独立设计维度：

1. 学生生成数据占比 $\lambda$；
2. token-level divergence $D$。

其混合目标可以写为：

$$
\begin{aligned}
\mathcal L_{\mathrm{GKD}}
=&
(1-\lambda)
\mathbb E_{(x,y)\sim\mathcal D}
\left[D(q\|\pi)(y\mid x)\right]
\\
&+
\lambda
\mathbb E_{x\sim\mathcal D_x}
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[D(q\|\pi)(y\mid x)\right].
\end{aligned}
$$

其中：

- $\lambda=0$：纯 supervised/off-policy KD；
- $0<\lambda<1$：混合 KD；
- $\lambda=1$：纯 on-policy KD。

GKD 通常不通过采样过程反向传播，因此可视为 on-policy data collection + supervised distribution matching。

---

### 6.2 MiniLLM

> 原始论文：[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)。

MiniLLM 直接最小化序列分布上的 Reverse KL：

$$
\min_\theta
D_{\mathrm{KL}}
\left(
\pi_\theta(y\mid x)
\middle\|
q(y\mid x)
\right).
$$

核心组件包括：

- sequence-level Reverse KL；
- on-policy Monte Carlo sampling；
- single-step exact decomposition；
- future return policy gradient；
- teacher-mixed sampling；
- length normalization。

Teacher-mixed sampling 形式为：

$$
\tilde p(y_t\mid s_t)
=
\alpha q(y_t\mid s_t)
+
(1-\alpha)\pi_\theta(y_t\mid s_t).
$$

它能把小学生从严重退化区域拉回教师支持集附近，但严格来说不再是纯学生 on-policy。

---

### 6.3 Self-Distilled Reasoner / OPSD

> 原始论文：[Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)。

OPSD 使用同一个模型构建 privileged teacher 和 inference-condition student：

$$
p_T(\cdot\mid x,y^\star),
\qquad
p_S(\cdot\mid x).
$$

学生 rollout 后，优化 full-vocabulary token divergence：

$$
\mathcal L
=
\mathbb E_{\hat y\sim p_S}
\left[
\frac1T\sum_t
D\left(
\operatorname{sg}[p_T(\cdot\mid x,y^\star,\hat y_{<t})]
\middle\|
 p_S(\cdot\mid x,\hat y_{<t})
\right)
\right].
$$

其重要设计包括：

- privileged solution conditioning；
- full-vocabulary dense feedback；
- pointwise divergence clipping；
- 无需外部大 teacher。

需要警惕 privileged information leakage：教师可能通过格式、答案 token 或近似复制参考解法提供过强捷径，而学生未必真正学到可泛化的推理过程。

---

### 6.4 Entropy-Aware OPD

> 原始论文：[Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079)。

Entropy-Aware OPD 在标准 Reverse-KL OPD 上，对高教师熵位置增加 Forward KL：

$$
\mathcal L_t
=
D_{\mathrm{KL}}(\pi_t\|q_t)
+
\lambda_t D_{\mathrm{KL}}(q_t\|\pi_t),
$$

其中 $\lambda_t$ 由 $H(q_t)$ 决定。

目的：

- 低熵位置继续使用 RKL 追随明确 mode；
- 高熵位置通过 FKL 保留多个合理候选。

---

### 6.5 Trust-Region On-Policy Distillation

> 原始论文：[Trust Region On-Policy Distillation](https://arxiv.org/abs/2606.01249)。

Trust-Region OPD 关注师生分布差异过大时的训练不稳定。

一种自然的可靠区域判定是：

$$
\mathcal R_t
=
\mathbf 1
\left[
D(q_t,\pi_t)\le \tau_D
\right]
$$

或基于 sampled-token log-ratio：

$$
\mathcal R_t
=
\mathbf 1
\left[
\left|
\log q_t(y_t)-\log\pi_t(y_t)
\right|
\le\tau_r
\right].
$$

可靠区域使用标准 OPD，异常区域采用：

- gradient clipping；
- masking；
- Forward-KL fallback；
- teacher-prefix off-policy guidance。

其核心观点是：

> 教师在学生严重偏离的状态上未必能提供稳定、可学习的局部梯度，不应对所有状态一视同仁。

---

### 6.6 $\beta$-OPSD

> 原始论文：[$\beta$-OPSD: Deriving with Policy Optimization, Training with Self-Distillation](https://arxiv.org/abs/2607.28582)。

$\beta$-OPSD 从 KL-regularized policy optimization 出发：

$$
\max_\pi
\mathbb E_{y\sim\pi}
\left[
R(y;x,c)
\right]
-
\beta
D_{\mathrm{KL}}
\left(
\pi\|\pi_{\mathrm{ref}}
\right).
$$

选择 teacher-to-reference log-ratio reward：

$$
R(y;x,c)
=
\log
\frac{p_T(y\mid x,c)}
{\pi_{\mathrm{ref}}(y\mid x)}.
$$

得到：

$$
\mathcal J_\beta(\pi)
=
\mathbb E_{y\sim\pi}
\left[
\log\frac{p_T(y\mid x,c)}
{\pi_{\mathrm{ref}}(y\mid x)}
\right]
-
\beta D_{\mathrm{KL}}
\left(
\pi\|\pi_{\mathrm{ref}}
\right).
$$

其闭式最优策略为：

$$
\pi_\beta^\star(y\mid x,c)
=
\frac{
\pi_{\mathrm{ref}}(y\mid x)^{1-1/\beta}
 p_T(y\mid x,c)^{1/\beta}
}{Z_\beta(x,c)}.
$$

这是一种 reference 与 privileged teacher 的几何插值。

- $\beta=1$：$\pi_\beta^\star=p_T$，恢复 vanilla OPSD teacher endpoint；
- $\beta\to\infty$：$\pi_\beta^\star\to\pi_{\mathrm{ref}}$；
- $\beta>1$：在 reference 稳定性和 teacher guidance 之间折中。

由于序列归一化常数不可计算，实践中使用局部 logits 插值近似：

$$
\tilde z_t
=
\left(1-\frac1\beta\right)
 z_{\mathrm{ref},t}
+
\frac1\beta z_{T,t},
$$

$$
\tilde q_{\beta,t}
=
\operatorname{softmax}(\tilde z_t).
$$

然后蒸馏学生去逼近 $\tilde q_{\beta,t}$。该方法还结合 return-to-go，以提高 token-level 更新与 sequence-level 目标的一致性。

---

## 第 7 章：OPD、SFT、离线 KD 与 RLVR 的关系

### 7.1 SFT

SFT 目标：

$$
\mathcal L_{\mathrm{SFT}}
=
-
\mathbb E_{(x,y^\star)\sim\mathcal D}
\left[
\sum_t
\log\pi_\theta(y_t^\star\mid x,y_{<t}^\star)
\right].
$$

特点：

- 固定专家状态；
- 单一 hard target；
- 低方差；
- 存在 exposure bias。

---

### 7.2 离线 Token-level KD

$$
\mathcal L_{\mathrm{offline-KD}}
=
\mathbb E_{(x,y)\sim\mathcal D}
\left[
\frac1T\sum_t
D_{\mathrm{KL}}(q_t\|\pi_t)
\right].
$$

特点：

- 固定前缀；
- 教师 soft distribution；
- 比 SFT 提供更丰富监督；
- 仍有状态分布错位。

---

### 7.3 RLVR / GRPO

一般 RLVR 目标：

$$
\max_\theta
\mathbb E_{y\sim\pi_\theta}
\left[r(x,y)\right].
$$

GRPO 使用组内标准化 advantage：

$$
A_i
=
\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}.
$$

特点：

- on-policy 或 near-on-policy；
- reward 常是序列级稀疏信号；
- 可超过给定参考答案风格；
- credit assignment 和采样成本高。

---

### 7.4 OPD

$$
\mathcal L_{\mathrm{OPD}}
=
\mathbb E_{y\sim\pi_\theta}
\left[
\sum_t D(q_t,\pi_t)
\right].
$$

特点：

- 使用学生状态；
- dense token-level feedback；
- 比纯 outcome reward 更高 sample efficiency；
- 上限受教师、特权信息和师生兼容性限制。

可以概括为：

$$
\boxed{
\text{OPD}
=
\text{RL 的学生状态分布}
+
\text{KD 的密集分布监督}
}
$$

---

### 7.5 方法对比

| 方法 | 状态来源 | 监督粒度 | 是否需要教师 | 主要优点 | 主要问题 |
|---|---|---:|---:|---|---|
| SFT | 固定参考轨迹 | hard token | 否 | 稳定简单 | exposure bias |
| Offline KD | 固定/教师轨迹 | full-vocab token | 是 | dense supervision | 状态错位 |
| GKD/OPD | 学生轨迹 | full-vocab token | 通常是 | 修复学生状态 | teacher 成本高 |
| Sampled OPD | 学生轨迹 | sampled token | 是 | 通信低 | 高方差 |
| OPSD | 学生轨迹 | full-vocab token | 无外部教师 | 利用特权答案 | 信息泄漏风险 |
| RLVR/GRPO | 学生轨迹 | sequence reward | 否/奖励器 | 直接优化结果 | 稀疏、高采样成本 |

---

## 第 8 章：工程实现

### 8.1 推荐训练流水线

一个稳定的 reasoning OPD pipeline：

1. **Cold start**：先用 SFT 或 offline KD，使学生进入合理 reasoning support；
2. **Rollout**：SGLang/vLLM 等服务侧使用 $\pi_{\mathrm{old}}$ 生成学生轨迹；
3. **Teacher evaluation**：教师在相同学生 token 序列上 teacher-forcing；
4. **Loss construction**：选择 full-vocab divergence 或 sampled-token advantage；
5. **Optimization**：Megatron/FSDP 侧更新学生；
6. **Weight synchronization**：训练权重同步到 rollout engine；
7. **Diagnostics**：监控策略滞后、KL、熵、token mask、teacher-student disagreement。

---

### 8.2 Full-vocabulary 实现模板

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute teacher||student token-level KL on fixed trajectories.

    Shapes:
        student_logits: [batch, seq, vocab]
        teacher_logits: [batch, seq, vocab]
        loss_mask:      [batch, seq]
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher/student logits must have the same shape")
    if student_logits.shape[:-1] != loss_mask.shape:
        raise ValueError("loss_mask shape must match logits without vocab dim")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    tau = temperature
    student_logp = F.log_softmax(student_logits / tau, dim=-1)

    with torch.no_grad():
        teacher_logp = F.log_softmax(teacher_logits / tau, dim=-1)
        teacher_p = teacher_logp.exp()

    token_kl = torch.sum(
        teacher_p * (teacher_logp - student_logp), dim=-1
    )

    mask = loss_mask.to(token_kl.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return tau * tau * torch.sum(token_kl * mask) / denominator
```

关键点：

- teacher logits 必须 stop-gradient；
- prompt token、padding token、图像 token 与不可训练位置需要正确 mask；
- packed sequence 中必须防止跨样本 attention 和 loss 串扰；
- teacher 与 student 的 tokenizer、chat template 和 prefix 必须严格一致。

---

### 8.3 Full-vocabulary Reverse KL

```python
def masked_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    student_logp = F.log_softmax(student_logits, dim=-1)
    student_p = student_logp.exp()

    with torch.no_grad():
        teacher_logp = F.log_softmax(teacher_logits, dim=-1)

    token_kl = torch.sum(
        student_p * (student_logp - teacher_logp), dim=-1
    )
    mask = loss_mask.to(token_kl.dtype)
    return torch.sum(token_kl * mask) / mask.sum().clamp_min(1.0)
```

数值上应对 teacher log-prob 做有限值检查。理论 softmax 概率大于 0，但低精度、top-k 截断或量化接口可能产生 $-\infty$。

---

### 8.4 Sampled-token PPO-style 实现

```python
def sampled_opd_loss(
    current_logp: torch.Tensor,
    old_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-style sampled-token OPD surrogate.

    All log-prob tensors have shape [batch, seq] and correspond to the
    same sampled response tokens.
    """
    with torch.no_grad():
        advantage = teacher_logp - old_logp
        mask = loss_mask.to(advantage.dtype)

        valid_adv = advantage[loss_mask.bool()]
        if valid_adv.numel() > 1:
            advantage = (
                advantage - valid_adv.mean()
            ) / valid_adv.std(unbiased=False).clamp_min(1e-6)

    ratio = torch.exp(current_logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(
        ratio, 1.0 - clip_eps, 1.0 + clip_eps
    ) * advantage

    token_loss = -torch.minimum(unclipped, clipped)
    return torch.sum(token_loss * mask) / mask.sum().clamp_min(1.0)
```

这是实用 surrogate，而不是 full-vocab RKL 的逐点等价替代。

---

### 8.5 Rollout/Train Logprob 对齐

如果 rollout engine 和 training engine 不完全一致，应监控：

$$
\Delta_t^{\mathrm{rollout/train}}
=
\log\pi_{\mathrm{train}}(y_t\mid s_t)
-
\log\pi_{\mathrm{rollout}}(y_t\mid s_t).
$$

建议统计：

- mean / median / p95 / p99 / max absolute difference；
- sequence summed log-ratio；
- importance ratio $\rho_t$ 分布；
- clipped token 比例；
- 不同 token 类型的差异；
- 不同 sequence length bucket 的差异。

如果 $|\Delta_t|$ 持续较大，所谓“on-policy”已经退化为 off-policy 近似。

常见来源：

- 权重同步滞后；
- sampling temperature/top-p 不一致；
- logits processor 不一致；
- tokenizer/chat template 不一致；
- TP/EP kernel 数值差异；
- train/rollout 使用不同精度；
- image preprocessing 或位置编码不一致。

---

### 8.6 Teacher Logits 的系统成本

全词表 teacher logits 的元素数量：

$$
N_{\mathrm{elem}}
=
B\times T\times V.
$$

若使用 BF16，每个元素 2 字节：

$$
M_{\mathrm{logits}}
\approx
2BTV\ \text{bytes}.
$$

例如：

- $B=64$；
- $T=4096$；
- $V=150000$。

则仅一份 logits 大约：

$$
64\times4096\times150000\times2
\approx 78.6\ \text{GB}.
$$

因此工程上常用：

- token chunking；
- vocabulary parallel loss；
- teacher/student 同机流水；
- top-$k$ logits + residual mass；
- sampled-token logprob；
- 在线消费 teacher logits，避免完整落盘；
- recompute 交换显存。

---

### 8.7 Top-k Teacher Approximation

若只保留教师 top-$k$ 集合 $S_k$，其剩余概率质量为：

$$
r
=
1-
\sum_{v\in S_k}q(v).
$$

一种近似是：

- 精确保留 $S_k$ 中的教师概率；
- 将尾部质量 $r$ 聚合成一个 bucket；
- 或按学生尾部分布重新分配。

但对 Reverse KL 而言，学生可能在教师 top-$k$ 外分配大量概率，因此简单丢弃 tail 会严重低估：

$$
\sum_{v\notin S_k}
\pi(v)
\log\frac{\pi(v)}{q(v)}.
$$

所以 top-$k$ 压缩通常对 FKL 更容易处理，对 RKL 需要额外 tail correction。

---

### 8.8 长序列与长度归一化

不归一化序列 loss：

$$
\mathcal L(y)
=
\sum_{t=1}^{T}\ell_t
$$

会让长序列天然具有更大梯度权重。常见归一化：

#### Token average

$$
\mathcal L(y)
=
\frac1T\sum_t\ell_t.
$$

#### Batch-global token average

$$
\mathcal L
=
\frac{\sum_i\sum_t m_{i,t}\ell_{i,t}}
{\sum_i\sum_t m_{i,t}}.
$$

#### Sequence-balanced average

$$
\mathcal L
=
\frac1B\sum_i
\frac{1}{T_i}
\sum_t\ell_{i,t}.
$$

三者语义不同：batch-global token average 更偏向长样本，sequence-balanced average 让每个 prompt 权重相等。

---

## 第 9 章：训练诊断与方法选择

### 9.1 必须监控的指标

#### 分布指标

- token FKL / RKL / JSD；
- teacher entropy 与 student entropy；
- top-1 agreement；
- top-$k$ probability mass overlap；
- sampled-token teacher/student log-ratio；
- pointwise divergence 的 p95/p99/max。

#### On-policy 程度

- rollout/train logprob difference；
- importance ratio 分布；
- rollout weight staleness；
- PPO clipping fraction。

#### 训练有效性

- grad norm；
- non-zero loss token ratio；
- teacher/student KL 是否下降；
- reward/accuracy 是否上升；
- response length 与 EOS rate；
- repetition rate；
- high-entropy token 上的 diversity。

#### 系统指标

- teacher forward latency；
- rollout throughput；
- logits 通信带宽；
- train/rollout GPU utilization；
- 每有效训练 token 的成本。

---

### 9.2 常见失败模式

#### 失败模式 1：学生访问不到教师 mode

表现：

- RKL 降不下去；
- sampled-token teacher logprob 极低；
- 生成长期停留在错误 reasoning family。

处理：

- 增加 SFT/offline KD cold start；
- teacher-mixed sampling；
- 增加 FKL/JSD；
- teacher-prefix continuation；
- 使用 trust-region mask。

#### 失败模式 2：纯 RKL 导致多样性下降

表现：

- student entropy 快速下降；
- 多次采样高度同质化；
- 高熵教师位置被压成单一 token。

处理：

- Entropy-Aware FKL；
- JSD；
- entropy bonus；
- 提高 rollout temperature，但要单独验证 train/inference mismatch。

#### 失败模式 3：教师输出主导风格而非推理

表现：

- KL 下降，但 correctness 不涨；
- 高 divergence token 主要是连接词、格式词、标点；
- 模型模仿长篇措辞，却没有改善关键计算步骤。

处理：

- pointwise clipping；
- reasoning span mask；
- delta distillation；
- error-localized distillation；
- 对答案、格式、思考内容分开加权。

#### 失败模式 4：OPSD 特权信息泄漏

表现：

- teacher 几乎直接复述答案；
- 训练 loss 很低，out-of-domain 泛化差；
- 去掉 reference solution 后能力消失。

处理：

- 限制 teacher prompt 中的答案暴露方式；
- 使用 verifier/error feedback 而非完整解法；
- 加入 RLVR 或真实环境反馈；
- 对 teacher/student 表达风格做控制；
- 评估反事实和 OOD 样本。

#### 失败模式 5：错误实现 sampled RKL

表现：

- 使用 `student_logp - teacher_logp` 直接 backward；
- loss 数值变化，但期望更新接近零或方向异常；
- grad 主要来自其他辅助项。

处理：

- full-vocab RKL；
- 正确 REINFORCE/PPO surrogate；
- 单元测试解析梯度与 Monte Carlo 梯度；
- 检查 advantage 是否 stop-gradient。

---

### 9.3 方法选择建议

#### 学生与教师接近，能承受 full logits

优先：

$$
\text{student rollout}
+
\text{full-vocab JSD/RKL}
+
\text{stop-gradient trajectory}.
$$

理由：稳定、实现简单，通常是最可靠 baseline。

#### 师生容量差距大

优先：

- SFT/offline KD cold start；
- GKD 混合学生与教师/数据轨迹；
- JSD 或 FKL；
- trust-region fallback。

不要一上来只用 sampled RKL。那相当于让一个刚学会走路的学生，只在自己误入的荒地里寻找老师留下的脚印。

#### 词表大、序列长、通信受限

优先：

- sampled-token OPD；
- token-level PPO clipping；
- return-to-go 或 truncated return；
- 记录 old logprob；
- 强化 rollout/train 对齐检查。

#### 有 verified solution，但不想维护外部教师

优先：

- OPSD；
- privileged context teacher；
- pointwise clipping；
- 防止答案泄漏；
- 与 RLVR 混合验证。

#### 想只迁移 reasoning tuning 增量

优先：

- OPD$^2$ / delta signal；
- teacher 与 teacher-base 必须 tokenizer/架构兼容；
- 注意额外 teacher-base forward 成本。

#### OPSD 训练不稳定

优先：

- $\beta$-OPSD reference interpolation；
- trust-region mask；
- 较大 $\beta$ 起步，再逐步靠近 teacher；
- return-to-go 与局部 clipping。

---

## 第 10 章：总结

On-Policy Distillation 的统一表达是：

$$
\boxed{
\underbrace{s_t\sim d_\mu^t}_{\text{在哪里学习}}
+
\underbrace{q(\cdot\mid s_t,c)}_{\text{向谁学习}}
+
\underbrace{D(q_t,\pi_t)}_{\text{如何对齐}}
+
\underbrace{\widehat{\nabla_\theta\mathcal L}}_{\text{如何估计梯度}}
}
$$

四部分分别回答：

1. **采样分布**决定学生在哪些状态接受监督；
2. **教师构造**决定监督中包含什么额外信息；
3. **散度方向**决定学生如何处理教师的 head、tail 和多模态结构；
4. **梯度估计**决定训练是低方差的 local surrogate，还是更接近完整 sequence objective。

最重要的数学结论包括：

1. on-policy 描述状态分布，不描述 KL 方向；
2. sequence Reverse KL 精确分解在学生状态分布上；
3. sequence Forward KL 精确分解在教师状态分布上；
4. 学生状态上的 Forward KL 是 DAgger-style surrogate，不等于 sequence Forward KL；
5. sampled-token log-ratio 不能简单当作普通监督 loss 直接反向传播；
6. full-vocab stop-gradient OPD 低方差但忽略状态分布梯度；
7. return-to-go 能将未来偏差归因给早期 token，但会提高方差；
8. OPSD 的新增信息来自 privileged context，而不是模型凭空“自我进化”；
9. 在真实系统中，权重滞后和训推 logprob diff 会决定方法究竟有多 on-policy。

一个务实的默认 recipe 是：

$$
\boxed{
\text{SFT/offline KD cold start}
\rightarrow
\text{student rollout}
\rightarrow
\text{teacher evaluate same prefixes}
\rightarrow
\text{full-vocab JSD/RKL baseline}
\rightarrow
\text{按成本切换 sampled PG}
\rightarrow
\text{必要时加入 FKL、trust region、reference 与 RTG}
}
$$

---

## 参考文献

以下优先列出原始论文或官方论文页面。2026 年方法多为近期 arXiv 预印本，结论仍可能随版本更新。

### 基础知识蒸馏、序列蒸馏与模仿学习

1. Hinton, G., Vinyals, O., & Dean, J. **Distilling the Knowledge in a Neural Network**. 2015.  
   https://arxiv.org/abs/1503.02531

2. Kim, Y., & Rush, A. M. **Sequence-Level Knowledge Distillation**. EMNLP 2016.  
   https://arxiv.org/abs/1606.07947  
   https://aclanthology.org/D16-1139/

3. Ross, S., Gordon, G. J., & Bagnell, J. A. **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**. AISTATS 2011.  
   https://arxiv.org/abs/1011.0686  
   https://proceedings.mlr.press/v15/ross11a.html

4. Williams, R. J. **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning**. Machine Learning, 1992.  
   https://link.springer.com/article/10.1007/BF00992696

5. Wen, Y., Li, Z., Du, W., & Mou, L. **f-Divergence Minimization for Sequence-Level Knowledge Distillation**. ACL 2023.  
   https://arxiv.org/abs/2307.15190

### 核心 On-Policy Distillation

6. Agarwal, R., Vieillard, N., Zhou, Y., et al. **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**. ICLR 2024.  
   https://arxiv.org/abs/2306.13649  
   重点：GKD、学生 rollout、Forward/Reverse KL 与 generalized JSD、stop-gradient sampling。

7. Gu, Y., Dong, L., Wei, F., & Huang, M. **MiniLLM: Knowledge Distillation of Large Language Models**. ICLR 2024.  
   https://arxiv.org/abs/2306.08543  
   重点：sequence-level Reverse KL、policy-gradient 推导、single-step decomposition、teacher-mixed sampling。

8. Wu, T., Tao, C., Wang, J., et al. **Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models**. 2024.  
   https://arxiv.org/abs/2404.02657  
   重点：对简单 mode-seeking/mode-covering 叙事的修正、FKL/RKL 的 head-tail 训练动态、Adaptive KL。

### On-Policy Self-Distillation 与 2026 年扩展

9. Zhao, S., Xie, Z., Liu, M., et al. **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models**. 2026.  
   https://arxiv.org/abs/2601.18734  
   重点：privileged solution teacher、full-vocabulary OPSD、pointwise divergence clipping。

10. Ye, T., et al. **On-Policy Context Distillation for Language Models**. 2026.  
    https://arxiv.org/abs/2602.12275  
    重点：学生轨迹上的 context-conditioned teacher 与 Reverse-KL distillation。

11. Jin, W., Min, T., Yang, Y., et al. **Entropy-Aware On-Policy Distillation of Language Models**. 2026.  
    https://arxiv.org/abs/2603.07079  
    重点：高教师熵位置加入 Forward KL，平衡精确性与多样性。

12. Li, Y., et al. **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**. 2026.  
    https://arxiv.org/abs/2604.13016  
    重点：OPD 成功条件、师生兼容性、训练动态和经验配方。

13. Xing, X., Wang, H., Gao, B., et al. **Trust Region On-Policy Distillation**. 2026.  
    https://arxiv.org/abs/2606.01249  
    重点：师生分布差异较大时的可靠区域、outlier clipping/masking 与 off-policy guidance。

14. Heo, B., Hwang, J., Yun, S., & Han, D. **On-Policy Delta Distillation**. 2026.  
    https://arxiv.org/abs/2607.15161  
    重点：teacher 与 teacher-base 的 log-prob delta signal，迁移 reasoning tuning 增量。

15. Xu, J., Liu, M., Zhang, J., Goldstein, T., & Huang, F. **$\beta$-OPSD: Deriving with Policy Optimization, Training with Self-Distillation**. 2026.  
    https://arxiv.org/abs/2607.28582  
    重点：KL-regularized policy optimization、reference-teacher 几何插值、局部 logit realization、return-to-go。

### 综述与补充材料

16. Song, M., et al. **A Survey of On-Policy Distillation for Large Language Models**. 2026.  
    https://arxiv.org/abs/2604.00626

17. Lu, K., & Thinking Machines Lab. **On-Policy Distillation**. 2025-10-27; updated in 2026.  
    https://thinkingmachines.ai/blog/on-policy-distillation/  
    适合作为 sampled-token OPD 与 RL-style 实现的工程补充材料，不替代原始论文。

---

### 引用说明

- 本文中的数学统一框架包含对多篇论文记号的重新整理，未完全沿用任一论文的字母命名。
- “Forward KL / Reverse KL”统一以教师 $q$ 和学生 $\pi$ 为基准。
- 第 0.9、0.11、0.12 节中的链式分解和梯度公式是基于自回归概率分解与 log-derivative trick 的统一推导。
- 2026 年工作截至本文日期多数仍是预印本，应用时应检查 arXiv 最新版本与作者代码。
