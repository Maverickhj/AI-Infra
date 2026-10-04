---
type: knowledge
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# 从固定轨迹到一次 GRPO / PPO 更新

> 本章是 authored trajectory 的 CPU 数值参考。四条 response 和 reward 由人编写，没有调用生成引擎。两层完整 decoder 提供冻结特征，只训练 LM head 和独立 critic head。数值、梯度和 SGD 更新经过 CPU PyTorch 对照；这不证明真实 NeMo RL rollout、跨引擎 refit 或模型收益。

## 先定位整模、批次与版本

复用 G03 的 embedding → 两层 GQA / residual / SwiGLU / residual → final RMSNorm；参数和输入固定，得到每个 prefix 的 $x_{ij}\in\mathbb R^8$。策略 head 为 $W\in\mathbb R^{27\times8}$，critic head 为 $u\in\mathbb R^8$：

$$
z_{ij}=Wx_{ij},\qquad
\ell_{ij}=\log\operatorname{softmax}(z_{ij})_{a_{ij}},\qquad
v_{ij}=u^\top x_{ij}.
$$

backbone 冻结仍然是完整模型前向，不是全参数微调。两个 head 不共享可训练参数。PPO 的旧 value、advantage 和 returns 固定，不让 actor loss 反传进 critic。

两组 prompt 各两条 authored response：奖励分别 1/0 和 0.5/0.5。有效 response 长度为 2/3/2/2，总有效 token 数为 9；最后一条物理 response 长度为 4，中间两格 mask=0。奖励是给定标量，不能从轨迹名字推断模型答题正确率。“全部同分”将奖励设为 0.5，仅用于反例。

generation 和 previous 使用 authored version 0；常规 current 为另一组 authored version 1。version 1 **不是声称由历史优化器训练得到**。本页演算一次 version 1→2 的 SGD；force-on-policy 从与 generation/previous 相同的 version 0 开始，更新到 1。

## 独立数学符号表

| 符号 | 含义与 shape |
|---|---|
| $i,j$ | trajectory 与完整 token 序列的 action 位置，均从 0 计数 |
| $B,S,H,V$ | 4 条轨迹，右 padding 后 13 格，隐藏维 8，词表 27 |
| $a_{ij},m_{ij}$ | 固定 action ID 与有效 response mask，均 $[B,S]$ |
| $x_{ij},W,u$ | prefix 特征 $[H]$、策略 head $[V,H]$、critic head $[H]$ |
| $\ell^g,\ell^p,\ell^\theta,\ell^r$ | generation、previous、current、reference 的 action logprob，均 $[B,S]$ |
| $R_i,\mu_g,s_g$ | 序列奖励、prompt 组均值与样本标准差 |
| $A_{ij},r_{ij}$ | 固定 advantage 与 current/previous probability ratio |
| $\epsilon,\beta$ | ratio clip 半径 0.2、KL 系数 0 或 0.02 |
| $\gamma,\lambda$ | GAE 折扣 0.9、衰减 0.95；手算另用 $\lambda=0.8$ |
| $v^{old},v^\phi,T$ | 旧 value、当前 critic prediction 与固定 return，均 $[B,S]$ |
| $\delta_k$ | 相邻有效 action 间的 TD residual |
| $N_i,N$ | 每条有效 token 数与全批有效 token 数；$N=\sum_iN_i$ |
| $\eta_\theta,\eta_\phi$ | actor/critic SGD 学习率 0.08/0.05 |
| $\operatorname{sg}$ | stop-gradient：前向不改值，反向导数为零 |

本例 CPU float64 的绝对/相对容差均为 $10^{-10}$，不能直接移用为 BF16 或分布式训练容差。

## 四类 logprob 对齐同一个 action

action 在完整序列位置 $j$，预测它的是 prefix 结尾 $j-1$ 的 final norm 特征。$j=0$ 为不参与 loss 的 dummy 槽位。prompt、padding 和显式排除的 response 槽位 mask=0；其 logprob 可以存在，但梯度必须为零。

| 字段 | 权重与用途 | 当前梯度 |
|---|---|---|
| generation | 采样时的生成 policy；本页仅在固定 action 上计算参考值 | 无 |
| previous | 批次更新前重算，固定在本轮更新期间 | 无 |
| current | 当前可训练 policy 前向 | 有 |
| reference | 固定参考 policy，用于指定 KL 分支 | 无 |

固定 R-LOSS 的完整字段保存 $[B,S]$，损失入口对 mask、advantage、prev/generation/reference 取第二格开始的切片，与 forward 已取 next-token 的 $[B,S-1]$ 对齐。参考保留开头 dummy 便于查看，但 mask=0，求和等价。不能再 shift 一次，也不能把位置相同误当作概率相等。

生产 generation 与 previous 可能因引擎、过滤、温度或旧版本不同而不等。本页无 top-k/top-p 过滤、温度 1、无 actor off-policy correction；generation=previous 是参考设定，不是 RL 恒等式。

## GRPO：组内奖励得到优势

本例关闭 leave-one-out；同 prompt 的 $n_g>1$ 条有效轨迹：

$$
\mu_g=\frac1{n_g}\sum_{i\in g}R_i,\qquad
s_g^2=\frac1{n_g-1}\sum_{i\in g}(R_i-\mu_g)^2.
$$

$$
A_{ij}=m_{ij}
\begin{cases}
(R_i-\mu_g)/(s_g+10^{-6}),&s_g>0,\\
R_i-\mu_g,&s_g=0.
\end{cases}
$$

R-UTIL 用 $E[R^2]-E[R]^2$ 再乘 $n_g/(n_g-1)$ 完成 Bessel 修正，不能换成总体标准差。R-ADV 先广播序列优势；参考显示时再乘 mask，最终 loss 有效位置相同。

手算 $[1,0]$：均值 0.5，标准差 $\sqrt{0.5}$，优势约 $[0.70710578,-0.70710578]$。同分 $[0.5,0.5]$ 或 singleton 的优势为零。GRPO 无 critic 更新。同分仅使 PG 项为零；开启 reference KL 后仍可有 KL 梯度。

## PPO：terminal reward、GAE 与独立 critic

将 reward 放到最后一个 mask=1 的 action，其他位置 reward 为零。本例关闭 reward 中的 KL、advantage whitening、decoupled lambda，不对截断 episode bootstrap；terminal 后 value=0。

有效 action 编号为 $k=0,\ldots,N_i-1$，masked gap 不算新的环境步：

$$
\delta_k=r^{reward}_k+\gamma v^{old}_{k+1}-v^{old}_k,\qquad
A_k=\sum_{q=k}^{N_i-1}(\gamma\lambda)^{q-k}\delta_q,\qquad
T_k=A_k+v^{old}_k.
$$

Python 用 reverse carry，TS 独立直接求有限和。R-ADV 在 mask=0 处保留 next_values/last_gae_lam，最终只把 masked advantage 清零；masked returns 可非零，value loss 必须继续用 mask。跳过 mask 后额外乘一次折扣会改变算法。

手算 $v^{old}=[0.2,0.4]$、terminal reward=1、$\gamma=0.9,\lambda=0.8$：$\delta_1=0.6,\delta_0=0.16$，$A=[0.592,0.6]$，$T=[0.792,1]$。插入 masked value=999 不应改变有效位置结果。

当前 critic clip 半径 $c=0.2$：

$$
\bar v=\operatorname{clip}(v^\phi,v^{old}-c,v^{old}+c),\qquad
L_V=\frac1{2N}\sum_{ij}m_{ij}
\max\big((v^\phi_{ij}-T_{ij})^2,(\bar v_{ij}-T_{ij})^2\big).
$$

value loss 始终按有效 token 归约，与 actor 的归约选项独立。手算 $v^\phi=[1,0.5]$、$v^{old}=[0,0]$、$T=[0,1]$：逐 token 含 $1/2$ 的值为 $[0.5,0.32]$，平均后对 value 的梯度为 $[0.5,0]$。第二格走被 clip 的较大误差，clip 外导数为零。

固定 R-PPO 先 value_model.train 再 policy.train。参考对不相连的图分别 backward/SGD，更新顺序可交换；这不验证生产调度、分布式 optimizer 或 critic warmup。

## 正负优势 clipping 与两种归约

$$
r=\exp(\ell^\theta-\ell^p),\qquad
\bar r=\operatorname{clip}(r,1-\epsilon,1+\epsilon),\qquad
q=\max(-Ar,-A\bar r).
$$

这是最小化损失的 max。$A>0,r>1+\epsilon$ 限制继续提高好动作概率；$A<0,r<1-\epsilon$ 限制继续降低坏动作概率。

手算 $r=[1.5,0.5,0.5,1.5],A=[2,2,-2,-2]$ 得 $q=[-2.4,-1,1.6,3]$。四格平均对 current logprob 的梯度为 $[0,-0.25,0,0.75]$。误写 min 会选错分支。

$$
L_{\mathrm{token}}=\frac{\sum_{ij}m_{ij}q_{ij}}{N},\qquad
L_{\mathrm{sequence}}=\frac1B\sum_i\frac{\sum_jm_{ij}q_{ij}}{N_i}.
$$

两条序列的 loss 为 $[1]$ 与 $[3,3,3]$ 时，token mean=2.5，sequence mean=2。本页改变的是 loss reduction；没有开启 sequence-level importance ratio，ratio 仍逐 token。

## KL：前向为一的权重仍有导数

固定 k3、非 IS sampling-weight 分支，设 $d=\ell^r-\ell^\theta$：

$$
k=\exp(\ell^\theta-\operatorname{sg}(\ell^\theta))
\big(\exp(d)-1-d\big),\qquad L=L_{\mathrm{PG}}+\beta\,\operatorname{reduce}(k).
$$

detach anchor 固定且未 clamp 时，两部分都求导：

$$
\frac{\partial k}{\partial\ell^\theta}
=(e^d-1-d)+(1-e^d)=-d=\ell^\theta-\ell^r.
$$

删除或 detach 外部权重，错误导数为 $1-e^d$；loss 数值相同不证明梯度相同。R-UTIL 将输入 $d$ clamp 到 $\pm20$，同时 detach 对应采样权重；再将乘积 clamp 到 $\pm10$。参考覆盖这些分支。这里是采样式 penalty，不是枚举全词表的精确 KL。关闭 KL 为 $\beta=0$，页面仍可显示 reference logprob 解释来源。

## Force-on-policy、backward 与一次 SGD

$$
r=\exp(\ell^\theta-\operatorname{sg}(\ell^\theta))=1,\qquad
\frac{\partial r}{\partial\ell^\theta}=1.
$$

不能替成常数 ones_like。本页 force 从 generation=previous=current version 0 开始，每 batch 只允许一次更新；强制 ratio 不会使陈旧轨迹变成真实 on-policy。有限差分必须将 detach anchor 固定在基点，否则每次扰动都重建 anchor 会错误得出零导数。

用 log-softmax 导数独立复核整个 head：

$$
\frac{\partial L}{\partial W_{vh}}
=\sum_{ij}\frac{\partial L}{\partial\ell^\theta_{ij}}
\big(\mathbf1[a_{ij}=v]-p_{ijv}\big)x_{ijh},
\qquad W'=W-\eta_\theta\nabla_WL.
$$

critic 独立更新 $u'=u-\eta_\phi\nabla_uL_V$。CPU 检查 actor backward 后 critic.grad 为空，critic backward 不改变 actor gradient；GRPO critic 不变。页面 head 行梯度来自全批累计，不能误称只来自选定 token。

## Export、refit 与生成版本

“应用一次参考 SGD”将 policy 快照切到更新后参数；generation 仍为 version 0。“导出 policy 快照”固定 version 和参数 SHA256。“执行参考 refit”先复制到候选 generation，校验 export version/hash、generation version、完成 ack/hash，全通过才发布新快照。

hash 是头标识加 row-major float64 little-endian 字节，本项目不声称 NeMo RL 使用此 wire format。错误版本、hash 或未完成 ack 必须拒绝同步。更改算法/配置重置一次更新流程。

refit 后在同组固定 action 上重算 logprob，与更新后 policy 应相符。这证明参考参数复制和合约，不证明新采样、真实 engine refit 或第二次 RL 迭代；后者归 R02 验证。

## 验证与源码边界

test:rl-reference 对 GRPO/PPO、token/sequence、KL 与 force 配置逐字段比较 Python autograd 和 TS 解析梯度，含手算、同分组、masked gap、错误 clip、constant ratio、错误 KL 导数、真实 SGD 和版本/hash/ack 反例。浏览器另验控件、实际数值、导出、源码、离线数学、键盘和窄屏。

来源：R-LOSS 的对齐/clip/KL/value loss，R-UTIL 的组标准差/KL clamp，R-ADV 的 GRPO/GAE，R-PPO 的独立训练，R-GRPO 的 refit 成功检查，均固定到 source.lock commit。高层代码不冒充 kernel、runtime lineage 或生产依赖已兼容。
