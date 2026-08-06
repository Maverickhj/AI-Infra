---
type: knowledge
status: growing
created: 2026-07-29
updated: 2026-08-05
domains:
  - transformer
  - positional-encoding
aliases:
  - RoPE 原理
reviewed: false
---

# Transformer RoPE 原理解析

> 从二维旋转到相对位置注意力的直观推导、实现方式与长上下文扩展

## 阅读导览

RoPE（Rotary Position Embedding，旋转位置编码）的核心思想可以概括为：

> 把 token 的位置编码成旋转角度，分别旋转 Query 和 Key，使它们的点积自然依赖两个 token 的相对位置。

本文从二维旋转出发，依次说明 RoPE 如何表达相对位置、如何扩展到高维、如何进入注意力计算，以及代码实现、重要性质和长上下文外推问题。

## 1. RoPE 想解决什么问题

Transformer 的注意力计算为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

对于第 $m$ 个 token 和第 $n$ 个 token，它们的注意力分数是：

$$
s_{m,n}=\frac{q_m^\top k_n}{\sqrt d}
$$

其中：

- $q_m$：第 $m$ 个 token 的 Query；
- $k_n$：第 $n$ 个 token 的 Key；
- $d$：每个注意力头的维度。

单看 $q_m^\top k_n$，模型并不知道 token 位于序列中的哪个位置。例如：

```text
猫 吃 鱼
鱼 吃 猫
```

两句话包含相同的 token，却有完全不同的顺序和含义。因此，注意力机制需要某种位置编码。

RoPE 的目标是：在 Query 和 Key 中编码位置，并让注意力分数自然感知两个 token 之间的相对距离。

## 2. 从二维旋转开始

先看一个二维向量：

$$
x=\begin{bmatrix}x_1\\x_2\end{bmatrix}
$$

把它逆时针旋转 $\phi$，可以乘以旋转矩阵：

$$
R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}
$$

旋转后的向量为：

$$
x'=R(\phi)x
$$

展开后：

$$
x'_1=x_1\cos\phi-x_2\sin\phi
$$

$$
x'_2=x_1\sin\phi+x_2\cos\phi
$$

旋转不会改变向量长度：

$$
\lVert R(\phi)x\rVert=\lVert x\rVert
$$

它只改变方向。RoPE 的想法很直接：把 token 的位置编号转换成旋转角度。

假设基础角频率为 $\theta$，第 $m$ 个位置使用的旋转角度为：

$$
\phi_m=m\theta
$$

因此，第 $m$ 个 token 的向量被旋转为：

$$
x'_m=R(m\theta)x_m
$$

## 3. 为什么旋转能够表示相对位置

这是 RoPE 最关键的数学性质。假设二维 Query 和 Key 都是：

$$
q=k=\begin{bmatrix}1\\0\end{bmatrix}
$$

Query 位于位置 $m$，Key 位于位置 $n$。旋转后：

$$
q_m=R(m\theta)q=\begin{bmatrix}\cos(m\theta)\\\sin(m\theta)\end{bmatrix}
$$

$$
k_n=R(n\theta)k=\begin{bmatrix}\cos(n\theta)\\\sin(n\theta)\end{bmatrix}
$$

计算点积：

$$
q_m^\top k_n=\cos(m\theta)\cos(n\theta)+\sin(m\theta)\sin(n\theta)
$$

利用恒等式 $\cos A\cos B+\sin A\sin B=\cos(A-B)$，得到：

$$
q_m^\top k_n=\cos((m-n)\theta)
$$

最终结果不再分别依赖 $m$ 和 $n$，而是依赖位置差 $m-n$。例如，位置对 $(2,5)$ 和 $(102,105)$ 的位置差都为 $3$，因此位置旋转对注意力分数的影响具有相同形式。

> RoPE 使用绝对位置执行旋转，却通过点积让相对位置影响注意力。

## 4. 更一般的矩阵推导

对于任意 Query 向量 $q_m$ 和 Key 向量 $k_n$，RoPE 先进行旋转：

$$
\widetilde q_m=R(m\theta)q_m,\qquad \widetilde k_n=R(n\theta)k_n
$$

它们的点积是：

$$
\widetilde q_m^\top\widetilde k_n=q_m^\top R(m\theta)^\top R(n\theta)k_n
$$

旋转矩阵满足：

$$
R(\phi)^\top=R(-\phi),\qquad R(\alpha)R(\beta)=R(\alpha+\beta)
$$

因此：

$$
R(m\theta)^\top R(n\theta)=R((n-m)\theta)
$$

最终得到：

$$
\boxed{\widetilde q_m^\top\widetilde k_n=q_m^\top R((n-m)\theta)k_n}
$$

这个公式说明，Query 和 Key 分别使用绝对位置 $m,n$ 旋转，但点积中的位置项只依赖相对位置 $n-m$。

## 5. 如何扩展到高维向量

真实 Transformer 中，一个注意力头的维度可能是 $64$、$128$ 或 $256$。RoPE 把维度两两分组：

$$
(x_0,x_1),(x_2,x_3),\ldots,(x_{d-2},x_{d-1})
$$

每两个维度构成一个二维平面。第 $i$ 对维度使用独立频率：

$$
\theta_i=10000^{-2i/d},\qquad i=0,1,\ldots,\frac d2-1
$$

第 $m$ 个位置对应的旋转角度为 $m\theta_i$，于是：

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}=
\begin{bmatrix}\cos(m\theta_i)&-\sin(m\theta_i)\\\sin(m\theta_i)&\cos(m\theta_i)\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
$$

展开为：

$$
x'_{2i}=x_{2i}\cos(m\theta_i)-x_{2i+1}\sin(m\theta_i)
$$

$$
x'_{2i+1}=x_{2i}\sin(m\theta_i)+x_{2i+1}\cos(m\theta_i)
$$

## 6. 为什么需要不同旋转频率

如果所有二维分组都使用相同频率，位置信息会很单一。RoPE 使用一组逐渐减小的频率：

$$
\theta_0,\theta_1,\theta_2,\ldots
$$

由于 $\theta_i=10000^{-2i/d}$，随着 $i$ 增大，$\theta_i$ 逐渐减小：

- 前面的维度旋转得快，适合区分邻近位置；
- 后面的维度旋转得慢，适合表达更大范围。

可以把它类比为钟表：秒针变化快，分针较慢，时针更慢。多种频率共同作用，让模型可以在多个尺度上区分位置关系，如距离 $1$ 与 $2$、距离 $10$ 与 $20$、距离 $1000$ 与 $2000$。

## 7. RoPE 在 Attention 中的完整形式

普通 Attention 的 Query 和 Key 为：

$$
q_m=W_Qx_m,\qquad k_n=W_Kx_n
$$

使用 RoPE 后：

$$
\widetilde q_m=R_mW_Qx_m,\qquad \widetilde k_n=R_nW_Kx_n
$$

注意力分数变为：

$$
s_{m,n}=\frac{\widetilde q_m^\top\widetilde k_n}{\sqrt d}=\frac{q_m^\top R_{n-m}k_n}{\sqrt d}
$$

完整的注意力输出为：

$$
\operatorname{Attention}_m=\sum_n\operatorname{softmax}_n\left(\frac{q_m^\top R_{n-m}k_n}{\sqrt d}\right)v_n
$$

这里：

- $q_m,k_n$ 表示内容信息；
- $R_{n-m}$ 表示相对位置信息；
- $v_n$ 通常不应用 RoPE。

RoPE 一般只作用于 Query 和 Key，因为位置信息需要影响“应该关注谁”，即注意力权重；Value 则是被加权汇总的内容。

## 8. 从复数角度理解

二维旋转也可以用复数表示。把二维向量 $(x_1,x_2)$ 写成：

$$
z=x_1+\mathrm{i}x_2
$$

由欧拉公式：

$$
e^{\mathrm{i}\phi}=\cos\phi+\mathrm{i}\sin\phi
$$

复数乘以 $e^{\mathrm{i}\phi}$，相当于旋转 $\phi$：

$$
z'=ze^{\mathrm{i}\phi}
$$

因此，第 $m$ 个位置的 RoPE 可以写成：

$$
z'_m=z_me^{\mathrm{i}m\theta}
$$

两个位置之间的相位关系是：

$$
e^{-\mathrm{i}m\theta}e^{\mathrm{i}n\theta}=e^{\mathrm{i}(n-m)\theta}
$$

这里再次出现相对位置 $n-m$。因此，Rotary Position Embedding 的名字非常直观：把位置编码成复数相位或二维旋转。

## 9. 与其他位置编码方法对比

传统正弦位置编码通常把位置向量直接加到 token 表示上：

$$
h_m=x_m+p_m
$$

其中：

$$
p_m=[\sin(m\theta_0),\cos(m\theta_0),\ldots]
$$

RoPE 则使用旋转：

$$
h'_m=R_mh_m
$$

| 方法        | 操作                     | 主要效果                  |
| --------- | ---------------------- | --------------------- |
| 正弦位置编码    | $x_m+p_m$              | 把绝对位置加入表示，相对关系由模型间接学习 |
| RoPE      | $R_mx_m$               | 通过旋转让注意力点积自然感知相对位置    |
| 可学习绝对位置编码 | $x_m+e_m$              | 为每个位置学习一个向量           |
| 相对位置 Bias | $q_m^\top k_n+b_{m-n}$ | 直接给注意力分数增加相对位置偏置      |

RoPE 不把位置向量加到内容上，而是保持向量长度不变，通过改变 Query 和 Key 的方向，让相对位置直接进入注意力匹配。

## 10. 一个具体数值例子

假设基础频率为：

$$
\theta=\frac{\pi}{6}
$$

Query 位于 $m=2$，Key 位于 $n=5$。它们的位置角度分别为：

$$
m\theta=\frac{\pi}{3},\qquad n\theta=\frac{5\pi}{6}
$$

相对角度为：

$$
(n-m)\theta=3\cdot\frac{\pi}{6}=\frac{\pi}{2}
$$

如果原始 Query 和 Key 都是 $[1,0]^\top$，旋转后点积为：

$$
\cos\left(\frac{\pi}{2}\right)=0
$$

把它们同时向后移动 $100$ 个位置，令 $m'=102,n'=105$，相对距离仍为 $3$，因此点积中的位置影响仍是：

$$
\cos\left(3\cdot\frac{\pi}{6}\right)=0
$$

这说明 RoPE 对整体位置平移具有良好的相对位置性质。

## 11. 代码中通常如何实现

设输入 `x` 的最后一维为：

```text
[x0, x1, x2, x3, ...]
```

把偶数维和奇数维分别取出并旋转：

```python
x_even = x[..., 0::2]
x_odd = x[..., 1::2]

out_even = x_even * cos - x_odd * sin
out_odd = x_even * sin + x_odd * cos
```

最后交错组合：

```text
[out_even0, out_odd0, out_even1, out_odd1, ...]
```

另一种常见实现定义 `rotate_half`：

$$
\operatorname{rotate\_half}(x)=[-x_1,x_0,-x_3,x_2,\ldots]
$$

因此：

$$
\operatorname{RoPE}(x)=x\cos\phi+\operatorname{rotate\_half}(x)\sin\phi
$$

对应伪代码：

```python
def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return interleave(-x_odd, x_even)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```

不同框架可能采用不同的维度配对方式：相邻维度配对，或前半维度与后半维度配对。只要 `cos`、`sin` 与配对规则一致，数学上通常只是维度排列不同。

## 12. RoPE 的重要性质

### 12.1 不改变向量长度

因为旋转矩阵是正交矩阵：

$$
R_m^\top R_m=I
$$

所以：

$$
\lVert R_mx\rVert=\lVert x\rVert
$$

RoPE 不会直接放大或缩小 Query、Key 的范数，只改变方向。

### 12.2 位置信息直接进入注意力分数

RoPE 让普通点积 $q_m^\top k_n$ 显式变成：

$$
q_m^\top R_{n-m}k_n
$$

相对位置由此直接参与注意力匹配。

### 12.3 没有额外可训练参数

经典 RoPE 的频率通常固定为 $\theta_i=10000^{-2i/d}$，因此不需要维护一个长度为最大上下文长度的位置 Embedding 表。它可以计算任意位置的旋转角度，但“能够计算”不等于模型能正确使用无限长的位置。

### 12.4 不保证距离越远，注意力越小

单个频率中的位置项类似：

$$
\cos((m-n)\theta)
$$

余弦函数会振荡，并非单调下降。RoPE 的作用是让不同距离产生不同相位，让模型能够识别相对距离；它不是简单的距离衰减函数。

## 13. 长上下文外推为什么会遇到问题

位置 $m$ 对应的角度是 $m\theta_i$。当上下文长度显著超过训练范围时，主要会出现以下问题。

### 13.1 高频维度旋转过快

对于较大的 $\theta_i$，位置稍微增加，角度就会变化很多。若模型训练时只见过长度 $4096$，推理突然扩展到 $128\text{K}$，就会遇到训练中未见过的相位组合。

### 13.2 旋转具有周期性

由于：

$$
\sin(\phi+2\pi)=\sin\phi,\qquad \cos(\phi+2\pi)=\cos\phi
$$

位置编码具有周期性。多频率组合可以显著扩大可区分范围，但并不意味着无限无歧义。

### 13.3 训练与推理长度分布不一致

即使公式可以计算位置 $1000000$，模型参数也可能只在较短位置范围内训练过。因此，长上下文外推通常需要调整 RoPE 频率，例如：

- Position Interpolation；
- Linear Scaling；
- NTK-aware Scaling；
- YaRN；
- LongRoPE。

这些方法的共同思路通常是减慢旋转速度，或针对不同频率采用不同缩放方式，让训练长度范围映射到更长的推理范围。

最简单的线性缩放把位置 $m$ 替换为 $m/s$，旋转角度从 $m\theta_i$ 变为：

$$
\frac{m}{s}\theta_i
$$

这相当于让位置变化得更慢。

## 14. 最终直觉与总结

可以把每个 Query 和 Key 想象成许多根二维钟表指针：

- 第一根指针转得很快；
- 第二根稍慢；
- 第三根更慢；
- 每根指针都转到与位置 $m$ 对应的角度。

当第 $m$ 个 Query 与第 $n$ 个 Key 做点积时，相同频率的指针比较夹角，而夹角由 $n-m$ 决定。

RoPE 的核心链路是：

$$
\boxed{\text{位置}\rightarrow\text{旋转角度}\rightarrow\text{Query/Key 方向变化}\rightarrow\text{点积感知相对距离}}
$$

最值得记住的三个公式是：

$$
\boxed{R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}}
$$

$$
\boxed{\widetilde q_m=R_mq_m,\qquad \widetilde k_n=R_nk_n}
$$

$$
\boxed{\widetilde q_m^\top\widetilde k_n=q_m^\top R_{n-m}k_n}
$$

一句话概括：

> RoPE 把每个位置编码成旋转角度，分别旋转 Query 和 Key，使它们的点积自然依赖两个 token 的相对位置。

## 相关笔记

- [[RoPE 长上下文外推方法综述|RoPE 长上下文外推方法综述]]
