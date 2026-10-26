---
type: knowledge
status: draft
created: 2026-10-25
updated: 2026-10-26
domains: [llm, attention, performance]
source:
  - https://spaces.ac.cn/archives/10091
  - https://spaces.ac.cn/archives/8265
ai_generated: true
reviewed: false
---

# MLA 计算强度分析基线

> [!warning]
> 本文包含 AI 生成的推导，尚未完成人工核验。计算量、显存访存量和计算强度均为明确假设下的理论估算，不是内核实测结果。

## 1. 分析目标与阅读顺序

本文要解释一个具体问题：为什么多头潜在注意力（Multi-head Latent Attention，MLA）在解码时，可以通过增加计算量来减少缓存读取，并因此获得性能收益？分析按预填充（prefill）与解码（decode）两个阶段，以及展开（expanded）与吸收（absorbed）两种计算形式组织，分别统计计算量、显存访存量和计算强度（arithmetic intensity）。

| 阶段 | 展开形式 | 吸收形式 |
|---|---|---|
| 预填充 | 已建立单位置算子成本；已给出批量投影与键值复用的基本模型；完整批量访存待分析 | 待分析 |
| 解码 | 已建立单个解码词元的注意力核心与前向成本；已明确批次扩展的边界 | 待分析 |

各场景采用相同的推导顺序：从一个词元（token）位置和一个注意力头（head）开始，统计投影与归一化成本；再分析一对查询（query，Q）与键（key，K）的匹配，以及对应值（value，V）的加权；随后扩展到一条序列，最后考虑批次（batch）和多头。单头、单位置、单查询只是分步推导的简化条件，不是文档适用范围的限制。

预填充中的单位置分析，统计的是输入序列中某个位置的算子成本，并不表示输入长度为 1。在因果注意力（causal attention）中，该位置仍可访问当前位置及此前的序列前缀。

标准自回归解码每步为每条活跃序列处理一个新输入词元，因此每条序列的查询长度为 Q_len=1。该词元在每个头中各有一个查询向量；批大小为 $B$、头数为 $h$ 时，一步解码共处理 $B$ 个查询位置、$Bh$ 个查询向量。因此，“单查询”只在固定序列和头之后才指一个向量。推测解码（speculative decoding）中的多词元验证不属于这里的简化模型。

一个键值位置只是注意力归一化范围中的一项，不能独立完成归一化。若确实只有一个可见键值位置，其权重恒为 1；本文按位置统计的成本不采用这种退化情形。

## 2. 共用符号与统计口径

### 2.1 符号与维度

使用行向量记法（row-vector convention）。局部推导固定注意力头 $s$，必要时省略头下标。

| 符号 | 含义 |
|---|---|
| $B$ | 批大小，即活跃序列的数量 |
| $S$ | 无前缀缓存时，等长预填充的输入长度 |
| $Q_{\mathrm{len}}$ | 每条序列本次处理的查询位置数 |
| $h$ | 注意力头数 |
| $t,i$ | 查询位置、可见键值位置 |
| $L,L_r$ | 单条序列的可见键值长度、批次中第 $r$ 条序列的可见键值长度 |
| $d$ | 模型隐藏维度 |
| $d_c,d_c'$ | 键值压缩维度、查询压缩维度 |
| $d_k,d_r,d_v$ | 内容 Q/K 维度、位置分支维度、V 维度 |
| $p=d_k+d_r$ | 展开后的 Q/K 头维度 |
| $b$ | 每个元素的字节数；16 位浮点格式（bfloat16，BF16）占 2 字节 |
| $N$ | 一次投影处理的词元位置数 |
| $M$ | 同一份 K/V 在片上复用的查询数 |
| $\boldsymbol x_t$ | 词元隐藏状态，$1\times d$ |

### 2.2 展开形式的前向计算

先通过降维投影（down-projection）得到压缩表示，再进行均方根归一化（Root Mean Square Normalization，RMSNorm）：

$$
\begin{aligned}
\boldsymbol z_t^Q&=\boldsymbol x_tW_c',&
\boldsymbol c_t'&=\mathrm{RMSNorm}(\boldsymbol z_t^Q),\\
\boldsymbol z_i^{KV}&=\boldsymbol x_iW_c,&
\boldsymbol c_i&=\mathrm{RMSNorm}(\boldsymbol z_i^{KV}).
\end{aligned}
$$

再通过展开投影（up-projection）生成各头的 Q、K、V，并计算注意力：

$$
\begin{aligned}
\boldsymbol q_t^{(s)}&=[\boldsymbol c_t'W_{qc}^{(s)},\ \boldsymbol c_t'W_{qr}^{(s)}\mathcal R_t]
\in\mathbb R^{1\times p},\\
\boldsymbol k_i^{(s)}&=[\boldsymbol c_iW_{kc}^{(s)},\ \boldsymbol x_iW_{kr}\mathcal R_i]
\in\mathbb R^{1\times p},\\
\boldsymbol v_i^{(s)}&=\boldsymbol c_iW_v^{(s)}\in\mathbb R^{1\times d_v},\\
s_{t,i}^{(s)}&=(\boldsymbol q_t^{(s)}\boldsymbol k_i^{(s)\top})\sigma,
\qquad \sigma=1/\sqrt p,\\
\alpha_{t,i}^{(s)}&=\frac{\exp(s_{t,i}^{(s)})}{\sum_{j\le t}\exp(s_{t,j}^{(s)})},\\
\boldsymbol o_t^{(s)}&=\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol v_i^{(s)}.
\end{aligned}
$$

将各头的结果拼接后，进行输出投影（output projection）：

$$
\boldsymbol y_t
=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O
=\sum_{s=1}^{h}\boldsymbol o_t^{(s)}W_O^{(s)},
\qquad W_O^{(s)}\in\mathbb R^{d_v\times d}.
$$

单头的输出成本只表示该头对最终结果的贡献。完整多头矩阵乘法通常已包含跨头求和，不应再无条件叠加一次独立的求和成本。

### 2.3 计算量与显存访存量如何计数

- 一次乘法或加法计为一次浮点运算（floating-point operation，FLOP），运算总次数记为 FLOPs；融合乘加（fused multiply-add，FMA）计 2 FLOPs。矩阵乘法使用 $2mnk$ 的主项估算。
- 指数、比较、除法和平方根倒数分别列出操作次数，不将它们与普通加乘视为等成本指令。平方根倒数在公式中记为 rsqrt。
- 显存访存量指高带宽内存（High Bandwidth Memory，HBM）与片上存储之间的数据搬运量，不等于张量占用容量。同一份数据重复读取时，需要重复计数。
- 理论模型统一采用每元素 $b$ 字节，包含 RMSNorm 权重；实际数据类型（dtype）不同时，应按张量分别核算。片上累加可以使用 32 位浮点格式（FP32）。
- 降维投影、压缩表示的 RMSNorm 和位置 K 投影由各头共享。它们按新词元的位置数计数，不按历史查询与键的配对数计数。
- 分数缩放（score scaling）按每个分数一次乘法计数，$\sigma$ 已预先计算。若计算内核（kernel）将缩放移到 Q 或其他操作中，应按实际执行方式调整。
- 偏置（bias）、注意力输入端的归一化、残差连接（residual connection）、随机失活（dropout）、掩码（mask）处理、缓存写入和通信不纳入当前合计，见第 7 节。

### 2.4 归一化与位置编码的公式和成本

#### 2.4.1 RMSNorm：对压缩表示归一化

对 $n$ 维向量：

$$
r=\operatorname{rsqrt}\left(\frac1n\sum_{j=1}^{n}z_j^2+\epsilon\right),
\qquad \widehat c_j=z_jr\gamma_j.
$$

直接实现包含 $n$ 次平方、$n-1$ 次求和、1 次均值缩放、1 次加稳定项，以及各 $n$ 次归一化乘法和权重缩放乘法。因此采用：

$$
R(n)=4n+1\ \mathrm{FLOPs},
$$

另列一次 rsqrt，并读取 $n$ 个权重 $\gamma$。压缩表示的 RMSNorm 已纳入合计，rsqrt 不折算为普通 FLOPs。

#### 2.4.2 旋转位置编码（RoPE）：二维旋转如何加入位置信息

旋转位置编码（Rotary Position Embedding，RoPE）将位置映射为旋转角度。以下将相邻的特征坐标两两分组，并采用行向量记法。“二维”指一对特征坐标，不指图像等二维位置坐标。构造参考苏剑林的[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)。

设位置为 $m$，一对坐标为 $\boldsymbol u=[u_0,u_1]$，对应角频率为 $\theta$。行向量右乘的旋转矩阵定义为：

$$
R_{\mathrm{2D}}(\phi)=
\begin{bmatrix}
\cos\phi&\sin\phi\\
-\sin\phi&\cos\phi
\end{bmatrix},
\qquad \phi=m\theta.
$$

因此：

$$
\begin{aligned}
\widetilde{\boldsymbol u}_m
&=\boldsymbol uR_{\mathrm{2D}}(m\theta)\\
&=[u_0\cos(m\theta)-u_1\sin(m\theta),\quad
u_0\sin(m\theta)+u_1\cos(m\theta)].
\end{aligned}
$$

与列向量记法（column-vector convention）的对应关系为：

$$
\widetilde{\boldsymbol u}_m^\top
=R_{\mathrm{2D}}(m\theta)^\top\boldsymbol u^\top
=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}
\boldsymbol u^\top.
$$

原文按列向量左乘旋转矩阵，本文按行向量右乘其转置；两种写法得到相同的坐标。每对坐标执行 4 次乘法、2 次加减法，共 6 FLOPs。

#### 2.4.3 RoPE：多维旋转由二维分块组成

设位置分支的维度 $d_r$ 为偶数，将 $\boldsymbol u\in\mathbb R^{1\times d_r}$ 划分为 $d_r/2$ 对坐标，第 $j$ 对使用角频率 $\theta_j$：

$$
\theta_j=\mathrm{base}^{-2j/d_r},
\qquad j=0,\ldots,d_r/2-1.
$$

经典 RoPE 使用 $\mathrm{base}=10000$；具体模型的频率基数和缩放方式按配置确定。本文的旋转只作用于 $d_r$ 维位置分支，不作用于整个模型维度 $d$ 或展开后的头维度 $p=d_k+d_r$。

将各对坐标的旋转矩阵放到对角线上，得到分块对角矩阵（block-diagonal matrix）：

$$
\mathcal R_m
=\operatorname{blockdiag}
\left(R_{\mathrm{2D}}(m\theta_0),R_{\mathrm{2D}}(m\theta_1),\ldots,R_{\mathrm{2D}}(m\theta_{d_r/2-1})\right)
\in\mathbb R^{d_r\times d_r}.
$$

以 $d_r=4$ 为例：

$$
\mathcal R_m=
\begin{bmatrix}
\cos(m\theta_0)&\sin(m\theta_0)&0&0\\
-\sin(m\theta_0)&\cos(m\theta_0)&0&0\\
0&0&\cos(m\theta_1)&\sin(m\theta_1)\\
0&0&-\sin(m\theta_1)&\cos(m\theta_1)
\end{bmatrix}.
$$

一般情形下，每对坐标分别计算：

$$
\begin{aligned}
\widetilde u_{m,2j}
&=u_{2j}\cos(m\theta_j)-u_{2j+1}\sin(m\theta_j),\\
\widetilde u_{m,2j+1}
&=u_{2j}\sin(m\theta_j)+u_{2j+1}\cos(m\theta_j).
\end{aligned}
$$

实现时只需逐元素乘法和每对坐标的加减法，无需执行稠密的 $d_r\times d_r$ 矩阵乘法。假设正余弦系数已准备好，则：

$$
F_{\mathrm{RoPE}}=6\frac{d_r}{2}=3d_r\ \mathrm{FLOPs/vector}.
$$

坐标配对的物理顺序可以通过置换调整。交错布局（interleaved）将一对坐标相邻存放；分半布局（split-half）分别从向量前后两半取出配对坐标。具体内核的布局必须与权重和特征顺序一致。

#### 2.4.4 RoPE：内积为什么只引入相对位置

每个二维分块都满足旋转的复合规则和正交性，因此：

$$
\mathcal R_m\mathcal R_n=\mathcal R_{m+n},
\qquad
\mathcal R_m^\top=\mathcal R_{-m},
\qquad
\mathcal R_t\mathcal R_i^\top=\mathcal R_{t-i}.
$$

设 $\boldsymbol u_t,\boldsymbol w_i\in\mathbb R^{1\times d_r}$ 为尚未施加 RoPE 的 Q、K 位置分支向量。旋转后的内积为：

$$
\begin{aligned}
(\boldsymbol u_t\mathcal R_t)
(\boldsymbol w_i\mathcal R_i)^\top
&=\boldsymbol u_t\mathcal R_t\mathcal R_i^\top\boldsymbol w_i^\top\\
&=\boldsymbol u_t\mathcal R_{t-i}\boldsymbol w_i^\top.
\end{aligned}
$$

旋转引入的位置依赖通过相对偏移 $t-i$ 进入内积。MLA 使用解耦 RoPE（decoupled RoPE），只旋转位置分支；内容分支不旋转，从而保留吸收变换所需的线性结构。

#### 2.4.5 归一化指数函数（softmax）：从分数得到权重

固定查询位置 $t$ 和头 $s$，将 $L$ 个可见位置的未归一化分数（logits）记为 $s_i=s_{t,i}^{(s)}$，其中已包含分数缩放。Softmax 将它们转化为非负、和为 1 的权重：

$$
\alpha_i=\operatorname{softmax}(\boldsymbol s)_i
=\frac{\exp(s_i)}{\sum_{j=1}^{L}\exp(s_j)},
\qquad
\alpha_i\ge0,\quad \sum_{i=1}^{L}\alpha_i=1.
$$

数值稳定的 softmax（stable softmax）利用平移不变性（shift invariance）：先从每一项中减去最大分数，再计算指数和归一化：

$$
\begin{aligned}
m&=\max_{1\le j\le L}s_j,\\
e_i&=\exp(s_i-m),\\
\ell&=\sum_{j=1}^{L}e_j,\\
\alpha_i&=e_i/\ell.
\end{aligned}
$$

这一变换保持 softmax 的数学结果：

$$
\frac{\exp(s_i-m)}{\sum_j\exp(s_j-m)}
=\frac{\exp(-m)\exp(s_i)}
{\exp(-m)\sum_j\exp(s_j)}
=\frac{\exp(s_i)}{\sum_j\exp(s_j)}.
$$

对于有限分数，$s_i-m\le0$，可以避免直接对较大正数计算指数。V 不参与 softmax；得到的权重随后用于对 V 加权求和：

$$
\boldsymbol o_t^{(s)}=\sum_{i=1}^{L}\alpha_i\boldsymbol v_i^{(s)}.
$$

| 操作 | 每个查询、每个头的次数 |
|---|---:|
| 求最大值 | $L-1$ 次比较 |
| 减去最大值 | $L$ 次减法 |
| 指数 | $L$ 次指数运算 |
| 求和 | $L-1$ 次加法 |
| 归一化 | $L$ 次除法，或 1 次倒数与 $L$ 次乘法 |

约 $5L$ 次标量操作只是粗略的次数统计，不表示这些指令具有相同硬件成本。当前 FLOPs 合计不包含 softmax 操作。在线 softmax（online softmax）按分块维护归一化状态，其累加器更新和额外开销见第 7.1 节；不能重复累计两种归一化实现的成本。

### 2.5 如何用计算强度判断性能瓶颈

$$
I=\frac{F}{D}\quad\mathrm{FLOPs/Byte},
\qquad
T\gtrsim\max\left(\frac FP,\frac D{B_{\mathrm{HBM}}}\right).
$$

屋顶线模型（Roofline model）用计算能力与访存带宽约束运行时间。$F,D$ 必须对应相同统计范围；$P$ 为相应精度与算子的计算吞吐量，$B_{\mathrm{HBM}}$ 为显存带宽。两类瓶颈的分界点为 $P/B_{\mathrm{HBM}}$。上述关系用于性能估算，不等于实测延迟。

## 3. 预填充 × 展开

### 3.1 单个词元位置：哪些算子执行一次

预填充为每个输入位置生成 Q、K、V。下表以一个位置和一个头展示算子成本，同时标明哪些操作由多头共享。注意力中的查询与键配对次数另行累计。

| 算子 | 公式 | 运算量 | 执行频率 |
|---|---|---:|---|
| Q 降维投影 | $\boldsymbol z_t^Q=\boldsymbol x_tW_c'$ | $2dd_c'$ FLOPs | 每个新词元一次，多头共享 |
| Q 压缩表示的 RMSNorm | $\boldsymbol c_t'=\mathrm{RMSNorm}(\boldsymbol z_t^Q)$ | $4d_c'+1$ FLOPs，另有 1 次 rsqrt | 每个新词元一次，多头共享 |
| KV 降维投影 | $\boldsymbol z_i^{KV}=\boldsymbol x_iW_c$ | $2dd_c$ FLOPs | 每个新词元一次，多头共享 |
| KV 压缩表示的 RMSNorm | $\boldsymbol c_i=\mathrm{RMSNorm}(\boldsymbol z_i^{KV})$ | $4d_c+1$ FLOPs，另有 1 次 rsqrt | 每个新词元一次，多头共享 |
| Q 展开投影 | $[\boldsymbol c_t'W_{qc},\boldsymbol c_t'W_{qr}]$ | $2d_c'(d_k+d_r)$ FLOPs | 每个当前查询、每个头一次 |
| Q 旋转 | $(\boldsymbol c_t'W_{qr})\mathcal R_t$ | $3d_r$ FLOPs | 每个当前查询、每个头一次 |
| 内容 K 展开投影 | $\boldsymbol c_iW_{kc}$ | $2d_cd_k$ FLOPs | 每个新 K、每个头一次 |
| 位置 K 投影 | $\boldsymbol x_iW_{kr}$ | $2dd_r$ FLOPs | 每个新 K 一次，所有头共享 |
| K 旋转 | $(\boldsymbol x_iW_{kr})\mathcal R_i$ | $3d_r$ FLOPs | 每个新 K 一次，所有头共享 |
| V 展开投影 | $\boldsymbol c_iW_v$ | $2d_cd_v$ FLOPs | 每个新 V、每个头一次 |
| QK 内积 | $\boldsymbol q_t\boldsymbol k_i^\top$ | 约 $2p$ FLOPs | 每对查询与键、每个头一次 |
| 分数缩放 | $s_{t,i}=(\boldsymbol q_t\boldsymbol k_i^\top)\sigma$ | 1 FLOP | 每对查询与键、每个头一次 |
| Softmax | $\alpha_{t,i}=\operatorname{softmax}_i(s_{t,i})$ | 对长度 $L$ 的整组分数为 $O(L)$ | 每个查询、每个头做一次整体归一化 |
| V 加权累加 | $\boldsymbol o_t\mathrel{+}=\alpha_{t,i}\boldsymbol v_i$ | 约 $2d_v$ FLOPs | 每对查询与值、每个头一次 |
| 输出投影 | $\boldsymbol o_tW_O^{(s)}$ | $2d_vd$ FLOPs | 历史位置求和完成后，每个查询、每个头一次 |

每个位置的 K、V 只生成一次；Q 投影和输出投影也按查询位置执行一次。只有 QK 内积、分数缩放和 V 加权累加按查询与键的配对数累计。

KV 降维投影与位置 K 投影可融合为输出宽度 $d_c+d_r$ 的矩阵乘法，合计 $2d(d_c+d_r)$ FLOPs。其中已经包含位置投影的 $2dd_r$，不能重复计数。

### 3.2 单个位置：明确访存假设后估算成本

本节统计单向量算子的理想访存成本，不代表实际预填充的批量访存量。假设：权重初始未驻留片上，每个元素从显存读入一次；输入读取一次；中间激活在片上复用；输出写回一次；正余弦与分数缩放系数已在片上。Softmax、注意力对历史 K/V 的读取及持久缓存写入不计入本节。

固定一个头，并将共享路径计一次，定义：

$$
\begin{aligned}
A&=d_c'p+d_c(d_k+d_v)+dd_r+d_vd,\\
C&=d(d_c'+d_c),\\
G&=d_c'+d_c.
\end{aligned}
$$

$A$ 为展开投影、位置 K 投影和输出投影的权重元素数；$C$ 为降维投影的权重元素数；$G$ 为 RMSNorm 的权重元素数。

| 算子组 | 计算量 | 显存访存量 |
|---|---:|---:|
| 降维投影 | $2C$ | 权重：$bC$ |
| 压缩表示的 RMSNorm | $4G+2$，另有 2 次 rsqrt | 权重：$bG$；压缩表示在片上复用 |
| 展开投影、位置 K 投影和输出投影 | $2A$ | 权重：$bA$ |
| Q/K 旋转 | $6d_r$ | 正余弦系数在片上复用 |
| 词元输入与输出 | — | 输入读取与输出写回：$2bd$ |

因此：

$$
F_{\mathrm{token,ops}}\approx2(A+C)+4G+2+6d_r,
\qquad
D_{\mathrm{token,cold}}\approx b(A+C+G+2d).
$$

$$
I_{\mathrm{token,cold}}
\approx\frac{2(A+C)+4G+2+6d_r}{b(A+C+G+2d)}.
$$

该结果仅统计每个词元的投影、归一化、RoPE 和输出投影，未包含与序列前缀交互的注意力计算，也未计入 Q/K/V 等中间张量写入显存的成本。公式下标 cold 只标识本节的假设，不是一个独立的标准模型名称；权重初始未驻留片上，也不自动保证运行时只读一次。实际预填充仍需根据算子融合、分块和批量复用推导访存量。

### 3.3 从一对查询与键扩展到整条序列

对查询位置 $t$，每个可见键值位置的普通加乘运算量为：

$$
F_{\mathrm{pair}}=2p+1+2d_v=2(p+d_v)+1.
$$

采用无前缀缓存、位置从 1 开始的因果注意力，该查询位置的注意力核心计算量为 $tF_{\mathrm{pair}}$。访存量还取决于其他查询位置能否复用同一份 K/V，因此不能将解码中的独立 K/V 读取量直接按位置累加，作为预填充的访存量。

序列长度为 $S$ 时，因果注意力中的查询与键配对数为 $S(S+1)/2$。单头简化下，算子与注意力核心计算量可扩展为：

$$
F_{\mathrm{prefill,expanded}}^{(h=1)}
\approx
S F_{\mathrm{token,ops}}
+\frac{S(S+1)}2 F_{\mathrm{pair}}.
$$

该式仍不含 softmax 等单列操作。$S=1$ 时注意力会退化，不能将按配对数统计的主项理解为专门针对单位置优化后的执行成本。

### 3.4 扩展到批次与多头：哪些数据能够复用

等长预填充一次处理的词元位置数为 $N=BS$。以单个投影 $Y=XW$ 为例：

$$
I_{\mathrm{projection}}
\approx
\frac{2Nd_{\mathrm{in}}d_{\mathrm{out}}}
{b(Nd_{\mathrm{in}}+d_{\mathrm{in}}d_{\mathrm{out}}+Nd_{\mathrm{out}})}.
$$

该式假设输入和权重各读一次，输出写回一次。多个词元复用同一份权重，使权重访存量不必随词元数等比例增加。

注意力内核中，如果同一键值数据块（tile）在片上被 $M$ 个查询复用，仅考虑这次 K/V 读取所支撑的运算：

$$
I_{\mathrm{KV,reuse}}
\approx
\frac{M[2(p+d_v)+1]}{b(p+d_v)}.
$$

$M$ 是内核实际实现的复用次数，不保证等于 $S$。该式尚未计入 Q 的读取、输出写回、K/V 重复加载、中间张量落盘和缓存写入。完整的批量访存与多头合计仍待分析，不能把单位置的理想访存估算当作预填充的计算强度。

## 4. 解码 × 展开

### 4.1 一个解码词元需要读取哪些缓存

标准解码每条活跃序列的查询长度为 Q_len=1。展开形式下，查询张量的一种布局为 $Q\in\mathbb R^{B\times h\times1\times p}$。固定序列和头之后，注意力的输入形状为：

$$
\boldsymbol q_t\in\mathbb R^{1\times p},
\qquad K_{\le t}\in\mathbb R^{L\times p},
\qquad V_{\le t}\in\mathbb R^{L\times d_v}.
$$

本节假设历史 K、V 已展开并保存在键值缓存（KV cache）中，每个位置读取一次；当前词元的 Q、K、V 仅生成一次。展开计算与缓存布局是两个不同维度：如果缓存中只存压缩表示、使用时才展开历史 K/V，还需计入历史展开投影及中间读写，本节的缓存访存模型不适用。

### 4.2 单个历史位置：计算量与读取量

先固定一条序列、一个头和一对查询与键。Q 已生成，历史 K/V 已展开并缓存。此处只统计注意力核心计算，不计投影权重读取和当前词元的缓存写入。

假设 Q 和输出累加器保留在片上，每个历史 K/V 从显存读取一次。Softmax 与注意力计算融合，不把完整分数和权重写回显存。在线 softmax 的累加器重缩放等实现开销在第 7 节单列。

| 算子 | 运算量 | 显存读取 | 显存写入 |
|---|---:|---:|---:|
| QK 内积 | 约 $2p$ FLOPs | K 读取：$bp$ Bytes | 分数在片上，0 |
| 分数缩放 | 1 FLOP | 分数与缩放系数在片上，0 | 0 |
| Softmax | 每个键值位置摊销约 5 次标量操作 | 分数在片上，0 | 权重在片上，0 |
| V 加权累加 | 约 $2d_v$ FLOPs | V 读取：$bd_v$ Bytes | 累加器在片上，0 |
| **合计，softmax 操作单列** | **$2(p+d_v)+1$ FLOPs** | **$b(p+d_v)$ Bytes** | **不按键值位置写回输出** |

计入 QK 内积、分数缩放和 V 加权累加后，该历史位置的计算强度为：

$$
I_{\mathrm{pair,KV}}
\approx\frac{2(p+d_v)+1}{b(p+d_v)}
=\frac2b+\frac1{b(p+d_v)}
\quad\mathrm{FLOPs/Byte}.
$$

### 4.3 一个解码词元：累计 $L$ 个可见位置

QK 内积、分数缩放和 V 加权累加的计算量为：

$$
F_{\mathrm{core}}\approx L[2(p+d_v)+1].
$$

仅统计历史键值缓存读取时：

$$
D_{\mathrm{KV}}=bL(p+d_v),
\qquad
I_{\mathrm{core,KV}}\approx\frac{2(p+d_v)+1}{b(p+d_v)}.
$$

再计入当前 Q 从显存读取一次、注意力输出写回一次，并仍采用每元素 $b$ 字节：

$$
D_{\mathrm{core}}\approx b\left[L(p+d_v)+p+d_v\right]
=b(L+1)(p+d_v).
$$

$$
\boxed{
I_{\mathrm{core}}\approx
\frac{L[2(p+d_v)+1]}{b(L+1)(p+d_v)}
\xrightarrow{L\gg1}\frac{2(p+d_v)+1}{b(p+d_v)}
}
$$

如果 Q 已在片上，或输出直接交给融合的后续算子，相应访存可以省去。K/V 重复读取、分数落盘、分块并行的中间结果等额外流量，需要单独计入。

### 4.4 数值例子

设 $d_k=128$、$d_r=64$、$d_v=128$，故 $p=192$；缓存使用 BF16，$b=2$。

| 指标 | 单头、单个历史位置 |
|---|---:|
| QK 内积 | $2\times192=384$ FLOPs |
| 分数缩放 | 1 FLOP |
| V 加权累加 | $2\times128=256$ FLOPs |
| 计算量合计 | **641 FLOPs** |
| K 读取 | $2\times192=384$ Bytes |
| V 读取 | $2\times128=256$ Bytes |
| 键值读取合计 | **640 Bytes** |
| 基于键值读取的计算强度 | **$641/640=1.0015625$ FLOPs/Byte** |

计数约定：只统计 QK 内积和 V 加权累加时为 640 FLOPs；加入一次分数缩放后为 641 FLOPs。片上缩放不增加显存读写。降维投影和 RMSNorm 是每个新词元的一次性成本，在第 4.5 节计入，不按历史位置重复累计。

取 $L=8192$，固定一条序列、一个头和一个解码词元：

$$
F_{\mathrm{core}}=641\times8192=5{,}251{,}072\ \mathrm{FLOPs},
$$

$$
D_{\mathrm{KV}}=640\times8192=5{,}242{,}880\ \mathrm{Bytes}=5\ \mathrm{MiB}.
$$

计入一次 Q 读取和输出写回后，访存量增加 640 Bytes，计算强度为 $5{,}251{,}072/5{,}243{,}520\approx1.00144025$ FLOPs/Byte。$1\ \mathrm{MiB}=2^{20}$ Bytes。分数缩放已计入，softmax 操作仍单列。

### 4.5 一个解码词元：加入投影和归一化后的合计

在第 4.3 节基础上，加入当前词元的降维投影、RMSNorm、展开投影、RoPE 和输出投影。$A,C,G$ 采用第 3.2 节定义。

假设所有权重从显存读入一次，当前隐藏状态读入一次后复用；中间激活、Q 和累加器保留在片上，最终输出写回一次。固定一个头，并计入一次共享路径成本：

| 算子 | 运算量 | 显存读取 | 显存写入 |
|---|---:|---:|---:|
| Q 降维投影 | $2dd_c'$ | 权重 $bdd_c'$；输入 $bd$ | 0，结果留片上 |
| Q 压缩表示的 RMSNorm | $4d_c'+1$，另有 1 次 rsqrt | $\gamma_Q$：$bd_c'$ | 0 |
| KV 降维投影 | $2dd_c$ | 权重 $bdd_c$；复用输入 | 0 |
| KV 压缩表示的 RMSNorm | $4d_c+1$，另有 1 次 rsqrt | $\gamma_{KV}$：$bd_c$ | 0 |
| Q 展开投影 | $2d_c'p$ | 权重 $bd_c'p$ | 0 |
| Q 旋转 | $3d_r$ | 0 | 0 |
| K 展开投影与位置投影 | $2d_cd_k+2dd_r$ | 权重 $b(d_cd_k+dd_r)$ | 0 |
| K 旋转 | $3d_r$ | 0 | 0 |
| V 展开投影 | $2d_cd_v$ | 权重 $bd_cd_v$ | 0 |
| QK 内积与分数缩放 | $L(2p+1)$ | 展开 K：$bLp$ | 0 |
| Softmax | 特殊操作和求和单列，见第 2.4 节 | 分数在片上，0 | 0 |
| V 加权累加 | $2Ld_v$ | 展开 V：$bLd_v$ | 0 |
| 输出投影 | $2d_vd$ | 权重 $bd_vd$ | 输出 $bd$ |

$$
\boxed{
F_{\mathrm{decode,expanded}}^{(B=1,h=1)}
\approx2(A+C)+4G+2+6d_r+L[2(p+d_v)+1]
}
$$

$$
\boxed{
D_{\mathrm{decode,expanded}}^{(B=1,h=1)}
\approx b[A+C+G+2d+L(p+d_v)]
}
$$

$$
\boxed{
I_{\mathrm{decode,expanded}}^{(B=1,h=1)}
\approx
\frac{2(A+C)+4G+2+6d_r+L[2(p+d_v)+1]}
{b[A+C+G+2d+L(p+d_v)]}
}
$$

合计已包含降维投影、压缩表示的 RMSNorm 和分数缩放；2 次 rsqrt、softmax 及在线分块实现的额外操作仍单列。

访存模型暂不计当前 K/V 的缓存写入。若 $L$ 包含当前位置，且其 K/V 由片上直接供注意力计算使用，可减去对应的缓存读取量。由原始隐藏状态生成的压缩表示不再作为外部输入重复读取。

$B=1,h=1$ 只是本层推导的简化条件。扩展到多头时，共享的降维投影、RMSNorm 和位置 K 路径只计一次，不能把全部合计直接乘以 $h$。

### 4.6 扩展到批次与多头

标准解码每条序列的查询长度为 Q_len=1，一个批次的查询位置数为 $N=B$。固定一个头时，各项成本按以下规则扩展：

| 组成部分 | 批次扩展规则 |
|---|---|
| 投影权重 | 可在 $B$ 个新词元间复用；矩阵—向量乘法（GEMV）与矩阵—矩阵乘法（GEMM）的执行方式和权重读取量随批大小改变 |
| 查询位置 | 每条序列一个，总计 $B$ 个 |
| 可见键值位置 | 第 $r$ 条序列为 $L_r$，合计 $\sum_r L_r$ |
| 历史键值复用 | 不同序列的 K/V 通常不同；增加 $B$ 不自动使同一份 K/V 被多个查询复用 |
| 多头路径 | 各头独立的操作按头数累计，共享路径单独计数 |

在历史展开缓存每项只读取一次的相同假设下，固定一个头的注意力核心计算可扩展为：

$$
F_{\mathrm{core,batch}}\approx[2(p+d_v)+1]\sum_{r=1}^{B}L_r,
\qquad
D_{\mathrm{KV,batch}}\approx b(p+d_v)\sum_{r=1}^{B}L_r.
$$

在这一口径下，仅以键值读取为分母的计算强度保持不变。投影计算强度、整个前向的计算强度和内核执行效率仍可能随批大小改变。完整的批次与多头合计、前缀共享及实际内核访存量仍待分析。

## 5. 预填充 × 吸收

### 5.1 单个词元位置

待分析。沿用单位置算子、一对查询与键、整条序列的推导顺序，分别统计 Q 侧的吸收变换、压缩空间中的注意力，以及移至输出侧的 V 投影。分数缩放保持模型原有的 $\sigma$，不因吸收变换改用压缩维度重新计算。

### 5.2 扩展到批次与多头

待分析。分别考虑多个查询之间的复用、跨头共享的 K/V 复用，以及中间张量写入显存的成本，不能直接沿用展开形式的访存量。

## 6. 解码 × 吸收

### 6.1 单个解码词元

待分析。每条序列的查询长度为 Q_len=1；在相同统计范围内，分别计算 Q 侧变换、共享压缩缓存读取、压缩空间中的注意力和输出投影。

### 6.2 扩展到批次与多头

待分析。头数与批大小分别处理。共享缓存的实际读取次数取决于内核能否跨头复用，不能将单头访存量直接乘以 $h$，作为共享缓存的总流量。

得到计算量与访存量之后，再用屋顶线模型比较增加计算和减少访存带来的性能收益。

## 7. 实现相关的额外成本

### 7.1 在线 softmax 为什么还需要重缩放

融合后的注意力可以避免完整分数和权重写入显存，但仍需在多个键值块之间维持数值稳定的累加。设每块包含 $B_K$ 个位置，共有 $n_{\mathrm{tile}}=\lceil L/B_K\rceil$ 块。已有状态为最大值 $m$、指数和 $\ell$ 和尚未归一化的输出累加器 $\boldsymbol u$。处理当前数据块的分数时，更新为：

$$
\begin{aligned}
m'&=\max(m,\max_j s_j),\\
\beta&=\exp(m-m'),\\
p_j&=\exp(s_j-m'),\\
\ell'&=\beta\ell+\sum_jp_j,\\
\boldsymbol u'&=\beta\boldsymbol u+\sum_jp_j\boldsymbol v_j,\\
\boldsymbol o&=\boldsymbol u/\ell\quad\text{（全部 tiles 处理完成后）}.
\end{aligned}
$$

除 QK 和 V 加权累加外，累加器重缩放与最终归一化的向量运算量约为 $O(n_{\mathrm{tile}}d_v+d_v)$；吸收形式的对应维度可能为 $d_c$。内核可能省略首块重缩放、使用以 2 为底的指数函数 exp2，或采用其他求和方式。显式计算归一化权重与在线归一化输出是两种实现，不能把两者的全部归一化成本重复累计。

### 7.2 当前合计之外的成本与适用边界

| 项目 | 统计要求 |
|---|---|
| 缓存写入 | 单头展开形式的完整 K/V 每个新词元写入 $b(p+d_v)$ Bytes；压缩缓存和跨头共享 RoPE 需按各自布局统计 |
| 压缩缓存与运行时展开 | 计入实际展开的历史位置的展开投影和中间读写 |
| 中间结果写入显存 | 独立算子的分数、权重、压缩表示、Q/K/V 可能产生额外显存读写 |
| 物理维度补齐 | 数学维度不一定等于内核实际计算的宽度；参考分块融合注意力算法（FlashAttention）的路径可能将 V 补齐（padding）至 Q/K 维度 |
| 布局变换 | 形状重排或视图可只修改元数据；连续化、实际拼接和广播复制可能搬运数据 |
| 分段并行与结果合并 | 按键值长度分段并行（split-K）时，计入局部输出和归一化统计量的读写及合并运算 |
| RoPE 系数 | 正余弦系数生成、读取、位置缩放和幅度缩放按实现单列 |
| 掩码、偏置、类型转换与量化 | 按配置累计；逻辑掩码不等于读取稠密掩码张量 |
| 通信与缓存元数据 | 张量并行（tensor parallelism，TP）、上下文并行（context parallelism，CP）的通信，以及分页缓存索引等，需与普通计算量和显存数据读写分开统计 |
| 模型块与训练 | 注意力输入端归一化、残差连接、前馈网络、专家混合、反向传播、激活保存与重计算不在当前范围内 |

投影与 RMSNorm 是否融合，决定中间压缩表示是否需要经过一次显存往返；RMSNorm 权重的实际数据类型也需明确。单头例子的 641 FLOPs / 640 Bytes 是所声明假设下的注意力核心参考值，不是完整 MLA 内核的实测计算强度。

## 8. 来源与验证状态

- 苏剑林：[《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》](https://spaces.ac.cn/archives/10091)，2024-05-13。沿用其第 2、3 部分的行向量公式，并补充压缩表示的 RMSNorm。
- 苏剑林：[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)，2021-03-23，2026-10-26 查阅。第 2.4 节采用二维旋转与多维分块对角构造，明确转换为行向量记法；算术计数按本文统计约定推导。
- [DeepSeek-V2 技术报告](https://arxiv.org/html/2405.04434v5)与[官方参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py)，2026-10-25 查阅。参考实现用于确认 RMSNorm、分数缩放、缓存更新及部分路径的 V 补齐，不代表所有部署内核。
- 本地代码核对：Megatron-LM/megatron/core/transformer/multi_latent_attention.py 中的降维投影、压缩表示归一化与吸收路径，以及 experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py 中的在线 softmax 更新。核对时仓库 HEAD 为 83392004c81b6e477002ae84d4f0b1271518c1d1。
- 本文的计算量、显存访存量和计算强度均为所列假设下的推导，尚未运行模型或进行内核性能分析。实际读取次数、算子融合、缓存布局与吞吐量需结合目标实现验证。
