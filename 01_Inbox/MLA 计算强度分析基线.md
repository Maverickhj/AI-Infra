---
type: knowledge
status: draft
created: 2026-10-25
updated: 2026-10-27
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
| 预填充 | 已推导单位置、单序列及多头多批次的计算量；已在明确中间张量边界和分块复用假设后给出访存量与计算强度 | 已推导 Q 侧变换、潜在空间注意力和后置 V 投影；已给出查询与跨头复用下的访存模型及展开形式对照 |
| 解码 | 已建立单个解码词元的注意力计算与前向成本；已明确批次扩展的边界 | 已推导单位置、多头与批次成本；已计入当前词元的缓存写入，并在相同显存边界下与展开形式比较 |

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

后续按公式项统计成本，统一使用以下分支符号；分支下标 $C$、$R$ 分别标识内容与位置分支：

$$
\begin{aligned}
\boldsymbol q_{t,C}^{(s)}&=\boldsymbol c_t'W_{qc}^{(s)}
\in\mathbb R^{1\times d_k},&
\boldsymbol q_{t,R}^{(s)}&=(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t
\in\mathbb R^{1\times d_r},\\
\boldsymbol k_{i,C}^{(s)}&=\boldsymbol c_iW_{kc}^{(s)}
\in\mathbb R^{1\times d_k},&
\boldsymbol k_{i,R}&=(\boldsymbol x_iW_{kr})\mathcal R_i
\in\mathbb R^{1\times d_r}.
\end{aligned}
$$

因此 $\boldsymbol q_t^{(s)}=[\boldsymbol q_{t,C}^{(s)},\boldsymbol q_{t,R}^{(s)}]$，$\boldsymbol k_i^{(s)}=[\boldsymbol k_{i,C}^{(s)},\boldsymbol k_{i,R}]$。成本表使用这些公式项或已定义的张量符号指代操作与数据。

将各头的结果拼接后，进行输出投影（output projection）：

$$
\boldsymbol y_t
=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O
=\sum_{s=1}^{h}\boldsymbol o_t^{(s)}W_O^{(s)},
\qquad W_O^{(s)}\in\mathbb R^{d_v\times d}.
$$

单头的输出成本只表示该头对最终结果的贡献。完整多头矩阵乘法通常已包含跨头求和，不应再无条件叠加一次独立的求和成本。

为便于成本分析，本文按实际运算划分统计范围，并约定以下简称。分组与边界由本文明确，运算采用常规的线性投影与注意力定义。

| 运算与简称 | 本文的统计范围 |
|---|---|
| 多个词元的线性投影，后文简称“投影” | 将多个位置的向量组成矩阵，一起执行 $Y=XW$；包括 Q/KV 降维投影、Q/K/V 展开投影和输出投影。各行分别变换，使用同一份权重 |
| 注意力计算（QK、softmax、V 聚合），后文简称“注意力核心” | 从已生成的 Q、K、V 开始，计算 QK 分数与缩放、softmax 权重及 V 加权聚合，得到各头输出；投影、RMSNorm、RoPE 和缓存构建在此范围之外 |

RMSNorm 与 RoPE 分别属于归一化和位置编码，按独立操作统计；完整前向合计将它们与投影、注意力计算一起累计。

注意力核心的数学过程包含 softmax。本文的 $F_{\mathrm{core}}$ 等计算量公式只累计 QK 内积、分数缩放和 V 加权累加，softmax 操作与在线归一化的额外成本按第 2.4、7.1 节单列。因此，$I_{\mathrm{core}}=F_{\mathrm{core}}/D_{\mathrm{core}}$ 表示所列主要算术操作对应的计算强度，完整内核的执行成本还需加入这些单列项。

### 2.3 计算量与显存访存量如何计数

- 一次乘法或加法计为一次浮点运算（floating-point operation，FLOP），运算总次数记为 FLOPs；融合乘加（fused multiply-add，FMA）计 2 FLOPs。矩阵乘法使用 $2mnk$ 的主项估算。
- 指数、比较、除法和平方根倒数分别列出操作次数，不将它们与普通加乘视为等成本指令。平方根倒数在公式中记为 rsqrt。
- 显存访存量指高带宽内存（High Bandwidth Memory，HBM）与片上存储之间的数据搬运量，不等于张量占用容量。同一份数据重复读取时，需要重复计数。
- 理论模型统一采用每元素 $b$ 字节，包含 RMSNorm 权重；实际数据类型（dtype）不同时，应按张量分别核算。片上累加可以使用 32 位浮点格式（FP32）。
- $\boldsymbol x_tW_c'$、$\boldsymbol x_iW_c$、$\mathrm{RMSNorm}(\boldsymbol z_t^Q)$、$\mathrm{RMSNorm}(\boldsymbol z_i^{KV})$ 和 $\boldsymbol k_{i,R}=(\boldsymbol x_iW_{kr})\mathcal R_i$ 由各头共享，按新词元的位置数计数。
- 分数缩放（score scaling）按每个分数一次乘法计数，$\sigma$ 已预先计算。若计算内核（kernel）将缩放移到 Q 或其他操作中，应按实际执行方式调整。
- 偏置（bias）、注意力输入端的归一化、残差连接（residual connection）、随机失活（dropout）、掩码（mask）处理和通信不纳入当前合计，见第 7 节。缓存写入按各节边界计数；第 6.4 节显式计入当前词元的缓存写入，不额外复制缓存。

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

| 操作 | 公式项 | 运算量 | 执行频率 |
|---|---|---:|---|
| Q 降维投影 | $\boldsymbol z_t^Q=\boldsymbol x_tW_c'$ | $2dd_c'$ FLOPs | 每个新词元一次，多头共享 |
| Q 压缩表示的 RMSNorm | $\boldsymbol c_t'=\mathrm{RMSNorm}(\boldsymbol z_t^Q)$ | $4d_c'+1$ FLOPs，另有 1 次 rsqrt | 每个新词元一次，多头共享 |
| KV 降维投影 | $\boldsymbol z_i^{KV}=\boldsymbol x_iW_c$ | $2dd_c$ FLOPs | 每个新词元一次，多头共享 |
| KV 压缩表示的 RMSNorm | $\boldsymbol c_i=\mathrm{RMSNorm}(\boldsymbol z_i^{KV})$ | $4d_c+1$ FLOPs，另有 1 次 rsqrt | 每个新词元一次，多头共享 |
| Q 展开投影 | $[\boldsymbol c_t'W_{qc}^{(s)},\boldsymbol c_t'W_{qr}^{(s)}]$ | $2d_c'(d_k+d_r)$ FLOPs | 每个当前查询、每个头一次 |
| Q 旋转 | $(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t$，仅乘 $\mathcal R_t$ | $3d_r$ FLOPs | 每个当前查询、每个头一次 |
| 内容 K 展开投影 | $\boldsymbol c_iW_{kc}^{(s)}$ | $2d_cd_k$ FLOPs | 每个新 K、每个头一次 |
| 位置 K 投影 | $\boldsymbol x_iW_{kr}$ | $2dd_r$ FLOPs | 每个新 K 一次，所有头共享 |
| K 旋转 | $(\boldsymbol x_iW_{kr})\mathcal R_i$，仅乘 $\mathcal R_i$ | $3d_r$ FLOPs | 每个新 K 一次，所有头共享 |
| V 展开投影 | $\boldsymbol c_iW_v^{(s)}$ | $2d_cd_v$ FLOPs | 每个新 V、每个头一次 |
| QK 内积 | $\boldsymbol q_t^{(s)}\boldsymbol k_i^{(s)\top}$ | 约 $2p$ FLOPs | 每对查询与键、每个头一次 |
| 分数缩放 | $s_{t,i}^{(s)}=(\boldsymbol q_t^{(s)}\boldsymbol k_i^{(s)\top})\sigma$，仅乘 $\sigma$ | 1 FLOP | 每对查询与键、每个头一次 |
| Softmax | $\alpha_{t,i}^{(s)}=\operatorname{softmax}_i(s_{t,i}^{(s)})$ | 对长度 $L$ 的整组分数为 $O(L)$ | 每个查询、每个头做一次整体归一化 |
| V 加权累加 | $\boldsymbol o_t^{(s)}\mathrel{+}=\alpha_{t,i}^{(s)}\boldsymbol v_i^{(s)}$ | 约 $2d_v$ FLOPs | 每对查询与值、每个头一次 |
| 输出投影 | $\boldsymbol o_t^{(s)}W_O^{(s)}$ | $2d_vd$ FLOPs | 历史位置求和完成后，每个查询、每个头一次 |

每个位置的 K、V 只生成一次；Q 投影和输出投影也按查询位置执行一次。只有 QK 内积、分数缩放和 V 加权累加按查询与键的配对数累计。

KV 降维投影与位置 K 投影可融合为输出宽度 $d_c+d_r$ 的矩阵乘法，合计 $2d(d_c+d_r)$ FLOPs。其中已经包含位置投影的 $2dd_r$，不能重复计数。

### 3.2 单个位置：明确访存假设后估算成本

本节统计单向量算子的理想访存成本，不代表实际预填充的批量访存量。假设：权重初始未驻留片上，每个元素从显存读入一次；输入读取一次；中间激活在片上复用；输出写回一次；正余弦与分数缩放系数已在片上。Softmax、注意力对历史 K/V 的读取及持久缓存写入不计入本节。

固定词元位置 $t$ 和一个头 $s$，并将该位置的共享公式项计一次，定义：

$$
\begin{aligned}
A&=d_c'p+d_c(d_k+d_v)+dd_r+d_vd,\\
C&=d(d_c'+d_c),\\
G&=d_c'+d_c.
\end{aligned}
$$

$A$ 对应 $W_{qc}^{(s)}$、$W_{qr}^{(s)}$、$W_{kc}^{(s)}$、$W_v^{(s)}$、$W_{kr}$ 和 $W_O^{(s)}$ 的权重元素数；$C$ 对应 $W_c'$、$W_c$；$G$ 对应两次 RMSNorm 的权重。

| 公式项或张量 | 计算量 | 显存访存量 |
|---|---:|---:|
| $\boldsymbol x_tW_c'$、$\boldsymbol x_tW_c$ | $2C$ | 权重：$bC$ |
| $\mathrm{RMSNorm}(\boldsymbol z_t^Q)$、$\mathrm{RMSNorm}(\boldsymbol z_t^{KV})$ | $4G+2$，另有 2 次 rsqrt | 权重：$bG$；$\boldsymbol c_t'$、$\boldsymbol c_t$ 在片上复用 |
| $\boldsymbol c_t'W_{qc}^{(s)}$、$\boldsymbol c_t'W_{qr}^{(s)}$、$\boldsymbol c_tW_{kc}^{(s)}$、$\boldsymbol c_tW_v^{(s)}$、$\boldsymbol x_tW_{kr}$、$\boldsymbol o_t^{(s)}W_O^{(s)}$ | $2A$ | 权重：$bA$ |
| $(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t$、$(\boldsymbol x_tW_{kr})\mathcal R_t$，仅乘旋转矩阵 | $6d_r$ | 正余弦系数在片上复用 |
| $\boldsymbol x_t$ 读取、$\boldsymbol y_t$ 写回 | — | 输入读取与输出写回：$2bd$ |

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

本节固定一条序列、一个注意力头，假设没有前缀缓存，采用位置从 1 开始、包含当前位置的因果注意力。计算量分为两类：

| 类别 | 成本 | 执行次数 |
|---|---|---|
| 单位置操作 | $F_{\mathrm{token,ops}}$ | 每个词元位置一次 |
| 注意力交互 | $F_{\mathrm{pair}}$ | 每对查询与可见键值位置一次 |

第 3.2 节的单位置成本为：

$$
F_{\mathrm{token,ops}}\approx2(A+C)+4G+2+6d_r.
$$

它包含该位置的降维投影、RMSNorm、展开投影、Q/K 的 RoPE 和输出投影。在模型维度固定时，这部分计算量不随可见历史长度增长。输出投影虽然需要等待注意力聚合完成，但处理的向量维度固定，因此仍按每个位置一次计数。

查询位置 $t$ 可以访问位置 $1,\ldots,t$，因此它与 $t$ 个键值位置交互。每次交互的普通加乘运算量为：

$$
F_{\mathrm{pair}}
=\underbrace{2p}_{\text{QK 内积}}
+\underbrace{1}_{\text{分数缩放}}
+\underbrace{2d_v}_{\text{V 加权累加}}
=2(p+d_v)+1.
$$

将位置 $t$ 自身的单位置操作与该查询的注意力计算合并，可得：

$$
F_t\approx F_{\mathrm{token,ops}}+tF_{\mathrm{pair}}.
$$

以长度 $S=4$ 的序列为例：

| 查询位置 $t$ | 可见键值位置 | 交互次数 | 该位置的计算量 |
|---|---|---:|---|
| 1 | 1 | 1 | $F_{\mathrm{token,ops}}+F_{\mathrm{pair}}$ |
| 2 | 1、2 | 2 | $F_{\mathrm{token,ops}}+2F_{\mathrm{pair}}$ |
| 3 | 1、2、3 | 3 | $F_{\mathrm{token,ops}}+3F_{\mathrm{pair}}$ |
| 4 | 1、2、3、4 | 4 | $F_{\mathrm{token,ops}}+4F_{\mathrm{pair}}$ |
| 合计 | — | 10 | $4F_{\mathrm{token,ops}}+10F_{\mathrm{pair}}$ |

推广到长度 $S$，逐位置求和：

$$
\begin{aligned}
F_{\mathrm{prefill,expanded}}^{(h=1)}
&\approx\sum_{t=1}^{S}F_t\\
&\approx\sum_{t=1}^{S}
\left(F_{\mathrm{token,ops}}+tF_{\mathrm{pair}}\right)\\
&=S F_{\mathrm{token,ops}}
+\left(\sum_{t=1}^{S}t\right)F_{\mathrm{pair}}\\
&=S F_{\mathrm{token,ops}}
+\frac{S(S+1)}2F_{\mathrm{pair}}.
\end{aligned}
$$

$S$ 个位置各执行一次固定成本，形成第一项；注意力矩阵包含对角线的下三角区域共有 $S(S+1)/2$ 个元素，对应第二项。代入两类成本的定义：

$$
\boxed{
F_{\mathrm{prefill,expanded}}^{(h=1)}
\approx
S\left[2(A+C)+4G+2+6d_r\right]
+\frac{S(S+1)}2\left[2(p+d_v)+1\right]
}
$$

历史 K、V 在各自位置只生成一次，其成本已计入第一项；其他查询读取它们时，不能再次计入投影和归一化。预填充可以并行处理多个位置，但并行执行不改变上述数学操作总数。

该式仍不含 softmax、单列的 rsqrt 等操作，也不利用首个位置权重恒为 1 的特殊优化。内核的分块、补齐或掩码执行可能产生额外计算，需要与该主项估算区分。

这里尚未推导整条序列的访存量。它还取决于不同查询位置能否复用同一份 K/V，不能直接将解码的独立 K/V 读取量按配对数累加，作为预填充的访存量。

### 3.4 扩展到批次与多头：哪些数据能够复用

本节继续采用无前缀缓存、长度为 $S$ 的因果注意力，增加批大小 $B$ 和头数 $h$。一次处理的词元位置数为 $N=BS$。先推导不依赖内核分块的数学计算量，再明确张量写回显存的边界，估算访存量和计算强度。这里的批次包含不同序列，不假设共享前缀。

#### 3.4.1 多头的计算量：共享路径不能重复累计

将第 3.2 节的单位置成本拆成共享路径与各头路径：

| 路径 | 公式项 | 每个词元位置的计算量 |
|---|---|---:|
| 多头共享 | $\boldsymbol x_tW_c'$、$\boldsymbol x_tW_c$、$\mathrm{RMSNorm}(\boldsymbol z_t^Q)$、$\mathrm{RMSNorm}(\boldsymbol z_t^{KV})$、$\boldsymbol k_{t,R}=(\boldsymbol x_tW_{kr})\mathcal R_t$ | $f_{\mathrm{shared}}=2[C+dd_r]+4G+2+3d_r$ |
| 每头独立 | $\boldsymbol q_t^{(s)}=[\boldsymbol c_t'W_{qc}^{(s)},(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t]$、$\boldsymbol k_{t,C}^{(s)}=\boldsymbol c_tW_{kc}^{(s)}$、$\boldsymbol v_t^{(s)}=\boldsymbol c_tW_v^{(s)}$、$\boldsymbol o_t^{(s)}W_O^{(s)}$ | $f_{\mathrm{head}}=2[d_c'p+d_c(d_k+d_v)+d_vd]+3d_r$ |

其中 $C=d(d_c'+d_c)$、$G=d_c'+d_c$。$\boldsymbol k_{t,R}$ 由所有头共享；$(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t$ 按头执行，因此旋转成本累计 $h$ 次。$[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O$ 按一个矩阵乘法计数，$W_O$ 的形状为 $hd_v\times d$，不另计逐头输出相加。

每个词元位置的固定成本为：

$$
F_{\mathrm{token,ops}}^{(h)}
\approx f_{\mathrm{shared}}+h f_{\mathrm{head}}.
$$

每条序列、每个头有 $S(S+1)/2$ 对有效查询与键，因此：

$$
\boxed{
F_{\mathrm{prefill,expanded}}^{(B,h)}
\approx BS\left(f_{\mathrm{shared}}+h f_{\mathrm{head}}\right)
+Bh\frac{S(S+1)}2F_{\mathrm{pair}}
}
$$

这里 $F_{\mathrm{pair}}=2(p+d_v)+1$。共享的 K 位置分支虽然只生成一次，但仍需与每个头各自的 Q 做内积，因此注意力交互项整体乘以 $Bh$。令 $B=h=1$，该式退化为第 3.3 节结果。

#### 3.4.2 多个词元的线性投影：复用同一份权重

预填充将 $N=BS$ 个词元位置的向量组成矩阵，一起执行线性投影。这些位置可以来自同一条序列，也可以来自批次中的不同序列；每一行独立做变换，所有行使用同一份投影权重。

以单个投影 $Y=XW$ 为例，$X$ 的形状为 $N\times d_{\mathrm{in}}$，$W$ 为 $d_{\mathrm{in}}\times d_{\mathrm{out}}$。本节只统计该矩阵乘法，RMSNorm 与 RoPE 另外累计。假设输入和权重各从显存读一次，输出写回一次：

$$
I_{\mathrm{projection}}
\approx
\frac{2Nd_{\mathrm{in}}d_{\mathrm{out}}}
{b(Nd_{\mathrm{in}}+d_{\mathrm{in}}d_{\mathrm{out}}+Nd_{\mathrm{out}})}.
$$

权重项 $bd_{\mathrm{in}}d_{\mathrm{out}}$ 不随 $N=BS$ 增长，而计算量随 $N$ 增长，因此增大批大小或序列长度，可以通过权重复用提高投影的计算强度。激活读写仍随 $N$ 增长，计算强度不会无限增加：

$$
I_{\mathrm{projection}}
\xrightarrow{N\to\infty}
\frac{2d_{\mathrm{in}}d_{\mathrm{out}}}
{b(d_{\mathrm{in}}+d_{\mathrm{out}})}.
$$

多个头可以合并为更宽的投影矩阵，共享输入读取；各头的权重仍不同，需要全部计数。因此，增加头数不能直接视为复用同一份投影权重。本节讨论 $B$、$h$ 的变化时，均保持其他维度不变。

#### 3.4.3 注意力计算：同一序列内的查询复用 K/V

本节分析 Q、K、V 已生成之后的注意力计算，即第 2.2 节约定的注意力核心。先固定一条序列和一个头。假设 Q 从显存读取一次，输出写回一次，分数与权重通过融合的 softmax 保留在片上，不写回完整注意力矩阵。

设 $\rho$ 为该头全部 K/V 元素从显存读取的平均次数。此处显式存储完整的 $p$ 维 $\boldsymbol k_i^{(s)}=[\boldsymbol k_{i,C}^{(s)},\boldsymbol k_{i,R}]$：$\boldsymbol k_{i,R}$ 被复制到各头的 $\boldsymbol k_i^{(s)}$ 中，因此各头的 K/V 读取分别计数。整个批次的注意力计算成本如下，其中 softmax 运算量单列：

| 项目 | 多头、多批次的合计 |
|---|---:|
| 因果注意力有效 Q–K 位置对应的计算量 | $Bh\,S(S+1)F_{\mathrm{pair}}/2$ |
| $\boldsymbol q_t^{(s)}$ 读取 | $bBS hp$ |
| $\boldsymbol k_i^{(s)}$、$\boldsymbol v_i^{(s)}$ 读取 | $bBS h\rho(p+d_v)$ |
| $\boldsymbol o_t^{(s)}$ 写回 | $bBS hd_v$ |

“有效 Q–K 位置”指注意力矩阵中满足 $i\le t$ 的查询—键配对位置，即包含对角线的下三角区域。每个位置对应的 $F_{\mathrm{pair}}$ 包含 QK 内积、分数缩放和该位置的 V 加权累加，不包含 softmax 的成本。

由此得到：

$$
D_{\mathrm{core}}
\approx bBS h(1+\rho)(p+d_v),
$$

$$
\boxed{
I_{\mathrm{core}}
\approx
\frac{(S+1)F_{\mathrm{pair}}}
{2b(1+\rho)(p+d_v)}
}
$$

当 $S$ 与 $\rho$ 固定时，$B$、$h$ 从比值中约去：增加独立序列或独立头，会同时增加注意力计算与 K/V 读取，不自动提高这部分的计算强度。

为了具体估算 $\rho$，设一个查询数据块包含连续的 $M$ 个位置，块内查询在片上复用 K/V；每个查询块读取其末尾位置之前的完整 K/V 前缀，不同查询块之间不复用显存读取。令 $J=\lceil S/M\rceil$，则：

$$
\rho=\frac1S\sum_{j=1}^{J}\min(jM,S).
$$

若 $S$ 可以被 $M$ 整除，$S=JM$，则：

$$
\rho=\frac{M(1+2+\cdots+J)}S=\frac{J+1}2
=\frac{S/M+1}2.
$$

例如 $S=4$、$M=2$：查询位置 1、2 组成的块读取 2 个 K/V 位置；查询位置 3、4 组成的块读取 4 个位置，共读取 6 个位置，而不是按每对交互读取 $1+2+3+4=10$ 个位置。因此 $\rho=6/4=1.5$。分块内仍应用因果掩码，计算量按 10 对有效交互计数。

当 $M=1$ 时，$\rho=(S+1)/2$，对应各查询分别读取自己的 K/V 前缀；当 $M=S$ 时，$\rho=1$，对应整条序列的查询共享一次 K/V 读取。后一情形只是理想参照，能否实现取决于片上容量。

$I_{\mathrm{core}}$ 用于估算注意力核心主要算术操作对应的计算强度。为了进一步解释查询复用为什么能够提高计算强度，引入局部指标 $I_{\mathrm{KV,reuse}}$，回答：**一份 K/V 从显存读入一次，供多个有效查询使用后，每读取一个字节能支撑多少计算？**

对确实被 $M$ 个有效查询复用的一份 K/V，该次读取的数据量为 $b(p+d_v)$，支撑的计算量为 $M F_{\mathrm{pair}}$，因此：

$$
I_{\mathrm{KV,reuse}}
\approx\frac{M F_{\mathrm{pair}}}{b(p+d_v)}.
$$

固定维度时，分母不随复用次数变化，分子随有效查询数增加。这个比值说明一次数据搬运被更多计算分摊的收益。式中的 $M$ 要求这些查询都实际使用该份 K/V；查询块包含 $M$ 个位置，并不保证其中每份 K/V 都被 $M$ 个有效查询使用，因为部分配对可能被因果掩码遮住。

两项指标的统计范围为：

| 指标 | $I_{\mathrm{core}}$ | $I_{\mathrm{KV,reuse}}$ |
|---|---|---|
| 统计范围 | 整个注意力核心 | 一份 K/V 的一次读取及其复用 |
| 计算量 | 全部有效 Q–K 位置对应的计算量 | 使用该份 K/V 的 $M$ 个有效查询对应的计算量 |
| 访存量 | Q 读取、K/V 读取、注意力输出写回 | 仅该份 K/V 的一次读取 |
| 分析目的 | 估算注意力核心整体的计算与访存比例 | 解释增加单次读取的复用次数为何提高计算强度 |

两者均沿用 softmax 等额外操作单列的口径，不包含投影与缓存构建。$I_{\mathrm{KV,reuse}}$ 的分子、分母仅覆盖局部，不能替代 $I_{\mathrm{core}}$。第 8.1 节给出同一执行示例下两者的数值对比。

上面的 $\rho$ 模型从整条序列统计 K/V 的平均读取次数，计入了因果前缀长度的变化；实际的分块宽度、缓存命中、重复加载和掩码执行可能不同，需要按实现调整。

#### 3.4.4 合并投影与注意力：明确中间张量边界

计算量可以直接相加；访存量还需要确定算子之间哪些张量写回显存。本节选择以下可核对的参考路径：

1. 读取 $\boldsymbol x_t$ 一次，执行 $\boldsymbol x_tW_c'$、$\boldsymbol x_tW_c$、两次 RMSNorm 和 $(\boldsymbol x_tW_{kr})\mathcal R_t$，写出 $\boldsymbol c_t'$、$\boldsymbol c_t$、$\boldsymbol k_{t,R}$。
2. 将上述结果各读一次并跨头复用，计算并写出 $\boldsymbol q_t^{(s)}=[\boldsymbol c_t'W_{qc}^{(s)},(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t]$、$\boldsymbol k_t^{(s)}=[\boldsymbol c_tW_{kc}^{(s)},\boldsymbol k_{t,R}]$、$\boldsymbol v_t^{(s)}=\boldsymbol c_tW_v^{(s)}$。
3. 读取 $\boldsymbol q_t^{(s)}$、$\boldsymbol k_i^{(s)}$、$\boldsymbol v_i^{(s)}$，计算注意力并写出 $\boldsymbol o_t^{(s)}$；$\boldsymbol k_i^{(s)}$、$\boldsymbol v_i^{(s)}$ 的平均读取次数为 $\rho$。
4. 读取 $[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]$，计算并写出 $\boldsymbol y_t=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O$。

假设各阶段的权重在 $N=BS$ 个位置间复用，各元素从显存读一次；前处理和展开阶段内部的临时结果不写回显存；不计持久 KV cache 的额外写入。该路径用于明确计数边界，不表示目标框架一定采用这些融合方式。

所有矩阵投影的权重元素数为：

$$
W_h=C+dd_r+h\left[d_c'p+d_c(d_k+d_v)+d_vd\right].
$$

其中共享投影权重只计一次，各头的展开与输出权重按 $h$ 计数；另有 $G$ 个 RMSNorm 权重。按张量累计显存流量：

| 数据 | 整个预填充的显存访存量 |
|---|---:|
| 矩阵投影与 RMSNorm 权重 | $b(W_h+G)$ |
| $\boldsymbol x_t$ 读取、$\boldsymbol y_t$ 写回 | $2bNd$ |
| $\boldsymbol c_t'$、$\boldsymbol c_t$、$\boldsymbol k_{t,R}$：写出、再读入 | $2bN(G+d_r)$ |
| $\boldsymbol q_t^{(s)}$、$\boldsymbol o_t^{(s)}$：写出、再读入 | $2bNh(p+d_v)$ |
| $\boldsymbol k_t^{(s)}$、$\boldsymbol v_t^{(s)}$ 写出 | $bNh(p+d_v)$ |
| $\boldsymbol k_i^{(s)}$、$\boldsymbol v_i^{(s)}$ 供注意力读取 | $bNh\rho(p+d_v)$ |

$\boldsymbol k_{t,R}$ 的生成成本只计一次；本路径将其复制到各头的 $\boldsymbol k_t^{(s)}$ 中，复制后的写入与读取已包含在该张量的流量中。若 $\boldsymbol k_{t,R}$ 单独存储并跨头复用，应重新计算相应流量。

合计得到：

$$
\boxed{
D_{\mathrm{prefill,expanded}}^{(B,h)}
\approx b\left\{
W_h+G+BS\left[2d+2(G+d_r)+h(3+\rho)(p+d_v)\right]
\right\}
}
$$

$$
\boxed{
I_{\mathrm{prefill,expanded}}^{(B,h)}
\approx
\frac{
BS(f_{\mathrm{shared}}+h f_{\mathrm{head}})
+Bh\,S(S+1)F_{\mathrm{pair}}/2
}{
b\left\{
W_h+G+BS[2d+2(G+d_r)+h(3+\rho)(p+d_v)]
\right\}
}
}
$$

这个合计与第 3.2 节的访存边界不同：它显式包含压缩表示、Q/K/V 和注意力输出经过显存的流量，不能由 $BS$ 倍单位置访存量代替。若取消阶段间写回、权重重复加载或增加缓存写入，应修改对应项。

#### 3.4.5 数值例子：批大小、头数与查询复用分别改变什么

取 $d=1024$、$d_c'=d_c=256$、$d_k=d_v=128$、$d_r=64$，故 $p=192$；$b=2$、$S=128$，查询块大小取 $M=32$，则 $\rho=(128/32+1)/2=2.5$。

$$
f_{\mathrm{shared}}=1{,}181{,}890,\qquad
f_{\mathrm{head}}=491{,}712,\qquad
F_{\mathrm{pair}}=641.
$$

下面使用第 3.4.4 节的阶段边界；访存量单位为 Bytes，计算强度单位为 FLOPs/Byte：

| $B$ | $h$ | 总计算量（FLOPs） | 总访存量（Bytes） | 总计算强度 | 仅注意力核心的计算强度 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 219,513,152 | 2,941,952 | 74.61 | 18.46 |
| 1 | 8 | 697,131,776 | 9,536,512 | 73.10 | 18.46 |
| 4 | 8 | 2,788,527,104 | 22,807,552 | 122.26 | 18.46 |
| 4 | 16 | 4,971,926,528 | 41,157,632 | 120.80 | 18.46 |

固定 $h=8$，从 $B=1$ 增至 $B=4$，计算量增加 4 倍，但权重只读一次，因此总访存量增加不到 4 倍，总计算强度提高。各序列的 K/V 不同，注意力核心的计算强度保持不变。

固定其他维度，增加 $h$ 会增加独立头的权重、激活与注意力交互；共享路径只增加了复用，并不保证总计算强度提高。本例中总计算强度略降，说明“头数增加”与“批量权重复用增加”不能等同。

固定 $B,h$，仅改变查询块大小 $M$，注意力核心的结果为：

| $M$ | K/V 平均读取次数 $\rho$ | 注意力核心计算强度（FLOPs/Byte） |
|---:|---:|---:|
| 1 | 64.5 | 0.99 |
| 32 | 2.5 | 18.46 |
| 128 | 1 | 32.30 |

这里计算量相同，计算强度的提高来自多个查询对 K/V 读取的复用。$M=128$ 是理想参照，不能仅根据表格认定其可实现或性能最优。

以上结果仍将 softmax、rsqrt 与在线 softmax 的额外操作单列，未计分块补齐等额外执行成本；没有进行内核实测。整个前向的合计计算强度只反映所声明边界的总成本，各阶段仍可能分别受计算或访存限制，不能用合计值替代逐算子的性能判断。

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

| 公式项 | 运算量 | 显存读取 | 显存写入 |
|---|---:|---:|---:|
| $\boldsymbol q_t^{(s)}\boldsymbol k_i^{(s)\top}$ | 约 $2p$ FLOPs | $\boldsymbol k_i^{(s)}$ 读取：$bp$ Bytes | 分数在片上，0 |
| $(\boldsymbol q_t^{(s)}\boldsymbol k_i^{(s)\top})\sigma$，仅乘 $\sigma$ | 1 FLOP | 分数与 $\sigma$ 在片上，0 | 0 |
| $\alpha_{t,i}^{(s)}=\operatorname{softmax}_i(s_{t,i}^{(s)})$ | 每个键值位置摊销约 5 次标量操作 | 分数在片上，0 | 权重在片上，0 |
| $\boldsymbol o_t^{(s)}\mathrel{+}=\alpha_{t,i}^{(s)}\boldsymbol v_i^{(s)}$ | 约 $2d_v$ FLOPs | $\boldsymbol v_i^{(s)}$ 读取：$bd_v$ Bytes | 累加器在片上，0 |
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

| 公式项 | 运算量 | 显存读取 | 显存写入 |
|---|---:|---:|---:|
| $\boldsymbol z_t^Q=\boldsymbol x_tW_c'$ | $2dd_c'$ | 权重 $bdd_c'$；$\boldsymbol x_t$：$bd$ | 0，结果留片上 |
| $\boldsymbol c_t'=\mathrm{RMSNorm}(\boldsymbol z_t^Q)$ | $4d_c'+1$，另有 1 次 rsqrt | $\gamma_Q$：$bd_c'$ | 0 |
| $\boldsymbol z_t^{KV}=\boldsymbol x_tW_c$ | $2dd_c$ | 权重 $bdd_c$；复用 $\boldsymbol x_t$ | 0 |
| $\boldsymbol c_t=\mathrm{RMSNorm}(\boldsymbol z_t^{KV})$ | $4d_c+1$，另有 1 次 rsqrt | $\gamma_{KV}$：$bd_c$ | 0 |
| $[\boldsymbol c_t'W_{qc}^{(s)},\boldsymbol c_t'W_{qr}^{(s)}]$ | $2d_c'p$ | 权重 $bd_c'p$ | 0 |
| $(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t$，仅乘 $\mathcal R_t$ | $3d_r$ | 0 | 0 |
| $\boldsymbol c_tW_{kc}^{(s)}$、$\boldsymbol x_tW_{kr}$ | $2d_cd_k+2dd_r$ | 权重 $b(d_cd_k+dd_r)$ | 0 |
| $(\boldsymbol x_tW_{kr})\mathcal R_t$，仅乘 $\mathcal R_t$ | $3d_r$ | 0 | 0 |
| $\boldsymbol v_t^{(s)}=\boldsymbol c_tW_v^{(s)}$ | $2d_cd_v$ | 权重 $bd_cd_v$ | 0 |
| $(\boldsymbol q_t^{(s)}\boldsymbol k_i^{(s)\top})\sigma$，累计 $L$ 个位置 | $L(2p+1)$ | $\boldsymbol k_i^{(s)}$：$bLp$ | 0 |
| $\alpha_{t,i}^{(s)}=\operatorname{softmax}_i(s_{t,i}^{(s)})$ | 特殊操作和求和单列，见第 2.4 节 | $s_{t,i}^{(s)}$ 在片上，0 | 0 |
| $\boldsymbol o_t^{(s)}=\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol v_i^{(s)}$ | $2Ld_v$ | $\boldsymbol v_i^{(s)}$：$bLd_v$ | 0 |
| $\boldsymbol o_t^{(s)}W_O^{(s)}$ | $2d_vd$ | 权重 $bd_vd$ | $\boldsymbol y_t$：$bd$ |

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

$B=1,h=1$ 只是本层推导的简化条件。扩展到多头时，$\boldsymbol x_tW_c'$、$\boldsymbol x_tW_c$、两次 RMSNorm 和 $\boldsymbol k_{t,R}$ 只计一次，不能把全部合计直接乘以 $h$。

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

本节参考 vLLM FlashMLA 路径采用的基本组织：执行 $\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$，在共享 $[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 上计算注意力，再执行 $(\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol c_i)W_v^{(s)}$ 和 $[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O$。实现可以在部分短预填充中选择这条路径；本节只分析其数学操作与数据搬运，不展开调度与内核细节。

为与第 3 节直接比较，继续采用无前缀缓存、$B$ 条长度为 $S$ 的序列、$h$ 个头和包含对角线的因果注意力，令 $N=BS$。已有长前缀时，查询位置数与缓存位置数需要分别计数，不直接套用本节的等长公式。

成本符号沿用第 2 节：$F$ 表示所列主要算术操作的计算量，单位为 FLOPs；$D$ 表示显存读写量，单位为 Bytes；$I=F/D$ 表示对应计算强度，单位为 FLOPs/Byte。下标与上标用于标明计算形式和统计范围，具体含义在首次使用时说明。

### 5.1 计算组织：Q 变换、潜在空间聚合与后置 V 投影

$\boldsymbol c_t'$、$\boldsymbol c_i$、$\boldsymbol q_{t,C}^{(s)}$、$\boldsymbol q_{t,R}^{(s)}$ 和 $\boldsymbol k_{i,R}$ 沿用第 2.2 节的定义。符号上的帽号标识本节吸收形式的注意力输入。通过 $\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$ 构造 $\widehat{\boldsymbol q}_t^{(s)}$，并定义对应的 $\widehat{\boldsymbol k}_i$、$\widehat{\boldsymbol v}_i$；后两者由各头共享，因此不带头下标 $s$：

$$
\begin{aligned}
\widehat{\boldsymbol q}_t^{(s)}
&=[\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top},
\ \boldsymbol q_{t,R}^{(s)}]
\in\mathbb R^{1\times(d_c+d_r)},\\
\widehat{\boldsymbol k}_i
&=[\boldsymbol c_i,\ \boldsymbol k_{i,R}]
\in\mathbb R^{1\times(d_c+d_r)},\\
\widehat{\boldsymbol v}_i
&=\boldsymbol c_i\in\mathbb R^{1\times d_c}.
\end{aligned}
$$

记 $p_a=d_c+d_r$ 为 $\widehat{\boldsymbol q}_t^{(s)}$、$\widehat{\boldsymbol k}_i$ 的维度，下标 $a$ 表示 absorbed（吸收形式）；$\boldsymbol q_t^{(s)}$、$\boldsymbol k_i^{(s)}$ 的维度仍为 $p=d_k+d_r$。两种形式的内积相同：

$$
\begin{aligned}
\widehat{\boldsymbol q}_t^{(s)}\widehat{\boldsymbol k}_i^\top
&=(\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top})\boldsymbol c_i^\top
+\boldsymbol q_{t,R}^{(s)}\boldsymbol k_{i,R}^\top\\
&=\boldsymbol q_{t,C}^{(s)}(\boldsymbol c_iW_{kc}^{(s)})^\top
+\boldsymbol q_{t,R}^{(s)}\boldsymbol k_{i,R}^\top.
\end{aligned}
$$

分数仍乘模型原有的 $\sigma=1/\sqrt p$；变换后的维度 $p_a$ 不用于重新定义缩放系数。Softmax 权重 $\alpha_{t,i}^{(s)}$ 因而保持相同。用 $\boldsymbol u_t^{(s)}$ 表示第 $s$ 个头对 $\boldsymbol c_i$ 加权后得到的 $d_c$ 维向量，再计算 $\boldsymbol o_t^{(s)}$ 和 $\boldsymbol y_t$：

$$
\begin{aligned}
\boldsymbol u_t^{(s)}
&=\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol c_i
\in\mathbb R^{1\times d_c},\\
\boldsymbol o_t^{(s)}
&=\boldsymbol u_t^{(s)}W_v^{(s)}
=\sum_{i\le t}\alpha_{t,i}^{(s)}
(\boldsymbol c_iW_v^{(s)}),\\
\boldsymbol y_t
&=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O.
\end{aligned}
$$

运行时按 $\boldsymbol c_t'W_{qc}^{(s)}$、$\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$ 两步统计；$\boldsymbol u_t^{(s)}W_v^{(s)}$ 与 $[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O$ 也按两步统计。权重保持分别存储，不预先相乘为新的大矩阵。

### 5.2 单个词元位置：哪些投影换了执行位置

固定一个头，并将共享路径计一次。$C=d(d_c'+d_c)$、$G=d_c'+d_c$ 分别沿用第 3.2 节的降维投影权重元素数与 RMSNorm 权重元素数；$f_{\mathrm{shared}}$、$f_{\mathrm{head}}$ 沿用第 3.4.1 节，分别表示每个词元位置的多头共享成本与每头独立的固定成本。下表的权重读取量假设每个元素读一次，激活读写在第 5.5 节统一累计。

| 公式项 | 计算量 | 权重读取量 | 频率 |
|---|---:|---:|---|
| $\boldsymbol x_tW_c'$、$\boldsymbol x_tW_c$ | $2C$ | $bC$ | 每个词元一次，多头共享 |
| $\mathrm{RMSNorm}(\boldsymbol z_t^Q)$、$\mathrm{RMSNorm}(\boldsymbol z_t^{KV})$ | $4G+2$，另有 2 次 rsqrt | $bG$ | 每个词元一次，多头共享 |
| $\boldsymbol k_{t,R}=(\boldsymbol x_tW_{kr})\mathcal R_t$ | $2dd_r+3d_r$ | $bdd_r$ | 每个词元一次，多头共享 |
| $\boldsymbol q_t^{(s)}=[\boldsymbol c_t'W_{qc}^{(s)},(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t]$ | $2d_c'p+3d_r$ | $bd_c'p$ | 每个查询、每个头一次 |
| $\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}=(\boldsymbol c_t'W_{qc}^{(s)})W_{kc}^{(s)\top}$，仅右乘 $W_{kc}^{(s)\top}$ | $2d_kd_c$ | $bd_kd_c$ | 每个查询、每个头一次 |
| $\boldsymbol o_t^{(s)}=\boldsymbol u_t^{(s)}W_v^{(s)}$ | $2d_cd_v$ | $bd_cd_v$ | 注意力完成后，每个查询、每个头一次 |
| $\boldsymbol o_t^{(s)}W_O^{(s)}$ | $2d_vd$ | $bd_vd$ | 每个查询、每个头一次 |

前三行合计为 $f_{\mathrm{shared}}$。$\boldsymbol q_t^{(s)}$ 一行同时包含 $2d_c'd_k$、$2d_c'd_r$ 的矩阵乘法与 $3d_r$ 的旋转；下一行复用已生成的 $\boldsymbol q_{t,C}^{(s)}$，不重复计算 $\boldsymbol c_t'W_{qc}^{(s)}$。

在无前缀缓存的预填充中，查询位置数与新 K/V 位置数相等。$\boldsymbol c_iW_{kc}^{(s)}$ 与 $\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$ 的单位置成本均为 $2d_cd_k$；$\boldsymbol c_iW_v^{(s)}$ 与 $\boldsymbol u_t^{(s)}W_v^{(s)}$ 的单位置成本均为 $2d_cd_v$。因此每头的固定成本仍为：

$$
f_{\mathrm{head}}
=2[d_c'p+d_kd_c+d_cd_v+d_vd]+3d_r.
$$

它与第 3.4.1 节相同，每个词元位置的固定成本仍是 $f_{\mathrm{shared}}+h f_{\mathrm{head}}$。吸收没有在这个等长场景中消除投影计算，而是改变其执行对象与中间结果。

### 5.3 从一对有效 Q–K 位置扩展到整条序列与批次

固定一个头，$\widehat{\boldsymbol q}_t^{(s)}$、$\widehat{\boldsymbol k}_i$ 为 $p_a=d_c+d_r$ 维，$\boldsymbol u_t^{(s)}=\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol c_i$ 为 $d_c$ 维。将每对有效 Q–K 位置的主项计算量记为 $F_{\mathrm{pair,abs}}$：下标 $\mathrm{pair}$ 表示一对满足 $i\le t$ 的查询—键位置，$\mathrm{abs}$ 是 absorbed（吸收形式）的缩写。计算量为：

$$
\begin{aligned}
F_{\mathrm{pair,abs}}
&=\underbrace{2(d_c+d_r)}_{\text{QK 内积}}
+\underbrace{1}_{\text{分数缩放}}
+\underbrace{2d_c}_{\text{潜在 V 加权累加}}\\
&=2(2d_c+d_r)+1.
\end{aligned}
$$

用 $F_{\mathrm{prefill,absorbed}}^{(B,h)}$ 表示整个吸收形式预填充的主项计算量；下标 $\mathrm{prefill,absorbed}$ 表示阶段与计算形式，上标 $(B,h)$ 表示统计时采用的批大小与头数。位置 $t$ 有 $t$ 对有效位置，因此 $B=h=1$ 时的合计为：

$$
\begin{aligned}
F_{\mathrm{prefill,absorbed}}^{(B=1,h=1)}
&\approx\sum_{t=1}^{S}
\left(f_{\mathrm{shared}}+f_{\mathrm{head}}+tF_{\mathrm{pair,abs}}\right)\\
&=S(f_{\mathrm{shared}}+f_{\mathrm{head}})
+\frac{S(S+1)}2F_{\mathrm{pair,abs}}.
\end{aligned}
$$

扩展到批次与多头：

$$
\boxed{
F_{\mathrm{prefill,absorbed}}^{(B,h)}
\approx BS(f_{\mathrm{shared}}+h f_{\mathrm{head}})
+Bh\frac{S(S+1)}2F_{\mathrm{pair,abs}}
}
$$

令 $F_{\mathrm{pair,exp}}=F_{\mathrm{pair}}=2(d_k+d_r+d_v)+1$ 表示第 3.3 节中展开形式每对有效 Q–K 位置的主项计算量，其中 $\mathrm{exp}$ 是 expanded（展开形式）的缩写。两种形式的固定成本相同，计算量之差来自注意力维度：

$$
F_{\mathrm{prefill,absorbed}}-F_{\mathrm{prefill,expanded}}
\approx Bh\frac{S(S+1)}2\cdot2(2d_c-d_k-d_v).
$$

当 $2d_c>d_k+d_v$ 时，吸收形式增加注意力主项计算量。Softmax、rsqrt、在线归一化与分块补齐等成本仍单列；首个位置的退化优化也不计入上述主项。

### 5.4 注意力的显存读取：查询复用与跨头复用分别计数

先看一个 K/V 位置。共享 K 为 $[\boldsymbol c_i,\boldsymbol k_{i,R}]$，V 直接引用其中的 $\boldsymbol c_i$。本节假设读入的 $\boldsymbol c_i$ 在片上同时用于 QK 与 V 聚合。将一个位置的数据完整读入一次的显存读取量记为 $D_{\mathrm{KV,load,abs}}$；下标 $\mathrm{KV}$ 将范围限定为 K/V 数据，$\mathrm{load}$ 表示对该位置的一次读取，$\mathrm{abs}$ 表示吸收形式。因此：

$$
D_{\mathrm{KV,load,abs}}=b(d_c+d_r)=bp_a.
$$

这里没有将同一个 $\boldsymbol c_i$ 作为 K、V 各从显存读取一次；若实现确实重复加载，应增加相应流量。

固定一条序列、一个头，令 $\rho_a$ 为每个共享 K/V 元素在整条序列的注意力计算中从显存读入的平均次数；下标 $a$ 与 $p_a$ 一致，表示吸收形式。$\widehat{\boldsymbol q}_t^{(s)}$ 读一次、$\boldsymbol u_t^{(s)}$ 写一次，将该注意力范围的总访存量记为 $D_{\mathrm{core,abs}}$；$\mathrm{core}$ 沿用第 2.2 节的注意力核心范围，不含投影与缓存构建。于是：

$$
D_{\mathrm{core,abs}}^{(B=1,h=1)}
\approx bS[(p_a+d_c)+\rho_a p_a].
$$

其中 $p_a+d_c=2d_c+d_r$ 来自 Q 读取和潜在输出写回，$\rho_a p_a$ 来自共享 K/V 读取。

扩展到多头时，不能从“所有头引用同一份 K/V”直接推出只读一次。设每组最多有 $g$ 个头在片上复用同一次 K/V 读取，其中 $g$ 为满足 $1\le g\le h$ 的整数；用 $n_g=\lceil h/g\rceil$ 表示头组数量，最后一组可以少于 $g$ 个头。每个头组采用相同的查询分块方式，组间不复用显存读取；$\rho_a$ 统计每个头组内部由查询分块产生的平均读取次数。

| 数据 | 多头、多批次的显存访存量 |
|---|---:|
| $\widehat{\boldsymbol q}_t^{(s)}$ 读取 | $bBS hp_a$ |
| $[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 读取，$\boldsymbol c_i$ 同时用于 V 聚合 | $bBS\rho_a n_g p_a$ |
| $\boldsymbol u_t^{(s)}$ 写回 | $bBS hd_c$ |

将注意力范围内的主项计算量除以 $D_{\mathrm{core,abs}}$，得到 $I_{\mathrm{core,abs}}$；它是吸收形式注意力核心的计算强度，softmax 等额外操作仍单列。整个批次的访存量与计算强度分别为：

$$
\boxed{
D_{\mathrm{core,abs}}
\approx bBS[h(2d_c+d_r)+\rho_a n_g(d_c+d_r)]
}
$$

$$
\boxed{
I_{\mathrm{core,abs}}
\approx
\frac{h(S+1)F_{\mathrm{pair,abs}}}
{2b[h(2d_c+d_r)+\rho_a n_g(d_c+d_r)]}
}
$$

当 $g=1$ 时，$n_g=h$，每个头仍独立读取共享缓存；当 $g=h$ 时，$n_g=1$，每个查询块的 K/V 读取可由所有头复用。批大小 $B$ 从比值中约去，不同序列的 K/V 仍分别读取。

将第 3.4.3 节展开形式的平均读取次数 $\rho$ 在本节记为 $\rho_e$，下标 $e$ 表示 expanded。若查询分块采用该节的假设，$\rho_a$ 可按同一公式估算，其中 $M$ 表示每个查询块的位置数。但两种形式的 Q 与聚合输出宽度不同，片上容量可能限制查询块大小及跨头复用规模，因此 $\rho_a$、$\rho_e$ 需要分别确定。

用 $D_{\mathrm{KV,abs}}$、$D_{\mathrm{KV,exp}}$ 分别表示整个批次在吸收、展开形式下的注意力 K/V 读取量，均不包含 Q 读取与聚合输出写回。在两种形式的查询复用相同，即 $\rho_a=\rho_e$，且 $h$ 可被 $g$ 整除时：

$$
\frac{D_{\mathrm{KV,abs}}}{D_{\mathrm{KV,exp}}}
=\frac{n_g(d_c+d_r)}{h(p+d_v)}
=\frac{d_c+d_r}{g(p+d_v)}.
$$

这个比值体现跨头复用的作用：只有实际共同使用一次显存读取，才会获得对应的流量收益。Q 读取、输出写回以及投影中间结果还需另外计入。

### 5.5 完整前向：逐项累计中间结果的显存读写

为与第 3.4.4 节的参考路径对照，本节按以下公式项执行，并保留阶段间的显存写回：

1. 读取 $\boldsymbol x_t$，计算并写出 $\boldsymbol c_t'=\mathrm{RMSNorm}(\boldsymbol x_tW_c')$、$\boldsymbol c_t=\mathrm{RMSNorm}(\boldsymbol x_tW_c)$ 与 $\boldsymbol k_{t,R}=(\boldsymbol x_tW_{kr})\mathcal R_t$；阶段内部临时结果留在片上。
2. 计算并写出 $\boldsymbol q_t^{(s)}=[\boldsymbol c_t'W_{qc}^{(s)},(\boldsymbol c_t'W_{qr}^{(s)})\mathcal R_t]\in\mathbb R^{1\times p}$。
3. 读取 $\boldsymbol q_t^{(s)}$，计算并写出 $\widehat{\boldsymbol q}_t^{(s)}=[\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top},\boldsymbol q_{t,R}^{(s)}]\in\mathbb R^{1\times p_a}$。
4. 读取 $\widehat{\boldsymbol q}_t^{(s)}$、$[\boldsymbol c_i,\boldsymbol k_{i,R}]$，计算注意力并写出 $\boldsymbol u_t^{(s)}=\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol c_i\in\mathbb R^{1\times d_c}$。
5. 读取 $\boldsymbol u_t^{(s)}$，计算并写出 $\boldsymbol o_t^{(s)}=\boldsymbol u_t^{(s)}W_v^{(s)}\in\mathbb R^{1\times d_v}$。
6. 读取 $[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]$，计算并写出 $\boldsymbol y_t=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O\in\mathbb R^{1\times d}$。

各投影的权重在 $N=BS$ 个位置间复用，按每个元素从显存读一次统计。沿用第 3.4.4 节的 $W_h$，表示所有 $h$ 个头及共享路径的矩阵投影权重元素总数，不含 RMSNorm 权重。该数量仍与展开形式相同：

$$
W_h=C+dd_r+h[d_c'p+d_cd_k+d_cd_v+d_vd],
$$

另有 $G$ 个 RMSNorm 权重。分数与 softmax 权重不写回显存，按以下边界累计激活流量：

| 数据 | 整个预填充的显存访存量 |
|---|---:|
| 矩阵投影与 RMSNorm 权重 | $b(W_h+G)$ |
| $\boldsymbol x_t$ 读取、$\boldsymbol y_t$ 写回 | $2bNd$ |
| $\boldsymbol c_t'$：写出、再读入 | $2bNd_c'$ |
| $[\boldsymbol c_t,\boldsymbol k_{t,R}]$ 写出 | $bNp_a$ |
| $\boldsymbol q_t^{(s)}=[\boldsymbol q_{t,C}^{(s)},\boldsymbol q_{t,R}^{(s)}]$：写出、再读入 | $2bNhp$ |
| $\widehat{\boldsymbol q}_t^{(s)}=[\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top},\boldsymbol q_{t,R}^{(s)}]$：写出、再读入 | $2bNhp_a$ |
| $[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 供注意力读取 | $bN\rho_a n_g p_a$ |
| $\boldsymbol u_t^{(s)}$：写出、再读入 | $2bNhd_c$ |
| $\boldsymbol o_t^{(s)}=\boldsymbol u_t^{(s)}W_v^{(s)}$：写出、再读入 | $2bNhd_v$ |

用 $D_{\mathrm{prefill,absorbed}}^{(B,h)}$ 表示上表全部数据的显存读写量，$I_{\mathrm{prefill,absorbed}}^{(B,h)}$ 表示完整前向主项计算量与该读写量之比；下标与上标沿用第 5.3 节的阶段、形式和统计范围约定。因此：

$$
\boxed{
D_{\mathrm{prefill,absorbed}}^{(B,h)}
\approx b\left\{
W_h+G+BS[
2d+2d_c'+p_a
+2h(p+p_a+d_c+d_v)
+\rho_a n_g p_a]
\right\}
}
$$

$$
\boxed{
I_{\mathrm{prefill,absorbed}}^{(B,h)}
\approx
\frac{
BS(f_{\mathrm{shared}}+h f_{\mathrm{head}})
+Bh\,S(S+1)F_{\mathrm{pair,abs}}/2
}{
b\left\{
W_h+G+BS[
2d+2d_c'+p_a+2h(p+p_a+d_c+d_v)+\rho_a n_g p_a]
\right\}
}
}
$$

该模型已经包含共享 K/V 的写出。若该缓冲区直接作为持久 KV cache，这次写出也完成新位置的缓存数据写入；若再复制到独立缓存，需要增加复制的读取和写入，不能把缓存构建统一视为零成本或重复计数。

上述显存边界是明确的参考选择。若 $\boldsymbol q_t^{(s)}\to\widehat{\boldsymbol q}_t^{(s)}$ 在片上衔接，可省去 $\boldsymbol q_t^{(s)}$ 的 $2bNhp$；若 $\boldsymbol u_t^{(s)}W_v^{(s)}$ 的结果直接在片上参与 $[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O$，可省去 $\boldsymbol o_t^{(s)}$ 的 $2bNhd_v$。取消写回不改变这里按两步矩阵乘法统计的计算量。实际权重重复加载、压缩表示反复读取或额外布局搬运则需要追加流量。

### 5.6 与展开形式对照：降低 K/V 读取不等于降低全部成本

沿用第 3.4.5 节的数值：$d=1024$、$d_c'=d_c=256$、$d_k=d_v=128$、$d_r=64$、$b=2$，以及 $B=1$、$h=8$、$S=128$。于是：

$$
p=192,\qquad p_a=320,\qquad
F_{\mathrm{pair,exp}}=641,\qquad F_{\mathrm{pair,abs}}=1153.
$$

为单独观察跨头复用，暂令两种形式的查询块大小均为 $M=32$，故 $\rho_e=\rho_a=2.5$。展开形式读取 $\boldsymbol q_t^{(s)}$ 并写回 $\boldsymbol o_t^{(s)}$，吸收形式读取 $\widehat{\boldsymbol q}_t^{(s)}$ 并写回 $\boldsymbol u_t^{(s)}$；统计范围到这两种聚合结果写回为止，softmax 额外操作单列：

| 形式 | 注意力主项计算量（FLOPs） | Q 读取与聚合输出写回（Bytes） | K/V 读取（Bytes） | 核心总访存量（Bytes） | 核心计算强度（FLOPs/Byte） |
|---|---:|---:|---:|---:|---:|
| 展开 | 42,336,768 | 655,360 | 1,638,400 | 2,293,760 | 18.46 |
| 吸收，$g=1$，各头独立读取 | 76,153,344 | 1,179,648 | 1,638,400 | 2,818,048 | 27.02 |
| 吸收，$g=8$，全部头共用读取 | 76,153,344 | 1,179,648 | 204,800 | 1,384,448 | 55.01 |

本例中，$[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 的宽度 $d_c+d_r=320$ 恰好等于单头 $\boldsymbol k_i^{(s)}$、$\boldsymbol v_i^{(s)}$ 的宽度合计 $p+d_v=320$。因此 $g=1$ 时 K/V 读取量相同；$g=8$ 时，K/V 读取量降至展开形式的 $1/8$。$\widehat{\boldsymbol q}_t^{(s)}$、$\boldsymbol u_t^{(s)}$ 的宽度合计为 $p_a+d_c=576$，大于 $\boldsymbol q_t^{(s)}$、$\boldsymbol o_t^{(s)}$ 的 $p+d_v=320$，对应的计算量与激活读写增加。

进一步加入投影、归一化与中间张量流量，展开形式使用第 3.4.4 节边界，吸收形式使用第 5.5 节边界：

| 形式 | 完整前向主项计算量（FLOPs） | 完整前向访存量（Bytes） | 合计计算强度（FLOPs/Byte） |
|---|---:|---:|---:|
| 展开 | 697,131,776 | 9,536,512 | 73.10 |
| 吸收，$g=1$ | 730,948,352 | 11,158,528 | 65.51 |
| 吸收，$g=8$ | 730,948,352 | 9,724,928 | 75.16 |

在全部头共用读取的参照下，注意力 K/V 读取大幅下降，但完整前向的计算量仍增加约 4.85%，访存量增加约 1.98%。合计计算强度提高，不代表执行时间下降：它也可能来自计算量增加。需要结合各阶段吞吐量、显存带宽与实际融合边界判断。

对于长序列稠密预填充，吸收形式增加的注意力计算按有效 Q–K 配对数增长，展开形式通常更有利于控制计算量；短查询、长缓存等场景的成本平衡需要单独分析。这里的结果用于建立对照基线，实际可实现的 $M$、$g$、平均读取次数与融合收益仍需验证。

## 6. 解码 × 吸收

本节沿用第 5 节的运行时计算顺序：先计算 $\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$，在压缩缓存上完成注意力，再计算 $\boldsymbol u_t^{(s)}W_v^{(s)}$ 和输出投影。各投影权重分别存储，不预先合并为新的矩阵。

标准解码每条序列只增加一个查询位置，即 $Q_{\mathrm{len}}=1$。可见键值长度 $L$ **包含当前新位置**，因此已有缓存包含 $L-1$ 个位置。历史位置的 $\boldsymbol c_i$、$\boldsymbol k_{i,R}$ 已生成，本步只生成当前位置的表示；不重新执行历史位置的降维投影、RMSNorm 或 RoPE。

与第 5 节的无前缀预填充相比，这里的关键变化是查询长度与缓存长度不同。不能将第 5 节公式中的 $S$ 直接替换为 $L$：每个头本步只有 $L$ 对有效 Q–K 位置，而不是 $L(L+1)/2$ 对。

### 6.1 吸收形式的前向公式与张量维度

固定一条序列和一个头，沿用第 5.1 节的张量与维度。为区分查询生成与后续内积，将已生成的 $\widehat{\boldsymbol q}_t^{(s)}$ 的内容分支、位置分支分别记为 $\widehat{\boldsymbol q}_{t,C}^{(s)}$、$\widehat{\boldsymbol q}_{t,R}^{(s)}$：

$$
\begin{aligned}
\widehat{\boldsymbol q}_{t,C}^{(s)}
&=\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}
\in\mathbb R^{1\times d_c},\\
\widehat{\boldsymbol q}_{t,R}^{(s)}
&=\boldsymbol q_{t,R}^{(s)}
\in\mathbb R^{1\times d_r},\\
\widehat{\boldsymbol q}_t^{(s)}
&=[\widehat{\boldsymbol q}_{t,C}^{(s)},\ \widehat{\boldsymbol q}_{t,R}^{(s)}]
\in\mathbb R^{1\times p_a},\\
\widehat K_{\le t}
&\in\mathbb R^{L\times p_a},\\
\widehat V_{\le t}
&\in\mathbb R^{L\times d_c},\\
p_a&=d_c+d_r.
\end{aligned}
$$

$\widehat K_{\le t}$ 的第 $i$ 行为 $[\boldsymbol c_i,\boldsymbol k_{i,R}]$，$\widehat V_{\le t}$ 的第 $i$ 行直接引用同一份 $\boldsymbol c_i$，不另存一份潜在 V。计算过程为：

$$
\begin{aligned}
s_{t,i}^{(s)}
&=(\widehat{\boldsymbol q}_t^{(s)}\widehat{\boldsymbol k}_i^\top)\sigma\\
&=\left[\widehat{\boldsymbol q}_{t,C}^{(s)}\boldsymbol c_i^\top
+\widehat{\boldsymbol q}_{t,R}^{(s)}\boldsymbol k_{i,R}^\top\right]\sigma,
\qquad \sigma=1/\sqrt p,\\
\alpha_{t,i}^{(s)}&=\operatorname{softmax}_{i\le t}(s_{t,i}^{(s)}),\\
\boldsymbol u_t^{(s)}&=\sum_{i\le t}\alpha_{t,i}^{(s)}\boldsymbol c_i
\in\mathbb R^{1\times d_c},\\
\boldsymbol o_t^{(s)}&=\boldsymbol u_t^{(s)}W_v^{(s)}
\in\mathbb R^{1\times d_v},\\
\boldsymbol y_t&=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O.
\end{aligned}
$$

分数缩放仍使用 $p=d_k+d_r$。$\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$、$\boldsymbol u_t^{(s)}W_v^{(s)}$ 和输出投影均按每个查询、每个头一次计数；只有内积和加权累加按 $L$ 个位置累计。

### 6.2 一个历史位置的 Attention 计算分析

#### 6.2.1 单头、单历史位置：计算量与显存读取

先假设 $\widehat{\boldsymbol q}_t^{(s)}$ 已生成，查询与输出累加器保留在片上，分数与权重不写回显存。下表的 QK 运算是已生成查询与缓存向量的内积；$\widehat{\boldsymbol q}_{t,C}^{(s)}=\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$ 的生成成本不在表中，而是在第 6.4 节按每个查询、每个头一次计入。读入的 $\boldsymbol c_i$ 在片上同时用于 QK 内积和 V 聚合；不将它作为两份独立数据重复读取。

| 公式项 | 主项计算量 | 显存读取 | 显存写入 |
|---|---:|---:|---:|
| $\widehat{\boldsymbol q}_{t,C}^{(s)}\boldsymbol c_i^\top$ | $2d_c$ FLOPs | $\boldsymbol c_i$：$bd_c$ Bytes | 0 |
| $\widehat{\boldsymbol q}_{t,R}^{(s)}\boldsymbol k_{i,R}^\top$ | $2d_r$ FLOPs | $\boldsymbol k_{i,R}$：$bd_r$ Bytes | 0 |
| $s_{t,i}^{(s)}$ 中的分数缩放，仅乘 $\sigma$ | 1 FLOP | 0 | 0 |
| $\alpha_{t,i}^{(s)}=\operatorname{softmax}_{i\le t}(s_{t,i}^{(s)})$ | 按整组 $L$ 个分数单列，见第 2.4.5 节 | 0 | 0 |
| $\boldsymbol u_t^{(s)}\mathrel{+}=\alpha_{t,i}^{(s)}\boldsymbol c_i$ | $2d_c$ FLOPs | 复用已读入的 $\boldsymbol c_i$，0 | 0 |
| **合计，softmax 等额外操作单列** | **$F_{\mathrm{pair,abs}}=2(2d_c+d_r)+1$ FLOPs** | **$bp_a=b(d_c+d_r)$ Bytes** | **0** |

这里将两个分支内积的主项记为 $2(d_c+d_r)$；分支求和已包含在这一主项估算中，不另加一次。Softmax 需要对全部可见位置归一化，不能为一个历史位置独立计算权重。

单个历史位置的数据量为 $b(d_c+d_r)$，而不是 $b(2d_c+d_r)$。后者会将 $\boldsymbol c_i$ 作为 K、V 重复计数。若实际内核分别加载这两次使用的数据，应按实际读取次数补充流量。

#### 6.2.2 累计到 $L$ 个可见位置：单头 Attention 成本

固定一个头，将上表累计到 $L$ 个可见位置，并计入 $\widehat{\boldsymbol q}_t^{(s)}$ 读取一次、$\boldsymbol u_t^{(s)}$ 写回一次：

$$
\begin{aligned}
F_{\mathrm{core,abs}}^{(B=1,h=1)}
&\approx L F_{\mathrm{pair,abs}},\\
D_{\mathrm{core,abs}}^{(B=1,h=1)}
&\approx b[Lp_a+p_a+d_c],\\
I_{\mathrm{core,abs}}^{(B=1,h=1)}
&\approx\frac{L F_{\mathrm{pair,abs}}}{b[Lp_a+p_a+d_c]}.
\end{aligned}
$$

$F_{\mathrm{core,abs}}$、$D_{\mathrm{core,abs}}$、$I_{\mathrm{core,abs}}$ 沿用第 5 节的注意力核心范围，本节均指一步解码，不含 Q 生成、Q 侧矩阵变换、后置 V 投影或缓存写入；这些操作在第 6.4 节计入完整前向。

### 6.3 一个历史位置的 Attention：扩展到多头与 batch

本节以第 6.2 节的单头、单历史位置成本为起点，先统计一个历史位置在多个头中的计算与读取，再累计各序列的可见位置，得到整个批次一步解码的 Attention 成本。查询读取与聚合输出写回仍按每个查询、每个头一次计数，不按历史位置重复累计。

#### 6.3.1 单个历史位置：多头计算与 KV 读取

沿用第 5.4 节的 $g$ 和 $n_g$：每组最多 $g$ 个头在片上共用一次 K/V 读取，$n_g=\lceil h/g\rceil$ 为头组数量，$1\le g\le h$。最后一组可以少于 $g$ 个头。这里 $g$ 表示实际执行中的跨头复用，不是仅由缓存的共享存储布局决定的常数。

本节假设每个头组对每个可见位置读一次，不同组之间不复用 HBM 读取；同一组内的 $\boldsymbol c_i$ 同时用于各头的内积和加权累加。跨头复用时，K/V 数据不必全部驻留片上，可以逐块读入并由组内各头使用。每个头仍有独立的 Q、分数、softmax 和 $\boldsymbol u_t^{(s)}$，因此计算量不因共享读取而减少。

对一条序列的一个可见历史位置，成本从第 6.2.1 节扩展为：

| 项目 | 单头 | 全部 $h$ 个头 |
|---|---:|---:|
| 该位置对应的主项计算量 | $F_{\mathrm{pair,abs}}$ | $hF_{\mathrm{pair,abs}}$ |
| $[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 读取，$\boldsymbol c_i$ 同时用于 V 聚合 | $bp_a$ | $bn_gp_a$ |
| 该位置的累加结果写回 | 0，累加器保留在片上 | 0，累加器保留在片上 |

计算量按 $h$ 个独立头累计，KV 读取按 $n_g$ 个头组累计。扩展到 batch 时，对不同序列的有效位置分别累加，不能假设不同序列共享同一次 KV 读取。

#### 6.3.2 累计各序列的可见位置：一步解码的 Attention 成本

标准解码只有一个查询位置，无法利用多个查询位置复用同一份 K/V。按本节的单遍读取假设，第 5.4 节“每个头组内的平均读取次数”对应 $\rho_a=1$。增加批大小也不会改变这一点：不同序列的 K/V 不同，本节不假设共享前缀。

设第 $r$ 条序列的可见长度为 $L_r$，所有序列均有相同的头数与头组配置。一步解码的注意力成本为：

| 项目 | 多头、多批次合计 |
|---|---:|
| 有效 Q–K 位置对应的主项计算量 | $hF_{\mathrm{pair,abs}}\sum_{r=1}^{B}L_r$ |
| $\widehat{\boldsymbol q}_t^{(s)}$ 读取 | $bBhp_a$ |
| $[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 读取，$\boldsymbol c_i$ 同时用于 V 聚合 | $bn_gp_a\sum_{r=1}^{B}L_r$ |
| $\boldsymbol u_t^{(s)}$ 写回 | $bBhd_c$ |

于是：

$$
\boxed{
D_{\mathrm{core,abs}}^{(B,h)}
\approx b\left[Bh(p_a+d_c)+n_gp_a\sum_{r=1}^{B}L_r\right]
}
$$

$$
\boxed{
I_{\mathrm{core,abs}}^{(B,h)}
\approx\frac{hF_{\mathrm{pair,abs}}\sum_{r=1}^{B}L_r}
{b[Bh(p_a+d_c)+n_gp_a\sum_{r=1}^{B}L_r]}
}
$$

当各序列长度相同，均为 $L$ 时，$B$ 从比值中约去：

$$
I_{\mathrm{core,abs}}^{(B,h)}
\approx\frac{hL F_{\mathrm{pair,abs}}}
{b[h(2d_c+d_r)+n_gL(d_c+d_r)]}
\xrightarrow{L\to\infty}
\frac{h}{n_g}\frac{F_{\mathrm{pair,abs}}}{b(d_c+d_r)}.
$$

若 $h$ 可被 $g$ 整除，则 $h/n_g=g$。同一次读取可供 $g$ 个头使用，因而能支撑约 $g$ 倍的注意力计算；$g=1$ 时各头独立读，$g=h$ 时所有头共用一次读。对于不能整除的情况，应保留 $h/n_g$，不能直接用 $g$ 替换。

这里假设 Q 各读一次、输出各写一次。实际内核若反复加载 K/V、将同一份 $\boldsymbol c_i$ 分两次读取，或通过分段并行产生中间输出与合并，应增加对应的读写和计算，见第 7 节。

### 6.4 一步解码的完整前向：计算量与显存读写量

在第 6.3 节的注意力计算之前，本步生成 $\boldsymbol c_t'$、$\boldsymbol c_t$、$\boldsymbol k_{t,R}$、$\boldsymbol q_t^{(s)}$ 和 $\widehat{\boldsymbol q}_t^{(s)}$；注意力完成后生成 $\boldsymbol o_t^{(s)}$ 和 $\boldsymbol y_t$。各操作的计算量与权重读取量沿用第 5.2 节表格，本步只有 $N=B$ 个新位置。

特别是，$\boldsymbol q_{t,C}^{(s)}W_{kc}^{(s)\top}$ 每个头执行一次，成本为 $2d_kd_c$；$\boldsymbol u_t^{(s)}W_v^{(s)}$ 每个头执行一次，成本为 $2d_cd_v$。展开形式则为当前新位置执行 $\boldsymbol c_tW_{kc}^{(s)}$ 和 $\boldsymbol c_tW_v^{(s)}$，成本分别相同。两种形式均复用历史缓存，不重复生成历史 K/V。

因此，共享固定成本仍为 $f_{\mathrm{shared}}$，每头固定成本仍为 $f_{\mathrm{head}}$：

$$
\begin{aligned}
f_{\mathrm{shared}}&=2[C+dd_r]+4G+2+3d_r,\\
f_{\mathrm{head}}&=2[d_c'p+d_kd_c+d_cd_v+d_vd]+3d_r.
\end{aligned}
$$

用 $F_{\mathrm{decode,absorbed}}^{(B,h)}$ 表示一步解码从输入 $\boldsymbol x_t$ 到输出 $\boldsymbol y_t$ 的主项计算量；下标分别表示解码阶段和吸收形式，上标表示批次与多头范围：

$$
\boxed{
F_{\mathrm{decode,absorbed}}^{(B,h)}
\approx B(f_{\mathrm{shared}}+h f_{\mathrm{head}})
+hF_{\mathrm{pair,abs}}\sum_{r=1}^{B}L_r
}
$$

每条序列另有 2 次 rsqrt，每头对 $L_r$ 个分数执行 softmax；在线归一化的额外操作仍按第 7.1 节单列。

为了统计显存流量，采用第 5.5 节相同的阶段边界：$\boldsymbol c_t'$、$\boldsymbol q_t^{(s)}$、$\widehat{\boldsymbol q}_t^{(s)}$、$\boldsymbol u_t^{(s)}$、$\boldsymbol o_t^{(s)}$ 在相邻阶段间各写出、读入一次；前处理内部临时结果留在片上。当前 $[\boldsymbol c_t,\boldsymbol k_{t,R}]$ **直接写入持久 KV Cache**，不再复制一份。假设权重初始未驻留片上，每个元素在本步读一次并在 $B$ 个新位置间复用；共享激活的读取也在各头间复用。

沿用第 5.5 节的权重元素数 $W_h$ 与 $G$：

$$
W_h=C+dd_r+h[d_c'p+d_cd_k+d_cd_v+d_vd],
\qquad G=d_c'+d_c.
$$

| 数据 | 一步解码的显存读写量 |
|---|---:|
| 矩阵投影与 RMSNorm 权重 | $b(W_h+G)$ |
| $\boldsymbol x_t$ 读取、$\boldsymbol y_t$ 写回 | $2bBd$ |
| $\boldsymbol c_t'$：写出、再读入 | $2bBd_c'$ |
| 当前 $[\boldsymbol c_t,\boldsymbol k_{t,R}]$ 写入 KV Cache | $bBp_a$ |
| $\boldsymbol q_t^{(s)}$：写出、再读入 | $2bBhp$ |
| $\widehat{\boldsymbol q}_t^{(s)}$：写出、再读入 | $2bBhp_a$ |
| 全部可见 $[\boldsymbol c_i,\boldsymbol k_{i,R}]$ 供注意力读取 | $bn_gp_a\sum_{r=1}^{B}L_r$ |
| $\boldsymbol u_t^{(s)}$：写出、再读入 | $2bBhd_c$ |
| $\boldsymbol o_t^{(s)}=\boldsymbol u_t^{(s)}W_v^{(s)}$：写出、再读入 | $2bBhd_v$ |

令 $D_{\mathrm{decode,absorbed}}^{(B,h)}$ 为上表总显存读写量，$I_{\mathrm{decode,absorbed}}^{(B,h)}$ 为该范围内的主项计算强度，则：

$$
\boxed{
D_{\mathrm{decode,absorbed}}^{(B,h)}
\approx b\left\{
W_h+G+B[2d+2d_c'+p_a+2h(p+p_a+d_c+d_v)]
+n_gp_a\sum_{r=1}^{B}L_r
\right\}
}
$$

$$
I_{\mathrm{decode,absorbed}}^{(B,h)}
\approx\frac{F_{\mathrm{decode,absorbed}}^{(B,h)}}
{D_{\mathrm{decode,absorbed}}^{(B,h)}}.
$$

$L_r$ 包含当前新位置，因此上表既计当前缓存写入，也计注意力从缓存读入当前数据：这是一次写和一次读，属于两次不同的数据搬运。若当前数据直接留在片上供所有头组使用，可减去 $bBn_gp_a$ 的当前数据读取；若另有缓存复制，则追加复制的读取和写入。

融合可以省去对应的中间往返，例如 $\boldsymbol q_t^{(s)}$ 的 $2bBhp$、$\boldsymbol o_t^{(s)}$ 的 $2bBhd_v$；这不改变上述分步矩阵乘法的 FLOPs。该模型是明确边界下的参考值，不表示所有中间向量能同时驻留片上，也不保证实际权重只加载一次。

### 6.5 吸收与展开的成本比较：公式推导与数值例子

先比较相同有效 Q–K 位置范围内的计算、缓存容量与读取量，再统一完整前向的显存读写边界，最后通过同一组维度的数值例子观察各项成本如何变化。

#### 6.5.1 Attention 计算、KV Cache 容量与读取量

为与第 4 节一致，展开形式仍假设持久缓存存储各头完整的 $\boldsymbol k_i^{(s)}$ 和 $\boldsymbol v_i^{(s)}$，其中 $\boldsymbol k_{i,R}$ 复制到各头；吸收形式只存共享 $[\boldsymbol c_i,\boldsymbol k_{i,R}]$。如果展开计算使用压缩持久缓存并临时展开历史 K/V，必须另计历史展开计算及中间流量，不能直接使用本节对照。

先比较相同有效位置范围内的注意力计算与 K/V 读取：

| 项目 | 展开形式 | 吸收形式 |
|---|---:|---:|
| 每头、每对有效 Q–K 位置的主项计算量 | $F_{\mathrm{pair,exp}}=2(p+d_v)+1$ | $F_{\mathrm{pair,abs}}=2(2d_c+d_r)+1$ |
| 当前查询与聚合输出的核心读写 | $bBh(p+d_v)$ | $bBh(p_a+d_c)$ |
| 全部可见 K/V 的注意力读取 | $bh(p+d_v)\sum_rL_r$ | $bn_gp_a\sum_rL_r$ |
| 全部可见位置的 KV Cache 数据容量 | $bh(p+d_v)\sum_rL_r$ | $bp_a\sum_rL_r$ |
| 当前新位置的 KV Cache 写入 | $bBh(p+d_v)$ | $bBp_a$ |

容量按每份数据存一份计数，不含分页空闲空间与元数据。吸收形式的容量不随 $n_g$ 增长，但运行时读取可能由 $n_g$ 个头组分别发生，因此容量比与读取比不同：

$$
\frac{\text{吸收缓存容量}}{\text{展开缓存容量}}
=\frac{p_a}{h(p+d_v)},
\qquad
\frac{D_{\mathrm{KV,abs}}}{D_{\mathrm{KV,exp}}}
=\frac{n_gp_a}{h(p+d_v)}.
$$

在本节的单遍读取模型下，降低注意力 K/V 读取量需要 $n_gp_a<h(p+d_v)$；当 $h$ 可被 $g$ 整除时，等价于 $p_a<g(p+d_v)$。即使缓存容量明显降低，若跨头读取复用不足，也可能没有对应的流量收益。

#### 6.5.2 完整前向：计算量差异与统一的显存读写边界

两种形式当前词元的固定计算量相同，完整前向的主项计算量之差为：

$$
F_{\mathrm{decode,absorbed}}^{(B,h)}
-F_{\mathrm{decode,expanded}}^{(B,h)}
\approx 2h(2d_c-d_k-d_v)\sum_{r=1}^{B}L_r.
$$

当 $2d_c>d_k+d_v$ 时，吸收形式增加每个有效 Q–K 位置的计算量。计算增加来自更宽的内积与加权累加；缓存读取下降来自压缩存储及实际的跨头复用。这两种变化必须分别计数。

完整前向对照也需要一致的显存边界。令 $D_{\mathrm{decode,expanded}}^{(B,h)}$ 在本节表示：采用第 3.4.4 节的阶段间写回方式，每步只生成 $B$ 个新位置，并将新 K/V 的写出直接作为持久缓存写入。展开形式每个可见位置、每个头读一次。因此：

$$
\begin{aligned}
F_{\mathrm{decode,expanded}}^{(B,h)}
&\approx B(f_{\mathrm{shared}}+h f_{\mathrm{head}})
+hF_{\mathrm{pair,exp}}\sum_{r=1}^{B}L_r,\\
D_{\mathrm{decode,expanded}}^{(B,h)}
&\approx b\left\{
W_h+G+B[2d+2(G+d_r)+3h(p+d_v)]
+h(p+d_v)\sum_{r=1}^{B}L_r
\right\}.
\end{aligned}
$$

其中 $3h(p+d_v)$ 来自 Q 与输出的各一次写出和读入，共 $2h(p+d_v)$，加上当前 K/V 写入的 $h(p+d_v)$。这与第 4.5 节“中间激活留片上、未计缓存写入”的理想边界不同；第 6.5.4 节的完整前向数值对照统一使用这里的边界，不将两种口径混合比较。

#### 6.5.3 数值例子：Attention 计算与 KV 读取

沿用第 5.6 节的维度：$d=1024$、$d_c'=d_c=256$、$d_k=d_v=128$、$d_r=64$、$b=2$、$h=8$。改为每条序列一步解码、可见长度 $L=8192$，包含当前新位置。于是：

$$
\begin{aligned}
p&=192,&p_a&=320,\\
F_{\mathrm{pair,exp}}&=641,&F_{\mathrm{pair,abs}}&=1153,\\
f_{\mathrm{shared}}+h f_{\mathrm{head}}&=5{,}115{,}586,&
W_h+G&=2{,}556{,}416.
\end{aligned}
$$

先固定 $B=1$，仅统计注意力核心。下表计算量单位为 FLOPs，访存量单位为 Bytes，计算强度单位为 FLOPs/Byte；softmax 等额外操作单列：

| 形式 | 主项计算量 | Q 读取与聚合输出写回 | K/V 读取 | 核心总访存量 | 核心计算强度 |
|---|---:|---:|---:|---:|---:|
| 展开 | 42,008,576 | 5,120 | 41,943,040 | 41,948,160 | 1.0014 |
| 吸收，$g=1$，$n_g=8$ | 75,563,008 | 9,216 | 41,943,040 | 41,952,256 | 1.8012 |
| 吸收，$g=2$，$n_g=4$ | 75,563,008 | 9,216 | 20,971,520 | 20,980,736 | 3.6015 |
| 吸收，$g=8$，$n_g=1$ | 75,563,008 | 9,216 | 5,242,880 | 5,252,096 | 14.3872 |

本例 $p_a=p+d_v=320$。$g=1$ 时，吸收形式虽然增加计算，K/V 读取量却没有降低；计算强度提高不能单独证明优化有效。$g=8$ 时，压缩 K/V 一次读取由 8 个头共用，K/V 读取从 $40\ \mathrm{MiB}$ 降至 $5\ \mathrm{MiB}$。

两种吸收执行方式的缓存数据容量均为 $5\ \mathrm{MiB}$，区别在于同一份数据实际读了几遍。当前词元的缓存写入量从展开形式的 $5{,}120$ Bytes 降至吸收形式的 $640$ Bytes；这部分与注意力读取分开累计。

#### 6.5.4 数值例子：完整前向与 batch 扩展

在第 6.5.3 节的相同维度下，再加入全部投影、归一化、RoPE、中间张量读写和当前缓存写入，采用第 6.4、6.5.2 节的显存边界：

| $B$ | 形式 | 完整前向主项计算量（FLOPs） | 完整前向访存量（Bytes） | 合计计算强度（FLOPs/Byte） |
|---:|---|---:|---:|---:|
| 1 | 展开 | 47,124,162 | 47,077,632 | 1.0010 |
| 1 | 吸收，$g=1$ | 80,678,594 | 47,090,304 | 1.7133 |
| 1 | 吸收，$g=8$ | 80,678,594 | 10,390,144 | 7.7649 |
| 4 | 展开 | 188,496,648 | 172,972,032 | 1.0898 |
| 4 | 吸收，$g=8$ | 322,714,376 | 26,222,080 | 12.3070 |

$B=1,g=8$ 时，完整前向的主项计算量增加约 71.20%，显存访存量减少约 77.93%。K/V 读取降至原来的 $1/8$，但完整前向还要读取相同的投影与 RMSNorm 权重，并处理更宽的中间向量，因此总访存量不会恰好下降 8 倍。

固定 $g=8$，从 $B=1$ 增至 $B=4$，注意力核心的计算强度仍为 $14.3872$ FLOPs/Byte：不同序列没有共用 K/V。完整前向的计算强度提高，是因为本步权重在更多新位置间复用。这里仍依赖“每份权重在整个批次中从显存读一次”的参考假设。

### 6.6 性能判断：增加计算为何可能缩短解码时间

对注意力核心或具体投影，分别使用第 2.5 节的屋顶线关系 $T\gtrsim\max(F/P,D/B_{\mathrm{HBM}})$。如果展开形式主要受缓存读取带宽限制，吸收形式又能实现跨头读取复用，则更少的 HBM 流量可能抵消增加的计算。需要检验增加计算后的 $F/P$ 是否仍足够小，以及实际的 $D/B_{\mathrm{HBM}}$ 降低了多少。

例如，第 6.5.3 节全部头共用读取时，注意力主项计算量约增加至 1.80 倍，核心访存量约降至原来的 0.125 倍。这为降低带宽受限的解码延迟提供了条件，但不能直接推导出 8 倍加速：计算吞吐、Q 变换、后置投影、softmax、分段合并和启动开销仍可能限制速度。完整前向包含依次执行的多个算子，合计 $I$ 也不能代替逐算子判断。

该收益与 prefill 的区别是：标准 decode 每条序列的查询长度固定为 1，单头难以通过查询分块复用历史 K/V，因此跨头读取复用尤为重要。长序列 prefill 已可让多个查询共用 K/V 读取，展开形式也能获得较高的计算强度；吸收形式增加的计算更可能成为主要成本。

本节还限定了两点。第一，短缓存下投影权重和固定成本可能占主导，压缩历史读取带来的收益较小。第二，若预先合并 $W_{qc}^{(s)}W_{kc}^{(s)\top}$ 或 $W_v^{(s)}W_O^{(s)}$，矩阵尺寸、权重元素数与运行时 FLOPs 都会改变，不能沿用本节的 $W_h$ 和 $f_{\mathrm{head}}$。本节的结论针对分别存储权重、运行时顺序执行矩阵乘法的路径。

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
| 缓存写入与复制 | 第 6.4、6.5 节已计入当前词元的直接缓存写入，不再追加同一次写入；其他边界需明确是否包含。展开缓存每个新位置写入 $bh(p+d_v)$ Bytes，共享 $[\boldsymbol c_t,\boldsymbol k_{t,R}]$ 写入 $bp_a$ Bytes；额外复制、独立存储 $\boldsymbol k_{t,R}$ 的布局分别统计 |
| 压缩缓存与运行时展开 | 按实际展开的历史位置计入 $\boldsymbol c_iW_{kc}^{(s)}$、$\boldsymbol c_iW_v^{(s)}$ 的计算和中间读写 |
| 中间结果写入显存 | $s_{t,i}^{(s)}$、$\alpha_{t,i}^{(s)}$、$\boldsymbol c_t'$、$\boldsymbol c_i$、$\boldsymbol q_t^{(s)}$、$\boldsymbol k_i^{(s)}$、$\boldsymbol v_i^{(s)}$ 等在独立算子间可能产生额外显存读写 |
| 物理维度补齐 | 数学维度不一定等于内核实际计算的宽度；参考分块融合注意力算法（FlashAttention）的路径可能将 V 补齐（padding）至 Q/K 维度 |
| 布局变换 | 形状重排或视图可只修改元数据；连续化、实际拼接和广播复制可能搬运数据 |
| 分段并行与结果合并 | 按键值长度分段并行（split-K）时，计入局部输出和归一化统计量的读写及合并运算 |
| RoPE 系数 | 正余弦系数生成、读取、位置缩放和幅度缩放按实现单列 |
| 掩码、偏置、类型转换与量化 | 按配置累计；逻辑掩码不等于读取稠密掩码张量 |
| 通信与缓存元数据 | 张量并行（tensor parallelism，TP）、上下文并行（context parallelism，CP）的通信，以及分页缓存索引等，需与普通计算量和显存数据读写分开统计 |
| 模型块与训练 | 注意力输入端归一化、残差连接、前馈网络、专家混合、反向传播、激活保存与重计算不在当前范围内 |

投影与 RMSNorm 是否融合，决定中间压缩表示是否需要经过一次显存往返；RMSNorm 权重的实际数据类型也需明确。单头例子的 641 FLOPs / 640 Bytes 是所声明假设下的注意力核心参考值，不是完整 MLA 内核的实测计算强度。

## 8. 延伸：Attention kernel 优化

本章从第 3.4 节的访存模型出发，记录注意力内核如何通过数据复用和算子融合减少显存流量，并作为后续分析具体内核的入口。以下示例是成本模型推导，不是内核实测。

### 8.1 逐查询计算与查询分块：同一份 K/V 读取几次

固定一条序列、一个头，即 $B=h=1$，序列长度 $S=4$，采用包含当前位置的因果注意力。用 $KV_i$ 表示位置 $i$ 的 K、V 向量，合计 $p+d_v$ 个元素。

两种执行方式都计算相同的有效 Q–K 位置：

~~~text
             K₁  K₂  K₃  K₄
查询 q₁       ✓
查询 q₂       ✓   ✓
查询 q₃       ✓   ✓   ✓
查询 q₄       ✓   ✓   ✓   ✓
~~~

共有 $1+2+3+4=10$ 对有效位置。比较时假设 Q 各读一次、注意力输出各写一次，分数与 softmax 权重留在片上；不同查询块之间没有缓存复用。这里的“查询块”由同一序列、同一个头的多个查询位置组成。

设位置 $i$ 的 K/V 从显存读取 $r_i$ 次，则平均读取次数为：

$$
\rho=\frac1S\sum_{i=1}^{S}r_i,
\qquad
D_{\mathrm{KV}}=b(p+d_v)\sum_{i=1}^{S}r_i
=bS\rho(p+d_v).
$$

**逐查询计算。** 每个查询分别读取自己的 K/V 前缀，处理完成后不保留给下一个查询：

~~~text
处理 q₁ → 读取 KV₁
处理 q₂ → 读取 KV₁、KV₂
处理 q₃ → 读取 KV₁、KV₂、KV₃
处理 q₄ → 读取 KV₁、KV₂、KV₃、KV₄
~~~

因此 $r_1=4$、$r_2=3$、$r_3=2$、$r_4=1$：

$$
\rho_{\mathrm{basic}}=\frac{4+3+2+1}{4}=2.5.
$$

**查询分块计算。** 取查询块大小 $M=2$，让读入片上的一份 K/V 供块内两个查询使用：

~~~text
查询块 [q₁, q₂]
  读取 KV₁ → q₁、q₂ 共用
  读取 KV₂ → q₂ 使用，q₁ 对应位置被因果掩码遮住

查询块 [q₃, q₄]
  读取 KV₁ → q₃、q₄ 共用
  读取 KV₂ → q₃、q₄ 共用
  读取 KV₃ → q₃、q₄ 共用
  读取 KV₄ → q₄ 使用，q₃ 对应位置被因果掩码遮住
~~~

此时 $r_1=r_2=2$、$r_3=r_4=1$：

$$
\rho_{\mathrm{blocked}}=\frac{2+2+1+1}{4}=1.5.
$$

| K/V 位置 | 逐查询计算的读取次数 | 两查询分块的读取次数 |
|---|---:|---:|
| $KV_1$ | 4 | 2 |
| $KV_2$ | 3 | 2 |
| $KV_3$ | 2 | 1 |
| $KV_4$ | 1 | 1 |
| 合计 | 10 | 6 |
| 平均次数 $\rho$ | 2.5 | 1.5 |

K/V 读取量由 $10b(p+d_v)$ 减至 $6b(p+d_v)$，减少 40%。这里减少的是显存读取次数，有效 Q–K 配对仍为 10 对。

继续取 $p=192$、$d_v=128$、$b=2$。每对有效位置的主项计算量为 $F_{\mathrm{pair}}=2(p+d_v)+1=641$ FLOPs，因此两种方式均有：

$$
F_{\mathrm{core}}\approx10\times641=6{,}410\ \mathrm{FLOPs}.
$$

Q 读取与注意力输出写回合计为 $bS(p+d_v)=2{,}560$ Bytes。代入 $D_{\mathrm{core}}=bS(1+\rho)(p+d_v)$：

| 指标 | 逐查询计算 | 两查询分块 |
|---|---:|---:|
| 有效 Q–K 位置对应的计算量，softmax 单列 | 6,410 FLOPs | 6,410 FLOPs |
| K/V 读取 | 6,400 Bytes | 3,840 Bytes |
| Q 读取与注意力输出写回 | 2,560 Bytes | 2,560 Bytes |
| 注意力核心访存量 $D_{\mathrm{core}}$ | 8,960 Bytes | 6,400 Bytes |
| 计算强度 | 0.72 FLOPs/Byte | 1.00 FLOPs/Byte |

总核心访存量减少约 28.6%，计算强度提高 40%。两者的百分比不同，因为 Q 读取和输出写回保持不变。

**局部复用收益与整个核心的计算强度。** 在两查询分块的执行方式中，查询块 $[q_1,q_2]$ 读入 $KV_1$ 一次，供两个有效查询使用。该次读取为 $b(p+d_v)=640$ Bytes，支撑 $2F_{\mathrm{pair}}=1{,}282$ FLOPs，因此：

$$
I_{\mathrm{KV,reuse}}
\approx\frac{1{,}282}{640}
\approx2.00\ \mathrm{FLOPs/Byte}.
$$

它说明同一次 K/V 读取服务两个有效查询时的局部收益。整个注意力核心则为：

$$
I_{\mathrm{core}}
\approx\frac{6{,}410}{3{,}840+2{,}560}
\approx1.00\ \mathrm{FLOPs/Byte}.
$$

整体值更低，一是因为部分 K/V 读取未服务满两个有效查询，例如第一个查询块中的 $KV_2$ 只供 $q_2$ 使用；二是因为整体还计入 Q 读取和输出写回。局部指标用于解释复用机制，整体指标用于汇总该执行方式的注意力核心成本。

这里仍未计入投影、K/V 生成与缓存写入、softmax 和在线归一化的额外操作，以及掩码块内可能实际执行的额外计算。真实内核通常同时对查询和 K/V 分块，无须把整个前缀放入片上；分块越大，占用的片上存储也越多，性能不一定越好。

在相同分块下，片上缓存命中、数据布局与执行调度也会影响 HBM 流量。因此 $\rho$ 需要对应明确的存储层级与复用假设，不是由 Attention 数学公式确定的常数。

### 8.2 FlashAttention：数据复用与中间矩阵融合

上述示例体现了 FlashAttention 的分块复用思路。FlashAttention 还结合在线 softmax，将 QK、归一化和 V 聚合融合，避免完整的 $S\times S$ 分数与权重矩阵写回显存。在线状态更新见第 7.1 节。

~~~text
分开执行：
QK → 分数写入显存 → softmax → 权重写入显存 → 与 V 相乘

分块融合：
读入 Q/K/V 数据块 → 在片上计算分数、更新归一化状态与输出累加器
                → 处理后续数据块 → 写回最终输出
~~~

FlashAttention 保持精确注意力的数学定义，浮点求和顺序可带来数值差异。不同版本的循环顺序会改变 Q、K/V 和输出的读写次数，不能仅凭 $\rho$ 判断性能。

第 8.1 节的两种方式均已假设分数与权重不写回显存，所以示例只比较 K/V 复用，未展示从分开执行到融合执行的全部收益。分析具体实现时，还需核对 Q 与输出的读写次数及片上资源占用。

### 8.3 其他 Attention 优化方法改变哪些成本

不同方法可能作用于不同层次，也可以组合使用。下表用于定位成本变化，不表示各方法收益可以直接相加。

| 方法 | 主要优化方向 | 与本例的关系 |
|---|---|---|
| FlashAttention | 分块复用、算子融合，减少 HBM 中间读写 | 本例展示其中的数据复用思路；完整收益还包括避免中间矩阵写回 |
| FlashAttention-2 | 减少非矩阵乘法操作，改善线程块与块内分工 | 即使 HBM 流量相近，也可能通过更高的执行效率加速 |
| PagedAttention | 分页管理 KV cache，减少碎片，支持缓存共享 | 主要改善缓存容量利用与服务吞吐，不自动降低同一份 K/V 的读取次数 $\rho$ |
| GQA、MQA 与 MLA 压缩缓存 | 通过跨头共享或压缩表示，减少缓存元素数 | 改变每个位置需要存储和搬运的数据量；属于模型结构或表示层面的变化 |
| 稀疏注意力 | 只计算选定的 Q–K 位置 | 改变配对范围与计算量；相对于原始稠密注意力，可能改变模型定义或引入近似 |

后续分析具体内核时，可沿用同一顺序：先确认 Attention 定义与有效位置，再确定张量布局和读写边界，统计 HBM 流量与实际执行计算量，最后结合并行分工和测量结果判断瓶颈。$\rho$ 描述 K/V 的重复读取，只覆盖其中一项成本。

## 9. 来源与验证状态

- 苏剑林：[《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》](https://spaces.ac.cn/archives/10091)，2024-05-13。沿用其第 2、3 部分的行向量公式，并补充压缩表示的 RMSNorm。
- 苏剑林：[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)，2021-03-23，2026-10-26 查阅。第 2.4 节采用二维旋转与多维分块对角构造，明确转换为行向量记法；算术计数按本文统计约定推导。
- [DeepSeek-V2 技术报告](https://arxiv.org/html/2405.04434v5)与[官方参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py)，2026-10-25 查阅。参考实现用于确认 RMSNorm、分数缩放、缓存更新及部分路径的 V 补齐，不代表所有部署内核。
- 本地代码核对：Megatron-LM/megatron/core/transformer/multi_latent_attention.py 中的降维投影、压缩表示归一化与吸收路径，以及 experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py 中的在线 softmax 更新。核对时仓库 HEAD 为 83392004c81b6e477002ae84d4f0b1271518c1d1。
- [vLLM MLA 的数据搬运友好路径](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/model_executor/layers/attention/mla_attention.py#L93)与[FlashMLA 短预填充路径说明](https://github.com/vllm-project/vllm/blob/ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc/vllm/v1/attention/backends/mla/flashmla.py#L109)，2026-10-26 核对。版本为 ffcabb8c7ab3e066e1c77b55d7673e5d08cf26cc。第 5、6 节采用运行时 Q 侧变换、共享潜在 K/V 和后置 V 投影的组织；分块复用与显存边界由本文明确假设，不代表该内核实测流量。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135v2)，2022，2026-10-26 查阅。第 8.2 节参考其分块、融合与避免完整注意力矩阵写回显存的机制。
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)，2023，2026-10-26 查阅。第 8.3 节参考其摘要对非矩阵乘法成本和并行分工的说明。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，2023，2026-10-26 查阅。第 8.3 节参考其摘要对 KV cache 碎片与缓存共享的说明。
- 本文的计算量、显存访存量和计算强度均为所列假设下的推导，尚未运行模型或进行内核性能分析。实际读取次数、算子融合、缓存布局与吞吐量需结合目标实现验证。
