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
> 本文包含 AI 生成的推导，尚未完成人工核验。FLOPs、HBM traffic 和 arithmetic intensity 均为明确假设下的理论估算，不是 kernel benchmark。

## 1. Analysis scope

本文以 execution phase 与 computation form 为组织维度，分析 MLA 的 FLOPs、HBM traffic 和 arithmetic intensity，并为 decode 中增加 FLOPs、减少 KV cache traffic 的性能分析建立统一模型。

| Execution phase | Expanded | Absorbed |
|---|---|---|
| Prefill | 已建立单 token 的 operator cost；已给出 batch projection 与 KV reuse 的基本模型；完整 batch traffic 待分析 | 待分析 |
| Decode | 已建立单 decode token 的 Attention core 与 forward cost；已明确 batch 扩展的边界 | 待分析 |

各场景采用相同的推导层级：单个 token position 的 projection 与 normalization；单个 query–key pair 的 Attention contribution；单条 sequence 的可见 KV positions；最后扩展到 batch 和 multi-head。Single-head、single-token、single-query 均为局部推导的简化条件，不构成文档适用范围的限制。

Prefill 中的 single-token analysis 指输入序列中某个位置的 operator cost，不表示 prompt length 为 1。Causal Attention 的该位置仍可访问其 prefix。

标准 autoregressive decode 每条 active sequence 每步处理一个新输入 token，因此每条 sequence 的 Q_len=1。该 token 对应每个 head 的一个 query vector；若 batch size 为 $B$、head count 为 $h$，则该 step 有 $B$ 个 query positions、$Bh$ 个 query vectors。仅在固定 sequence 和 head 后，single query 才表示一个向量。Speculative decoding 等多 token 验证路径不属于 Q_len=1 的简化模型。

一个 KV position 是完整 softmax domain 中的一项。可见 KV length 为 1 时，softmax weight 恒为 1；本文的 per-position contribution 不采用该退化情形。

## 2. Common definitions

### 2.1 Notation

使用 row-vector convention，局部推导固定 head $s$，必要时省略 head index。

| Symbol | Definition |
|---|---|
| $B$ | Batch size，active sequences 的数量 |
| $S$ | 无 prefix cache、等长 prefill 的 prompt length |
| $Q_{\mathrm{len}}$ | 每条 sequence 本次处理的 query positions 数 |
| $h$ | Head count |
| $t,i$ | Query position、可见 KV position |
| $L,L_r$ | 单条 sequence 的可见 KV length、batch 中第 $r$ 条 sequence 的可见 KV length |
| $d$ | Model hidden dimension |
| $d_c,d_c'$ | KV latent dimension、query latent dimension |
| $d_k,d_r,d_v$ | Content Q/K dimension、RoPE dimension、V dimension |
| $p=d_k+d_r$ | Expanded Q/K head dimension |
| $b$ | 每个存储元素的 byte width；BF16 为 2 Bytes |
| $N$ | 一次 projection 处理的 token positions 数 |
| $M$ | 单次 KV load 在片上供其复用的 query 数 |
| $\boldsymbol x_t$ | Token hidden state，$1\times d$ |

### 2.2 Expanded forward

Down-projection 与 latent RMSNorm：

$$
\begin{aligned}
\boldsymbol z_t^Q&=\boldsymbol x_tW_c',&
\boldsymbol c_t'&=\mathrm{RMSNorm}(\boldsymbol z_t^Q),\\
\boldsymbol z_i^{KV}&=\boldsymbol x_iW_c,&
\boldsymbol c_i&=\mathrm{RMSNorm}(\boldsymbol z_i^{KV}).
\end{aligned}
$$

Q/K/V 与 Attention：

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

Output projection：

$$
\boldsymbol y_t
=[\boldsymbol o_t^{(1)},\ldots,\boldsymbol o_t^{(h)}]W_O
=\sum_{s=1}^{h}\boldsymbol o_t^{(s)}W_O^{(s)},
\qquad W_O^{(s)}\in\mathbb R^{d_v\times d}.
$$

单 head 的 output cost 表示该 head 的 contribution。完整 multi-head GEMM 通常已包含跨 head 的 reduction，不应再无条件叠加一次独立求和。

### 2.3 FLOPs 与 HBM traffic

- 一次 multiplication 或 addition 计 1 FLOP，FMA 计 2 FLOPs；GEMM 使用 $2mnk$ 的主项估算。
- Exp、comparison、division、rsqrt 分别列 operation count，不与普通 addition/multiplication 视为等成本指令。
- HBM traffic 指 HBM 与 on-chip storage 之间的数据传输，区别于 tensor capacity。重复 load 必须重复计数。
- 理论模型统一采用 byte width $b$，包含 RMSNorm weights；实际 dtype 不同时按 tensor 分别核算。On-chip accumulation 可使用 FP32。
- Down-projection、latent RMSNorm 和 positional K projection 由各 head 共享；它们按新 token positions 计数，不按历史 query–key pairs 计数。
- Score scaling 按每个 score 一次 multiplication 计数，$\sigma$ 已预先计算。若 kernel 将 scaling 移到 Q 或其他操作，按实际执行方式调整。
- Bias、attention input normalization、residual、dropout、mask processing、cache writes 和 communication 不纳入当前 aggregate，见第 7 节。

### 2.4 RMSNorm、RoPE 与 softmax

#### 2.4.1 RMSNorm

对 $n$ 维向量：

$$
r=\operatorname{rsqrt}\left(\frac1n\sum_{j=1}^{n}z_j^2+\epsilon\right),
\qquad \widehat c_j=z_jr\gamma_j.
$$

直接实现包含 $n$ 次 square、$n-1$ 次 reduction addition、1 次 mean scaling、1 次 epsilon addition，以及各 $n$ 次 normalization multiplication、affine multiplication。因此采用：

$$
R(n)=4n+1\ \mathrm{FLOPs},
$$

另列一次 rsqrt，并读取 $n$ 个 $\gamma$ elements。Latent RMSNorm 已纳入 aggregate，rsqrt 不折算为普通 FLOPs。

#### 2.4.2 RoPE：二维 rotation

RoPE 将 position 映射为 rotation angle。以下采用相邻 coordinate pairing 和 row-vector convention；“二维”指一对 feature coordinates，不指图像等二维 position coordinates。构造参考苏剑林的[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)。

设位置为 $m$，一对坐标为 $\boldsymbol u=[u_0,u_1]$，对应 angular frequency 为 $\theta$。Row-vector rotation matrix 定义为：

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

与 column-vector convention 的对应关系为：

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

原文的 column-vector rotation 与本文的 row-vector rotation 互为 transpose；数值变换一致。每对坐标执行 4 次 multiplication、2 次 addition/subtraction，共 6 FLOPs。

#### 2.4.3 RoPE：多维 block-diagonal form

设 RoPE branch dimension $d_r$ 为偶数，将 $\boldsymbol u\in\mathbb R^{1\times d_r}$ 划分为 $d_r/2$ 个 coordinate pairs，第 $j$ 对使用 frequency $\theta_j$：

$$
\theta_j=\mathrm{base}^{-2j/d_r},
\qquad j=0,\ldots,d_r/2-1.
$$

经典 RoPE 使用 $\mathrm{base}=10000$；具体模型的 base、frequency scaling 按配置确定。本文涉及的旋转维度是 $d_r$，而非 model dimension $d$ 或 expanded head dimension $p=d_k+d_r$。

多维 rotation matrix 为：

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

一般情形逐 pair 计算：

$$
\begin{aligned}
\widetilde u_{m,2j}
&=u_{2j}\cos(m\theta_j)-u_{2j+1}\sin(m\theta_j),\\
\widetilde u_{m,2j+1}
&=u_{2j}\sin(m\theta_j)+u_{2j+1}\cos(m\theta_j).
\end{aligned}
$$

因此实现采用 elementwise multiplication 与 pairwise addition/subtraction，无需执行 dense $d_r\times d_r$ GEMM。Sin/cos coefficients 已准备好时：

$$
F_{\mathrm{RoPE}}=6\frac{d_r}{2}=3d_r\ \mathrm{FLOPs/vector}.
$$

Coordinate pairing 的物理 layout 可通过 permutation 调整；具体 kernel 的 interleaved 或 split-half layout 应与权重及 feature ordering 一致。

#### 2.4.4 RoPE：relative position identity

各二维 block 满足 rotation composition 和 orthogonality，因此：

$$
\mathcal R_m\mathcal R_n=\mathcal R_{m+n},
\qquad
\mathcal R_m^\top=\mathcal R_{-m},
\qquad
\mathcal R_t\mathcal R_i^\top=\mathcal R_{t-i}.
$$

设 $\boldsymbol u_t,\boldsymbol w_i\in\mathbb R^{1\times d_r}$ 为未施加 RoPE 的 query/key position-branch vectors。旋转后的内积为：

$$
\begin{aligned}
(\boldsymbol u_t\mathcal R_t)
(\boldsymbol w_i\mathcal R_i)^\top
&=\boldsymbol u_t\mathcal R_t\mathcal R_i^\top\boldsymbol w_i^\top\\
&=\boldsymbol u_t\mathcal R_{t-i}\boldsymbol w_i^\top.
\end{aligned}
$$

Rotation 引入的 position dependence 通过 relative offset $t-i$ 进入 dot product。MLA decoupled RoPE 将该运算限制在 position branch；content branch 不施加 rotation，保留 absorption 所需的线性结构。

#### 2.4.5 Softmax 与 stable softmax

固定 query position $t$ 和 head $s$，将 $L$ 个可见 KV positions 的 logits 记为 $s_i=s_{t,i}^{(s)}$，其中已包含 score scaling。Softmax 定义为：

$$
\alpha_i=\operatorname{softmax}(\boldsymbol s)_i
=\frac{\exp(s_i)}{\sum_{j=1}^{L}\exp(s_j)},
\qquad
\alpha_i\ge0,\quad \sum_{i=1}^{L}\alpha_i=1.
$$

Stable softmax 使用 shift invariance，将最大 logit 从每一项中减去：

$$
\begin{aligned}
m&=\max_{1\le j\le L}s_j,\\
e_i&=\exp(s_i-m),\\
\ell&=\sum_{j=1}^{L}e_j,\\
\alpha_i&=e_i/\ell.
\end{aligned}
$$

该变换保持 softmax 的数学结果：

$$
\frac{\exp(s_i-m)}{\sum_j\exp(s_j-m)}
=\frac{\exp(-m)\exp(s_i)}
{\exp(-m)\sum_j\exp(s_j)}
=\frac{\exp(s_i)}{\sum_j\exp(s_j)}.
$$

对于有限 logits，$s_i-m\le0$，避免直接计算较大正数的 exp。V 不参与 softmax；normalization 后的 weights 用于：

$$
\boldsymbol o_t^{(s)}=\sum_{i=1}^{L}\alpha_i\boldsymbol v_i^{(s)}.
$$

| Operation | Count / query / head |
|---|---:|
| Maximum reduction | $L-1$ comparisons |
| Shift | $L$ subtractions |
| Exponentiation | $L$ exp operations |
| Sum reduction | $L-1$ additions |
| Normalization | $L$ divisions，或 1 次 reciprocal 与 $L$ 次 multiplications |

约 $5L$ 次 scalar operations 是粗略 operation count，不表示这些指令具有相同硬件成本。当前 FLOPs aggregate 不包含 softmax operations；online softmax 的 tile-wise accumulator update 与 implementation overhead 见第 7.1 节，不能与另一套 normalization 实现重复累计。

### 2.5 Arithmetic intensity 与 Roofline

$$
I=\frac{F}{D}\quad\mathrm{FLOPs/Byte},
\qquad
T\gtrsim\max\left(\frac FP,\frac D{B_{\mathrm{HBM}}}\right).
$$

$F,D$ 必须对应相同 execution scope；$P$ 为相应 precision/operator 的 compute throughput，$B_{\mathrm{HBM}}$ 为 HBM bandwidth。Roofline ridge point 为 $P/B_{\mathrm{HBM}}$，上述时间关系不等于 benchmark latency。

## 3. Prefill × expanded

### 3.1 单 token position：operator cost

Prefill 为每个输入 position 生成 Q/K/V。下表先以一个 token position 和一个 head 展示 operator cost，并保留 shared operations 的归属；Attention 的 query–key pair 数另行累计。

| Operator | Formula | FLOPs / operation count | Frequency |
|---|---|---:|---|
| Query down-projection | $\boldsymbol z_t^Q=\boldsymbol x_tW_c'$ | $2dd_c'$ FLOPs | 每个新 token 一次，多头共享 |
| Query latent RMSNorm | $\boldsymbol c_t'=\mathrm{RMSNorm}(\boldsymbol z_t^Q)$ | $4d_c'+1$ FLOPs，另有 1 次 rsqrt | 每个新 token 一次，多头共享 |
| KV down-projection | $\boldsymbol z_i^{KV}=\boldsymbol x_iW_c$ | $2dd_c$ FLOPs | 每个新 token 一次，多头共享 |
| KV latent RMSNorm | $\boldsymbol c_i=\mathrm{RMSNorm}(\boldsymbol z_i^{KV})$ | $4d_c+1$ FLOPs，另有 1 次 rsqrt | 每个新 token 一次，多头共享 |
| Query up-projection | $[\boldsymbol c_t'W_{qc},\boldsymbol c_t'W_{qr}]$ | $2d_c'(d_k+d_r)$ FLOPs | 每个当前 query、每个头一次 |
| Query RoPE | $(\boldsymbol c_t'W_{qr})\mathcal R_t$ | $3d_r$ FLOPs | 每个当前 query、每个头一次 |
| Content K up-projection | $\boldsymbol c_iW_{kc}$ | $2d_cd_k$ FLOPs | 每个新 K、每个头一次 |
| Positional K projection | $\boldsymbol x_iW_{kr}$ | $2dd_r$ FLOPs | 每个新 K 一次，所有头共享 |
| Key RoPE | $(\boldsymbol x_iW_{kr})\mathcal R_i$ | $3d_r$ FLOPs | 每个新 K 一次，所有头共享 |
| V up-projection | $\boldsymbol c_iW_v$ | $2d_cd_v$ FLOPs | 每个新 V、每个头一次 |
| QK dot product | $\boldsymbol q_t\boldsymbol k_i^\top$ | 约 $2p$ FLOPs | 每个 query-key 对、每个头一次 |
| Score scaling | $s_{t,i}=(\boldsymbol q_t\boldsymbol k_i^\top)\sigma$ | 1 FLOP | 每个 query-key 对、每个头一次 |
| Softmax | $\alpha_{t,i}=\operatorname{softmax}_i(s_{t,i})$ | 对长度 $L$ 的整组分数为 $O(L)$ | 每个 query、每个头做一次整体归一化 |
| V weighted accumulation | $\boldsymbol o_t\mathrel{+}=\alpha_{t,i}\boldsymbol v_i$ | 约 $2d_v$ FLOPs | 每个 query-value 对、每个头一次 |
| Output projection contribution | $\boldsymbol o_tW_O^{(s)}$ | $2d_vd$ FLOPs | 历史位置求和完成后，每个 query、每个头一次 |

历史位置的 K/V generation 每个 position 执行一次。Q projection 与 output projection 每个 query position 执行一次。仅 QK、score scaling、V weighted accumulation 按 query–key pairs 累计。

KV down-projection 与 positional K projection 可融合为输出宽度 $d_c+d_r$ 的 GEMM，合计 $2d(d_c+d_r)$ FLOPs，已包含 positional projection 的 $2dd_r$，不能重复计数。

### 3.2 单 token position：cold-read cost model

本模型用于保存单向量 operator cost，不代表实际 prefill 的 batch HBM traffic。假设该 position 使用的 weights 从 HBM load 一次，输入读一次；intermediate activations 在片上复用，output 写回一次，sin/cos 与 scaling coefficients 已在片上。Softmax、Attention KV loads 和 persistent cache writes 在此不计入。

固定一个 head，并将 shared path 计一次，定义：

$$
\begin{aligned}
A&=d_c'p+d_c(d_k+d_v)+dd_r+d_vd,\\
C&=d(d_c'+d_c),\\
G&=d_c'+d_c.
\end{aligned}
$$

$A$ 为 up-projection、positional K projection、output projection 的 weight elements；$C$ 为 down-projection weight elements；$G$ 为 RMSNorm weight elements。

| Operator group | FLOPs | HBM traffic |
|---|---:|---:|
| Down-projection | $2C$ | Weights：$bC$ |
| Latent RMSNorm | $4G+2$，另有 2 次 rsqrt | Weights：$bG$；latent activations 片上复用 |
| Up-projection、positional K projection、output projection | $2A$ | Weights：$bA$ |
| Q/K RoPE | $6d_r$ | Sin/cos 片上复用 |
| Token input/output | — | Input read 与 output write：$2bd$ |

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

这是 per-token projection/normalization/RoPE/output cost 的 cold-read reference，未包含该位置与 prefix 的 Attention contribution。Softmax 以及 materialized Q/K/V 的 intermediate traffic 不在该式内。Prefill 的实际 operator fusion、tiling 与 batch reuse 必须另建模型。

### 3.3 Query–key pair 与 sequence 维度

对单个 query position $t$，每个可见 KV position 的普通 arithmetic contribution 为：

$$
F_{\mathrm{pair}}=2p+1+2d_v=2(p+d_v)+1.
$$

该位置的 Attention core 主项为 $tF_{\mathrm{pair}}$，采用无 prefix cache、1-based causal positions。其 KV traffic 取决于与其他 query positions 的复用，不能把 decode 的独立 KV load 模型直接逐 position 累加为 prefill traffic。

全序列长度为 $S$ 时，causal query–key pairs 为 $S(S+1)/2$。单 head 简化下，operator 与 Attention core 的 FLOPs 扩展为：

$$
F_{\mathrm{prefill,expanded}}^{(h=1)}
\approx
S F_{\mathrm{token,ops}}
+\frac{S(S+1)}2 F_{\mathrm{pair}}.
$$

该式仍不含 softmax operations 等单列项目；$S=1$ 的 softmax 退化不属于 per-pair 主项估算的独立执行模型。

### 3.4 Batch 与 multi-head 扩展

等长 prefill 的 token positions 为 $N=BS$。以单个 projection $Y=XW$ 为例：

$$
I_{\mathrm{projection}}
\approx
\frac{2Nd_{\mathrm{in}}d_{\mathrm{out}}}
{b(Nd_{\mathrm{in}}+d_{\mathrm{in}}d_{\mathrm{out}}+Nd_{\mathrm{out}})}.
$$

该式假设 input、weights 各 load 一次，output store 一次。Weight reuse 使 HBM traffic 不随 token count 等比例增加。

Attention kernel 中，若同一 KV tile 在片上供 $M$ 个 query 复用，仅考虑该次 KV load 支撑的 arithmetic：

$$
I_{\mathrm{KV,reuse}}
\approx
\frac{M[2(p+d_v)+1]}{b(p+d_v)}.
$$

$M$ 是 kernel tile 的实际 reuse count，不保证等于 $S$。Q/output traffic、KV reload、materialized intermediate tensors 与 cache writes 尚未纳入该式。完整 batch HBM model 和 multi-head aggregate 待分析，不能将 per-token cold-read intensity 视为 prefill intensity。

## 4. Decode × expanded

### 4.1 单 decode token 与 cache model

标准 decode 对每条 active sequence 使用 Q_len=1。Expanded query tensor 的一种 layout 为 $Q\in\mathbb R^{B\times h\times1\times p}$；固定 sequence 和 head 后，Attention operands 为：

$$
\boldsymbol q_t\in\mathbb R^{1\times p},
\qquad K_{\le t}\in\mathbb R^{L\times p},
\qquad V_{\le t}\in\mathbb R^{L\times d_v}.
$$

本节假设历史 expanded K/V 已持久存储在 KV cache，每个位置读取一次；当前 token 的 Q/K/V generation 只执行一次。Expanded computation 与 expanded cache layout 必须区分：若持久保存 compressed cache、运行时再展开历史 K/V，需要额外计入 historical up-projection 与 intermediate traffic，本节 cache model 不适用。

### 4.2 单个 KV position：Attention core

该层级固定一个 sequence、一个 head 和一个 query–key pair。Q 已生成，历史 K/V 已展开并写入 KV cache；仅统计 Attention core，不含 projection weights 和当前 token 的 cache write。

假设 Q、output accumulator 保留在片上；每个历史 K/V 从 HBM load 一次；softmax 与 Attention 融合，不将完整 scores、weights 写回 HBM。Online softmax 的 accumulator rescaling 等 implementation overhead 在第 7 节单列。

| Operator | FLOPs / operation count | HBM read | HBM write |
|---|---:|---:|---:|
| QK dot product | 约 $2p$ FLOPs | K load：$bp$ Bytes | Scores 在片上，0 |
| Score scaling | 1 FLOP | Scores 与 scaling coefficient 在片上，0 | 0 |
| Softmax | 每 KV position 摊销约 5 次 scalar operations | Scores 在片上，0 | Weights 在片上，0 |
| V weighted accumulation | 约 $2d_v$ FLOPs | V load：$bd_v$ Bytes | Accumulator 在片上，0 |
| **Total，softmax operations 单列** | **$2(p+d_v)+1$ FLOPs** | **$b(p+d_v)$ Bytes** | **不按 KV position store output** |

计入 QK、score scaling 和 V weighted accumulation，对该 KV position 的 arithmetic intensity 为：

$$
I_{\mathrm{pair,KV}}
\approx\frac{2(p+d_v)+1}{b(p+d_v)}
=\frac2b+\frac1{b(p+d_v)}
\quad\mathrm{FLOPs/Byte}.
$$

### 4.3 单个 decode token：扩展到 $L$ 个 KV positions

QK、score scaling 与 V weighted accumulation 的 FLOPs：

$$
F_{\mathrm{core}}\approx L[2(p+d_v)+1].
$$

仅统计 historical KV cache load：

$$
D_{\mathrm{KV}}=bL(p+d_v),
\qquad
I_{\mathrm{core,KV}}\approx\frac{2(p+d_v)+1}{b(p+d_v)}.
$$

若再计入当前 Q 从 HBM 读一次，以及 Attention 输出写回一次，并仍采用相同存储宽度 $b$：

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

Q 位于片上或 output 由融合的后续 operator 直接使用时，相应 HBM traffic 可省略。KV reload、scores materialization、split-K partial outputs 等额外 traffic 需单独计入。

### 4.4 Numerical example

设 $d_k=128$、$d_r=64$、$d_v=128$，故 $p=192$；cache 使用 BF16，$b=2$。

| Metric | Single head / KV position |
|---|---:|
| QK dot product | $2\times192=384$ FLOPs |
| Score scaling | 1 FLOP |
| V weighted accumulation | $2\times128=256$ FLOPs |
| Arithmetic total | **641 FLOPs** |
| K load | $2\times192=384$ Bytes |
| V load | $2\times128=256$ Bytes |
| KV traffic total | **640 Bytes** |
| KV-based arithmetic intensity | **$641/640=1.0015625$ FLOPs/Byte** |

计数约定：仅统计 QK 和 V weighted accumulation 时为 640 FLOPs；计入一次 score scaling 后为 641 FLOPs。Score scaling 使用片上数据，不增加 HBM traffic。Down-projection 和 RMSNorm 属于每个新 token 的一次性成本，在第 4.5 节计入，不按历史位置重复累计。

取 $L=8192$，固定一个 sequence、一个 head 和一个 decode token：

$$
F_{\mathrm{core}}=641\times8192=5{,}251{,}072\ \mathrm{FLOPs},
$$

$$
D_{\mathrm{KV}}=640\times8192=5{,}242{,}880\ \mathrm{Bytes}=5\ \mathrm{MiB}.
$$

计入一次 Q load 和 output store 后，HBM traffic 增加 640 Bytes，arithmetic intensity 为 $5{,}251{,}072/5{,}243{,}520\approx1.00144025$ FLOPs/Byte。$1\ \mathrm{MiB}=2^{20}$ Bytes；score scaling 已计入，softmax operations 仍单列。

### 4.5 单 decode token：forward aggregate

除第 4.3 节 Attention core 外，加入当前 token 的 down-projection、RMSNorm、up-projection、RoPE 和 output projection。$A,C,G$ 采用第 3.2 节定义。

假设 weights cold-load 一次，当前 hidden state 读一次后复用；intermediate activations、Q、accumulators 在片上，最终 output 写回一次。固定一个 head，并计入一次 shared path：

| Operator | FLOPs / operation count | HBM read | HBM write |
|---|---:|---:|---:|
| Query down-projection | $2dd_c'$ | 权重 $bdd_c'$；输入 $bd$ | 0，结果留片上 |
| Query latent RMSNorm | $4d_c'+1$，另有 1 次 rsqrt | $\gamma_Q$：$bd_c'$ | 0 |
| KV down-projection | $2dd_c$ | 权重 $bdd_c$；复用输入 | 0 |
| KV latent RMSNorm | $4d_c+1$，另有 1 次 rsqrt | $\gamma_{KV}$：$bd_c$ | 0 |
| Query up-projection | $2d_c'p$ | 权重 $bd_c'p$ | 0 |
| Query RoPE | $3d_r$ | 0 | 0 |
| Key up-projection / positional projection | $2d_cd_k+2dd_r$ | 权重 $b(d_cd_k+dd_r)$ | 0 |
| Key RoPE | $3d_r$ | 0 | 0 |
| V up-projection | $2d_cd_v$ | 权重 $bd_cd_v$ | 0 |
| QK dot product 与 score scaling | $L(2p+1)$ | Expanded K：$bLp$ | 0 |
| Softmax | Special operations 与 reduction 单列，见第 2.4 节 | Scores 在片上，0 | 0 |
| V weighted accumulation | $2Ld_v$ | Expanded V：$bLd_v$ | 0 |
| Output projection | $2d_vd$ | 权重 $bd_vd$ | 输出 $bd$ |

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

该 aggregate 包含 down-projection、latent RMSNorm 和 score scaling；2 次 rsqrt、softmax operations、online softmax implementation overhead 仍单列。

HBM model 暂不计当前 K/V 的 cache writes。若 $L$ 包含当前位置且其 K/V 由片上直接供 Attention 使用，可以减去对应 KV load。由原始 hidden state 生成的 latents 不再作为外部输入重复 load。

$B=1,h=1$ 是该推导层级的简化。推广到 multi-head 时，shared down-projection、RMSNorm 与 positional K path 只计一次，不能将全部 aggregate 直接乘以 $h$。

### 4.6 Batch 与 multi-head 扩展

标准 decode batch 的 query positions 为 $N=B$，每条 sequence 的 Q_len=1。对于固定 head：

| Component | Batch 扩展规则 |
|---|---|
| Projection weights | 可在 $B$ 个新 token 间复用；projection 的 GEMV/GEMM 形态和 weight traffic 随 batch 改变 |
| Query positions | 每条 sequence 一个，总计 $B$ 个 |
| Visible KV positions | 第 $r$ 条 sequence 为 $L_r$，合计 $\sum_r L_r$ |
| Historical KV reuse | 不同 sequence 的 KV 通常不同；增加 $B$ 不自动使同一 KV 被多个 query 复用 |
| Multi-head paths | Head-specific operations 按 head 累计，shared paths 单独计数 |

在相同的 one-load expanded cache model 下，固定 head 的 Attention core 可扩展为：

$$
F_{\mathrm{core,batch}}\approx[2(p+d_v)+1]\sum_{r=1}^{B}L_r,
\qquad
D_{\mathrm{KV,batch}}\approx b(p+d_v)\sum_{r=1}^{B}L_r.
$$

仅此 KV-based intensity 保持不变；projection intensity、完整 forward intensity 与 kernel efficiency 仍可能随 batch 改变。Full batch/multi-head aggregate、prefix sharing 与实际 kernel traffic 待分析。

## 5. Prefill × absorbed

### 5.1 单 token position

待分析。沿用单 token operator、query–key pair、sequence 的推导层级，分别计算 query-side absorption、latent-space Attention 和 output-side V projection。Score scaling 保持模型原有的 $\sigma$，不因 absorption 改为按 latent dimension 重新计算。

### 5.2 Batch 与 multi-head 扩展

待分析。显式区分 query reuse、cross-head shared KV reuse 与 intermediate materialization；不得直接采用 expanded 的 HBM traffic。

## 6. Decode × absorbed

### 6.1 单 decode token

待分析。每条 sequence 的 Q_len=1；在相同 execution scope 下分别统计 query transformation、shared compressed cache load、latent-space Attention 与 output projection。

### 6.2 Batch 与 multi-head 扩展

待分析。保留 head count 与 batch size 为独立维度。Shared cache 的物理读取次数由 kernel 的 cross-head reuse 决定；不能将 single-head traffic 直接乘以 $h$ 作为 shared-cache aggregate。

完成 FLOPs 与 HBM model 后，再以 Roofline 比较 increased arithmetic 与 reduced memory traffic 的收益。

## 7. Implementation scope

### 7.1 Online softmax

Fused Attention 避免完整 scores/weights 的 HBM materialization，但仍执行跨 KV tiles 的 numerically stable accumulation。设 KV tile size 为 $B_K$，tile count 为 $n_{\mathrm{tile}}=\lceil L/B_K\rceil$；旧状态为 maximum $m$、denominator $\ell$ 和未归一化 accumulator $\boldsymbol u$。对当前 tile 的 scores：

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

除 QK 与 weighted accumulation 主项外，accumulator rescaling 和 final normalization 的 vector arithmetic 为 $O(n_{\mathrm{tile}}d_v+d_v)$；absorbed 中对应维度可能为 $d_c$。实际 kernel 可省略首 tile rescaling、采用 exp2 或其他 reduction 策略。Explicit normalized weights 与 online output normalization 是不同实现，不能将二者的全部 normalization cost 重复累计。

### 7.2 Aggregate exclusions 与 implementation-dependent costs

| Item | Accounting requirement |
|---|---|
| Cache writes | 单 head expanded 的完整 K/V payload 为每新 token $b(p+d_v)$ Bytes；compressed cache 和 multi-head shared RoPE 需使用各自 layout |
| Compressed cache + runtime expansion | 计入实际展开的历史位置的 up-projection 与 intermediate traffic |
| Intermediate materialization | 独立算子的 scores、weights、latents、Q/K/V 可能产生额外 HBM read/write |
| Physical padding | Mathematical dimensions 不一定等于 kernel execution width；参考 FlashAttention 路径可能将 V pad 至 Q/K dimension |
| Layout operations | Reshape/view 可只修改 metadata；contiguous、materialized concatenation、broadcast copy 可能产生实际 traffic |
| Split-K / partial-output reduction | 计入 partial outputs、normalization statistics 的读写与 merge arithmetic |
| RoPE coefficients | Sin/cos generation、load、position scaling、amplitude scaling 依实现单列 |
| Mask、bias、dtype conversion、quantization | 按配置累计；logical mask 不等于读取 dense mask |
| Communication 与 cache metadata | TP/CP communication、paged-cache indices 等与普通 FLOPs/HBM payload 分开统计 |
| Transformer block / training | Attention input normalization、residual、FFN/MoE、backward、activation saving、recomputation 在当前 execution scope 之外 |

Projection + RMSNorm 的 fusion 决定 latent 是否经过 HBM round trip；RMSNorm weights 的实际 dtype 也需明确。Single-head numerical example 641 FLOPs / 640 Bytes 对应上述声明下的 Attention core reference，不是完整 MLA kernel 的 measured intensity。

## 8. Sources 与 validation status

- 苏剑林：[《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》](https://spaces.ac.cn/archives/10091)，2024-05-13。公式使用其 Part 2、Part 3 的 row-vector convention，并补充 latent RMSNorm。
- 苏剑林：[《Transformer升级之路：2、博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)，2021-03-23，2026-10-26 查阅。第 2.4 节采用其二维 rotation 与多维 block-diagonal 构造，并显式转换为 row-vector convention；算术计数按本文统计约定推导。
- [DeepSeek-V2 technical report](https://arxiv.org/html/2405.04434v5) 与 [official reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py)，2026-10-25 查阅。Reference implementation 用于确认 RMSNorm、score scaling、cache update 与部分路径的 V padding，不代表所有 deployment kernels。
- Local source inspection：Megatron-LM/megatron/core/transformer/multi_latent_attention.py 的 down-projection、latent normalization、absorption paths，以及 experimental_attention_variant/ops/tilelang_sparse_mla_fwd.py 的 online softmax update。核对时 repository HEAD 为 83392004c81b6e477002ae84d4f0b1271518c1d1。
- FLOPs、HBM traffic、arithmetic intensity 均为声明假设下的推导；尚未进行 model execution 或 kernel profiling。具体 load counts、fusion、cache layout 与 throughput 需结合目标实现验证。
