---
type: knowledge
status: seed
created: 2026-10-13
updated: 2026-10-13
domains:
  - megatron
  - distributed-training
  - numerical-accuracy
  - inference-consistency
aliases:
  - Megatron Forward 跨并行数值一致性
  - Megatron 并行策略与 Logprobs Diff
ai_generated: true
reviewed: false
---

# Megatron 不同并行策略对 Forward 数值一致性的影响

> [!warning]
> 本文由 AI 根据技术调研整理，尚未完成人工核验。**源码确认、数学推导、构造性实验、工程假设**具有不同证据强度；下文分别标明。不能用本文直接判定具体生产模型的 $10^{-2}$ logprob diff 是正常现象。
>
> **调研日期**：2026-10-13。**主要实现基线**：Megatron-Core `core_v0.19.2`、Transformer Engine `v2.18`，并参考 PyTorch、cuBLAS、NCCL 官方文档和特定版本源码；不意味着本地运行环境就是这些版本。

## 1. Executive Summary

### 1.1 背景与研究目标

本地实验现象（**用户提供，尚未通过本文独立复现**）：

- 相同 Megatron 并行配置，开启确定性计算后，run-to-run logprobs diff = 0。
- 相同权重和输入下，**不同并行配置之间**观测到 $10^{-2}$ 量级的 logprobs diff；具体统计指标尚未限定。
- 目前尚未定位最早出现差异的 operator。

目标不是证明“浮点误差正常”，而是建立一条**可量化、可重放、可干预验证**的证据链：

$$
\text{比较语义一致}
\longrightarrow
\text{logits / normalization 分离}
\longrightarrow
\text{最早差异边界}
\longrightarrow
\text{相同输入算子重放}
\longrightarrow
\text{高精度干预与成本评估}.
$$

### 1.2 核心结论

1. **确定性 $\neq$ 跨并行位级一致**。同一个浮点程序可以稳定地复现其结果；TP、CP、EP 等变化后，GEMM shape、分块、输出 dtype、归约/合并顺序乃至 kernel 可能改变，已经是另一个浮点程序。[R1][R2]
2. **先排除比较对象不一致**。权重重建、token/position 对齐、vocab padding、目标词 ID、logprob 归一化域、temperature、融合路径等问题，都可以制造比单纯舍入更严重的差异。
3. **优先研究 TP/ETP row-parallel partial GEMM、CP 分块 attention 的 LSE/output 合并、MoE router/expert/combine**。这些是可以从实现定位到的候选机制；但具体起点必须由本地实验决定。[R3][R4][R5][R6]
4. **Column-parallel GEMM 也可能更早产生差异**：虽然它通常不需要 forward 对多个 partial sums 做跨卡 SUM，但 GEMM 的 $N$ 维度、tiling、backend 可能变化。
5. **单纯的 AllGather / AllToAll / Send-Recv 不直接做数值求和**；SUM AllReduce / ReduceScatter 和带加权/归约的融合通信才是明确的算术候选边界。[R7]
6. **$10^{-2}$ 可以通过 BF16 中间舍入构造出来，但不能据此宣布任何实际 $10^{-2}$ diff 都合理**。必须区分 max/mean/P99、单 token/序列累积，以及是否出现路由/argmax 等离散决策变化。
7. **统一 FP64 log-softmax 是第一个高价值对照**。它可以在不重新运行模型的情况下区分模型主体差异与最后的归一化、后处理差异。

### 1.3 排查优先级

| 优先级 | 检查对象 | 排查价值 |
|---|---|---|
| P0 | 比较对象、输入/权重/状态、词表、logits→logprobs | 低成本排除语义及测量问题 |
| P1 | TP/ETP：局部 GEMM 输出 + SUM collective | 明确的分块和舍入边界 |
| P1 | CP：局部 attention、LSE 和 output correction | 可能改变 attention 的计算图 |
| P1，MoE 适用 | Router top-k、expert GEMM、token combine | 连续数值误差可能触发离散路径 |
| P2 | SP、kernel/fusion、通信计算 overlap、量化状态 | 变化路径广，需要针对实验隔离 |
| 控制组 | DP-only、PP-only | 计算路径本应较接近，适合发现环境或语义不一致 |

## 2. 数学符号表

**约定**：除特别说明外，logits/logprobs 针对同一样本的**单个预测位置**；$Y=XW$ 是数学排列，而 PyTorch `F.linear` 的存储权重一般为 $[N,K]$。

| 符号 | 含义 |
|---|---|
| $A,B$ | 两套并行或执行配置 |
| $\mathcal C,\widehat F_{\mathcal C}$ | 执行配置及其实际浮点程序 |
| $X,W,Y$ | Linear 输入、数学排列权重、输出，$Y=XW$ |
| $M,N,K$ | GEMM 的行数、输出维度、归约维度 |
| $P,r$ | 分片数与分片编号，$r=0,\ldots,P-1$ |
| $R_b(\cdot)$ | 舍入到 BF16（默认 round-to-nearest ties-to-even） |
| $s_r$ | 第 $r$ 个分片的局部部分和 |
| $h_\ell,f_\ell,J_\ell$ | 第 $\ell$ 个残差子层的输入、分支函数及其雅可比矩阵 |
| $I,\delta h_\ell,\eta_\ell$ | 单位矩阵、上游输入扰动、当前实现新引入的误差 |
| $Q,K_{\mathrm{attn}},V_{\mathrm{attn}}$ | Attention 的 Query/Key/Value，避免与 GEMM 的 $K$ 混淆 |
| $O_j,\lambda_j$ | 第 $j$ 个 KV 分块的局部 attention 输出及 log-sum-exp |
| $\mathcal V,\mathcal V_r$ | 完整归一化词表及第 $r$ 个词表分片 |
| $z_i,p_i,g_i$ | 第 $i$ 个 token 对应 logit、概率和 logprob |
| $y$ | 当前预测位置的目标 token ID |
| $\operatorname{LSE}(z)$ | $\log\sum_{i\in\mathcal V}\exp(z_i)$ |
| $\delta z,\Delta g_y,\epsilon$ | logits 扰动、目标 logprob 差及 $\lVert\delta z\rVert_\infty$ |
| $\tau$ | Temperature，默认 $\tau=1$ |
| $k,m$ | MoE top-k，以及第 $k$、$k+1$ 名路由分数的差 |
| $T$ | 序列汇总或统计时的有效 token 数 |

| $\theta,\theta_{\mathrm{old}}$ | RL 当前优化权重与 rollout 采样时的旧权重版本 |
| $s_t,a_t$ | 第 $t$ 个预测位置的上下文状态与采样 token |
| $\pi^{\mathrm{train}}_\theta,\pi^{\mathrm{rollout}}_\theta$ | 相同权重下训练/rollout 引擎实际实现的 token 条件分布 |
| $\delta_t$ | 同一旧权重下的 train–rollout token logprob 差（TIM） |
| $\Delta_t^{\mathrm{update}}$ | 同一训练引擎下，当前与旧策略 logprob 差 |
| $\rho_t^{\mathrm{beh}},\rho_t^{\mathrm{train}}$ | 分母分别为实际 rollout 行为分布与 trainer 旧策略分布的比率 |
| $\widehat A_t,\varepsilon_{\mathrm{clip}}$ | RL 中第 $t$ 个 token 的优势估计、PPO 风格裁剪阈值 |

## 3. 确定性与数值误差的计算原理

### 3.1 三个不同目标

| 目标 | 判据 | 不代表 |
|---|---|---|
| Deterministic Execution | 相同执行条件下不受未受控随机性/竞争影响 | 任意等价程序得到同样的结果 |
| Bitwise Reproducibility | dtype、逻辑排列、完整数据位一致 | 变更并行布局后仍一致 |
| Numerical Equivalence | 满足事先约定的误差指标与容差 | 位级相等，或任务行为一定相同 |

同配置满足：

$$
\widehat F_{\mathcal C}(X)
=
\widehat F_{\mathcal C}(X)
$$

并不推出：

$$
\widehat F_A(X)
=
\widehat F_B(X).
$$

PyTorch 文档明确说明数学等价的 batched/sliced 运算可能给出不同浮点结果；cuBLAS 的复现保证也有 GPU、工具包、SM/stream 等适用条件。参见 [R1][R2]。

**实验注意**：打印出 `diff=0` 不等于位级一致。应在逻辑重排后，比较原始 dtype、数据位和非有限值情况；不要只依赖有限精度格式化输出。

### 3.2 TP 局部结果舍入：与 collective 顺序不同的第二种误差

考虑 row-parallel GEMM：

$$
Y=XW=\sum_{r=0}^{P-1}X_rW_r.
$$

TP=1 近似为完整点积后舍入；TP=$P$ 可能先分别计算局部部分和：

$$
s_r=X_rW_r,
$$

局部 GEMM 落盘到低精度 Tensor 后，再执行跨卡 SUM。**局部输出 dtype** 与 **GEMM 的内部累加 dtype** 是两个不同概念。

即使假定最终合并使用高精度，也一般不能保证：

$$
R_b\left(\sum_r s_r\right)
=
R_b\left(\sum_rR_b(s_r)\right).
$$

把已经舍入的 BF16 partial 转成 FP32 再 AllReduce，**不能恢复局部输出已丢失的尾数**。

#### BF16 构造性示例（非真实 Megatron 复现）

令：

$$
X=(2,\;2^{-7},\;2^{-7},\;0).
$$

完整求和：

$$
R_b(2+2^{-7}+2^{-7})=2.015625.
$$

两块分开舍入：

$$
R_b(2+2^{-7})=2,\qquad
R_b(2^{-7}+0)=2^{-7}.
$$

重新合并后舍入：

$$
R_b(2+2^{-7})=2.
$$

将结果作为二分类 logits 的第二项，目标 token 为第 0 项，用同一 FP64 log-softmax 计算：

| 路径 | Logits | 目标 logprob |
|---|---|---|
| 完整求和后舍入 | $[0,2.015625]$ | $-2.1407032312$ |
| 分块舍入后合并 | $[0,2]$ | $-2.1269280110$ |

因此：

$$
|\Delta g_y|\approx 0.0137752202.
$$

**解释**：这是一个构造性数值例子，只证明跨分块舍入足以产生 $10^{-2}$ logprob 差；不表示实际模型的 diff 已找到来源，也不等价于生产 NCCL 算法行为。

### 3.3 Transformer 误差传播：区分“原有误差传播”与“本算子新误差”

写残差子层为：

$$
h_{\ell+1}=h_\ell+f_\ell(h_\ell).
$$

对局部可微函数作一阶近似：

$$
\delta h_{\ell+1}
\approx
(I+J_\ell)\delta h_\ell+\eta_\ell.
$$

这里，$J_\ell\delta h_\ell$ 对应既有扰动的传播，$\eta_\ell$ 对应当前执行实现引入的额外差异。**误差不是必然逐层单调增长**：它可能放大、抵消，也可能在舍入后暂时消失。

Attention 打分的一阶扰动包含：

$$
\delta(QK_{\mathrm{attn}}^\top)
\approx
\delta Q\,K_{\mathrm{attn}}^\top
+
Q\,\delta K_{\mathrm{attn}}^\top.
$$

归一化分母、Value 内容和输出投影决定后续响应；不能仅凭长序列就推断误差一定单调增大。

### 3.4 从 logits 到 logprobs 的可检验上界

固定完整词表、目标 ID、temperature 与后处理。定义：

$$
g_y(z)=z_y-\operatorname{LSE}(z).
$$

若另一条路径使用 $z+\delta z$，则：

$$
\Delta g_y
=
\delta z_y-
\left[\operatorname{LSE}(z+\delta z)-\operatorname{LSE}(z)\right].
$$

由 LSE 的单调性有：

$$
\min_i \delta z_i
\le
\operatorname{LSE}(z+\delta z)-\operatorname{LSE}(z)
\le
\max_i \delta z_i.
$$

于是（针对数学函数以及统一高精度重新计算）：

$$
\boxed{
|\Delta g_y|
\le
\max_i\delta z_i-\min_i\delta z_i
\le
2\lVert\delta z\rVert_\infty
}
$$

若 logits 都除以 $\tau$，上界相应为 $2\lVert\delta z\rVert_\infty/\tau$。

两个重要推论：

- $g_y(z+c\mathbf 1)=g_y(z)$：公共平移被抵消。
- $\partial g_y/\partial z_i=\mathbf 1_{i=y}-p_i$：稳定 log-softmax **不会因为 $p_y$ 很小，就必然出现 $1/p_y$ 级的绝对敏感性**。

**排错判据**：若完整词表 logits 的 max-abs diff 只有 $10^{-5}$，而两侧用同一 FP64 log-softmax 重算后的目标 logprob diff 仍是 $10^{-2}$，应先怀疑词表、目标、位置、后处理或数据比较不一致。只比较目标词自身的 logit 不足以使用这个上界。[R8]

## 4. Megatron-Core / TE 的实际计算路径

### 4.1 TP：RowParallelLinear 与 ColumnParallelLinear

#### Row-parallel

MCore `RowParallelLinear.forward` 的基本模式是局部 GEMM 生成 `output_parallel`，再按配置：

- 普通 TP：`reduce_from_tensor_model_parallel_region`，通常对应 SUM AllReduce；
- TP+SP：`reduce_scatter_to_sequence_parallel_region`；
- 某些专家路径：返回局部结果，由外部 MoE dispatcher 继续处理。

`tensor_parallel/mappings.py` 的普通 reduce 对输入 Tensor 直接调用分布式归约；**不能假定它自动转为 FP32**。[R3]

TE 普通 Linear 路径会根据激活 dtype 选择 GEMM 输出 dtype，随后进行相关 row-parallel 归约。**是否真正落盘 BF16、FP16 或 FP32，应以当前代码版本和观测到的 Tensor 为准**。[R4]

将误差拆成三个可验证阶段：

1. **local GEMM**：分片让 $K$ 由 $K$ 变为 $K/P$，可能改变内部分块及 kernel；
2. **local output cast**：每个 partial output 落到低精度 Tensor；
3. **collective SUM**：归约的 dtype、算法、顺序和中间舍入。

不要通过“改 AllReduce 为 FP32 后 diff 未下降”，就否定 TP 分块的作用；差异可能在局部 GEMM 输出时已出现。

#### Column-parallel

Column-parallel forward 往往没有同样的跨 TP 部分和归约，但 GEMM 从 $[M,K]\times[K,N]$ 变为 $[M,K]\times[K,N/P]$。shape/layout 不同可能改变算法和累加组织，因此最早差异仍可能出现在 QKV projection 或 FC1，而非只能在 FC2/输出投影的 row-parallel 合并处。[R1][R3]

#### Vocab-parallel embedding

常见路径是每个有效 token 只有所属词表 rank 提供非零 embedding，之后合并。这个路径与多个非零 partial sums 的 row-linear 有本质差别。**若 embedding 即明显不同，优先查 embedding shard / vocab ID / checkpoint 映射**。[R3]

### 4.2 SP：Sequence Parallel 不是 hidden 维度的 LayerNorm 并行

SP 主要沿 sequence 维度分布相关激活，配合 TP 的 AllGather/ReduceScatter。它**不等于将单 token 的 LayerNorm/RMSNorm hidden 维统计分拆到各个 rank**。[R9]

重点比较：

- 固定 TP 时，关闭与开启 SP；
- row-parallel forward 使用的 AllReduce vs ReduceScatter；
- local token 数量、Tensor layout、融合和 overlap 改变；
- TE Userbuffers/通信融合路径是否被启用。

ReduceScatter 与 AllReduce 不保证跨路径位一致，但也**不能断言必然不同**。需捕获相同 partial inputs 做 collective replay。[R3][R4]

### 4.3 CP：Context Parallel 的 KV 通信与 Attention 合并

TE 实现中有 `p2p`、`all_gather`、`a2a` 和组合 CP 路径。它们可能对应不同的 attention 分块和计算执行方式，并非同一个 attention kernel 外面加不同通信包装。[R5]

对于 query 对不同 KV 块的计算，记局部输出及 log-sum-exp 为 $O_j,\lambda_j$，数学上有：

$$
\lambda_{\mathrm{global}}
=
\log\sum_j\exp(\lambda_j),
$$

$$
O=\sum_j\exp(\lambda_j-\lambda_{\mathrm{global}})O_j.
$$

TE CP 代码存在 LSE correction / output correction helpers。因此不能笼统说“CP 只是做了一次 BF16 attention-output AllReduce”；应分别定位：

~~~text
Q/K/V (global coordinates)
  -> 每个 KV 块的局部 attention
  -> (O_j, LSE_j)
  -> LSE 合并
  -> output correction / weighted combine
  -> 最终 dtype cast
~~~

即使 softmax/LSE 统计的一部分在 FP32 中执行，也不自动意味着**所有**局部输出与最终合并始终保留 FP32；要观测真实 Tensor dtype。[R5]

**另一个容易漏掉的来源**：CP 改变本地 sequence shape，可能让 attention *之前* 的 QKV GEMM 选用不同 kernel。不可从“只有 CP 改了”直接得出“差异首先来自 CP attention”。

#### CP token 对齐

某些 MCore CP 分片采用 zigzag 方式。若完整序列切成 $2P$ 个连续 chunk，rank $r$ 取第 $r$ 块和第 $2P-r-1$ 块。例如 $P=4$：

~~~text
rank 0: 0,7
rank 1: 1,6
rank 2: 2,5
rank 3: 3,4
~~~

直接按 rank concat hidden states 会制造伪差异。Packed sequence、具体 attention mask 和切分实现均需结合当前代码还原。[R10]

### 4.4 EP 与 ETP：三条相互独立的数值路径

#### (1) Router 分数可能触发离散专家选择改变

定义第 $k$ 与 $k+1$ 名路由分数差为：

$$
m = \text{score}_{(k)}-\text{score}_{(k+1)}.
$$

对于标准 top-k，当所有分数的绝对扰动不超过 $\epsilon$，如果 $m>2\epsilon$，入选专家集合不会改变（忽略额外 capacity/group 逻辑）。当 margin 很小时，即使 router logits 的整体误差很小，也可能发生专家集合变化。

必须保存：

~~~text
router_logits
selected_global_expert_ids
routing_weights
topk_margin
capacity / dropping masks（若启用）
~~~

组路由、score function、expert bias、token drop 都是独立语义变量。[R6]

#### (2) Expert GEMM 改变

EP 调整专家放置和本地专家/token 组织；ETP 调整专家内部的 Tensor Parallel。即使专家集合相同，local batch shape、grouped GEMM 路径、分片和 reduction 也可能不同。按 `(global_expert_id, sample_id, token_position)` 对齐，不能用 `(rank_id, local_expert_id)` 对齐。[R6]

#### (3) Combine dtype 与合并顺序

MCore 部分 dispatcher 路径将待归约 hidden states 转换为 `local_probs.dtype` / `self.probs.dtype` 再 ReduceScatter，之后转换回原始 dtype。**因此改变 router/probability dtype，可能同时改变 combine 的计算精度，而不仅是 router GEMM。**[R6]

融合/非融合 `unpermute` 及不同 dispatcher 算法亦可能改变数值路径。Router 的辅助负载均衡 loss 主要影响训练/反向；纯 forward、固定权重时不能把 aux-loss 系数直接当作输出差异来源。应区分影响路由的 bias/capacity/state。

### 4.5 DP 和 PP：有价值的控制组

**DP-only**：对同一份本地输入和相同权重，普通 dense forward 不需要跨 DP rank 合并输出；若出现差异，优先核查 microbatch、数据预处理、dropout、MoE capacity、量化 scaling 与 kernel shape，不能单凭“DP 变化”判定浮点误差。

**PP-only**：主要移动层和激活，点对点发送相同数值 Tensor 本身不执行加法。需要核查 `pipeline_dtype`、stage 边界、global layer ID、是否发生额外 cast，以及 pipeline/microbatch 变化。[R11]

### 4.6 Logits / Vocab-parallel CE 需单独调查

标准 MCore vocab-parallel cross entropy 大致是：

~~~text
logits.float()
 -> local max
 -> TP MAX
 -> subtract max
 -> gather selected target logit
 -> local exp / sum
 -> TP SUM
 -> log(sum_exp) - target_logit
~~~

因此**不能预设这条标准 CE 路径全在 BF16 中归一化**；跨 TP 的分区仍可能改变局部 exp 求和和 TP SUM 的计算组织。[R12]

需要单独识别实际使用的是 MCore native vocab-parallel CE、自定义 RL logprob、TE fused CE，还是其他算子。若代码中有原地操作，snapshot 必须在操作前 `detach().clone()`，只保存 Tensor 引用可能导致数据事后被改写。

还必须记录 padded vocabulary 和对 logits 的有效 mask。若两配置把不同的额外词表行纳入归一化，就已经不是单纯的浮点分块误差。

### 4.7 NCCL、cuBLAS、TE deterministic 的具体边界

- PyTorch deterministic 只约束其所覆盖的非确定性行为，不要求不同算法实现位级一致。[R1]
- MCore `training/determinism.py` 的确定性配置还要求关闭部分 fusion/overlap，并涉及 `NCCL_ALGO=Ring`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8` 等约束。**是否生效以及是否与当前 release 一致，须读取本地实现和运行时环境**。[R13]
- NCCL SUM 对 BF16 的具体舍入/运算取决于 NCCL 版本、协议和路径；不能自动假定所有设备端 BF16 SUM 都采用 FP32 全程累加。[R14]
- 限制 NCCL Ring 仅能约束一部分 collective 选择，不会抹平局部 GEMM $K/P$ 分块与 local output cast。
- FlashAttention 的 deterministic 选项主要控制 backward 的确定性行为，不能拿它直接解释跨 CP/TP 的 forward 差异。[R15]

## 5. 实验设计：从 1e-2 定位到具体算子

### 5.1 P0：建立“实验契约”

分别保存以下内容；**不要只保存启动脚本**。

| 文件 | 必须包含 |
|---|---|
| `manifest.json` | commit、PyTorch/MCore/TE/CUDA/NCCL、GPU 型号、world/group/rank、实际 DP/TP/PP/CP/EP/ETP/SP、microbatch、各 dtype、attention backend、fusion、overlap、quantization/FP8 状态 |
| `inputs.json` + 小规模 Tensor snapshot | 同一 sample ID、input IDs、labels、positions、packing、attention mask、有效 token mask、temperature、目标词和词表范围 |
| `weights_manifest.json` | canonical global weights 的 shape/dtype/hash、QKV 与 gate/up 映射、全局专家编号、embedding/LM head、RoPE、normalization、其他 state/buffers |

固定同一逻辑输入和权重；调试阶段不执行 optimizer update。保留原本一致的 forward 模式，独立记录 eval/no-grad 对照，避免模式改变引入新的 backend 差异。

**先做两项轻量断言**：

1. 对相同配置，检验指定 snapshot 是否真正逐位相等，而非打印 `0.0000`。
2. 对跨配置，确认**重组后的 canonical 权重**和相同预测位置/目标 label 精确对齐。

### 5.2 P0：归一化隔离实验

对小批量固定预测位置采集**完整词表 logits**，分别得到：

$$
g_y^{A,\mathrm{ref}}
=
z_y^A-\operatorname{LSE}_{\mathrm{FP64}}(z^A),
$$

$$
g_y^{B,\mathrm{ref}}
=
z_y^B-\operatorname{LSE}_{\mathrm{FP64}}(z^B).
$$

并与 $g_y^{A,\mathrm{native}}$、$g_y^{B,\mathrm{native}}$ 比较。

| 现象 | 初步解释 |
|---|---|
| FP64 reference 仍不同 | 差异已存在于 logits/模型主体 |
| reference 接近，native 差很大 | 原生归一化、后处理、低精度保存 |
| 完整 logits 一致，native 不同 | logits 之后的路径应优先调查 |
| 真实词表一致，扩大归一化域后变化 | padded vocab 或 mask 问题 |
| FP64 结果不满足第 3.4 节上界 | 逻辑坐标、目标或数据处理没有对齐 |

补充保存 `target_logit`、`LSE` 和 `target_logprob`，避免把分子和归一化分母混成一个 diff 指标。

### 5.3 P1：按语义边界而不是仅按 Python module 名比较

第一轮只采集：

~~~text
embedding
layer_i.attention_branch input/output
layer_i.mlp_or_moe_branch input/output
layer_i.final_hidden
final_norm
lm_head_logits
~~~

找到按执行顺序最早的异常边界，再深入：

**Dense attention / MLP**

~~~text
norm -> qkv projection -> q/k norm / RoPE
     -> attention core -> output projection local GEMM
     -> collective -> residual add

norm -> fc1 -> activation -> fc2 local GEMM
     -> collective -> residual add
~~~

**MoE**

~~~text
router logits/topk/weights
 -> dispatch before/after
 -> (global expert, original token) fc1/activation/fc2
 -> ETP reduction
 -> weighted combine before/after
~~~

必须区分 fused 模块内部的不同位置、bias 前后、partial output 与完整 output。普通 `nn.Module` hook 对融合 CUDA kernel 的内部阶段无能为力；需要 instrumentation、独立 kernel replay 或 profiler。

### 5.4 逻辑坐标的正确还原

任何两个 snapshot 比较前，都要有最少如下索引：

~~~text
(sample_id, global_token_position,
 absolute_layer_id, operator_semantics,
 global_feature_or_head_or_expert_id)
~~~

| 策略 | 还原规则 |
|---|---|
| TP | 按 QKV/head/MLP/vocab 的真实布局恢复，不要盲目 concat |
| SP | 重建原始 sequence 顺序 |
| CP | 逆 zigzag / packed-sequence 分配 |
| PP | 使用绝对 layer ID，而不是 stage 内序号 |
| EP | 使用全局 expert ID + 原始 token 位置 |
| ETP | 在 expert 内还原 feature shard，并区分归约前后 |

对于 replicated Tensor，先检验多个副本是否相同，比较时选择一个副本；**不要取平均掩盖差异**。

### 5.5 最关键的方法：相同输入的算子重放

如果只有：

$$
Y_A=f_A(X_A),\qquad Y_B=f_B(X_B),
$$

直接比较 $Y_A,Y_B$ 不能区分上游输入误差与本算子数值差异。

补充重放 $f_B(X_A)$，获得恒等分解：

$$
\boxed{
Y_B-Y_A =
\underbrace{f_B(X_B)-f_B(X_A)}_{\text{上游误差在 B 路径传播}}
+
\underbrace{f_B(X_A)-f_A(X_A)}_{\text{固定输入后的实现差异}}.
}
$$

此实验必须固定模型权重、mask、状态、dtype 规则；记录 replay 采用的 shape/kernel。过度缩小输入可能切换 kernel，不能直接代表原始大 batch 实现。

### 5.6 TP / ETP 的 operator replay 矩阵

针对捕获的 row-parallel local GEMM partials：

| Case | local partials | merge | 回答什么 |
|---|---|---|---|
| R1 | 原生低精度输出 | 原生 SUM | 基线复现 |
| R2 | 与 R1 **完全相同**的已舍入 partials | FP32 SUM 再转回 | collective 运算精度贡献 |
| R3 | 与 R1 相同 partials | gather 后固定顺序 FP64 SUM | 和高精度合并差多少 |
| R4 | 尽可能保留高精度 local GEMM 输出 | FP32 merge | partial cast/局部计算贡献 |
| R5 | 完整逻辑 $X,W$ | FP64 小范围参考 GEMM | 各路径相对 reference 的误差 |

注意两种实验不同：

- BF16 inputs + 原 GEMM 的 FP32 output；
- 将 BF16 inputs/weights 转成 FP32 再做 GEMM。

后者可能改变计算路径，不可混称“只改变输出精度”。

如果 R2 未明显改善，只能说明该干预无效，**不能推出 TP 不是根因**；问题可能早于 collective。

### 5.7 CP 专项：固定 Q/K/V 后检查合并

~~~text
fix global Q,K,V
 -> CP=1 reference
 -> CP path p2p / all_gather / a2a（仅比较受支持的组合）
 -> compare each block O_j / LSE_j
 -> compare corrected LSE
 -> compare weighted output and cast
~~~

逐步检查：全局 KV 覆盖、causal/padding mask、block 顺序、输出 dtype。某个 CP backend 更接近参考不等于另外一个必定是 bug，必须落到具体数值或语义差异。

### 5.8 MoE 专项：先自然路由，再固定路由

1. 自然执行，统计选中专家集合不一致比例、router margin 分布、routing weight diff。
2. 诊断性地冻结 expert IDs 与 routing weights，保持原始逻辑 token 顺序。
3. 对比 expert GEMM、ETP reduction、combine。
4. 若冻结路由后显著收敛，说明离散路由分支是重要贡献；仍有差异则继续追踪 expert/merge。

固定路由仅用于归因，不作为生产修复建议。

### 5.9 最低指标集与统计边界

| 类别 | 指标 |
|---|---|
| 精确 | shape/dtype、逻辑 ID、位级一致、NaN/Inf 数量 |
| 绝对 | MaxAbs、MAE、RMSE、P95/P99 |
| 相对 | Relative L2、reference RMS、零值附近行为 |
| 传播 | signed mean、非零差异比例、最早异常层 |
| MoE | expert-set mismatch、top-k weight/margin diff |
| 输出 | target logit、LSE、target logprob、argmax 变化 |

差值和累计统计宜在 FP32/FP64 完成，不能低精度相减后才转 FP32。标记指标是 per-token 还是序列 sum/mean，及采样分位数还是完整数据分位数。

## 6. 工程建议与性能代价

| 改进方式 | 可能改善 | 限制与代价 |
|---|---|---|
| 统一 FP32 logprob 计算与保存 | 归一化和低精度存储差异 | 标准 MCore CE 已部分在 FP32，未必有效 |
| SUM collective 使用 FP32 payload | 归约舍入 | BF16→FP32 payload 字节数翻倍；总时延需实测 |
| local GEMM FP32 输出后再合并 | 提前舍入的 partial output | 缓冲区更大，可能需改 TE/kernel |
| FP32 residual | 残差路径累积舍入 | 不修复前面的 GEMM/collective |
| FP32 router/probs | 路由和部分 combine 精度 | 可能同时改变多个阶段，需实验拆分 |
| 控制 attention backend / fusion | 减少未受控路径变化 | 并不保证跨 shape/CP 位一致 |
| CP output correction 更高精度 | 分块 LSE/output 合并 | 无法恢复每块 attention 已丢失的精度 |
| 禁用部分 overlap/fusion | 更易观测与归因 | 降低吞吐，增加 kernel/通信等待 |
| FP64 reference / replay | 研究误差来源 | 通常仅适于小规模诊断 |

### 6.1 不应误用的改进建议

- **FP32 AllReduce 不是万能药**：需要先确定 partial outputs 有没有提前舍入。
- **NCCL Ring 不是跨 TP 位级一致承诺**：它不控制 local GEMM 分块。[R7][R14]
- **设置 PyTorch deterministic 不是控制所有 TE/NCCL kernel 的统一开关**：应核对真正生效的 backend。[R1][R13]
- **DP 梯度 FP32 / optimizer state FP32 不是固定权重 forward 的直接修复**。
- **Tensor/sequence parallel 并不天然精度更差**：不能只凭 $P$ 更大就推断数值误差单调增大；参照正确的高精度结果比较各自误差。
- **不要用一个通用 BF16 tolerance 代替业务判据**。

如果对一个目标词，$\Delta g_y=0.01$，则概率比例为：

$$
\frac{p_y^B}{p_y^A}=e^{0.01}\approx1.01005.
$$

这是约 $1\%$ 的**该位置条件概率比例**差异，不等于最终训练质量变化 $1\%$。对于 RL/OPD，应进一步评估 sequence logprob sum、policy ratio、KL 指标及尾部 token 的影响；不可假定 token 误差独立抵消。

## 7. 结论与待验证问题

### 7.1 已确认的机制（源码/数学层面）

- TP/ETP 可改变 GEMM 分块、局部输出 dtype 与 SUM 边界。
- CP 可改变 Attention 分块和 LSE/output correction 的计算流程。
- EP/MoE 包含独立的 router 离散选择、expert GEMM 和 combine。
- SP 会改变分片/collective 与局部 shape；纯数据移动本身不应被当作求和。
- 稳定 log-softmax 对固定归一化词表的 logits 扰动有明确绝对上界。
- 同配置位级复现不推出跨并行位级一致。

### 7.2 仍待在实际运行中验证

- $10^{-2}$ 是哪个统计指标，最早在哪个全局 token、哪层、哪个 operator 出现？
- 当前使用的 MCore/TE/NCCL 与实际 kernel、dtype、融合开关是什么？
- 重组后的权重、词表、position、mask 和目标 ID 是否精确对齐？
- 相同输入下，local GEMM 输出、高精度合并、CP LSE correction 各贡献多大？
- MoE 是否发生 expert-set 变化？是否仅集中在 margin 很小的 token？
- 这些变化会不会影响 RL/OPD 训练目标、KL 和最终质量？

**最有价值的产物是一条具体归因记录**：在固定输入与权重后，最早发生差异的算子是哪个，差异来自 partial computation / cast / SUM / attention merge / route 的哪一步，哪项精度干预使误差收敛，以及相应的性能代价。没有这些证据，不应该将 $10^{-2}$ 直接判为“正常浮点误差”。

## 8. RL / Post-training 领域相关研究与系统性影响

> [!note]
> 本章补充截至 **2026-10-13** 可查的 RL/Post-training 论文与工程报告。研究结论分为：**论文/项目在其给定实验条件下报告的结果**、**可以由定义和公式直接推导的结论**、以及**对当前 Megatron 跨并行实验的待验证启发**。不能把 Train–Rollout Mismatch、跨 TP 数值不一致和真正的异步 Off-policy 漂移直接当成同一问题。

### 8.1 为什么 RL 比普通 SFT 更关心 Forward 数值一致性？

SFT 的 token-level cross entropy 主要由训练引擎计算；而典型 GRPO/PPO/RLVR 把**采样行为**和**梯度更新**拆到两套执行系统：

~~~text
rollout engine（SGLang / vLLM 等）
  -> 按某一权重版本生成 token，保存 token IDs 和 old logprobs
  -> training engine（Megatron / FSDP 等）
  -> 对相同 token 重算当前 logprobs、构造 policy ratio、clip 和 loss
  -> 梯度更新；随后同步新权重
~~~

因此即使权重完全相同，也可能有两个**不同的实际概率分布**：

$$
\pi_{\theta}^{\mathrm{rollout}}(a_t\mid s_t)
\ne
\pi_{\theta}^{\mathrm{train}}(a_t\mid s_t).
$$

这就是 **Training-Inference Mismatch（TIM）** 的一种来源；它不是随机 seed 不同的同义词，也不等于模型权重版本已经过期。尤其在不同 TP、CP、EP、batch shape、FP8 scaling 和 kernel 实现之间，TIM 可能来自本报告第 4 节讨论的真实计算路径差异。[R16][R18]

**三种应分开记录的偏差：**

| 偏差 | 两侧权重版本 | 核心原因 | 应先控制的变量 |
|---|---|---|---|
| 跨并行布局 Forward Diff | 相同 | TP/CP/EP/ETP 等改变局部算子与合并 | canonical weights、输入、layout、dtype、kernel |
| TIM（训推数值不一致） | 相同 | rollout 与 trainer 的模型/kernel/量化实现不同 | 相同 token、prefix、mask、词表和精度语义 |
| Policy Staleness / Off-policy Drift | 不同 | 异步 rollout、PPO/GRPO 多次 mini-step 更新 | 权重版本、生成时间、optimizer step、replay age |

**重要边界**：纯粹的 teacher–student KL 或 OPD 中教师与学生本来不同的分布，是**预期中的建模差异**，不能直接称为 TIM。只有同一个逻辑策略、同一个权重快照在训练/推理路径给出不同数值时，才是本章关心的系统级 mismatch。

### 8.2 Mismatch 如何进入 PPO / GRPO 的 Importance Ratio？

设 rollout 使用旧权重 $\theta_{\mathrm{old}}$ 生成 $a_t$，其上下文为 $s_t$。在同一权重快照下定义 token-level 数值差：

$$
\delta_t
=
\log\pi^{\mathrm{train}}_{\theta_{\mathrm{old}}}(a_t\mid s_t)
-
\log\pi^{\mathrm{rollout}}_{\theta_{\mathrm{old}}}(a_t\mid s_t).
$$

在 trainer 上，当前与旧权重之间**真正希望用于更新的** logprob 变化可以写为：

$$
\Delta_t^{\mathrm{update}}
=
\log\pi^{\mathrm{train}}_{\theta}(a_t\mid s_t)
-
\log\pi^{\mathrm{train}}_{\theta_{\mathrm{old}}}(a_t\mid s_t).
$$

若 denominator 使用实际行为策略的 rollout logprob，则：

$$
\rho_t^{\mathrm{beh}}
=
\frac{\pi^{\mathrm{train}}_{\theta}(a_t\mid s_t)}
{\pi^{\mathrm{rollout}}_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
=
\exp\left(\Delta_t^{\mathrm{update}}+\delta_t\right).
$$

若把旧 logprob 在 trainer 重算作为 denominator，得到：

$$
\rho_t^{\mathrm{train}}
=
\frac{\pi^{\mathrm{train}}_{\theta}(a_t\mid s_t)}
{\pi^{\mathrm{train}}_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
=
\exp\left(\Delta_t^{\mathrm{update}}\right).
$$

所以有精确关系：

$$
\boxed{
\rho_t^{\mathrm{beh}}
=
\rho_t^{\mathrm{train}}\,e^{\delta_t}.
}
$$

**关键区分**：

- 若还未执行任何 optimizer update，且当前权重与 rollout 权重确实相同，理论上没有 policy-update drift；此时 $\rho_t^{\mathrm{train}}=1$，但 $\rho_t^{\mathrm{beh}}=e^{\delta_t}$ 仍可能偏离 1。
- **执行过 PPO/GRPO mini-step 后，ratio 不等于 1 本来就是正常现象**；不能把所有 $r\ne1$ 都诊断成 TIM。
- 使用 trainer-recomputed old logprob 可使第一种 ratio 接近 1，但此时 denominator 不再是实际采样所用概率；并不等于已消除两套策略的数值差异。
- 直接使用 rollout logprob 保留了真实 denominator，但 trainer 的当前分布和更新梯度可能仍不能完全反映 rollout engine 部署后的分布。VeXact 的消融实验专门研究了这两类失败路径。[R18]

以 PPO 风格 surrogate 为例：

$$
L_t^{\mathrm{clip}}
=
\min\left(
\rho_t\widehat A_t,\;
\operatorname{clip}(\rho_t,1-\varepsilon_{\mathrm{clip}},1+\varepsilon_{\mathrm{clip}})
\widehat A_t
\right).
$$

若 $\delta_t=0.01$，则 $e^{\delta_t}\approx1.01005$；这只是约 1% 的**单 token 比率乘性扰动**，**不必然**触发 PPO clipping。但当正常参数更新已接近 clipping 边界、负优势 token 分布不均、MoE top-k 出现离散跳变，或按序列合并误差时，其训练影响可能与平均 $|\delta_t|$ 很不相称。[R18][R23]

还有一个单独的语义前提：若 rollout 采样经过 temperature、top-p/top-k 或其他过滤，**参与实际采样的行为分布**与 raw model logprobs 可能不同。必须先明确对比的是 raw policy，还是已处理的 sampling policy，不能在公式中混用。

### 8.3 与本问题最直接相关的系统研究

以下优先选取直接研究**计算路径数值对齐**的工作，而不是泛泛讨论 RL 稳定性。

| 工作 | 直接研究对象 | 主要结果与边界 | 对当前 Megatron 实验的价值 |
|---|---|---|---|
| **TBIK**，ICML 2026 | **不同 TP size、batch size 的数值一致性** | BIO+TBIK 在论文支持的 vLLM/FSDP 配置中实现位级一致；不是任意 MCore 并行组合的保证。[R16] | 优先研究分片内部与跨卡 **Reduction Tree** 如何统一 |
| **RL-Kernel**，2026-09 | Megatron train ↔ vLLM rollout 的严格 logprob parity | Qwen3-8B BF16、Megatron TP4/CP2 的 CUDA/ROCm strict path，团队报告 200 steps 内 mismatch 指标为 0。[R17] | 最贴近 Megatron；学习逐算子的数值执行契约与 provenance |
| **VeXact**，2026-05 / v2 2026-09 | Zero-TIM 对照组及 mismatch 对 RL 的影响 | FSDP 与 VeXact 统一 kernel/模型路径；在论文设置中揭示 TIM 独立导致的训练不稳定。[R18] | 引入 zero-mismatch baseline，区分误差根因与优化后果 |
| **NeMo-RL DTensor TP 指南** | TP plan、row-wise SUM 与 batch-variant kernels | 官方 nightly 文档记录：某些实验中对齐 microbatch 后 ratio 恢复；row-wise TP 是另一个贡献因素。[R21] | 直接增加 TP plan 和 microbatch-shape 对照 |

#### TBIK：跨 TP 的计算树一致性

Zhang 等在 ICML 2026 提出 **Tree-Based Invariant Kernels（TBIK）**，让 GPU 内 GEMM 与 GPU 间归约采用跨 TP 配置一致的计算树；实现基于 Triton，并结合 **Batch-Invariant Operations（BIO）** 对齐 batch 相关路径。[R16]

原论文的 AIME'24 Table 2（L40S；数值单位为 $10^{-3}$）：

| 模型 | 原生 BF16 | BIO | BIO + TBIK |
|---|---:|---:|---:|
| Qwen3-8B | 10.01 | 9.10 | **0** |
| Mistral-7B-Instruct | 17.37 | 14.10 | **0** |
| Llama-3.1-8B-Instruct | 26.48 | 27.54 | **0** |
| Qwen3-32B | 8.01 | 7.79 | **0** |

这里的指标是作者定义的 **Average Maximum Probability Divergence**，**不是目标 token logprob MAE**。论文还报告，在特定 Qwen3-8B / H20 / TP4 设置下，BIO+TBIK 相比原生 BF16 的端到端延迟开销约 **22%–63%**。不能将这些数字直接外推为你的 Megatron 实验误差预算或性能代价。

**直接启发**：对一个 row-parallel GEMM，不能只统一 NCCL `NCCL_ALGO`；还要对齐 local GEMM 的累加树、partial output cast、跨 GPU 合并树和输出舍入。如果要求 bitwise invariance，需要的是**端到端算术执行协议**，而不是简单提高 SUM dtype。

#### RL-Kernel：Megatron × vLLM 的执行契约

RL-Kernel 将数值一致性要求具体化为：相同逻辑输入与状态、相同 reduction domain、分片与 merge tree、精度/舍入边界，以及 `exp`/`log`/`rsqrt`/`SiLU` 等数值 primitive 的实现。[R17]

其技术报告列出的涉及范围包括 RMSNorm、Attention、GEMM、SwiGLU、linear logprob 与 distributed collectives；并使用训练/rollout 各自采用 Production（P）还是 Aligned（R）实现的 **P/P、P/R、R/P、R/R** 对照，追踪具体是哪个算子带来差异。

团队报告的两组代表性 end-to-end strict path：

| GPU | 训练布局 | Rollout | 模型 | 结果 |
|---|---|---|---|---|
| 8×H100（CUDA） | Megatron TP4/CP2/PP1 | 2×vLLM TP4 | Qwen3-8B BF16 | 200 steps，`mismatch_count=0`、`max_abs_diff=0` |
| 8×MI300X（ROCm） | Megatron TP4/CP2/PP1 | 2×vLLM TP4 | Qwen3-8B BF16 | 200 steps，两个指标同为 0 |

**边界**：这是项目团队对其版本、固定逻辑输入、受控 backend / fallback 的**自报告结果**；不能直接证明所有其他 GPU、模型、MoE、CP 实现或动态批处理路径也具有位级一致性。

特别值得借鉴的是它的 **execution provenance**：仅在配置中打开 `strict` 并不构成证明，还要确认实际 kernel、通信路径、CUDA/HIP Graph、fallback 状态，且比较对象确实相同。

#### NeMo-RL：同 TP 下 microbatch 也能改变 logprob

NeMo-RL DTensor TP 的官方指南指出两个独立因素：[R21]

1. `train_micro_batch_size` 与 `logprob_batch_size` 不同，导致 batch-dependent GEMM/归约/融合路径变化；该文档所述实验中对齐后 `probs_ratio` 恢复到 1。
2. Row-wise TP 需要跨卡 partial sum，与 column-wise 分片的归约结构不同；指南提供了调整 TP plan 的思路。

这不是对 Megatron 的逐位保证，且 nightly 文档可能持续修改。**工程建议**：在测试跨 TP 数值一致性前，先保证不同配置的 local token/batch shape 和对应 kernel 信息可追溯，避免把 batch shape 变化归因给 TP 算术本身。

### 8.4 TIM 是否会影响 RL 的真实优化？VeXact 的诊断意义

Zhong 等的 **Diagnosing Training-Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference**（arXiv:2605.14220v2，2026-09-29）构建 VeXact：使用和 FSDP 训练侧对齐的模型/kernel、batch-invariant 路径，尽量隔离 TIM 对 REINFORCE/GRPO 稳定性的影响。[R18]

其观察包括：

- Dense Qwen3-1.7B 和 MoE Qwen3-30B-A3B 的特定 REINFORCE 实验中，非精确 rollout 路径出现了 reward/gradient 不稳定，而 zero-mismatch baseline 更稳定。
- 在一段报告的 Qwen3-8B token 轨迹样例中，出现绝对 logprob 差 $0.133$；这是**个别实例**，不能代表总体均值。
- PPO/GRPO 中直接拿 trainer-recomputed old logprob 当 denominator，或完全绕过 recomputation，分别有不同偏差；PPO ratio 之外还需检查由哪个执行路径计算的梯度。
- 论文发现平均 KL 指标未必能第一时间反映按优势符号区分的更新方向偏差。

**实验上的最重要启示**：应区分 “`train_logp - rollout_logp` 是否不同” 与 “这种不同是否**因果性**改变梯度、reward 或训练质量”。先建立可控的 zero-mismatch baseline，再用单变量干预隔离 TP / CP / EP，不能只比较两条自由运行的 reward 曲线。

### 8.5 MoE 专项：Rollout Routing Replay（R3）

Ma 等的 **Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers**（ICML 2026）提出 **Rollout Routing Replay（R3）**。[R19]

问题是：即使权重一致，router 分数出现微小差异，也可能在 top-k 的边界切换专家集合，使训练和 rollout 计算不再位于同一个稀疏分支。R3 记录 rollout 的 routing 分布/选择信息并在训练侧重放，以降低训练与推理路由不一致所带来的额外偏差。作者报告 policy KL 差异降低及训练稳定性改善。

这里必须分开两类“修复”：

- **Routing replay**：对齐走哪些专家，优先修复离散路由变化。
- **Numerical kernel alignment**：对齐每个专家的 GEMM、ETP reduce、combine、attention 和最后 logprob。

R3 不承诺后者全部位级相等。因此若你在 TP/EP/ETP 改变后遇到 $10^{-2}$ diff，建议先分析不同 token 的 expert-set mismatch 与 router margin，再用固定专家 ID/权重的 replay 单独测专家执行路径。

### 8.6 算法补偿：IcePop、CIS 与 FP8 Calibrated Clipping

系统级对齐通常成本较高，RL 研究也有**允许 mismatch 存在，降低其训练伤害**的路线。

| 工作 | 主要机制 | 不能解决什么 |
|---|---|---|
| **IcePop**（Ring-1T / AReaL） | 对重要性比率超出给定区间的 token 进行 masking / clipping，降低异常 token 对更新的影响。[R20] | 不会令 train/rollout hidden states、logits、router bitwise 一致 |
| **Calibrated Importance Sampling（CIS）**，2026-09 | 研究 log-odds displacement 与 token confidence，设计 confidence-aware 截断以控制权重矩并权衡偏差。[R22] | 是采样/更新权重校正，不是 GEMM/CP 的数值修复 |
| **Calibrated Clipping for FP8 RL**，2026-09 | 研究 FP8 量化噪声对 importance ratio、负优势 token clipping 的影响，按高精度参考分布校准剪裁范围。[R23] | 不提供跨并行位级一致保证 |

IcePop 的 AReaL 实现可通过 `use_decoupled_loss` 与 token-level ratio rejection 配置启用；超阈值 token 不贡献对应 loss。是否值得部署，要看 rejection 比例、有效梯度 token 数，以及是否伤害长尾样本。[R20]

这些措施是**容错/补偿层**，不应拿来证明 underlying forward kernel 已数值一致。也不能简单声称它们“恢复真正 on-policy”，因为过滤与裁剪本身可能引入目标函数偏差。

### 8.7 应在本地 Megatron 实验中增加哪些观测？

**第一层：相同权重、固定 token 的纯数值对齐**（本报告第 5 节的延伸）。

| 实验 | 固定变量 | 只改变 | 主要指标 |
|---|---|---|---|
| E0 | 所有输入与权重 | 无，同配置重复 | 原始 dtype 的 bitwise parity |
| E1 | 权重、token、batch shape | TP=1/2/4/8 | earliest divergent operator、partial GEMM、collective |
| E2 | TP、权重、逻辑 tokens | CP=1/2/4（仅受支持组合） | QKV、block LSE/output correction |
| E3 | TP/CP、权重、token | EP/ETP、MoE dispatcher | expert-set diff、expert output、combine |
| E4 | 完整 logits | native logprob vs 统一 FP64 log-softmax | target-logit / LSE / logprob 的误差分解 |
| E5 | 同一语义输入 | training/rollout Production/Aligned 2×2 | 由哪一端的算子实现造成差异 |

**第二层：控制变量的 RL 敏感性实验**：

1. 在固定权重版本下保存完整 rollout 样本，严格记录行为策略版本、token IDs、mask、原生 rollout logprobs 与 trainer 重算 logprobs。
2. 分开计算 $\delta_t$、$\rho_t^{\mathrm{beh}}$、$\rho_t^{\mathrm{train}}$；按层、token 位置、response length、router margin、优势符号、是否接近 clipping 边界分组。
3. 记录 `p50/p95/p99/max |delta_logp|`、异常 token 比例、clip fraction、expert-set mismatch、update 前后 ratio，并同时比较完整 logits 和 LSE。
4. 在**同一固定样本**上，比较原生 kernel、统一高精度合并、固定 routing 与完全对齐 kernel 的差别；优先定位最早引入误差的 operator。
5. 在独立受控训练中比较 zero-mismatch/reference 与 native mismatch，记录梯度差、KL、有效样本数、reward 和验证指标。不同动态 rollout 轨迹的 reward 对比不能单独当作算子级因果证据。

**要特别防止的假阳性**：不同 rollout weight version、position / attention mask、KV cache 逻辑语义、tokenizer/vocab mapping、packing 以及 raw/temperature/top-p 后处理，本质上让两边比较的不是同一个条件概率。应先通过 `comparable=true` 的数据契约，否则所有后续差异统计都失去解释力。

### 8.8 研究优先级与未解决问题

基于现有论文和当前实验现象，建议：

1. **优先复现 TBIK 的跨 TP reduction-tree 论证**：在 Megatron BF16 row-linear 和 column-linear 上，以最小 operator replay 比较 local GEMM partial、cast 和 collective；这是与“改 TP 后 diff 约 $10^{-2}$”最直接的对应关系。
2. **借鉴 RL-Kernel 的 numerical contract / execution provenance**：把 reduction domain、分块、merge tree、输入/内部/输出精度、数值 primitive 和 fallback 写入 snapshot 元数据。
3. **参考 VeXact 做 zero-mismatch control**：在真实 GRPO/PPO 训练前先问“相同 $\theta$ 的 $\delta_t$ 是否为零”，随后才研究 clipped ratio 与训练稳定性。
4. **MoE 优先追加 R3 风格 routing replay 对照**：先隔离是否更换专家，再测 expert GEMM、ETP 和 combine。
5. **IcePop/CIS/Calibrated Clipping 作为第二阶段的算法补偿研究**：在确认 mismatch 来源和规模后，量化偏差、丢弃 token 比例与训练收敛，而不是以 mask 掩盖根因。

**尚无通用结论的部分**：TBIK 并未覆盖任意 Megatron CP/EP/ETP 路径；RL-Kernel 的严格一致结果有特定模型、硬件和执行契约约束；VeXact 的论文主线是 FSDP/train–rollout；NeMo-RL nightly 是工程案例。它们共同支持“跨并行数值路径值得作为 RL 一等公民来处理”，但**仍不足以断言你目前的 $10^{-2}$ diff 来自哪一层，更不足以预测它一定导致 RL 崩溃**。

本章的长期研究方向可概括为：

> **Megatron-Core 多维并行下的 Forward Numerical Consistency：从 Reduction / Routing / Attention 逐算子归因，到 PPO/GRPO/OPD 的训练目标敏感性。**

## 9. References

以下链接优先固定已讨论的源码 tag；网页文档可能继续更新。**链接用于溯源，不表示本文已在本地 GPU 逐一实验验证。**

- **[R1]** PyTorch, [Numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html), [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- **[R2]** NVIDIA, [cuBLAS Documentation — Results Reproducibility](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).
- **[R3]** NVIDIA Megatron-Core `core_v0.19.2`, [tensor_parallel/layers.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/tensor_parallel/layers.py), [mappings.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/tensor_parallel/mappings.py).
- **[R4]** NVIDIA Transformer Engine `v2.18`, [pytorch/module/linear.py](https://github.com/NVIDIA/TransformerEngine/blob/v2.18/transformer_engine/pytorch/module/linear.py); MCore [extensions/transformer_engine.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/extensions/transformer_engine.py).
- **[R5]** NVIDIA Transformer Engine `v2.18`, [context_parallel.py](https://github.com/NVIDIA/TransformerEngine/blob/v2.18/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py), [Attention backend & debugging](https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html).
- **[R6]** NVIDIA Megatron-Core `core_v0.19.2`, [moe/router.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/transformer/moe/router.py), [experts.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/transformer/moe/experts.py), [token_dispatcher.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/transformer/moe/token_dispatcher.py), [moe_utils.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/transformer/moe/moe_utils.py).
- **[R7]** NVIDIA, [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) and [Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
- **[R8]** Blanchard, Higham, Higham, [Accurately computing the log-sum-exp and softmax functions](https://academic.oup.com/imajna/article/41/4/2311/5893596), IMA Journal of Numerical Analysis.
- **[R9]** Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198).
- **[R10]** NVIDIA Megatron-Core `core_v0.19.2`, [core/utils.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/utils.py).
- **[R11]** NVIDIA Megatron-Core `core_v0.19.2`, [pipeline_parallel/p2p_communication.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/pipeline_parallel/p2p_communication.py).
- **[R12]** NVIDIA Megatron-Core `core_v0.19.2`, [tensor_parallel/cross_entropy.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/tensor_parallel/cross_entropy.py), [fusions/fused_cross_entropy.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/fusions/fused_cross_entropy.py).
- **[R13]** NVIDIA Megatron-Core `core_v0.19.2`, [training/determinism.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/training/determinism.py), [training/arguments.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/training/arguments.py).
- **[R14]** NVIDIA NCCL `v2.28.9-1`, [device/reduce_kernel.h](https://github.com/NVIDIA/nccl/blob/v2.28.9-1/src/device/reduce_kernel.h)（特定版本设备端归约实现；不外推到所有运行环境）。
- **[R15]** Dao-AILab, [FlashAttention README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md); Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135).

- **[R16]** Zhang et al., [Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch](https://proceedings.mlr.press/v306/zhang26ag.html), ICML 2026; [arXiv HTML, v2](https://arxiv.org/html/2511.17826)（TBIK，跨 TP/BIO，论文 Table 2 与延迟对照）。
- **[R17]** RL-Kernel Team, [RL-Kernel v0.1.0: Train–Rollout Bitwise Consistency](https://rl-align.github.io/RL-Kernel/blog/2026-09-18-rl-kernel-amd-vime-qwen3-8b-train-rollout-bitwise-consistency/) (2026-09-18); [项目仓库](https://github.com/RL-Align/RL-Kernel)（Megatron/vLLM strict path、execution provenance、200-step CUDA/ROCm 自报告）。
- **[R18]** Zhong et al., [Diagnosing Training-Inference Mismatch in LLM Reinforcement Learning via a Zero-Mismatch Reference](https://arxiv.org/html/2605.14220v2), arXiv:2605.14220v2 (updated 2026-09-29); [VeXact](https://github.com/verl-project/vexact).
- **[R19]** Ma et al., [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://proceedings.mlr.press/v306/ma26aw.html), ICML 2026; [arXiv:2510.11370](https://arxiv.org/abs/2510.11370)（Rollout Routing Replay / R3）。
- **[R20]** Ling Team, [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://arxiv.org/abs/2510.18855) (2025); AReaL [GRPO Series: IcePop](https://github.com/areal-project/AReaL/blob/main/docs/en/algorithms/grpo_series.md).
- **[R21]** NVIDIA NeMo-RL, [DTensor Tensor Parallel Accuracy Issue](https://docs.nvidia.com/nemo/rl/nightly/guides/dtensor-tp-accuracy.html)（**nightly 文档，可能变动**；microbatch 与 row-wise TP 诊断）。
- **[R22]** Yu et al., [Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It](https://arxiv.org/abs/2609.32444) (2026-09); [CIS project](https://kzhao5.github.io/CIS-website/)（Calibrated Importance Sampling）。
- **[R23]** Chen et al., [Towards Full Pipeline FP8 Reinforcement Learning for LLMs](https://arxiv.org/abs/2609.22870) (2026-09)（FP8 importance-ratio 噪声与 Calibrated Clipping）。

---

**后续实测记录建议**：独立存放在 `50_Experiments/`，按本仓库实验笔记规范保存环境、输入、关键快照统计、首个差异边界、精度干预及性能结果；待确认后再将结论提炼到 `20_Knowledge/`。
