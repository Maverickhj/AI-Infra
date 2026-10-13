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

## 8. References

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

---

**后续实测记录建议**：独立存放在 `50_Experiments/`，按本仓库实验笔记规范保存环境、输入、关键快照统计、首个差异边界、精度干预及性能结果；待确认后再将结论提炼到 `20_Knowledge/`。
