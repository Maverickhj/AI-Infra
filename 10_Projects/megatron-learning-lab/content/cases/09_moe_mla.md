---
type: knowledge
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# MoE / MLA：沿完整模型追踪一条 token

从 authored messages、单次 shift 和 assistant mask 进入：embedding → 两层 attention/residual → FFN norm → dense 或 MoE/residual → final norm → untied head → masked CE。浏览器和 CPU 都复用 G03 公共 decoder 骨架，MoE 替换 FFN，MLA 替换 attention。R1-Distill-Qwen 保持 GQA，不进入 MLA 演算。

全尺寸模型图来自既有 config 档案，部分 revision 未锁定，不是已加载权重。下面三个确定性小模型均为 architecture-scaled/authored，$V=27,H=8,L=2$。没有执行 Bridge、NCCL、Flash MLA 或 GPU。

## 独立数学符号表

| 符号 | 定义与 shape / 单位 |
|---|---|
| $B,S,N$ | batch、物理长度、非 padding token 数；本演算 $B=1$ |
| $H,V,L$ | hidden width、词表大小、decoder 层数 |
| $E,k,F,E_s$ | routed 专家数、每 token top-k、单专家 FFN 宽度、shared 专家数 |
| $x_t,z_{t,e}$ | FFN norm 后 hidden $[H]$；router logits 整体 $[S,E]$ |
| $s_{t,e},w_{t,e},b_e,\gamma$ | 原始分数、组合权重、选择偏置、routed 缩放 |
| $\mathcal T_t,f_e,P_e,\alpha$ | 所选专家集合、辅助路由频率、平均概率、辅助系数 |
| $n_e,\eta_b$ | 实际派发计数、bias 更新幅度 |
| $W_g,W_u,W_d$ | 存储 $[F,H],[F,H],[H,F]$；行向量右乘其转置 |
| $n_h,d_n,d_r,d_v,r_{kv},r_q$ | MLA 头数、内容 QK/RoPE/V 每头维度、KV/Q latent rank |
| $c_t,q_h^N,q_h^R,k_t^R$ | normalized KV latent $[r_{kv}]$、内容 Q、旋转 Q、共享 positional K |
| $U_h^K,U_h^V$ | KV up 切片，存储 $[d_n,r_{kv}]$、$[d_v,r_{kv}]$ |
| $a_{h,t,s},u_{h,t}$ | causal attention 概率、weighted latent sum $[r_{kv}]$ |
| $\epsilon,\sigma,R_t$ | RMSNorm epsilon、sigmoid、position $t$ 的 RoPE 旋转 |
| EP, ETP, EDP | 专家分布、单专家张量分片、专家数据副本 |

## 三种模型的实际分支

| 模型 | 全尺寸结构档案 | 两层 authored 参考 |
|---|---|---|
| Qwen3 MoE | 48 层 MoE，128 routed/top-8，无 shared，GQA + QK norm | 两层 MoE，4 routed/top-2，$F=4$，复用 G03 GQA |
| DeepSeek-V2-Lite | 27 层，首层 dense，64 routed/top-6，2 shared，直接 Q | L0 dense/L1 MoE，2 shared，直接 $[8]\to[2,4]$ Q |
| DeepSeek-V3 | 61 层，前三层 dense，256 routed/top-8，1 shared，低秩 Q | L0 dense/L1 MoE，1 shared，Q down $8\to4$、norm、up $4\to[2,4]$ |

两层参考保留 dense→MoE 边界，不能把 V3 的前三层说成一层。页面同时展示两种分布。微型 MLA 为 $n_h=2,d_n=d_r=d_v=2,r_{kv}=3$；V3 的 $r_q=4$，V2-Lite 没有 Q latent norm。所有模型都有完整 head/CE，不是孤立专家 MLP。

## Router、权重与排列

$$
z_t=W_{\mathrm{router}}x_t,\qquad
y_t^{\mathrm{routed}}=\sum_{e\in\mathcal T_t}w_{t,e}
W_{d,e}\bigl(\mathrm{SiLU}(W_{g,e}x_t)\odot W_{u,e}x_t\bigr).
$$

公式用列向量表达，代码 token 是行向量，权重按 $[\mathrm{out},\mathrm{in}]$ 保存。gate/up 必须按同一专家和同一 FFN 列配对。

Qwen 路径选择 norm_topk_prob=True：先取 top-k logits，再在所选项上 softmax。V2 参考遵循固定 Bridge pre-softmax：先对全部专家 softmax，再取 top-k，不额外归一化。V3 先 sigmoid，bias 只影响选择；组合取回未经加 bias 的原分数：

$$
s_{t,e}=\sigma(z_{t,e}),\quad
\mathcal T_t=\mathrm{TopK}(s_t+b),\quad
w_{t,e}=\gamma\frac{s_{t,e}}{\sum_{j\in\mathcal T_t}s_{t,j}+10^{-20}}.
$$

本例 V3 将四名专家分成两组，先选一组。固定 Core 的组分数为组内 top-$(k/\mathrm{group\_topk})$ 之和；这里是两分数之和，不是固定取 max。并列时参考选择较小专家编号，不承诺生产 torch.topk 的 tie 顺序。

手算 logits $[\log1,\log2,\log3,\log4]$：全 softmax 为 $[.1,.2,.3,.4]$。选择专家 3、2，V2 权重为 $[.4,.3]$，总和 .7；Qwen 为 $[4/7,3/7]$，总和 1。V3 若两个原分数都是 .5，$\gamma=2.5$，各项为 1.25。“权重和必须为 1”不是通用规则。

dispatch 保存 token、expert、weight，按专家聚集输入，计算专家输出，乘一次权重，再用逆映射按 token 求和。同一 token 两个专家输出 2、6，权重 .75、.25，恢复为 3。重复相同 token/expert 是错误，top-k 中两个不同专家正常。

shared 对每个有效 token 计算，不进入 top-k，不乘 routed scaling：

$$
y_t=y_t^{\mathrm{routed}}+y_t^{\mathrm{shared}}.
$$

两个 shared 可合并成宽度 $2F$ 的 paired gate/up 与 down，等价于独立 SwiGLU 输出相加。“漏 shared”和“错归一化”会改变完整模型 CE。

## Padding、辅助项与 bias

本参考显式从 MoE 计算中移除 suffix padding，它们的 routed/shared 输出、计数和辅助项均为零。assistant loss mask 为零的 prompt 仍是有效 token，仍参与路由。

固定 Core 的 compute_routing_scores_for_aux_loss 实际用取反的 padding_mask；部分 docstring 的 True/False 与代码相反。该参数本身不保证主路由概率归零；教学有效 token 过滤是显式策略，不能冒充所有 Core 默认行为。

辅助项独立从无 bias、无 group 限制的分数取 top-k。softmax 分数直接使用，sigmoid 分数先在全部 $E$ 专家上归一化。实际 dispatch 的 counts 不能代替辅助 counts。设辅助集合为 $\widehat{\mathcal T}_t$：

$$
f_e=\frac{\sum_{t\ \mathrm{valid}}\mathbf1[e\in\widehat{\mathcal T}_t]}{Nk},
\quad P_e=\frac1N\sum_{t\ \mathrm{valid}}p_{t,e},
\quad \mathcal L_{\mathrm{aux}}=\alpha E\sum_e f_eP_e.
$$

counts/top-k 指标不求导，概率保留梯度。Qwen 使用 aux_loss，V2/V3 使用 seq_aux_loss。本例 $B=1$ 时归约相同，不扩展成任意变长 batch 等价。Qwen 系数 .001、V3 .0001 对应固定 provider；V2 .001 是本 authored fixture 的显式设置，未冒充已加载 HF 训练超参数。

生产实现用 MoEAuxLossAutoScaler 附加梯度，且有有效 token、累积、group scaling。本例显式优化 $\mathcal L_{\mathrm{CE}}+\sum_l\mathcal L_{\mathrm{aux},l}$，限单次 CPU 参考，不声称复刻分布式 scaler。

V3 bias 是独立控制量，不是 SGD 参数。排除 padding 的实际派发计数在 global batch finalize 阶段通过 TP×CP×DP 聚合，然后：

$$
\bar n=\frac1E\sum_en_e,\qquad
b_e\leftarrow b_e+\eta_b\,\mathrm{sign}(\bar n-n_e).
$$

只对 training 且未冻结的 router 更新，然后清临时计数。不是每个 token 更新，也不是 sigmoid 导数。页面显示“本批计数产生的下一次 bias”，切 token 不累积更新。counts $[1,1,0,0]$，均值 .5，$\eta_b=.01$：前两项减 .01，后两项加 .01。

## MLA：norm 位置决定能否吸收

pre-attention norm 后，V2-Lite 直接投影 Q；V3 做 Q down→RMSNorm→up。KV down 同时产生 latent 与独立 positional key，只有 latent 经 RMSNorm 后进入 KV up：

$$
c_t=\mathrm{RMSNorm}(W^{DKV}x_t),\quad
k_{h,t}^{N}=U_h^Kc_t,\quad v_{h,t}=U_h^Vc_t.
$$

$W^{DKV}$ 表示 combined down 的 latent 行；positional 行另取。RoPE 只作用于分离的位置子空间，不旋转内容 latent，也不能把 KV norm 移到 up 之后。RMSNorm 非线性，不能并入常数权重。

训练展开路径为：

$$
a_{h,t,s}=\mathrm{softmax}_{s\le t}
\left(\frac{(q_{h,t}^N)^\top U_h^Kc_s+
(R_tq_{h,t}^{R,\mathrm{raw}})^\top(R_sk_s^{R,\mathrm{raw}})}
{\sqrt{d_n+d_r}}\right),\qquad
o_{h,t}=\sum_{s\le t}a_{h,t,s}U_h^Vc_s.
$$

标准 RoPE，本例 $d_r=2$，旋转角为 position。没有 YaRN 扩展，固定源码的 mscale 在本例设为 1。在相同权重、norm、mask、position、scale 下可重排：

$$
\widetilde q_{h,t}=(U_h^K)^\top q_{h,t}^{N},\qquad
u_{h,t}=\sum_{s\le t}a_{h,t,s}c_s,\qquad
o_{h,t}=U_h^Vu_{h,t}.
$$

分母始终为 $\sqrt{2+2}=2$，吸收后不能改成 $\sqrt{r_{kv}+d_r}=\sqrt5$。手算 $c=[1,2,3]$、$U^K=[[1,0,0],[0,1,1]]$、$q=[2,3]$：展开与重排内积都为 17。attention logits $[0,\log3]$、values $[2,4]$，输出为 3.5。

## 训练张量与 decode cache

训练视图展示 causal Q/K/V 和 weighted value；decode 视图逐 token 追加 $[c_t,R_tk_t^R]$，只读已有 prefix，weighted latent 再乘 $U_h^V$。固定 Core 中，Q/V 吸收需 cache_mla_latents 且 is_decode_only；prefill/mixed 可走展开路径。这里只验证数学参考，没有调用 Flash MLA。

每层每 token 保存 $r_{kv}+d_r=5$ 个 float64 数，即 40 bytes；全长 $S$ 为 $40S$ bytes。展开 K/V 为 $n_h(d_n+d_r+d_v)=12$ 个数，即 96 bytes/token。这只是数组 payload，不含 allocator、梯度或临时 buffer，不是 GPU 显存。训练激活也不是 decode cache。

“错 latent norm”“错 scale”“漏位置旋转”分别破坏等价。正常展开、吸收、逐 prefix cache 应一致。导出标记 reference，训练与 cache 切片分开。

## 专家组与验收

只提供 EP1/2、ETP1/2、EDP1。参考编号是 ep_rank × ETP + etp_rank；EP 分专家，ETP 分 paired FFN 列并 SUM row partial，EDP1 为单成员。不是任意 Core world rank，不能机械乘到 G04 dense world。EDP2、overlap、capacity drop、FP8 padding、MTP 未验证，不显示为可用。

四种组合对照完整未切分模型 logits、CE+aux 和所有可训练参数梯度。CPU 验证 router/MLA 有限差分、冻结参数、实际 optimizer 更新与 padding 不影响有效输出；TypeScript 对照独立 Torch 的三种 family、长短样本和 padding。浏览器须验证控制变化、离线源码、导出、数学、键盘和窄屏。

源码链：B-Q3M/B-DS2/B-DS3 → C-ROUTER/C-MOEUTIL → C-DISPATCH/C-EXPERT → C-FINALGRAD。MLA 看 C-MLA 的 direct Q、latent norm、cached latent、V up 和 score scale 分支。逐字原文与推导分开，静态阅读不标 runtime trace。
