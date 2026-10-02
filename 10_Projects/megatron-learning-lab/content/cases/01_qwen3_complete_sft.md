---
type: knowledge
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Case 01：一条对话如何完成 Qwen3 的一次 SFT 更新

## 学习目标与基线

模型：`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`。参考源码：Bridge v0.6.2 与配套 Core；实际环境按最小接口兼容合约选择，不要求版本相等。先讨论 TP=PP=CP=1、不启用 packing、不启用 MTP 的路径，再引入变更。

面向有一定基础的初学者，默认已理解 tensor/Linear/autograd。基础只作定位，本章精讲整模计算差异与 SFT 数据/更新语义，不从独立 MLP 实验起步。以下 shape 根据官方 config 推导，并非采集结果；真实 tokenizer、权重和 execution trace 由实验生成。来源：[B-Q3][B-RECIPE][C-GPT][C-ATTN]，参见 ../../research/REFERENCES.md。

## 数学符号

| 符号 | 含义 | 本例 |
|---|---|---|
| $B$ | microbatch 的序列数 | 首轮1 |
| $S$ | 输入模型的 token 位置数，不包含移位前额外的末尾 token | 短序列，运行时记录 |
| $H$ | residual stream 的 hidden size | 1024 |
| $L$ | decoder 层数 | 28 |
| $n_q,n_{kv}$ | query heads 与 KV heads 数 | 16、8 |
| $d$ | 每个 Q/K/V head 的维度 | 128，由 config 显式给出 |
| $F$ | 单个 gate/up 分支宽度 | 3072 |
| $V,V_{pad}$ | HF 词表与 padding 后词表大小 | 151936；后者由 provider 确认 |
| $p$ | TP world size | 基线1 |
| $x_t,y_t$ | 第 t 个输入 token 和它的 next-token 目标 | $y_t=x_{t+1}$ |
| $m_t$ | 与目标对齐的 loss mask | 0 或1 |
| $z_t$ | 第 t 个位置的词表 logits | 逻辑长度 $V_{pad}$ |
| $W_g,W_u,W_d$ | gate/up/down 的逻辑权重 | 维度见 FFN 小节 |
| $g,u,h,o$ | FFN 的 gate/up/激活乘积/输出 | 中间宽度 F，输出宽度 H |
| $D_Q,D_K,D_V$ | Q/K/V 总投影宽度 | 2048、1024、1024 |
| $X$ | 当前子层的 hidden states | `[S,B,H]` |
| $\theta$ | 本次训练的模型参数集合 | Full SFT 是全部可训练参数 |
| $N$ | 全局有效目标 token 数 | $\sum m_t$，需定义归约范围 |
| $\ell_t$ | 第 t 个有效目标的负对数似然 | $-\log\operatorname{softmax}(z_t)_{y_t}$ |

## 1. 基础速览：样本在整模中的位置

一条对话经实际 template/tokenizer 得到 `[B,S]` tokens；embedding 后进入 `[S,B,1024]` residual stream，依次经过 28 层 decoder、final norm 与 LM head，再由 SFT 目标产生梯度和参数更新。首页只展开当前层，但完整层栈始终可见。

本节不重讲词嵌入、矩阵乘法或 autograd。直接进入 attention；完整样本的 shift/mask 和权重来源在后面的核心小节核对。先区分三个对象：模型结构、已加载权重、一次执行的输入，三者不能互相替代。

## 2. 核心精讲：Attention 不是“1024 除以16”

本模型 $d=128$，所以：

$$
D_Q=n_qd=2048,\quad D_K=D_V=n_{kv}d=1024.
$$

语义步骤：RMSNorm → QKV projection → 按 GQA groups 拆分 → Q/K norm → Q/K RoPE → causal attention → 合并 heads → output projection → residual add。

| 张量/算子 | 逻辑 shape，TP1 |
|---|---|
| residual 输入 | `[S,B,1024]` |
| fused QKV projection 输出 | `[S,B,4096]` |
| Q | `[S,B,16,128]` |
| K、V | 各 `[S,B,8,128]` |
| attention head 输出 | `[S,B,16,128]` |
| 合并 Q heads | `[S,B,2048]` |
| output projection 输出 | `[S,B,1024]` |

GQA 中每组2个 Q heads 对应1组 K/V。选中某个 head 时，需要显示它读取哪个 KV group，而不把 K/V 的 head 数“补成16”冒充真实存储。

数学上可解释 attention score/softmax/weighted V；TE/FlashAttention 执行时不一定物化整块 `[S,S]` score。网页的概念矩阵与真实 trace 要使用不同标签。[C-ATTN]

**源码追踪题：** `SelfAttention.get_query_key_value_tensors` 如何拆分 mixed_qkv？为什么 mapped QKV 不是简单地将所有 Q、所有 K、所有 V 顺序拼接？QK norm 的模块或 fused backend 在哪里？[B-QKV][C-ATTN]

## 3. 在整层中定位 FFN：在完整 layer 中解释 SwiGLU

语义上：

$$
g=XW_g,\quad u=XW_u,\quad h=\operatorname{SiLU}(g)\odot u,\quad o=hW_d.
$$

这里 $X$ 指进入 FFN 的归一化 hidden states；$W_g,W_u,W_d$ 分别为 gate、up、down 的逻辑权重，$g,u,h,o$ 为中间结果。代码的 Linear 权重以 `[out,in]` 存储，不等于上式书写的逻辑矩阵方向。

补充符号：$W_g,W_u\in\mathbb R^{H\times F}$，$W_d\in\mathbb R^{F\times H}$；$g,u,h$ shape 为 `[S,B,F]`，$o$ 为 `[S,B,H]`。gated fc1 融合后宽度是 $2F=6144$。

FFN 之后第二次 residual add，使输出仍为 `[S,B,1024]`，再交给下一层。MLP 的教学只占这一层内部的一步，不另起一个与整模无关的故事。[B-Q3]

## 4. 核心精讲：SFT 样本、监督位置与权重来源

### 同一条样本与监督范围

使用本包 `content/samples.jsonl` 的两轮对话。网页左侧保留原始 messages，右侧依次显示 rendered text、token id、role span、input/target 对应、loss mask。

第一条可以是：用户问“2+3 等于多少？”，assistant 回“5。”，用户再问“再乘以2呢？”，assistant 回“10。”。这些是课程自造样本，不是伪装的 tokenizer 输出。

**先预测：** 在 assistant loss 模式下，哪两个回复参与 loss？切到 last_turn 后发生什么？用户 token 不参与目标 loss，是否意味着它们不参与 forward，或者 prompt 位置的表示永远没有梯度？答案是否定的：后续响应依赖上下文。

使用真实 chat template，记录是否启用 thinking、special tokens、generation prompt。训练序列与 rollout prompt 的渲染目的不同，不要对完整 SFT 会话重复添加待生成 assistant 前缀。[B-SFTDATA]

### 只做一次 next-token 移位

对于移位前的 token 串 $[x_0,\ldots,x_S]$，语义上构造 inputs $[x_0,\ldots,x_{S-1}]$ 与 labels $[x_1,\ldots,x_S]$。loss mask 应跟随目标 token，而非仍指向原输入位置。

具体移位可能已经发生在 dataset/collator，不能在网页导出器和模型端再各做一次。实验应核对该版本实际 batch，并保存 input/label 的少量位置对照。禁止仅凭“这是 Hugging Face 模型”就套用 HF forward 内部 shift 的假设。

### 结构构建不等于权重加载

页面显示：HF id/revision → Qwen3Bridge.provider_bridge → GPTModelProvider 的 spec 和 provide → MCore GPTModel。

同时展示“权重来源”和“结构来源”。recipe 中 load_weights=False 只构建配置，不证明预训练权重已载入。SFT 实验必须在 setup 后核验 checkpoint/import 的来源，保留加载日志、关键权重统计和映射校验。[B-RECIPE][B-SFTENTRY][B-PROVIDER]

## 5. 核心精讲：Final norm、LM head 与 token loss

28 层完成后经过 final norm 和 LM head，产生逻辑 `[B,S,V_pad]` logits。该 checkpoint 配置 tie_word_embeddings=true，需要解释同一份权重在 input embedding 和 output head 的共享语义；PP 环境中的物理处理另作源码核对。[HF-qwen3-06b][C-GPT]

默认后处理在有 labels 时直接返回 token losses。课程同时展示“概念中存在 logits”和“forward 实際返回 token loss”两个事实，避免把返回值再次当 logits 做 CE。[C-GPT]

SFT 目标的一个明确教学基线为：

$$
L_{SFT}=\frac{\sum_t m_t\ell_t}{N},\qquad N=\sum_t m_t.
$$

这是选定的全局 token 均值目标，不声称全部 recipe 都采用相同缩放路径。实际 Bridge callback 返回 loss sum 和 num_tokens，最终归一化必须结合调度与梯度 finalize 路径验证。[B-LOSS]

## 6. 核心精讲：Backward 与 optimizer 是整模流程的一部分

网页的 backward 不应简单反放 forward 动画。至少解释：head 的梯度来源、residual 两条分支如何合流、attention/FFN 对输入和权重的梯度、共享 embedding 权重的梯度汇集。

真实实验记录少量层级 grad norm 与选定权重更新摘要，不 dump 全部 activations/gradients，不对每个模块注入 GPU synchronize。forward/backward hook 能观察的边界与 fused/custom autograd 的不可见部分需明确。

Full SFT 与 LoRA 使用不同 trainable 参数集合。LoRA 页必须先枚举实际 trainable names，不能凭模块叫 linear 就认定已插入 adapter。

## 7. 保存和恢复

显示三个不同状态：预训练权重导入、训练 checkpoint resume、HF 权重导出。后两者不是同一个文件格式或恢复能力。验证 resume 的 global step、优化器/scheduler/RNG 等需要的状态；模型权重导出则用于跨引擎推理或部署。

## 8. 两个必要的反例

1. 输入相同但重复移位 labels，网页要能定位每个被监督位置错在哪里。
2. 两个长短不同的样本分给两个 DP ranks，比较局部均值的均值与全局 sum/count。两者在有效 token 数不同的时候不等价。

## 9. 本章完成标准

用户能从一条对话走到一次更新；能解释1024→2048的 Q 投影；能比较 Qwen2.5 与 Qwen3 的 QKV bias/QK norm；能指出损失在何处被 mask/归一化；能打开对应固定版本源码；能区分真实权重 SFT 与结构 smoke test。
