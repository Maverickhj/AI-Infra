---
type: knowledge
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Case 02：从 Qwen 整模切换到 DeepSeek 的 MLA + MoE

## 本章的上下文

参考源码固定为 Bridge v0.6.2 及其 Core，实际 profile 按最小兼容合约验证。读者已具备基础 Transformer 知识，公共骨架只概括，重点精讲 MLA/MoE 差异。首先展示完整 tokens→embedding→61层 decoder→final norm→LM head 的 DeepSeek-V3 骨架，再对比 Qwen3。所有参数量和 shape 区分 config 推导与实测，不自动下载全量 V3。

来源：[B-DS3][B-DSMAP][C-MLA][C-MOE][HF-deepseek-v3]，见 ../../research/REFERENCES.md。

## 数学符号

| 符号 | 含义 | V3 配置 |
|---|---|---|
| $H,L$ | hidden size 与 decoder 层数 | 7168、61 |
| $n_h$ | MLA heads 数量 | 128 |
| $r_q,r_{kv}$ | query 与 KV 的低秩维度 | 1536、512 |
| $d_n,d_r,d_v$ | no-PE、RoPE、value head 维度 | 128、64、128 |
| $c_t^Q,c_t^{KV}$ | 第 t 个 token 压缩后的 Q/KV 表示 | 逻辑维度 $r_q,r_{kv}$ |
| $q^N,q^R,k^N,k^R$ | 内容/位置两部分 Q 与 K | 按对应维度展开 |
| $E,k,E_s$ | routed experts、top-k、shared expert 数 | 256、8、1 |
| $p_{t,e}$ | 第 t 个 token 分配给 expert e 的组合权重 | 由选定路由分支确定 |
| $x_t,h_t$ | MoE 输入与输出 hidden state | 宽度 H |
| $f_e,f_s$ | routed expert 与 shared expert 的 FFN | 各自参数 |
| $\mathcal T_t$ | 第 t 个 token 选中的 expert 集合 | 大小8 |

## 1. 一个模型里存在不同 layer 类型

layer0–2 为 dense FFN，layer3–60 为 MoE。前端应该用 layer-type strip 展示并允许点击，不能将61层画成相同的“MoE方块”。MTP 配置是额外可见分支，不偷偷增加到 decoder layer count 中。[B-DS3][HF-deepseek-v3]

## 2. Attention 换成 MLA，并非多几个 head

展示两条语义路线：

- Q：hidden states → q_down → latent norm → q_up → 拆为 no-PE 与 RoPE 部分。
- KV：hidden states → kv_down → 压缩 KV 与位置 K 分离 → latent norm → kv_up 得到内容 K/V；位置 K 应用 RoPE，再按实际实现组合。

V3 的 query 每头总打分维度为 $d_n+d_r=192$，value 每头维度128，两者不同。KV 压缩维度512不能误标为每个 head 的 K 维度。相应 HF 参数 q_a/q_b/kv_a/kv_b 可追到 Bridge 的 down/up 参数映射。[B-DSMAP][C-MLA]

## 3. 训练与 decode 展示两个视图

训练/teacher-forced 视图解释序列内 QK/V 计算及需要保存或重算的中间值。decode 视图解释 latent cache、位置 K、逐 token cache 增长和条件 absorption。

源码中的 cached-latent inference 分支与普通 up-projection 分支分开。不要把“只缓存512维 latent”的推理结论直接用作训练显存估算。[C-MLA]

## 4. 路由、专家、组合构成一条完整 FFN 路线

语义上，某 token 的 MoE 输出包含 routed experts 的加权组合以及 shared expert 分支：

$$
h_t=\sum_{e\in\mathcal T_t}p_{t,e} f_e(x_t)+f_s(x_t).
$$

$x_t$ 是进入该 MoE 的 token hidden state，$h_t$ 是该 FFN 输出。归一化、缩放、group-limited top-k 和 expert-bias 的细节以选定代码分支为准，不能凭该概念式重写整个 router。

页面与源码节点对应：route → preprocess → dispatch → routed_experts_compute → combine → postprocess。shared expert 可以与通信 overlap，因此语义上的并行分支不等于固定时间顺序。[C-MOE]

## 5. 要比较的是三类不同模型

| 特性 | Qwen3-30B-A3B | DeepSeek-V2-Lite | DeepSeek-V3 |
|---|---:|---:|---:|
| Attention | GQA | MLA | MLA |
| Q 压缩 rank | 不适用 | null，直接 Q 路径 | 1536 |
| KV 压缩 rank | 不适用 | 512 | 512 |
| 初始 dense 层数 | 0 | 1 | 3 |
| Routed experts | 128 | 64 | 256 |
| Top-k | 8 | 6 | 8 |
| Shared experts | 0 | 2 | 1 |

V2-Lite 是更小的 MLA/MoE 实验候选，但依然要先做资源预算与 Bridge 导入审计。R1-Distill-Qwen 则是 Qwen2 dense 家族，不能填入本表冒充 MLA。[HF-qwen3-30ba3b][HF-deepseek-v2-lite][HF-deepseek-v3][HF-dsr1-distill-qwen15b]

## 6. 与 SFT、RL 直接连接

SFT：response loss 之外，可能存在 router 辅助目标；先看当前 configured coefficient，再解释它对 backward 的贡献。packed padding 的排除、专家负载不均和保存路由状态都属于实际训练问题。

RL：固定同一条 rollout，用 generation 与 training backend 比较 logprob，并在 MoE 场景检查 routed expert IDs 是否一致。不能把所有差异都归因于“模型更新”，也不能假设用同一权重就能 bitwise 一致。Router Replay 作为正式框架中可验证的进阶功能，与默认无 replay 的 baseline 分开。

## 7. 运行级别的诚实边界

原始 V3 checkpoint 的架构资料和源代码可完整学习；没有资源时，实测区域明确未运行。architecture-scaled 配置改变 layers/hidden/experts 后必须重新检查分片约束，标记随机初始化，与原模型 revision 使用不同 architecture id。不要将该实验的 loss 曲线标成 DeepSeek-V3 SFT。

## 本章验收

切换 DeepSeek 时实际改变 attention 计算图；dense/MoE layer 布局准确；能看到 q/kv 的 down/up 与位置分离；能解释 shared expert 与 routed expert 的差别；能明确分开训练激活与 decode cache；能沿固定源码定位每一步。
