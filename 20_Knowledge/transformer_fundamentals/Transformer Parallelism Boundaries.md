---
type: knowledge
status: seed
created: 2026-08-25
updated: 2026-08-25
domains:
  - llm
  - transformer
  - inference
aliases:
  - Transformer parallelism boundary
source:
  - "[[2017 - Attention Is All You Need]]"
ai_generated: true
reviewed: false
---

# Transformer Parallelism Boundaries

> [!warning]
> 本文包含 AI 整理的可迁移结论，尚未完成人工核验。理论 operation count 不等同于具体 kernel、hardware 或 distributed-system performance。

## Three different claims

| Claim | Verdict | Scope |
|---|---|---|
| 同一 self-attention layer 的 positions 可以批量计算 | 成立 | Layer-level computation；不存在 RNN 式的 position-by-position recurrence |
| Teacher-forcing training 中 target positions 可以并行计算 | 成立 | 完整 shifted target 已知；causal mask 保证每个 query 只能使用合法 prefix |
| Autoregressive inference 可以一次生成所有未知 tokens | 不成立 | $y_{t+1}$ 依赖已经生成的 $y_t$，generation loop 仍然 sequential |

Transformer 移除的是 layer 内 recurrence，不是 autoregressive factorization。

## Original-paper cost model

设 $n$ 为 sequence length、$d$ 为 representation dimension、$r$ 为 restricted attention neighborhood size：

| Layer type | Per-layer complexity | Minimum sequential operations | Maximum path length |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

当 $n<d$ 时，paper-level algebraic comparison 给出 $n^2d<nd^2$。这只是 operation-count model；实际 runtime 还受 memory bandwidth、kernel fusion、parallel occupancy、communication 与 hardware shape 影响。

## Training versus inference

Teacher forcing 已知 target sequence，因此可以构造整张 causal attention matrix：

$$
M_{t,j}=
\begin{cases}
0, & j\le t,\\
-\infty, & j>t.
\end{cases}
$$

所有 rows 可作为一个 batched operation 计算，但 row $t$ 仍不能看到 future keys。Parallel computation 没有改变 information dependency。

Inference 时 future token 尚不存在，无法预先作为 K/V 加入 sequence。即使每一步内部的 attention 是 parallel tensor computation，steps 之间仍满足：

$$
y_t\rightarrow y_{t+1}\rightarrow y_{t+2}.
$$

## Engineering boundary

Maximum path length $O(1)$ 描述一个 layer graph 中任意 positions 的连接距离，不保证：

- end-to-end latency 为 $O(1)$；
- attention memory 为 $O(1)$；
- distributed communication 为 $O(1)$；
- autoregressive decode steps 可以被消除。

## Related notes

- [[Transformer Architecture Causal Chain]]
- [[2017 - Attention Is All You Need]]
