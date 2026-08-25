---
type: knowledge
status: seed
created: 2026-08-25
updated: 2026-08-25
domains:
  - research-methods
  - machine-learning
aliases:
  - ML paper evidence reading
source:
  - "[[2017 - Attention Is All You Need]]"
ai_generated: true
reviewed: false
---

# Reading ML Paper Results and Ablations

> [!warning]
> 本文包含 AI 整理的方法论，尚未完成人工核验。它用于校准 claim strength，不能代替对具体论文实验设计的复查。

## Evidence ladder

阅读一个 reported result 时，先判断它处于哪一层：

| Label | Meaning | Safe wording |
|---|---|---|
| Observed result | 论文在特定 setup 中报告的 measurement | “论文报告……” |
| Local comparison | Tested configurations 之间的差异 | “在这些 settings 中，A 高于 B……” |
| Mechanism hypothesis | 作者对差异原因的解释 | “作者推测……；该实验没有隔离验证机制” |
| General claim | 跨 tasks、scales 或 implementations 的规律 | 需要更广泛、可复现且控制充分的证据 |

不要把 local comparison 直接升级为 general claim，也不要把 mechanism hypothesis 改写成已证明的 causal mechanism。

## Reconstruct the evaluation pipeline

一个 score 通常属于完整 pipeline：

$$
\text{data}
\rightarrow
\text{training configuration}
\rightarrow
\text{checkpoint selection}
\rightarrow
\text{decoding / inference}
\rightarrow
\text{metric}.
$$

至少核对：

- Dataset、split 与 preprocessing / tokenization。
- Model configuration、optimizer、schedule 与 regularization。
- Checkpoint selection、averaging 或 ensemble。
- Decoding algorithm 与 hyperparameters。
- Metric definition、reduction unit 与 reporting convention。

因此，system-level benchmark score 通常不是 architecture-only causal effect。

## Audit an ablation

对每个 ablation row，列出：

1. Changed variables。
2. Held-constant variables。
3. Dataset split 与 selection role。
4. Number of runs、error bars 与 statistical test。
5. Measurement 的 scale 是否足以支持解释。

若多个 variables 同时改变，该 row 是 configuration comparison，不是 single-factor ablation。论文未报告 repeated runs 或 uncertainty 时，小数点级差异不能自动解释为稳定优势。

## Compare metrics only after checking scope

两个 scalar 可能评价完全不同的对象。比较前先确认：

- Prediction unit 是 token、wordpiece、word、sequence 还是 sample。
- Reduction 是 mean、sum、micro average、macro average 还是 weighted average。
- Metric 作用于 training distribution、development split 还是 held-out test。
- Metric 是否依赖 decoding，而不是只依赖 model probabilities。

例如 token-level perplexity 与 sequence-level BLEU 可以朝不同方向变化，因为它们的 evaluation object 与 aggregation 不同。

## Preserve source conflicts

若 abstract、table 与 narrative 给出不一致数字：

- 同时记录各位置的原值；
- 标明 conflict，而不是静默选择一个数字；
- 不在没有 source evidence 时猜测哪个是 typo。

## Worked source

[[2017 - Attention Is All You Need]] 的 Tables 1–3 分别对应 theoretical comparison、test pipeline result 与 dev-set model variations，是区分 evidence types 的一个实例。
