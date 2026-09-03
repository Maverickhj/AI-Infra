---
type: project
status: active
created: 2026-08-20
updated: 2026-08-25
domains:
  - llm
  - transformer
  - paper-study
source:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1412.6980
  - https://arxiv.org/abs/1711.05101
ai_generated: true
reviewed: false
---

# LLMZeroToHero Paper Learning

> [!warning]
> 本项目包含 AI 生成的课程组织与讲解，尚未完成人工核验。论文学术事实以原始论文为准；LLMZeroToHero 的中英对照 PDF 仅作阅读便利，不能单独作为证据。

## 目标

按“已有约束 → 瓶颈 → 机制 → 证据 → 边界 → 后续问题”的因果链精读 LLMZeroToHero 论文，并把跨论文、经人工核验后仍可复用的结论沉淀到 `20_Knowledge/`。

## 当前课程

- [[MISSION]]：学习任务的边界与成功标准。
- [课程范围索引](CURRICULUM.md)：论文 → lesson / learning-record 编号范围与状态；变动时以此为准。
- [资源清单](RESOURCES.md)：首篇论文的原始来源与本地阅读材料。
- [Math Foundations Index](reference/math-foundations-for-attention.html)：按 Lesson 按需进入的数学前置索引；一般原理与论文实例分开维护。
- [Attention Glossary](reference/attention-glossary.html)：英文术语与中文解释的速查表。
- [课程 0001：从串行瓶颈到 scaled dot-product attention](lessons/0001-transformer-sequential-bottleneck.html)
- [课程 0002：Multi-Head Attention](lessons/0002-multi-head-attention.html)
- [课程 0003：Encoder、Decoder 与 Cross-Attention](lessons/0003-attention-roles-in-transformer.html)
- [课程 0004：Position-wise Feed-Forward Network](lessons/0004-position-wise-ffn.html)
- [课程 0005：Residual Connection 与 LayerNorm](lessons/0005-residual-layernorm.html)
- [课程 0006：Positional Encoding](lessons/0006-positional-encoding.html)
- [课程 0007：Output Softmax 与 Weight Tying](lessons/0007-output-softmax-weight-tying.html)
- [课程 0008：Why Self-Attention](lessons/0008-why-self-attention.html)
- [课程 0009：Adam、Warmup 与 Learning-Rate Schedule](lessons/0009-adam-warmup-lr-schedule.html)
- [Adam 0009A：从 Noisy SGD 到 Adaptive Moments](lessons/0009a-adam-adaptive-moments.html)
- [Adam 0009B：Zero Initialization Bias 与 Bias Correction](lessons/0009b-adam-bias-correction.html)
- [Adam 0009C：Effective Step、Experiments 与 Convergence Evidence](lessons/0009c-adam-effective-step-evidence.html)
- [课程 0010：Batching、Dropout 与 Label Smoothing](lessons/0010-batching-dropout-label-smoothing.html)
- [课程 0011：Results、Ablation 与 Evidence Boundaries](lessons/0011-results-ablation-evidence.html)
- [课程 0012：Transformer 全文因果链复盘与知识沉淀判断](lessons/0012-transformer-causal-review-promotion.html)
- [AdamW 0013：L2 正则与 Weight Decay 的等价性破解](lessons/0013-adamw-decoupled-weight-decay.html)
- [速查：Attention 的因果链与张量形状](reference/attention-causal-chain.html)

## 剩余路线

Adam 0009A–0009C 与 Transformer 0010–0012 已完成第一轮学习。Lesson 0012 的 Cross-Attention final synthesis retrieval 被跳过，保留为 `[待核验]`；它不阻止 source / knowledge seed 的创建，但这些 notes 仍需人工 review。

Adam 支线不改变 Transformer 主线编号。数学前置按 model-agnostic 主题加入 Math Foundations，architecture mapping 与论文专用推导留在 Lesson；若某个概念需要多轮巩固，课程总数会相应调整。

AdamW 从 0013 开始第一轮学习；★ 必读篇目中还剩 RoPE。编号范围以 `CURRICULUM.md` 为准。

## 本轮沉淀产物

- Paper sources：[[2017 - Attention Is All You Need]]、[[2015 - Adam - A Method for Stochastic Optimization]]。
- Transformer knowledge：[[Transformer Architecture Causal Chain]]、[[Transformer Parallelism Boundaries]]。
- Optimization knowledge：[[Adam Optimizer - Moments, Bias Correction, and Evidence Boundaries]]。
- Research method：[[Reading ML Paper Results and Ablations]]。
- Final learning record：[[0012-transformer-causal-review-promotion]]。

其中 Adam 与 Transformer 通过“architecture 的 training dynamics requirement → Adam moments / adaptive scaling → Transformer-specific warmup schedule”建立双向 causal links；两篇论文的 evidence scope 仍保持独立。

## 沉淀规则

单篇阅读的解释、疑问和未复算结论留在本项目。跨论文可复用、但仍由 AI 整理的内容可以先以 `status: seed`、`reviewed: false` 进入 `20_Knowledge/`；只有经人工复核后，才提升为成熟知识。
