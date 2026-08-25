# 已建立：Transformer Results、Ablation 与 Evidence Boundaries

## 1. Reported score 属于完整 evaluation pipeline

Table 2 的 BLEU 不是 Transformer architecture 单独产生的数字。它属于以下完整 pipeline：

1. Training data、tokenization、model configuration、optimizer、schedule 与 regularization 决定 trained parameters。
2. Base model 对最后 $5$ 个 checkpoints 求平均，big model 对最后 $20$ 个 checkpoints 求平均。
3. Beam search 使用 beam size $4$、length penalty $\alpha=0.6$，maximum output length 为 input length $+50$。
4. 最终 decoded sequences 在指定 test set 上计算 BLEU。

因此，若 checkpoint selection、decoding hyperparameters 或 metric setup 改变，即使 architecture 不变，reported BLEU 也可能改变。Table 2 支持的是 paper-defined translation system 的整体结果，不是 architecture-only causal effect。

需要保留的 source inconsistency：

- Abstract 与 Table 2 报告 Transformer big 在 WMT 2014 EN–FR 上得到 $41.8$ BLEU。
- §6.1 narrative 写成 $41.0$ BLEU。

不能静默把二者当成无歧义的同一个数值。

## 2. Training Cost (FLOPs) 的含义

论文采用的 estimate 为：

$$
C_{\mathrm{est}}
=T_{\mathrm{train}}
\times N_{\mathrm{GPU}}
\times F_{\mathrm{sustained}}.
$$

其中：

- $T_{\mathrm{train}}$ 是 training wall-clock seconds；
- $N_{\mathrm{GPU}}$ 是 GPU 数量；
- $F_{\mathrm{sustained}}$ 是作者估计的单张 GPU sustained single-precision throughput，单位为 $\mathrm{FLOPs/s}$。

对 Transformer big 的 P100 context：

$$
\begin{aligned}
C_{\mathrm{est}}
&=(3.5\times24\times3600)\,\mathrm{s}
\times8
\times9.5\times10^{12}\,\mathrm{FLOPs/s}\\
&\approx2.3\times10^{19}\,\mathrm{FLOPs}.
\end{aligned}
$$

这个 quantity 是 paper-defined arithmetic estimate，不等于 measured energy、cloud price、peak memory、communication cost 或现代 accelerator 上的实际 efficiency。

## 3. Table 3 rows (A)–(E)

### (A) Number of heads

在论文设计的 approximately constant-compute variations 中：

- Single head 得到 $24.9$ BLEU，比 best tested setting 的 $25.8$ 低 $0.9$。
- Head count 增加到 $32$ 时，BLEU 又下降到 $25.4$。

用户已经正确拒绝了“head num 越多，模型效果越好”的 monotonic claim。更精确的 evidence label 是：这是 tested settings 内的 local observation，说明 single head 与过多 heads 都可能较差；它不是关于任意 Transformer 最佳 head count 的普遍结论。

需要进一步修正的是：head-count result 本身不是作者提出的 mechanism hypothesis。它直接来自 rows (A) 的 observed comparison。

### (B) Key dimension

把 $d_k$ 从 base 的 $64$ 降到 $32$ 或 $16$ 时，BLEU 分别降至 $25.4$ 与 $25.1$。这是 local observation。

作者由此提出的 mechanism hypothesis 是：determining compatibility 可能并不容易，更复杂的 compatibility function 可能比 dot product 有益。Table 3 没有直接证明该 hypothesis。

### (C) Capacity

在 tested settings 中，减少 layers 或缩小 $d_{\mathrm{model}}$、$d_{\mathrm{ff}}$ 通常使结果变差，较大的 tested configurations 通常更好。

但 big row 同时改变 width、head count、dropout、training steps 与 parameter count，不是 single-factor ablation，不能把全部增益归因给其中一个 variable，也不能推出无限增加 capacity 会持续改善。

### (D) Regularization

$P_{\mathrm{drop}}=0$ 时 BLEU 为 $24.6$，低于 base configuration 的 $25.8$；这支持 dropout 在该 EN–DE development setup 中有助于避免 overfitting。

Label-smoothing variations 也再次显示 PPL 与 BLEU 不必同方向变化，但不能由该表确定所有 tasks 的最佳 dropout 或 smoothing strength。

### (E) Positional representation

Learned positional embeddings 得到 $25.7$ BLEU，sinusoidal base 得到 $25.8$。可以说两者在该 comparison 中结果近似，不能推出它们在 length extrapolation、其他 tasks 或更长 context 上等价。

## 4. Metric 与 dataset scope 不能混用

Table 2 与 Table 3 的 evidence scope 不同：

| Dimension | Table 2 | Table 3 |
|---|---|---|
| Dataset split | newstest2014 test | EN–DE newstest2013 development set |
| Purpose | 与 prior systems 比较 | 比较 Transformer variations |
| Checkpoint averaging | Base last $5$，big last $20$ | 不使用 |
| Model-selection role | 最终 test report | 参与 architecture / hyperparameter exploration |

Table 3 的 PPL 是 per-wordpiece perplexity：

$$
\operatorname{PPL}
=\exp\left(
-\frac{1}{N}
\sum_{i=1}^{N}
\log p(y_i\mid y_{<i},x)
\right),
$$

其中 $N$ 是有效 wordpiece positions 数量。它不能直接与 per-word PPL 比较，因为 reduction unit 不同。

BLEU 则评价最终 decoded sequence 与 references 的 n-gram overlap。Per-wordpiece PPL、BLEU、development-set variation 和 held-out test result 回答的是不同问题，不能只因为它们都是单个 scalar 就放在同一个证据层级。

## 可迁移的 evidence-reading 规则

1. 先恢复完整 evaluation pipeline，再解释 reported score。
2. 区分 observed result、local comparison、mechanism hypothesis 与 general claim。
3. Ablation 必须检查哪些 variables 真正保持不变；多项配置同时变化时不能做 single-factor attribution。
4. 比较 metrics 前先核对 dataset split、tokenization / reduction unit、checkpoint selection 与 decoding setup。
5. 论文没有报告 repeated runs、error bars 或 statistical tests 时，小数点级差异不能自动解释成稳定的真实优势。

Primary source: Vaswani et al., *Attention Is All You Need*, §6、Tables 2–3。
