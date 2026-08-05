---
title: "RoPE 长上下文外推方法综述"
subtitle: "Position Interpolation、Linear Scaling、NTK-aware Scaling、YaRN 与 LongRoPE"
author: "技术学习笔记"
type: knowledge
status: growing
created: 2026-07-22
updated: 2026-08-05
domains:
  - transformer
  - positional-encoding
  - long-context
aliases:
  - RoPE 长上下文扩展
lang: zh-CN
reviewed: false
---

# RoPE 长上下文外推方法综述

## 目录

- 1. 结论先行
- 2. 统一数学视角：扩展方法究竟在改什么
- 3. 为什么原始 RoPE 难以直接外推
- 4. Position Interpolation
- 5. NTK-aware Scaling
- 6. YaRN
- 7. LongRoPE
- 8. 总览对比
- 9. 如何选择
- 10. 工程实现检查清单
- 11. 如何验证“有效上下文长度”
- 12. 方法关系图
- 13. 参考文献与原始材料
- 14. 最小记忆版

> **文档目的**  
> 本文从 RoPE 的频率与相位表示出发，解释五类常见的长上下文扩展术语和方法，并重点澄清它们之间的继承关系。RoPE 的基础推导见 [[20_Knowledge/transformer_fundamentals/Positional Encoding/Transformer_RoPE_原理解析|Transformer RoPE 原理解析]]。这里的“支持 128K / 1M 上下文”不仅指程序能够接受这么长的输入，还要求模型在长距离检索、语言建模和原有短上下文能力上保持可用。只修改 `max_position_embeddings`，通常只能让程序更有勇气地犯错。

## 1. 结论先行

1. **Position Interpolation（PI）与 Linear Scaling 的核心数学操作相同。** PI 是论文中的方法与训练方案名称；Linear Scaling 通常是框架中对该操作的工程命名：将位置索引除以扩展倍数，等价于将全部 RoPE 逆频率统一除以该倍数。
2. **NTK-aware Scaling 不再均匀压缩所有频率。** 它通过增大 RoPE base，使高频维度少缩放、低频维度多缩放，以保留近邻 token 的位置分辨率。
3. **Dynamic NTK 是 NTK-aware 的推理时动态版本。** 它随当前序列长度改变 base，短序列保持原 RoPE，超出训练长度后逐步增强缩放。它可作为零微调基线，但对 KV Cache 的实现一致性要求很高。
4. **YaRN 是“按频率分段缩放 + 注意力温度修正”。** 高频维度保持外推，中频平滑过渡，低频执行完整插值，并调整 attention logits 的尺度。它尤其适合配合少量长上下文微调扩展到 32K–128K。
5. **LongRoPE 将缩放因子从人工公式变成搜索变量。** 它针对每个 RoPE 维度搜索不同缩放系数，同时搜索保留原始 RoPE 的起始 token 区间；再通过分阶段微调和二次搜索扩展到百万级上下文。
6. **方法选择不能只看“最大长度”。** 还要看是否有长文本训练预算、是否必须保持短上下文能力、推理框架是否原生支持，以及是否完成长距离检索和困惑度验证。

## 2. 统一数学视角：扩展方法究竟在改什么

设一个注意力头中实际应用 RoPE 的维度为 $d_r$，把维度两两分组，编号为

$$
i=0,1,\ldots,\frac{d_r}{2}-1.
$$

RoPE base 记为 $B$，通常为 $10000$。第 $i$ 个二维分组的逆频率为

$$
\omega_i=B^{-\frac{2i}{d_r}}.
$$

位于位置 $p$ 的 token 在这一维度上的旋转相位为

$$
\phi_i(p)=p\omega_i.
$$

对应波长为

$$
\lambda_i=\frac{2\pi}{\omega_i},
$$

它表示该二维分组完成一次完整旋转需要跨越多少个 token。

当 $i$ 较小时，$\omega_i$ 较大、$\lambda_i$ 较短，属于**高频维度**，擅长区分局部位置差异；当 $i$ 较大时，$\omega_i$ 较小、$\lambda_i$ 较长，属于**低频维度**，更适合表达较大尺度的位置变化。[1]

令模型原始训练上下文为 $L_0$，目标上下文为

$$
L_t=sL_0,
$$

其中 $s>1$ 是扩展倍数。所有 RoPE 扩展方法，本质上都在决定新的相位函数

$$
\widetilde\phi_i(p)=p\widetilde\omega_i,
$$

或者等价地决定如何把位置 $p$ 映射成新的有效位置。它们之间的主要区别是：

- 是否对所有维度使用相同缩放；
- 是否保留高频维度；
- 缩放是否随当前序列长度动态变化；
- 缩放因子由公式给定，还是由搜索得到；
- 是否同时修正 attention logits 的尺度。

![[20_Knowledge/images/图片 1.png]]

## 3. 为什么原始 RoPE 难以直接外推

原始模型只在 $p\le L_0$ 的位置范围内训练。直接把同一组频率用于 $p>L_0$，会产生训练中未出现过的相位组合。问题并不是正弦余弦在数值上算不出来，而是模型没有学会如何解释这些新相位。

从注意力分数看，RoPE 后的 Query 和 Key 点积依赖相对位置：

$$
\widetilde q_m^\top\widetilde k_n
=q_m^\top R((n-m)\omega_i)k_n.
$$

当 $|n-m|$ 远超训练范围时，各频率分组中的相位进入分布外区域，可能造成注意力分数和注意力熵异常。Position Interpolation 论文的理论分析指出，相比直接外推，把新位置压回训练位置范围更稳定；论文报告其插值上界至少比外推上界小约 $600$ 倍。[2]

然而，简单压缩全部频率也会付出代价：相邻位置之间的角度差同步缩小，模型区分近邻 token 顺序的能力可能下降。后续方法基本都在解决这组矛盾：

> 既要把远距离位置压入可处理范围，又不能把局部位置分辨率一起压扁。

## 4. Position Interpolation

### 4.1 核心操作

Position Interpolation 将目标上下文中的位置 $p$ 线性映射回原训练范围：

$$
\widetilde p=\frac{p}{s}.
$$

因此新的相位为

$$
\widetilde\phi_i(p)=\frac{p}{s}\omega_i.
$$

等价地，也可以不修改位置索引，而修改逆频率：

$$
\widetilde\omega_i=\frac{\omega_i}{s}.
$$

因此，目标位置 $p=L_t=sL_0$ 被映射到原训练边界 $L_0$：

$$
\widetilde p(L_t)=\frac{sL_0}{s}=L_0.
$$

所有新位置都落入模型见过的相位范围，而不是继续向外推。[2]

### 4.2 Position Interpolation 与 Linear Scaling 的关系

这两个名称经常被文档和代码写成两个方法，实际上它们的核心变换相同：

| 名称                                   | 常见语境             | 核心公式                        |
| ------------------------------------ | ---------------- | --------------------------- |
| Position Interpolation               | 论文方法、理论解释、配套微调流程 | $p\mapsto p/s$              |
| Linear Scaling / Linear RoPE Scaling | 推理框架或模型配置中的实现名称  | $\omega_i\mapsto\omega_i/s$ |

因为 RoPE 相位是 $p\omega_i$，所以

$$
\frac{p}{s}\omega_i=p\frac{\omega_i}{s}.
$$

Hugging Face Transformers 的官方实现也明确说明：对 position IDs 做缩放，与对 inverse frequencies 做缩放完全等价，其 `linear` RoPE 类型直接执行 `inv_freq /= factor`。[8]

因此，本文不把 Linear Scaling 当成与 PI 完全独立的算法。真正可能不同的是**使用方式**：

- 只在推理时打开 linear scaling，不做任何训练；
- 使用 PI 公式，并在目标长上下文数据上进行少量微调。

前者更便宜，但质量通常不如后者稳定。

### 4.3 训练与效果

PI 论文将 LLaMA 系列模型扩展到最多 32K 上下文，使用不超过 1000 个微调步骤，并在 passkey retrieval、语言建模和长文档摘要任务上进行了验证。[2]

### 4.4 优点

- 数学与实现最简单；
- 模型架构不变；
- 与 FlashAttention、张量并行和已有训练基础设施兼容；
- 作为长上下文微调的初始化通常比直接外推稳定。

### 4.5 局限

所有维度都缩放为原频率的 $1/s$：

$$
\frac{\widetilde\omega_i}{\omega_i}=\frac1s.
$$

这意味着最高频维度也被压缩。扩展倍数较大时，相邻位置的相位差变得很小，局部顺序信息和短上下文能力可能受损。PI 更像可靠、简单的基线，而不是对所有扩展倍率都最优的最终方案。

## 5. NTK-aware Scaling

### 5.1 术语来源与边界

“NTK-aware Scaled RoPE”最初由社区作者 bloc97 在 2023 年公开提出，而不是首先出现在正式论文中。[3] YaRN 论文随后对 NTK-aware、Dynamic NTK 和 NTK-by-parts 进行了系统化描述和实验比较。[5]

因此引用 NTK-aware 时，应区分：

- **原始社区方案**：改变 RoPE base 的静态 NTK-aware scaling；
- **Dynamic NTK**：根据当前序列长度动态改变有效 base；
- **NTK-by-parts**：按频率区间选择外推、混合或插值，是 YaRN 的频率缩放基础。

### 5.2 Static NTK-aware：通过改变 base 实现非均匀缩放

PI 对所有频率统一除以 $s$。NTK-aware 改为增大 RoPE base：

$$
B'=B\,s^{\frac{d_r}{d_r-2}}.
$$

新的逆频率为

$$
\widetilde\omega_i
=(B')^{-\frac{2i}{d_r}}
=\omega_i\,s^{-\frac{2i}{d_r-2}}.
$$

因此每一维的频率保留比例是

$$
\frac{\widetilde\omega_i}{\omega_i}
=s^{-\frac{2i}{d_r-2}}.
$$

它具有两个重要端点：

- 最高频分组 $i=0$：
  
  $$
  \widetilde\omega_0=\omega_0,
  $$
  
  完全不缩放，保留局部位置分辨率。

- 最低频分组 $i=d_r/2-1$：
  
  $$
  \widetilde\omega_i=\frac{\omega_i}{s},
  $$
  
  缩放程度与 PI 相同。

中间维度在两者之间连续过渡。直觉上，相当于把“插值压力”更多分配给低频维度，而不是让高频维度一起牺牲。[3][5]

### 5.3 Dynamic NTK

静态 NTK-aware 需要预先设定固定缩放参数。Dynamic NTK 则根据当前序列长度 $\ell$ 动态调整有效参数。原始社区方案以及 Hugging Face 的 `dynamic` 实现使用类似形式：

$$
a(\ell)
=\alpha\frac{\ell}{L_0}-(\alpha-1),
$$

并在 $\ell\le L_0$ 时保持 $a(\ell)=1$，随后更新 base：

$$
B'(\ell)=B\,a(\ell)^{\frac{d_r}{d_r-2}}.
$$

于是短序列使用原始 RoPE，序列超过原训练长度后才逐步减慢旋转速度。[4][8]

需要注意：不同框架中 `factor`、$\alpha$ 与“目标长度倍数”的语义并不总是完全相同，不能只看到 `factor=8` 就假设每个实现都会得到完全一致的 8 倍映射。

### 5.4 KV Cache 一致性问题

Dynamic NTK 的频率会随当前序列长度改变。如果旧 token 的 Key 已经用旧频率应用 RoPE 并写入 KV Cache，而新 token 使用新频率，那么缓存中的 Key 和当前 Query 不再位于同一套旋转坐标系中。

YaRN 论文明确指出，严格的动态实现需要缓存应用 RoPE **之前**的 Key，或在缩放变化时重新计算已缓存 Key 的旋转；否则会产生频率不一致。[5]

工程中常见的处理方式包括：

- 一个请求开始后固定本次请求使用的 scale；
- 预先按目标最大长度确定频率，而不逐 token 改变；
- 缓存 pre-RoPE Key，并在需要时重新旋转；
- 使用框架提供并已验证的 Dynamic RoPE 实现，不自行拼接旧 KV Cache。

### 5.5 优点与局限

**优点**

- 高频位置分辨率优于 PI；
- 零微调场景下通常是比线性缩放更强的快速基线；
- 短序列可保持原始 RoPE，尤其是动态版本。

**局限**

- 原始术语来自社区方案，公式和参数语义在实现间存在差异；
- Static NTK-aware 的部分维度仍可能进入外推区域；
- 目标上下文倍数与参数不总是一一对应；
- Dynamic NTK 与 KV Cache、连续批处理和 chunked prefill 的交互容易出错；
- 大倍率扩展仍通常需要长上下文微调。

## 6. YaRN

### 6.1 从“连续缩放”走向“按频率分段”

YaRN 的频率部分建立在 NTK-by-parts 上。它不再使用一条 base-change 曲线处理所有维度，而是根据每个维度在原始上下文中完成的旋转次数，划分三类：

1. **高频维度**：原训练窗口中已完成很多次旋转，保留原频率，不插值；
2. **低频维度**：原训练窗口中旋转次数很少，执行完整线性插值；
3. **中频维度**：在两者之间平滑过渡。[5]

第 $i$ 个维度在原始窗口内的旋转次数为

$$
\rho_i=\frac{L_0\omega_i}{2\pi}.
$$

YaRN 常用 $\beta_{\text{fast}}=32$ 与 $\beta_{\text{slow}}=1$ 作为边界。工程实现先把旋转次数边界转换成维度边界：

$$
i(\beta)=
\frac{d_r\ln\left(\frac{L_0}{2\pi\beta}\right)}{2\ln B}.
$$

记对应的高频边界和低频边界为 $i_f$ 与 $i_s$，定义线性 ramp：

$$
\gamma_i=
\operatorname{clip}
\left(
\frac{i-i_f}{i_s-i_f},0,1
\right).
$$

新的逆频率可写成

$$
\widetilde\omega_i
=(1-\gamma_i)\omega_i
+\gamma_i\frac{\omega_i}{s}.
$$

因此：

- 高频区 $\gamma_i=0$，使用原频率；
- 低频区 $\gamma_i=1$，使用完整 PI；
- 中频区使用两者的线性混合。

Hugging Face 的官方实现正是构造 `inv_freq_extrapolation` 与 `inv_freq_interpolation`，再通过维度 ramp 混合二者。[8]

### 6.2 Attention temperature / magnitude scaling

YaRN 还观察到，长上下文扩展会改变 attention logits 的统计尺度，因此引入 attention temperature 修正。常见实现使用

$$
m(s)=
\begin{cases}
1, & s\le1,\\
1+0.1\ln s, & s>1,
\end{cases}
$$

并通过缩放 RoPE 的 cosine/sine embedding 来等价影响 Query、Key 和 attention logits。Hugging Face 的 `yarn` 实现也以该形式推导默认 `attention_factor`。[5][8]

这一部分很重要：YaRN 并不是单纯换一组频率，它还在补偿上下文拉长后 softmax 温度和注意力熵的变化。

### 6.3 训练效率与效果

YaRN 论文报告：

- 相比之前方法，使用约 **10 倍更少的训练 token**；
- 使用约 **2.5 倍更少的训练步骤**；
- LLaMA 2 扩展实验仅微调约 **400 步**；
- 使用 64K 长度训练数据，可外推并验证到 128K；
- Dynamic-YaRN 在无微调情况下可实现超过 2 倍的上下文扩展。[5]

### 6.4 优点

- 保留高频维度，局部位置关系优于统一 PI；
- 低频维度仍获得足够压缩，能够覆盖更远距离；
- attention scaling 改善扩展后的注意力统计；
- 微调 token 和步骤需求较低；
- 不改变 attention 算法结构，与 FlashAttention 兼容。

### 6.5 局限

- $\beta_{\text{fast}}$、$\beta_{\text{slow}}$ 与 attention factor 仍是人工规则；
- 零微调下不一定能达到配置声明的完整目标长度；
- 高倍率扩展仍依赖长上下文训练和验证；
- 不同框架对 `mscale`、`attention_factor`、`original_max_position_embeddings` 的配置方式存在版本差异。

## 7. LongRoPE

### 7.1 两类非均匀性

LongRoPE 认为 YaRN 等方法仍使用人工分组规则，没有充分利用 RoPE 的复杂非均匀性。它重点建模两类差异：[6]

1. **维度非均匀性**：不同 RoPE 维度应使用不同缩放系数；
2. **位置非均匀性**：序列开头的一段 token 对注意力较重要，应减少或取消位置插值。

对每个二维分组搜索缩放因子 $\lambda_i$，并搜索起始 token 阈值 $n_0$。概念上可写为

$$
\widetilde\omega_i(p)=
\begin{cases}
\omega_i, & p<n_0,\\
\omega_i/\lambda_i, & p\ge n_0.
\end{cases}
$$

其中 $\lambda_i$ 不再由统一公式确定，而是通过验证集困惑度搜索得到。论文为维度缩放引入单调约束：高频维度通常缩放较少，低频维度缩放较多，以减少搜索空间并符合 NTK 相关直觉。[6]

工程配置中，LongRoPE 通常表现为长度为 $d_r/2$ 的数组：

- `long_factor[i]`：长序列时第 $i$ 个频率的缩放系数；
- `short_factor[i]`：短序列时用于恢复短上下文性能的系数。

Hugging Face 的实现使用

$$
\widetilde\omega_i
=\frac{1}{\lambda_i B^{2i/d_r}},
$$

并根据当前长度选择 `long_factor` 或 `short_factor`。[8]

### 7.2 搜索而不是手写规则

LongRoPE 使用进化搜索，以少量长文本样本上的困惑度作为目标，搜索：

- 每个 RoPE 维度的缩放系数；
- 保留原 RoPE 的起始 token 数；
- 满足频率维度上的单调约束。

论文将 PI、NTK 与 YaRN 的缩放曲线放入初始种群，以便更快找到优于人工规则的解。[6]

这带来的关键变化是：

> YaRN 问“哪些频率应该外推或插值”；LongRoPE 问“针对这个具体模型和目标长度，每一维究竟缩放多少才最好”。

### 7.3 渐进式扩展

直接在 2M token 上微调成本极高，也缺少足够长的训练文本。LongRoPE 采用分阶段策略：[6]

1. 从原模型搜索 128K / 256K 的缩放系数；
2. 先用 128K 配置微调约 400 步；
3. 切换到 256K 配置再微调约 600 步；
4. 在得到的 256K 模型上再次搜索；
5. 不进行 2M 长度微调，直接把有效上下文扩展到 2048K；
6. 额外搜索短上下文缩放系数，在长度小于约 8K 时使用，以恢复原短上下文能力。

论文报告其非均匀插值在无微调场景下可实现约 8 倍扩展，并最终用不超过 1000 个、最长 256K 的微调步骤扩展到 2048K。[6]

### 7.4 优点

- 针对具体模型和目标长度优化，灵活性高；
- 同时处理维度与 token 位置的非均匀性；
- 渐进式训练避免直接在百万 token 长度微调；
- 提供短上下文 readjustment，减少短任务能力损失；
- 已集成到 Microsoft Phi-3 的部分 128K 模型中。[7]

### 7.5 局限

- 需要额外搜索流程和验证样本；
- 缩放因子依赖模型、head dimension、原训练长度和目标长度，不宜跨模型复制；
- 百万级“可接受输入长度”不等于所有百万级任务都可靠；
- 训练和推理的显存、通信与注意力计算成本仍然存在，RoPE 扩展并没有消除 $O(L^2)$ attention 成本；
- 需要框架正确支持 per-dimension factor 数组和短/长上下文切换。

## 8. 总览对比

| 方法                  | 频率缩放策略                              | 是否通常需要微调     | 短上下文保持 | 主要优势                | 主要风险             |
| ------------------- | ----------------------------------- | ------------:| ------ | ------------------- | ---------------- |
| PI / Linear Scaling | 所有 $\omega_i$ 统一除以 $s$              | 推荐需要         | 一般     | 最简单、稳定、兼容性强         | 高频分辨率同步损失        |
| Static NTK-aware    | 通过增大 base 连续非均匀缩放                   | 可零微调试用       | 中等     | 保留高频，零微调常优于 PI      | 参数语义不统一，部分维度外推   |
| Dynamic NTK         | base 随当前序列长度变化                      | 可零微调试用       | 较好     | 短序列保持原 RoPE，逐步扩展    | KV Cache 频率一致性复杂 |
| YaRN                | 高频外推、中频混合、低频 PI；加 attention scaling | 推荐少量微调       | 较好     | 32K–128K 扩展质量与效率平衡好 | 启发式边界和框架配置差异     |
| LongRoPE            | 搜索每维 factor + 起始 token 阈值；短/长两套因子   | 大倍率通常需要分阶段微调 | 好      | 可针对模型优化并扩展到百万级      | 搜索和工程复杂度最高       |

## 9. 如何选择

### 9.1 从 4K / 8K 扩展到 8K / 16K

如果没有训练预算，先测试 **Dynamic NTK**，同时保留原 RoPE 作为短序列基线。必须验证：

- 原长度以内的 perplexity 是否退化；
- 目标长度附近是否突然崩溃；
- KV Cache 开启和关闭的结果是否一致；
- chunked prefill 与普通 prefill 是否一致。

如果允许少量训练，PI / Linear Scaling 配合长上下文微调是最简单可靠的基线。

### 9.2 扩展到 32K

优先级通常为：

1. **YaRN + 少量长上下文微调**：质量、训练成本和框架兼容性平衡较好；
2. **PI + 微调**：实现最简单，适合先建立可复现基线；
3. **Dynamic NTK**：适合快速零微调评估，但不应只凭能生成到 32K 就认定成功。

### 9.3 扩展到 64K / 128K

建议使用：

- 已有成熟 YaRN 配置的模型和框架；
- LongRoPE / 已发布的 per-dimension scaling 配置；
- 分阶段长上下文微调，而不是一次把训练长度从 4K 跳到 128K。

训练数据应混合短文本和长文本，避免模型只适应长位置而损失短任务表现。

### 9.4 扩展到 1M 以上

这不再是修改一个 RoPE 参数的问题。更合理的路线是 LongRoPE 式流程：

- 搜索模型专属的 per-dimension factors；
- 先扩展并微调到中间长度，例如 128K / 256K；
- 在扩展后的模型上进行二次搜索；
- 为短上下文单独 readjust；
- 配合稀疏注意力、分块注意力、Ring Attention、Context Parallel 或其他系统优化解决计算和显存问题。

RoPE 只解决位置分布问题，不会凭空把二次复杂度变成慈善项目。

## 10. 工程实现检查清单

### 10.1 配置语义

确认以下字段是否来自**原始模型配置**，不要把扩展后的值反过来当原始值：

- `rope_theta` / base；
- `head_dim` 与实际 rotary dimension；
- `partial_rotary_factor`；
- `original_max_position_embeddings`；
- `factor`；
- YaRN 的 `beta_fast`、`beta_slow`、`attention_factor`；
- LongRoPE 的 `short_factor` 与 `long_factor` 数组长度。

Hugging Face 当前官方 RoPE 工具支持 `default`、`linear`、`dynamic`、`yarn`、`longrope` 等类型，但不同 Transformers 版本与具体模型仍可能使用 `rope_scaling` 或 `rope_parameters` 等不同字段。[8]

### 10.2 KV Cache

对 Dynamic NTK / Dynamic YaRN：

- 检查已缓存 Key 是 pre-RoPE 还是 post-RoPE；
- 检查序列长度增长时是否重算频率；
- 检查旧 Key 是否需要重新旋转；
- 对比 `use_cache=False` 和 `use_cache=True` 的 logits；
- 对比一次性 prefill 与分块 prefill 的 logits。

### 10.3 数值精度

长位置下的相位乘法可能对低精度计算更敏感。通常应在至少 FP32 中生成 inverse frequencies、position-frequency outer product、sin/cos，再转换到模型 dtype。需要重点测试：

- BF16 / FP16 / FP8 下的相位误差；
- 非连续 position IDs；
- sequence parallel / context parallel 后的位置编号；
- packed sequence 中每个样本的位置重置逻辑。

### 10.4 不要只改最大长度

至少同时确认：

- tokenizer / data collator 不会提前截断；
- causal mask 能覆盖目标长度；
- FlashAttention 或其他 kernel 支持目标长度；
- KV Cache 分配与分页策略可承受目标长度；
- RoPE cache 没有按原始长度固定分配；
- 训练位置编号与推理位置编号完全一致。

## 11. 如何验证“有效上下文长度”

建议把验证分成四层。

### 11.1 数学与实现一致性

- 对同一 $p,i$，验证 position scaling 与 inverse-frequency scaling 的相位相等；
- 验证原长度以内的动态方法退化为原 RoPE；
- 验证不同 batch size、prefill chunk 大小和 KV Cache 策略输出一致。

### 11.2 语言建模

在长文档上绘制 sliding-window perplexity 随上下文长度的曲线。重点观察：

- 原训练长度以内是否退化；
- 超过某个长度后是否出现 perplexity cliff；
- 目标长度附近是否仍稳定，而不是只在开头稳定。

### 11.3 检索能力

使用 passkey / needle-in-a-haystack，覆盖：

- 不同上下文长度；
- needle 在开头、中间、末尾的不同深度；
- 多 needle 和干扰项；
- 需要组合多个远距离证据的任务。

单个五位数 passkey 成功，只能证明模型偶尔能找到一根针，不能证明它理解了整片草垛。

### 11.4 短上下文回归

长上下文扩展后仍需测试原任务：

- 常规语言模型 perplexity；
- MMLU、代码、数学和指令遵循；
- 4K / 8K 输入下的速度与质量；
- 不同 prompt 长度上的 attention entropy。

## 12. 方法关系图

```text
Original RoPE
    |
    +-- Position Interpolation (PI)
    |       `-- 工程实现名：Linear Scaling
    |
    +-- NTK-aware Scaling
    |       +-- Static NTK-aware：改变 base
    |       +-- Dynamic NTK：base 随当前长度变化
    |       `-- NTK-by-parts：按频率区间混合
    |               `-- YaRN：NTK-by-parts + attention scaling
    |
    `-- LongRoPE
            +-- 每个维度独立 factor
            +-- 起始 token 阈值
            +-- 进化搜索
            +-- 渐进式扩展
            `-- 短上下文 readjustment
```

## 13. 参考文献与原始材料

[1] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu. **RoFormer: Enhanced Transformer with Rotary Position Embedding**. arXiv:2104.09864, 2021.  
<https://arxiv.org/abs/2104.09864>

[2] Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian. **Extending Context Window of Large Language Models via Positional Interpolation**. arXiv:2306.15595, 2023.  
<https://arxiv.org/abs/2306.15595>

[3] bloc97. **NTK-Aware Scaled RoPE allows LLaMA models to have extended context size without fine-tuning**. Original community post, 2023.  
<https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/>

[4] emozilla. **Dynamically Scaled RoPE further increases performance of long-context LLaMA with zero fine-tuning**. Original community post, 2023.  
<https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/>

[5] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. **YaRN: Efficient Context Window Extension of Large Language Models**. arXiv:2309.00071, 2023; updated version 2026.  
<https://arxiv.org/abs/2309.00071>  
Official implementation: <https://github.com/jquesnelle/yarn>

[6] Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, Mao Yang. **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens**. ICML 2024; arXiv:2402.13753.  
<https://arxiv.org/abs/2402.13753>

[7] Microsoft Research. **LongRoPE official repository and Phi-3 integration notes**.  
<https://github.com/microsoft/LongRoPE>  
<https://www.microsoft.com/en-us/research/publication/longrope-extending-llm-context-window-beyond-2-million-tokens/>

[8] Hugging Face Transformers. **Utilities for Rotary Embedding / modeling_rope_utils.py**. Official implementation reference for linear, dynamic NTK, YaRN and LongRoPE.  
<https://huggingface.co/docs/transformers/internal/rope_utils>  
<https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py>

[9] Xiaoran Liu et al. **Scaling Laws of RoPE-based Extrapolation**. arXiv:2310.05209, 2023.  
<https://arxiv.org/abs/2310.05209>

## 14. 最小记忆版

- **PI / Linear**：所有频率一起除以 $s$。
- **NTK-aware**：高频少缩放，低频多缩放；通常通过改变 base 实现。
- **Dynamic NTK**：根据当前序列长度动态改变 NTK 缩放。
- **YaRN**：高频不缩放，中频渐变，低频做 PI，再修正 attention 温度。
- **LongRoPE**：不再手写缩放曲线，而是搜索每个维度和起始位置的最佳缩放，并分阶段扩展。
