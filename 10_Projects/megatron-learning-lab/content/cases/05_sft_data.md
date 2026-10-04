---
type: knowledge
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# 从一条对话到 SFT 的有效目标

这里接续完整 Qwen-style 模型的 input → embedding → decoder → loss。先决定“什么是一个目标”，模型再对每个输入位置产生下一 token 分布。页面保留原始 messages；人工 token fixture 只验证数据机制，不声称调用官方 tokenizer 或运行 Bridge。AI 整理，待人工核验。

## 独立数学符号表

| 符号 | 含义 / shape |
|---|---|
| $B$ | 未打包时样本行数 |
| $S_b$ | 样本 $b$ 的有效输入长度，包含声明的控制符 |
| $x_{b,t}$ | 输入 token 的教学词表索引 |
| $u_{b,t}$ | 未移位监督 mask，取 0 或 1 |
| $y_{b,t}$ | 位置 $t$ 的目标 token；末尾无目标 |
| $m_{b,t}$ | 与 target 对齐的 loss mask |
| $p_{b,t}(v)$ | 给定样本前缀，对词表 token $v$ 的预测概率 |
| $\ell_{b,t}$ | token 交叉熵，单位 nat |
| $N$ | 所有有效目标的 mask 总和 |
| $L$ | 全局有效 token 平均 loss |
| $A_{ij}$ | attention 允许矩阵，true 表示可读 |
| $c_b$, $\tilde c_b$ | 不含 padding / 含 padding 的累计长度 |
| $a$ | 每个 packed 样本的 padding 对齐倍数 |

## 单次 shift 和 target mask

未移位序列长为 $S_b$。对 $0\le t<S_b-1$：

$$
y_{b,t}=x_{b,t+1},\qquad m_{b,t}=u_{b,t+1}.
$$

最后一个输入位置保留，label 为 ignore_index=-100，mask=0；中间不监督的 label 同样设为 -100。不能把 input 的 role 当成 target 的监督角色。人工例子中的 assistant 控制符本身不监督，但它的位置预测第一个 response token，那个目标要监督。

多轮样本的索引 7 输入是 assistant 控制符，目标为“5”；索引 8 的输入“5”预测“。”；索引 9 的输入“。”预测 EOS。这三项 mask 都为 1。assistant 模式两次回复合计 6 项，last_turn 为 3 项；full 为所有有效相邻目标，BOS 自己不会成为被监督目标。

固定 Bridge direct-HF 路径：DirectSFTDataset 绑定 collate_fn（B-DIRECTSFT）；text_chat_collate_fn 构造未移位 mask 后，由 _build_text_sft_batch 调 build_shifted_labels_and_loss_mask（B-SFTCOLLATE / B-CONVERSATION）。该函数保留输入长度，在尾部补 ignore_index 和零 mask。gpt_step 将 tokens/labels 交给 GPT，loss callback 再做 mask sum/count（B-STEP / B-LOSS）。这里不从 HF 模型常见内部 shift 行为类推所有 Bridge 路径；legacy GPT-SFT、MTP、其他 collator 要分别确认。

三种模式：assistant 监督全部回复；last_turn 只监督最后回复；full 监督完整有效序列。真实模板的 role 边界、special-token 排除由源码 apply_chat_loss_mode 与 tokenizer 共同决定。固定实现的 last_turn 在可获知时用最终回复 prefix 边界，否则退到最后连续 assistant-mask span。fixture 已知精确 message 边界，故不模拟该回退分支。

## 截断、空回复与 padding

fixture 在末尾截断，不补造被截掉的 EOS；last_turn 的“最后”按原始 messages 决定。截断掉最后回复时得到 N=0，而不是悄悄监督上一轮。空 response 依本 fixture 约定仍有 EOS 目标，真实模板是否相同留 R01 核验。

padding 是物理对齐，不是训练样本，不进入 token count 和上下文。教学图里 padding query 全零且不执行 softmax；不能对全为负无穷的分数做 softmax 并把 NaN 当成训练结果。真实 fused backend 对 padding 的处理属于其接口合约。

## Packing 保留独立问题

先各自构造 labels/mask，再拼接；样本末尾 label 已被忽略，不能令前一条 EOS 预测后一条 BOS。position 每个样本从零起。正确允许矩阵满足：

$$
A_{ij}=[\mathrm{sample}(i)=\mathrm{sample}(j)]\,[\mathrm{pos}(j)\le\mathrm{pos}(i)]\,[i,j\text{ 均有效}].
$$

仅有全局 causal 三角矩阵不够：第二个样本会读取第一个样本。页面可点击 query 查看哪些 key 被允许，故障模式实际重算泄漏后的参考 loss。

长短样本未打包是两行 [B,Smax]；打包可变成一条物理 [1,T]，这里的 microbatch=1 不等于一个原始样本。THD 的 $T$ 是 token 轴，H/D 是 attention head 轴，不要解释为多个独立 batch。

$$
c_{b+1}=c_b+S_b,\qquad
\tilde c_{b+1}=\tilde c_b+a\left\lceil S_b/a\right\rceil.
$$

固定 B-PACK 分别生成 cu_seqlens_q/kv 与可选的 cu_seqlens_q/kv_padded；后者描述物理偏移。将二者混成一个数组会让后续样本读错位置。没有额外 padding 时两者相等。表中保留每个样本内 position，padding 区延续编号但仍被排除。

## 从 token loss 到全局归约

$$
\ell_{b,t}=-\log p_{b,t}(y_{b,t}),\quad
N=\sum_{b,t}m_{b,t},\quad
L=\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}{N}.
$$

两项有效目标的概率是 1/2、1/4 时，loss sum 为 $3\log 2$，N=2，均值约 1.03972077084 nat。若一组一个目标、loss=1，另一组三个目标、每项 loss=3，则全局均值是 2.5，而均值之均值是 2。mask 为零只消除该位置的直接监督，不会阻止 prompt 表示影响后续受监督预测和梯度。

页面的 loss 使用小型因果上下文函数验证数据机制：query/key 为教学 ID 除以 10，value 为 ID 除以 7；softmax 加权 value 得到标量 context，再令第 v 个 logit 为 cos((v+1)(context+1)+position/3)/4。它是具备上下文依赖的确定性参考，不是 Qwen decoder 的 loss，更不是训练结果。G03 用完整微型 decoder 接续。

## 验证和故障边界

Python 标准库参考独立计算编码、mask、softmax/CE；TS 是页面实际使用的模块。96 组配置覆盖三种模式、单/双样本、截断和 padding；容差预先固定 atol=rtol=1e-11。packed/unpacked 的有效目标序列、sum/count 和 loss 必须一致。重复 shift、mask 跟随 input、跨样本预测、跨样本 attention 和均值之均值都是故障测试，不是只检查数组存在。

外部 tokenizer trace 读取器只接收结构化文本字段与有限大小数组，检查 revision/template hash、mask 长度和整数 ID。导入标 external_unverified，不以自报 revision 证明官方执行，不运行 JSON 中的任何代码。真实 template、special token 边界及同权重训练数据链路在 R01 实测。
