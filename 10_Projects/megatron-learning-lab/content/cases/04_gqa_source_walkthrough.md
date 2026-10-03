---
type: knowledge
status: draft
created: 2026-10-03
updated: 2026-10-03
ai_generated: true
reviewed: false
---

# GQA：一个 query 如何从整层输入走到 residual

> 本文为 AI 编写、待人工核验的教学单元。数值来自确定性 CPU reference；真实 HF、Bridge、RL 均为 not_run。

## 从完整模型进入

保留 tokens→embedding→decoder层栈→final norm→head→loss。这里只展开选中层的 attention，输出仍送入同层 FFN。无需重学矩阵乘法；要回答的是：哪个 Q 使用哪组 K/V、mask 在哪条轴上、为何输出必须投影回 residual 宽度。

真实 Qwen3-0.6B 配置 H=1024、Q heads=16、KV heads=8、d=128，Q 宽2048，输出投影2048→1024。Qwen2.5-0.5B H=896、Q heads=14、KV heads=2、d=64，Q 宽896。结构来自 config，标签是 derived，不能解释成已加载的权重。

教学 fixture 另取 B=1、S=4、H=8、n_q=4、n_kv=2、d=4。刻意保留 H≠n_q d，投影16→8。数据、权重均人工构造，architecture_origin=authored_scaled_gqa、weights_origin=authored_fixture、evidence=reference。每层复用同一 fixture，选层只改变整模和源码上下文，绝非该层的观测激活。

## 独立数学符号表

| 符号 | 意义 / 本单元逻辑 shape |
|---|---|
| B,S,H | batch、序列长度、residual宽度；1、4、8 |
| n_q,n_kv,d | Q heads、KV groups、每head维度；4、2、4 |
| t,j,h,g(h) | query位置、key位置、Q head、对应KV group，均从0编号 |
| X,U | attention原始输入、输入RMSNorm输出；[B,S,H] |
| γ,γ_Q,γ_K,ε | 输入及Q/K norm gains；ε=10⁻⁶，在均方后、开方前相加 |
| W_QKV,b,M | 逻辑[in,out] QKV权重[8,32]、bias[32]、mixed输出[1,4,32] |
| Q,K,V | [1,4,4,4]、[1,4,2,4]、[1,4,2,4] |
| Q̂,K̂,Qᴿ,Kᴿ | head norm前后及RoPE后Q/K；shape保持 |
| p,θ,φ_i | position ID、RoPE theta=10000、角度p θ^(-2i/d) |
| s,z,z̃,P | 原始score、缩放score、masked score、概率；[B,S,n_q,S] |
| O,C | 加权V的head输出[1,4,4,4]、合并结果[1,4,16] |
| W_O,b_O,Y,R | 投影权重[16,8]、零bias[8]、投影输出和residual输出[1,4,8] |

数组为简洁省略 B=1 轴。数字显示舍入6位；JSON导出保留浮点精度，负无穷编码为字符串 `-Infinity`，不是有限零值。

## 1. 保存 residual，再做 RMSNorm

$$
U_{t,i}=\gamma_i\frac{X_{t,i}}{\sqrt{\frac1H\sum_{k=0}^{H-1}X_{t,k}^2+\epsilon}}.
$$

沿hidden轴归一化，不减均值，不能错当LayerNorm。最终加回的是X，不是U。输入shape和输出同为[1,4,8]。fixture显式列出input_gain，不依赖随机种子或隐式默认值。

源码 C-LAYER 的输入norm/residual片段是调用入口：spec可把norm融合进QKV线性层，不能凭概念图声称真实执行有单独的norm kernel。验证可用零输入检查ε避免零除，再比较非零输入所有分量的共同缩放因子。

## 2. 先投影，再按组拆分

$$
M=UW_{QKV}+b,\qquad g(h)=\left\lfloor\frac{h}{n_q/n_{kv}}\right\rfloor.
$$

这里每组2个Q。group0存 `[Q0,Q1,K0,V0]`，group1存 `[Q2,Q3,K1,V1]`，每个head连续4列。投影后32列的前16列包含第一组的K/V，绝非“前16列全是Q”。C-ATTN grouped split与B-QKV参数映射说明的是这种布局。

Qwen3-style关闭QKV bias并启用QK norm；Qwen2.5-style加上fixture显式bias并关闭QK norm。两者都仍是缩小的教学结构，不是正式模型权重。验证：两种mixed输出之差应逐元素等于bias；h=0/1应读group0，h=2/3应读group1。

## 3. Q/K head norm 与 RoPE 的顺序

Qwen3-style对Q/K每个head的4维向量分别RMSNorm，Q/K使用独立gain；V不做head norm。C-ATTN的QK norm片段在普通路径按模块存在性执行。Qwen2.5-style旁路，norm关闭来自模型档案。B-Q2片段直接证明QKV bias配置，没有显式qk_layernorm=false；另可查看C-ATTN的模块存在性条件，不能把条件定义假装成当前执行了norm。

随后对Q/K做split-half RoPE。d=4时配对(0,2)、(1,3)，角度分别p和p/100；position IDs明确为[0,1,2,3]。

$$
\begin{pmatrix}a'\\b'\end{pmatrix}=
\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
$$

C-ROPE的`rotary_interleaved=false`分支把前后两半配对。若把(0,1)配对会得到另一种布局；即使维度正确也不是本fixture声明的计算。位置0应原样返回，每对平方和保持。C-ATTN另有RoPE调用入口，真实fused与动态推理路径不应混成这段普通实现。

## 4. 点积、缩放、mask：先保留分数，再限制可见性

$$
s_{t,h,j}=Q^R_{t,h}\cdot K^R_{j,g(h)},\qquad z_{t,h,j}=s_{t,h,j}/\sqrt d.
$$

选定query/head后，页面展开四项乘积及其和，并列出该head对所有key的score。这里d=4，故除以2；不是除以H=8，也不是除以head数4。遗漏缩放不会改变shape，却改变softmax和最终输出，页面可实际运行此反例。

$$
\widetilde z_{t,h,j}=\begin{cases}z_{t,h,j}&j\le t\\-\infty&j>t.\end{cases}
$$

query0只看key0；query2看key0、1、2。未来位置填负无穷，乘0是错误反例：零仍可能占概率。fixture无padding、packing、attention bias或滑动窗口；真实batch的mask必须按实际接口转换，不能直接把这里“true表示允许”传给所有后端。

## 5. Softmax与读取V

$$
P_j=\frac{\exp(\widetilde z_j-m)}{\sum_k\exp(\widetilde z_k-m)},\quad
m=\max_k\widetilde z_k,\qquad
O_{t,h,i}=\sum_jP_{t,h,j}V_{j,g(h),i}.
$$

softmax沿key轴，不是head轴或feature轴。减最大值避免指数溢出；masked位置恰为0，每个允许行和为1。全mask行在本教学实现报错，而不是输出伪造的全零或NaN成功结果；这不是对所有Megatron backend的行为声明。

同组两个Q共享K/V却可能有不同概率。query0只允许一个key，因此其head输出恰为V[0,g]；query3则是四个V的加权和。页面的“错group”同时改错K与V读取，结果与正确路径比较，不能只变一个标签。

这里的score→scale→mask→softmax→V是数学等价参考。对应真实证据是C-ATTN `core_attention`调用边界；TE/FlashAttention可分块融合计算，不保证物化完整[S,S]矩阵。没有把本地标量Python/TS程序冒充TE源码或性能实现。

## 6. 合并、投影、加回原始输入

$$
C_t=\operatorname{concat}_h O_{t,h},\quad Y=CW_O+b_O,\quad R=X+Y.
$$

合并4个4维Q head得到16维，不求均值；W_O把16投回8，才能与8维X相加。C-ATTN linear_proj负责投影，C-LAYER bias/dropout/add入口连接residual。教学b_O=0、dropout=0，不测试训练随机数或并行通信。

核验：C[t,4h+i]=O[t,h,i]；展开任意Y[t,j]的16项乘加；R-X必须等于Y。不要把归一化后的U作为residual，也不要直接把16维C与8维X相加。

## 7. 本单元怎样验证

Python标准库参考不读取前端结果；TypeScript生产模块同时被页面与`npm run test:gqa`调用。双方比较norm、mixed QKV、Q/K/V、head norm、RoPE、score、scale、probabilities、head output、merged、projection与residual。mask负无穷按位置单独断言。

固定容差仅用于这个float64小fixture：abs_error≤10⁻¹⁰+10⁻¹⁰×abs(reference)。还需独立因果测试：逐个边界改变所有未来输入，比较最终residual的更早行不变。行和、mask零概率、KV group与投影维度均有断言。head/token小数、越界、NaN、空输入必须拒绝。

数值通过不证明真实BF16/GPU parity、实际权重导入或训练正确。Final norm专属源码仍可显示缺口；HF/Bridge/RL保持not_run。本目标只交付GQA参考精讲，等待用户审阅，不自动启动SFT/RL训练。
