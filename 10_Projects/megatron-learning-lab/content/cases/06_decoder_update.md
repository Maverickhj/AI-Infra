---
type: knowledge
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# 同一条 SFT 样本经过完整微型 decoder

这是两层 architecture-scaled Qwen3-style 参考：V=27、H=8、Q heads=4、KV groups=2、head_dim=4、FFN=12。输入取 G02 的完整人工 token 序列，权重是显式 authored 数值，没有下载或加载 Qwen checkpoint。CPU autograd/update 是实际执行的教学参考；HF/Bridge SFT 仍未运行。

## 独立数学符号表

| 符号 | 含义 / shape |
|---|---|
| $S,V,H,F$ | 序列长度、词表数27、隐藏维8、FFN维12 |
| $E$ | embedding 存储矩阵 $[V,H]$ |
| $X,R,Y$ | 某子步骤的输入、残差输入、输出 $[S,H]$ |
| $\gamma,\epsilon$ | RMSNorm 的增益 $[H]$ 与稳定项1e-6 |
| $W_g,W_u,W_d$ | gate/up 存储 $[F,H]$，down 存储 $[H,F]$ |
| $G,U,Z$ | gate/up/product 激活 $[S,F]$ |
| $W_h$ | head 存储 $[V,H]$，tied 时就是 $E$ |
| $z_t,p_t,y_t,m_t$ | logit向量、概率、目标ID、target mask |
| $N,L$ | 有效目标数、masked mean CE |
| $\theta,g,\eta$ | 参数、总梯度、学习率0.03 |
| $v_k,\mu$ | SGD momentum buffer、动量系数0.9 |

## Norm 从实际 residual stream 进入

$$
\mathrm{RMSNorm}(X)_{t,i}=\frac{X_{t,i}\gamma_i}{\sqrt{\frac1H\sum_j X_{t,j}^2+\epsilon}}.
$$

输入 RMSNorm 后是已有 GQA 运算图；attention 输出投影回 H 维并加入残差。FFN 之前再次归一化。最后一层结束还有独立 final norm；这三处并不共享参数。页面展示选定 token 的实际分母、输入和输出前8维。

Core 的 norm 可能在 TE 的 LayerNormLinear 中融合。C-BLOCK 的 final_layernorm 构造/调用是明确源码入口；它不是 norm kernel。无 MTP 时主要由 post_process/post_layer_norm 决定所属 stage；有 MTP 时该固定版本检查含最后 decoder 层的 stage。不能把“永远在最后物理 PP stage”写成所有分支规则。

## 两层 attention 与 SwiGLU

每层 GQA 直接复用 G01 Python/TS 运算图。这里仅把有界序列长度从4扩展到1–64，并给同一标量图提供 PyTorch sqrt/exp，以保留 autograd；没有新增另一套 attention 算法。G01 四 token 数值与因果反例仍独立复验。

存储权重为 [out,in]，逻辑计算右乘转置：

$$
G=\mathrm{RMSNorm}(R)W_g^\top,\qquad U=\mathrm{RMSNorm}(R)W_u^\top,
$$

$$
Z=\mathrm{SiLU}(G)\odot U,\qquad
\mathrm{SiLU}(g)=g\,\sigma(g),\qquad Y=R+ZW_d^\top.
$$

融合 fc1 的存储形状是 [2F,H]=[24,8]，前后各12维分别为 gate/up。对 up 做 SiLU、把 gate 仅做 sigmoid、把 product 当矩阵乘法都会改变结果。C-MLP 的真实实现保留 fused 和 unfused 分支；教学参考选显式、无 bias、无 FFN dropout、glu_linear_offset=0、无 clamp 的路径。实际 TE 可能把 norm 与线性投影融合，片段必须按配置理解。

## Head 与 masked CE

最后 norm 输出右乘 $W_h^\top$ 得到 [S,V] logits。tied 使用同一个 embedding 参数，untied 使用独立 head。两者初始数值相同，所以初始 forward 可以相同，但梯度归属不同。

$$
\ell_t=-\log p_t(y_t),\quad
p_t=\mathrm{softmax}(z_t),\quad
L=\frac{\sum_t m_t\ell_t}{N},\quad N=\sum_t m_t.
$$

对 logits 的解析梯度：

$$
\frac{\partial L}{\partial z_{t,v}}=\frac{m_t}{N}\left(p_t(v)-[v=y_t]\right).
$$

因此 untied head 的解析梯度等于上式矩阵转置乘 final norm 激活，CPU 测试直接核对它与 autograd。tied 参数则要汇集 lookup 和 head 两条路径：

$$
\nabla_E L=\nabla_E^{\mathrm{lookup}}L+\nabla_{W_h}^{\mathrm{head}}L.
$$

测试把同初始权重的 untied 梯度相加，对照 tied 总梯度；漏掉 head 项的反例确实不等价。mask=0 的 prompt 位置没有直接 CE，但后续目标的 attention 仍依赖其 activation，页面显示该位置的非零梯度。不要将 mask 当成 detach。

## Backward、更新与恢复

CPU 使用实际 PyTorch autograd；不是前端动画倒放。对 embedding、QKV、FFN、final gain 的选定元素，用中心差分：

$$
g_{\mathrm{FD}}=\frac{L(\theta+h)-L(\theta-h)}{2h},\qquad h=10^{-5}.
$$

梯度验收预先固定 $|g-g_{\mathrm{FD}}|\le10^{-6}+10^{-4}|g_{\mathrm{FD}}|$；有限差分期间关闭 dropout，避免用随机变化冒充梯度。前向 Python/TS float64 对照使用 atol=rtol=1e-10。将 final gain 置零、重复 shift、用全序列长度作分母都由反例检测。

$$
v_k=\mu v_{k-1}+g_k,\qquad\theta_{k+1}=\theta_k-\eta v_k.
$$

第一次更新 momentum 从零开始，退化为 $\theta'=\theta-0.03g$；页面可逐项手算。冻结的 l0_input_gain 没有 grad 且值完全不变。保存/恢复测试实际序列化模型参数、optimizer momentum 和 CPU RNG，再对同一输入执行下一步。该测试开启0.1的 embedding dropout，以检验 RNG 恢复；故意扰动 RNG 会导致下一步不同。连续与恢复的下一步 loss/参数须完全一致。

## 样本、来源与局限

浏览器的 forward 跟随当前 authored 样本/监督模式/token/layer/head共享选项；CPU 梯度卡固定绑定 arithmetic-multiturn / assistant / tied，显示其采集时间、代码 hash 与参数切片，不把其他选项的画面冒充重新执行的 CPU 训练。改变源码后 data:decoder 重新生成小型证据，完整 gate 重新执行。

随机或人工权重的 CPU loss 下降，只证明一次可导更新与恢复流程，不代表预训练 SFT、GPU 性能或能力提高。真正 checkpoint、官方 tokenizer、跨引擎对齐与 HF export 在 R01 验证；训练曲线仍无实测 GPU 数据。
