## IDEA-20260815-Megatron-LM sequence packing 性能问题

> [!warning]
> 本文由 AI 根据原始想法整理，尚未完成人工核验。

- **记录时间**：2026-08-15
- **状态**：seed

### 一句话

梳理 Megatron-LM 在单个 batch 的 forward-backward 中使用 sequence packing（pack）或不使用（no pack）时，对 `micro_batch_size` 和训练性能的影响。

### 待确认问题

- pack 后，`micro_batch_size` 是否可以大于 `1`？需要明确其在 batch 组织、模型输入和 kernel 调用中的具体语义与限制。
- pack 在什么数据分布、序列长度、模型配置和并行策略下会比 no pack 更快？
- 在可比条件下，pack 相对 no pack 的吞吐提升通常是多少？应如何定义和测量提升？
- 是否存在 pack 比 no pack 更慢的场景？若存在，瓶颈来自数据准备、attention kernel、负载不均衡、通信，还是其他环节？

### 初步代码观察（2026-08-15）

当前本地 Megatron-Bridge 的 Qwen3-VL 路径中，`src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py::pack_or_pad_batch_sequences` 从 `tokens.shape` 取得 `batch_size`，并构造长度为 `batch_size + 1` 的 `cu_seqlens` 后传入 `PackedSeqParams(qkv_format="thd")`。`pack_sequences_in_batch=True` 时会把该对象传给模型；这里没有 `micro_batch_size == 1` 的代码限制。因此，对该路径而言，`micro_batch_size > 1` 在数据结构上是允许的；实际可取上限仍受显存、TP/PP/CP/EP、CUDA Graph 和所选 kernel 支持情况约束。

同一函数会先将 `tokens` 的每一行 pad/truncate 到 `target_len`，再以每行均为 `target_len` 构造边界。故仅切换此开关不必然减少物理 token 数或 FLOPs；还需核查 dataset/collator 是否已将短样本真正组合为较少的有效 token，或是否能提供反映真实子序列长度的 `cu_seqlens`。

### 理论评估框架

可以先做理论上的 FLOPs、显存和 break-even 估算，但不能直接从 FLOPs 推出真实的百分比加速。前提是比较时固定模型、精度、并行配置和**有效训练 token 数**，并明确 pack 的含义：

1. **去 padding 的 ragged/varlen pack**：只为真实 token 计算；可直接减少单步计算。
2. **填满固定容量的 document packing**：把多个短样本填入固定长度容器；单步可能处理更多有效 token。此时应优先比较有效 token/s，不能只比较 step time。

设 `B` 为 `micro_batch_size`，`S` 为 no pack 中每个样本的物理序列长度，`L_i` 为 pack 后第 `i` 个由 `cu_seqlens` 划定、实际送入 kernel 的子序列长度，$T_{\mathrm{phys}} = \sum_i L_i$ 为 pack 的物理 token 数，$u = T_{\mathrm{phys}} / (BS)$ 为相对于 no pack 容量的物理 token 比例。有效训练 token 数应另由 `loss_mask` 统计，并在对照中保持一致。`H` 为 hidden size；`\alpha`、`\beta` 分别汇总全部 Transformer 层 forward-backward 中与 token 数成正比的线性/MLP 计算及与 token 对数成正比的 attention 计算系数。

在 dense Transformer、并且 packed attention 不跨越子序列边界的近似下：

$$
F_{\mathrm{no\ pack}} \approx \alpha B S H^2 + \beta B S^2 H,
$$

$$
F_{\mathrm{pack}} \approx \alpha T_{\mathrm{phys}} H^2 + \beta \left(\sum_i L_i^2\right) H.
$$

其中，线性/MLP 部分的理论节省比例为 $1-u$；attention 部分的理论节省比例为：

$$
1 - \rho_{\mathrm{attn}}, \qquad
\rho_{\mathrm{attn}} = \frac{\sum_i L_i^2}{B S^2}.
$$

相应的计算量加速上界可记为 $R_{\mathrm{FLOPs}} = F_{\mathrm{no\ pack}} / F_{\mathrm{pack}}$。若当前实现的每个边界均为 `L_i = S`，则 $T_{\mathrm{phys}} = BS$、$\rho_{\mathrm{attn}} = 1$，该模型预测不到来自“消除 padding”的 FLOPs 收益；即使这些位置的 `loss_mask` 为零，它们仍可能参与计算。

`micro_batch_size` 的显存上限也可先估算。设 $N_{\mathrm{layer}}$ 为层数，$d$ 为 activation dtype 字节数，$\gamma$ 为由保存的中间结果、重计算和并行切分决定的系数，则：

$$
M_{\mathrm{act}} \approx \gamma N_{\mathrm{layer}} T_{\mathrm{phys}} H d + M_{\mathrm{attn\ temp}} + M_{\mathrm{comm}},
$$

其中真正去 padding 时 $T_{\mathrm{phys}} = T$，固定形状或已 pad 的路径通常仍为 $T_{\mathrm{phys}} = BS$。只有总显存（参数、优化器状态、梯度、activation、临时缓冲和通信 buffer）低于可用显存时，增大 `B` 才可行。

### 从理论到真实速度

FLOPs 只给出上界；真实单步时间应写为：

$$
t_{\mathrm{no\ pack}} = \frac{F_{\mathrm{no\ pack}}}{P_{\mathrm{no\ pack}}} + t_{\mathrm{comm,no\ pack}} + t_{\mathrm{other,no\ pack}},
$$

$$
t_{\mathrm{pack}} = \frac{F_{\mathrm{pack}}}{P_{\mathrm{pack}}} + t_{\mathrm{prepare}} + t_{\mathrm{comm,pack}} + t_{\mathrm{other,pack}}.
$$

只有 $t_{\mathrm{pack}} < t_{\mathrm{no\ pack}}$ 时 pack 才更快。`P` 是实际 kernel 吞吐；它不是常数。pack 在下列情况可能反而更慢：$u$ 和 $\rho_{\mathrm{attn}}$ 已接近 `1`、变长/小矩阵降低 GEMM 或 attention kernel 效率、`cu_seqlens`/重排/DataLoader 的准备成本显著，或并行通信与负载不均衡抵消了计算节省。

因此，理论计算可以回答“预期有没有足够收益值得试”，并给出理论上限；要回答“快多少”，至少需要用一次小规模 profiler 测量来校准 $P$、准备时间和通信时间。

### 后续处理

- [ ] 从真实 batch 记录 `L_i` 直方图，计算 $u$、$\rho_{\mathrm{attn}}$、$F_{\mathrm{no\ pack}}$、$F_{\mathrm{pack}}$ 和 $R_{\mathrm{FLOPs}}$。
- [ ] 确认 dataset/collator 提供的 `cu_seqlens` 是否反映真实子序列边界，而非仅记录已 pad 的 `target_len`。
- [ ] 设计 pack 与 no pack 的对照实验，固定模型、有效 token 数、并行配置和精度；同时记录 step time 与有效 token/s。
- [ ] 用 profiler 校准 attention、GEMM、数据准备和通信时间，并记录显存峰值。

### 补充材料

- **相关笔记**：[[Qwen3.5-VL SFT 的 Megatron 代码链路与观测指南]]
- **待确认结论**：目前不预设 pack 一定更快或 `micro_batch_size` 的固定限制，等待代码核查与实测。

### 后续处理状态

<!-- 状态可选：seed、exploring、promoted、dropped。升格后在此链接正式项目。 -->

- **处理决定**：
- **正式项目**：
- **决定日期**：
- **原因**：
