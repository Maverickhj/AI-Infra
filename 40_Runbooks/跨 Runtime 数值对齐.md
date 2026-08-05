---
type: runbook
status: seed
created: 2026-08-05
updated: 2026-08-05
domains: [numerics, inference, training]
aliases: [Cross-runtime Numerical Alignment]
tags: [alignment, sglang, megatron, transformers]
ai_generated: true
reviewed: false
---

# 跨 Runtime 数值对齐

## 适用场景

Megatron、Transformers、SGLang 或其他训练/推理 runtime 在 hidden states、logits、selected-token log-prob 上出现差异。

## 对齐原则

从输入身份向模型末端逐级证明。不要先比较最终 log-prob，再从几十个潜在差异中猜原因。

## Phase 0：冻结变量

记录：

- 两侧 commit、模型配置和 checkpoint/weight version。
- hardware、kernel backend、dtype/autocast/quantization。
- TP/PP/CP/DP、batching、prefill/decode 方式。
- 关闭随机采样或固定 seed；禁用不相关的 dump/hook。

## Phase 1：证明是同一个请求

同时匹配：

- request/session/sample ID。
- 原始 prompt 与媒体 hash。
- token IDs hash、长度和关键 special tokens。
- processor 版本与媒体预处理 metadata。

仅按执行次序、pass 编号或 module name 匹配不可靠，尤其在 continuous batching 和多进程环境中。

## Phase 2：证明模型输入一致

比较：

- token embeddings 或注入前 hidden states。
- position IDs / RoPE inputs。
- attention mask、cu-seqlens、packed boundaries。
- media features、offset 和 injection positions。

任何一项不一致都应先在其 producer 层修复。

## Phase 3：证明权重映射一致

- 参数名映射、transpose/reshape、QKV packing 顺序。
- TP/EP shard 的 global coordinate 与 gather 规则。
- PP/VPP stage 是否拥有目标层。
- MTP、vision/projector、embedding/LM head 等特殊层是否完整转换。
- weight sync 的完成时刻是否早于目标请求。

可对关键参数计算 per-shard 与重建后 hash，但 hash 一致仍不证明 forward 语义一致。

## Phase 4：逐层定位首个分叉

建议 capture 点：

1. embedding / media injection 输出。
2. 每层 norm 输入输出。
3. Q/K/V 或 attention 输出。
4. MLP/MoE 输出。
5. residual 后 hidden state。
6. final norm、LM head raw logits。
7. processed logits 和 selected-token log-prob。

先用 summary（shape、dtype、min/max/mean/norm、NaN/Inf、hash）定位，再保存少量完整 tensor，避免 dump 本身造成 OOM 或并发竞争。

## Shard 变换

比较前统一到同一坐标系：

- 明确 global tensor path 与 layer index。
- 记录 TP/PP/CP/EP rank。
- 按真实参数/激活分片轴 gather 或 slice。
- 去除 vocabulary padding 或 storage-only padding。
- 对 packed token 使用 request offsets 选择相同 token。

## 误差判断

同时报告：

$$
\max |x-y|,\quad \operatorname{mean}|x-y|,\quad
\frac{\lVert x-y\rVert_2}{\lVert x\rVert_2+\epsilon}.
$$

还应报告 dtype、元素数、阈值、NaN/Inf 和 top-k token 是否改变。阈值应根据 dtype、层数与业务语义设定，不存在一个适用于所有模型的绝对容差。

## 首个分叉后的检查顺序

1. 输入/权重 layout 或 shard mapping。
2. mask、position、cache offset。
3. autocast、内部累积与输出 dtype。
4. fused/native/compiled 分支及优先级。
5. epsilon、归约顺序和近似函数。
6. 未完成同步、错误请求映射或 dump 竞争。

## 完成标准

- 找到首个可重复分叉点。
- 解释差异由哪个具体 producer/分支造成。
- 修复后从该点到最终消费端满足明确阈值。
- 保留输入身份、权重版本、shard mapping 和实验环境。

## 相关笔记

- [[20_Knowledge/Inference/推理系统与训练对齐|推理系统与训练对齐]]
- [[20_Knowledge/Transformer/张量形状与计算约定|张量形状与计算约定]]
- [[20_Knowledge/Multimodal/多模态模型数据流|多模态模型数据流]]
