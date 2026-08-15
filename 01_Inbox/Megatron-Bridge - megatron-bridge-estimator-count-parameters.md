---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains: []
aliases: []
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/codex/dev-v0.5.1-minimind/docs/debug/megatron-bridge-estimator-count-parameters.md
sync_source: docs/debug/megatron-bridge-estimator-count-parameters.md
ai_generated: false
reviewed: false
---

> [!todo]
> 自动剪藏自 `Maverickhj/Megatron-Bridge`，待整理到 `20_Knowledge/`。
> 来源：https://github.com/Maverickhj/Megatron-Bridge/blob/codex/dev-v0.5.1-minimind/docs/debug/megatron-bridge-estimator-count-parameters.md
> 同步脚本：`scripts/sync_ai_infra_inbox.py`；如要手工编辑，请先把本笔记移出 `01_Inbox/`。

## 原文元数据

```yaml
type: how-to
status: growing
created: 2026-08-15
updated: 2026-08-15
domains:
  - megatron
  - memory-planning
  - estimator
aliases:
  - _count_parameters 逻辑
related:
  - "[[qwen35-vl-800m-sft-memory]]"
  - "[[qwen35-vl-800m-sft-smoke]]"
  - "[[megatron-lm 链路可观测性 roadmap]]"
```

## 原文

# `_count_parameters` 计算逻辑

本文记录 `theoretical_memory_utils.py` 中 `_count_parameters()` 的参数估算逻辑。
该函数是 Megatron-Bridge 内置显存 estimator 的参数计数入口，位于：

`src/megatron/bridge/training/utils/theoretical_memory_utils.py:424`

## 1. 函数目标

`_count_parameters()` 不加载权重，也不遍历实际 module。它只读取
`model_config`，用公式估算三类参数量：

```text
dense_transformer_parameters
routed_expert_parameters
embedding_parameters
```

返回结构：

```python
@dataclass
class _ParameterCounts:
    dense_transformer: float
    routed_experts: float
    embeddings: float
```

其中：

```text
total_parameters = dense_transformer + routed_experts + embeddings
```

后续 `estimate_training_memory()` 会把这些参数按 TP/PP/EP/ETP 分片，再乘以
`bytes_per_parameter` 计算显存。

## 2. 输入配置

`_count_parameters()` 使用的主要字段：

- `hidden_size`
- `num_layers`
- `num_attention_heads`
- `num_query_groups`
- `kv_channels`
- `ffn_hidden_size`
- `vocab_size`
- `share_embeddings_and_output_weights`
- `gated_linear_unit`
- `activation_func`
- MoE 相关字段：
  - `num_moe_experts`
  - `moe_layer_freq`
  - `moe_ffn_hidden_size`
  - `moe_shared_expert_intermediate_size`
  - `moe_latent_size`
- MTP 相关字段：
  - `mtp_num_layers`

## 3. 总流程

```text
_count_parameters(model_config)
-> 读取 hidden_size
-> _get_layer_counts(model_config)
-> _ffn_projection_factor(model_config)
-> 计算 attention_parameters_per_layer
-> 计算 layernorm_parameters
-> 计算 dense_mlp_parameters
-> 计算 shared_expert_parameters
-> 如果 MoE：
   -> 计算 routed_expert_parameters
   -> 可选计算 latent_projection_parameters
-> 汇总 dense_transformer_parameters
-> 计算 embedding_parameters
-> 返回 _ParameterCounts
```

## 4. Layer 数量

`_get_layer_counts()` 计算 dense 和 MoE layer 数。

### 非 MoE 模型

```text
dense = num_layers
moe = 0
```

然后通过 `_with_mtp_layers()` 追加 MTP 层。

### MoE 模型

`moe_layer_freq` 可能是：

- `int`：

```text
layer_idx % moe_layer_freq == 0 -> MoE layer
```

- `list/tuple`：

```text
moe_layer_freq[i] == 1 -> 第 i 层是 MoE layer
```

最终返回：

```text
dense = num_layers - num_moe_layers
moe = num_moe_layers
```

### MTP

`_with_mtp_layers()` 会根据最后一个 decoder layer 的类型，把 MTP 层近似成：

- dense MTP layer
- 或 MoE MTP layer

因此 `_count_parameters()` 并不是精确读取 MTP 模块结构，而是按额外 decoder
layer 估算。

## 5. Attention 参数量

代码：

```python
query_projection_size = model_config.kv_channels * model_config.num_attention_heads
query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

num_query_groups = (
    model_config.num_query_groups
    if model_config.num_query_groups
    else model_config.num_attention_heads
)

attention_parameters_per_layer = (
    2
    * hidden_size
    * hidden_size
    * (
        (1 + num_query_groups / model_config.num_attention_heads)
        * query_projection_to_hidden_size_ratio
    )
)
```

### 5.1 变量含义

设：

```text
h = hidden_size
H = num_attention_heads
d = kv_channels
G = num_query_groups
```

则：

```text
query_projection_size = H * d
```

```text
query_projection_to_hidden_size_ratio = (H * d) / h
```

### 5.2 四个投影

Q、K、V、output 四个权重：

```text
Q: h -> H * d
K: h -> G * d
V: h -> G * d
O: H * d -> h
```

参数分别为：

```text
Q = h * H * d
K = h * G * d
V = h * G * d
O = H * d * h = h * H * d
```

合并：

```text
Total = Q + K + V + O
      = 2*h*H*d + 2*h*G*d
      = 2*h*d*(H + G)
```

用 `ratio = (H * d) / h` 代换：

```text
H * d = h * ratio
```

得到：

```text
Total = 2 * h * h * ratio * (1 + G/H)
```

也就是代码中的：

```text
2 * hidden_size^2 * ratio * (1 + num_query_groups / num_attention_heads)
```

### 5.3 每一项含义

- `hidden_size * hidden_size`：基础投影规模。
- `2`：Q/O 和 K/V 合并后的公共因子。
- `1 + G/H`：MHA 时为 `2`；GQA 时小于 `2`，体现 KV 共享带来的参数下降。
- `ratio`：把 `H * d` 归一化成 `h` 的倍数。

### 5.4 示例

Qwen3.5-VL 0.8B smoke 配置示例：

```text
h = 1024
H = 8
G = 2
d = 128
ratio = 1
```

则：

```text
attention_parameters_per_layer
= 2 * 1024 * 1024 * ((1 + 0.25) * 1)
= 2,621,440
```

拆开：

```text
Q = 1,048,576
K = 262,144
V = 262,144
O = 1,048,576
Total = 2,621,440
```

## 6. LayerNorm 参数量

```python
layernorm_parameters = (4 * hidden_size * layer_counts.total) + (2 * hidden_size)
```

逻辑：

- 每个 transformer layer 假设有 attention 前 LN 和 MLP 前 LN。
- 每个 LN 假设包含 weight + bias。
- 因此每层为 `2 * 2 * h = 4h`。
- 额外 `2h` 对应 final LayerNorm。

这里不是“每层 4 次 LayerNorm 操作”，而是：

```text
每层 2 个 LayerNorm 模块
每个 LayerNorm 有 weight + bias
每层参数 = 2 * 2h = 4h
```

`+ 2h` 也不是 word embedding 或 lm_head 的 norm，而是 decoder 之后的 final
LayerNorm。word embedding 和 lm_head 的参数是在第 9 节单独计算的。

对 Qwen 这类模型还要注意：

- RMSNorm 通常只有 `weight`，没有 `bias`，实际每个 norm 更接近 `1h`。
- Qwen attention 可能额外有 Q/K LayerNorm，这里没有显式区分。

所以这是通用 LayerNorm 估算，不是对 Qwen 结构的精确计数。

## 7. Dense MLP 参数量

```python
dense_mlp_parameters = (
    ffn_projection_factor
    * hidden_size
    * model_config.ffn_hidden_size
    * layer_counts.dense
)
```

`ffn_projection_factor`：

- SwiGLU：`gated_linear_unit=True` 且 `activation_func=SiLU`，为 `3`。
  - `gate_proj + up_proj + down_proj`
- 普通 MLP：为 `2`。
  - `fc1 + fc2`

因此每个 dense layer 的 MLP 参数量为：

```text
factor * h * ffn_hidden_size
```

### 7.1 `ffn_projection_factor` 来源

该值来自 `_ffn_projection_factor()`：

```python
if (
    getattr(model_config, "gated_linear_unit", False)
    and getattr(model_config, "activation_func", None) == F.silu
):
    return 3.0
return 2.0
```

普通 MLP 是两个线性层：

```text
fc1: h -> ffn_hidden_size
fc2: ffn_hidden_size -> h
```

因此：

```text
参数 = 2 * h * ffn_hidden_size
factor = 2
```

SwiGLU 是三个线性层：

```text
gate_proj: h -> ffn_hidden_size
up_proj:   h -> ffn_hidden_size
down_proj: ffn_hidden_size -> h
```

因此：

```text
参数 = 3 * h * ffn_hidden_size
factor = 3
```

只有 `gated_linear_unit=True` 且 `activation_func=SiLU` 时才按 SwiGLU 处理。
GeGLU 等其他 gated MLP 如果激活函数不是 SiLU，当前实现仍会返回 `2.0`，可能低估。

## 8. MoE 参数量

### 8.1 Shared expert

```python
shared_expert_parameters = (
    ffn_projection_factor
    * hidden_size
    * moe_shared_expert_intermediate_size
    * layer_counts.moe
)
```

只对 MoE layer 计算，不乘 expert 数量。

### 8.2 Routed expert

如果模型没有 `moe_latent_size`：

```text
routed_expert_parameters =
    factor
    * h
    * moe_ffn_hidden_size
    * num_moe_experts
    * moe_layers
```

如果模型有 `moe_latent_size`：

```text
routed_expert_parameters =
    factor
    * moe_latent_size
    * moe_ffn_hidden_size
    * num_moe_experts
    * moe_layers
```

同时增加 latent projection：

```text
latent_projection_parameters =
    2 * h * moe_latent_size * moe_layers
```

Routed expert 参数单独返回，便于后续按 EP/ETP 分片。

### 8.3 Latent projection

当配置了 `moe_latent_size` 时，latent MoE 的结构是：

```text
hidden_states [h]
  -> fc1_latent_proj: h -> moe_latent_size
  -> token dispatcher
  -> routed experts 输入维度为 moe_latent_size
  -> token combiner
  -> fc2_latent_proj: moe_latent_size -> h
  -> 输出回 hidden
```

Megatron 中的实际模块是：

```text
fc1_latent_proj
fc2_latent_proj
```

参考：

`3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:273`
`3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:285`

两个投影的参数量分别是：

```text
fc1_latent_proj: h * moe_latent_size
fc2_latent_proj: moe_latent_size * h
```

因此每个 MoE layer 的 latent projection 参数为：

```text
2 * h * moe_latent_size
```

再乘 MoE layer 数：

```text
latent_projection_parameters =
    2 * hidden_size * moe_latent_size * moe_layers
```

注意：

- 这两个投影是每层共享的，不是每个 expert 一份。
- 所以它们不乘 `num_moe_experts`。
- 它们被加进 `dense_transformer_parameters`，而不是 `routed_expert_parameters`。
- 在 latent MoE 下，routed expert 的输入维度从 `hidden_size` 变成
  `moe_latent_size`，因此 expert 参数也相应改变。

## 9. Embedding 参数量

```python
embedding_size = hidden_size * _get_vocab_size(model_config)
embedding_parameters = embedding_size

if not model_config.share_embeddings_and_output_weights:
    embedding_parameters += embedding_size
```

- `_get_vocab_size()` 会返回 padding 后的 vocab size。
- 如果 embedding 和 output head 共享权重，只算一次。
- 如果不共享，则再增加一份 output projection 参数。

## 10. 汇总公式

```text
dense_transformer_parameters =
    attention_parameters_per_layer * total_layers
    + layernorm_parameters
    + dense_mlp_parameters
    + shared_expert_parameters
    + latent_projection_parameters
```

注意：

- attention 和 LayerNorm 按 `total_layers` 计算。
- dense MLP 只按 dense layer 计算。
- shared expert 和 latent projection 按 MoE layer 计算。
- routed expert 单独返回。

## 11. 已知限制

- 不读取实际权重，只依赖 `model_config`。
- 不遍历 vision tower、visual projection 等非 text-decoder 模块。
- MTP 被近似成额外 decoder layer。
- RMSNorm 与 LayerNorm 的差异没有精确区分。
- 线性层 bias 未计入 attention/MLP 参数估算。
- 对 Qwen3.5-VL 0.8B，该函数估算约 `0.66B`，而实际 DDP bucket 为
  `0.87B`；做显存规划时应以后者为准。

## 12. 关联文档

- 显存估算：[[qwen35-vl-800m-sft-memory]]
- 最小训练链路：[[qwen35-vl-800m-sft-smoke]]
- 通用观测 roadmap：[[megatron-lm 链路可观测性 roadmap]]
