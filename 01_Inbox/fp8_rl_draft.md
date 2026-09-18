# FP8 RL 方案速览与阅读材料

整理日期：2026-09-18。关注训练计算、参数存储、权重同步和rollout之间的精度契约。以下链接以官方文档、作者论文及实现团队技术文章为主；main/latest文档会变化，落地时需固定版本。这里不是本项目的兼容性或性能验收报告。

## 先区分四件事

| 维度 | 需要确认的问题 |
|---|---|
| 参数存储与更新 | 常驻参数是BF16还是FP8？master weights、优化器状态和更新计算是什么精度？ |
| 训练计算 | 哪些GEMM使用FP8？量化权重、activation和反向计算的recipe是什么？ |
| 权重同步 | 传BF16后由推理端量化，还是传训练侧生成的FP8权重及scale？ |
| Rollout计算 | 权重、activation、KV cache分别是什么精度？是否与训练前向匹配？ |

“FP8训练”不等于以上全部使用FP8。NeMo RL的文档提供`fp8`开启而`fp8_param=false`的配置；SkyRL也介绍了FP32 master更新后重新量化到FP8参数存储的路线。[NeMo RL](https://docs.nvidia.com/nemo/rl/latest/fp8.html)、[SkyRL](https://www.anyscale.com/blog/fp8-reinfinforcement-learning-in-skyrl)

## 主要路线

| 路线 | 核心做法 | 值得关注的收益与代价 | 代表材料 |
|---|---|---|---|
| BF16训练 + FP8 rollout | 保留训练计算精度，生成阶段在线量化 | 优先加速生成；需处理训练策略与采样策略差异 | verl、FP8-RL |
| FP8训练计算 + FP8 rollout | 两侧对齐低精度计算，参数存储另行配置 | 同时优化训练与生成；相同FP8 dtype仍不保证数值一致 | NeMo RL |
| 统一前向量化流程 | 对齐量化位置、精度和粒度 | 从计算图设计上减少训推差异，保留较高精度更新状态 | Jet-RL |
| FP8参数存储及all-gather | 高精度master更新，低精度参数副本参与通信和计算 | 进一步减少显存/通信；受硬件、TE和优化器集成限制 | SkyRL |

这些路线可以组合，不是互斥产品分类。收益不能直接跨模型、硬件和rollout长度比较。

## 值得看的材料

### 1. verl：FP8 RL 官方文档（建议先读）

[FP8 RL in verl](https://github.com/verl-project/verl/blob/main/docs/low_precision/fp8.md)

- 同时说明rollout-only与端到端FP8两种模式，适合建立整体概念。
- 重点看训练/推理backend支持矩阵、SGLang ignored-layer规则，以及token-level TIS对照实验。
- 对本项目的启发：模型构建与在线权重同步必须使用相同的量化排除规则。

### 2. NeMo RL：FP8 配置与硬件边界

[FP8 Quantization in NeMo RL](https://docs.nvidia.com/nemo/rl/latest/fp8.html)

- 重点看`fp8`、`fp8_recipe`与`fp8_param`的独立配置；文档示例中`fp8_param=false`。
- 阅读Hopper/Blackwell的recipe支持差异，不把一种设备上的配置直接套到另一种设备。
- 适合回答“FP8训练是否必须把常驻参数也变成FP8”。

### 3. NVIDIA：端到端FP8 RL的工程与数值分析

[Run High-Throughput Reinforcement Learning Training with End-to-End FP8 Precision](https://developer.nvidia.com/blog/run-high-throughput-reinforcement-learning-training-with-end-to-end-fp8-precision/)（2026-04-20）

- 比较BF16、仅rollout FP8、训推均FP8；讨论不同引擎kernel引入的数值差异。
- 重点看权重128×128、activation 1×128的blockwise recipe，以及概率误差指标和importance sampling。
- KV cache/attention量化是额外能力，不能因为linear使用FP8就认为它们也已开启。

### 4. Jet-RL：为什么要统一训练与rollout精度流程

[Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow](https://arxiv.org/abs/2601.14243) · [已阅读的v1正文](https://arxiv.org/html/2601.14243v1)

- 建议重点读第4节：让推理计算图与训练前向的量化位置、精度及粒度对齐。
- 保留较高精度权重副本和BF16算子间梯度传递，同时在GEMM上使用FP8。
- 其长序列实验中的不稳定现象是特定实验结果，不能推广为所有BF16训练+FP8 rollout都必然失败。

### 5. SkyRL：FP8参数存储、通信与master更新

[FP8 Reinforcement Learning in SkyRL: Preserving Policy Consistency Across Training and Rollout](https://www.anyscale.com/blog/fp8-reinfinforcement-learning-in-skyrl)

- 重点看FP32 master更新后重新量化、FP8参数all-gather、权重与block scales直接同步。
- 讨论小幅RL更新是否被量化吞掉，以及FP32 scale与power-of-two scale的对照。
- 对本项目的启发：`fp8_param_gather=false`可以是验证基线，但不是普遍最优配置；开启后仍可保留高精度master更新。

### 6. FP8-RL：实用rollout栈与分布修正

[FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning](https://arxiv.org/abs/2601.18150)（当前页面为2026-04-10修订的v2）

- 从每步权重变化、在线量化与同步成本出发，讨论W8A8 rollout、KV cache和QKV scale重校准。
- 关注token-level TIS/MIS等rollout correction；是否提升吞吐和保持学习行为需在同一实验配置中比较。
- 适合补充阅读；此处根据官方摘要整理，未逐项复现论文实验。

## 对当前VL项目的启发

以下是结合本项目的建议，不是上述材料已经验证过的VL结论：

1. 保留现有BF16基础参数、FP8 expert计算和`fp8_param_gather=false`作为已跑通基线；FP8参数存储/gather另做受控对照。
2. 先固定同一权重版本、量化模块范围、block shape、scale算法和权重同步契约，再比较BF16/FP8。修复HF量化配置传播优先于调整通信精度开关。
3. 记录logp绝对差、分位数/尾部、KL、重要性权重及真实reward，不只看有符号diff_mean。概率比是exp(logp差)，少量大偏差也值得关注。
4. 保留Vision/router/attention等模块的明确精度策略；不要从纯语言实验直接推断多模态效果。
5. R07只验证了4步功能链路，使用dummy reward；它不能证明长期学习稳定性或completion质量。参见[本地实验记录](deepvision_fp8_local_run.md)。

推荐阅读顺序：verl概览 → NeMo配置 → NVIDIA工程分析 → Jet-RL精度流程 → SkyRL存储/同步细节；FP8-RL论文作为补充。
