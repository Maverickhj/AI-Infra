---
type: source
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Megatron-Bridge 稳定版源码调研

## 0. 调研边界与结论

此前调研核验于 2026-09-30，当时记录的 Bridge 正式 release 为 **v0.6.2**，发布于 2026-09-18 23:54 UTC，即新加坡时间 9 月 19 日。release 不是 draft 或 prerelease。以官方 GitHub API、固定 tag/gitlink 与实际源码为准，不以可能滞后的文档首页版本横幅为准。[RELEASE-BR][TAG-BR]

本次深入读取模型 bridge、recipe 实体及别名、数据预处理、训练入口、loss、MCore GPT/Attention/MLA/MoE 片段，以及 NeMo RL 的 worker、训练后处理与 clipped policy loss。具体读取范围见 `source-evidence.json`。**这是静态代码调研，未实际运行 Bridge、NeMo RL 或 GPU。未看到用户已有 Codex 实现及失败日志，不能将其效果差归因于某个未经检查的具体 bug。**

引用 ID 对应 [REFERENCES.md](REFERENCES.md)。

> Review 修订：本报告保留固定版本的源码证据，不能被理解为要求 learning-lab 运行环境逐项照搬这些依赖。目标读者为有一定基础的初学者，基础铺垫缩减，核心实现精讲。实际环境规则见 [最小兼容策略](../profiles/minimum-compatibility.md)。本轮只修订方案，不重新宣称完成了 GPU 或跨版本兼容验证。

## 1. 固定参考源码，按所需接口验证实际环境

| 对象 | 本次核验结果 | 工程含义 |
|---|---|---|
| Bridge | v0.6.2 / `c0e164ed2aedac4ad1c877780e2564a19d5d54ec` | 模型与 SFT 主线 |
| 配套 Core | `a0f793dfa4e776d99a8aa63bed74e0fafc91b0db` | 参考读取来自 Bridge gitlink；实际环境可采用经合约验证的版本 |
| Python | `>=3.12,<3.13` | 上游该版本包声明，非所有 lab profile 的统一 Python 要求 |
| Transformers | `>=5.8,<=5.12.1` | 这是依赖约束，不是已安装版本 |
| Transformer Engine | 源码 pin `15288857644368b8b7bdddd9bead7d6d3d04de80` | 参考依赖记录；所选运行路径使用 TE 时核验实际 ABI/接口/行为 |
| NeMo RL | 正式版 v0.7.0 / `81aa43dda4765b0429cf31dab44441e4e4383911` | 独立的 RL 实践与源码层 |

Bridge 的 uv 配置复用系统 Torch 等组件，Core 指向本地 submodule。稳定 release 也可能包含实验 API、开发依赖或已知限制。“stable”不是所有功能/硬件组合已被验证的保证。[BR-GITLINK][BR-PYPROJECT]

### RL 依赖存在两种来源，不可混成一套

NeMo RL v0.7.0 发布说明的镜像组件表写的是 Bridge `0.5.0+554c7b9`；而该 tag 的 `.gitmodules` 和 gitlink 指向 `yuki-97/Megatron-Bridge@151168f36372132c8fb8f926501f573019feb967`，后者再指向 `yuki-97/Megatron-LM@7b8495881dee1689db9d5ed636d8b6b48eacf260`。[RL-RELEASE][RL-GITLINK]

这不证明发布镜像不能用，但说明**容器发布说明、源代码 checkout 和实际 import 环境不能混为一谈**。这些是参考来源，实际环境可以不同，也可在验证通过后供 SFT/RL 共用。每个运行 profile 单独记录实际版本、模块 `__file__`、源码标识和容器 digest（仅使用容器时需要）；以该场景的接口/语义测试作为准入，不以版本名称判定兼容。

稳定版要求用于建立可追溯的调研参考。运行时只满足 learning-lab 当前功能所需的最小接口与语义，既不要求所有依赖独立最新，也不要求逐项匹配参考 release 的 pin。

## 2. Qwen2.5 的入口实际是 Qwen2Bridge

Qwen2.5-0.5B 官方配置的 `architectures` 为 `Qwen2ForCausalLM`，`model_type` 为 `qwen2`。Bridge 注册表因此走 `Qwen2Bridge`，不是按字符串“qwen25”寻找一个新模型类。[HF-qwen25-05b][B-Q2]

代码明确设置 RMSNorm、gated MLP、普通 Linear 无 bias，但 **QKV 有 bias**。映射不仅重命名：Q/K/V 合并用 `QKVMapping`，gate/up 合并用 `GatedMLPMapping`。embedding、最终 norm、LM head 和 attention output projection 都有对应映射。[B-Q2]

**课程要求：** 首先用完整 decoder 展示公共骨架，再点开一层解释其权重布局；不能把“模型支持”简化成换个标题或 hidden size。

## 3. Qwen3 不是 Qwen2.5 的缩放版

Qwen3Bridge 明确启用 QK norm，关闭 QKV bias。norm 的参数映射可追到 HF 的 `q_norm` 与 `k_norm`。[B-Q3]

Qwen3-0.6B 配置：28 层、hidden size 1024、16 个 Q heads、8 个 KV heads、显式 head_dim=128、FFN 中间宽度 3072。于是：

- Q 总投影宽度为 16×128=2048，而非 1024。
- K/V 各为 8×128=1024。
- 逻辑 QKV 合计宽度为 4096。
- attention 合并 heads 后为 2048，再由 output projection 投回 1024。
- SwiGLU 的 gate/up 各 3072，融合 fc1 的输出宽度是 6144。

这来自官方 config 与投影结构的推导，**不是执行测量结果**。MCore 的 SelfAttention 依据 query_projection_size 与 kv_projection_size 构建 QKV，随后按 query groups 拆分，并分别应用 Q/K norm。[HF-qwen3-06b][C-ATTN]

不要把所有模型统一成 `head_dim=hidden_size/heads`。也不要直接把 Qwen3-0.6B 当作 Base checkpoint；本课程使用的 HF ID 是已后训练版本，Base 是不同的模型 ID。

## 4. Bridge 的模型构建是可追踪的，不是一个黑盒“转换”

阅读路线为：HF config / architecture → AutoBridge 的注册分发 → family bridge 的 provider 配置 → GPTModelProvider 的 spec、词表 padding 和 stage 设置 → MCore GPTModel。[B-Q2][B-Q3][B-DS3][B-PROVIDER]

`GPTModelProvider` 会处理 callable/ModuleSpec、PP/VPP 的 pre/post-process、词表 padding，再实例化 MCore GPTModel。前端必须同时保留 `HF vocab_size` 与 `padded_vocab_size`，不能永远使用原始词表长度计算本地输出分片。[B-PROVIDER]

真实 `GPTModel.forward` 调用 `_preprocess`、decoder 和 `_postprocess`。默认后处理路径在有 labels 时返回每 token loss，无 labels 时返回 logits；还存在 output_processor/MTP 等分支。不要在所有场景把 forward 输出标成 logits，或重复做一次 cross entropy。[C-GPT]

## 5. 语义模块与真实 nn.Module 不是一一对应

Dense Qwen 的 input/post-attention norm 可映射到 TE fused Linear 的 `layer_norm_weight`，因此“hook 没看见一个独立 RMSNorm”不能说明 norm 没运行。MoE 分支的 pre-MLP norm 又可以是独立模块。[B-Q2][B-Q3][B-Q3M][B-DSMAP]

QKVMapping 是按 GQA group 交错排布，不是对 Q、K、V 三块直接做一次普通 cat 后就结束。它的文档与结构明确描述 group 内若干 Q heads 配一组 K/V 的布局。[B-QKV]

**课程要求：** 图中每一步有 semantic operator 与 backend implementation 两个字段。比较 HF/Bridge 的权重和 activations 之前先进行布局还原。需要中间值但 fused kernel 不暴露时，标“不可直接观测”，不要伪造 hook 输出。

## 6. Qwen3 MoE 是全模型中的 FFN 替换，不是另一种 attention

Qwen3-30B-A3B 使用 Qwen3 GQA/QK norm，官方配置 48 层、128 routed experts、每 token top-8；该配置没有 shared expert。bridge 中可见 router、TEGroupedMLP/SequentialMLP 权重映射、alltoall dispatcher 和辅助 loss 配置。[HF-qwen3-30ba3b][B-Q3M]

MCore MoELayer 将计算分为 route、preprocess、dispatch、routed_experts_compute、combine、postprocess；shared expert 有单独计算与 overlap 路径。[C-MOE]

“激活参数约 A3B”不表示只需常驻这些权重。lab 必须分开显示总参数、活跃专家、每 rank 常驻权重、token 分配和通信量。

## 7. DeepSeek 课程要区分三种东西

### DeepSeek-V2-Lite

官方配置有 27 层、首层 dense、64 routed experts、top-6、2 shared experts，以及 MLA。它的 `q_lora_rank=null`，因此不能把 V3 的 Q 压缩路径原样画过去；KV 仍有低秩路径。[HF-deepseek-v2-lite]

本次已核对官方 config、公共 mapping 和 MCore MLA 片段；独立 V2Bridge 的完整行为仍列入 Codex 实施审计，不把它写成已完成 GPU 支持验证。

### DeepSeek-V3

使用 MLAModelProvider。官方配置中 61 层、前 3 层 dense、256 routed experts、top-8、1 shared expert；Q 压缩 rank1536、KV rank512、no-PE维128、RoPE维64、V维128。Bridge 据 `first_k_dense_replace` 生成层类型列表。[HF-deepseek-v3][B-DS3]

公共映射把 HF 的 q_a/q_b、kv_a/kv_b 分别映射到 down/up projections。MCore MLA 源码中，训练路径展开 Q/K/V；cached-latent inference 路径另有压缩缓存与 absorption 条件。**不能把推理 KV-cache 图当成训练激活图。** [B-DSMAP][C-MLA]

Bridge 的 V3 路由配置包含 sigmoid、expert bias、fp32 router，同时保留 `seq_aux_loss`，缺省系数 0.0001。不要把“auxiliary-loss-free balancing”口号改写成“这里绝无辅助 loss”。V3 FP8 权重导入还会处理块级 scale_inv，直接 cast 成 BF16 不是等价反量化。[B-DS3]

全量 V3 不是入门机器上的默认运行任务。第一轮交付必须有真实全结构课程；执行实验可另设 architecture-scaled 随机配置，但它不是原始 V3 checkpoint，更不能报告 V3 能力结果。

### DeepSeek-R1-Distill-Qwen

官方 config 明确是 Qwen2ForCausalLM/qwen2，可作为 RL 行为与 checkpoint 案例；**不能拿它代替 DeepSeek 的 MLA/MoE 结构课。** [HF-dsr1-distill-qwen15b]

## 8. SFT recipe 不等于预训练权重已经加载

`qwen3_600m_sft_config` 是旧名称别名，实际落在 h100 目录的 `qwen3_600m_sft_1gpu_h100_bf16_config`。该 recipe 的 provider 使用 `load_weights=False`，而 `checkpoint.pretrained_checkpoint` 在相关代码中只是待配置示例。[B-ALIAS][B-RECIPE]

`finetune` 明确要求 pretrained_checkpoint 或 resume checkpoint，并委托到 pretrain 的 setup/train 路径。因此 lab 的“开始 SFT”之前必须核验权重来源和加载结果。模型初始化成功、loss 能计算，不足以证明进行了预训练模型 SFT。[B-SFTENTRY][B-PRETRAIN]

H100 recipe 名称不等于已验证适配其他 GPU。真实运行需针对当前 case 检查机器、驱动、所需 CUDA/TE kernel、dtype 与实际使用的通信能力，不把全部上游硬件能力作为门槛。

## 9. SFT 的核心工程问题是 token 语义与归一化

当前 `ChatSFTPreprocessingConfig` 有 `assistant`、`last_turn`、`full` 三种 loss_mode；prompt-completion 预处理是不同的数据合约，默认不调用 chat template。源码还拒绝将结构化 conversation 悄悄摊平成无模板文本。[B-SFTDATA]

`gpt_step.forward_step` 返回模型输出及绑定 loss_mask 的 loss callback；`masked_next_token_loss` 做加权求和并返回 num_tokens，而不是在该函数中对每个 rank 各自取均值。[B-STEP][B-LOSS]

因此必须一路解释并验证 scheduler、DP/CP、gradient accumulation 最终在哪里归一化。不同 rank 有不同数量有效 tokens 时，“局部均值再平均”与全局 token 均值通常不同。

Packed metadata 在该版本既支持新的 q/kv 专用字段，也兼容旧字段。THD+CP 不只是把 sequence 均分，代码根据 cu_seqlens 获取切分索引；中间 PP stage 也可能需要 metadata。[B-STEP]

Qwen3 长上下文 SFT recipe 有 `calculate_per_token_loss=True` 与 `average_in_collective=False` 等特定设置，注释解释了全 mask 的 CP rank 导致 NaN 的风险。课程先短序列、CP1，再将 packed/CP 作为真实对照，不从128K起跑。[B-RECIPE]

## 10. Bridge 的 RL 示例不能直接当生产训练模板

`examples/rl/rlhf_with_bridge.py` 明确说明是 AI 生成的教育演示：HF 生成、sentiment reward、简单 REINFORCE 风格 loss，非完整 PPO。它适合说明 Bridge 的模型/权重接口，不足以作为完整 RL 课程的可信实现终点。[B-RLDEMO]

真实参考采用 NeMo RL 正式版本。已核对 policy worker、`models/megatron/train.py`、clipped policy loss 和 Qwen2.5 PPO 配置。SGLang 的 generation 初始化存在，但这不代表所有 vLLM recipe 换一个 backend 字段就等价可用。[R-WORKER][R-TRAIN][R-GRPO][R-PPOCFG]

## 11. RL 至少分清四份 logprob

实际 `ClippedPGLossDataDict` 包括 advantages、prev_logprobs、generation_logprobs、reference_policy_logprobs、token_mask、sample_mask；当前 next-token logprobs 作为 loss 函数输入。源码对 mask/advantages/旧 logprob 做位置偏移，并用 token×sample mask 归约。[R-LOSS]

学习页面因此必须分开：

| 数据 | 含义 |
|---|---|
| generation_logprobs | rollout 后端实际采样时记录 |
| prev_logprobs | 更新前训练后端对固定 trajectory 的重算 |
| current_logprobs | 当前待求导 policy 在训练 forward 中产生 |
| reference_logprobs | 固定参考模型，用于选定的 KL 正则 |

即使权重版本相同，训推后端、采样分布处理和 MoE 路由也可能产生差别。PPO ratio 与 generation↔training 的校正不是同一个概念。初始实验使用 temperature=1、无 top-k/p 截断，先减少分布定义歧义；以后再比较 sampling-filtered 与原始 model logprob。

`force_on_policy_ratio` 分支以 detach 保留梯度路径，而不是把整个 ratio 替换成 Python 常数1。不能凭名称重写算法。要对选定 loss 分支做逐项数值和梯度对照。[R-LOSS]

## 12. 仍需在实现时验证，而非本次已经证明

完整 SFT 数据 collator 的标签移位与截断边界；全部 train_step/梯度 finalize 的缩放；NeMo RL 最终合并配置和完整迭代调用顺序；不同 backend 的真实导入兼容；除 Qwen3-0.6B 外模型的 HF commit；用户当前 GPU/驱动；真实数值误差与性能；所有 GPU 实验。

这些项目是明确的后续验收点，不应阻止先交付准确的整模课程和源码阅读能力，也不能被“mock pass”替代。

## 13. 正式 runner 与默认数据语义也需要读代码

本版本存在 `scripts/training/run_recipe.py`，支持 model/完整 recipe 选择、sft/lora/dora 模式、checkpoint参数和尾随ConfigContainer覆盖。sequence length由dataset拥有，runner同步到model，不是单独改一个model字段。[B-RUNNER]

`default_squad_config`采用prompt-completion预处理，`default_tulu3_config`采用ChatSFTPreprocessingConfig。两者都能做SFT，但不能用SQuAD路径的成功代替多轮chat mask验证。[B-DATASETUTIL]
