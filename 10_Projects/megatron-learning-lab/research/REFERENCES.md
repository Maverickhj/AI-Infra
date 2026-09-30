---
type: source
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# 一手资料与审阅入口

核验日期：2026-09-30。`latest` API 只用于发布发现，课程链接必须落到锁定 SHA。

- [RELEASE-BR] [Bridge 最新正式 release](https://api.github.com/repos/NVIDIA-NeMo/Megatron-Bridge/releases/latest)
- [TAG-BR] [固定 release tag](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.6.2)
- [BR-GITLINK] [Bridge 的 Core gitlink](https://api.github.com/repos/NVIDIA-NeMo/Megatron-Bridge/contents/3rdparty/Megatron-LM?ref=v0.6.2)
- [BR-PYPROJECT] [Bridge 环境声明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/pyproject.toml)
- [RL-RELEASE] [NeMo RL 正式 release](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.7.0)
- [RL-GITLINK] [NeMo RL 的 Bridge gitlink](https://api.github.com/repos/NVIDIA-NeMo/RL/contents/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge?ref=v0.7.0)

## 固定版本代码

- [B-Q2] [src/megatron/bridge/models/qwen/qwen2_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/qwen/qwen2_bridge.py)：完整文件：Qwen2/Qwen2.5 配置与权重映射。
- [B-Q3] [src/megatron/bridge/models/qwen/qwen3_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/qwen/qwen3_bridge.py)：完整文件：QK norm、无 QKV bias、条件 MTP 映射。
- [B-Q3M] [src/megatron/bridge/models/qwen/qwen3_moe_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/qwen/qwen3_moe_bridge.py)：完整文件：MoE router、专家权重、TE/Sequential 分支。
- [B-DS3] [src/megatron/bridge/models/deepseek/deepseek_v3_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/deepseek/deepseek_v3_bridge.py)：完整文件：MLA provider、dense/MoE 排布、shared expert、FP8 导入。
- [B-DSMAP] [src/megatron/bridge/models/deepseek/common.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/deepseek/common.py)：MLA q/kv down/up、norm、shared expert、expert bias 的映射。
- [B-ALIAS] [src/megatron/bridge/recipes/qwen/qwen3.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/recipes/qwen/qwen3.py)：旧 recipe 名称到 h100 显式 recipe 的兼容别名。
- [B-RECIPE] [src/megatron/bridge/recipes/qwen/h100/qwen3.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/recipes/qwen/h100/qwen3.py)：0.6B pretrain/SFT，以及 CP8 128K 配置片段。
- [B-SFTDATA] [src/megatron/bridge/data/sft_processing.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/data/sft_processing.py)：chat 与 prompt-completion 类型、loss_mode、规范化。
- [B-SFTENTRY] [src/megatron/bridge/training/finetune.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/training/finetune.py)：完整文件：checkpoint 前置条件，委托 pretrain。
- [B-PRETRAIN] [src/megatron/bridge/training/pretrain.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/training/pretrain.py)：setup/训练/验证生命周期片段。
- [B-TRAIN] [src/megatron/bridge/training/train.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/training/train.py)：train 准备 forward callback、训练模式与调度选择；未逐行审完整 train_step。
- [B-STEP] [src/megatron/bridge/training/gpt_step.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/training/gpt_step.py)：packed metadata/CP 切分，forward 返回 output+loss callback。
- [B-LOSS] [src/megatron/bridge/training/losses.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/training/losses.py)：完整文件：loss sum、num_tokens，非函数内局部均值。
- [B-PROVIDER] [src/megatron/bridge/models/gpt_provider.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/gpt_provider.py)：provide 的 spec、vocab padding、pre/post-process 和 MCoreGPTModel 构建。
- [B-QKV] [src/megatron/bridge/models/conversion/param_mapping.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/models/conversion/param_mapping.py)：AutoMapping 后端识别与 QKVMapping 的 GQA interleave 合约。
- [B-RLDEMO] [examples/rl/rlhf_with_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/examples/rl/rlhf_with_bridge.py)：教学 REINFORCE 演示声明与 export/refit；非完整 PPO。
- [C-GPT] [megatron/core/models/gpt/gpt_model.py](https://github.com/NVIDIA/Megatron-LM/blob/a0f793dfa4e776d99a8aa63bed74e0fafc91b0db/megatron/core/models/gpt/gpt_model.py)：forward 的 preprocess/decoder/postprocess 与 labels/logits 分支。
- [C-ATTN] [megatron/core/transformer/attention.py](https://github.com/NVIDIA/Megatron-LM/blob/a0f793dfa4e776d99a8aa63bed74e0fafc91b0db/megatron/core/transformer/attention.py)：QKV 实例化、分组切分、QK norm、TP>KV heads 分支。
- [C-MLA] [megatron/core/transformer/multi_latent_attention.py](https://github.com/NVIDIA/Megatron-LM/blob/a0f793dfa4e776d99a8aa63bed74e0fafc91b0db/megatron/core/transformer/multi_latent_attention.py)：down projection、latent norm、up projection 与 cached-latent inference 分支。
- [C-MOE] [megatron/core/transformer/moe/moe_layer.py](https://github.com/NVIDIA/Megatron-LM/blob/a0f793dfa4e776d99a8aa63bed74e0fafc91b0db/megatron/core/transformer/moe/moe_layer.py)：route/preprocess/dispatch/experts/combine/postprocess；shared expert overlap。
- [R-WORKER] [nemo_rl/models/policy/workers/megatron_policy_worker.py](https://github.com/NVIDIA-NeMo/RL/blob/81aa43dda4765b0429cf31dab44441e4e4383911/nemo_rl/models/policy/workers/megatron_policy_worker.py)：真实 policy worker 接入 setup/data/train 组件、router replay 数据约束。
- [R-TRAIN] [nemo_rl/models/megatron/train.py](https://github.com/NVIDIA-NeMo/RL/blob/81aa43dda4765b0429cf31dab44441e4e4383911/nemo_rl/models/megatron/train.py)：model_forward、ProcessedMicrobatch、postprocess、replay 生命周期。
- [R-GRPO] [nemo_rl/algorithms/grpo.py](https://github.com/NVIDIA-NeMo/RL/blob/81aa43dda4765b0429cf31dab44441e4e4383911/nemo_rl/algorithms/grpo.py)：导入链、资源划分、policy/ref 初始化与 vLLM/SGLang generation 选择；非完整训练循环逐行审计。
- [R-LOSS] [nemo_rl/algorithms/loss/loss_functions.py](https://github.com/NVIDIA-NeMo/RL/blob/81aa43dda4765b0429cf31dab44441e4e4383911/nemo_rl/algorithms/loss/loss_functions.py)：ClippedPGLoss 配置、输入字段、next-token 对齐与 KL/logp 分支。
- [R-PPOCFG] [examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-1n8g-megatron-valuetp2sp-dynbatch.yaml](https://github.com/NVIDIA-NeMo/RL/blob/81aa43dda4765b0429cf31dab44441e4e4383911/examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-1n8g-megatron-valuetp2sp-dynbatch.yaml)：完整配置：Qwen2.5 policy/value、GSM8K、TP/CP、vLLM、8 GPU。

## 官方模型配置

- [HF-qwen25-05b] [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json)。revision 状态：待获取完整 HF commit；本次已读官方 config 字段。
- [HF-qwen3-06b] [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/raw/c1899de289a04d12100db370d81485cdf75e47ca/config.json)。revision 状态：官方 API 和固定 revision config 已核对。
- [HF-qwen3-30ba3b] [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)。revision 状态：待获取完整 HF commit；本次已读官方 config 字段。
- [HF-deepseek-v2-lite] [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json)。revision 状态：待获取完整 HF commit；本次已读官方 config 字段。
- [HF-deepseek-v3] [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)。revision 状态：待获取完整 HF commit；本次已读官方 config 字段。
- [HF-dsr1-distill-qwen15b] [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json)。revision 状态：待获取完整 HF commit；本次已读官方 config 字段。

## 追加核对的运行入口

- [B-RUNNER] [scripts/training/run_recipe.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/scripts/training/run_recipe.py)：正式 recipe runner 的参数与 dataset→model seq_length 同步说明。
- [B-DATASETUTIL] [src/megatron/bridge/recipes/utils/dataset_utils.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c0e164ed2aedac4ad1c877780e2564a19d5d54ec/src/megatron/bridge/recipes/utils/dataset_utils.py)：GPTSFT数据配置与SQuAD prompt-completion、Tulu chat的区别。
