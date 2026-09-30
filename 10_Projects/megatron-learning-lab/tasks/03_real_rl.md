---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# P3：接口兼容的 RL 环境中的完整更新

以 NeMo RL v0.7.0 为源码参考，优先复用已有 RL 环境；分别验收 rl_grpo 或 rl_ppo 所需接口与数据/计算语义，实际版本不必一致。记录所用 Bridge/Core/rollout 后端及适配器；只处理必要冲突，不要求重装正式镜像。不能用参考静态源码图冒充另一 runtime 的执行图。

第一实验：同步GRPO，小Qwen checkpoint，单轮、短输出、固定generation/update版本、temperature1、无top-k/p截断。复用正式algorithm、policy worker、loss、generation、refit。用真正rollout，不能把人工写好的responses当生成。

第二实验：以官方 `examples/configs/recipes/llm/ppo-qwen2.5-1.5b-gsm8k-1n8g-megatron-valuetp2sp-dynbatch.yaml` 为对照，检查继承后的完整配置再缩小规模。它使用vLLM与独立value；新增SGLang profile时单独验证，不只改标签。

必须记录 prompt/group/trajectory id；generation/prev/current/ref logprob；sample/token mask；实际advantage estimator；loss reduction；采样配置；training/generation权重版本；refit完成证据。先做2个真实迭代，只称闭环smoke，不称能力提升。

测试：next-token对齐；同组基线；全组同分；正负advantage的clip路径；beta=0不要求ref；beta>0对应固定ref；更新前后权重版本；refit后固定token logprob检查；长短response归约。force_on_policy_ratio必须保留梯度，禁止替换成常数。

MoE扩展再加routed_experts与router replay，不把其缺失当默认通过。async/partial/OPD后置。
