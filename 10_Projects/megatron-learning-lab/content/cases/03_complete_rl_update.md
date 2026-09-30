---
type: knowledge
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Case 03：同一条 trajectory 如何完成一次真实 RL 更新

## 软件与模型边界

源码参考：NeMo RL v0.7.0，见 `source.lock.json`。实际 RL/SFT 环境均按所需接口与语义验收，可复用兼容环境，不要求依赖逐项匹配参考版。先做同步、同一权重版本的 rollout→update；不默认开启 async/partial rollout/多轮工具。

第一条自建小实验使用 Qwen3-0.6B，明确是 lab 配置，需按实际运行环境验证。第二条对照从已核对的官方 Qwen2.5-1.5B-Instruct/GSM8K PPO recipe 出发；该 recipe 使用 Megatron policy/value 与 vLLM generation，不标成 SGLang 已验证。[R-PPOCFG]

面向有一定基础的初学者，RL 基本用途只概括，重点精讲数据与梯度语义。这里的重点不是再画一次“RL有reward”，而是从 policy worker 实際需要的数据出发。来源：[R-WORKER][R-TRAIN][R-GRPO][R-LOSS]，见 ../../research/REFERENCES.md。

## 数学符号

| 符号 | 含义 |
|---|---|
| $i,t$ | trajectory 与 token 位置 |
| $G$ | 同一 prompt 的采样组大小 |
| $a_{i,t}$ | 当前 action token |
| $s_{i,t}$ | 生成它时的上下文 |
| $R_i$ | 完整 response 的 reward |
| $A_{i,t}$ | 送入 loss 的 advantage |
| $\theta,\theta_{old},\theta_{ref}$ | 当前、更新前、固定参考 policy 的权重 |
| $\ell^{gen},\ell^{prev},\ell^{cur},\ell^{ref}$ | 生成、更新前重算、当前可导、固定参考的 token logprob |
| $r_{i,t}$ | policy update ratio，$\exp(\ell^{cur}-\ell^{prev})$ |
| $m_{i,t}$ | 有效 action token mask |
| $s_i$ | 有效 sample mask |
| $\epsilon$ | clipping 范围，具体上下界依配置 |
| $\beta$ | reference KL 的系数 |
| $\mu_R,\sigma_R$ | 同 prompt 组内 reward 均值、标准差 |
| $\delta$ | advantage 标准化的数值稳定项，仅用于教学公式 |
| $V_\phi$ | PPO value model，参数为 $\phi$ |

## 1. 全周期与持久标识

每条数据带 prompt_id、group_id、trajectory_id、rollout_weight_version 和 tokenizer revision。页面按如下阶段导航：

`固定 policy 版本 → 渲染 prompt → rollout → reward → 重算 logprob → advantage → actor loss/backward/update → export/refit → 下一次 rollout`。

不要将不同 prompt 的 responses 混成一个 GRPO group，也不要在组内丢弃样本后仍沿用错误的组基线。

## 2. 先让生成与训练共享同一份 token 序列

rollout backend 保存实际生成的 token IDs、停止位置、结束原因、response mask 和 generation_logprobs。训练阶段读取这份 token IDs，不重新 tokenize 一段可能经过格式化的字符串。

单轮实验先使用 temperature=1、无 top-k/p 截断，并记录实际 sampling config。长回复截断、EOS、padding 的处理要能在页面逐 token 检查。

## 3. Reward 与 advantage

小规模实验可用简单可验证的算术题，奖励函数单元测试与训练环分开。真实 rollout 的 responses 由模型生成，不用人工伪造 responses 冒充 rollout。

一个用于教学的组标准化形式为：

$$A_i=\frac{R_i-\mu_R}{\sigma_R+\delta}.$$

该式解释组内比较，不声称覆盖所有实现分支。实际 GRPOAdvantageEstimator 的 normalize/leave-one-out 配置、标准差定义、全组同分行为要沿正式版本继续审计并导出结果。页面允许对比“概念式”和“该次运行的 estimator”。

group 全部同分时可没有有用的组内优势；两次迭代 reward 没上涨不自动说明系统错误。Smoke test 验证闭环与数值，不声称证明训练有效。

## 4. 四种 logprob 必须分开

| 数量 | 生成位置 | 是否参与当前梯度 |
|---|---|---|
| generation_logprobs | rollout backend | 否 |
| prev_logprobs | update 前的 training backend 重算 | 否 |
| current_logprobs | 当前 policy forward | 是 |
| reference_policy_logprobs | 固定参考模型 | 否 |

正式 loss 的 data contract 中保留前述字段；next_token_logprobs 作为当前输入。code 对 token_mask、advantages 和旧 logprob 使用目标位置偏移，不能用未移位的 response mask。[R-LOSS]

分别显示两个差值：generation↔prev 检查训推差异，current↔prev 表示优化过程相对旧 policy 的变化。前者不是 PPO ratio 的同义词。

## 5. Loss 与 SFT 在哪里分叉

SFT 使用正确回复目标的 masked NLL；policy gradient 使用采样 action 的 logprob、advantage、ratio、clipping、可选 KL。它不是把每个 response 当成同权重正确标签再做一次 SFT。

作为 baseline 教学目标可写为负的 clipped surrogate 加 KL；但是实现必须调用或封装正式 `ClippedPGLossFn` 的选定配置，不能把课程公式复制成未经验证的新 loss。

`token_mask × sample_mask` 决定有效贡献，归约可以 token-level 或 sequence-level。两者会改变长短 response 的权重；不能擅自“统一规范”为一种均值。[R-LOSS]

一个容易写错的分支：ratio 前向值为1并不代表它是可替换的常数。正式 force_on_policy_ratio 路径使用 current.detach 作为旧 logprob，以保留当前梯度。用字面常数1改写会改变计算图。[R-LOSS]

## 6. 从 worker 找到真正训练入口

先读 MegatronPolicyWorkerImpl 的职责划分，再沿其 imports 进入 setup、data、train。`ProcessedMicrobatch` 带输入、CP 分片、packed params、masks 与可选 routed_experts；model_forward 后由选定 postprocessor 计算 logprob 或 loss。[R-WORKER][R-TRAIN]

这不是 Bridge SFT gpt_step 的同一数据合约。两个场景共享底层模型和调度能力，但外层控制器、batch contract、loss 与状态生命周期不同。

## 7. Optimizer 更新后还没有完成 RL 迭代

训练权重变为新版本后，要 export/refit 到 generation backend，并验证实际采用的新版本。网页显示：training_weight_version、generation_weight_version、refit 状态与下一批 rollout 的 provenance。

SFT checkpoint resume、HF 格式权重导出、in-memory refit 分开讲。不能把“落盘 save 成功”直接当成“推理服务权重已更新”。

## 8. PPO 是第二条完整路线，不只是改 GRPO 标签

官方 Qwen2.5 PPO recipe 同时配置 policy 和 value：value 模型有自己的 TP/SP、batching 和 optimizer。页面必须增加 value predictions、returns/GAE 与 value loss，显示 actor/critic 两条梯度路径。[R-PPOCFG]

复用现有训练框架实现，不以手写几行 reward-weighted CE 冒充 PPO。

## 9. MoE / SGLang / OPD 扩展

MoE：记录路由 metadata，并审计 Router Replay。缺少 routed_experts 时启用了 replay 的路径会报错，而不是自动退化为“近似相同”。[R-WORKER][R-TRAIN]

SGLang：代码中存在 backend 初始化路线，但特定模型、refit、memory offload、sampling/logprob 参数仍要单独验收。先保留一个原始官方可执行 recipe，再建立新 backend profile，防止多个改动一起引入而无法定位。

OPD：以后在完整循环上引入 teacher logprob 和 distillation objective，先保证符号、mask 和版本定义一致，不在首轮同时实现所有 RL 算法。

## 10. 本章验收

至少两次真实同步迭代能追踪同一权重版本的 samples；四份 logprob 不混用；mask/advantage/ratio 逐 token 可核查；policy 实际更新；refit 后 generation 使用新版本；所有指标有对应 run manifest。无 GPU 时本章源码与流程照常可读，但实测卡片必须显示未执行。
