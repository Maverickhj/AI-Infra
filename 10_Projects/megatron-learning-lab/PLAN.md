---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# megatron-learning-lab 实施方案

## 1. 产品目标

面向有一定基础的初学者：已会基本 Python/PyTorch、tensor shape、forward/backward 和训练流程，但尚不熟悉 Megatron 的整模实现与分布式训练。最终能独立解释并沿源码定位：一条对话如何变为模型输入；Qwen/DeepSeek 完整 decoder 如何计算；SFT 与 RL 为何共享骨干却有不同的数据、目标与状态；分布式并行如何改变同一条链；真实执行的输出如何验证解释。

不把课程做成 MLP demo，不把源码学习做成通用代码搜索器，不把实际场景做成“loss=0”的运行占位。第一阶段不新增账号、数据库、远程 GPU 调度、内置 AI 问答或平台专属存储。

### 学习深度

基础内容概括，核心内容精讲，补充材料按需查阅。基础铺垫合并为一页可跳过的速览，默认正文篇幅目标 10%–15%；不从 Python、矩阵乘法、Linear 或 autograd 重新开课。

Qwen/DeepSeek 计算、参数映射、SFT mask/shift/归一化、RL 概率与更新、并行布局与通信必须展开为“数学/shape → 源码 → 演算 → 反例 → 验证”。完整规则见 [CURRICULUM.md](CURRICULUM.md)。压缩的是铺垫与重复，不是关键假设与符号表。

## 2. 三条同时可见的学习轴

**模型轴**：Qwen2.5 Dense → Qwen3 Dense → Qwen3 MoE → DeepSeek-V2-Lite/V3。始终展示从 tokens 到 LM head 的全结构，用户可展开某层/算子。切换模型时不仅换参数，还要变更 operator graph、参数布局、norm 与 expert 类型。

**场景轴**：teacher-forced forward → SFT update → 同步 GRPO update → PPO actor/critic → checkpoint export/refit。首页就可看到 SFT 与 RL 的整体差别，不把 RL 藏在最后一个空章节。

**系统轴**：单 GPU → TP/DP → PP/SP/CP → EP → recompute/overlap。所有并行知识依附于上述完整模型和场景，不重新建立一套不相关的 toy 数学题。

## 3. 页面规格

| 页面 | 默认呈现 | 必须发生的真实交互 |
|---|---|---|
| Model Atlas | 4 个主模型完整骨架与准确差异 | 切换 Qwen2.5/Qwen3/DeepSeek，图与 shape 改变 |
| Sample Journey | 一条多轮 conversation 到训练目标 | token role、mask、shift、padding、packing 联动 |
| Model Walkthrough | Qwen3-0.6B 全层视图，可展开一层 | Q/K/V、norm、RoPE、SwiGLU、残差、LM head 到源码 |
| SFT Step | 加载→batch→forward/loss→backward→step→save | 权重来源、trainable 参数、loss sum/count、更新状态可查 |
| RL Cycle | rollout→reward→logp→advantage→update→refit | 固定一个 trajectory，查看四类 logprob 与权重版本 |
| Run Compare | 两次配置/执行的差别 | 来源、mask、数值误差、通信/显存指标逐项比较 |

一个页面可采用 tabs，但不要将所有内容永久压成四栏。主画布与当前步骤说明是视觉中心；源码通过侧栏/下方面板按需展开。公式与符号表可单独阅读。默认不自动播放，支持上一步/下一步、键盘操作、deep link 到 case/step。

## 4. 当前步骤的数据合约

每个 step 具备 id、教学问题、输入输出语义、shape 推导、适用模型/配置、source IDs、分支条件、观察方法、自测。共有一个 `selectedModel / scenario / layer / step / sourceLane` 状态，不允许各组件维护不一致的副本。

来源类型必须独立显示：

| evidence | 含义 |
|---|---|
| static_source_read | 固定源码版本已读过，未运行 |
| derived | 根据配置和数学计算，例如投影 shape |
| reference | 教学实现或 HF 单独运行结果，不是 Megatron |
| observed_bridge | 指定 Bridge 环境实际采集 |
| observed_rl | 指定 RL 环境实际采集 |

“缩小版”“模拟”“官方原模型”“实测”不能只写在页脚。每份 run manifest 都记录 architecture_origin、weights_origin、source_lane、模型/tokenizer revision、backend、dtype、实际模块路径、parallel groups 与配置 hash。

## 5. 源码学习方式

优先使用本包 27 个证据入口，按真实 case 扩展，不先建全仓调用图。阅读顺序为：recipe 和 family bridge → provider/spec → GPT forward → layer/attention/MLA/MoE → batch/loss/调度 → 状态与通信。

代码定位使用固定 repository+commit+path+symbol。`source.lock.json` 只固定参考证据，不强制安装该版本组合。实际环境不同则记录 runtime source mapping，不静默替换参考原文。语义图、静态调用路线、runtime trace 分别标注。不能把一个理论步骤声称为实际 kernel 或独立 nn.Module。

### 最小运行兼容

优先复用现有环境，只要求所选 case/profile 的接口、数据与计算语义满足 lab 合约，不要求逐包匹配 Megatron-Bridge 锁文件。依赖缺失或冲突只修复必要部分，未使用功能不安装、不验收。检查实际 import/签名只是第一层，仍须通过 forward 返回类型、mask/shift、loss 归约、梯度/update、save/refit 等所需行为测试。

具体规则和按场景的能力矩阵见 [最小兼容策略](profiles/minimum-compatibility.md)。SFT/RL 保留各自运行来源记录；只有确有依赖冲突才要求隔离环境。版本不同本身不判失败，版本相同也不自动通过。

## 6. 实验层级

E0：无需 GPU。完整课程、源码、根据 config 推导的 shape 和小数据合约可正常学习。不得用 random logits 伪造模型行为。

E1：HF 参考。真实 tokenizer/config，小模型真实权重进行 teacher-forced forward，采集少量 tensor 与 sampled-token logprob。只安装该路径必需依赖，存在冲突时才与 Bridge 隔离。

E2：Bridge 真实短序列 SFT。Qwen3-0.6B 优先，Qwen2.5-0.5B 作对照。固定权重与数据，TP=PP=CP=1、BF16、短序列开始。所选训练路径需要 TE 时检查其兼容性；允许其他经验证的 backend，但必须标明实际路径与差异，不把替换实现冒充官方 backend。

E3：固定模型增加并行、LoRA、packing/CP，比较 loss/grad、source path 与真实内存。按能力分别启用，不要求一次组合所有开关。

E4：以 NeMo RL 稳定版源码为参考、在通过最小兼容验收的环境中执行同步 GRPO；同 rollout/update 版本，先无多轮工具和异步。第二条完整例子使用已经核对的 Qwen2.5 PPO recipe，引入独立 value 训练与 GAE。SGLang 作为真实 backend 对照任务，不盲改 vLLM recipe。

E5：Qwen3 MoE、DeepSeek。课程首轮即存在，真实大模型运行延后到资源和兼容性满足时。V3 architecture-scaled 用于结构/通信实验，不能替代原模型功能与能力验证。

## 7. SFT 与 RL 必测行为

SFT：证明不是随机初始化；验证 template/tokenizer 一致；检查单次 shift、assistant/last_turn/full mask；padding 及 packed 跨样本边界；全局有效 token 归一化；nonzero gradient 和实际参数变化；PEFT 下冻结参数不变；保存后 resume 包含所需训练状态，而 HF export 只是另一个用途。

RL：generation/prev/current/ref 分离；输入 token 与 mask 对齐；prompt/group/trajectory ID 保留；advantage estimator 的实际分支可解释；ratio/clip/KL 与选定实现对照；update 前后版本明确；refit 后固定输入的 logprob 检查；短 rollout 两次迭代完成只证明闭环，不证明能力提高。

MoE：路由、expert assignment、combine 与 shared expert 单独展示；replay 和未 replay 明确区别；aux loss、padding 排除、专家偏置不能被推理导出遗漏。MLA 训练激活与 decode latent KV cache 不是同一个图。

## 8. 交付阶段及硬门槛

P0 内容与兼容边界：保留参考 source lock/model manifests，新增按需 runtime 合约；迁移旧站的可用模块，压缩基础铺垫。**不能只产出审计报告便称首轮完成。**

P1 可学习的整模闭环：Qwen3 完整 token→loss→update 概念路线；Qwen2.5 比较；DeepSeek 完整差异；SFT mask 操作；RL 全周期课程。E0 可运行，实测位显示未运行。至少 3 个用户路线的浏览器测试通过。

P2 真实 SFT：E1/E2，预训练权重、loss、gradient/update、save/resume、trace 导入。无 GPU 时留下可执行 runbook 与明确未执行状态，P1 仍可独立验收。

P3 同步 RL：通过场景接口与语义验收的 RL 环境、实际 rollout/reward/logprob/update/refit。复用算法与 worker，不自行写一套 PPO。将 runtime lineage 带入每条 trace。

P4 并行与 MoE/MLA：在相同模型/场景上启用并行和大模型案例，最后才做性能优化。

每个阶段先确定 3–6 项行为验收，再实现与测试。P1 额外验证基础速览可跳过、核心说明未被统一压缩；P2/P3 额外验证版本不同但合约通过可接受、版本相同但语义不符应拒绝。不按代码行数、路由数量、动画数量或章节目录数量宣告完成。

## 9. 技术实现建议

复用旧工程中合适的 React/TypeScript、Markdown/公式渲染和静态构建；不为重做而重做。公式正文 Markdown 使用 `$...$`/`$$...$$`，保留独立符号表。数据来自本包 model/case JSON，case reducer 驱动组件。

源码与课程可离线缓存，用户输入数据/私有源码不公开上传。Sites 只是可选部署，构建与验收不依赖其权限。浏览器只展示导出的数据，不执行 arbitrary Python、shell、远程训练或私有 GitHub token。

## 10. 验证报告应回答什么

“实现了什么用户行为；测过哪些路径；测量来自什么环境；哪些结果只是推导；哪些测试未运行；哪项限制阻碍下一阶段”。不得把 unit test 通过写成 Megatron 数值对齐完成；不得把无法访问 GPU 隐藏在绿色 mock 指标后。
