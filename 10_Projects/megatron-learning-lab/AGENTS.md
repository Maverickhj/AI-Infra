# megatron-learning-lab 工作约束

## 目标

交付面向有一定基础的初学者、由完整模型和真实训练场景驱动的学习体验。基础概括，核心精讲；前置基础合并为一页可跳过速览，按 CURRICULUM.md 控制深度。优先 Qwen3-0.6B 全链路、Qwen2.5 对照、DeepSeek MLA+MoE 差异。不得以独立 MLP、通用源码索引或空课程目录替代。

## 版本与语义

source.lock.json 固定调研参考源码，不是安装锁。实际环境优先复用，只要求当前 case/profile 的接口与数据/计算语义一致，不强制匹配上游 Python/Torch/CUDA/TE/Bridge/Core 的全部版本。按 profiles/minimum-compatibility.md 检查必要接口、ABI 与实际行为；只修复所需依赖，不批量升级、不安装所有 optional extras。

Bridge-SFT 与 NeMo-RL 的实际来源分别记录，可在兼容验证通过后共用环境，冲突时再隔离。运行源码和参考源码分别定位。版本不同不自动失败，版本相同不自动通过；缺失功能只阻塞依赖它的 profile。

模型图来自完整 HF config 与 family bridge。head_dim 优先显式字段；DeepSeek-R1-Distill-Qwen 不是 MLA；SFT recipe 创建结构不代表权重已加载；带 labels 的 GPT 默认输出可为 token loss。遵守 research/FINDINGS.md。

## 实施

先读旧工程再复用，不删除/重置/覆盖未提交修改，不自动 commit/push、公开 Sites 或启动远程 GPU 作业。不要递归扫描用户 home。

首轮范围由 CODEX_START.md 控制。每个任务必须同时实现内容、交互和行为测试，不先做万能平台。源码阅读器从现有 evidence entries 做起，不做全仓动态调用图。

## 证据

derived/reference/observed_bridge/observed_rl 明确分开。没有运行不生成实测 loss、耗时、显存、heatmap；未完成显示未完成。静态审阅只证明读过记录范围，不能冒充 runtime trace。

实际实验必须锁 checkpoint/tokenizer revision、输入与配置 hash、backend、dtype、parallel groups、实际模块路径和源代码 lineage。获取不到 revision 时可展示带警告的 config 档案，不得开始标称 reproducible 的 observed run。

源码原文与解释/伪代码分开，不编行号、不修改原文。少量必要 tensor 捕获，默认禁止全模型激活/logits dump，profiling 与数值对齐分别运行。

## 文档与验收

中文教学，保留独立数学符号表，新增符号同步更新。Markdown 行内 `$...$`、块级 `$$...$$`。每章以整模/样本定位进入核心，不重复基础推导。精讲内容必须保留数学/shape、实现分支、演算、反例与验证；符号表不可因缩短篇幅删除。

只报告实际执行的测试。无 GPU 不阻止 E0 课程与页面完成，但不伪造 E2/E4。结束提交实现范围、启动命令、测试记录、待验证项和阻塞原因。
