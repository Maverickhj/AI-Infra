---
type: runbook
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# Trace、来源与验收合约

## 来源不是一个 verified 布尔值

`static_source_read` 表示某个固定源码片段读过；`derived` 表示公式与配置推导；`reference` 表示独立参考实现或 HF 运行；`observed_bridge` / `observed_rl` 表示对应环境的实际执行。

静态源码匹配、数值断言通过、运行完成、性能测量可信是不同属性，不能相互代替。本包全部 model-cases 是 derived，execution_status=not_run。

## 兼容不是版本相等

参考源码使用 `source.lock.json`；运行环境使用各自的 resolved manifest。每次实际执行新增 `runtime_profile` 和 `compatibility`，后者包含 `status`、`required_capabilities`、`checks`。只有全部必需检查有通过记录，才能声明 profile supported；未用的可选能力不要求通过。

`source_lane` 是实际来源的标识，不限定必须为三种预设 lane 名称，也不要求其依赖 SHA 与参考相等。旧 trace 可通过显式元数据迁移适配器补全已有信息；缺少真实测试记录时保持未核验，不能填充伪造 supported。

## 每次实际执行必须记录

run_id、source_lane、runtime_profile、compatibility 检查记录、实际源码 commits 或内容 hash、镜像 tag/digest（如使用容器）、实际模块 import 路径、所用 Python/Torch/CUDA/TE/Transformers/NCCL 版本（未使用组件记不适用）、硬件、backend、dtype、parallelism、完整最终配置及SHA256、输入样本与token ids的SHA256、模型与tokenizer完整revision、chat_template hash、weights_origin、architecture_origin、开始/结束时间、实际命令和结果文件列表。

原模型使用官方checkpoint时记录完整HF revision；自有checkpoint另外记录checkpoint文件哈希、上游模型revision与训练步。随机architecture-scaled模型要写明修改字段和seed，不把它标成原模型的数值结果。`verify_handoff.py`只对课程官方HF checkpoint场景提供最低元数据检查，不是通用训练平台校验器。

必须输出effective config，不仅输出用户覆盖参数。presets、默认配置、环境变量和runtime更新均可能改变最终配置。网页只读导出数据，不具备任意命令执行能力。

## SFT 逐 token 视图

保留原始conversation的sample_id、role/span、完整input_ids、是否已shift、labels、loss_mask、attention边界、position_ids、有效token数；packed case还需sample边界、cu_seqlens_q/kv及padded版本和实际CP切分索引。

精确规定哪个位置的logit预测哪个target。不得在collator和loss中各shift一次。assistant mask取决于实际template和预处理，不手写几个角色special token冒充tokenizer结果。没有tokenizer快照时显示文本span与“尚未tokenize”，不得捏造官方token ids。

捕获该microbatch的loss sum和token count；追踪最终global normalization/gradient scaling。样本内prompt token虽可不直接计loss，但通过attention影响response，其相关梯度不必为零。全mask的局部CP rank与整个global batch没有有效token的情况分别处理。

## RL trajectory 视图

保留prompt_id、group_id、trajectory_id、response/action mask、sample mask、完整tokens、response长度、EOS/truncation原因、policy_version_at_rollout、policy_version_at_recompute、policy_version_at_update、采样参数和每个logprob字段的分布定义。

四种量不能覆盖彼此：generation_logprobs、prev_logprobs、next_token_logprobs（current）、reference_policy_logprobs。reference未启用时明确缺省，不以全零tensor代替模型观测。通过选定loss分支核对位置偏移，避免token/sequence归一化、policy ratio和train/rollout correction混为一谈。

每组reward/advantage有明确来源和计算规则。优势全相同、response为空、被过滤sample、过长截断、模型版本落后分别有测试。同步首例要求先完成update/refit再下一批rollout，但仍然检查训推后端数值差异。

## 收集限制

源码位置是固定commit+path+symbol的锚点。runtime事件记录rank/group、时间域和采样范围。CPU发起调用时间不能写成GPU kernel完成时间；未测量不能画真实毫秒轴。全词表logits、attention矩阵和长序列activations默认不完整dump。

TE融合模块的中间值不可直接hook时，显示“不可直接观测”；通过参考路径得到的中间值标reference。数值调试与性能profiling分开，记录同步操作、hook与tensor复制带来的扰动。

## 本地源码锚点工具

```bash
python tools/audit_sources.py \
  --repo bridge=/absolute/path/to/Megatron-Bridge \
  --repo core=/absolute/path/to/Megatron-LM
```

工具只对显式路径执行git show/rev-parse，读取版本锁指定commit，即使当前worktree不同或有改动也不checkout/reset。blob不匹配、符号缺失或多义会失败，不猜测新的行号。不提供路径的仓库标为not_checked。

参考 commit 不在本地时只是无法完成该源码证据检查，不据此判定运行环境不兼容。实际 profile 的源码映射与行为测试单独执行。

工具不解决所有动态构建、import alias、C++/TE代码或runtime派发。真实后端对象需要在训练环境单独观测，不将AST索引宣称为调用图。
