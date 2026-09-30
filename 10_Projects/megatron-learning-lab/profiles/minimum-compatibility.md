---
type: runbook
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# 最小接口兼容与运行环境策略

## 1. 两类版本记录，各司其职

`source.lock.json` 固定此前稳定版源码调研的引用位置，是阅读证据，不是 learning-lab 的安装锁。其 Python、Torch、TE、Transformers、Bridge/Core 版本记录不得转化为全项目的硬性版本相等检查。

实际运行环境优先复用已有可用环境，并按本文件的场景合约验收。满足所选路径的接口、数据和计算语义即可，不要求与上游锁文件逐包一致。成功执行后记录实际版本、源码位置和本地改动，以便复现这一条路径。

底层依赖的真实 Python/CUDA ABI、硬件和包声明约束仍须满足；不能以“最小化”为理由忽略 import 错误、kernel 不支持或训练行为差异。出现冲突时只调整必要依赖，或为冲突任务隔离环境，不先全量重装。

## 2. 一致性验收检查什么

| 层面 | 最小要求 | 不能仅凭什么判定通过 |
|---|---|---|
| 可调用接口 | 所用构造/forward/loss/step/save/refit 可调用；参数可由显式适配器映射 | import 成功、函数名相同 |
| 数据合约 | shape/layout/dtype、shift、mask、packed 边界、rank/group、logprob 含义与所选课程一致 | JSON key 存在、shape 看起来相同 |
| 计算语义 | 选定 forward/loss/梯度/更新/恢复行为满足已声明断言 | loss 有限、程序未报错 |
| 来源可追溯 | 原始源码与运行源码分开记录；测试绑定当前 profile、版本与配置 | 版本号与参考版相同 |

允许“字段名/返回封装不同，经薄适配器归一到 lab 合约”。例如 batch 字段名、SBH/BSH 布局、旧/新 packed metadata 可以显式映射；必须记录转换和测试。不能把改变 loss reduction、丢弃 mask、伪造 logprob、替换 MoE 路由或把 token loss 再当 logits 算 CE 称为兼容适配。

## 3. 按场景最小化能力集

| Profile | 必需能力 | 不应强制安装/启用 |
|---|---|---|
| `read_only` | 课程、模型元数据、固定源码入口、HTML 交互/静态构建 | Torch、CUDA、Bridge、TE、Ray、rollout 后端 |
| `hf_reference` | 所选模型的 tokenizer/config、权重加载、teacher-forced forward、目标 token logprob | Megatron 分布式环境、全部模型 backend |
| `bridge_sft` | 所选完整模型的构建/权重导入；batch/forward/loss；backward/step；save/resume；trace 导出 | 未使用的 CP/EP、FP8、长序列、MoE、RL 环境 |
| `parallel_sft` | 已验证 SFT 加当前启用的 group/layout/collective、梯度归约；启用 packing 才检查其边界 | 其他并行维度或所有排列组合 |
| `rl_grpo` | rollout/action token、reward/group、实际算法所需 logprob/mask/advantage、可导 policy loss/update、refit 与版本标记 | critic；未启用 KL 时的参考模型；异步/多轮/Router Replay |
| `rl_ppo` | 已验证 RL 数据与更新路径，加 value/returns/GAE/value loss 和相应状态 | 与该 PPO 变体无关的扩展 |
| `moe_mla` | 所选模型的 MLA/MoE 结构、映射和路由；启用 EP/replay 时再验收这些能力 | 其他家族或仅为最新特性存在的依赖 |

未启用功能记 `not_applicable`，不可伪装成 `supported`；需要但缺失的能力只阻塞相应 profile。无 GPU 的机器仍可完成 `read_only`。同一环境通过 SFT 不代表通过 RL；通过 GRPO 不代表通过 PPO。

## 4. 实施顺序

1. 选择 case/profile，列出实际必需接口，不遍历安装所有 optional extras。
2. 盘点已有环境：实际 import 路径、包版本、所需二进制能力；读取目标函数签名和实际实现。
3. 执行小规模行为探针：模型权重来源、forward 返回类型、shift/mask、loss sum/count/归约、梯度与更新；RL 再检查 sampling/logprob/refit。阈值在实验前声明。
4. 存在可机械适配的差异时新增薄适配器，并保存 adapter 源码与测试；其他差异标 `unsupported`，不能吞掉异常或静默退化。
5. 测试通过后输出该环境的 resolved manifest。只在依赖冲突或隔离确有需要时分环境；SFT/RL 的来源记录始终分开，即使实际共用一个环境。

状态为 `not_checked`、`supported`、`unsupported`、`not_applicable`。`supported` 只针对所选模型、配置和能力路径，必须附检查名、执行命令、断言/容差及结果；函数签名检查不是数值测试。

## 5. 最小报告内容

profile id、case/model id、所需能力及检查状态、实际软件版本/import 路径、实际源码 revision 或内容 hash、adapter revision/hash、effective config、输入/checkpoint/tokenizer 标识、检查命令、断言与容差、测试结果、未启用和未验证能力。

不使用容器时 image digest 标 `not_applicable`。没有 TE 的只读/HF 路径不要求 TE 版本；使用 TE 的训练路径必须记录并检查实际 TE。额外可选包的版本不同不能单独导致失败。

`profiles/compatibility-contract.json` 仅定义 lab 的验收策略；`tools/verify_handoff.py` 检查报告字段与状态，不能自动证明真实框架兼容。真实行为探针由 Codex 在选定运行环境实现和执行，未执行保持 `not_checked`。

## 6. 源码阅读工具与兼容性探针不能混用

`tools/audit_sources.py` 检查参考 commit 的 blob/symbol 是否与证据清单一致，不判断运行环境是否合格。当前本地仓库没有参考 commit，只意味着该参考尚不可本地查阅，可继续使用固定远程链接。

实际运行版本不同，应追加 runtime source mapping，以实际 import 文件/符号重新定位，保留参考原文。缺失实际符号时停止该源码映射，不猜行号；但不因此删除历史参考或重装整个环境。

## 7. 两个必须测试的反例

版本与参考不同，但所需接口及小规模行为测试均通过：接受该 profile，显示实际来源。

版本与参考相同，但 loss 返回类型、mask/归约或 refit 行为与要求不符：拒绝该 profile，定位不一致项。

结论：最小化的是环境和功能依赖，不是语义验证。
