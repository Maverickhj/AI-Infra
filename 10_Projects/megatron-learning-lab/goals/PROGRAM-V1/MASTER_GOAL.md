---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# PROGRAM-V1：连续完成 Megatron Learning Lab 的一组里程碑

## 目标与基线

在现有工程上交付一套可学习、可演算、源码可追溯的完整模型与 SFT/RL 学习工作台，并准备和验证实际训练接入。用一个原生 `/goal` 驱动下列相关里程碑，不逐阶段等待用户发“继续”，不重建另一套站点。

本次读取的 main 为 `538afeb1745f69ec7c51d20d467b83a011d23752`（2026-10-03），已有 GQA 子步骤、CPU/TS 对照、源码映射和课程。G01 的 PROGRESS 记录 ready_for_review、51 项 Python、16 项浏览器与数值校验；这是提交者记录，本次规划没有独立复跑这些测试。M00 必须重验当前工作区，不将上述计数当成永久验收阈值。

## 执行入口与规则优先级

启动 PROGRAM-V1 时，本文件取代旧 CODEX_START.md 的首轮范围，以及 G01 中“只做 G01、完成后停止、不做后续阶段”的范围限制。G01 的技术断言、质量标准、数学边界和安全约束仍保留；不改写其历史 evidence 来迎合新工作区。

读取顺序：仓库/项目 AGENTS.md → 本文件 → plan.json、STATE.json、PROGRESS.md → STAGES.md 中当前阶段 → 必要源码、课程和 G01 历史记录。不要每轮重读所有长文。项目内其他同名 GOAL.md 不会自行激活原生 Goal。

本次授权的是当前项目范围内的一系列实现和测试，不是无限 backlog。后续每阶段可以有 checkpoint、自审和阶段报告，但通过后立即进入下一个依赖满足的阶段。只在整体验收完成、全部剩余工作明确阻塞、用户暂停/中断或会话预算限制时停止。

## 阶段与依赖

| 阶段 | 交付结果 | 执行条件 |
|---|---|---|
| M00 | 接续 G01，重跑基线，建立本轮证据 | 现有本地开发环境 |
| G02 | SFT 样本、shift、mask、packing 的可演算流程 | CPU；真实 tokenizer 缺失时保留明确边界 |
| G03 | 完整微型 decoder、loss、backward、update、resume | CPU 参考，非只有 FFN |
| G04 | 同一完整模型中的 TP/DP 布局与梯度归约 | CPU 数值切分参考，不冒充 NCCL |
| G05 | PP 时间调度、SP/CP 序列语义与 packing | CPU 逻辑时间与 attention 边界验证 |
| G06 | Qwen MoE 与 DeepSeek MLA/MoE 的完整差异 | 确定性小模型参考，非全量 V3 |
| G07 | GRPO/PPO 概率、优势、梯度与版本更新教学闭环 | CPU 数值参考，非真实 rollout |
| G08 | 最小兼容检查、SFT/RL 接入脚本与统一 trace 接口 | 可做配置/合约测试，不需启动 GPU |
| R01 | 真实 HF 对齐、Bridge SFT、保存/恢复 | 已授权且资源就绪 |
| R02 | 真实同步 RL rollout/update/refit | 已授权且资源就绪，依赖 R01 |
| G09 | 全站整合、Run Compare、全部回归与交付报告 | 不依赖 R01/R02 成功；必须披露其状态 |

`plan.json` 是固定队列和依赖的机器可读入口；STAGES.md 是各阶段的具体交付/验收。阶段不是新仓库、独立原生 Goal 或需要人工接力的工单。

## 每一轮如何推进

1. 先 `git status --short`，读取 STATE/PROGRESS，运行 `python tools/program_plan.py next`。若有新提交或无关改动，先比较，不 reset、不丢弃。
2. 选择当前可执行阶段，记录 3–6 个行为验收及准确文件范围。按“内容和数学 → 独立参考/反例 → UI 和源码 → 测试/自审”完成一个垂直切片，不先搭万能平台。
3. 运行阶段数值/合约用例、相关浏览器路径和共享回归，产出本轮命令、退出码、日志、源码指纹。通过后更新阶段报告与 STATE，继续下一阶段，不停下来询问是否继续。
4. 若独立工作被外部条件阻塞，记录 blocker、实际尝试、证据和解阻条件，继续其他依赖满足的阶段。依赖失败不能绕过；没有新假设不要重复安装或联网。
5. 最终 G09 在同一个最新源码快照上重跑全部软件阶段检查。不能拼接多次不同工作区的旧绿色报告作为整体验收。

默认一个工作区串行写入；子代理仅作只读审阅或写互不重叠的文件，整合由主会话负责。不要另建会话/Goal 来覆盖当前目标，也不要通过 shell 无限循环调用 Codex。

## 品质与资源边界

读者是有一定基础的初学者，基础一页可跳过；核心仍须符号表、shape、计算、源码条件、反例和验证。既不能只做卡片，也不能把高层框架代码伪装成 kernel。

沿用 source.lock.json 的参考快照，不为“追最新”自动改写整个项目。实际环境按 `profiles/minimum-compatibility.md` 最小接口和语义验收。复用已有依赖；确需 CPU 依赖时仅安装当前路径所需、遵守已有审批，不全量复制上游 lock。CPU 环境不可用也要如实 blocked，不能把未执行的数值验收改成文字说明。

允许修改本项目 web/content/experiments/tests/tools/profiles、必要 research/package scripts 和当前阶段记录；不改仓库级设置，不迁移其他笔记，不批量升级依赖，不减弱断言，不默认跳过旧测试。不随意修改 PROGRAM 的目标/验收来求通过。

本次提交计划不等于授权后续 GPU、集群、模型大下载、系统配置或远程服务。R01/R02 依 RESOURCES.md 的一次性资源授权清单执行；未授权时先完成 G08 的脚本/接入，不能实际开跑。默认不 commit/push，不公开部署，不上传私有数据。需要后续自动提交时，由用户另行指定分支和写入策略。

## 证据、回归与历史记录

`tools/program_plan.py` 只检查计划/状态结构和证据文件完整性、提示下一阶段；不执行测试，不控制原生 Goal，也不能证明报告里的测试结果真实。实际完成必须满足 VALIDATION.md 的运行和审读要求。

G01 final 绑定旧 evidence 的内容 hash。后续实现可能修改其依赖；保留旧报告作为历史，不修改旧 evidence 然后宣称原报告仍对当前版本有效。M00 用当前基线与 test:gqa 重验。后续复用 G01 的测试与执行工具，不反复要求旧 final 的历史指纹匹配整个新项目。

涉及已有阶段的公共模块修改，立即重跑受影响阶段；先前阶段报告仍只说明当时的快照。G09 必须包含全部阶段回归和新前端用户路线。

## 完成状态

阶段 `validated` 表示对应快照已执行验收，`reviewed` 仍为 false，并不要求用户每阶段确认才能继续。

- `full_project_ready_for_review`：软件阶段和 R01/R02 都已验证，G09 的最终报告覆盖当前整合状态。
- `software_ready_runtime_blocked`：M00/G02–G09 全部验证，但 R01/R02 仍缺资源/授权/接口。只可报告“软件学习版完成、真实运行待验证”，不可将总项目标为完成；保存 blocked 状态后停止无效等待，资源就绪可恢复。
- `blocked`：仍有软件必需验收无法完成且没有独立可执行阶段。报告具体缺口，不能用文档占位转成完成。

预算、上下文、断网或用户暂停都不是成功。恢复同一线程时先查看 `/goal` 和本地记录；新线程使用 START_PROMPT，重新读账本和最新 HEAD 后接续，不重做已验证阶段。每次恢复重新检查阻塞条件是否真的改变。

## 官方依据

2026-10-04 核对：[Using Goals in Codex](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex)、[Follow a goal](https://learn.chatgpt.com/use-cases/follow-goals)。原生 Goal 是线程内的持久目标；本目录是项目内阶段约定，不是官方多 Goal 调度 API。沿用现有 sandbox/审批与客户端预算，文件不会扩大权限或保证一次会话耗时/额度足以完成全部阶段。
