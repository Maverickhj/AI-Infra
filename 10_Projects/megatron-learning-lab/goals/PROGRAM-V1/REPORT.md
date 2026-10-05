---
type: project
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# PROGRAM-V1 交付与证据索引

当前状态：**software_ready_runtime_blocked**。M00/G02–G09 的软件学习版已通过验收；R01/R02 的真实 HF/Bridge/GPU/SFT/RL 仍未执行，不将整个项目或原生 Goal 标为完成。

验收时的基点为 Git d060187da36b40defd0c599774495b0c6be51441，结果覆盖当时其后的未提交改动，具体内容以源码清单指纹为准。验收期间没有 commit/push、公开部署、模型下载或 GPU/Ray 训练；验收完成后用户明确授权提交和推送这批交付改动。所有代码与检查在 minimind-megatron-bridge-dev-1 的项目目录执行，reviewed=false 表示仍待人工审阅。

## 当前同一源码快照的验收

最终入口：容器内运行 npm run test:release。实际完成时间：2026-10-05T00:31:20.824632+00:00。

- [最终报告](../../runs/program-v1/G09/20261005T002421065004Z/report.json)，SHA256：`6a8d85f74e1bf9e935d382c5292558ee548e3ada344f250c1aeca9997bf5c3d2`。
- [独立复核](../../runs/program-v1/G09/20261005T002421065004Z/audit.json)：逐项核对报告、命令/日志/artifact hash、当前源码、依赖、配置、dist 及生产字体记录。
- [源码清单](../../runs/program-v1/G09/20261005T002421065004Z/source-manifest.json)，指纹：`2b71778880f36b2ecae8e8bfc4b963f191215ca14079eac0ad339d3179812ad3`。
- [依赖记录](../../runs/program-v1/G09/20261005T002421065004Z/dependencies.json)，文件 SHA256：`0779a326e62d8225ca2c803301198939e36735ff4d4ead17a3e428530c657ec0`。
- [配置清单](../../runs/program-v1/G09/20261005T002421065004Z/configuration-manifest.json)，指纹：`f8848248fd35e8fe307bc42ec416af8f45ae29d61757a9249c8d48dffa30f2ae`。
- [构建清单](../../runs/program-v1/G09/20261005T002421065004Z/dist-manifest.json)，指纹：`602d77d056cc1765bce2b3c96693b1d283bfb1a597a75ccedb4cef5a6f9c537e`。
- [机器可读阶段索引](../../runs/program-v1/G09/20261005T002421065004Z/stage-index.json) 与 [阶段账本](STATE.json)。

| 检查 | 实际结果 |
|---|---|
| 全部 Python | 178 项通过，无 skip |
| 完整 Chromium | 99 项通过；skipped/flaky/unexpected=0 |
| 独立生产 smoke | 3 项通过；skipped/flaky/unexpected=0 |
| 独立数值/合约入口 | 8 项入口全部执行通过，见下表 |
| TypeScript / Vite | typecheck 与生产构建通过 |
| 固定源码 | 49 个完整文件 / 111 段原文逐字及 Git blob 核验 |
| 数据/路由/trace/diff | 数据生成、handoff、源码路由、trace 合约、git diff --check 通过 |
| 证据完整性 | gate 内检查与外部 report_issues 复核均通过 |

正常 git diff --check 成功时输出为空：原始空日志保持不变，[命令回执](../../runs/program-v1/G09/20261005T002421065004Z/empty-output-diff.json) 记录真实 argv、退出码、空文件 SHA256 和 output_bytes=0，没有伪造成功文本。

| 专项 | 场景与数值范围 | 最大绝对误差 | 证据 |
|---|---|---|---|
| test:gqa | 1872 标量 / 48 个 masked 位置 | 3.552713678800501e-15 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-gqa.log) |
| test:sft-data | 96 场景 / 3014 标量 | 1.4210854715202004e-14 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-sft-data.log) |
| test:decoder | 6 场景 / 26790 标量 | 1.7763568394002505e-15 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-decoder.log) |
| test:tp-dp | 实际 CPU 全模型/梯度检查 | 见专项日志 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-tp-dp.log) |
| test:sequence | 25 场景 / 5448 标量 | 4.440892098500626e-16 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-sequence.log) |
| test:moe-mla | 9 场景 / 72033 标量 | 2.886579864025407e-15 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-moe-mla.log) |
| test:rl-reference | 16 场景 / 54957 标量 | 7.993605777301127e-15 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-rl-reference.log) |
| test:runtime-contracts | 53 跨语言案例，2 份新 CPU trace | 见专项日志 | [日志](../../runs/program-v1/G09/20261005T002421065004Z/regressions/test-runtime-contracts.log) |

误差只适用于各自声明的 authored CPU float64 fixture；不能作为 BF16/GPU 或真实引擎的验收结果。各专项保留手算、独立 CPU 对照、错误实现反例和预先声明的容差，未通过减少断言或跳过旧测试完成验收。

## 生产与用户路线

生产配置单独在 127.0.0.1:5174 启动同一 dist，不复用开发服务；检查真实脚本来自 /assets/*.js，没有 Vite dev 客户端。覆盖样本/监督 → 完整 decoder/更新恢复 → TP/DP/PP → DeepSeek MLA/MoE → PPO/value/refit → 当前源码/缺运行来源 → Run Compare，并验证基础可跳过。

直接及嵌套 hash 路由重载、本地 KaTeX 字体、离线切换/公式、键盘焦点/横向表格滚动、390 px 窄屏和缺少梯度/通信证据的降级均已执行。浏览器为 Chromium 153.0.8010.12，字体均来自本地，外部请求列表为空；Firefox/WebKit 未执行。

已实际查看本轮归档的 [手机截图](../../runs/program-v1/G09/20261005T002421065004Z/production-artifacts/release--G09-narrow-produc-933c2-yboard-and-missing-evidence-chromium-production/comparison-mobile.png) 与 [桌面截图](../../runs/program-v1/G09/20261005T002421065004Z/production-artifacts/release--G09-narrow-produc-933c2-yboard-and-missing-evidence-chromium-production/comparison-desktop.png)。手机在键盘操作后的表格局部横向滚动可见，页面无整体横向溢出；桌面可查看 A/B loss、未采集梯度、来源/版本和派生通信边界。导入仍为 imported_claim；零差不等于运行身份、完整梯度或跨引擎等价。

## 各阶段报告索引

下列 M00/G02–G08 报告属于历史快照；本轮上述 8 项专项和完整基线重新验证最终整合代码，未拼接旧绿灯代替当前验收。完整报告 SHA256 保存在 STATE.json。

| 阶段 | 状态 | 报告 / SHA256 前缀 |
|---|---|---|
| M00 · 接续与基线 | validated | [report](../../runs/program-v1/M00/20261004T150812693403Z/report.json) · 684e6b85b000 |
| G02 · SFT 样本与监督 | validated | [report](../../runs/program-v1/G02/20261004T153109683149Z/report.json) · 665388bd6bbe |
| G03 · 完整 decoder 与更新 | validated | [report](../../runs/program-v1/G03/20261004T155324142852Z/report.json) · 341f259d8256 |
| G04 · TP / DP | validated | [report](../../runs/program-v1/G04/20261004T164510932004Z/report.json) · f5d4980f19f8 |
| G05 · PP / SP / CP | validated | [report](../../runs/program-v1/G05/20261004T170057490328Z/report.json) · 44e4ad896a07 |
| G06 · MoE / MLA | validated | [report](../../runs/program-v1/G06/20261004T174656964753Z/report.json) · 13b69f18abf5 |
| G07 · RL 数值与更新 | validated | [report](../../runs/program-v1/G07/20261004T183620097709Z/report.json) · 26ed98a0a886 |
| G08 · 真实引擎接入软件 | validated | [report](../../runs/program-v1/G08/20261004T232220705471Z/report.json) · 6b68a771300d |
| R01 · 真实 HF / Bridge SFT | blocked_external | [当前缺口](../../runs/program-v1/G09/20261005T002421065004Z/runtime-readiness.json) |
| R02 · 真实同步 RL | blocked_external | [当前缺口](../../runs/program-v1/G09/20261005T002421065004Z/runtime-readiness.json) |
| G09 · 整合与最终交付 | validated | [report](../../runs/program-v1/G09/20261005T002421065004Z/report.json) · 6a8d85f74e1b |

## 真实运行缺口与恢复条件

[本轮只读前提检查](../../runs/program-v1/G09/20261005T002421065004Z/runtime-readiness.json) 记录三个样例均 not_ready/not_run：指定 Qwen3 config/weights/tokenizer snapshot 缺失，RL 还缺实际 NeMo checkout，PPO 缺 critic 文件；资源授权清单不存在。本轮只读元数据未导入 torch/ray/transformers/nemo_rl；构建及 CPU 参考不启动真实训练。

R01 仍需真实 tokenizer/template 与三种 mask、单次 shift、同权重 HF/Bridge 选定 LP/激活误差、非零梯度/参数更新、save/resume、HF export 和当前网页导入。R02 仍需真实生成 actions、reward/group/masks、可导 LP、正式 GRPO/PPO 分支、actor/critic 更新、两次同步迭代及 refit 后固定输入训推对齐。

解除条件：明确设备/容器及归属、设备数量、步数、时限、已有 pinned 模型/tokenizer 和 NeMo 路径、数据/输出目录及是否允许下载；PPO critic 单独授权。上次只读发现的 6 GiB GPU 低于当前全参数 Adam 配置约 7.77 GiB 的权重/master/moments 推导项（还未计梯度/激活）；这是理论下界项，不是显存实测。不能默换小模型/LoRA 后宣称原验收完成。

资源就绪后先 R01，再 R02；若集成代码或依赖改变，将 G09 标为 revalidate 并重跑最终 gate。当前没有剩余可独立执行的软件阶段，原生 Goal 保持未完成。

## 启动、限制与证据保全

启动与复验命令见 [README](../../README.md)，具体自审见 [G09-REVIEW](G09-REVIEW.md)。当前 Python prefix 为 /opt/venv，实际依赖版本按该解释器优先级记录；版本元数据不能证明 CUDA/ABI/训练兼容性。前端主 chunk 约 1.65 MB（未 gzip），构建有体积警告，未做用户设备性能基准。

新 checkout 缺少 ignored runs 时，不能直接继承 validated：需要补回对应证据或依 planner 重跑缺失阶段，缺固定源码缓存时再显式获取。原始模型、长日志、截图、生成数据和缓存均沿用现有忽略规则，没有加入 Git。

诊断保留：[首次源码变化拒绝](../../runs/program-v1/G09/20261005T000716382044Z/report.json)、[第二轮账本完整性诊断](../../runs/program-v1/G09/20261005T001441019715Z/integrity-diagnostic.json)、[初始生产路径失败](../../runs/program-v1/G09/preflight/production.log)。这些记录没有用作 G09 的成功证据。
