---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# PROGRAM-V1 进度

当前：M00/G02/G03 validated，G04 in_progress。当前 HEAD `8a61e3b`，工作区接续已有 G01。所有代码/测试在 `minimind-megatron-bridge-dev-1` 执行，不 commit/push。

## M00 · 2026-10-04

报告：`runs/program-v1/M00/20261004T150812693403Z/report.json`，源码指纹和实际文件清单见同目录 `source-manifest.json`。完整 baseline、GQA 数值、完整源码缓存、自审均通过；Chromium 16 项，skipped/flaky/unexpected 均为 0。自审见 M00-REVIEW.md。G01 原证据保持历史，不重写为当前版本成功。

已确认既有 Node/Chromium、CPU Torch 可用。R01/R02 资源尚未授权；不影响软件阶段。新增 `tools/program_gate.py` 保存真实子进程命令/日志/不可变源码清单，不安装依赖、不改 STATE、不代替语义审读。

## G02 · 开工范围与行为验收

范围：`content/fixtures/sft-data.json`、`experiments/sft_data_reference.py`、`web/sft/{compute.ts,SftDataJourney.tsx}`、`content/cases/05_sft_data.md`、`tests/{sft-data.mjs,test_sft_data.py,browser/sft-data.spec.ts}`；接入修改 `web/main.tsx`、`web/data.ts`、`web/style.css`、`package.json`。固定版本 collator 片段核实后扩充 `research/source-evidence.json`、`content/source-snippets.json` 及源码覆盖测试。本阶段报告、自审与 STATE 随验收更新。

行为验收：①保留原 messages，三种监督模式与 target 对齐；②EOS、截断、空回复、padding/零计数有显式结果；③两种长度样本 pack/unpack 的目标、mask 和参考 loss 相同；④重复 shift、跨样本 target/attention、均值之均值反例均检出；⑤真实 tokenizer trace 仅通过读取接口导入，保留来源/未验证边界；⑥浏览器可切换并查看逐 token 表、边界、源码，旧路线继续通过。

G02 已完成：报告 `runs/program-v1/G02/20261004T153109683149Z/report.json`（含实际源码清单、桌面/窄屏截图），79 Python、51 Chromium 通过，skipped/flaky/unexpected=0；33完整源码/59片段通过；96数值配置最大误差1.4210854715202004e-14。自审 G02-REVIEW.md。原聚合离线源码测试因覆盖增长超时，已逐源码文件参数化，全部原文/行号/注释断言保留；失败报告未覆盖。仅本阶段结果有效，后续公共修改另行回归。

## G03 · 开工范围与行为验收

范围：复用扩展 `experiments/gqa_reference.py`、`web/gqa/compute.ts` 的序列长度与可导数值后端；新增 `experiments/decoder_reference.py`、`content/fixtures/decoder-reference.json`、`web/decoder/{compute.ts,DecoderWalkthrough.tsx}`、`content/cases/06_decoder_update.md`、`tests/{decoder-numeric.mjs,test_decoder_reference.py,browser/decoder.spec.ts}`。接入修改 `web/main.tsx`、`web/data.ts`、`web/style.css`、`package.json`，补固定 Core final norm/FFN 源码与相关路由测试。只改当前项目。

验收：①两层 embedding→GQA/residual→SwiGLU/residual→final norm→head→masked CE；②TS复用GQA forward与CPU PyTorch共享GQA计算对照；③选定参数autograd与有限差分；④冻结参数不动、prompt激活仍可有梯度、共享head梯度汇集；⑤CPU优化器更新与同输入/RNG的save/resume下一步一致；⑥norm/FFN/head张量子步骤和源码可联动查看，旧G01/G02及全部回归保持通过。所有权重为明确authored/architecture-scaled，不下载模型、不执行GPU。

G03 已完成：`runs/program-v1/G03/20261004T155324142852Z/report.json`，当前阶段指纹 `4b59dfecb11f69c44dedfb1df49e7c567725177b268c05a52b7da8e979a07d86`。81项Python（含真实CPU子进程六项检查）、56项Chromium全部通过；GQA与decoder数值通过，后者26790值最大误差1.78e-15；35完整源码/65片段通过。报告保存本次CPU梯度/更新/恢复证据和截图。Final norm调用缺口已补，仍明确不是kernel。新增README等整体验收留G09。

## G04 · 开工范围与行为验收

同一G03参数与样本保持不变。新增 `experiments/tp_dp_reference.py`、`tests/{tp_dp_cpu_checks.py,test_tp_dp_reference.py,tp-dp.mjs,browser/parallel.spec.ts}`、`web/parallel/ParallelInspector.tsx`、`content/cases/07_tp_dp.md`。复用扩展 `experiments/decoder_reference.py` 与 GQA 运算图投影回调，仅提供in-memory TP切分；接入package/data生成、web状态/样式/入口，按需补固定TP/词表/梯度归约源码。

行为验收：①TP1/2在完整两层模型恢复输出及所有可训练参数梯度；②GQA组内QKV与成对gate/up分片、row partial sum、词表分片可查；③V27在TP2中用物理V28槽位承载，但教学CE明确排除额外槽位，不能把此约定冒充任意Core默认；④不同有效长度和梯度累积下DP全局sum/count与单模型参考一致；⑤漏reduce、错gate配对/gather、均值之均值等反例；⑥rank/group/shape联动与非法配置报错。collective仅CPU reference simulation，通信量注明字节口径，无NCCL或耗时实测。

## 继承记录

2026-10-04 读取 main `538afeb1745f69ec7c51d20d467b83a011d23752`，G01 报告 ready_for_review。继承的是代码与历史记录，不是已独立复跑的断言。M00 保持 pending。

G01 的范围到此保留为历史；本项目当前入口为 MASTER_GOAL。各阶段通过后自动继续，不以“等待用户审阅”作为每阶段停止点。最终的人工 review 标记仍保留为 false。

## 每次 checkpoint

记录阶段/子步骤、实际修改路径、当前 HEAD 与源码指纹、运行命令/结果/日志、解决的问题、剩余 blocker、下一阶段。只记录可复核过程摘要，不写完整私有推理、私有端点、密钥或原始大 tensor。

更新 STATE.json；validated 需报告/hash，blocked_external 需解除条件。终止时写清是 full_project_ready_for_review、software_ready_runtime_blocked 还是 blocked，不把会话停止当成功。

## 2026-10-05 · 提交前 checkpoint

用户授权 commit & push，保存当前阶段成果。M00/G02/G03 的历史验收记录保持不变；G04 当前仅完成完整模型 TP/DP CPU 参考、rank 映射和行为检查，课程与交互尚未完成，仍为 `in_progress`，不代表 PROGRAM-V1 全部完成。

在 `minimind-megatron-bridge-dev-1` 内执行 `tools/goal_gate.py baseline`：82 项 Python、56 项 Chromium、构建、handoff 和片段校验全部通过，无 skipped/flaky。报告 `runs/goal-g01/20261004T161114113098Z/report.json`，SHA256 `a01c5c116e487d70dec46a9758ce7592d691e9572238973dd61402dd7fffc742`。另通过 `npm run test:gqa`、`npm run test:sft-data`、`npm run test:decoder`、`npm run test:tp-dp`；完整源码缓存校验为 35 文件/65 片段。G03 当前 CPU 子测试为 7 项，G04 CPU 子测试为 6 项。全部为 reference 验证，无 GPU/NCCL 实测。

本次 Git 范围仅 `10_Projects/megatron-learning-lab` 的源文件、教学内容、测试与进度记录。运行报告/生成数据继续留在本地，不加入 Git；可在同一开发容器复跑上述命令。下一步仍为 G04 课程、rank inspector 和浏览器行为验收。
