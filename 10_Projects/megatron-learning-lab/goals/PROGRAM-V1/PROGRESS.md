---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# PROGRAM-V1 进度

当前：M00/G02/G03/G04/G05/G06 validated，G07 in_progress。2026-10-05 用户再次授权 commit & push，提交前 HEAD `7cff80b`；本次保存 G06 成果和 G07 数值核心，G07 课程与页面尚未完成。所有代码/测试在 `minimind-megatron-bridge-dev-1` 执行。

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

## G05 · 后续实现范围与行为验收

G04完整门槛正在稳定快照运行；此处只记录下阶段范围。计划新增 `experiments/sequence_reference.py`、`web/sequence/{compute.ts,SequenceJourney.tsx}`、`content/cases/08_pipeline_sequence.md`、`tests/{test_sequence_reference.py,sequence.mjs,browser/sequence.spec.ts}`，接入既有状态/入口/package/source archive。使用G03真实两层分配与G02 authored样本/packed metadata，不引入新平台。

验收：①PP1/2、microbatch1–8的non-interleaved 1F1B，绑定两层和embedding/head归属，所有前后向及跨stage依赖满足；②activation生存期、逻辑bubble和手算时间轴，不推断GPU利用率；③SP沿TP组sequence维切分、all-gather与row reduce-scatter，对照同模型dense FFN；④CP按普通全序列zigzag或THD每文档zigzag分片，保留有效/物理cu_seqlens并检查可分性；⑤远端KV分块softmax合并等于完整block-causal attention，丢远端KV/跨样本泄漏/非法padding反例报错；⑥页面切PP/microbatch/layout/CP/rank/query同步更新，源码与手机布局可用。

G04 已完成：报告 `runs/program-v1/G04/20261004T164510932004Z/report.json`，指纹 `ca9586f5680a96ba5696bd67a9f97e09fd99e96400bd3904089c9895a03e60d0`。82项Python、63项Chromium全部通过，无skipped/flaky；test:gqa、test:tp-dp、38完整源码/70片段及自审通过。报告归档本次CPU切片与桌面/手机截图。课程静态解析复用修复了旧GQA路径性能回退，未扩大测试时限；全部失败报告保留。

G05 开始实施。容器中的临时草稿已通过5项独立Python检查和25组跨语言/CPU数值对照（5448值，最大误差4.44e-16），这些只是草稿预检，不能替代写入后完整阶段验收。

G05首轮四条新浏览器路径全部通过；集成后旧GQA源码长流程超30秒，失败trace已保存至runs/program-v1/G05/preflight-gqa-timeout。正文memo消除了重复解析，但未展开整章仍产生大量公式DOM；进一步将五处课程折叠正文按展开状态挂载，完整内容、原测试与30秒时限保留。此公共渲染改动需重跑所有课程路径。


## 2026-10-05 · G04/G05 提交 checkpoint

G05 完整门槛通过，报告 `runs/program-v1/G05/20261004T170057490328Z/report.json`，源码指纹 `f2aff69966f4b035a6e7bbcbbc9f9dce9aff63264dd4030f85fef0e5cabe06fe`。87 项 Python、70 项 Chromium、构建、handoff、GQA 数值、sequence 数值和固定源码校验全部通过，skipped/flaky/unexpected=0。sequence 对照 25 组、5448 数值，最大误差 4.440892098500626e-16；41 完整源码/78 片段逐字验证通过。桌面/窄屏截图已查看并归档到报告目录，STATE 保存报告哈希。

五处课程正文按展开状态挂载，保留所有精讲内容、数学符号表和键盘操作；完整浏览器回归已覆盖这次共享渲染改动。G04 的历史报告保持原快照，当前快照再次覆盖 G04 四条浏览器路径及 CPU 数值检查。全部仍为 reference 验证，没有 GPU/NCCL 或真实训练实测。

本次用户明确授权 commit & push，范围仅本项目源文件、课程、测试和进度记录；运行日志、截图及生成数据继续留在本地，不加入 Git。提交前再次校验当前源码指纹与报告一致。下一阶段为 G06 的 MoE/MLA 完整模型交互与独立数值验收，PROGRAM-V1 尚未全部完成。


## G06 · 开工范围与行为验收

复用 `experiments/decoder_reference.py` / `web/decoder/compute.ts` 公共两层 backbone，提取 attention/FFN 扩展点并立即回归 G03/G04/G05。新增 `experiments/moe_mla_reference.py`、`content/fixtures/moe-mla-reference.json`、`web/family/{compute.ts,FamilyWalkthrough.tsx}`、`content/cases/09_moe_mla.md`、`tests/{moe_mla_cpu_checks.py,test_moe_mla_reference.py,moe-mla.mjs,browser/family.spec.ts}`；接入状态、入口、样式、package 和固定来源档案。

行为验收：①三种模型复用 embedding→两层 attention/FFN residual→final norm→head→CE，Qwen MoE无shared、V2直接Q、V3低秩Q；②top-k、分组/归一化/缩放、permutation→expert→combine恢复token，padding与辅助项明确；③V3 bias计数/符号更新符合固定源码，不作为SGD参数；④MLA expanded与normalized-latent吸收式及逐prefix cache一致，错误norm/scale/RoPE反例有效；⑤EP1/2、ETP1/2、EDP1只在参考已验证组合中展示；⑥模型/层/token/专家/训练或decode选择实际改变数据，CPU对照、源码、导出、键盘/离线/窄屏与旧路线通过。

G06 首轮八项 CPU 检查中七项通过；bias 手算发现 float64 参考中误将计数均值降为 float32，引入 2.235e-10 偏差。改为保留 logits dtype 后重验，预设 1e-10 容差不变；生产 router 的 fp32 边界仍明确保留。


## G06 · 完整阶段验收完成

报告 runs/program-v1/G06/20261004T174656964753Z/report.json，源码指纹 c859a265881f1e27f0f6570f01301545c123d1d7519e14ebaf1d9e9788d391ba。88项Python、79项Chromium全部通过，skipped/flaky/unexpected=0；构建、handoff、GQA、MoE/MLA数值、46完整源码/98片段、自审通过。MoE/MLA 9组与4种专家布局共72033数值，最大误差2.886579864025407e-15，含八项实际CPU梯度/反例检查。报告归档当前桌面/窄屏截图与源码清单，STATE保存报告hash。此前失败trace保持在preflight-group-expectation。

G03/G04/G05公共decoder扩展点已立即通过各独立数值检查，本轮完整基线再覆盖旧课程。新增子步骤源码显式选择不改变旧默认路由。训练激活与decode cache独立导出，参考数据未标observed。进入G07 GRPO/PPO，真实训练仍未获资源授权，不能以CPU参考代替R01/R02。


## G07 · 开工范围与行为验收

新增 `content/fixtures/rl-reference.json`、`experiments/rl_reference.py`、`web/rl/{compute.ts,RLJourney.tsx}`、`content/cases/10_rl_reference.md`、`tests/{rl_cpu_checks.py,test_rl_reference.py,rl-reference.mjs,browser/rl.spec.ts}`，接入已有RL场景、状态/来源/package；复用G03完整两层decoder的固定特征。CPU参考仅训练明确列出的LM head和独立critic head，backbone冻结以便独立推导全部策略参数梯度，不把它称为全参数训练或实际rollout。

验收：①四条固定authored轨迹、两组prompt与不同响应长度，generation/prev/current/reference按action位置对齐；②GRPO组内Bessel标准差/同分组与PPO terminal reward、GAE跨mask carry、returns、value loss两条完整路线；③正负advantage clipping、token/sequence归约、KL开关及其采样权重梯度、force-on-policy detach；④CPU autograd与独立解析/手算/有限差分，真实SGD actor/critic互不串梯度并改变参数；⑤导出/refit以版本+参数hash+ack校验，错版本/未完成拒绝同步，固定actions不冒充新生成；⑥交互数值、步骤源码、参考导出、离线数学、键盘与窄屏及全旧回归通过。

已读取固定R-LOSS/R-GRPO以及新缓存R-ADV/R-UTIL/R-PPO；GitHub tree API限流一次后改读固定commit raw原文成功，不重试API、不改source.lock。源码实际std带Bessel修正，force ratio和KL sampling weight需保留detach梯度；GAE保持跨masked位置的累积状态。

## 2026-10-05 · G06 与 G07 数值核心提交 checkpoint

用户明确授权 commit & push，本次范围仅本项目源文件、课程、测试与进度记录。G06 的历史阶段验收保持不变；G07 当前完成 authored 轨迹、冻结整模 backbone 的可训练策略/critic head、独立数值参考与行为检查，课程、页面及完整阶段验收尚未完成，STATE 保持 in_progress。

当前快照在开发容器内执行 tools/goal_gate.py baseline 全部通过：89 项 Python、79 项 Chromium，skipped/flaky/unexpected=0；数据生成、handoff、片段检查、生产构建和 diff 检查通过。报告 runs/goal-g01/20261004T181156949024Z/report.json，SHA256 fedb8f4de67193d78062169d80983d98a797d7c2fc394bb2e82d0893f7a80c6f；源码指纹 a41440d4d8d417609aff794586fa755f723c6ccdab02f69efe012970c4953089。

npm run test:rl-reference 通过 12 组配置、41221 数值、8 项真实 CPU 检查，最大误差 7.993605777301127e-15，atol=rtol=1e-10。日志 runs/commit-check-hMEIzm65/rl-reference.log。完整源码缓存校验通过 46 文件/98 片段。所有结果属于 reference/software 验证，没有 GPU 或真实 rollout；PROGRAM-V1 尚未全部完成。

运行报告、截图、源码缓存和生成数据留在本地，不加入 Git。下一步继续 G07 的课程、交互、固定源码档案与完整浏览器验收。
