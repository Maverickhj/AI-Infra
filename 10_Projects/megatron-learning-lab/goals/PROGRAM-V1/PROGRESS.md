---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# PROGRAM-V1 进度

当前：M00/G02/G03/G04/G05/G06/G07/G08/G09 validated；R01/R02 blocked_external，项目状态 software_ready_runtime_blocked。验收基点为 d060187；用户现已明确授权提交和推送 G09 交付改动，原生 Goal 未完成。所有代码/测试在 minimind-megatron-bridge-dev-1 执行。

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

G07 首轮新浏览器路径五项通过、一项失败：测试请求 force=true/KL=false，但 CPU --forward 清单尚未导出该组合，导致找不到参考行。已保留失败 trace 至 runs/program-v1/G07/preflight-missing-force-kl-off，并将真实 CPU 导出矩阵从12组扩展为完整16组，不移除 force/KL-off 行为断言。其余数值、GAE、独立 critic、参数导出、错误 refit 拒绝与实际内存复制、离线源码/数学及窄屏路径已通过；仍需完整阶段门槛。

G07 扩展后的16组54957数值最大误差7.993605777301127e-15、8项CPU检查与六条新浏览器路径全部通过。已实际查看桌面/窄屏截图，发现数字和 token 可被折行，补数字表格 nowrap 并保持局部横向滚动；增加对应样式断言，由完整门槛复验。49完整源码/111片段逐字检查通过。

## G07 · 完整阶段验收完成

报告 runs/program-v1/G07/20261004T183620097709Z/report.json，源码指纹 1f9669f279adec5ceeb38c9f5a57881e58b515720da876c406378dfe73ad1b52，报告 SHA256 26ed98a0a886b56a51ca8389b1a100c2239f846344dc1385d3cd90db1b8a316a。89项Python、88项Chromium通过，skipped/flaky/unexpected=0；构建、handoff、GQA回归、RL16组54957数值/8项CPU检查、49完整源码/111片段和自审全部通过。当前源码指纹及每条日志hash已复核；最终桌面/窄屏截图实际查看，数字保持单行、局部横向滚动，已复制到不可变报告目录并加入hash。R01/R02仍未执行，进入G08。

## G08 · 开工范围与行为验收

计划新增 experiments/runtime/ 下的 trace合约、来源探针、HF/Bridge/NeMo-RL薄适配器与采集器，tools/runtime_cli.py、profiles/runtime-plan.example.json、content/cases/11_runtime_contracts.md、content/fixtures/runtime-reference.json、web/runtime/{trace.ts,RuntimeLab.tsx}、tests/{test_runtime_contracts.py,runtime-contracts.mjs,browser/runtime.spec.ts}，接入现有状态/入口/package/源码档案。

验收：①只读dry-run不下载、不初始化GPU/训练/集群，按实际CLI签名生成命令并输出显式配置和未核验项；②保留HF/Bridge/RL不同运行来源，适配字段不改变shift/mask/归约/版本语义；③同版本语义错拒绝、异版本经显式适配和行为验证接受，未知映射拒绝，test double明确synthetic；④统一trace记录revision、input/config hash、backend/dtype/groups/layout/版本/采集边界，非法来源/缺mask/无效输入拒绝；⑤前端仅只读导入与比较，不执行JSON内命令，参考导入、错误反馈、离线/键盘/窄屏与旧路线通过。

G08只读盘点已保存 runs/program-v1/G08/research/environment-metadata.json：本地Bridge editable HEAD 2c173377b584acf1d92d6d1f09d9b150d1773139，包metadata0.5.1；Core0.18.2、Torch2.12.0a0、Transformers5.8.1等。NeMo RL未安装。Bridge仓库已有无关devcontainer/docs/AGENTS改动，本任务不修改。本地run_recipe.py是dataset推断mode的新CLI，Qwen3 recipe无hf_path参数且内部固定HF ID；不能照抄参考旧CLI或把--hf_path当离线保证。固定NeMo RL官方两个run脚本已缓存，尚待审读与适配，不声称其runtime兼容。


## 2026-10-05 · G07 与 G08 trace 合约提交 checkpoint

用户明确授权 commit & push，范围仅本项目源文件、课程、测试和进度记录。G07 保持已验证；G08 已有 Python/TypeScript 只读 trace 校验、实际 CPU reference 样例和 AST CLI 探针，仍为 in_progress。HF/Bridge/NeMo RL 薄适配器、采集器、完整配置与 dry-run、导入页面及完整 G08 阶段验收尚未完成；R01/R02 未运行，PROGRAM-V1 未全部完成。

当前快照在 minimind-megatron-bridge-dev-1 内复跑 tools/goal_gate.py baseline：102 项 Python、88 项 Chromium 全部通过，skipped/flaky/unexpected=0；数据生成、handoff、片段检查、生产构建和 diff 检查通过。报告 runs/goal-g01/20261004T190659876534Z/report.json，SHA256 c9a24a9c7bb3887ffa58c98f979850717457ced74e78d3f00e52bafdbded06b2。当前 PROGRAM-V1 源码指纹 181c3acb59f253b4946656c08f3fa2250a6523810a7131c563ed320fb3dbc826 已复核，基线日志 hash 全部匹配。

另通过 npm run test:rl-reference：16 组配置、54957 数值、8 项真实 CPU 检查，最大误差 7.993605777301127e-15，atol=rtol=1e-10。TypeScript trace 检查接受 2 份 CPU reference，拒绝 15 个损坏/不合法输入；Python 新增 13 项合约测试包含在上述 102 项中。完整源码缓存校验通过 49 文件/111 片段。补充日志与源码清单在 runs/commit-check-_soev9ou/。

这些检查证明当前参考数值和软件行为，不证明生产 runtime 兼容；导入元数据仍标 imported_claim，不作为真实运行证明。日志、截图、源码缓存、生成数据及临时凭据不加入 Git。下一步继续 G08 适配器、采集器、显式配置与只读导入交互。


## G08 · 薄适配、采集器与只读导入子 checkpoint

已新增 experiments/runtime/adapters.py 和 capture.py：显式 HF logits/Bridge token-loss 或 BSH/SBH logits/NeMo next-token 到 action 位置映射；普通 Bridge batch 追加真实末 target 的 context-only 槽，保留末位置监督，不重复 shift；未知 layout、缺 mask、packing/reset 和过滤样本拒绝。LossTap/BridgeForwardTap 委托原实现并返回原对象，ParameterSlice 观察实际 backward/SGD；不自写生产 trainer。采集器限定记录数、校验后才写文件、不覆盖既有记录，保存已加载 callable 的源码和实际配置值。正式运行入口及资源约束仍待接线。

新增 web/runtime/RuntimeLab.tsx、content/cases/11_runtime_contracts.md、tests/browser/runtime.spec.ts、tests/runtime-contracts.mjs、tests/runtime_adapter_cpu_checks.py、tests/test_runtime_adapters.py、tests/test_runtime_capture.py，接入原有导航/URL 状态和 package test:runtime-contracts。双槽导入保留来源、mask、loss、版本、源码/config/input hash；导出已校验数据，比较要求同一输入与身份。JSON 中的命令/HTML 仅为文本，校验状态防止旧异步结果覆盖新输入。

已通过 24 个 Python/TypeScript 交叉接受/拒绝用例、两份新执行 CPU reference 与现有样例逐字段对照、13 个 Python trace 合约和6个采集器测试。CPU 适配/观察器10项预检通过后，为 Ray 类 worker 传输补一个序列化反例，修正 LossTap 反序列化期间缺 delegate 时的递归 getattr；扩展后的11项已在 runs/program-v1/G08/runtime-contracts-serialization.log 通过；本次提交前又在 runtime-contracts-commit.log 复跑通过。runtime-contracts-preflight.log 保留前一10项版本。

四条新 Chromium 路径通过（42.0s），日志 runs/program-v1/G08/browser-import-label.log，实际查看桌面/手机截图 runs/program-v1/g08-{desktop,mobile}.png。首轮3通过/1失败因序列控件标签定位，trace保留在preflight-import；为控件补显式 accessible label 后重验，没有延长超时或删断言。测试含 SFT token 表/导出、PPO mask/版本、恶意文本不执行、非法输入清除旧结果、来源/离线数学/键盘/窄屏。共享完整回归与最终 G08 gate 尚未执行，STATE 继续 in_progress。

运行来源研究补读本地 Bridge AutoBridge.from_hf_config/from_hf_pretrained/to_megatron_provider/load_hf_weights/save_hf_pretrained、DatasetProvider、gpt_step、finetune、callbacks 与配置序列化。当前 Qwen3 recipe 固定 HF ID 且 load_weights=False，通用 runner 只允许已导出 recipe 名，不能把本地自定义 factory 路径塞进 --recipe。配置序列化实际来自当前 Core ConfigContainerBase.to_dict；Bridge/Core 与固定阅读档案仍分开。NeMo 两个固定 CLI 已完整读到 main/setup/train 分支；真正官方框架未安装或执行。继续实现明确本地权重来源的最小正式入口、effective config/dry-run 和资源绑定，再做整体验收。


## G08 · 提交前配置与回归收尾

新增 read_only_config.py 的解析检查只使用 Hydra/OmegaConf，不导入 Torch/Ray/NeMo launcher。六项配置测试覆盖继承、_override_、纯算术、命令行 override、环境/动态 resolver/路径越界拒绝，并与固定 NeMo 官方 helper 的 GRPO/PPO 配置逐字段一致。顶层 override 引用尚未合并的父变量会与当前官方 helper 一样失败；此边界有明确反例，未猜测结果。此前正例误用了该分支的失败日志保留在 runs/program-v1/G08/config-preflight.log。

固定 helper 与两份 YAML 原文及许可证保存至 tests/fixtures/nemo-config/，以固定 commit、Git blob、SHA256 核对；required tests 不再依赖被忽略的 runs 缓存。样例是配置解析测试输入，不是已授权的训练 profile。本次读取实际配置依赖为 Hydra 1.3.2、OmegaConf 2.3.1、PyYAML 6.0.3。

Bridge 配置探针走真实 AutoBridge.from_hf_config → to_megatron_provider(load_weights=False) → ConfigContainer.to_dict，在网络、CUDA、分布式初始化 guard 下通过。上游 optimizer.grad_norm_skip_threshold 的正无穷明确保存为 float_sentinel 标签；NaN 和非有限测量仍拒绝，并补独立测试。报告 runs/program-v1/G08/research/bridge-config-probe.json，effective config SHA256 c3689b3c505221032b93bb1656066bc2e30b29a67efae04be81933831b3fa37c；只证明配置构造/序列化，未创建模型或加载权重，不是训练兼容结论。

npm run test:runtime-contracts 已通过 24 项跨语言合约、两份新执行 CPU reference、11 项 CPU 适配检查、13 项 Python trace 合约、7 项采集器和6项配置测试。日志 runs/program-v1/G08/runtime-contracts-commit.log。首轮提交基线在111项 Python 中发现课程符号表标题不符合 handoff 的“数学符号”约定；修正标题与对应浏览器内容断言，保留失败报告 runs/goal-g01/20261004T195902593337Z/report.json，并重新运行完整基线。

原始源码文件归档方式的预检快照已通过完整基线：111项 Python、92项 Chromium，skipped/flaky/unexpected=0；数据生成、handoff、片段核验、生产构建和 diff 检查通过。报告 runs/goal-g01/20261004T200104491224Z/report.json，SHA256 3e03c74901b89c8108c404425ac23b30807fa8197b263376cb0aa3cadb8141bc；PROGRAM-V1 源码指纹 d1572551d19fab8811b77a1688fb955cb3b381989c30dbc95cf0c9af5dfa2e09。全部日志 hash 与当前源码指纹逐项核对；补充源码清单、49完整源码/111片段核验及截图归档位于 runs/commit-check-runtime/。

用户本轮明确授权 commit & push，仅提交本项目代码、课程、固定小型测试源码样例及进度；日志、截图、生成数据与临时认证文件不入 Git。G08 仍为 in_progress，完整正式入口、dry-run 资源绑定及最终阶段验收尚待完成。R01/R02 未执行，PROGRAM-V1 未全部完成。

固定上游 GRPO YAML 原文含尾空格；为保持原文/哈希且通过 Git whitespace 检查，三个固定源码文件以 manifest.json 中的 JSON 字符串逐字归档。测试先验证 SHA256/Git blob，再还原到临时目录调用官方 helper；不修改原文，不关闭 whitespace 检查。提交前完整基线随后再次复验。

最终提交快照（JSON 原文归档后）再次通过111项 Python、92项 Chromium，skipped/flaky/unexpected=0，全部基线检查通过；专项 runtime 合约也再次通过，日志 runs/program-v1/G08/runtime-contracts-commit-archive.log。最终报告 runs/goal-g01/20261004T200900295032Z/report.json，SHA256 6faf808b788721161729f30474aea342fd270dae8b1485baf288babbf9e8d4b0；源码指纹 4863cd7c69b9b36d5b356027c7760a3866fe2e1407592d32edc30ae1212eeecf，全部日志与源码 hash 复核通过。最终核验记录和截图位于 runs/commit-check-runtime-archive/；已实际查看最终桌面/手机截图，布局与局部表格滚动正常。G08 状态及真实运行边界保持不变。

## 2026-10-05 · G08 本地运行入口提交 checkpoint

用户本轮明确授权 commit & push，提交范围为本项目当前 HF/Bridge 入口、plan/tokenizer/进程约束、CLI、测试与小型配置样例，以及本段进度记录。没有启动真实 GPU、下载模型或创建 Ray 集群；G08 仍为 in_progress，NeMo 正式执行接线、完整 G08 验收及 R01/R02 尚未完成。

新增 experiments/runtime/{plan,token_data,launch,worker,hf_entry,bridge_dataset,bridge_entry,config_probe}.py、profiles/runtime-{plan,data}.example.json、tests/test_runtime_{plan,token_data,launch}.py、tests/runtime_bridge_config_checks.py 和完整公开模型 config 样例，接入 tools/runtime_cli.py 与 tests/runtime-contracts.mjs。dry-run 只检查指定元数据、文件存在性和实际源码签名；正式入口要求 plan SHA256 与明确资源授权、路径、GPU、步数和时限一致，worker 再次核对。进程管理保留实际退出码、限制时间与日志大小；子进程退出 0 但缺少完成结果也记为失败。NeMo run 仍明确拒绝，不把配置解析当作训练支持。

HF 入口调用本地完整 checkpoint 与真实 tokenizer；Qwen IM 映射独立核对模板 token IDs/offsets，支持 assistant/last_turn/full，拒绝未知格式和超长样本，不修改模板或截断。Bridge 通过官方 AutoBridge、DatasetProvider、finetune、checkpoint/export 委托实现；显式观测 HF 权重导入完成、训练 callback、选定 activation/gradient/参数片段。BF16 模型权重与 FP32 optimizer master 参数分开记录，CPU 反例验证 master 已更新但 BF16 舍入后可不变；adapter 修正 CPU batch 与设备端输出的 label/mask 对齐。这些真实入口代码尚未经过实际 GPU 训练验证，保存/恢复/导出数值等价性仍待 R01。

真实 Bridge 配置构造 CLI 已完成，报告 runs/program-v1/G08/constructed-bridge-config.json，配置 SHA256 283313578f339b8be0f5d236f0b3b45a561de0cb1e31caeb4d9b4bff3bf75400。阶段明确为 constructed_before_framework_finalize：保留全部 308 个 model 字段，以 JSON 字符串和 hash 归档；运行时最终配置将在 on_train_start 捕获。禁止网络、模型/权重与分布式初始化，拦截一次可选 FlashInfer CUDA 初始化请求，实际 CUDA/model/weights 均未初始化。第一次使用 AssertionError 的失败日志 bridge-entry-config-preflight.log 保留；改为可选模块会处理的 CUDA 不可用 RuntimeError 后四项检查通过，未安装 optional extras。配置嵌套字符串由 trace 外层 hash 绑定，尚未增加独立内层配置 schema/hash 校验，不宣称已验证这些内层语义。

提交前专项日志 runs/program-v1/G08/runtime-entry-commit.log 通过：24 项跨语言合约、2 份新执行 CPU reference、11 项 CPU 适配、13 项 trace 合约、7 项采集器、6 项 NeMo 配置、7 项 plan、4 项 tokenizer、5 项真实/模拟进程管理和4项 Bridge 配置/CPU 检查。完整源码缓存核验通过 49 文件/111 片段；实际 torchrun 参数解析接受所生成的单进程 argv，未执行 launcher。上述数字不代表 GPU 或 NeMo runtime 验证。

可在开发容器内复验：

```bash
env -u PYTHONPATH uv run --no-project python -S tools/runtime_cli.py dry-run --plan profiles/runtime-plan.example.json
npm run test:runtime-contracts
env -u PYTHONPATH uv run --no-project python -S tools/goal_gate.py baseline
```

样例中的输出路径需按实际计划填写，模型/tokenizer 缓存目前缺失，样例不含任何训练授权。HF profile 需将显式参数名改为 model.norm.weight；Bridge 使用 decoder.final_layernorm.weight。资源授权与实际运行需另行绑定完整计划。日志、截图、缓存、生成数据和临时 Git 认证文件不加入 Git。

提交快照完整基线通过 127 项 Python、92 项 Chromium，skipped/flaky/unexpected 均为 0；数据生成、handoff、片段检查、构建和 diff 检查通过。报告 runs/goal-g01/20261004T205810440656Z/report.json，SHA256 944246ddf7c8bde260a8ebb257f34e98d4522b948f637d47845e3250241f891c；PROGRAM-V1 源码指纹 ef44ea7a2f39ce26d39a2901a3e853dafeb65a33bf2f9f55a546013d3aa36805。已逐项核对当前源码和所有日志 hash，本地核验档案位于 runs/commit-check-runtime-entry/。此次提交不将 G08 或 PROGRAM-V1 标为完成。

## G08 · NeMo 配置与同步入口接线范围

上一轮完成源码提交和当前快照完整回归，属于有效进展。本轮重新 fetch 后 HEAD 与 origin/main 均为 a0290a2，工作区干净，planner 仍指向 G08。计划修改 experiments/runtime/{plan,launch,worker}.py、tools/runtime_cli.py 与现有专项入口/测试，新增 nemo_config.py、nemo_entry.py、对应配置/委托测试、小型 GRPO/PPO frozen config 与 JSONL 样例，并更新课程、源码档案和本进度。验收包括：配置 hash 变更拒绝；GPU/节点/步数/模型/数据/日志路径与授权绑定；PPO critic 独立授权；真实官方 setup/train 委托仅在授权后导入；同步 loss/refit 采集不改变返回对象；所有 synthetic 测试明确标记，未知分支继续拒绝。

## 2026-10-05 · G08 NeMo 配置与 CPU 观察器提交 checkpoint

用户明确授权 commit & push，本次仅提交本项目已完成的配置约束、入口委托边界、loss/参数观察器、CLI、测试与小型样例；不启动 GPU 或 Ray，不下载模型。G08 保持 in_progress，生产 NeMo observer、supervisor 接线、真实 rollout/refit 与最终阶段验收仍未完成。正式 launch 和直接 nemo_entry.run 都明确拒绝未完成的 RL 路径；不存在通过缺失模块导入间接失败的入口。

freeze-nemo 只读解析 YAML 后独占写入 frozen JSON，返回实际文件 SHA256；inspect-nemo-config 核对 exact bytes、路径、单卡/单节点、两轮样例、生成长度、batch、优化器和 loss 分支。PPO critic 使用独立身份及授权字段，policy/value tokenizer 均绑定本地同一 snapshot；不允许运行时 overrides、采样过滤、动态 batch、packing、量化和远程 logger。两份样例输出路径仍为占位符，未赋予资源授权；调整配置后须重新计算文件 hash 并重新绑定计划。

nemo_entry 的可注入委托边界执行固定官方 launcher 函数体，依赖显式 synthetic；检查原 setup/train 参数、配置变更拒绝、只创建本地 Ray context 的参数及异常恢复。nemo_capture 对固定官方 actor/critic loss 函数体执行真实 CPU forward/backward，接口为 synthetic，比较独立标量计算；用 2 token/1 token 的不等 mask 验证全局 token 归约，拒绝微批均值的平均值。worker mixin 委托实际 CPU SGD 测试 worker，核对返回对象、梯度、选定参数更新、失败后的 hook 清理和已有记录提前拒绝；这不证明实际 NeMo worker 或 GPU 兼容。

提交预检已通过 25 项 plan/授权/委托/进程检查、9 项 loss/worker CPU 检查、7 项配置解析/冻结检查，日志 runs/program-v1/G08/nemo-commit-{plan,capture,freeze}-preflight.log。新测试已接入常规发现和 test:runtime-contracts，固定 loss 原文块保留行号、SHA256、Git blob 与许可证。运行日志、截图、源码缓存与认证材料不加入 Git。

最终提交快照已在开发容器内完成完整基线：141 项 Python、92 项 Chromium 全部通过，skipped/flaky/unexpected=0；数据生成、handoff、源码片段、构建及 diff 检查通过。报告 runs/goal-g01/20261004T214617560389Z/report.json，SHA256 1f7aa0d2a9ea6e1bbc1f8de519eaaa6d1a2b64ecef89dd41ea9d4317bc474bda；PROGRAM-V1 源码指纹 1cd12ec17a6346eaa47473c90db5cb423a659069bc46263ef7cc432a84b21d1d。全部日志 hash 与当前源码快照一致，核验记录在 runs/commit-check-nemo-contracts/。专项 test:runtime-contracts 全部通过，完整源码缓存 49 文件/111 片段与新 loss 2 文件/7 原文块逐字核验通过。G08、R01/R02 的未完成状态不变。

## G08 · 官方 worker 扩展与生成事件接线

从 761e8f4 恢复，远端与本地一致、工作区干净，planner 无完整性问题并继续 G08。上一轮验证与提交属于有效进展。本轮范围：新增 experiments/runtime/nemo_extension.py、nemo_events.py、nemo_observer.py 及对应合约测试，按需修改 nemo_entry.py、nemo_worker.py、nemo_config.py、launch.py、worker.py 与专项测试入口，再更新 trace/refit 展示和课程。正式 GPU/RL 仍无资源授权，本轮只执行软件/CPU 验证。

行为验收：①官方 Policy 的 worker_extension_cls_fqn 在独立 Ray initializer 按名称加载，保留原 trainer；②指定当前 Python 环境，拒绝默认 uv 建环境或未知 worker；③记录真实生成输入/输出、奖励、mask 与 rollout/update/refit 次序，错误版本或错配 action 拒绝；④官方 refit 调用完成与权重 hash 验证分开；⑤失败恢复驱动 hooks/registry，输出有界且不覆盖；⑥正式入口仍须资源与源码/输入复查，合约测试始终标明 synthetic。

源码新增读到固定 worker_groups.py/ray_actor_environment_registry.py、experience/rollouts.py、models/megatron/setup.py。确认 driver patch 不会传入 IsolatedWorkerInitializer，因此采用官方扩展 FQN；NEMO_RL_PY_EXECUTABLES_SYSTEM=1 是当前上游公开分支，registry 仍逐项校验实际 sys.executable。nemo_extension 的 5 项 stdlib 合约已通过，日志 runs/program-v1/G08/nemo-extension-preflight.log。新增源码研究只说明静态范围，不证明 NeMo runtime 可运行。

事件账本与 worker 扩展共 12 项 stdlib 检查通过（nemo-events-preflight.log）。refit 的跨语言首轮发现 TypeScript 新分支误用了 Python helper 名 integer，导致合法 acknowledged trace 被拒绝；失败日志 nemo-refit-contract-preflight.log 保留，改用现有 int helper，未放宽版本/hash 断言，随后复验。

修正后的 runtime 合约通过 31 项跨语言用例及全部既有 CPU 检查。新浏览器用例首次 4 通过/1 失败：测试比较的是 stringify 上传前的 JS 对象，其中 -0 在上传 JSON 时已经编码为 0；导出保留了实际上传值。测试改为保存真实上传 JSON，并逐字段比较导出和该 JSON，未删除字段或放宽数值精度。首轮日志及失败 trace 保存于 runs/program-v1/G08/refit-display/，随后复验。

## 2026-10-05 · G08 同步观察器提交 checkpoint

用户明确授权 commit & push，本次提交本项目的官方 worker 扩展、HF 转换/加载调用观察、生成事件账本、NeMo 同步观察器及入口接线，连同 refit 证据展示、课程和行为测试。仅在 megatron-bridge 开发容器内执行 CPU/软件验证；没有启动 GPU、Ray 集群、下载模型或运行正式 RL。G08 保持 in_progress，不将本次提交视为完整阶段验收。

正式入口现在在框架导入前复查资源授权与 worker receipt，绑定当前 Python、离线环境和本次输出目录的转换缓存。观察器保留官方返回对象，核对 rollout/action/mask/reward/version 关联、实际 backward/参数更新及最终固定输入概率对齐。refit 调用成功只记录 acknowledged 和 weight_hash_verified=false；CPU 测试引擎与奖励明确标为 synthetic，不提升为 observed_rl。

提交检查补接已有 checkpoint loading 与 CPU observer 测试到标准回归入口，并修正采集器用例计数。专项 test:runtime-contracts 通过 31 项跨语言合约、2 份新执行 CPU reference、4 项加载调用检查、4 项观察器集成检查及全部既有用例；日志 runs/commit-check-nemo-observer/runtime-contracts.log。完整固定源码缓存核验通过 49 文件/111 原文片段。检查新增与改动 Python 文件的语法，并确认提交候选不含密钥或大文件。

后续 G08 仍需补足正式启动接口/元数据检查、RL supervisor 分支覆盖，以及强制超时下 Ray 子进程清理的验证，再进行完整阶段自审。内层配置字符串的独立 schema/hash 校验和 R01/R02 的实际 tokenizer、模型、GPU 训练及 refit 数值验证仍未完成；当前合约与软件回归不证明生产运行兼容性。

最终提交快照完整基线通过 160 项 Python、93 项 Chromium，skipped/flaky/unexpected 均为 0；数据生成、handoff、源码片段、构建与 diff 检查全部通过。报告 runs/goal-g01/20261004T225424088348Z/report.json，SHA256 62567bdce64cef18bba059838d6319c36f243b1aecb47ce241744f7321955928，源码指纹 08eaac3e38d5ffc667a3a759519f48da899af941cbbbbb2f8f92f42d164ae6f2。已逐项核对当前源码和日志 hash，核验档案位于 runs/commit-check-nemo-observer/。G08 和真实运行阶段状态不变。

## G08 · 启动前检查与独占进程回收

从 cacaa13 恢复，fetch 后与 origin/main 一致、工作区干净，原生 Goal active，planner 继续 G08，无完整性错误。上一轮完成 160 Python/93 Chromium 回归与用户要求的提交，属于有效进展；本轮恢复不提交/推送。范围：experiments/runtime/{plan,source_probe,nemo_metadata,launch,process_guard}.py、对应 tests/test_runtime_*.py 与 tests/runtime-contracts.mjs；按需补 trace 内层配置校验和课程/运行说明，再完成阶段自审与 gate。

行为验收：①正式框架导入前拒绝模型/tokenizer remote-code 映射与未知家族；②绑定实际 Policy/worker/setup/refit/rollout 等关键源码及接口，变更后的 receipt 不可复用；③RL 分支只接受精确授权、受控环境和真实完成状态；④超时、日志上限和 leader 提前退出后，清理本次调用拥有的 setsid/双重派生子进程，保留无关进程；⑤合约测试无 GPU/模型/Ray 初始化，元数据与模拟成功不作为真实 runtime 证明。

本轮启动检查 19 项通过（nemo-startup-preflight.log），随后追加 supervisor 异常终结 receipt 的用例。独立 Linux reaper 的 7 项真实 stdlib 子进程检查通过（process-guard-preflight.log），覆盖 setsid/双重派生/TERM 忽略/父进程退出、无关 sibling 和调用者 reaper 状态。合约专项通过 43 项跨语言判定、全部 CPU/配置/observer 检查（runtime-startup-config-preflight.log）；新增完整配置面板的 6 条运行页浏览器路径通过，当前完整基线 170 Python/94 Chromium 通过，待阶段 gate 全部结束后更新 STATE。

只读资源查询识别 GPU0 为 RTX 3060 Laptop、6144 MiB。按固定 Qwen3 config 推导 596049920 参数，现有 BF16 weights/FP32 master/Adam moments 为 14 bytes/parameter，合计约 7.77 GiB，尚不含梯度/激活/临时缓冲；这是 derived 下界项，不是显存实测。模型/tokenizer snapshot、NeMo checkout 和授权清单缺失；已保存 runs/program-v1/resource-proposal.json，并通过异步问题请求可用资源信息。未执行 GPU 计算，资源缺口不阻止 G09。

G08 gate 全部通过并核对当前源码、各日志和 artifact hash：runs/program-v1/G08/20261004T232220705471Z/report.json，SHA256 6b68a771300de3f429cf181a08bc3e750e12b1f756fe6863f9139af8744cf17f，源码指纹 4932bb5e59c5b1e6f2f5697f00bfcfe9b0184c03160102bf4fdfb7b14b7cf8e1。STATE 将 G08 标为 validated，R01/R02 标记具体资源/依赖阻塞，继续 G09；未将总 Goal 标完成。

## G09 · 整合与同一快照交付范围

依赖 G02–G08 已验证。范围：web/runtime/trace.ts、RuntimeLab.tsx、必要 experiments/runtime/{contracts,reference_trace}.py 与小型参考 fixture，tests/runtime-contracts.mjs、tests/browser/release.spec.ts、独立 production Playwright 配置、tools/release_gate.py、package.json、README.md 和阶段报告。行为验收：①同数据定义比较 shape/loss/选定梯度/版本并明确不匹配原因；②通信理论由显式张量尺寸计算，真实通信缺失保持未采集；③现有样本→整模→更新→并行→MLA/MoE→RL/源码路线可达；④生产预览验证直接/嵌套路由、离线数学/字体、窄屏、键盘及缺证据降级；⑤同一最终快照重跑全部数值/源码/浏览器/构建检查，生成报告索引/源码和依赖配置指纹；⑥真实运行仍阻塞时仅报告 software_ready_runtime_blocked。


## 2026-10-05 · G08 验收与 G09 比较界面提交 checkpoint

用户明确授权 commit & push。本次提交范围为当前 G08 启动前元数据/源码检查、专属子进程回收、内层配置结构/hash 校验，以及 G09 已实现的 shape/loss/选定梯度/版本比较和通信预算，连同 CPU 参考数据、课程、进度记录及测试。不包含尚未实现的 production smoke、release gate、最终 README/报告整合；G09 继续保持 in_progress。

53 项跨语言合约与全部 runtime CPU/配置检查已通过，日志 runs/program-v1/G09/comparison-contracts.log。新增浏览器行为检查覆盖相同输入的 loss/梯度/更新差、DP4 的 2688 B/rank 派生发送与接收、缺少尺寸或梯度、不兼容坐标及非法尺寸导入后清除旧结果。通信预算不代表 NCCL 实测，导入来源保持 imported_claim。真实 GPU/SFT/RL 仍未执行，资源阻塞不变。

提交快照完整基线通过 172 项 Python、96 项 Chromium，skipped/flaky/unexpected=0；数据生成、handoff、源码片段、构建与 diff 检查全部通过。报告 runs/goal-g01/20261004T234439348128Z/report.json，SHA256 e9ca38243d43feab8cba6b97d916a4817b9b44d7c1a7fcb38cb89e9cc3ad8dc2，PROGRAM 源码指纹 02fdc3ffa4d13a36244a7bd70ddcdd53ba0fe137f828f2087f9de0a7f53808d4。当前源码与全部基线日志 hash 已复核，核验档案在 runs/commit-check-runtime-diagnostics/；已查看比较界面的手机截图，表格保持局部滚动。该提交不代表 G09 完整发布验收或真实训练完成。


## G09 · 生产预览与最终验收执行

从 d060187 恢复并 fetch，HEAD 与 origin/main 一致、工作区原本干净，planner 继续 G09；上一轮提交和 172 Python/96 Chromium 验证属于有效进展。新增范围为 tests/browser/release.spec.ts、tests/playwright.release.config.ts、tools/release_gate.py 及验收工具测试；更新 package.json、README.md、第 11 课符号/通信推导、G09-REVIEW.md、REPORT.md 和本进度。独立生产预览在 127.0.0.1:5174，拒绝复用现有服务；不公开部署。最终 gate 绑定实现、课程、测试、README/自审、依赖与配置，生产截图归档至本次不可变报告目录。R01/R02 资源和模型缺口不变，继续软件验收。

生产首轮 smoke 为 1 通过/2 失败：集成测试在 Update/resume 子步骤后直接找 Decoder 层按钮，另一条对默认折叠符号表内公式断言可见。修正测试为先返回 Decoder、先展开符号表，保留原断言与超时；失败 JSON/截图/trace 已归档到 G09/preflight/production.log 和 first-failure-artifacts/。窄屏与缺失证据用例已通过；尚未宣布生产 gate 成功。

G09 首轮最终回归过程中发现依赖元数据采集错误：-S 的 sysconfig 路径会跳过实际虚拟环境并读到被遮蔽的系统包。已用正常 CPU 解释器的 metadata.version 与实际 prefix/sys.path 修正，并增加回归测试；仅查询元数据，确认未导入训练框架。首轮报告 20261005T000716382044Z 保留诊断且不能作为最终证据，等待其当前子进程终结后再启动修正版本的完整 gate。

G09 第二轮 20261005T001441019715Z 全部执行检查通过，但 report_issues 拒绝空 diff.log artifact。该轮保留为“执行成功、账本完整性未通过”的诊断，不更新 STATE。修正为记录真实零输出命令回执，仍保留原始日志/hash，并新增真实子进程回执测试和 gate 内最终完整性检查。源码改变后重新执行同一最终快照的全套验收。

## G09 · 最终同一快照验收完成

修正后的 gate runs/program-v1/G09/20261005T002421065004Z/report.json 已通过全部实际检查与内置报告完整性验证，外部 audit_current_release 再次核对当前 source/dependencies/config/dist 和所有日志/artifact hash。报告 SHA256 6a8d85f74e1bf9e935d382c5292558ee548e3ada344f250c1aeca9997bf5c3d2；源码指纹 2b71778880f36b2ecae8e8bfc4b963f191215ca14079eac0ad339d3179812ad3。实际 178 Python、99 Chromium、3 独立生产 smoke，skipped/flaky/unexpected=0；8 专项全部通过，固定源码 49 文件/111 原文片段通过。已查看本轮不可变归档的手机和桌面截图；具体索引、指纹、数值范围、启动与解阻条件见 REPORT.md。

STATE 将 G09 标为 validated，项目状态为 software_ready_runtime_blocked，R01/R02 保持 blocked_external。软件阶段已无独立剩余工作；真实模型/tokenizer、NeMo 与明确资源授权仍缺失。本轮是有效进展，并在软件完成后首次到达仅剩外部运行资源的停点；没有将原生 Goal 标完成。本轮没有提交/推送、公开部署、模型下载或 GPU/Ray 训练。
