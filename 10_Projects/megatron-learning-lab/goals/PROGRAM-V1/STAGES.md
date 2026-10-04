---
type: project
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# PROGRAM-V1 分阶段交付与验收

以下均为后续实施要求，不声称新功能已经存在。公共要求见 MASTER_GOAL/VALIDATION。新增 npm 测试入口必须真实执行测试，不能用 echo、零用例、全部 skip 或读取旧 JSON 代替。交付路径可在阶段开工时细化，但不得删减行为要求。

## M00：接续现有实现

保留 `538afeb` 中的 GQA 13 个子步骤、Python/TS 参考、29 个源码证据入口及其新增课程。以上数量仅来自提交记录，不作为固定计数断言。检查 G01 的实现与报告引用，获取当前 Git 状态、运行环境及可访问缓存。

执行现有 `python tools/goal_gate.py baseline`、`npm run test:gqa` 和当前完整源码缓存校验（缓存可用时）。基线没有涵盖的必要源码核验不得写为通过；缺缓存只阻塞需要该缓存的核验，新数据/课程可先实现但不能虚假验收。继承报告在被忽略的 runs/ 中丢失时，重跑，不伪造。

验收：公式、语义源码路由、head 合法性无回退；GQA 数值/浏览器仍通过；明确本轮可用能力；阶段账本能接续而不是重新执行 P0。M00 不重新设计平台、不把准备工作扩展成多天环境迁移。

## G02：SFT 样本与监督

页面：在现有 Sample Journey 加 target 对齐、label shift、loss mask、padding、packing 交互，并保留原始 messages。一个输入串与两个长短不同样本贯穿全部操作。

数值与语义：assistant/last_turn/full；只 shift 一次；mask 跟随 target；EOS、截断、空 response；全局有效 token 数；pack 后 block-causal 边界、position/cumulative lengths 与 unpack 对照。明确 packed microbatch size 不是样本数。

没有真实 tokenizer 时使用命名为 authored token fixture 的可见 token 序列，不给它冠以 Qwen token IDs 或 chat template 实测；G02 的必须项是数据机制可演算。提供真实 tokenizer trace 的读取接口，实际官方 tokenizer/template 结果必须在 R01 补验。

源码：沿当前版本 dataset/collator → gpt_step → loss，核对实际 shift 所在位置，不从 HF 常见行为推断 Bridge。已有片段不足时补固定源码或明确调用边界。

验收：手工可核对的输入/target/mask 表；重复 shift、跨样本预测和跨样本 attention 泄漏均被反例测试抓住；packed/unpacked 有效目标及指定参考 loss 一致。浏览器可切模式、查看 mask 对齐与 packing 边界。测试入口 `test:sft-data`；浏览器标题包含 `[G02]`。

## G03：完整 decoder 到一次 SFT 更新

复用 GQA 计算，不另写第二套 attention。构造明确标记 architecture-scaled/random-or-authored 的微型 Qwen-style 两层模型：embedding → 两层 attention/residual/SwiGLU/residual → final norm → tied/untied head → masked CE → backward → optimizer。

CPU 参考优先复用已安装 PyTorch；不存在时按最小依赖申请或用可验证的微型解析梯度参考，不能将需要梯度的功能降级为 forward 数值。前端只展示选定 token/参数切片，不复制完整训练框架。

必须把 norm/FFN/head 从“说明卡片”变成有明确张量输入输出的子步骤，补 Final norm 真实源码调用/实现入口。区别逻辑权重方向与 [out,in] 存储，以及参数共享的梯度汇集。

验收：CE 与目标 token logprob 一致；选定参数解析梯度与有限差分对照；真实 CPU backward/update 改变预期参数，冻结参数保持不变；loss mask 为零不等于 prompt 激活无梯度；同一输入/RNG 下连续训练与 save/resume 的下一步一致。非实测 GPU 曲线仍不存在。测试入口 `test:decoder`；浏览器 `[G03]`。

## G04：同一模型中的 TP / DP

在 G03 的完整模型/样本上加 rank inspector，不另建孤立 Linear 网站。展示全局 tensor 与 rank-local 权重/激活/部分和；group 成员与 collective 的输入、输出、前反向含义可查。

TP：QKV/GQA、成对列/行 FFN、输出与词表分片，哪些中间结果不需 gather；至少 TP1/TP2 合法微型配置恢复输出和梯度。不得用 dense world-size 公式机械乘 EP。DP：不同有效长度的 batch，梯度累积、loss sum/count 与全局归约。

CPU 内存内切分/拼接必须标 reference simulation，不能标 dist/NCCL 实测；多进程 Gloo 可另设扩展但不成为普通浏览器的前置要求。通信量是明确计量口径的理论值，不能伪造耗时。

验收：重组权重/输出/梯度对齐单卡 CPU 参考；漏 reduce、错误 gather/concat、局部均值之均值等反例失败；rank/shape/group 联动正确；非法维度组合有错误信息。测试入口 `test:tp-dp`；浏览器 `[G04]`。

## G05：PP / SP / CP 与长序列语义

PP 时间轴绑定真实层分配和 microbatch ID。至少实现一种明确假设的 non-interleaved 1F1B 教学调度，展示依赖、activation 生命周期和 bubble；用逻辑单位时间，不标真实 GPU 利用率。没有实现的 VPP/overlap 不能靠调参宣称支持。

SP 展示 TP group 内 sequence 分片、gather/reduce-scatter 边界；CP 展示全上下文依赖与 KV 通信，说明简单局部 causal attention 为何不等价。接上 G02 packed metadata，区别普通序列与 THD 路径，不能只按 S/CP 画同样的条。

验收：调度无违反 forward/backward/跨 stage 依赖、无重复或缺失 microbatch；改变 PP/microbatch 更新逻辑时间轴；CP 参考重组与未切分 attention 一致，跨样本泄漏/丢远端 KV 反例失败；SP/CP 的 tensor layout 不混淆。测试入口 `test:sequence`；浏览器 `[G05]`。

## G06：MoE / MLA 的核心精讲

模型页复用公共 decoder 骨架，Qwen3 MoE 精讲 router → top-k → permute/dispatch → expert → combine；DeepSeek-V2-Lite/V3 精讲低秩 Q/KV、latent norm、分离 RoPE、shared experts 与 dense/MoE 层分布。V2-Lite 的直接 Q 路径不可替成 V3，R1-Distill-Qwen 不可替代 MLA。

参考 fixture 分开验证 routed/shared 输出、路由缩放/归一化、padding 排除及当前辅助项；路由 bias 训练更新机制以固定源码为准，不能简化成另一算法。MLA 训练展开式与数学可等价的 latent 读取式用小张量对照，norm/位置条件必须显式。

验收：专家计算结果按 token 恢复；重复派发、漏 shared、权重归一化错误被测试捕获；MLA 对照在声明假设下成立；训练激活与 decode cache 在 UI/trace 中分开；EP/ETP/EDP group 只显示已核验组合。测试入口 `test:moe-mla`；浏览器 `[G06]`。

## G07：RL 更新的数值与系统语义

同一组 authored trajectories 和可导 CPU policy 贯穿 reward → group advantage → 四类 logprob → ratio/clip/KL → backward/update → 版本/refit 合约。明确这是教学参考，不是真实模型生成；固定 sampled actions 不许伪装真实 rollout。

至少包括 GRPO 和 PPO 两条完整路线。PPO 增加 value prediction、returns/GAE、value loss 及独立 actor/critic 梯度，不只是换标签。OPD、异步和 partial rollout 留作后续扩展，不阻止本版交付。

验收：generation/prev/current/reference 含义及位置对齐；同组同分、正负 advantage clipping、token-vs-sequence 归约、reference KL 关闭/开启、force-on-policy detach 梯度；手算与独立 CPU loss/gradient 对照；不匹配 policy 版本/refit 未完成必须拒绝“已同步”。真实 rollout 和跨引擎 refit 留 R02。测试入口 `test:rl-reference`；浏览器 `[G07]`。

## G08：最小训练接入与 trace 合约

建立当前所需的 HF/Bridge-SFT/NeMo-RL 接入路径、显式 effective config、模块来源检查、可读 dry-run 与采集器；实现版本字段差异的薄适配层，不自写生产 PPO 或 Megatron trainer。

统一 trace schema 连接 Run Compare，记录模型/权重/tokenizer/config/input/revision、backend/dtype/group/layout、sample/trajectory/policy版本、来源类别与采集边界。前端只读导出数据，绝不执行上传 JSON 中的 shell/Python。

分层探针：接口存在/签名 → 数据语义 → 小规模行为 → runtime source mapping。使用 test double 的合约测试明确标 synthetic；这可验收接入软件，但不能声明真实 Bridge/NeMo RL 已兼容。必要的未知上游分支保持不支持或未核验，不能假造。

验收：真实命令在当前 CLI 合约下生成；dry-run 不下载权重、不初始化训练、不发起集群；错误 provenance、缺 mask、未知 mapping/版本、无效 trace 拒绝；同版本但语义错拒绝，异版本经验证适配可接受。测试入口 `test:runtime-contracts`；浏览器 `[G08]` 验证合法参考 trace 与非法 trace 导入。

## R01：真实 HF / Bridge SFT 验证

只有资源授权与模型/数据具备时执行。优先 Qwen3-0.6B，Qwen2.5 小模型对照按授权资源运行；完整 tokenizer/config/checkpoint revision 和真实 chat template 不得缺。基线短序列、并行1、BF16；资源清单指定步数、时限、设备及输出目录。

验收：不是 load_weights=False 的随机模型冒充 SFT；同输入/权重的 HF 与 Bridge selected-token logprob/激活误差，容差运行前声明；真实 assistant/last_turn/full mask 与单次 shift；非零梯度与预期参数更新；save/resume、HF export 各自验收；trace 导入当前网页。LoRA/packing/多卡后续可增，不应挤掉此最小闭环。

检查命令由 G08 生成并经 resource manifest 绑定，记录为 `runtime-sft`。没有 GPU、权限、权重或兼容环境时，记录 blocked_external；G09 仍须继续。

## R02：真实同步 RL 验证

使用 R01 已确认的模型/分布式训练接口，独立核验实际 NeMo RL 与 rollout backend，不强制复制参考依赖。首例同步、短回复、固定采样配置、两次迭代；优先复用已核验正式 recipe，SGLang 对照按实际能力启用。

验收：模型实际生成的 actions；reward/group/masks；更新前重算及当前可导 logprob；GRPO/PPO 按所选正式实现分支；policy 确实更新；export/refit 后 generation 使用新版本并固定输入复验。PPO 的 critic 资源须单独具备。这里两次迭代只证明闭环，不证明能力提高。

检查记录 `runtime-rl`。正式 recipe 未确认的资源/外部服务/成本不自动扩大；协议只有 test double 通过时仍为 blocked 或未执行，不能写 observed_rl。

## G09：整合发布与全部回归

默认入口仍是完整模型/任务，新增子课程可按需展开；基础速览可跳过。Run Compare 能区分 reference/derived/observed，比较明确来自同一数据定义的 shape、loss/grad、版本、通信理论值与实测，不显示虚构指标。

在最终当前源码快照重跑：全部 Python、test:gqa 及 G02–G08 的独立检查、完整 Chromium、typecheck/build、源码片段/路由、trace schema、git diff --check。生产预览单独 smoke，包括嵌套/直接路由、无 CDN 的公式/字体、窄屏、键盘、证据缺失降级。Firefox/WebKit 未执行须明确，不冒充已覆盖。

产出项目运行 README、各阶段报告索引、最终源码/依赖/配置指纹、状态汇总、残余限制。真实运行缺口单列，不能因为 CPU 版完整就将 R01/R02 标完成。默认不部署、不 commit/push。R01/R02 后续恢复并改变集成代码时，G09 重新标 revalidate 再出最终报告。

验收：一个用户能从样本走到完整模型/更新、比较并行策略与 MLA/MoE、理解 SFT/RL 目标差异，并沿当前语义定位源码；所有成功状态均有当前实际检查，所有未运行项都有边界。测试入口 `test:release`；浏览器 `[G09]`。
