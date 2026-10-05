---
type: experiment
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# G09 整合交付自审

## 审读范围与成功边界

对照 MASTER_GOAL、STAGES、VALIDATION、RESOURCES，审读当前 web/main.tsx、web/data.ts、CourseText.tsx、runtime/trace.ts、RuntimeLab.tsx、runtime/contracts.py、reference_trace.py；检查 G01–G08 数值入口和浏览器路线，再审读 release_gate.py、生产 Playwright 配置、release.spec.ts、README 与第 11 课。本次是代理自审，不替用户完成人工 review，reviewed 保持 false。

G09 验收软件、教材和运行接入合约。R01/R02 未运行，不能输出整个项目完成；最终只能在所有软件检查通过后给 software_ready_runtime_blocked。阶段历史报告用于接续和完整性核验，最终成功必须由本次同一源码快照的真实回归证明。

## 学习路线与既有阶段

默认仍为 Qwen3-0.6B 整模/Decoder；基础速览可键盘跳过；展开式核心章节使用 CourseDetails，独立符号表保留。顶层完整 HF 架构图与 authored 两层数值模型分别标注。Qwen2.5 的 bias/norm/head_dim、Qwen MoE 和 DeepSeek V2/V3 的路径差异未替换为通用 MLP 或同一标签。

现有数值入口保留以下断言，由最终 gate 重跑：

| 阶段 | 当前可核对的断言与证据入口 |
|---|---|
| M00/G01 | tests/gqa-numeric.mjs 的 Python/TS、causal boundary、head 合法性、omit_scale/wrong_group；gqa.spec 与源码路由 |
| G02 | tests/sft-data.mjs 的 3 种监督、target/mask、重复 shift、跨样本预测/attention、pack/unpack 与归约反例；sft-data.spec |
| G03 | decoder-numeric/decoder_cpu_checks 的完整两层 forward、CE、autograd/有限差分、冻结/共享梯度及 resume 下一步；decoder.spec |
| G04 | tp-dp/tp_dp_cpu_checks 的完整模型 TP1/2 输出/梯度重组、漏 reduce/错误 concat、DP sum/count；parallel.spec |
| G05 | sequence.mjs 的 PP1/2 与 microbatch1–8、SP、普通/THD CP、远端 KV/泄漏反例、布局与逻辑调度；sequence.spec |
| G06 | moe-mla/moe_mla_cpu_checks 的完整模型专家恢复、shared/重复 dispatch/路由权重反例、norm/RoPE 条件下 MLA 对照、已核验 EP/ETP；family.spec |
| G07 | rl-reference/rl_cpu_checks 的四类 LP、GRPO/PPO、value/GAE、detach/clip/KL 梯度、版本/refit 拒绝；rl.spec |
| G08 | runtime-contracts 的 CPU/配置/observer/启动/清理合约、Python/TS 同输入判定；runtime.spec 的合法/非法导入、来源、内层 config、导出 |

新增 release.spec 串联用户真实操作：样本与 last_turn → Decoder/更新恢复 → TP/DP/PP → DeepSeek MLA/MoE → PPO/value/refit → 当前源码和缺运行来源 → Run Compare。不只验证标题或文件存在。首次生产测试暴露测试操作与当前页面状态不符，已修正为先返回 Decoder、先展开默认折叠的符号表；没有扩大超时、删除断言或使用 skip。

## Run Compare 的定义与缺失数据

逐 token 比较要求相同任务、输入/mask/policy 版本 hash、模型/tokenizer revision 和模板。两侧 shape、dtype、来源与版本始终显示；不同任务不强行计算差。Loss 比较另外核对目标/归约/clip/KL/value 定义。选定梯度识别 authored、Bridge master、NeMo master 三种已知布局，要求有限数、相同坐标/角色/dtype 和更新前值；forward-only 缺梯度不会填零。

第 11 课新增独立符号与通信推导：记录的 head[27,8]+critic[8] 共 224 float64 元素，1792 B；复制式 DP ring 的 p=4 每 rank 发送和接收各为 2688 B。这个预算不包括 optimizer 分片、协议/拓扑/overlap/临时缓冲，也没有实际 NCCL 记录。UI 可变的是假设 p，原 trace 和实际运行版本不变；未知 shape 保持不可推导。数值手算、错误类型/坐标/尺寸和浏览器 rank 切换均保留测试。

上传始终 imported_claim，网页只读取 JSON，不执行路径/命令，不加载 pickle。Config 的结构/hash 不等于框架执行；selected scalar 不等于完整梯度/权重/恢复等价。observed schema 的字段合法也不能认证文件来源。真实未采集项不会变成模拟性能图。

## 生产预览与可访问性

独立配置 tests/playwright.release.config.ts 在 127.0.0.1:5174 启动当前 dist，reuseExistingServer=false；只运行 release.spec，不复用开发服务器。检查生产脚本来自 /assets/*.js 且不含 Vite dev 客户端，直接/嵌套 hash 路由重载保持模型状态。

第二轮生产 preflight 实际通过 3 条路径，skipped/flaky/unexpected=0；记录在 runs/program-v1/G09/preflight/production-second.log。浏览器为 Chromium 153.0.8010.12；观测到 KaTeX Math/Main/Size1/Size2/AMS 的本地 woff2 成功响应，外部请求为空。离线切换与公式展开没有 KaTeX/font 错误。Firefox/WebKit 没运行。

已实际查看生产手机与桌面截图：比较表和通信缺口可读，手机表格在局部横向滚动，焦点边框可见，页面整体不横向溢出。随后增加键盘滚动位置的明确断言，并将最终桌面截图定位到完整比较区；最终结果与截图由 release gate 再生成、审读。

## Gate 与交付完整性

release_gate 每轮新建目录，先核对历史依赖证据，再运行完整 baseline、8 个独立数值/合约入口、独立生产 smoke、完整固定源码逐字核验和此自审记录。Python 必须非零且无 skip；Chromium 必须非零、无 skipped/flaky/unexpected 并包含既有和 G02–G09 marker。控制测试证明缺/不完整 baseline 不接受、独立检查失败保留实际退出码并停止、旧输出不覆盖、runtime blocked 不变成 full completion。

源码清单包含实现/课程/测试/工具/有效计划/README/本自审，运行前后重新比较；依赖由与 CPU runner 相同的正常 Python 解释器查询已安装元数据和实际 sys.path，再结合 npm lock 记录，不导入 GPU 框架；配置单列文件 hash；构建树在 baseline build 后和生产 smoke 后必须相同。各命令真实 argv/退出码/日志 hash、基线子日志、独立检查子日志和生产截图拷贝归档，避免后续常规浏览器测试覆盖最终证据。STATE/PROGRESS/REPORT 属于结果账本，不制造源码指纹自引用。

README 现在给出容器环境、启动、生产预览、test:release、各阶段能力/边界、固定来源获取、真实运行授权入口与恢复策略。最终报告索引在 REPORT.md 与本轮 stage-index.json。源码参考缓存不是 runtime installation；版本元数据不是 GPU/ABI 兼容结论。当前单个前端主 chunk 约 1.65 MB（未 gzip）有构建体积警告；尚未做用户设备性能基准，不宣称性能达标。

## 未执行与解阻

本轮重新 dry-run 三个样例：均 not_ready/not_run；权重/tokenizer snapshot 缺失，RL 另缺 NeMo checkout/PPO critic 文件；runs/program-v1/resources.json 不存在。没有收到 GPU/SFT/RL 设备、步数、时限和输入的明确授权。本轮没有下载模型、初始化 GPU/Ray、执行训练、提交/推送或公开部署。

R01 仍要真实 tokenizer/一次 shift、同权重 HF/Bridge LP/激活、非零梯度/更新、save/resume、HF export 和 trace 导入；R02 仍要真实 actions/reward/mask、可导 LP、actor/critic 更新、两次同步迭代/refit 后固定输入训推对齐。当前 6 GiB 设备不能容纳既定全参数 Adam 推导的 7.77 GiB 权重/master/moments，不能默换成 LoRA 或小模型宣称验收。资源就绪后按授权先做 R01、再 R02；集成改变时 G09 revalidate。

依赖指纹审计发现 -S 下用 sysconfig 扫描会漏掉 /opt/venv，并误选系统同名包（Transformers 5.12.0、NumPy 2.1.0）。实际正常解释器优先虚拟环境，分别为 5.8.1、1.26.4，Core/Bridge/Ray 也能找到。已改成子进程元数据查询并保存实际 prefix/search path，确认 torch/ray/transformers/nemo_rl 均未被导入；新增独立解释器版本对照测试。首轮最终 gate 保留为诊断，源码变化将使其失效，最终成功以修正后整轮重跑为准。

修正依赖后整轮 177 Python/99 Chromium、8 专项及 3 生产 smoke 已执行通过，但外部账本核验发现正常 git diff --check 的零输出日志不满足 artifact 非空要求。保留原始空日志及 SHA256，用带真实 command/returncode/log hash/output_bytes=0 的命令回执作为非空 artifact；没有向日志伪造成功文本，也没有删除 diff 检查。新增真实静默子进程与日志篡改拒绝测试，并在 gate 返回成功前调用同一 report_issues 完整性检查。该轮不能更新 STATE，最终以修复后的整轮结果为准。
