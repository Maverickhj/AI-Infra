---
type: experiment
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# G08 接入软件自审

## 范围与验收结论边界

审读 experiments/runtime 的 plan、source_probe、read_only_config、nemo_config、capture、adapters、HF/Bridge/NeMo 入口、worker 扩展、loading/events/observer、launch/process_guard；对照 tests/runtime*、tests/test_runtime*、web/runtime 与第 11 课。这里验收软件与合约，R01/R02 的实际权重、tokenizer、CUDA/分布式 ABI、训练和引擎行为仍未运行。正式代码存在、AST 签名相符和 synthetic 测试均不能给实际 backend 标 supported。

HF 调用完整本地 checkpoint 的官方 tokenizer/config/model；Bridge 使用 AutoBridge/finetune、真实 HF 导入 hook、callback、checkpoint 与 HF export；NeMo 委托固定已审读 launcher/setup/train 和官方 Policy worker extension FQN，不另写生产 trainer。此前“生产路径未接线”的历史进度保留；当前 supervisor 已分发三个 backend，正式入口受精确 plan/resource/receipt 限制。

## 配置、来源与命令

plan 明确 immutable revision、输入 SHA256、路径、设备、步数、时限、dtype、capture slice 与预声明容差。dry-run/CLI parser 只读指定路径，不导入 launcher、加载权重或初始化集群。冻结 NeMo JSON 的精确 bytes/hash 绑定资源计划；PPO critic 身份和权限独立。未知 packing、异步、loss 分支、采样过滤、外部 logger 和 runtime overrides 拒绝。

RL 启动前拒绝未知模型/tokenizer、remote-code 映射、另一个 pretrained checkpoint、替换 chat template、共享旧转换缓存。静态检查 Policy、worker、setup/train、rollout/refit、tokenizer 与环境 registry 的必要接口，并对本地 nemo_rl Python 文件形成变更指纹；这不是审阅所有模块或证明其依赖已安装。worker 执行前重新检查输入/源码，与 receipt 不同则拒绝。固定参考缓存上的 GRPO/PPO 各 16 个接口位置已实际检查；报告 nemo-source-interface-probe.json 的缓存是研究子集，绝不称为可运行 NeMo 安装。

运行时实际 callable 路径和文件 hash 与参考 source.lock 分开，partial chat template 定位到真实底层函数。模型/critic/tokenizer bytes 单独记录。config 归档保留完整实际字段，不将手写子集称为完整配置；Bridge 308 个 model 字段实测构造通过。内层配置独立校验 JSON object、重复/危险 key、有限数、深度/大小、float sentinel 与 SHA256；仅该内层每对象允许 512 字段，trace 外层 256 字段不变。

本轮样例 dry-run 明确返回 not_ready/not_authorized/not_run：指定本地 Qwen3 snapshot 与 /opt/NeMo-RL 尚不存在，resources.json 缺失。检查结果保存在 runs/program-v1/G08/startup-dry-run/；没有根据样例自行赋权、下载权重或创建 Ray 集群。

## 数值、观察与 refit

保留 next-token/full-token、BS/SB、mask 和有效 token 归约的显式 adapter。CPU 反例覆盖双重 shift、错误 target/mask、微批均值平均以及同版本语义错误；版本不同的已验证 synthetic mapping 可接受。SFT 区分 FP32 optimizer master 更新与 BF16 参数舍入，未变的 BF16 scalar 不直接否定 master update。

NeMo loss tap 保持官方返回对象与可导图；固定公开 actor/critic loss 函数体执行真实 CPU forward/backward，独立标量诊断不回灌 trainer。Worker 记录真实选定参数/主梯度/step 和初始 HF 转换、checkpoint load 完成；调用完成不等于 HF 数值等价。生成事件按原输入、生成 token、LP、reward、mask 和版本核对，提前/stale/错序事件拒绝。

两轮 GRPO/PPO observer 测试用明确 synthetic 引擎和 reward，真实 CPU SGD/loss/backward；完成 actor/critic 更新后，最终同一输入分别在 refit 前后 greedy 生成，并比较 refit 后生成 token LP 与当前 policy LP。真实 runtime 才会生成 observed_rl，synthetic 测试结果保持 synthetic_nemo_contract/reference。failed refit、缺迭代、LP 不匹配没有成功 result。acknowledged 仅表示官方调用成功返回，weight_hash_verified=false；无权重 hash 不准提升为 synchronized。

## 资源、故障与清理

启动先拒绝未授权计划，再读取接口，最后创建专属输出；worker 重新核验精确资源/环境/输入。NeMo registry 仅允许当前 sys.executable，UV/HF offline，拒绝已有 Ray context/外部地址，创建自己的本地 context 后 finally shutdown。actor 转换 cache 与 critic 分开且必须新建，观察器不覆盖原始文件。

Linux 监督器运行在独立 Python 子进程，使用 subreaper 收养 double-fork/setsid 后代；用 pidfd 绑定身份，避免对复用的 PID 发信号。只沿本次子树读取 children，不按名字、全局 Ray 或进程组猜测归属。wall/log 上限、leader 正常提前退出、监督父进程被终止都会清理；SIGTERM 后至多两秒升级 SIGKILL，总清理上限五秒，单列清理结果。此边界不承诺内核失效/OOM 等外部故障时的资源管理。不可用的 Linux /proc/pidfd 环境在启动训练前拒绝。

7 项真实 stdlib 子进程检查已通过，包括独立会话、双重派生、忽略 TERM、父进程退出、无关 sibling 保活及调用者 subreaper 状态不变。6 项 synthetic 启动检查覆盖 GRPO/PPO 环境/receipt、源码变化、缺失或 synthetic 成功结果拒绝；监督异常会把 receipt 终结为 failed。

## UI、证据与验证

RuntimeLab 只校验数据，显示 imported_claim，不执行上传的 command/path。合法参考 trace 可导出原值；未知 mapping、错误 provenance、hash、mask、loss 和 refit 均拒绝并清空成功视图。完整配置有独立可读折叠面板，说明结构/hash 校验不证明执行。手机测试验证页面无整体溢出，token 表和长配置在局部滚动。

本轮专项通过 43 项 Python/TypeScript 同输入判定、实际 CPU 适配/worker/observer 与真实 Bridge 配置构造；日志 runtime-startup-config-preflight.log。运行页 6 条 Chromium 路径通过，包含 308 字段配置显示/导出和错误内层 hash 拒绝；日志 runtime-startup-browser.log。固定源码原文由阶段 gate 再逐字核验，完整基线与所有日志/源码指纹以新生成 G08 report 为准。

G08 软件验收不消除 R01/R02 的缺权重、缺 NeMo 安装和无 GPU 授权边界；不冒充 Firefox/WebKit、实际训练吞吐、显存、NCCL 或模型效果验证。

本次完整基线已实际通过 170 项 Python、94 项 Chromium，skipped/flaky/unexpected=0；报告 runs/goal-g01/20261004T232220750439Z/report.json。已查看当前构建生成的 g08-effective-config-mobile.png：内层配置折叠标题、hash/声明和局部代码滚动可读；未出现页面整体横向溢出。该截图是 synthetic 配置界面验证，不是 observed runtime 记录。
