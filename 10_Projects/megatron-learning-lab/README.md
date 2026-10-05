---
type: project
status: draft
created: 2026-09-30
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# megatron-learning-lab

从一条样本走到完整模型、loss、梯度、更新与源码的中文学习工作台。默认进入 Qwen3-0.6B 整模；Qwen2.5、Qwen3 MoE 与 DeepSeek MLA/MoE 的配置差异可直接对照。基础速览可跳过，核心课程按需展开并保留独立数学符号表。

当前阶段和验收证据见 [PROGRAM-V1 交付记录](goals/PROGRAM-V1/REPORT.md)、[阶段账本](goals/PROGRAM-V1/STATE.json) 与 [进度](goals/PROGRAM-V1/PROGRESS.md)。CPU 学习参考和运行接入软件不等于真实 pretrained 模型训练；R01/R02 必须使用获授权且就绪的资源单独验收。

## 在现有开发容器运行

所有项目代码、生成器和测试均在 `minimind-megatron-bridge-dev-1` 中执行。先从宿主机进入已有容器：

```bash
docker exec -it -w /opt/AI-Infra/10_Projects/megatron-learning-lab minimind-megatron-bridge-dev-1 bash
```

以下命令在容器内执行，复用已经准备好的 Node 与 Chromium 缓存：

```bash
export PATH=/data/cache/megatron-lab-node/bin:$PATH
export PLAYWRIGHT_BROWSERS_PATH=/data/cache/megatron-lab-browsers
export CUDA_VISIBLE_DEVICES=
unset PYTHONPATH
npm run dev
```

访问 http://localhost:5173。容器外需已有端口映射或编辑器转发；开发服务监听 0.0.0.0:5173，本项目未公开部署。默认开发启动会重新生成配置派生数据以及实际 CPU decoder/TP-DP 参考，可能需要几十秒。浏览器里载入样例或点击演算不会启动 GPU。

换用新容器时，需要 Node >=22.12、Python、uv、当前 CPU 参考所需的 PyTorch，以及 package-lock.json 对应的前端依赖和 Playwright Chromium。先按 [最小兼容策略](profiles/minimum-compatibility.md) 核查已有环境，再补所需依赖；源码参考锁不是 Python 安装锁。不要求复制所有上游 optional extras，也不在宿主机安装训练依赖。

## 可以沿着哪些路径学习

| 路径 | 可操作内容 | 数值与边界 |
|---|---|---|
| 样本与监督 | assistant/last_turn/full、一次 shift、截断、padding、pack/unpack、token 表 | 小词表 authored tokens；真实 tokenizer trace 可只读导入 |
| Qwen decoder | embedding、两层 GQA/residual/SwiGLU、final norm/head、masked CE | 完整微型架构的 TS/CPU 对照，不是 pretrained Qwen 权重 |
| 梯度与更新 | 选定参数 autograd/有限差分、冻结参数、共享 head、save/resume 下一步 | 实际 CPU 参考；不宣称 Bridge GPU resume 等价 |
| TP/DP/PP/SP/CP | rank/group、QKV/FFN/词表分片、sum/count、1F1B、序列与 packing 边界 | CPU 数值切分或逻辑时间；不当作 NCCL 耗时/显存 |
| MoE 与 MLA | router/top-k、dispatch/shared expert、EP/ETP、norm、latent/cache | 完整缩小模型参考；不等于全量 DeepSeek-V3 执行 |
| GRPO/PPO | 四类 logprob、mask、advantage/GAE、clip/KL/value、更新/refit 版本 | authored trajectories 与 CPU 更新；非真实 rollout |
| 运行对照 | 本地 JSON/粘贴、loss/梯度/更新差、来源/版本、完整配置、通信预算、导出 | 导入均标 imported_claim；缺失指标保持未采集 |

顶层模型图来自完整 HF config/family 映射，微型可演算模型另有明确标签。切换 Qwen3/Qwen2.5 会改变 head_dim、QKV bias 和 QK norm；DeepSeek-R1-Distill-Qwen 仍是 Qwen/GQA，不标 MLA。

地址栏 hash 保存 model/scenario/step/layer、数值子步骤与 rank 等状态，可直接打开或刷新，例如 `/#step=input&view=sample&sftLayout=packed`、`/#scenario=rl&rlAlgorithm=ppo&step=advantage`、`/#view=runtime`。生产预览也验证了嵌套路径下的 hash 入口；这不表示已经配置任意外部静态托管服务的 rewrite。

## 源码与证据

当前收录 49 个固定源码文件的 111 段原文，原始行号、Git blob、commit、许可证与中文讲解分开。源码弹窗按当前步骤、算法和分支定位；运行通道未建立时明确提示，不能用参考链接冒充实际 runtime 来源。

`derived` 是配置/公式推导；`reference` 包括明确的 authored CPU 或 synthetic 合约；`observed_bridge`、`observed_rl` 需要运行清单和数值验收。即使导入文件字段自称 observed，网页也只证明结构/数值合约通过，不认证运行身份。比较同一 input/mask/version 定义；梯度还核对参数坐标、角色、dtype 和更新前值。

PPO 样例记录完整可训练梯度 head[27,8] 与 critic[8] 的 float64 尺寸。DP 通信显示理想 ring AllReduce 的派生字节预算，实际通信仍为未采集；forward-only 样例缺梯度尺寸时不猜数值。

公式和 KaTeX 字体、课程、参考数据与源码摘录均随构建打包，无 CDN 依赖。完整源码全文仅保存在忽略的 `runs/source-cache/`；网页使用打包的摘录。

```bash
env -u PYTHONPATH python -S tools/verify_source_snippets.py
env -u PYTHONPATH python -S tools/verify_source_snippets.py --cache runs/source-cache
# 新环境缺完整缓存时，显式获取固定版本；不更新 source.lock
env -u PYTHONPATH python -S tools/verify_source_snippets.py --fetch --cache runs/source-cache
```

研究背景见 [FINDINGS](research/FINDINGS.md)、[REFERENCES](research/REFERENCES.md)、[学习深度](CURRICULUM.md) 和 [实施方案](PLAN.md)。

## 软件验收与生产预览

保持上述容器环境变量：

```bash
npm run test:runtime-contracts
npm run test:release
```

`test:release` 顺序执行完整 Python/元数据/handoff/源码/构建/Chromium 基线，以及 test:gqa、test:sft-data、test:decoder、test:tp-dp、test:sequence、test:moe-mla、test:rl-reference、test:runtime-contracts；随后对相同 dist 单独执行生产 smoke。所有检查必须实际通过，不能跳过、flaky 或零用例。

生产 smoke 启动专属 127.0.0.1:5174 服务，拒绝复用已有服务，覆盖整条学习路线、直接/嵌套入口、离线数学/字体、窄屏、键盘和缺少运行证据的降级。请先停止占用 5173/5174 的个人预览；测试不会自动关闭不属于自己的服务。Firefox/WebKit 没有执行。

每轮结果在 `runs/program-v1/G09/<run-id>/report.json`，包含真实命令/退出码/日志 hash、当前源码清单、已安装依赖版本与 lock 指纹、配置/构建清单、生产截图与阶段报告索引。完成后还需审读报告和截图，再更新 STATE；脚本不自动将阶段或原生 Goal 标完成。历史阶段报告只证明当时的源码快照。

单独查看生产构建：

```bash
npm run build
npm run preview
# 同一 dist 的独立 smoke，不执行构建或部署
npm exec -- playwright test --config tests/playwright.release.config.ts
```

普通 preview 使用 5173；独立 smoke 使用 5174。端口转发不会自动创建公开站点。

## 真实 HF / Bridge / NeMo 运行

接入脚本在 `experiments/runtime/`，只读 CLI 在 `tools/runtime_cli.py`：

```bash
env -u PYTHONPATH python -S tools/runtime_cli.py dry-run --plan profiles/runtime-plan.example.json
env -u PYTHONPATH python -S tools/runtime_cli.py dry-run --plan profiles/runtime-grpo-plan.example.json
env -u PYTHONPATH python -S tools/runtime_cli.py dry-run --plan profiles/runtime-ppo-plan.example.json
```

样例输出路径为占位符，不含资源授权。正式运行先填写实际 pinned model/tokenizer、数据、backend、设备、步数、时限和专属输出目录，并按 [资源约定](goals/PROGRAM-V1/RESOURCES.md) 绑定明确授权及完整 plan hash。PPO critic 需要独立授权；不可把 synthetic 测试的 authorized=true 复制为权限。

启动前核查模型/tokenizer 元数据、实际源码/接口、配置与输入 hash；正式采集保留实际 runtime lineage。Linux 专属 subreaper/pidfd 负责本次子树的时限、日志上限和退出清理。NeMo refit acknowledged 只代表官方调用返回，端到端权重 hash 尚未验证；R02 仍须固定输入下训推 logprob 对齐。

当前真实验证的缺口：没有明确 GPU/RL 授权；指定 Qwen3 权重/tokenizer snapshot 与 NeMo checkout 缺失。已发现的 GPU0 为 6 GiB，当前全参数 Adam 配置的权重、master、moments 推导约需 7.77 GiB，尚不含梯度/激活；这不是显存实测。实际 HF/Bridge 对齐、GPU SFT、save/resume、HF export、同步 rollout/update/refit 都留待 R01/R02，不能据 CPU 结果宣称完成。

## 接续开发

连续目标入口为 [MASTER_GOAL](goals/PROGRAM-V1/MASTER_GOAL.md) 与 [plan.json](goals/PROGRAM-V1/plan.json)，使用 `python -S tools/program_plan.py next` 读取下一阶段。该工具只检查账本/证据完整性，不代替运行验收。

[CODEX_START](CODEX_START.md)、[implementation-report.json](implementation-report.json) 和 [validation-report.json](validation-report.json) 保留首轮历史；其旧计数和范围不代表当前版本。后续若 R01/R02 改变集成代码，将 G09 标记 revalidate 并重跑整体验收。日志、截图、生成数据、模型、trace 与缓存沿用现有忽略规则，不加入 Git。
