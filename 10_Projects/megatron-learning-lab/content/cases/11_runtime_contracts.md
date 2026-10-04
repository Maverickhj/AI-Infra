---
type: knowledge
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# 从同一批输入到可比较的运行证据

> 本页实现 G08 的接入软件合约。当前可只读导入 trace、检查字段与数值、查看明确的数据映射；这不证明 Bridge 或 NeMo RL 已实际训练。内置 SFT/PPO 数据由 authored 两层模型在 CPU 执行生成，属于 reference。真实模型/引擎验证分别归 R01/R02。

## 从整模与样本进入

SFT 的观测点在整模 head 与 masked CE 之间；RL 的观测点在固定 action 的 logprob 与正式 loss 函数之间。采集器只拿所需的 token、mask、目标概率和选定参数切片，不保存全模型 logits 或激活。运行配置、权重、tokenizer 和源码身份与这些数值一起记录，才能解释差值。

本页两个槽位都支持本地 JSON 与粘贴。先核验单份合约，再判断两份输入是否可比较。网页不执行文件中的命令、不自动打开其中的路径、不反序列化 pickle，也不根据 `observed` 字样宣称已经运行。

## 独立数学符号表

| 符号 | 含义与 shape |
|---|---|
| $B,S,V,H$ | 批大小、完整序列长度、词表大小、隐藏维 |
| $x_{ij},y_{ij}$ | 输入 ID 与预测目标，均为 $[B,S]$ |
| $m_{ij},N$ | 二值监督 mask 与有效 token 数，$N=\sum_{ij}m_{ij}$ |
| $z_{ijv}$ | head logits，$[B,S,V]$；不是 token loss |
| $\ell_{ij}$ | 所选目标/action 的 logprob，$[B,S]$ |
| $L_\Sigma,L$ | loss sum 与全局有效 token 平均 |
| $v_g,v_p,v_\theta,v_{after}$ | generation、previous、current 与一次更新后的 policy 版本 |
| $\Delta_{max}$ | 两份可比较文件的最大绝对数值差 |

## HF、Bridge 与 NeMo 的位置含义

HF 返回 `logits[B,S,V]`，目标概率是：

$$
\ell_{ij}=z_{ij,y_{ij}}-\log\sum_v e^{z_{ijv}}.
$$

这里输入合约已经包含一次 next-token shift：有效位置的 $y_{ij}=x_{i,j+1}$。采集时直接在 logits 上 gather 目标，不能再把这份 shifted labels 交给 HF 自带的 shift-loss；否则会预测后面第二个 token。

普通 Bridge batch 的 `tokens` 和 `labels` 已由 dataset/collator 对齐。带 labels 的 GPT 返回值可能是 `token_loss[B,S]`，此时 $\ell=-\mathrm{token\_loss}$；不能对二维 loss 再做 softmax。无 labels 的 logits 路径需显式选择 BSH 或 SBH 映射，不能只看两个维度恰好相等就猜布局。

Bridge 的最后一个 label 仍可能有效。转成网页合约时，在输入末尾追加该真实 target，新增一格 mask=0；原有预测、loss 与有效 count 全部保留。这个 context-only 格子不执行第二次 shift。当前薄适配器明确拒绝 THD/packing 和 position reset；这些需要单独映射，不能静默当普通序列。

NeMo loss 的 current 参数是 next-token logprob $[B,S-1]$，而 generation/previous/reference、advantage 和 token mask 在 batch 中是完整 action 位置 $[B,S]$。适配器在 current 前加不参与 loss 的 dummy 槽，保持 action $j$ 由 prefix $j-1$ 预测。masked gap 的 returns 不应强制清零。force-on-policy 分支使用 current.detach() 作为 previous，仍需相同版本约束；它不证明陈旧样本变成真实 on-policy。

## Mask、归约与可以手算的反例

$$
L_\Sigma=-\sum_{ij}m_{ij}\ell_{ij},\qquad
L=L_\Sigma/N.
$$

设两条序列的有效 token loss 分别为 $[1]$、$[3,3,3]$，全局平均为 $10/4=2.5$；先分别平均再取平均得到 2，改变了目标。合约从逐 token 值独立重算 sum/count/mean。RL 另按显式 token/sequence 归约检查，PPO value loss 单独验证。

CPU 测试用概率 $[1,2,3]/6$ 的手算、实际 cross-entropy 与 backward 检查 HF/Bridge 路径；另用实际 CPU PPO reference 检查 NeMo 字段与 action 对齐。合约 test double 标 synthetic，只证明适配与调用协议。LossTap 委托原 loss 一次，返回完全相同的 loss/metrics 对象，采集副本不替换正式计算图。

## 来源、配置与导入信任

每份 trace 保存 model/tokenizer revision、权重与架构来源、chat template hash、backend/dtype、实际 group 来源、运行源码路径/符号/hash、执行状态与采集边界。input/config 的 JSON 原文作为字符串保存，并对 UTF-8 字节算 SHA256；浏览器不重新排序或重写这些字符串后再验证，避免跨语言浮点序列化产生另一个 hash。

`derived` 只能带公式输入，不能带测量结果。`reference` 可表示明确的 authored CPU、synthetic contract 或实际 HF reference；`observed_bridge`/`observed_rl` 还要求对应 backend、真实 checkpoint 身份、不可变 revision、真实 group 和执行信息。但这些字段仍来自导入文件，校验通过统一标为 `imported_claim`：结构和数值一致，不等于运行身份认证。

实际配置采集调用框架容器的 to_dict()，再保留路径、dtype、callable 来源等明确类型。上游无界阈值用 {"float_sentinel":"+inf"} 或负号对应标签保存；NaN 以及测量中的所有非有限数仍拒绝。配置标签只是元数据，不参与 loss 计算。

只读 NeMo 配置解析支持已检查的 defaults 继承、立即子节 _override_、Hydra overrides 和纯 mul/div/max；拒绝环境读取、任意 resolver、Python YAML tag 与越界路径。固定源码配置样例随测试保存并核对 hash，不依赖临时 runs 缓存；它们是解析测试输入，不能直接作为获授权的训练资源配置。当前官方 helper 在合并前读取顶层 override，引用仅存在于父文件的顶层变量会失败，此分支明确拒绝，不猜结果。

程序版本相同不能挽救错误 shift/mask/loss；版本不同也不自动失败。选择显式 mapping 后，仍要实际执行接口、数据、小规模行为、源码映射各层探针。固定参考源码和本次运行源码分别保留。

## 比较与版本边界

两份文件的 task、input hash（包含 mask 与 policy 版本）、model/tokenizer revision、template hash 一致时：

$$
\Delta_{max}=\max_{ij}|\ell^{(A)}_{ij}-\ell^{(B)}_{ij}|.
$$

页面同时显示两侧来源与 config hash。配置不同需要逐项解释；即使差为零，也只说明导入的这些数值相同，不能推出梯度、optimizer、save/resume 或跨引擎等价。当前 schema 不接受未限定的耗时，因此不会凭模拟生成 GPU 性能图。

RL refit 只有完成 acknowledgment、export/ack hash 相同、generation 版本等于 after 才能标 synchronized。只递增一个版本整数不会自动替换生成引擎权重。

NeMo 当前参考实现等待传输和生成 worker 的结果后返回，但没有返回端到端权重 hash。这类事件只能标 acknowledged：记录实际调用完成、after 对应的 generation 版本、事件序号，并明确 weight_hash_verified=false；不能填造 export_hash/ack_hash。页面显示“文件记录 refit 调用完成；权重 hash 未核验”。这仍是导入字段，真实 R02 还要在最终 refit 前后使用同一输入生成，并将 refit 后生成 token 的 logprob 与训练 policy 重算值比较，不能用状态标签代替该验证。

## 当前可执行的只读入口

在开发容器的项目目录：

```bash
env -u PYTHONPATH uv run --no-project python -S tools/runtime_cli.py probe-bridge --root /opt/Megatron-Bridge
env -u PYTHONPATH uv run --no-project python -S tools/runtime_cli.py validate-trace path/to/single-trace.json
```

`probe-bridge` 只解析源码 AST，不导入 launcher。它发现当前 Qwen3 recipe 没有 hf_path 参数，所以不会把命令行上看似存在的参数误当作权重已经加载或离线保证。NeMo 探针同样不调用带 init_ray 的 main。实际配置采集与只读配置解析已有检查；HF/Bridge/NeMo 的入口、dry-run 资源绑定与采集器已接线；正式运行仍需精确资源授权和当前环境行为验收。样例输出路径须替换后重新绑定计划，不能直接当作已授权任务。

完整运行时配置保留为内层 JSON 和独立 SHA256。网页会拒绝重复 key、非有限数、危险 key、过深/过大配置和 hash 不符，并提供“查看运行时完整配置”。实际 Bridge model 配置有 308 个字段，因此仅配置内层每对象允许至多 512 字段；普通 trace 对象仍是 256。结构/hash 一致不证明该配置已执行，也不证明参数语义正确。

开发容器内可用 `tools/runtime_cli.py dry-run --plan profiles/runtime-grpo-plan.example.json` 查看命令与缺失前提，PPO 对应 `runtime-ppo-plan.example.json`。dry-run 不创建模型、GPU 或 Ray 集群；Linux 运行监督器使用独立 subreaper/pidfd 回收本次子进程。完整步骤与边界见项目 `profiles/minimum-compatibility.md`。

## 验证边界

重复 JSON key、非有限数、过深/过大数据、错误 hash、缺失 mask、未知 mapping、错误 provenance、重复 shift、跨文档监督、错误归约与 refit acknowledgment 都有拒绝路径。JSON 中的命令仅显示为文本。CPU 与 synthetic 合约不代替 GPU、官方 tokenizer、正式 trainer、真实 rollout 或资源上限验证。
