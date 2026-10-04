---
type: experiment
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# G07 自审

## 完整模型与可导路径

实际审读 experiments/rl_reference.py、web/rl/compute.ts 及 G03 backbone。每个 action j 使用完整两层 decoder 的 j-1 final norm，dummy j=0 和 padding 明确。backbone 冻结，完整词表 LM head [27,8] 与 critic [8] 可训练；fixture/current version1 明确 authored，不捏造历史 update。scope、课程与导出都标 reference/no rollout。

CPU torch.autograd 与实际 torch.optim.SGD 运行，TS 使用独立解析 log-softmax/head 梯度；没有用 TS 结果生成 CPU oracle。actor backward 不触及 critic，critic backward 不改变 actor gradient；GRPO critic 保持不变。PPO 旧 values/returns/advantages detached。固定源 R-PPO 先 critic 后 actor，而参考两个独立图顺序可交换，课程明确不验证生产调度。

## 数学与反例

GRPO 读取 R-UTIL 的 Bessel 修正，与 1/0 手算一致；同分和 singleton 优势零，KL 仍可改变全损失梯度。GAE Python reverse carry 与 TS 有效位置有限和独立；masked gap 不多折扣，returns 的 masked 槽位不造零。手算 0.2/0.4、gamma0.9/lambda0.8 得 A0.592/0.6、T0.792/1。

正负 advantage 四分支的 PG 值与 dlogp 有手算；token/sequence reduction 使用各自有效 count，不混淆 sequence ratio。value clipping 使用 max squared error、半系数与 token mean，两个 value 梯度手算 0.5/0。KL k3 在 exp(curr-curr.detach()) 的采样权重上保留导数，未 clamp 的结果为 curr-ref；错误 detach 得不同梯度，即使前向 loss 相等。input clamp 时 weight 也 detach，output clamp 在乘权重后。force ratio=1 不取常数；有限差分固定 detach anchor。

实际通过16组54957值，max_abs_error=7.993605777301127e-15，atol=rtol=1e-10 未更改；另8项CPU检查覆盖上述手算、错误算法、梯度隔离、真实更新与 refit 拒绝。数值检查先于完整门槛，最终仍以门槛快照为准。

## 来源与课程

读过固定 R-ADV、R-UTIL、R-PPO 对应 reviewed_ranges，扩充 R-LOSS 和 R-GRPO 的实际分支。全部新片段从完整固定 blob 提取，49文件/111片段检查通过；版权和中文注释分离。新版03课程只补本次静态审阅已完成之处，真实循环仍 not_run。新增10课程保留完整数学符号表、shape、版本、手算、导数和未支持配置；不把 fixed actions 叫新生成。

源片段只证明读过固定代码，不证明本机安装了 NeMo RL 或与其 runtime 兼容；也不把高层 Python 分支当 kernel。本地参考 hash/ack wire format 与生产 refit 传输区分。

## 交互、导出与 refit

RLJourney 接在现有6步路线。算法、轨迹、token、reduction、KL、force、同分组与错误实现有真实计算影响；配置改变会重置一次更新流程。错误实现可查看但禁止导出和应用参数。选定 head 行展示全批累计梯度，PPO 才显示 critic；导出保留各自来源、mask 与版本。

VersionDemo 应用一次 reference SGD 后 generation 仍旧；导出固定参数与 canonical f64le hash。refit 复制候选参数，完成版本/hash/ack校验后才发布 generation；在同一组固定 action 上重新 forward，对齐更新后 policy。错误版本/hash/未完成 ack 均拒绝，generation 保持旧值。没有真正调用 rollout engine，没有声称第二次采样。

## 浏览器与视觉审阅

六条实际 Chromium 新路径全部通过，数值 oracle 来自该次 CPU 子进程；覆盖所有13个token的四类logprob/梯度、PPO masked GAE/value/critic、token-sequence与KL、force/同分/错误梯度、216个导出权重、三类refit失败及成功后generation重算、URL归一化和导出、离线源码/KaTeX/键盘/窄屏。

首轮五项通过、一项因 CPU --forward 未导出 force/KL-off 而失败；保留失败trace并扩展CPU导出至完整16组，没有删除断言。第二轮六项通过。实际查看桌面和窄屏截图后发现数字换行影响阅读，新增局部表格nowrap与对应断言；这一最终样式由完整阶段门槛复验。表格仍允许局部横向滚动，不拉宽整页。

## 边界与最终检查

没有 HF checkpoint、GPU、真实生成、训练收益、NCCL、异步/partial rollout、off-policy correction、dual clip、CISPO、OPD、多轮工具、截断 bootstrap 或生产 optimizer 验证。课程明确这一阶段只证明参考数值、交互和合约；R01/R02 仍需资源授权。

运行完整 G07 gate 后再检查 source manifest、所有退出码、零skip/flaky、当前截图与日志hash；全部通过才更新STATE，并继续G08，不把自审文件存在当作完成证据。
