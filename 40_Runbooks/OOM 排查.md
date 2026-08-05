---
type: runbook
status: seed
created: 2026-08-05
updated: 2026-08-05
domains: [memory, operations]
aliases: [GPU OOM Triage]
tags: [oom, cuda, memory]
ai_generated: true
reviewed: false
---

# OOM 排查

> [!warning]
> 先定位峰值 phase 和 owner，再调整并行或模型配置。没有 traceback、rank 和输入规模时，只能给出候选原因，不能声称找到根因。

## 快速分流

1. 是 GPU OOM、host OOM、shared memory 还是 payload/序列化容量错误？
2. 发生在 rollout、prefill、decode、train forward、backward、optimizer、weight sync 还是 checkpoint？
3. 每次都在同一 rank/step/input 发生，还是随时间增长？
4. `allocated` 接近容量，还是 `reserved` 与 `allocated` 差距很大？
5. 是否刚发生 engine reinitialize、模型重载、batch/token 增长或 topology 变化？

## 证据清单

- 完整 OOM 消息和 innermost traceback。
- role、PID、rank、parallel coordinates、logical device。
- 当前 micro-batch 的 text tokens、raw media tokens、合并后 media tokens。
- 参数/gradient/optimizer 分片策略与 dtype。
- activation recompute、offload、attention backend。
- cache/pool 配置与 engine 实例数。
- allocator allocated/reserved/peak 和时间序列。

## 四类模式

### 容量型

同一输入稳定复现，allocated 接近设备上限。优先降低本次 forward token/media 规模、增加有效分片或减少常驻状态。

### 峰值重叠型

只在某个 phase 出现，例如完整 logits、all-gather、optimizer step 或 weight sync 与旧 buffer 同时存活。寻找生命周期重叠，优先流式处理或提前释放无用引用。

### 碎片型

reserved 很高但可满足大块连续申请的空间不足。记录分配历史与最大申请；不要只调用清 cache 并把症状消失当成根因修复。

### 泄漏/生命周期型

峰值随 step、retry 或 initialize 次数单调增长。检查 Python 引用、异步任务、hook、cache、IPC pool、线程/进程 cleanup 和重复 engine construction。

## 分阶段行动顺序

1. **复现与定位**：固定输入、step、rank 与 phase。
2. **输入 A/B**：降低 tokens/media；判断激活/临时张量是否主导。
3. **实现 A/B**：recompute、attention backend、logits 分块、cache/pool 大小。
4. **拓扑 A/B**：在满足 head/expert/layer/checkpoint 约束下调整 TP/PP/CP/EP。
5. **生命周期 A/B**：对比首次启动与 repeated initialize/retry。
6. **Profiler**：对仍不清楚的峰值记录 allocator snapshot 和分层 trace。

一次只改变一类杠杆，否则无法判断哪个假设成立。

## 多模态额外检查

- 区分 raw vision tokens、merge 后 feature 和 LLM vision tokens。
- 视觉 encoder 是否真的使用 CP/SP，而不是只有顶层 LLM 配置开启。
- vision blocks、projector 与首个 PP stage 是否形成局部热点。
- feature cache/pool 是每节点、每进程、每 tokenizer worker 还是每 DP instance 一份。

## 成功标准

- 越过原失败点并保持目标语义。
- 峰值下降与假设方向一致。
- 至少运行足够 step 排除逐步泄漏。
- 吞吐、数值与恢复行为没有不可接受回归。
- 将 A/B 记录为 [[90_Templates/Experiment|Experiment]]，而不是只保留最终配置。

## 相关笔记

- [[20_Knowledge/Memory/显存分析方法|显存分析方法]]
- [[20_Knowledge/Distributed/大模型并行训练|大模型并行训练]]
- [[20_Knowledge/Multimodal/多模态模型数据流|多模态模型数据流]]
