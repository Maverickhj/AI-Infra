---
type: knowledge
status: seed
created: 2026-10-27
updated: 2026-10-27
domains:
  - ai-infra
  - profiling
  - distributed-training
  - inference
  - reinforcement-learning
aliases:
  - LLM AI-Infra Profiling 技术体系
  - GPU Profiling 心智模型
ai_generated: true
reviewed: false
---

# LLM AI-Infra Profiling 技术体系调研

> [!warning]
> 本文整合 2026-10-15 的 AI 辅助调研，并于 2026-10-27 整理入库，**尚未经过人工逐条核验**。工具的配置参数、指标口径和版本兼容性，应以目标环境实际文档为准。文中的所有数值与可视化图均为**教学假设**，不是 GPU 实测数据。

> **阅读目标**：面向熟悉 LLM / Megatron-LM / SGLang，但 Profiling 经验较少的 AI-Infra 工程师。先理解执行时序、测量口径和因果证据，再学习工具与框架实践。
>
> **与已有资料的关系**：本文关注通用方法、设计原理与入门案例；Megatron 的具体排障命令、GPU 显存/Host Memory/网络与 Tensor Debug 参考 [[megatron_lm_profiling_debug_guide|Megatron-LM 训练引擎 Profiling & Debug 实践指南]]（位于 `Train/`）。

## 0. 核心观点与数学符号表

**Profiling 的核心不是“谁最忙”，而是“为什么这个任务直到此刻才完成”。**

推荐的分析路径：

```text
明确端到端目标
  → 取得无 Profiler 基线
  → 观察 Timeline 与关键依赖
  → 归因 Compute / Memory / Communication / CPU / I/O / Scheduling
  → 局部 Profiling 形成可证伪假设
  → 使用相同工作负载验证 End-to-End 收益
```

原先按“显存、卡间通信、跨 POD 通信、算子性能”分类并没有错，但更适合作为资源层。它没有回答：工作为什么没有按时提交、哪些通信被计算覆盖、谁先到 collective、GPU 空隙是否真的拖慢任务。

### 数学符号表

| 符号 | 含义与单位 / 范围 |
|---|---|
| $T_{\mathrm{step}}$ | 一个训练 step 的端到端完成时间，通常为 ms |
| $T_{\mathrm{profile}},T_{\mathrm{baseline}}$ | 带/不带 profiler 的可比窗口耗时 |
| $T_{\mathrm{comm}}$ | 选定口径下的通信持续时间；需说明是 API 区间还是设备区间 |
| $F$ | 名义浮点运算量，FLOP |
| $Q$ | 特定存储层级上实际搬运的字节数，byte |
| $Q_{\min}$ | 理想最小张量搬运字节数，**非**实际 HBM 流量 |
| $I=F/Q$ | 算术强度，FLOP/byte |
| $I_{\mathrm{ideal}}=F/Q_{\min}$ | 理想算术强度，FLOP/byte |
| $B$ | 对应内存层级的有效/理论带宽，byte/s；使用时注明口径 |
| $P$、$P_{\max}$ | 实际计算吞吐、相应精度和指令路径的计算上限，FLOP/s |
| $M,N,K$ | GEMM 维度：$A\in\mathbb{R}^{M\times K},B\in\mathbb{R}^{K\times N}$ |
| $s$ | 张量单元素字节数，如 FP16 为 2 byte |
| $R$ | 一个通信 collective 的参与 rank 数 |
| $\lambda_r,\lambda_t$ | 有效 rollout 样本生产、训练样本消费速率，样本/s |
| $q$ | 等待队列中的样本量，样本 |
| $t$ | 时间变量，秒；图中毫秒另行标注 |

## 1. 基础概念：先知道自己在测什么

| 类别 | 回答的问题 | 典型输出 |
|---|---|---|
| Monitoring | 机器与系统现在是否异常？ | GPU 利用率、内存/网络指标、请求队列 |
| Benchmarking | 在固定工作负载下表现如何？ | tokens/s、step time、TTFT/ITL、尾延迟 |
| Profiling | 时间和资源花在什么活动上、为何发生等待？ | Trace、算子表、内存历史、硬件计数器 |

三者互补。例如 **GPU Util 下降是现象**；固定工作负载下的吞吐变化是影响；Timeline 才能帮助寻找可解释的执行机制。Profiling 生成证据，不自动生成根因。

### 1.1 CPU Launch 与 CUDA 异步执行

一行 `y = x @ w` 的 Python 返回，通常不能等同于矩阵乘法在 GPU 上已经完成。CUDA 工作由 CPU 发起后，可在 GPU stream 上异步执行：

```text
时间 →
CPU     [准备] [提交 A] [提交 B] [其他工作...]
GPU                 [执行 A ...] [执行 B ...]
```

必须区分 **API 提交时间**、**GPU 开始时间** 与 **GPU 完成时间**。没有必要的同步时，CPU 计时往往只覆盖提交路径；CUDA Events 和明确同步边界可用于设备计时，但同步本身可能改变本来的流水执行。GPU 提交延迟也不天然等于浪费：已有任务可能正在占满设备。

### 1.2 Event / Range / Metric / Trace / Timeline / Critical Path

- **Event**：某瞬间发生的一件事。
- **Range/Span**：有开始和结束的一段工作，可能属于某个 CPU 线程、请求或模块。
- **Metric**：统计窗口和口径明确的数值，如带宽、cache hit、队列长度。
- **Trace**：一组带时间、活动对象及关联信息的记录；**Timeline** 是其可视化。
- **Critical Path**：在依赖和资源约束下，决定最终完成时间的执行链，不等于单个耗时最长的 kernel。

**热点 ≠ 关键路径**：累计耗时大的通信可能完全与计算重叠；短但每一步都阻止后续工作开始的 CPU 同步却可能决定吞吐。

### 1.3 Profiling 数据从哪里来？

| 技术 | 直觉 | 擅长 | 局限 |
|---|---|---|---|
| Sampling | 定期看程序在做什么 | CPU 调用栈热点统计 | 短事件可能漏采，结果受采样口径影响 |
| Instrumentation | 在阶段首尾加入标记 | 模块/step 语义与耗时 | 过度插桩有开销 |
| Tracing | 按时间记录多种活动 | 异步、依赖、并发、等待 | 数据量大，采集扰动 |
| Hardware Counters | 读取硬件内部活动计数 | SM、指令、cache、memory、warp 行为 | 部分采集需 replay，解释依赖架构 |

注意：CPU **火焰图的宽度通常表示聚合采样量，而不是程序执行顺序**；想看先后与并发，使用 Timeline。一个采样显示 42% 落在 tokenizer，并不意味着 tokenizer 占整个 GPU 训练 wall time 的 42%。

### 1.4 如何关联层级与跨主机时间？

```text
step / request_id / rank
  → Python / PyTorch range
  → CUDA API
  → stream 上的 GPU kernel / memcpy / NCCL
  → 设备和网络的指标
```

CUPTI 记录 CUDA 相关活动与 correlation ID；NVTX 负责给低层活动加上有意义的高层阶段名称。编译、融合、CUDA Graph 可能让“一个 PyTorch op ↔ 一个 kernel”的关系不再成立，不能靠时间邻近就认定因果关联。

分布式 Trace 还涉及主机时钟同步。跨机把两个 JSON 拼接并不保证微秒级因果顺序可信。要同时保留 step、rank、collective、request 等**语义关联**，并明确时钟映射和校准误差。

## 2. 从目标到资源：推荐的 Profiling 分层体系

| 层次 | 核心问题 | 重点 |
|---|---|---|
| 目标层 | 何谓更快、更有效？ | Step Time、有效 token/s、TTFT/ITL、质量目标所需 wall time |
| 执行层 | 工作如何推进？ | Timeline、GPU 空隙、等待、Critical Path、重叠 |
| 资源层 | 什么限制执行？ | Compute、GPU Memory 容量/带宽、NCCL、CPU、Host RAM、I/O |
| 框架层 | 为什么这么安排？ | TP/PP/CP/EP、MoE、batching、KV cache、RL 调度 |
| 验证层 | 优化真的有效吗？ | 无 profiler 的 A/B、正确性与方差、延迟/吞吐权衡 |

### 2.1 显存必须区分容量与带宽

- **容量与生命周期**：当前分配、峰值、缓冲、激活、临时 tensor、allocator 缓存和 OOM。
- **访问性能**：HBM/L2 流量、数据复用、带宽和计算吞吐。

它们是不同问题。高 Reserved 不必然表示碎片化；高 HBM 带宽也不直接说明内存分配异常。

### 2.2 通信按逻辑组和物理拓扑关联

```text
TP / DP / PP / CP / EP 的 collective 或 P2P
  → rank 布局与各自到达时间
  → GPU / NUMA / PCIe / NVLink / NIC
  → 节点与交换网络
  → 跨部署域（若实际发生）
```

“跨 POD”只是部署标签，应明确具体网络拓扑、通信模式和消息大小。**暴露在关键路径上的通信时间**通常比通信区间的简单总和更能解释端到端性能。

### 2.3 补上 CPU Runtime、Host Memory 和 I/O

Python 数据准备、tokenization、线程调度、launch、同步、Host-to-Device 搬运、checkpoint、RDMA 控制面，都可能影响 GPU 是否按时收到工作。CPU 利用率不高也可能处于等待状态；不可只依据 `nvidia-smi` 判断。

## 3. 主流工具：按问题分工

| 工具 | 层级与核心用途 | 避免误用 |
|---|---|---|
| **Nsight Systems** | CPU/GPU 时间线、stream、NVTX、NCCL、memcpy 与空隙 | 不能替代详细 kernel 微架构分析 |
| **PyTorch Profiler** | PyTorch op、shape、stack、CPU/CUDA 关联、`record_function` | self/total 与并发 wall time 不能混为一谈 |
| **Nsight Compute** | 单/少量 kernel 的指令、cache、带宽、occupancy、Roofline | replay/采集扰动下的时长不能直接代表线上 |
| **PyTorch CUDA Memory Snapshot** | 张量分配生命周期、allocator 状态、OOM 现场 | 无法覆盖所有绕过 allocator 的 CUDA/NCCL 内存 |
| **NCCL tracing / Flight Recorder** | collective 进展、异常等待与 rank 关联 | 不是交换机遥测 |
| **nccl-tests / RDMA perftest** | 通信基线测试 | Benchmark 并不直接解释真实训练的 NCCL 调用 |
| **perf / py-spy / BCC** | CPU/OS/线程/调用栈/运行队列/I/O | 不等于 GPU 硬件剖析 |
| **Memray** | Python/native 主机内存分配 | 不等于 CUDA Memory Snapshot |
| **DCGM** | 持续设备监控与指标 | 不能直接归因到某个 PyTorch operator |
| **Perfetto / HTA** | Trace 查看、查询与分布式离线分析 | 不负责所有底层数据采集 |

建议学习顺序：**PyTorch Profiler + Nsight Systems + Memory Snapshot → NCCL/CPU 专项 → Nsight Compute**。把单 kernel 研究放在确认 kernel 确实影响关键路径之后。

### 3.1 三种特别重要的分工

**CUPTI**：底层 CUDA Profiling 数据采集接口；**NVTX**：插入高层语义标记；**Perfetto**：查看/查询 Trace 的工具平台。这三个不是互相竞争的独立 GPU profiler。

**Nsight Systems** 应优先看“什么时候提交、执行、等待”；**Nsight Compute** 才深入“这一个 kernel 内部为什么快或慢”。

PyTorch Profiler 的 `record_shapes`、`with_stack`、`profile_memory` 并非免费。限制活动窗口，避免第一次就同时打开所有采集开关；保留无 profiler 的可比基线。

## 4. 四个直观案例（均为假设数据）

![四个 Profiling 基本案例](LLM%20AI-Infra%20Profiling%20技术体系调研.assets/profiling-concepts.svg)

> 图片以一个 SVG 汇总四个教学案例：计算通信重叠、显存分配与缓存、Roofline、集体通信中的迟到者。**所有数据都是人为构造，禁止作为实测结果引用**。

### 4.1 Timeline：通信 4 ms 不变，为什么从 10 ms 变成 6 ms？

设 A=2 ms，AllReduce=4 ms，计算 B=4 ms。假设 B 不依赖通信结果，并发执行时不存在资源争用。

顺序执行：

$$
T_{\mathrm{serial}}=2+4+4=10\ \mathrm{ms}.
$$

允许通信与 B 并发：

$$
T_{\mathrm{overlap}}=2+\max(4,4)=6\ \mathrm{ms}.
$$

工作量与通信时长没有减少，完成时间减少了。真正的问题是：**通信有多少时间未被计算覆盖、实际重叠时是否产生资源争用**。CUDA stream 是执行序列，不是拥有独立硬件资源的“另一张 GPU”。

### 4.2 Memory：释放张量后，Reserved 为什么还高？

- `memory_allocated`：目前由 tensor 使用的分配器内存。
- `memory_reserved`：PyTorch caching allocator 管理的内存，包括备用缓存块。
- 驱动层更宽口径的设备占用：可包含 CUDA 上下文、NCCL 或其他分配；不必等于前两者。

示意：Allocated 从 14 GiB 降到 4 GiB，但 Reserved 保留 16 GiB，**只能证明 allocator 仍保留部分内存**，不能由差值单独认定内存泄漏或碎片化。OOM 需要追踪峰值那一刻并存的激活、临时 tensor、logits、loss 和通信缓冲等对象；不属于 PyTorch allocator 的显存另行观察。

### 4.3 Kernel 与 Roofline：计算是否等得到数据？

给定一个内存层级的搬运量 $Q$，算术强度 $I=F/Q$，简化上界为：

$$
P\leq\min(P_{\max},BI).
$$

例如 FP32 向量加法，每元素理想读取两个 4-byte 输入、写一个 4-byte 输出，完成一次浮点加法：

$$
I_{\mathrm{ideal}}\approx\frac{1}{12}\ \mathrm{FLOP/byte}.
$$

对 FP16 方阵 GEMM（边长 $n$），假定每个输入完整读一次、结果只写一次：

$$
F\approx 2n^3,\qquad Q_{\min}=6n^2,\qquad
I_{\mathrm{ideal}}=\frac{n}{3}\ \mathrm{FLOP/byte}.
$$

更一般地，$A\in\mathbb{R}^{M\times K},B\in\mathbb{R}^{K\times N}$、单元素 $s$ byte 时：

$$
F\approx 2MNK,\qquad Q_{\min}=s(MK+KN+MN).
$$

这是**理想数据搬运**，不是 HBM 硬件计数器。真实表现受到 tile 重载、缓存命中、shape、可并行工作量、精度与指令路径等影响。高 occupancy 也不等于高计算吞吐：warp 驻留数量与有效指令执行效率是两回事。

### 4.4 Straggler：AllReduce 区间最长的 rank 未必是元凶

假设 Rank 0/1 各在 4 ms 时完成前置计算并开始参与 collective，Rank 2 在 8 ms 才到，之后再需 2 ms 完成。较早参与的 rank 可观察到约 6 ms 的通信相关区间，Rank 2 则约 2 ms。

这里故意将“等待”和“完成通信”分段，**真实 NCCL 可能交织推进通信，不能简单硬拆**。排查先对齐同一个 collective，比较 rank 到达时刻，再检查晚到 rank 之前的计算、调度或数据准备，随后才检验网络是否异常。

## 5. 通信 / 跨节点 / 跨 POD：观察链路不能只看 NCCL 名字

至少区分三类不同的量：

1. **CPU API 时间**：主机发起/等待调用的区间。
2. **GPU/通信执行时间**：设备侧 NCCL 活动与相关依赖的区间。
3. **物理链路活动**：NIC/交换机等真实发送、接收、拥塞与流控指标。

`nccl-tests` 的 `algbw` 和 `busbw` 是在测试定义下计算的指标。以 AllReduce 为例：

$$
\mathrm{busbw}=\mathrm{algbw}\times\frac{2(R-1)}{R}.
$$

这里的 busbw **不是从 NIC 直接读取的线速值**。测试 collective、消息大小、rank 布局和拓扑应尽可能接近应用中的实际条件。

跨节点排查建议：应用/通信 Trace → rank 与消息关联 → GPU/NIC 拓扑 → RDMA 端点测试 → 网络交换设备计数器与时间窗口 → 对照实验。RoCE 的 ECN/PFC/CNP 等指标要按具体设备口径分析；RDMA 数据面不能想当然地用普通 socket 观测代替。跨机器 trace 还应标记时钟同步误差。

## 6. 在三个 LLM 框架里落地

### 6.1 Megatron-LM / Megatron-Core

| 并行与结构 | 观察重点 | 常见候选瓶颈 |
|---|---|---|
| TP / SP | 层内 collective、GEMM shape、计算通信衔接 | 小 GEMM、通信未覆盖 |
| PP | stage 负载、微批次进度、Send/Recv | bubble、stage 不均衡 |
| DP / Distributed Optimizer | gradient ReduceScatter、参数 AllGather | 尾部通信、权重就绪 |
| CP | attention 数据交换与计算 | 长序列通信、调度不匹配 |
| EP / MoE | dispatch/permute、token 数、Grouped GEMM、combine | 负载失衡、重排/通信开销 |

仅画 3 个 stage、3 个微批次的**前向**：

```text
时间槽      1    2    3    4    5
Stage 0   F1   F2   F3    .    .
Stage 1    .   F1   F2   F3    .
Stage 2    .    .   F1   F2   F3
```

点表示此简化前向时间线的空槽，不代表完整 1F1B 训练。Pipeline bubble 与 stage 依赖相关，不必然能靠单独优化一个 GEMM 消除。

MoE 则需要同时记录各 expert token 量、rank 消息分布、permute/dispatch、Grouped GEMM、combine/unpermute；只看 All-to-All 耗时可能漏掉根因。Megatron-Bridge 有按 rank/step 开窗的 Nsight Systems 与 PyTorch Profiler 集成，具体参数以安装版本核对。

现有专门指南：[[megatron_lm_profiling_debug_guide|Megatron-LM 训练引擎 Profiling & Debug 实践指南]]。

### 6.2 SGLang Serving

```text
请求 → 入口/排队 → tokenizer / scheduler → prefill
    → decode 循环与 KV cache → 返回
```

SGLang 的 HTTP 在线测试、离线 Engine 测试、静态单 batch 测试属于不同测量边界，不能互相代替。

- **TTFT**（Time to First Token）：首 token 到达延迟，必须声明客户端或服务端计时起点。
- **ITL**（Inter-Token Latency）：相邻输出 token 的间隔。
- **TPOT**（Time per Output Token）：按定义口径统计的平均输出 token 时间。

首 token 慢先区分排队、输入处理、prefill 和传输；输出卡顿则关联 decode batch、请求混合、KV 长度、调度与 GPU 活动。不要仅凭 KV cache 占用高推断低效；应同时观察命中/淘汰、可容纳请求、排队和延迟。

SGLang 支持 `/start_profile` / `/stop_profile` 等入口。若为调试而关闭 CUDA Graph，结论只适用于新的诊断配置；必须回到生产配置做端到端验证。

**目标建议：在固定请求分布与延迟约束下，提高有效吞吐，而不是无条件堆高排队量。**

### 6.3 AReaL / RL Training Engine

```text
请求准备 → rollout → 工具/环境/reward → 样本整理与传输
                           ↘ training / weight sync ↗
```

这是概念图，不规定所有 RL 框架都按这个串行流程实现。异步训练应建立两条可关联的时间线：

- **任务时间线**：request、rollout、reward、train batch、weight version、长尾与反压。
- **设备时间线**：对应 GPU 计算、传输、通信、显存与空隙。

AReaL 的 `perf_tracer` 可用于阶段追踪、转换 Chrome Trace；但框架 range 不能直接替代 kernel 执行时间。需要结合 GPU profiling。

例如生产速率 $\lambda_r=120$ 样本/s，消费速率 $\lambda_t=80$ 样本/s，在无丢弃、无反压的假设下：

$$
\frac{dq}{dt}=\lambda_r-\lambda_t=40\ \mathrm{samples/s}.
$$

如果训练消费速率没变，单纯加快 rollout 不一定改善最终 RL 训练目标，反而可能扩大队列与数据陈旧问题。建议额外观察有效样本、丢弃率、权重版本差、尾部 rollout、达到目标质量的 wall time。

**VLM 扩展**：把分辨率、图片/视频数量、vision token 数、文本序列长度显式放入 trace 元数据，并区分 ViT、projector 和 LLM。输入量不同可能首先解释耗时不同，不能未经归一化就宣布性能回退。

## 7. 一套可以复用的性能排查方法

**Step 1：定义最终目标。** 明确 step time、tokens/s、TTFT/ITL、质量目标，记录模型、dtype、shape/长度分布、batch、并行布局、硬件拓扑、缓存与 CUDA Graph 状态。

**Step 2：无 Profiler 基线。** 预热后多次记录稳定窗口；区分编译、加载、graph capture、checkpoint 与业务执行，保留方差和 P95/P99 等有意义的分布。

**Step 3：小范围采集。** 先加阶段标记，定位候选时间窗；再选择 PyTorch Profiler、Nsight Systems、Memory Snapshot、NCCL 或 CPU 工具。不要首次就全 rank、全 step、全 stack/shape/allocation 采集。记录一次采集扰动：

$$
\mathrm{Overhead}_{\mathrm{obs}}=
\frac{T_{\mathrm{profile}}-T_{\mathrm{baseline}}}
     {T_{\mathrm{baseline}}}.
$$

该比值只对可比条件下的具体采集窗口成立，不能当作工具固定开销。

**Step 4：构建可证伪假设。** 例如“decode kernel 前的固定空隙与 CPU 调度区间一致；降低提交/调度开销后，GPU 空隙和客户端 ITL 应同步改善”。单凭相关不能认定因果，要做控制变量或干预实验。

**Step 5：对照实验验证。** 在相同工作负载上同时检查数值正确性、端到端性能及相关 Trace。局部优化不能直接等于全局收益。例如只有 20% wall time 的阶段加速 2 倍，且其他阶段不变，则：

$$
T_{\mathrm{new}}=0.8T_{\mathrm{old}}+0.1T_{\mathrm{old}}
=0.9T_{\mathrm{old}},\qquad
\mathrm{Speedup}\approx 1.11.
$$

真实系统的并发、资源竞争和依赖可能让结果偏离这条简化的 Amdahl 推导。

**三种高频误判**：
- 看到 GPU util 低就优化 kernel，而不看 CPU 提交/等待。
- 看到 NCCL 占比高就换网卡，而不看 rank 到达与 overlap。
- 看到 Reserved 高就认为碎片化，而不看 allocator 历史与活跃对象。

## 8. 入门练习：从可复现例子理解工具

配套实验脚本位于本项目下的 `Lab/`，不是生产 benchmark：

- [single_device_lab.py](Lab/single_device_lab.py)：CPU/GPU timeline、memory history、不同 shape GEMM；
- [two_rank_lab.py](Lab/two_rank_lab.py)：两 rank 人为迟到、Gloo/CPU 或 NCCL/CUDA 的 collective Trace。

```bash
# 时间线：先采集无 profiler 的基线
python Lab/single_device_lab.py --device cpu --mode timeline \
  --capture none --steps 100 --output out/timeline_baseline

# 在 CUDA 环境观察人为的 CPU 提交前等待
python Lab/single_device_lab.py --mode timeline --capture torch \
  --delay-ms 5 --output out/timeline_delay

# CUDA 显存历史：需要 CUDA 版 PyTorch + 足够空闲显存
python Lab/single_device_lab.py --mode memory --output out/memory

# CUDA GEMM：比较不同 M 下的执行
python Lab/single_device_lab.py --mode gemm --dim 1024 \
  --m-values 1 32 1024 --output out/gemm

# 两 rank：无 GPU 环境先用 Gloo 验证逻辑
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  Lab/two_rank_lab.py --cpu --steps 4 --slow-step 2 \
  --delay-ms 20 --numel 1024 --output out/rank
```

**Nsight Systems 采集示例（CUDA 路径尚未实测）**：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=out/timeline_nsys \
  python Lab/single_device_lab.py --mode timeline --capture nsys \
    --delay-ms 5 --output out/nsys_metadata
```

**运行限制和安全提示**：
- 代码采用 Python 3.10+ 语法，需 PyTorch；Memory Snapshot 使用下划线 API，核对 PyTorch 版本。
- 在当前整合环境中已进行 `py_compile`、CPU timeline 和两进程 Gloo/AllReduce 校验；**没有 GPU，CUDA/NCCL/NSYS 分支未经过真实设备验证**。
- `run.json` 中带 profiler 的墙钟时间包括采集/导出干扰，不代表真实性能基线。
- 显存实验为了阶段状态明确而显式同步；双 rank 实验加入 barrier 与 host sleep。这些是教学条件，不能移植到真实性能关键路径当作无扰动测量。
- Trace 可能含路径、机器名、调用栈和请求标识；外发前清理敏感数据。不要加载来源不可信的 pickle 快照。

建议的递进学习路径：

| 阶段 | 目标 | 必须能够解释 |
|---|---|---|
| A 异步与 Timeline | CPU launch、stream、同步、事件 | 为什么 CPU 返回早于 GPU 完成 |
| B 内存与 Kernel | allocator、显存峰值、理想/实测流量 | 为什么 Reserved 高不一定是内存泄漏 |
| C 通信与分布式 | collective、rank、等待传播、时钟 | 为什么最长 NCCL 区间不一定是网络瓶颈 |
| D 真实框架 | Megatron / SGLang / RL trace 与 A/B | 为什么某项局部优化没有提升真实吞吐 |

## 9. References（以原始官方/开源资料为主）

以下是对应概念的资料入口；本次整合不对链接所指的每个历史版本做逐项重测。

**原理与平台**
- [NVIDIA CUPTI Documentation](https://docs.nvidia.com/cupti/main/main.html)：Activity API、相关性标识与硬件数据采集。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：时间线、NVTX、CUDA/NCCL 分析。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：性能计数器、replay、kernel 分析。
- [NVIDIA Roofline 介绍](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)：算术强度与性能上界。
- [Perfetto Clock Sync](https://perfetto.dev/docs/concepts/clock-sync)：分布式时钟与 trace 同步。
- [Linux perf tutorial](https://perfwiki.github.io/main/tutorial/)；[py-spy](https://github.com/benfred/py-spy)；[BCC](https://github.com/iovisor/bcc)；[Memray](https://bloomberg.github.io/memray/overview.html)。

**PyTorch / 设备与通信**
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html)：CUDA 异步与测量。
- [PyTorch Profiler](https://docs.pytorch.org/docs/2.8/profiler.html)：活动、采集选项与开销。
- [PyTorch Memory](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html)：分配器快照与能力边界。
- [PyTorch HTA](https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html)；[NCCL Tests Performance](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)；[RDMA perftest](https://github.com/linux-rdma/perftest)。
- [DCGM User Guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)。

**LLM 框架**
- [Megatron-Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。
- [Megatron-Bridge Profiling](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/profiling.html)；[Nsight 分析 Pitfalls](https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/nemo-mbridge-perf-nsys-analysis/references/pitfalls.html)。
- [SGLang Benchmark and Profiling](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html)；[SGLang Production Metrics](https://docs.sglang.ai/references/production_metrics.html)。
- [AReaL Profiling Best Practices](https://github.com/areal-project/AReaL/blob/main/docs/en/best_practices/perf_profiling.md)；[AReaL 论文](https://arxiv.org/html/2505.24298v5)。

## 10. 后续验证 TODO

- [ ] 在明确的 NVIDIA GPU + PyTorch/CUDA/Nsight 版本上执行全部 CUDA/NCCL 实验，记录环境、基线与 profiler 扰动。
- [ ] 选一个真实 Megatron step，对齐 step → NVTX → NCCL/GEMM → rank 时序，并与现有 Megatron 专项指南交叉检查。
- [ ] 选一个真实 SGLang 请求分布，区分服务端/客户端 TTFT、ITL 与 GPU 活动。
- [ ] 针对 AReaL 的调度 Trace，验证任务区间到设备活动的可追踪性。
- [ ] 核对各版本的 CUPTI、NCCL profiler plugin、PyTorch Memory Snapshot API 差异，补充实测截图而不是教学图。
