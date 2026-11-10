---
ai_generated: true
aliases:
- SimAI Megatron Training Simulation
- SimAI 技术架构与 Megatron 训练仿真
created: 2026-11-09
domains:
- ai-infra
- distributed-training
- performance-modeling
- simulation
reviewed: false
status: seed
type: knowledge
updated: 2026-11-09
---

# SimAI 技术架构调研与 Megatron-LM 训练性能仿真方案

> [!warning] 本文为 AI 辅助源码调研与工程设计，尚未完成人工核验。论文结果、固定版本源码已实现能力、工程推导和建议验收目标分别说明。本文没有实际运行 SimAI、SimuMax 或 Megatron 训练实验，PoC 结果均为待执行目标。

相关总览：[[大规模 LLM 训练性能建模与仿真技术调研]]。本文进一步聚焦 SimAI / SimuMax 的源码架构与 Megatron Core 适配设计。

## 1. Executive Summary

### 1.1 核心结论

**SimAI 最强的部分是集群网络与 collective 仿真，不是 Megatron 模型执行语义或训练显存建模。** 它把训练 workload 转换成计算延迟和通信操作，再通过 SimCCL 将部分 NCCL collective 分解成 rank-to-rank flow，交给 ASTRA-sim 和 NS-3 推进。这样可以研究网络拓扑、链路竞争、拥塞控制、NIC placement 和 collective 算法对训练时间的影响。

**SimuMax 更适合快速回答模型、并行策略、计算成本、pipeline schedule 和显存容量问题。** 它提供模型、策略、系统三类配置，结合 FLOPs、shape-specific efficiency、通信成本与 activation lifetime，输出 step time、吞吐和显存分析。它的网络抽象没有 SimAI NS-3 那么细。

两者不是同一抽象层次的直接竞品：

- SimAI 是由 workload generator、计算画像、collective-to-flow 转换、系统事件调度和网络仿真组成的工具生态。
- SimuMax 是配置驱动的训练性能分析与轻量 schedule simulation 系统。
- SimAI 在网络拓扑和通信算法研究上更强；SimuMax 在计算/显存/策略快速评估上更完整。
- 两者的公开精度数据来自不同模型、硬件和验证方法，不能据此宣布某一个整体更准确。

**推荐路线：以 SimAI 作为网络仿真后端，以 Megatron Core 0.20.0 适配器作为 workload 前端，以真实 MCore profiling 作为计算成本来源；同时用 SimuMax 做计算、显存和策略预测的交叉校验。**

不要把 AICB 当作最终 Megatron 前端。它适合快速 PoC 和理解 SimAI workload 格式，但当前固定版本与 Megatron Core 0.20.0 之间存在明显语义缺口。

### 1.2 关键发现

1.  SimAI 主仓库当前核查基线为 2026-09-20 的 commit；其 AICB submodule 固定在 2025-12-27，明显早于 Megatron Core 0.20.0。
2.  AICB 的 SimAI training workload generator 没有 CP 参数传播；`dp_num` 只按 `world_size / (TP × PP)` 计算。ASTRA workload 也没有 CP communicator。
3.  AICB 把 `vpp` 写成 PP stage 本地层数，隐含”一层一个 virtual pipeline chunk”的假设；没有通用的 `num_layers_per_virtual_pipeline_stage` 映射。
4.  AICB 的 PP 通信是 header 中的聚合 `pp_comm`，ASTRA 报告阶段用公式估计，不是每个 microbatch 的 P2P flow。
5.  当前 ASTRA Workload parser 读取 `depen` 字段，但只用于日志输出；调度主要依赖固定的 Forward / Input-Grad / Weight-Grad 状态机，不是通用 DAG 执行器。
6.  AIOB 使用自建的 PyTorch mock 模型测量时间，不等同于实际 Megatron Core / Transformer Engine 的 kernel profile。通用 Megatron 路径使用 MHA 形状；MoE 通信代码注释明确基于 Megatron Core v0.13.0。
7.  SimAI 当前训练 workload 没有完整的 model-state / activation lifetime memory trace。Vidur 侧的推理显存能力不能直接当成训练显存能力。
8.  SimCCL v2.30 mock 支持一组特定 NCCL collective / algorithm / protocol 组合，不是完整 NCCL 源码的替代品。
9.  SimuMax v1.2 已有 CP A2A、MoE、显存和 pipeline schedule 模型，但 DP overlap 仍有 TODO；正式 B200 结果集中在单节点 8 GPU 和特定缩层模型。
10. Megatron Core 0.20.0 新增/强化了 GTP-remat、hierarchical/dynamic CP、ProcessGroupCollection 等语义。适配器应读取实际 process group 成员，而不是只根据 TP/PP/DP/EP/CP 数值推导。

### 1.3 技术选型结论

| 目标                                                 | 首选工具                                      | 原因                                     |
|:-----------------------------------------------------|:----------------------------------------------|:-----------------------------------------|
| 快速比较 Dense 模型 TP/PP/DP、显存和吞吐             | SimuMax                                       | 配置简单，计算与显存模型完整度较高       |
| 比较 NCCL collective、网络拓扑、NIC、拥塞和 rail     | SimAI NS-3                                    | 能将 collective 转成 flow 并模拟网络     |
| 快速做网络敏感性初筛                                 | SimAI Analytical                              | 成本低，但带宽模型较粗                   |
| 复现当前 Megatron Core 0.20.0 的 schedule / CP / MoE | 自研 Megatron Adapter                         | AICB 当前版本不能直接表达完整语义        |
| 长期建设可扩展的全栈系统                             | Megatron IR + ASTRA/NS-3 + SimuMax cost model | 统一工作负载，分离计算、通信、内存与调度 |

------------------------------------------------------------------------

## 2. 版本基线与研究边界

### 2.1 固定版本

以下 SHA 用于本报告源码结论。复现时应使用固定 SHA，不要在初始化 submodule 时无条件跟随远端 `master`。

| 项目                                                                                                           | 固定版本 / Commit                                           | 日期 UTC   | 在本方案中的角色        |
|:---------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:-----------|:------------------------|
| [SimAI](https://github.com/aliyun/SimAI/tree/a57d765c5478f319c5f5ab6fdc53f9cc7f461743)                         | `a57d765c5478f319c5f5ab6fdc53f9cc7f461743`                  | 2026-09-20 | 主仿真系统              |
| [AICB](https://github.com/aliyun/aicb/tree/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f)                           | `23eec3c48ca2d2d93dd888a4c7b22ab4421e782f`                  | 2025-12-27 | Workload / AIOB         |
| [SimCCL](https://github.com/aliyun/SimCCL/tree/fd7cd57d16f9bd42e3ccb70911c977e18ec294b9)                       | `fd7cd57d16f9bd42e3ccb70911c977e18ec294b9`                  | 2026-08-12 | NCCL collective-to-flow |
| [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud/tree/3e0c7c1bfbbe9f77890ddcf5e5b9c79fc6dd7437) | `3e0c7c1bfbbe9f77890ddcf5e5b9c79fc6dd7437`                  | 2026-08-12 | 网络仿真后端            |
| [SimAI-CLEM](https://github.com/aliyun/SimAI-CLEM/tree/4a91f8b655224c0b1d1c300aa09c7058f4da2707)               | `4a91f8b655224c0b1d1c300aa09c7058f4da2707`                  | 2026-09-20 | 可选 CCL emulator       |
| [SimuMax](https://github.com/MooreThreads/SimuMax/tree/4dcaae513ba7679d56c116a17cd4a691ec41e252)               | `4dcaae513ba7679d56c116a17cd4a691ec41e252`                  | 2026-05-11 | 对照分析模型            |
| [Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/685fb8bd970185e9ac3283fe32a5d6667bf4b268)           | `core_v0.20.0` / `685fb8bd970185e9ac3283fe32a5d6667bf4b268` | 2026-10-27 | 目标训练框架            |

资料截止：2026-11-09。SimAI 主仓库的最新核查 commit 为 2026-09-20；AICB submodule 比主仓库旧约九个月。Megatron Core 以 0.20.0 正式 tag 为基线。

### 2.2 SimAI 仓库的依赖结构

SimAI 的 `.gitmodules` 包含四个 submodule：

- `aicb`
- `SimCCL`
- `ns-3-alibabacloud`
- `SimAI-CLEM`

`astra-sim-alibabacloud` 是 SimAI 仓库内的 ASTRA-sim 扩展代码，不是独立 submodule。`vidur-alibabacloud` 是另一条推理 workload / serving 仿真路径，不属于本次训练仿真主链路。

复现时应执行：

``` bash
git checkout a57d765c5478f319c5f5ab6fdc53f9cc7f461743
git submodule update --init --recursive
```

不要直接照搬旧教程中的 `git submodule update --remote` 作为可复现安装步骤。`--remote` 会把 submodule 移动到远端分支的新提交，破坏本报告所依据的版本组合。

------------------------------------------------------------------------

## 3. 数学符号表

| 符号                           | 含义                                                                                   |
|:-------------------------------|:---------------------------------------------------------------------------------------|
| $N$                            | 参与训练的 GPU 总数                                                                    |
| $d_t,d_p,d_d,d_c,d_e,d_{et}$   | TP、PP、DP、CP、EP、Expert TP 并行度                                                   |
| $d_{ed}$                       | Expert DP 并行度                                                                       |
| $d_g$                          | 唯一训练样本的数据分发组大小；常规情况为 DP，启用 GTP-remat 时还需考虑相应数据分发维度 |
| $B_g,b,m$                      | Global batch、micro-batch size、每个 iteration 的 micro-batch 数                       |
| $S,H,L$                        | Sequence length、hidden size、Transformer layer 数                                     |
| $P_r$                          | 某 rank 持有的参数量                                                                   |
| $d_s$                          | optimizer state 的分片组大小                                                           |
| $F_i,Q_i$                      | 操作 $i$ 的 FLOPs、所建模存储层级的数据流量                                            |
| $\Pi_{\rm peak},\Pi_{\rm eff}$ | GPU 峰值与有效计算吞吐                                                                 |
| $W_{\rm HBM},W_{\rm net}$      | 有效 HBM 带宽、网络带宽                                                                |
| $\eta_i,\ell_i$                | 算子效率系数、固定启动开销                                                             |
| $G=(V,E)$                      | 执行 DAG，$V$ 为操作节点，$E$ 为依赖边                                                 |
| $s_i,f_i,\tau_i$               | 操作开始时间、完成时间、执行时长                                                       |
| $q,D,\alpha$                   | Collective group size、每 rank 输入数据量、通信启动延迟                                |
| $T_{\rm step}$                 | 稳态训练 iteration 时间                                                                |
| $X_{\rm tok}$                  | Token throughput                                                                       |
| $M_r(t),M_{\rm peak}$          | Rank $r$ 在时刻 $t$ 的显存占用、全 rank 最大峰值                                       |
| $E_x,k,n_e$                    | MoE expert 数、router top-k、expert $e$ 接收的 token 数                                |
| $n_{r,s}$                      | 从源 rank $r$ 发往目标 rank $s$ 的 token-expert assignment 数                          |
| $D_{r,s}$                      | Rank $r$ 到 rank $s$ 的实际通信数据量                                                  |
| $\rho$                         | MoE expert 最大负载与平均负载之比                                                      |
| $\epsilon_j$                   | 第 $j$ 个验证配置的绝对相对误差                                                        |
| $J$                            | 验证配置总数                                                                           |

除特别说明外，时间单位为秒，数据量为字节，带宽为 byte/s。GB 与 GiB、Gb/s 与 GB/s 必须区分。

------------------------------------------------------------------------

## 4. SimAI 整体技术架构

### 4.1 架构图

SimAI 的训练仿真主链路可以分成五层：

``` mermaid
flowchart TD
    A["模型配置 / 并行配置 / Batch / Sequence"] --> B["AICB Workload Generator"]
    B --> C["Workload Text"]
    B --> D["AIOB Compute Profile"]
    D --> C
    C --> E["ASTRA-sim Workload Parser"]
    F["SimCCL: Collective → P2P Flow"] --> G["Flow Models"]
    E --> H["ASTRA-sim System / Layer / Event Scheduler"]
    G --> H
    I["GPU / NIC / Switch / Link Topology"] --> J["NS-3 Network Backend"]
    K["Analytical Bus-BW Model"] --> H
    J --> H
    H --> L["Iteration / Layer / Communication Statistics"]
    L --> M["Step Time / Throughput / Bottleneck Analysis"]
```

其中计算和网络成本是两条独立输入：

- 计算成本来自 AIOB 或外部 profile database。
- 通信成本来自 ASTRA Analytical 或 SimCCL + NS-3。
- ASTRA-sim 负责把计算延迟、通信事件和 workload 状态机放到同一时间轴。

因此，SimAI 不会在每个模拟 rank 上真的执行 Transformer kernel。它执行的是预先抽象好的计算延迟和通信事件。

### 4.2 模块依赖图

``` mermaid
flowchart TD
    S["SimAI Repository"]
    S --> A["AICB"]
    S --> X["ASTRA-sim AlibabaCloud"]
    S --> C["SimCCL"]
    S --> N["ns-3 AlibabaCloud"]
    S --> E["SimAI-CLEM"]
    S --> V["Vidur AlibabaCloud"]

    A --> W["Training Workload"]
    A --> P["AIOB Compute Profile"]
    W --> X
    P --> W
    C --> F["Collective Flow Models"]
    F --> X
    X --> N

    E --> N
    V --> I["Inference / Serving Simulation"]
```

`Vidur` 属于推理服务模拟链路。它不应被当作 SimAI 训练显存模型的一部分。`SimAI-CLEM` 是可选 CCL emulator，也不是一般训练 workload 的必需依赖。

### 4.3 模块职责与可替代性

| 模块               | 当前职责                            | 必要性                           | 独立运行                   | 替代方案                                 | 替换难度 |
|:-------------------|:------------------------------------|:---------------------------------|:---------------------------|:-----------------------------------------|:---------|
| AICB               | Mocked model、workload 生成         | 官方训练路径需要；不是系统硬依赖 | 可单独生成 workload        | Megatron Adapter、Chakra trace           | 中高     |
| AIOB               | GPU 上测量模块计算时间              | 可选；需要计算画像时使用         | 可独立 profile             | MCore profiler、PyTorch Profiler、Nsight | 中       |
| SimCCL             | Collective 选择与 P2P flow 分解     | SimAI NCCL flow 路径需要         | 有 standalone binary       | NCCL trace parser、自研 decomposer       | 高       |
| ASTRA-sim          | Workload 状态机、事件调度、通信调用 | SimAI 主引擎                     | 可独立使用                 | 其他 DES 引擎                            | 很高     |
| Analytical backend | 估算 collective 时间                | Analytical 模式需要              | 不依赖 NS-3                | SimuMax network model、自研公式          | 低中     |
| NS-3 backend       | Packet / queue / congestion 仿真    | NS-3 模式需要                    | 可作为网络模拟器           | SST、OMNeT++、其他 flow/packet simulator | 高       |
| Topology generator | 生成预置网络拓扑文件                | 非必要                           | 可单独生成                 | 自定义拓扑文件生成器                     | 中       |
| SimAI-CLEM         | 用少量真实 GPU 做 CCL 网络仿真/诊断 | 可选专项                         | 需实际 GPU 和 patched NCCL | nccl-tests、真实 CCL benchmark           | 高       |
| Vidur              | Inference / serving workload        | 训练不需要                       | 可独立                     | 其他 serving simulator                   | 不适用   |

### 4.4 三种网络相关运行模式

| 模式                  | 主要执行路径                           | 主要用途                    | 主要限制                                  |
|:----------------------|:---------------------------------------|:----------------------------|:------------------------------------------|
| SimAI-Analytical      | Workload → ASTRA Analytical            | 快速估算通信和训练时间      | 有效带宽与 overlap 参数驱动，网络抽象较粗 |
| SimAI-Simulation      | Workload + SimCCL flows → ASTRA + NS-3 | 拓扑、队列、拥塞、NIC、rail | CPU 仿真开销更高；依赖 flow 映射正确      |
| SimAI-Physical / CLEM | 真实 CCL / 物理网络或 emulation        | CCL 诊断和真实网络行为研究  | 不是通用的大规模训练性能预测器            |

------------------------------------------------------------------------

## 5. SimAI 核心模块源码分析

### 5.1 AICB：训练 workload 生成

主要源码：

- [SimAI_training_workload_generator.py](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/workload_generator/SimAI_training_workload_generator.py)
- [MockedMegatron.py](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/workload_generator/mocked_model/training/MockedMegatron.py)
- [AiobMegatron.py](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/workload_generator/mocked_model/training/AiobMegatron.py)
- [utils.py](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/utils/utils.py)

#### 5.1.1 Mocked Model

AICB 通过自建模型模块描述 Transformer：

``` text
MockedMegatron
  ├── Embedding
  ├── Attention
  │    ├── QKV Linear
  │    ├── Attention
  │    └── Output Linear
  ├── MLP
  │    ├── Column Parallel Linear
  │    └── Row Parallel Linear
  ├── MoE
  │    ├── Router
  │    ├── Permutation / Dispatch
  │    ├── Expert GEMM
  │    └── Combine
  └── Grad / Parameter Operations
```

Mocked Model 的价值是构造 shape、模块顺序和通信模式，而不是执行一个完整的生产 Megatron 模型。

通用 AIOB attention 路径中：

``` python
self.num_query_groups_per_partition = (
    self.num_attention_heads_per_partition
)
```

因此它实际采用 MHA 形状。它有独立的 DeepSeek MLA 路径，但不能据此推断通用 Qwen GQA、任意 MLA 或 VLM 结构都已支持。

#### 5.1.2 AIOB 计算画像

AIOB 的 `AiobMegatron.py` 使用自己的 PyTorch mock model 和 `cuda_timing_decorator` 测量模块执行时间。

流程：

1.  解析模型参数。
2.  构建 mock 模型。
3.  在 GPU 上执行对应 forward / backward 操作。
4.  将测量结果写入 compute profile。
5.  `extract_averages()` 读取 profile。
6.  `SIMAI_workload_generator` 把成本填入 workload。

`--comp_filepath` 可以复用既有计算画像，避免每次重新 profile。

AIOB 的优点是可以快速获得 shape 相关 GPU 时间；限制是其模块实现不等于目标 Megatron Core / TE kernel 实现。对于当前 MCore 0.20，建议把 AIOB 当作起点，最终换成真实 MCore 运行时的 profile database。

#### 5.1.3 Workload 文件格式

当前 AICB 生成的 SimAI 文本 workload 头部包含：

``` text
HYBRID_TRANSFORMER_FWD_IN_BCKWD
model_parallel_NPU_group: ...
ep: ...
pp: ...
vpp: ...
ga: ...
all_gpus: ...
checkpoints: ...
checkpoint_initiates: ...
pp_comm: ...
```

每个 workload row 主要描述：

| 字段                  | 含义                      |
|:----------------------|:--------------------------|
| name                  | 模块名称                  |
| depen                 | dependency placeholder    |
| forward_compute_time  | 前向计算时间              |
| forward_comm_type     | 前向通信类型              |
| forward_comm_size     | 前向通信数据量            |
| backward_compute_time | 输入梯度阶段计算时间      |
| backward_comm_type    | 输入梯度通信类型          |
| backward_comm_size    | 输入梯度通信数据量        |
| dp_compute_time       | 权重梯度阶段计算时间      |
| dp_comm               | 权重梯度通信类型          |
| dp_comm_size          | 权重梯度通信数据量        |
| process_time          | 通信完成后的 update delay |

这是一种固定字段的 Layer 级描述，不是通用的 per-rank DAG。

#### 5.1.4 当前 AICB 对 Megatron Core 0.20 的缺口

**CP 没有进入 SimAI workload generator。**

在固定 AICB commit 中，`context_parallel_size` 的唯一搜索命中位于其他 workload applyer 代码，不在 `SimAI_training_workload_generator.py` 的生成路径中。

`utils.py` 的 DP 推导是：

``` python
args.dp_num = args.world_size // (
    args.tensor_model_parallel_size
    * args.pipeline_model_parallel
)

args.num_microbatches = (
    args.global_batch
    // (args.dp_num * args.micro_batch)
)
```

没有 CP 维度。SimAI generator 也没有把 CP 转换为独立 communicator 或通信事件。

**VPP 被简化为本地层数。**

`dump_file()` 写入：

``` python
vpp = self.args.num_layers
```

而 `num_layers` 在 PP 配置下已按 stage 分割。这相当于假设每个 virtual pipeline chunk 只包含一层，即：

$$
VPP = \frac{L}{PP \cdot L_{\rm per\ virtual\ chunk}}
$$

且固定 $L_{\rm per\ virtual\ chunk}=1$。生成器没有通用的 `num_layers_per_virtual_pipeline_stage` 映射。

**PP 通信只有聚合大小。**

AICB 写入：

``` python
pp_comm_value = (
    2 * micro_batch * seq_length * hidden_size
)
```

如果开启 SP，则除以 TP size。这个 `2` 隐含 BF16/FP16 的两字节元素宽度，没有从 dtype 自动读取。PP 通信通过 header 中的一个聚合数值表示，不是每个 microbatch 的 P2P send/recv。

**MoE 通信使用旧版语义。**

AICB 源码注释明确写明其 MoE communication based on Megatron Core v0.13.0。它能构造 EP AllToAll 和相关 TP 操作，但不能直接代表 MCore 0.20 的所有 dispatcher、DeepEP v2 和动态 routing 行为。

**Dependency 字段不是任意 DAG。**

ASTRA 的 `Workload.cc` 读取 `depen`，但在当前 parser 中主要用于输出日志。执行依赖来自固定的 workload state machine，而不是对任意 dependency graph 的通用调度。

**Optimizer 不是完整 AdamW 模型。**

`Work_Item.process_time` 默认值为 100。ASTRA `Layer` 将它作为通信完成后的 update delay。AIOB `Grad_param` 主要进行梯度缩放与分片操作，不是完整的 AdamW optimizer step benchmark。因此必须额外测量 optimizer kernel 和状态更新。

### 5.2 ASTRA-sim：系统事件调度

核心源码：

- [Workload.cc](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/workload/Workload.cc)
- [Layer.cc](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/workload/Layer.cc)
- [Sys.cc](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/system/Sys.cc)

ASTRA 主要由三部分构成：

| 模块     | 职责                                               |
|:---------|:---------------------------------------------------|
| Workload | 解析 workload，维护训练状态和层序列                |
| Layer    | 维护前向、输入梯度、权重梯度的 compute / comm 状态 |
| Sys      | 维护系统事件、collective 生成和通信完成回调        |

一层的固定状态流程：

``` mermaid
flowchart TD
    A["Forward Compute"] --> B["Forward Communication"]
    B --> C["Input-Grad Compute"]
    C --> D["Input-Grad Communication"]
    D --> E["Weight-Grad Compute"]
    E --> F["Weight-Grad Communication"]
    F --> G["Next Layer / Iteration"]
```

通信可以使用 Blocking / Non-Blocking barrier。非阻塞通信完成后通过 callback 通知 Workload，推进后续状态。

这能模拟固定 workload 内的计算与通信重叠，但不能自动还原任意 MCore stream/event DAG。

### 5.3 Analytical Backend：带宽模型

当前代码中的主要参数包括：

- GPU 类型。
- NVLink 带宽。
- 每 NIC 带宽。
- 每服务器 NIC 数量。
- 每服务器 GPU 数量。
- Collective group size。
- Collective 类型。

对于大消息，代码按 Ring 类模型估算：

$$
T_{\rm AR}
\approx
\frac{2D}{B_{\rm bus}}
\frac{q-1}{q}
$$

$$
T_{\rm AG/RS/A2A}
\approx
\frac{D}{B_{\rm bus}}
\frac{q-1}{q}
$$

其中 $D$ 为每 rank 输入数据量，$q$ 为 group size，$B_{\rm bus}$ 为有效 bus bandwidth。

当前代码对小于 1 MiB 的消息使用按 rank 数固定的 latency 表。大消息则使用 `cal_busbw()` 和数据量 / 节点数修正表。

该模式有三个重要边界：

1.  它不是任意拓扑图仿真，rank placement 由简化的 GPU / NIC / 节点参数推断。
2.  DP 分支在 TP 位于节点内时，将 DP group size 作为 `cal_busbw()` 的节点数量参数，隐含 DP ranks 分散在不同节点的假设。多个 DP ranks 同处一个节点时，应检查其结果。
3.  当前开源 binary 不支持旧教程中的 `-busbw busbw.yaml` 用户自定义文件。官方中文 Tutorial 已明确说明该 flag 未被当前 parser 接受。

### 5.4 Compute / Communication Overlap

Analytical 模式使用 overlap ratio 估计暴露通信时间：

$$
T_{\rm exposed}
=
T_{\rm comm}(1-r_{\rm overlap})
$$

分别存在 TP、DP、EP、PP overlap 参数。

这不是从真实执行图自动推导的 overlap。它是输入参数。

NS-3 模式则允许通信事件异步推进，计算和通信在时间轴上重叠。但计算成本仍是预先提供的固定时长，不能自动建模 GPU kernel 与通信并发时的 SM / HBM 资源干扰。

因此应区分：

- **逻辑重叠**：操作依赖允许并发。
- **网络竞争**：多个 flow 竞争 NIC / link / queue。
- **GPU 资源竞争**：计算 kernel 和通信路径争用 SM、HBM、copy engine。

SimAI 当前最强的是前两项，第三项需要额外校准或扩展。

### 5.5 PP 的特殊处理

在 `Layer::report()` 中，PP 暴露通信时间通过 `pp_commsize`、`pp_overlap_ratio`、`GA` 和 `VPP` 估算；pipeline bubble 则通过聚合公式计算：

$$
T_{\rm bubble}
\propto
T_{\rm pre}
\frac{PP-1}{GA \cdot VPP}
$$

这不是完整的 per-rank 1F1B / interleaved schedule 回放。

因此 SimAI 当前 workload 中的 PP 结果适合做粗粒度参考，不应直接用于比较复杂 VPP、非均匀 stage 或跨节点 PP placement。

### 5.6 SimCCL：Collective-to-Flow Translator

核心源码与文档：

- [SimCCL README](https://github.com/aliyun/SimCCL/tree/fd7cd57d16f9bd42e3ccb70911c977e18ec294b9)
- [Architecture](https://github.com/aliyun/SimCCL/blob/fd7cd57d16f9bd42e3ccb70911c977e18ec294b9/docs/design/architecture.md)
- [SimAI integration](https://github.com/aliyun/SimCCL/blob/fd7cd57d16f9bd42e3ccb70911c977e18ec294b9/docs/integration/integration-with-simai.md)

主要流程：

``` mermaid
flowchart TD
    A["Collective: type / size / group / ranks"] --> B["Algorithm & Protocol Selection"]
    B --> C["Ring / NVLS / PAT / Protocol"]
    C --> D["Flow Model Generation"]
    D --> E["Per-rank Send / Recv"]
    E --> F["Channels / Chunks / Peer Mapping"]
    F --> G["ASTRA / NS-3"]
```

当前公开 v2.30 mock 支持范围：

| Collective    | 当前可用算法 |
|:--------------|:-------------|
| AllReduce     | Ring、NVLS   |
| AllGather     | Ring、PAT    |
| ReduceScatter | Ring、PAT    |
| AllToAll      | Ring         |
| Broadcast     | Ring         |

Tree、CollNet 等不能仅凭 ASTRA 延迟表中存在相应条目就认定为完整可用算法。延迟表包含 Ring、Tree、CollNet、NVLS、NVLS_TREE、PAT 等组合，但 SimCCL 的自动选择和 flow 生成范围更窄。

PAT 的自动选择还有特定约束，例如多节点、每节点单 GPU、AllGather / ReduceScatter 和消息大小门槛。

SimCCL v2.30 的价值是能够独立生成 flow，不需要真实 GPU 或 NS-3。但它是 NCCL 行为的有限 mock，不是完整 NCCL 的替代实现。

### 5.7 NS-3 网络仿真与拓扑

SimAI NS-3 路径通过 SimCCL flow 和 ASTRA 网络前端连接 NS-3。

当前 `SimAI.conf` 包含 QCN、动态 PFC threshold、packet payload size、rate control 等配置。`entry.h` 还会根据 algorithm、protocol、intra-node / inter-node link type 选择发送延迟表；`AS_SEND_LAT` 可以覆盖默认发送延迟。

拓扑生成入口：

[gen_Topo_Template.py](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py)

预置模板包括 Alibaba HPN、Spectrum-X、DCN+ 和 Rail-Optimized 拓扑。当前内置模板的 pod 生成有单 pod 假设；多 pod 或特殊交换结构需要自定义拓扑文件。

NS-3 模式适合分析：

- NVLink / NIC / leaf-spine 的路径差异。
- 多个 collective 的链路竞争。
- 不同 rail 和 rank placement。
- PFC / QCN / DCQCN 相关网络行为。
- 拥塞对 iteration time 的影响。

它不会自动知道用户集群的真实拓扑。拓扑文件和 rank mapping 必须来自实际集群信息。

### 5.8 SimAI-CLEM、Physical 与 Vidur

**SimAI-CLEM** 是 CCL emulator：在少量真实 GPU 上执行 NCCL / CCL，将网络传输路径接入模拟网络。它适合 CCL 诊断与网络 what-if，不是完整训练 simulator。固定版本 README 以 NCCL 2.23.4 和 8×A100 环境为主要验证条件，MSCCL Executor 仍有开发状态说明。

**SimAI-Physical** 是物理网络流量生成 / 运行路径，需要 MPI 和实际网络环境。它不是无硬件大规模预测的替代品。

**Vidur AlibabaCloud** 属于 inference / serving 模拟路径。SimAI 的推理参数和 KV cache 显存能力不能直接当成训练 activation / optimizer memory 模型。

------------------------------------------------------------------------

## 6. SimAI 与 SimuMax 深入比较

### 6.1 两者的目标不同

| 项目     | SimAI                                           | SimuMax                                                     |
|:---------|:------------------------------------------------|:------------------------------------------------------------|
| 主要问题 | 大规模训练通信和网络拓扑如何影响训练性能        | 模型、并行策略和硬件配置如何影响训练时间与显存              |
| 输入     | workload、compute time、collective、topology    | model config、strategy config、system config                |
| 计算模型 | 外部计算画像 / AIOB 时间                        | FLOPs + shape-specific efficiency + Roofline                |
| 通信模型 | SimCCL flow + ASTRA / NS-3，或 Analytical busbw | 有效带宽、latency、Ring / A2A 解析成本                      |
| 调度     | ASTRA 固定 workload 状态机和事件调度            | PpSchedule / OptimizerSimulator 事件回放                    |
| 显存     | 当前训练 workload 缺少完整显存生命周期          | 模型状态、activation、recompute、rank-local memory timeline |
| 典型用途 | 网络设计、NCCL 算法、拥塞、rail、placement      | 快速配置搜索、MFU、memory feasibility、pipeline 比较        |

### 6.2 建模能力矩阵

| 维度              | SimAI 1.7 当前公开路径                           | SimuMax v1.2 当前公开路径                                        |
|:------------------|:-------------------------------------------------|:-----------------------------------------------------------------|
| Dense Transformer | 支持固定 workload 模板                           | 支持模型配置与计算建模                                           |
| GQA / MQA         | 通用 AIOB 路径按 MHA 处理；其他模型需单独适配    | `kv_head_num` 可表达 GQA                                         |
| MLA               | 有 DeepSeek 专用路径                             | 支持 MLA 配置                                                    |
| MoE               | EP AllToAll，主要为均衡 token 模型               | MoE Router / Permutation / Experts，带统计 tail / straggler 选项 |
| TP                | 支持常见 collective                              | 支持                                                             |
| DP                | 支持 collective，但 placement 有解析假设         | 支持 DP / optimizer shard                                        |
| PP                | `pp_comm` 聚合成本和 bubble 估计                 | 显式 pipeline schedule、P2P、VPP                                 |
| VPP               | `vpp` 隐含一层一个 chunk                         | 支持 interleaving schedule；不同模式验证范围不同                 |
| EP                | 支持部分 EP AllToAll                             | 支持 EP / ETP / EDP 模型                                         |
| CP                | 当前训练链路不支持                               | 支持 CP A2A 配置，正式验证范围有限                               |
| Optimizer         | Grad_param / fixed process delay，不是完整 AdamW | 有 optimizer cost / DP communication model                       |
| 计算通信 overlap  | ratio 或固定时长事件；没有完整 GPU 竞争模型      | schedule 级模拟；DP overlap 在分析路径仍有 TODO                  |
| 网络拓扑          | NS-3 packet / queue / congestion / topology      | 有效带宽和网络层次参数                                           |
| 训练显存          | 缺少完整 training activation lifetime            | 有 model state / activation / memory timeline                    |
| 新 GPU            | 重测 AIOB，或使用简化模型                        | 添加 accelerator config 和 shape efficiency profile              |
| 大规模网络        | 主要优势                                         | 需要简化外推                                                     |

### 6.3 SimAI 的优势究竟在哪里？

**优势一：Collective 不是一个固定时长，而是可以展开为真实网络 flow。**

SimCCL 将 AllReduce / AllGather / ReduceScatter / AllToAll 等操作分解成 rank-to-rank 通信，再由 NS-3 推进。这样可以分析 topology、queue、link sharing 和 congestion。SimuMax 的通信模型主要是公式和有效带宽。

**优势二：网络拓扑和拥塞模型粒度更细。**

SimAI 可研究多 rail、NIC placement、PFC / QCN、链路 oversubscription 和 packet-level 竞争。SimuMax 的网络参数更适合快速估算，不适合替代数据中心网络模拟。

**优势三：有独立的 CCL emulation 路径。**

SimAI-CLEM 可以用少量 GPU 执行真实 CCL 软件，同时把网络行为交给模拟网络。它有助于验证 collective 实现，但不能代替完整训练 workload 模拟。

**优势四：论文有千卡级训练验证。**

SimAI NSDI 2025 论文报告了 A100 / H100 集群上的大规模验证，最高覆盖 1,024 GPU，端到端 iteration time 偏差低于 3.9%，论文整体平均偏差约 1.9%。论文还报告通信时间平均偏差约 3.9%（A100）和 2.3%（H100）。

这些结果属于论文所用 workload、硬件、拓扑、成本画像和代码路径。它们不是 SimAI 1.7 + Megatron Core 0.20.0 + 任意 CP / MoE 配置的精度承诺。

### 6.4 SimuMax 的优势

**优势一：计算模型直接面向模型结构和 shape。**

SimuMax 的 `SystemConfig.compute_op_accuracy_time()` 使用算子 FLOPs、峰值 TFLOPs 和 shape-specific efficiency factor：

$$
T_{\rm compute}
=
\frac{F}
{\Pi_{\rm peak}\eta_{\rm shape}}
$$

有精确 shape profile 时使用 `accurate_efficient_factor`，缺失时回退到默认 efficiency factor。

**优势二：显存模型更完整。**

SimuMax 包含 model weight、gradient、optimizer state、activation cache、recompute 和 rank-local memory timeline。`SimuMemoryTracker` 会记录 static、cached、temporary allocation 的时间变化。

**优势三：Pipeline schedule 显式建模。**

`PpSchedule` 能生成 microbatch / virtual chunk 的执行序列，支持 P2P send / recv 和 interleaving schedule。它还会在 `OptimizerSimulator` 中建立 DP×CP reduce-scatter、expert-DP 通信、optimizer step 和参数 all-gather 事件。

**优势四：适合快速搜索。**

Model / Strategy / System 配置分离，便于批量比较 TP、PP、DP、EP、CP、FP8、recompute、VPP 和不同硬件效率假设。

### 6.5 SimuMax 的限制

1.  `SystemConfig.compute_net_op_time()` 明确标注当前以 Ring 算法为基础；它不是 NCCL 算法和 packet-level 网络模拟器。
2.  网络拓扑主要抽象为 intra-node / inter-node bandwidth、latency、efficiency 和通信系数，无法表达任意交换机图、PFC queue 和动态拥塞。
3.  `PerfLLM._compute_dp_time()` 中仍有 `TODO: support overlap`；`dp_comm_exposed_time` 当前直接等于 DP communication time。配置里存在 `overlap_grad_reduce`，不代表分析路径已经完整模拟 bucket overlap。
4.  v1.2 的正式 B200 结果集中在单节点 8 B200，不能推断多节点拥塞精度。
5.  正式 CP 结果主要是 Dense A2A case；不能推断所有 CP ring / hierarchical / dynamic CP 算法。
6.  VPP 的同步 schedule 有结果，异步 VPP 仍有边界。
7.  当前代码和验证契约以较早 Megatron selective recompute 语义为主，不能视为 MCore 0.20.0 的完整兼容实现。

### 6.6 精度结果如何比较？

| 系统                | 公开验证                           | 结果                                                | 解释                           |
|:--------------------|:-----------------------------------|:----------------------------------------------------|:-------------------------------|
| SimAI NSDI 2025     | A100 / H100，最高 1,024 GPU        | 端到端偏差低于 3.9%，论文整体平均约 1.9%            | 大规模网络和训练 workload 验证 |
| SimAI Compute Model | 已测 GPU 与未测 GPU 外推           | 实测画像路径约 0.5%–3.1%；新 GPU 解析外推约 13%–15% | 计算画像质量影响很大           |
| SimuMax Dense       | 单节点 8 B200，24 个缩层 case      | timing signed error 约 -11.37% 至 -0.18%            | 形状保留的缩层模型             |
| SimuMax MoE         | 单节点 8 B200，12 个缩层 case      | timing signed error 约 -13.54% 至 +0.06%            | DeepSeek V2 / V3 相关 case     |
| SimuMax CP          | 单节点 8 B200，7 个 Dense A2A case | timing signed error 约 -9.27% 至 -5.05%             | CP A2A，不代表所有 CP 算法     |

SimuMax 同一验证集的 allocated memory 误差也较小，但 allocated memory 不等于 NVML process total 或 PyTorch reserved memory。

这两组结果不是 apples-to-apples：硬件、模型、验证范围和误差口径都不同。当前不能从公开数字得出”SimAI 比 SimuMax 更准确”或相反的结论。

### 6.7 两者如何互补？

可以互补，但不是把 SimuMax JSON 文件直接交给 SimAI。

推荐组合：

``` mermaid
flowchart TD
    A["Megatron Core 0.20 Adapter"] --> B["Canonical Per-rank DAG"]
    B --> C["Compute Cost Database"]
    B --> D["Collective Metadata"]
    B --> E["Memory Lifetime"]
    C --> F["ASTRA Event Scheduler"]
    D --> G["SimCCL / Custom Flow Adapter"]
    G --> H["NS-3"]
    H --> F
    E --> I["SimuMax / Custom Memory Model"]
    F --> J["Step Time / Throughput / Bottleneck"]
    I --> K["Peak Memory / OOM Feasibility"]
    J --> L["Cross-validation with SimuMax"]
    K --> L
```

具体分工：

- SimuMax 提供 per-op compute cost、optimizer cost、显存估算和快速策略筛选。
- SimAI / SimCCL / NS-3 提供 collective flow 和网络竞争时间。
- ASTRA 负责统一事件时间线。
- 自研 Megatron IR 负责保留真实 schedule、dependencies、process groups 和 tensor lifetime。

不要同时让 SimuMax 和 ASTRA 各自完整调度同一套事件，否则会重复计算通信和 pipeline 时间。长期方案应只保留一个主事件调度器。

------------------------------------------------------------------------

## 7. Megatron Core 0.20.0 的执行语义与适配设计

### 7.1 为什么不能只读 Megatron 配置？

配置能确定模型结构和理论并行度，却不能完整确定：

- 实际 process group 的 rank 成员。
- 每个 microbatch 的执行顺序。
- Pipeline warmup / steady-state / cooldown。
- Gradient bucket 何时 ready、何时发起 collective。
- Param all-gather 的预取时点。
- MoE 每个 rank 的 token split。
- CP attention 使用的通信算法。
- Recompute 造成的重复计算和 activation 生命周期。

因此建议采用：

**静态配置解析 + 真实运行时 trace + 小规模成本画像** 的 hybrid workload extraction。

### 7.2 MCore 0.20 的 process group 语义

当前版本除了 TP / PP / DP / CP / EP / ETP，还包含 GTP-remat、Expert GTP-remat、hierarchical CP、dynamic CP 等维度。

适配器应读取 `ProcessGroupCollection` 和实际 process group ranks，而不是只根据并行度相乘。

重要 group 类型包括：

- TP group
- PP group
- CP group
- EP / Expert TP group
- DP group
- DP×CP group
- DP×CP×GTP-remat group
- Expert DP / Expert GTP-remat group

对于常规配置：

$$
B_g = b \cdot m \cdot d_g
$$

其中 $d_g$ 是持有不同训练样本的数据分发组大小。CP ranks 是同一个样本的 sequence shards，不应直接乘进 global batch。启用 GTP-remat 时，需要按其实际数据分发语义计算 $d_g$。

梯度同步组和 optimizer state 分片组可能与 global-batch 数据分发组不同。适配器必须记录真实 group members。

源码：

- [parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/parallel_state.py)
- [process_groups_config.py](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/process_groups_config.py)

### 7.3 并行行为映射

| 并行方式              | MCore 实际执行行为                                                     | IR 中应记录                                         | SimAI 映射                      |
|:----------------------|:-----------------------------------------------------------------------|:----------------------------------------------------|:--------------------------------|
| TP                    | Column / Row Parallel Linear 相关 gather / reduce-scatter / all-reduce | communicator、bytes、dtype、operation order         | SimCCL / ASTRA                  |
| PP                    | 每 microbatch 的 P2P Send / Recv，受 schedule 控制                     | src/dst rank、tensor bytes、microbatch、chunk、依赖 | 需要新增 P2P flow               |
| DP                    | DDP bucket gradient sync                                               | bucket size、ready time、start/wait                 | AllReduce / ReduceScatter       |
| Distributed Optimizer | ReduceScatter、optimizer update、参数 AllGather                        | optimizer shard group、bucket、prefetch event       | Collective + compute event      |
| EP                    | Router、Permutation、AllToAll、Expert GEMM、Unpermutation              | 每 rank 的 token split / send / recv bytes          | SimCCL A2A 或自定义 flow        |
| CP                    | Attention backend 对应的 ring / P2P / A2A 通信                         | CP group、attention backend、通信序列               | 需扩展 CP communicator          |
| SP                    | TP 域上的 activation sequence sharding                                 | TP group 与 tensor shape                            | TP collectives                  |
| Recompute             | 重复执行 forward 子图，减少 activation 保存                            | recompute module、重复 compute、activation lifetime | Compute event + memory lifetime |
| FP8                   | 低精度 GEMM、scale / amax、转换和通信 dtype                            | 每种 tensor 的 dtype、kernel path                   | 重新 profile 成本               |
| GTP-remat             | 新增的数据分发 / 重计算组语义                                          | 实际 process group、microbatch mapping              | 需扩展                          |

### 7.4 Pipeline schedule

MCore 的关键入口：

- `get_forward_backward_func()`
- `forward_backward_pipelining_without_interleaving()`
- `forward_backward_pipelining_with_interleaving()`
- `P2PCommunicator`

不能用一个统一 bubble ratio 代替这些 schedule。

对普通均衡 1F1B，可用下式作为 sanity check：

$$
\beta_{\rm bubble}
\approx
\frac{PP-1}{m+PP-1}
$$

但 VPP、非均匀 stage、异步 P2P、不同 microbatch grouping 都需要逐事件建模。

特别是 MCore 0.20 的非交错和交错 schedule 有不同的 P2P 行为；某些 overlap 选项不能在所有 schedule 中使用。

### 7.5 DP 与 Optimizer

重点适配：

- `DistributedDataParallel.start_grad_sync()`
- `DistributedDataParallel.finish_grad_sync()`
- `ParamAndGradBuffer` bucket
- `DistributedOptimizer.step_with_ready_grads()`
- `start_param_sync()` / `finish_param_sync()`

必须记录 bucket ready time、collective launch time、wait time、参数预取时间。只知道总梯度大小无法重建通信隐藏比例。

### 7.6 MoE

MCore `MoEAlltoAllTokenDispatcher` 典型路径：

``` text
Router
  |
  v
Token Permutation
  |
  v
EP AllToAll
  |
  v
TP / ETP AllGather
  |
  v
Grouped Expert GEMM
  |
  v
TP / ETP ReduceScatter
  |
  v
EP AllToAll
  |
  v
Token Unpermutation
```

实际路径依赖 dispatcher backend。DeepEP v2 等自定义通信实现不能简单视为 NCCL AllToAll。

MoE 适配必须采集每 rank 的实际 split sizes：

$$
D_{r,s} = n_{r,s} \cdot H \cdot b_{\rm elem}
$$

其中 $n_{r,s}$ 是 rank $r$ 发往 rank $s$ 的 token-expert assignments 数，$b_{\rm elem}$ 是 activation dtype 的字节宽度。

同时记录 expert load：

$$
\rho =
\frac{\max_e n_e}
{\frac{1}{E_x}\sum_e n_e}
$$

均衡路由作为基线，真实训练 trace 用于长尾验证。

### 7.7 CP

CP 不能简单地统一建模为 AllToAll。

MCore 的 CP attention 行为与 attention backend 相关。当前部分本地 DotProductAttention 路径不支持 CP，实际 CP 路径依赖 Transformer Engine attention 或其他专门实现。

因此应从真实运行时捕获 CP collective / P2P，而不是在配置文件中硬编码一个 CP AllToAll。

### 7.8 Recompute 与显存

每个 operation 至少应携带：

- input / output shape
- dtype
- tensor lifetime
- recompute 标记
- microbatch / layer / chunk
- memory alloc / free 事件

模型状态显存可先使用：

$$
M_{\rm state}
\approx
P_r\left(6+\frac{12}{d_s}\right)
\ \mathrm{bytes}
$$

该近似对应 BF16 参数、FP32 main gradients 和相关 Distributed Optimizer 分片模型。它不包含 activation、workspace、NCCL buffer 和 allocator reserved memory。

完整 peak memory：

$$
M_{\rm peak} =
\max_{r,t}
\left[
M_{{\rm state},r}(t)
+
M_{{\rm activation},r}(t)
+
M_{{\rm workspace},r}(t)
+
M_{{\rm comm},r}(t)
\right]
$$

SimAI 当前训练 workload 没有足够的 alloc/free 信息来可靠计算这一指标。建议使用 SimuMax 或自研 memory lifetime tracker，并用 MCore 实际 `max_memory_allocated`、`max_memory_reserved` 校准。

------------------------------------------------------------------------

## 8. 面向 Megatron Core 的仿真系统设计

### 8.1 推荐模块划分

| 模块                       | 实现方式                                               | 主要输入                            | 主要输出                                  |
|:---------------------------|:-------------------------------------------------------|:------------------------------------|:------------------------------------------|
| Megatron Config Normalizer | 新增                                                   | MCore CLI / YAML / ModelProvider    | 标准化 model / strategy / hardware config |
| Model Shape Analyzer       | 新增，复用 MCore model config                          | 模型结构、层、dtype                 | 每个模块的 tensor shape / FLOPs           |
| Process Group Extractor    | 新增，复用 `parallel_state` / `ProcessGroupCollection` | 真实 process groups                 | group rank list                           |
| Schedule Extractor         | 新增，复用 MCore schedule 语义                         | PP / VPP / microbatch               | 每 rank 执行依赖                          |
| Collective Tracer          | 新增，hook MCore / c10d wrappers                       | collective 调用                     | group、bytes、split、launch/wait          |
| Compute Profiler           | 复用 AIOB 思路，优先直接 profile MCore                 | shape、dtype、kernel backend        | per-op cost database                      |
| Optimizer Profiler         | 新增                                                   | AdamW / distributed optimizer       | optimizer cost / memory                   |
| MoE Router Profiler        | 新增                                                   | token routing                       | per-rank send / recv matrix               |
| Topology Builder           | 复用 SimAI topology generator 并扩展                   | rank placement / GPU / NIC / switch | NS-3 topology                             |
| Workload IR                | 新增                                                   | 所有上述信息                        | per-rank DAG                              |
| ASTRA Adapter              | 扩展                                                   | Workload IR                         | ASTRA events / flows                      |
| Result Analyzer            | 新增                                                   | SimAI / SimuMax outputs             | step time、throughput、bottleneck、误差   |

### 8.2 Canonical Workload IR

不要让 AICB 的 12 列文本格式成为长期接口。建议定义统一 IR。

每个 operation 至少包括：

``` text
Operation:
  id
  rank
  type: compute / collective / p2p / alloc / free
  name
  layer_id
  microbatch_id
  virtual_chunk_id
  phase: forward / backward / recompute / optimizer
  shape
  dtype
  flops
  bytes_read
  bytes_written
  cost_key
  stream_id
  dependencies[]
  communicator_id
  peer_ranks[]
  send_bytes[]
  recv_bytes[]
  memory_token
```

Run metadata：

``` text
Run:
  model_config
  strategy_config
  software_versions
  rank_to_node_gpu_map
  process_groups
  hardware_topology
  sequence_distribution
  routing_statistics
```

IR 可序列化成 JSON，或进一步适配 Chakra Execution Trace。

当前 SimAI ASTRA fork 没有直接消费 Chakra ET 的现成训练入口，因此仍需实现 `IR → ASTRA` adapter。第一阶段可以输出 legacy workload；后续切换为 DAG workload。

### 8.3 计算成本模型

对每个 operation：

$$
T_i =
\ell_i+
\max\left(
\frac{F_i}{\Pi_{{\rm eff},i}},
\frac{Q_i}{W_{{\rm HBM},i}}
\right)
$$

实际实现优先查 profile database：

``` text
key = (
    gpu_type,
    op_name,
    shape,
    dtype,
    kernel_backend,
    software_version
)
```

profile 至少区分：

- Attention forward / backward
- QKV / output projection
- MLP gate / up / down
- Norm / RoPE
- MoE router / grouped GEMM / permutation
- Cross entropy / logits
- Gradient accumulation
- AdamW optimizer- FP8 scale / conversion

AIOB 可以提供初始数据，但正式成本库应来自目标 MCore 0.20 的实际模块和 kernel。

### 8.4 通信成本模型

通信事件包含：

``` text
collective_type
group_ranks
message_size_per_rank
algorithm
protocol
launch_time
dependencies
```

SimCCL 能处理已支持的 NCCL collective。对于 unsupported collective、DeepEP 或自定义通信，提供 fallback：

1.  解析带宽模型。
2.  自定义 rank-to-rank flow。
3.  真实通信 benchmark lookup。

网络拓扑必须包含：

- GPU / NVLink / NVSwitch。
- PCIe root complex。
- NIC 与 GPU 亲和性。
- 每节点 NIC 数量和 rail。
- Leaf / Spine switch。
- Link bandwidth / latency。
- Oversubscription。
- PFC / ECN / DCQCN 等配置。

### 8.5 执行调度

统一事件模型：

$$
s_i =
\max\left(
\max_{j\in pred(i)} f_j,
a_i
\right)
$$

$$
f_i=s_i+\tau_i
$$

$$
T_{\rm step}=\max_i f_i-\min_i s_i
$$

当多个 operation 共享 HBM / SM / NIC / link 时，$\tau_i$ 应根据并发状态更新。当前 SimAI 的固定 compute delay 与网络流调度还没有完整实现 GPU 内部资源干扰，需要后续校准。

### 8.6 输出指标

| 指标                     | SimAI 当前能力                           | 额外工作                        |
|:-------------------------|:-----------------------------------------|:--------------------------------|
| Step Time                | 可输出 workload 结束时间                 | 校准 optimizer 和 schedule      |
| Throughput               | 可根据 batch / sequence / step time 计算 | 处理变长 token / padding        |
| MFU                      | 可推导                                   | 需要正确模型 FLOPs 和 GPU peak  |
| Compute / Communication  | 可输出 compute 与 exposed communication  | 统一时间口径                    |
| Pipeline Bubble          | 有聚合估计                               | 完整 PP schedule 需要扩展       |
| Network Link Utilization | NS-3 可输出网络层统计                    | 映射到训练关键路径              |
| GPU Utilization          | 不能直接得到真实 SM utilization          | 需要 GPU resource model / trace |
| Peak Training Memory     | 当前训练链路不足                         | 接入 SimuMax / memory tracker   |
| MoE Load Imbalance       | 需要外部 routing 数据                    | 增加 per-expert / per-rank 统计 |
| Bottleneck Attribution   | 可从通信与层统计推导                     | 输出最慢 rank / critical path   |

------------------------------------------------------------------------

## 9. 分阶段 Roadmap

### Phase 0：环境锁定与官方例子跑通

**目标：** 建立可复现 SimAI 工具链，验证 Analytical 和 NS-3 两种后端。

核心任务：

- 锁定 SimAI / AICB / SimCCL / ns-3 / MCore SHA。
- 构建 `SimAI_analytical`。
- 构建 `SimAI_simulator`。
- 运行官方 workload 和 microAllReduce 示例。
- 生成一份小型 topology。
- 保存编译环境、参数和结果。

依赖：SimAI、AICB、SimCCL、NS-3。

最小运行：

``` bash
git checkout a57d765c5478f319c5f5ab6fdc53f9cc7f461743
git submodule update --init --recursive

./scripts/build.sh -c analytical
./scripts/build.sh -c ns3
```

验收：

- 两种 binary 正常启动。
- Analytical workload 输出 EndToEnd CSV。
- NS-3 microAllReduce 正常结束。
- 固定 SHA 后可重复生成同样的 workload。

风险：NS-3 依赖、编译环境、submodule 分支漂移。

### Phase 1：最小 Dense PoC，尽快完成真机验证

<badge color="success">第一优先级</badge>

**目标：** 在一台 4×A800 节点上验证 AICB → SimAI Analytical → Megatron Core 的基本链路。

配置建议：

| 项目              | 配置               |
|:------------------|:-------------------|
| Model             | GPT-style Dense 7B |
| Layers            | 32                 |
| Hidden            | 4096               |
| Attention Heads   | 32                 |
| KV Heads          | 32，MHA            |
| FFN               | 16384              |
| Sequence          | 2048               |
| Dtype             | BF16               |
| TP / PP / CP / EP | 4 / 1 / 1 / 1      |
| DP                | 1                  |
| Micro-batch       | 1                  |
| Global batch      | 1                  |
| GA                | 1                  |
| Recompute         | Off                |
| SP                | Off                |
| VPP               | 1 / disabled       |
| Optimizer         | AdamW，单独测量    |

AICB workload：

``` bash
cd SimAI/aicb

bash scripts/megatron_workload_with_aiob.sh \
  --gpu_type A800 \
  --model_size 7 \
  --world_size 4 \
  --tp 4 \
  --pp 1 \
  --global_batch 1 \
  --micro_batch 1 \
  --seq_length 2048 \
  --aiob_enable \
  --flash_attn
```

该命令使用 AIOB 在当前 GPU 上生成 compute profile，随后输出到 `results/workload/`。从命令输出中获取完整 workload 文件名。

构建并运行 Analytical：

``` bash
cd ../
./scripts/build.sh -c analytical

./bin/SimAI_analytical \
  -w ./aicb/results/workload/<generated-workload>.txt \
  -g 4 \
  -nv <NVLINK_GBPS> \
  -nic <NIC_GBPS> \
  -n_p_s <NICS_PER_SERVER> \
  -g_p_s 4 \
  -r ./results/poc-a800-
```

`NVLINK_GBPS`、`NIC_GBPS` 和 `NICS_PER_SERVER` 必须从实际机器拓扑和通信 benchmark 获取，不要直接复制示例值。先用 `nvidia-smi topo -m` 确认 GPU 之间的连接方式。

Megatron Core 0.20.0 真机对照使用相同结构：

``` text
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_query_groups = 32
ffn_hidden_size = 16384
seq_length = 2048
TP = 4
PP = 1
CP = 1
EP = 1
micro_batch_size = 1
global_batch_size = 1
dtype = BF16
```

固定 Transformer implementation 和 FlashAttention backend。若 AIOB 和 MCore 使用的 kernel backend 不一致，先用实际 MCore profile 替换 AIOB cost database。

**先比较 Forward + Backward + TP Communication。** AIOB 的 `Grad_param` 不等同于完整 AdamW。真实 `optimizer.step()` 应单独 profile，再作为独立事件加回总 step time。

验收目标：

- 模型层数、shape、dtype、TP group 一致。
- TP collective 类型、次数、消息大小一致。
- 每层 compute cost 中位误差目标 ≤10%。
- TP collective 时间误差目标 ≤15%。
- 校准后的 step time MAPE 目标 ≤10%。
- 以上是工程验收目标，不是当前工具已有保证。

风险：

- AIOB 与 MCore 真实 kernel 不同。
- optimizer update 没有完整建模。
- `process_time=100` 是固定 update delay。
- 单节点 PoC 不验证 scale-out。

进入下一阶段的前置条件：能解释 compute、TP communication 和 optimizer 的独立误差。

### Phase 2：Dense TP + DP，多节点网络仿真

**目标：** 验证跨节点 AllReduce / ReduceScatter / AllGather 和 NS-3 topology。

建议：

| 项目         | 配置 |
|:-------------|:-----|
| 节点         | 2    |
| GPU / Node   | 4    |
| Total GPU    | 8    |
| TP           | 4    |
| PP           | 1    |
| DP           | 2    |
| CP / EP      | 1    |
| Global Batch | 2    |
| Micro-batch  | 1    |
| GA           | 1    |

核心任务：

- MCore DDP bucket instrumentation。
- Distributed Optimizer ReduceScatter / AllGather。
- 真实 rank-to-node placement。
- NCCL Tests 与 SimCCL flow 对照。
- 生成 2-node NS-3 topology。
- 先关闭 overlap，再单独开启 overlap。

运行示例：

``` bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  --ro -g 8 -gt H100 -bw 400Gbps -nvbw 1440Gbps

AS_SEND_LAT=6 ./bin/SimAI_simulator \
  -t 8 \
  -w <workload.txt> \
  -n <generated_topology> \
  -c ./astra-sim-alibabacloud/inputs/config/SimAI.conf
```

实际 GPU 类型、NIC 带宽和拓扑应替换为真实集群参数。

输出：

- DP collective latency。
- 每种消息大小的 algbw / busbw。
- Per-link throughput。
- Queue / congestion。
- Exposed communication。
- Step time 与吞吐。

验收：

- 实际 NCCL group ranks 与 IR 一致。
- Message size 与 NCCL trace 一致。
- 留出消息大小的 collective 时间误差目标 ≤15%。
- 留出 placement 的趋势排序正确。

### Phase 3：Pipeline Parallelism 与 VPP

**目标：** 将当前聚合 `pp_comm` 升级为逐 microbatch P2P 事件。

核心任务：

1.  读取 MCore `forward_backward_pipelining_without_interleaving()`。
2.  读取 `forward_backward_pipelining_with_interleaving()`。
3.  记录 P2PCommunicator 的 Send / Recv、partner、tensor size、microbatch、chunk。
4.  在 ASTRA 增加 P2P event / dependency。
5.  支持非均匀 pipeline stage。
6.  对齐 P2P overlap 与 warmup / cooldown。

实验：

- 4 GPU TP2 × PP2。
- 同样总 GPU 数比较 TP4 × PP1。
- 再加入 VPP2 / VPP4。
- 分别验证低 GA 与高 GA。

验收：

- 每个 microbatch 的 P2P 数量和 bytes 对齐。
- Pipeline schedule 的事件顺序一致。
- Bubble time 与真实 trace 趋势一致。
- 最慢 PP stage 识别正确。

### Phase 4：MoE EP / ETP 与负载不均衡

**目标：** 从均衡 token 假设升级到真实 router split。

核心任务：

- 复用 MCore `MoEAlltoAllTokenDispatcher`。
- 捕获每 rank 的 `input_splits` / `output_splits`。
- 捕获 EP AllToAll、TP / ETP AllGather / ReduceScatter。
- profile Grouped GEMM。
- 记录每 expert / rank 的 token histogram。
- 增加 DeepEP v2 专用通信成本模型。

实验：

- 8 GPU，EP1 / EP2 / EP4 / EP8。
- 固定 TP / PP / global batch。
- 比较均衡 routing 和受控偏斜 routing。
- 测量 AllToAll、Grouped GEMM 和最慢 rank。

验收：

- 每 rank send / recv bytes 对齐。
- Per-expert token 数对齐。
- 负载偏斜导致的 step time 增量方向正确。
- DeepEP 路径与 NCCL AllToAll 路径分别校准。

### Phase 5：CP、GTP-remat 与长序列

**目标：** 支持 MCore 0.20.0 的高级上下文并行与数据分发语义。

核心任务：

- CP group rank mapping。
- TE attention 的 CP ring / P2P / A2A 实际路径。
- Hierarchical CP。
- Dynamic CP。
- GTP-remat / Expert GTP-remat。
- CP + DP 梯度同步组。
- 长序列 activation lifetime 和 recompute。

实验：

- Sequence 8K / 32K / 128K。
- CP1 / CP2 / CP4。
- 固定 global batch。
- 对比 attention backend。
- 对比 recompute on/off。

验收：

- CP group 和 DP×CP group 语义正确。
- 每种 CP backend 的通信量和操作序列匹配。
- Peak allocated memory 预测目标误差 ≤10%。
- OOM 可行性判断正确。

### Phase 6：大规模预测与硬件设计空间搜索

**目标：** 从已验证的 4–16 GPU 扩展到 128 / 512 / 1024 GPU 设计空间。

核心任务：

- 多节点 rank placement。
- 多 rail / oversubscription。
- 多 pod 自定义拓扑。
- NCCL 算法和协议组合。
- 网络拥塞控制参数。
- GPU 代际切换。
- 计算画像的硬件外推。
- 不确定性区间和敏感性分析。

输出：

- Step time / tokens/s。
- MFU / scaling efficiency。
- TP / DP / EP / CP exposed communication。
- Pipeline bubble。
- Per-link congestion。
- 最慢 rank / stage。
- Hardware cost-performance frontier。

验收：

- 留出配置 MAPE、最大误差和 P95 同时报告。
- 不将预测差异小于模型误差的候选强行排序。
- 对没有目标硬件实测的结果标记为外推。
- 保留完整版本、配置和校准数据库。

------------------------------------------------------------------------

## 10. 校准与验证方法

### 10.1 Compute Profile

针对真实 MCore 0.20，建立 profile database：

``` text
GPU
dtype
kernel_backend
operation
input_shape
output_shape
parallel_degree
sequence_length
micro_batch
forward_time
dgrad_time
wgrad_time
recompute_time
```

采集：

- PyTorch Profiler。
- Nsight Systems。
- 必要时 Nsight Compute。
- MCore 模块级 hooks。
- CUDA Event 稳态时间。

每个 profile 应包含 GPU、驱动、CUDA、PyTorch、NCCL、TE、FlashAttention 版本。

### 10.2 Collective Calibration

使用 `nccl-tests`：

- AllReduce
- AllGather
- ReduceScatter
- AllToAll

覆盖消息大小 4 KiB 至 1 GiB、group size 2 / 4 / 8 / 16 / 32 等。

固定或记录：

- NCCL Algo。
- NCCL Proto。
- Channel 数。
- NIC / rail。
- Rank placement。
- NCCL 环境变量。

对比 SimCCL flow 和 NS-3 网络时间。

### 10.3 Overlap Calibration

使用 Nsight Systems 采集真实 timeline：

- GEMM 与 TP collective。
- Backward 与 DP bucket。
- EP AllToAll 与 expert GEMM。
- PP Send / Recv 与计算。
- Param AllGather 与下一轮 forward。

不要用独立通信 benchmark 的时间直接代表 overlap 场景中的 exposed communication。

### 10.4 Memory Calibration

记录：

- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`
- NVML process memory
- MCore activation / optimizer state
- NCCL buffer
- Workspace / temporary tensor

allocated、reserved 和设备总占用分别验证。

### 10.5 误差指标

$$
\epsilon_j =
\frac{|\hat T_j-T_j|}{T_j}
$$

$$
MAPE =
\frac{1}{J}\sum_{j=1}^{J}\epsilon_j
$$

同时报告：

- MAPE。
- Maximum relative error。
- P95 error。
- 每个并行策略的误差。
- 最慢 rank / stage 是否预测正确。
- 最优配置排序是否正确。

不要只拟合总 step time。应先验证 per-op compute、collective、P2P、schedule 和 memory，再检查总时间。

------------------------------------------------------------------------

## 11. 技术风险与限制

| 风险                     | 影响                                      | 处理方法                                   |
|:-------------------------|:------------------------------------------|:-------------------------------------------|
| AICB 与 MCore 版本不一致 | shape / schedule / communication 语义偏差 | 自研 MCore 0.20 adapter                    |
| AIOB 与 TE kernel 不一致 | compute cost 偏差                         | 真实 MCore profile                         |
| CP 缺失                  | 长序列策略预测错误                        | 新增 CP communicator 和 attention trace    |
| VPP 假设                 | bubble / P2P 错误                         | 从 MCore schedule 导出事件                 |
| PP 聚合通信              | 无法比较 P2P overlap 和拓扑               | 增加 P2P flow                              |
| `depen` 未参与调度       | 无法表达任意 DAG                          | 扩展 ASTRA Workload                        |
| MoE 均衡假设             | 低估 straggler                            | 记录 per-rank token split                  |
| Optimizer 不完整         | step time 偏低                            | 独立 profile AdamW / distributed optimizer |
| Training memory 缺失     | 无法判断 OOM                              | SimuMax / memory lifetime tracker          |
| NCCL 算法覆盖有限        | Flow 不等同真实 NCCL                      | NCCL trace / 自定义 fallback               |
| GPU / Network 资源竞争   | overlap 时间偏差                          | 共享资源模型和并发校准                     |
| 新 GPU 外推              | compute cost 不可靠                       | 目标 GPU 少量 benchmark                    |

------------------------------------------------------------------------

## 12. 最终技术选型建议

### 12.1 短期

先使用官方 AICB + SimAI Analytical 完成单机 Dense PoC，确认：

- Workload 生成正确。
- Compute profile 可用。
- TP collective 成本基本可信。
- 输出指标可以与 MCore 对照。

不要在第一阶段扩展 CP、VPP 或 MoE。

### 12.2 中期

开发 `Megatron-Core 0.20 Adapter`，将真实 MCore 执行语义转换为 canonical per-rank DAG。

优先完成：

1.  TP / DP。
2.  P2P / PP。
3.  Optimizer。
4.  EP / MoE。
5.  CP。

SimCCL 和 ASTRA / NS-3 继续作为网络后端。

### 12.3 长期

采用混合架构：

- SimuMax：计算效率、optimizer、memory 和快速策略搜索。
- SimAI Analytical：快速通信估算。
- SimCCL + NS-3：精细网络仿真。
- ASTRA：统一事件调度。
- Megatron Adapter：真实模型、process groups、schedule 和动态 routing。
- 自研 IR：连接所有模块，避免绑定 AICB 的旧 workload schema。

**最终目标不是把 Megatron 配置翻译成 SimAI 文本，而是把 Megatron 的真实分布式执行过程变成一个可重调度、可替换成本模型、可校准的 per-rank execution graph。**

这才是能够长期支持 TP / PP / DP / EP / CP、MoE、不同 GPU 和不同网络拓扑的基础。

------------------------------------------------------------------------

## 13. References

### SimAI 与论文

1.  [SimAI Repository](https://github.com/aliyun/SimAI)
2.  [SimAI 固定版本 commit](https://github.com/aliyun/SimAI/commit/a57d765c5478f319c5f5ab6fdc53f9cc7f461743)
3.  [SimAI NSDI 2025 Paper](https://www.usenix.org/system/files/nsdi25-wang-xizheng-simai.pdf)
4.  [SimAI Tutorial](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/docs/Tutorial.md)
5.  [SimAI Chinese Tutorial](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/docs/CN/Tutorial.md)
6.  [SimAI Layer.cc](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/workload/Layer.cc)
7.  [SimAI Workload.cc](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/workload/Workload.cc)
8.  [SimAI Sys.cc](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/system/Sys.cc)
9.  [SimAI NS-3 entry.h](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/astra-sim/network_frontend/ns3/entry.h)
10. [SimAI Topology Generator](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py)
11. [SimAI Network Configuration](https://github.com/aliyun/SimAI/blob/a57d765c5478f319c5f5ab6fdc53f9cc7f461743/astra-sim-alibabacloud/inputs/config/SimAI.conf)

### AICB / AIOB

12. [AICB Repository](https://github.com/aliyun/aicb)
13. [AICB SimAI Training Workload Generator](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/workload_generator/SimAI_training_workload_generator.py)
14. [AICB MockedMegatron](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/workload_generator/mocked_model/training/MockedMegatron.py)
15. [AICB AiobMegatron](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/workload_generator/mocked_model/training/AiobMegatron.py)
16. [AICB utils](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/utils/utils.py)
17. [AICB AIOB workload shell script](https://github.com/aliyun/aicb/blob/23eec3c48ca2d2d93dd888a4c7b22ab4421e782f/scripts/megatron_workload_with_aiob.sh)

### SimCCL / NS-3 / CLEM

18. [SimCCL Repository](https://github.com/aliyun/SimCCL)
19. [SimCCL Architecture](https://github.com/aliyun/SimCCL/blob/fd7cd57d16f9bd42e3ccb70911c977e18ec294b9/docs/design/architecture.md)
20. [SimCCL SimAI Integration](https://github.com/aliyun/SimCCL/blob/fd7cd57d16f9bd42e3ccb70911c977e18ec294b9/docs/integration/integration-with-simai.md)
21. [ns-3 AlibabaCloud](https://github.com/aliyun/ns-3-alibabacloud)
22. [SimAI-CLEM](https://github.com/aliyun/SimAI-CLEM)

### SimuMax

23. [SimuMax Repository](https://github.com/MooreThreads/SimuMax)
24. [SimuMax v1.2 Release](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/docs/release_v1.2.md)
25. [SimuMax FULL_RESULTS](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/docs/FULL_RESULTS.md)
26. [SimuMax B200 Validation Summary](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/docs/b200/b200_release_v1.2_summary.md)
27. [SimuMax System Model](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/docs/system.md)
28. [SimuMax Strategy Model](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/docs/strategy.md)
29. [SimuMax config.py](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/simumax/core/config.py)
30. [SimuMax perf_llm.py](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/simumax/core/perf_llm.py)
31. [SimuMax pipeline_schedule.py](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/simumax/core/transformer/pipeline_schedule.py)
32. [SimuMax memory tracker](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/simumax/core/simu_memory.py)
33. [SimuMax MoE module](https://github.com/MooreThreads/SimuMax/blob/4dcaae513ba7679d56c116a17cd4a691ec41e252/simumax/core/transformer/moe_module.py)

### Megatron Core 0.20.0

34. [Megatron Core 0.20.0 Release](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.20.0)
35. [Megatron Core parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/parallel_state.py)
36. [Megatron Core ProcessGroupCollection](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/process_groups_config.py)
37. [Pipeline schedules](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/pipeline_parallel/schedules.py)
38. [DistributedDataParallel](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/distributed/distributed_data_parallel.py)
39. [Param and Grad Buffer](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/distributed/param_and_grad_buffer.py)
40. [MoE Token Dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/transformer/moe/token_dispatcher.py)
41. [Distributed Optimizer](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/optimizer/distrib_optimizer.py)
42. [Transformer Block](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/transformer/transformer_block.py)
43. [Megatron Core Context Parallel Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
44. [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
45. [Megatron Core Distributed Optimizer Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)

### 其他

46. [NCCL Tests Performance Definitions](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
47. [ASTRA-sim upstream](https://github.com/astra-sim/astra-sim)
48. [Chakra](https://github.com/mlcommons/chakra)
49. [Anti-Defensive-Writing SKILL](https://github.com/Kiterlin/anti-defensive-writing)
