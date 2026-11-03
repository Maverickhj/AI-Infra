---
type: knowledge
status: draft
created: 2026-11-03
updated: 2026-11-03
domains: [GPU, CUDA, PyTorch, Attention, Performance Modeling]
ai_generated: true
reviewed: false
---

# GPU Attention 六阶段学习路线：从 Tensor 计算到 Kernel 性能分析

> [!warning]
> 本文依据 2026-11-03 的学习方案整理，含 AI 生成内容，尚未完成人工核验。下文资源数字是理论推导而非 GPU 实测；实验结果须记录硬件、版本与测量条件。

**主线：Attention Workload → Computation & Data Movement → GPU Architecture → GPU Execution → Performance Modeling → FlashAttention。**

本路线是 [[10_Projects/attention-gpu-learning-lab/_Project - Attention GPU Learning Lab|Attention GPU Learning Lab]] 的完整六阶段规划。[[10_Projects/attention-gpu-learning-lab/README|现有入门实验指南]] 保留阶段 0～6 的轻量练习、命令与已编写的脚本；本文侧重深入 GPU Kernel 分析及最终验收。学习项目与 nano-vllm 的 MLA 扩展工作独立。

## 0. 最终目标与执行原则

完成路线后，应能够独立：

1. 给定 Attention Shape、Dtype 与实现方式，推导 FLOPs、Footprint、逻辑 Memory Traffic，并分析实际 DRAM/L2 流量。
2. 解释一个 GEMM 的 Grid/Block/Warp/Thread 分工、Tile 复用、Register/Shared Memory 使用及 Tensor Core 路径。
3. 从 PyTorch API 追踪到 ATen/Dispatcher、CUDA Backend 或 cuBLAS、GPU Kernel，并在可控样例中检查 PTX/SASS。
4. 建立逐 Kernel Roofline/资源模型，通过 Timeline、计数器与单变量实验验证性能归因。
5. 从瓶颈出发解释 FlashAttention 的 Online Softmax、Fusion、Tiling、并行切分和架构专属 Pipeline，进而阅读核心 Forward 源码。

每个问题按 **What / How Much / Where / How / Why** 五个维度记录。优先级：**P0** 为核心必修，**P1** 为源码阅读与 Profiling 进阶，**P2** 为之后按需扩展。

预计 **66～94 小时**（每周 8 小时约 9～12 周）。这是学习预算，不是代码执行或学习成果的实测。统一坚持：**数学推导 → 可观测实验 → 运行证据 → 归因 → 最小验证**。

### 数学符号表

新符号首次使用时同时更新本表；下文代码与文档中的公式均使用 Markdown LaTeX。

| 符号 | 含义 |
|---|---|
| $B,H$ | Batch Size、Query Attention Head 数；主案例为 MHA |
| $N_q,N_k,N$ | Query、KV 序列长度；主案例中 $N_q=N_k=N$ |
| $D,D_v$ | Q/K Head Dimension、V Head Dimension；主案例 $D_v=D$ |
| $s$ | 存储元素字节数；FP16 时 $s=2$ |
| $Q,K,V,O$ | Query、Key、Value、Output |
| $S_0,S,P$ | 原始 Attention Scores、缩放后的 Scores、Softmax 概率 |
| $F$ | FLOPs，矩阵乘的一次乘加按 2 FLOPs 计 |
| $M$ | 指定范围内的 Tensor 存储占用（bytes） |
| $\mathcal V_{\mathrm{logical}}$ | 以算子边界输入读一次、输出写一次计算的逻辑 Traffic |
| $\mathcal V_{\mathrm{DRAM}},\mathcal V_{\mathrm{L2}}$ | 实测 DRAM、L2 接口流量，单位 bytes |
| $I$ | Arithmetic Intensity，单位 FLOP/byte，需注明 Traffic 所属层次 |
| $P_{\mathrm{compute}}, BW_{\mathrm{memory}}$ | 与执行路径匹配的理论计算吞吐及指定存储层带宽 |
| $T_{\min},T_k,T_{\mathrm{gap}}$ | 简化时间下界、第 $k$ 个 Kernel 时间、GPU 时间线可见间隙 |
| $m_t,n_t,k_t$ | GEMM Tile 的输出行、输出列及归约长度 |
| $B_r,B_c$ | 分块 Attention 的 Query、KV Tile 序列长度 |
| $m,\ell,u$ | Online Softmax 的运行最大值、指数和、未归一化输出累加器 |
| $m',\ell',u',\alpha$ | 新块合并后的状态及历史状态的重缩放系数 |
| $x_j,p_j,V_j$ | 第 $j$ 个 KV 块的缩放 Scores、指数权重和 Value Tile |

存储单位 **MiB = $2^{20}$ bytes**；计算量 **GFLOPs = $10^9$ FLOPs**。根据具体 GPU 区分 GDDR 与 HBM；本文统一称 Device DRAM。

## 1. 整体知识地图与投入

~~~mermaid
flowchart TD
    A["Stage 1：Workload / FLOPs / Traffic"] --> B["Stage 2：GPU 硬件与存储"]
    B --> C["Stage 3：线程映射与 Tiling"]
    A --> D["Stage 4：PyTorch 到 Kernel"]
    C --> D
    A --> E["Stage 5：性能建模与 Profiling"]
    C --> E
    D --> E
    D --> F["Stage 6：FlashAttention"]
    E --> F
~~~

| 阶段 | 核心问题 | 优先级 | 预计投入 | 核心产物 |
|---|---|---|---:|---|
| Stage 1 | Attention 有多少计算、存储和数据流动？ | P0；Causal/Backward 为 P1 | 8～10h | 理论资源账本 |
| Stage 2 | 哪些 GPU 资源完成计算与搬移？ | P0；专属硬件机制 P1 | 8～12h | 硬件资源与数据流图 |
| Stage 3 | 如何给线程分工并复用数据？ | P0；高级流水 P1 | 14～20h | Naive/Tiled/Tensor-Core GEMM 对照 |
| Stage 4 | Python API 最后执行了什么？ | P0；源码/反汇编 P1 | 10～14h | API→Kernel Trace 与源码图 |
| Stage 5 | 性能受谁限制，怎样验证？ | P0/P1 | 10～14h | 逐 Kernel Roofline 与测量表 |
| Stage 6 | FlashAttention 的设计为何有效？ | FA1/2 P0；FA3/4 P1 | 16～24h | 算法/源码/硬件/实测对照报告 |

## 2. 贯穿案例与统一资源基线

### 2.1 基线代码及条件

~~~python
import torch
import math

B, H, N, D = 2, 8, 1024, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

with torch.inference_mode():
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(D)
    probs = torch.softmax(scores, dim=-1)
    output = probs @ v
~~~

初始条件：单 GPU、Dense、Non-causal、Forward、无 Dropout、无 Autograd、Eager 模式；输入预先位于设备，不把随机 Tensor 生成算入 Attention 计时。Scaled Dot-Product Attention：

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}\right)V.
$$

依次计算 $S_0=QK^\top$、$S=S_0/\sqrt D$、$P=\operatorname{softmax}(S)$、$O=PV$。本例所有输入均为 FP16；内部累加方式由具体 Backend 决定，需单独检查。

### 2.2 Shape、Stride 与存储

| Tensor | Shape | FP16 大小 |
|---|---|---:|
| $Q,K,V$（各一个） | $[2,8,1024,64]$ | 各 2 MiB |
| $S_0,S,P$（各一个） | $[2,8,1024,1024]$ | 各 32 MiB |
| $O$ | $[2,8,1024,64]$ | 2 MiB |

连续 Q/K/V 的元素 Stride 为 $(524288,65536,64,1)$。$K$ 经过最后两维转置后的 Stride 为 $(524288,65536,1,64)$，一般是共享 Storage 的 View；仅转置并不等于搬移数据。

在给定 Python 生命周期、scores 变量重绑定覆盖旧 Tensor 后，函数末尾仍保留 $Q,K,V,S,P,O$，这批 Tensor 的数据量合计 **72 MiB**。这是活跃数据大小，不等于 GPU Allocator Reserved Memory、峰值内存或 DRAM Traffic。

### 2.3 FLOPs 与逻辑 Traffic

两个 GEMM 的计算量为：

$$
F_{QK^\top}=2BHN^2D,\qquad F_{PV}=2BHN^2D.
$$

因此：

$$
F_{\mathrm{GEMM}}=4BHN^2D
=4{,}294{,}967{,}296
\approx 4.295\ \mathrm{GFLOPs}.
$$

Scaling 有 $BHN^2=16{,}777{,}216$ 个逐元素缩放；Softmax 有 $BH N=16{,}384$ 行，每行长度 1024，包括 Max/Sum Reduction、Exp 与归一化。指数不能简单并入 Tensor Core FLOPs 吞吐上界。

逻辑读写模型：每个算子输入读一次、输出写一次，FP16 Scores 与 Probability 独立物化。

| 步骤 | 输入读 | 输出写 | 合计 |
|---|---:|---:|---:|
| $QK^\top$ | 4 MiB | 32 MiB | 36 MiB |
| Scaling | 32 MiB | 32 MiB | 64 MiB |
| Softmax | 32 MiB | 32 MiB | 64 MiB |
| $PV$ | 34 MiB | 2 MiB | 36 MiB |
| **合计** | | | **200 MiB** |

以两次 GEMM 的 FLOPs 为分子，对应粗粒度逻辑计算强度：

$$
I_{\mathrm{logical}}=\frac{4{,}294{,}967{,}296}{200\times 2^{20}}
=20.48\ \mathrm{FLOP/byte}.
$$

**三个口径分别记录：**

1. Tensor Footprint：数据对象占用空间；本例末尾活跃数据为 72 MiB。
2. Logical Traffic：特定算子边界模型下，约 200 MiB；实际 tiled GEMM 会重读部分输入，内部实现还可能产生额外操作。
3. Actual DRAM/L2 Traffic：由硬件计数器给出，受 Tile、缓存命中、Transaction 和实现细节影响。

仅数一次 $Q,K,V$ 的读和 $O$ 的写，端点数据量是 8 MiB。这是理解融合的最低端点数据量参考；并不宣称融合 Kernel 能在任何 Shape 下达到该流量。

### 2.4 Prefill 与 Decode

将长度分离，矩阵乘计算量为：

$$
F_{\mathrm{GEMM}}=4BHN_qN_kD.
$$

| 指标 | 主案例 Prefill 形状 | 单步 Decode 形状 |
|---|---:|---:|
| $N_q,N_k$ | 1024, 1024 | 1, 1024 |
| 两个 GEMM | 4.295 GFLOPs | 4.194 MFLOPs |
| Q/K/V 一次读 + O 写的端点量 | 8 MiB | 约 4.004 MiB |
| 端点计算强度 | 512 FLOP/byte | 约 1 FLOP/byte |

Decode 的 KV 历史仍需扫描，计算量与并行度却大幅减小。这里是单个独立 Attention 运算的对照，既不包含 Projection，也不包含真实 KV Cache 管理与调度开销。

### 2.5 各阶段复用规则

Stage 1 分析数据与公式；Stage 2 把数据映射到硬件层次；Stage 3 用 $QK^\top$ 和 $PV$ 对比矩阵形状；Stage 4 追踪 Eager 与 SDPA；Stage 5 验证时间与流量；Stage 6 用 Online Softmax 和融合改变执行数据流。

保持同一基线，之后每次实验只改变一类条件。新增 Causal、不同 $N_q/N_k$、FP32/BF16、Backward 时另开配置，不与主基线混算。

## 3. 推荐阅读路径

### AI Infra Book

| 章节 | 优先重点 | 为什么现在读 |
|---|---|---|
| [第三章：推理与训练负载](https://bojieli.github.io/ai-infra-book/) | P0：3.1.1、3.1.2；P1：3.4.1、3.4.3 | 建立 Prefill/Decode 的工作量、存储需求、Forward/Backward 语境 |
| [第四章：加速器架构](https://bojieli.github.io/ai-infra-book/) | P0：4.1.1～4.1.2、4.2.1～4.2.3、4.3、4.8.1、4.8.3；P1：4.4.1～4.4.2 | 对应 SM、Tensor Core、存储层次、计算与带宽约束 |
| [第五章：算子与运行时](https://bojieli.github.io/ai-infra-book/) | P0：5.1.1、5.1.3～5.1.5、5.2.1～5.2.2、5.3.1、5.3.3；P1：5.2.3～5.2.4、5.3.2 | 连接算子提交、线程组织、GEMM Tiling、Kernel Fusion 与 Pipeline |

建议顺序：**3.1 → 4.1/4.2/4.3 → 5.1/5.2 → 4.8 → 5.3**。章节名称、版本与书的实际目录发生变化时，在阅读笔记记录访问日期。

### 官方与论文资料

| 主题 | 资料 |
|---|---|
| CUDA 执行与存储 | [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)；[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) |
| GEMM Tiling | [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)；[WMMA / Warp Matrix Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions) |
| PyTorch 内部 | [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)；[CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)；[Custom C++/CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)；[SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |
| 分析与指令 | [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)；[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)；[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)；[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)；[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) |
| Triton | [Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)；[Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| FlashAttention | [FA1](https://arxiv.org/abs/2205.14135)；[FA2](https://arxiv.org/abs/2307.08691)；[FA3](https://arxiv.org/abs/2407.08608)；[FA4](https://arxiv.org/abs/2603.05451)；[实现仓库](https://github.com/Dao-AILab/flash-attention) |

## 4. Stage 1：Attention Workload Analysis

**核心问题：** Attention 表达式如何对应 Shape、计算量、存储与搬移？  
**前置：** 矩阵乘、Softmax、Tensor 基础；**投入：** 8～10h。

**P0 理论：** 逐行推导 $QK^\top$、Scaling、Softmax、$PV$ 的数据依赖、Shape/Stride、FP16 存储与内部计算精度；区分 Tensor Lifetime、逻辑 Traffic 和 DRAM Traffic。验证 $N$ 翻倍时 GEMM FLOPs 与 Score Matrix 面积均为 4 倍，$D$ 翻倍时 GEMM FLOPs 和 QKV 体积为 2 倍，$B$ 翻倍时工作量与数据量均为 2 倍。比较 Prefill $N_q=N_k$ 与 Decode $N_q=1$。

**P1：** Causal 有效元素与实际执行 FLOPs 的差别；Backward 的存储与重算。**P2：** GQA/MQA、变长、分页 KV Cache。

**可视化：** 计算 DAG；Shape/Stride 图；随 $N$ 增长的 FLOPs、QKV/Score 大小三条曲线。

**最小实验：**

1. 运行现有 attention_lab.py 的 tensors 模式，核对每个 Tensor 的 Shape、Stride、Contiguous、Storage Pointer；单独比较 transpose 与 contiguous。
2. 小 Shape 的 CPU FP64 Reference 与 GPU Eager 输出比较，报告 Max-Abs、RMSE、Relative-L2；大 Shape 使用经说明的 FP32 Reference，注明 TF32/Matmul 设置。
3. 分别扫描 $N\in\{256,1024,4096\}$、$D\in\{32,64,128\}$、$B\in\{1,2,8\}$；增加一个单 Query 的 Decode Shape，写出运行前预测。

**阅读：** AI Infra Book 第三章 3.1；PyTorch Tensor Views、SDPA 文档。

**验收：** 不查资料推导 4.295 GFLOPs、72 MiB、200 MiB 的来源；能解释 transpose 为什么不一定复制、Decode 的 KV 流量为何仍显著；分清 FP32 累加与 FP32 输出。

**产物：** 资源账本 + Tensor 元数据记录 + Shape 扫描预测/观察表。

## 5. Stage 2：GPU Hardware Architecture

**核心问题：** 计算在哪里执行，输入与累加器存在哪里，数据如何移动？  
**前置：** Stage 1 的账本、带宽/延迟/吞吐；**投入：** 8～12h。

**P0 理论：** GPU/SM/计算管线、Thread/Warp/Block 的逻辑与物理关系；Register File、Shared Memory、L1/L2、Device DRAM；SIMT、Warp Scheduling、Latency Hiding、Memory Bandwidth、Tensor Core 矩阵运算。用经典 GEMM 解释：Global→Shared 加载输入 Tile，再由线程/Warp 利用片上数据在 Register 中累加并写回；同一 Tile 的输入支持多个输出元素。

**P1：** 每 SM Register/Shared 容量与驻留 Block 的限制、架构专属矩阵操作。**P2：** DRAM 时序、缓存微架构细节。

**可视化：** 一个 GPU/SM/Memory Hierarchy 图；在图上标注 Q/K Tile、累加器和数据搬移箭头。

**最小实验：**

1. 制作设备规格卡：GPU 型号、Compute Capability、SM 数、L2、Register/Shared 限制、DRAM 类型与带宽、对应 Dtype 的 Dense 吞吐。
2. 做大型连续 Copy 和 GEMM 的尺度扫描，记录有效带宽与有效计算吞吐；不同范围预期会受缓存与并行度影响。
3. 手算一个输出 Tile 所需 Register/Shared 规模，和硬件容量对照。

**阅读：** AI Infra Book 第四章 4.1～4.3；CUDA Programming Model；CUTLASS Efficient GEMM。

**验收：** 能区分 Warp 与 CUDA Core、解释 SM 的资源分配与 Warp 就绪机制；能画出至少一种 GEMM 数据流和复用途径，解释为什么峰值算力高仍可能受数据供给限制。

**产物：** GPU 规格卡 + 标注数据位置和复用路径的硬件图。

## 6. Stage 3：GPU Execution Model

**核心问题：** 同一 GEMM 如何映射为 CUDA Grid/Block/Warp/Thread，并通过 Tiling 减少重复访问？  
**前置：** Stage 2、基本 C++/CUDA；**投入：** 14～20h。

**P0 理论：**

| 版本 | 分工 | 核心机制 |
|---|---|---|
| Naive CUDA GEMM | 每个 Thread 计算一个输出元素 | 全局访问、地址映射、Register 累加 |
| Shared-Tiled CUDA GEMM | 一个 Block 计算一块输出 | 合并访问、Shared Tile、同步与输入复用 |
| Tensor Core GEMM | Warp/协作线程执行矩阵运算 | Fragment/Operand Layout、MMA、额外片上资源 |

单个 GEMM Tile 的输入量与 FLOPs：

$$
F_{\mathrm{tile}}=2m_tn_tk_t,\qquad
\mathcal V_{\mathrm{tile,input}}=s(m_tk_t+k_tn_t).
$$

更大的 Tile 可以提高单次输入加载的计算复用，但增加 Register/Shared Memory 压力，减少驻留并行度；Occupancy 应和实际吞吐一起分析。

**P1：** Register Blocking、Bank Conflict、Warp Shuffle、Double Buffer。**P2：** CUTLASS/CuTe 高级 Layout 与架构特定调度。

**可视化：** 输出 Tile 的 Block/Warp/Thread 映射图；一个 Warp 的 32 个访存地址；加载、同步、计算、覆盖的时间图。

**最小实验：**

1. 先运行已有 scale_cuda.cu 的连续/跨行地址映射，对比 Coalescing，并手算 Warp 地址。
2. **待开发** Naive $QK^\top$ CUDA Kernel，再做 Shared-Tiled 版本；统一输入 FP16、FP32 累加、输出 FP16，加入非整齐边界 Shape 与数值检查。
3. **待开发** WMMA 版本，观察矩阵协同指令；对比 Naive/Tiled/WMMA 的时间和资源消耗。
4. 复用代码执行 $PV$，对照 $QK^\top$ 的输出形状、归约维与 Tile 数量；现有 triton_qk.py 可以先用于理解分块复用，不等同于完整 CUDA 三版本实验。

**阅读：** CUDA Best Practices 的 Coalescing/Shared Memory GEMM；WMMA；CUTLASS Efficient GEMM；AI Infra Book 5.2。

**验收：** 能指出每个输出元素由谁算、累加器在哪里、Barrier 保护何种依赖、哪些数据被重用，以及为什么 Tile 增大未必更快。

**产物：** 三种 GEMM 正确性、资源与性能对照；新的 CUDA GEMM 脚本属于后续任务。

## 7. Stage 4：PyTorch to GPU Execution

**核心问题：** 从 Python API 追踪到实际 Kernel、CUDA 库和 GPU 机器指令。  
**前置：** Stage 3、PyTorch 基本算子机制；**投入：** 10～14h。

**P0 理论：** Python API→ATen/Dispatcher→CUDA Backend/库调用→Kernel Launch；CUDA Stream 异步提交；PTX 虚拟 ISA 与具体架构 SASS 的区别；SDPA 在 Math、Flash、Efficient、cuDNN 等可用 Backend 间按运行时约束选择。

**源码入口：**

- [PyTorch Attention 入口](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp)：scaled_dot_product_attention 分支。
- [SDPA CUDA 选择](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)：条件检查与优先级。
- [Blas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp)：bmm / GEMM 调用路径。

阅读时以安装版本的 torch.version.git_version 找对应 Commit；上述 main 链接是源码导航，不直接代表本机执行版本。

**P1：** 将 Stage 3 自己的 GEMM 注册为 PyTorch Custom C++/CUDA Operator，再完整追踪源码、PTX、Cubin、SASS。**P2：** 深入 Dispatcher Key 优先级、复杂 JIT/编译器流程。

**可视化：** 同时标注 CPU 算子、CUDA API、GPU Kernel、Stream 同步的 Timeline 与对应源码图。

**最小实验：**

1. 现有 attention_lab.py 的 profile 模式：观察 QK、Scaling、Softmax、PV 的实际 Operator/Kernel；使用 Nsight Systems 查 CPU 提交与 GPU 时间线。
2. 现有 attention_lab.py 的 sdpa 模式：对照 Eager、强制 SDPA Math/Flash；不支持时记录具体失败，而不是把 fallback 误记为成功。实际 Kernel 名称以 Profiler 为准。
3. 对已有 scale_cuda.cu 使用 NVCC 生成 PTX、cuobjdump/nvdisasm 检查二进制；标出 Load/Store、运算、同步。之后追踪自己注册的 GEMM，避免一开始解析大型 cuBLAS 内核。

**阅读：** PyTorch CUDA Semantics、Custom C++/CUDA Operators；CUDA Binary Utilities、PTX ISA；AI Infra Book 5.1。

**验收：** 能说明为什么 Python 语句与 Kernel 数不一定一一对应；能定位当前 PyTorch 的 SDPA 后端并给出 Trace 证据；能完成一个自写 Kernel 的 API→源码→指令路径。

**产物：** PyTorch 执行链与 Nsight Systems Timeline；一个可控 Kernel 的 PTX/SASS 注释。

## 8. Stage 5：GPU Performance Modeling

**核心问题：** 理论上限是什么，实际性能为什么没有达到？  
**前置：** Stage 1、3、4；**投入：** 10～14h。

**P0 理论：**

$$
I=\frac{F}{\mathcal V},\qquad
T_{\min}=\max\left(
\frac{F}{P_{\mathrm{compute}}},
\frac{\mathcal V}{BW_{\mathrm{memory}}}
\right).
$$

$$
P_{\mathrm{attained}}\le
\min\left(P_{\mathrm{compute}},BW_{\mathrm{memory}}I\right).
$$

上述简化模型是理想计算/数据传输重叠条件下的资源下界。算力上界必须匹配 FP16 Tensor Core 或具体标量执行路径；DRAM Bytes 与 DRAM Bandwidth 配对，L2 Bytes 与 L2 Bandwidth 配对。Scaling、Exp、Reduction 的执行单元和瓶颈需另行考察。

显式 Attention 的不同 Kernel 串行执行时，使用逐 Kernel 模型比整体平均 AI 更有解释力：

$$
T_{\mathrm{GPU}}\gtrsim
\sum_k\max\left(\frac{F_k}{P_k},
\frac{\mathcal V_{\mathrm{DRAM},k}}{BW_{\mathrm{DRAM}}}\right)+T_{\mathrm{gap}}.
$$

$T_{\mathrm{gap}}$ 表示 Timeline 中不可与这些 Kernel 重叠的可见间隙；不能将所有 CPU Launch 时间直接相加到 GPU 时间上。

**P1：** DRAM/L2 分层 Roofline、Register Spill、Occupancy、Wave 数量、Pipeline Stall。**P2：** 架构特定性能计数器体系和详细指令吞吐模型。

**可视化：** Kernel Timeline；时间与 DRAM/L2 流量占比；带实测点的 Roofline；Tile Size 对时间与资源的影响。

**最小实验：**

1. 制定测量协议：固定输入、版本、Seed、Dtype 与频率/功耗条件；预热、重复，记录 Median/离散程度；CUDA Event 与同步 Wall Time 分开。
2. 用 PyTorch Profiler/Nsight Systems 定位 Kernel；用 Nsight Compute 选取少量必要指标，例如 DRAM Read/Write、L2、Achieved Occupancy、寄存器和计算管线利用。指标名通过本机版本查询。
3. 比较同一输入反复运行与轮换大工作集，观察热缓存；区分 Nsight Compute Replay/Cache Control 与正常应用。
4. 选择一项有明确预测的干预：改变 Tile Size、增加独立工作量、调整布局或比较 Fusion；用实测证据验证假设。

**阅读：** AI Infra Book 4.8；Nsight Compute Profiling Guide；PyTorch CUDA Semantics。

**验收：** 能说明逻辑 Traffic 和实际 DRAM Bytes 的差异、峰值分配与流量的区别、Profiler 如何扰动计时；能给出一次“预测→观测→解释→验证”的闭环。

**产物：** 单 Kernel/整体运行的性能模型与指标表，附一个经测试的瓶颈归因。

## 9. Stage 6：Understanding FlashAttention

**核心问题：** 如何通过 Online Softmax 与分块执行减少 $N^2$ 中间结果的显存搬移，并随硬件演进优化？  
**前置：** Stage 1、3、5；**投入：** 16～24h。

**P0 理论一：Online Softmax。** 对一行 Query 按 KV 块处理，已有状态为 $m,\ell,u$。新块的 Scaled Scores 为 $x_j$，则：

$$
m'=\max\bigl(m,\max(x_j)\bigr),\qquad
\alpha=e^{m-m'},\qquad p_j=e^{x_j-m'}.
$$

$$
\ell'=\alpha\ell+\sum p_j,\qquad
u'=\alpha u+p_jV_j.
$$

全部 KV 块结束后 $o=u/\ell$。当新块有更大的分数时，必须同时调整历史分母 $\ell$ 和分子 $u$ 的尺度。计算可以在片上维护这些状态，减少完整 $P$ 的显存物化。

**P0 理论二：IO-Awareness。** FlashAttention 的真实流量依赖 Tile 复用和循环顺序。举例，主案例若用 $B_r=64$，Q Tile 常驻且每个 Query Tile 完整遍历 KV，16 次读取 K/V：

- Q 读和 O 写为 4 MiB；
- K/V 每轮 4 MiB，合计 64 MiB；
- 逻辑 Tile 访问约 **68 MiB**，暂不计小型统计量与实现开销。

它是一个**指定循环顺序下的逻辑访问示例**，不是 FA 固定的 DRAM Traffic；L2 命中与更复杂排程会改变外部搬移。

**版本主线：**

| 版本 | 瓶颈 → 设计 → 硬件映射 | 优先级 |
|---|---|---|
| FA1 | $N^2$ 中间矩阵 IO → Online Softmax + Tiling + Fusion → 减少中间矩阵显存访问 | P0 |
| FA2 | 非 Matmul 工作、并行度与 Warp 共享通信 → 更好的 Query Block/Warp 分工与更少非 Matmul 工作 | P0 |
| FA3 | Hopper 的异步矩阵乘/搬移能力未充分重叠 → TMA、WGMMA、Warp Specialization、Pipeline | P1 |
| FA4 | Blackwell 等新架构中的非矩阵操作、Shared Memory 与流水限制 → 改进 Pipeline、Exp/重缩放和资源安排 | P1；深入源码 P2 |

不同 FA 版本的支持硬件、具体实现分支和 PyTorch 内置 SDPA Backend 标签不等价，逐次测试记录 Commit、安装包与实际 Kernel。

**P1：** Warp-Level Parallelism、Buffering、跨阶段重叠、Backward 的重算策略。**P2：** FP8/更低精度、架构特定 CuTe、Distributed/Long-context Attention。

**可视化：** Eager vs Fused 的数据流；Online Softmax $m,\ell,u$ 更新动画/表；FA2 与 FA3 Pipeline 时间线。

**最小实验：**

1. 现有 verify_models.py 和 attention_lab.py 的 online 模式：验证分块递推，尤其让最大分数出现在后面的块；再引入非整除 Tile。
2. 通过同一输入对照 Eager、PyTorch SDPA Math/Flash、明确版本的外部 FlashAttention，逐项记录输出误差、时间、内存和 Kernel 名称。注意外部 API 布局可能为 $[B,N,H,D]$，单独记录转置及 contiguous 成本。
3. 阅读 Triton Fused Attention Forward，并针对一个 Tile 参数做单变量实验，预测 Register/Shared 与时间变化。
4. 对照 [FA2 Forward](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)、[Softmax](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/softmax.h)、[FA3 Hopper Mainloop](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp)、[FA4 Blackwell](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py) 的目标架构与特定 Forward 分支，标记输入加载、MMA、Softmax、输出、等待/同步。

**阅读：** AI Infra Book 5.3；FA1/2/3/4 论文；Triton Fused Attention。

**验收：** 能证明 Online Softmax 的更新正确；从源码指出数据在何处保留及重复读取；解释 FA1→FA2→FA3→FA4 的瓶颈迁移，并以一个可测实验说明某项优化为何有效。

**产物：** 算法、IO、硬件映射、Profiler 与源码路径相互对应的 FlashAttention 技术分析。

## 10. 实验实施及记录模板

每次实验至少保存：

| 类别 | 必备字段 |
|---|---|
| Workload | $B,H,N_q,N_k,D,D_v$、Mask、Dropout、Forward/Backward、数据分布 |
| 数值 | 输入/输出 Dtype、累加与 Matmul 设置、Reference、Max-Abs、Relative-L2 |
| 实现 | 代码 Commit、PyTorch/Triton/CUDA/FA 版本、SDPA Backend、是否 JIT |
| 设备 | GPU/CC、SM、DRAM 类型、相关时钟、驱动版本 |
| 测量 | 热身次数、重复次数、时间边界、CUDA Event/Wall Time、Profiler 是否开启 |
| 资源 | FLOPs、Footprint、逻辑流量、DRAM/L2 指标、Register/Shared、Grid/Block |
| 结论 | 先验预测、观察证据、偏差解释、单变量验证、未证实的问题 |

建议流程：

1. 写下要验证的机制与预测变化。
2. 用小 Shape 验证正确性。
3. 用无 Profiler 运行记录标准时间。
4. 用 Profiler/NCU 采集能回答问题的最少指标。
5. 将指标与公式对照，改变一个变量复验。
6. 记录结果与下一步问题；GPU 结果未执行则写“待验证”，不填推测值。

仓库已经存在 attention_lab.py、scale_cuda.cu、triton_qk.py、triton_softmax.py、verify_models.py；详细使用命令见 [[10_Projects/attention-gpu-learning-lab/README|入门实验说明]]。Stage 3 的三套原生 CUDA GEMM、Stage 4 的自定义 PyTorch Operator 属于本路线新增的后续实现任务。

## 11. 最终验收项目

在主案例之外选一个长 Prefill Shape 和一个 $N_q=1$ 的 Decode Shape，至少比较 Eager、PyTorch SDPA 与一个明确版本的 FlashAttention。

完整报告必须包含：

- **计算预测**：各子操作 FLOPs、Tensor/Traffic 估算与瓶颈假设。
- **执行映射**：Dispatcher/Backend、实际 Kernel、Block/Warp、存储层次与关键指令。
- **数据证据**：逻辑字节与 DRAM/L2 的实测区别、缓存/重复读取归因。
- **时间证据**：端到端时间、Kernel Timeline、逐 Kernel 的模型与实测偏差。
- **优化验证**：一次针对单个假设的干预及效果。
- **源码定位**：至少一个关键优化对应到原论文、源码位置和硬件行为。

**最终验收：面对新 Shape，先预测、再测量、用源码和硬件计数器解释差异，并用最小实验验证归因。**

建议从 Stage 1 的三项交付开始：资源账本、Stride/View 核验、Prefill/Decode 对照。后续实践可在本项目下按需要增加实验记录与阶段笔记，无须预建空目录。
