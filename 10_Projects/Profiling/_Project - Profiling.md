---
type: project
status: active
created: 2026-10-27
updated: 2026-10-27
domains:
  - ai-infra
  - profiling
aliases:
  - Profiling Learning Project
ai_generated: true
reviewed: false
---

# Profiling 学习与实践

> [!warning]
> 这是用于导航的 AI 整理项目入口，尚未人工核验，具体 API、参数和性能结论以实验为准。

## 目标

形成从 **Monitoring / Benchmarking → CPU/GPU Timeline → Kernel / Memory / Communication → LLM Framework → 优化 A/B** 的 Profiling 能力。

## 资料

- [[LLM AI-Infra Profiling 技术体系调研]]：通用概念、核心机制、工具能力、四个可视化入门例子，以及 Megatron / SGLang / AReaL 的方法论。
- [[megatron_lm_profiling_debug_guide|Megatron-LM 训练引擎 Profiling & Debug 实践指南]]：Megatron-LM / Megatron-Bridge 训练场景的详细命令、显存、通信与排障。

## 练习

- [single_device_lab.py](Lab/single_device_lab.py)：CPU/GPU 时间线、显存生命周期、GEMM。
- [two_rank_lab.py](Lab/two_rank_lab.py)：Gloo/CPU 与 NCCL/CUDA 两 rank 等待传播。
- [四个概念可视化](LLM%20AI-Infra%20Profiling%20技术体系调研.assets/profiling-concepts.svg)：教学假设图，不代表实测。

## 下一步

- [ ] 实测两个入门脚本的 CUDA/NCCL 路径，并写明机器、驱动、torch 和 profiler 版本。
- [ ] 对 Megatron 小模型执行短窗口 Nsight Systems 采集，记录 rank/step 的关键路径。
- [ ] 给 SGLang 服务做固定请求分布下的无 profiler 基线与对照追踪。
