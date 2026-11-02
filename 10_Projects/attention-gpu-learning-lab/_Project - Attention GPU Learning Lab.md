---
type: project
status: active
created: 2026-11-02
updated: 2026-11-02
domains: [GPU, CUDA, PyTorch, Attention]
ai_generated: true
reviewed: false
---

# Attention GPU Learning Lab

> [!warning]
> 本项目学习方案由 AI 辅助生成，尚未完成人工核验。GPU 实验脚本尚未编译或运行，不包含实测 GPU 性能结论。

## 目标与范围

围绕同一个简化 Scaled Dot-Product Attention，建立从 Tensor 布局、PyTorch/ATen 调用到 GPU 执行及性能分析的完整理解，为阅读 FlashAttention 论文和 forward Kernel 做准备。

第一轮聚焦单 GPU、dense、non-causal、forward-only MHA；先理解数据流、线程协作、访存和数学模型，再进入融合与分块实现。

## 学习入口

- [[10_Projects/attention-gpu-learning-lab/README|完整学习路线、数学符号表、性能推导与阶段验收]]。
- GitHub 阅读入口：[README.md](README.md)。
- 来源链接、适用版本和架构边界均在完整路线中标明。

## 学习进度

- [ ] 阶段 0：Tensor 布局与数值语义。
- [ ] 阶段 1：CUDA scale 的线程映射与 Memory Coalescing。
- [ ] 阶段 2：PyTorch 调用链与 PTX/SASS。
- [ ] 阶段 3：计时、Profiler 与性能模型。
- [ ] 阶段 4：Triton scale + Softmax Fusion。
- [ ] 阶段 5：QK GEMM Tiling 与资源权衡。
- [ ] 阶段 6：Online Softmax 与 FlashAttention 前置验收。

## 实验文件

| 文件 | 用途 |
|---|---|
| [attention_lab.py](attention_lab.py) | Tensor 布局、数值对照、Profiler、长度扫描、online reference、SDPA 对照 |
| [scale_cuda.cu](scale_cuda.cu) | 原生 CUDA scale 的两种线程/访存映射 |
| [triton_softmax.py](triton_softmax.py) | 同一 Attention score 的 scale + Softmax 融合 |
| [triton_qk.py](triton_qk.py) | 同一 QK 运算的三种 query tile 配置 |
| [verify_models.py](verify_models.py) | CPU NumPy 核对公式、数值账和 online 递推 |
| [cpu_model_check.json](cpu_model_check.json) | 初次 CPU 模型验证的轻量结果摘要 |

## 已有验证与下一步

已完成 Python 语法检查、CPU NumPy 的 12 组分块递推验证、性能模型数字核算，以及 CUDA 两种索引映射的覆盖检查。CPU online reference 的最大绝对误差约为 5.55e-16。

GPU 编译、正确性和性能验证均待在实际设备执行。第一步按 README 的第一天安排，记录 GPU/CC、软件版本、Tensor 布局、实际 Kernel 与同步计时。

后续实验记录保留输入条件、准确命令、误差和测量口径；真实 trace、完整日志或大文件不直接放入 Git。
