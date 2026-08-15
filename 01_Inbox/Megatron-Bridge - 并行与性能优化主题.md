---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains:
  - megatron-bridge
  - performance
  - parallelism
aliases:
  - 并行与性能优化
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/parallelisms.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/performance-guide.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/moe-optimization.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/communication-overlap.md
ai_generated: true
reviewed: false
---

# Megatron-Bridge - 并行与性能优化主题

> [!todo]
> 待整理：从 `01_Inbox/` 中拆分主题；项目专属记录留在 `10_Projects/Megatron-Bridge/Notes/`，成熟且通用的结论可进入 `20_Knowledge/` 或由 `80_MOCs/` 汇总。

## 来源

- [parallelisms.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/parallelisms.md)
- [performance-guide.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/performance-guide.md)
- [moe-optimization.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/moe-optimization.md)
- [communication-overlap.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/communication-overlap.md)

## 待整理要点

- TP、PP、DP、CP、EP 等并行策略的选择与组合
- MoE 训练的通信、dispatch/combine 与优化路径
- 通信计算重叠、内存优化、性能定位的常用方法

## 整理时注意

- 只保留公开安全信息，不写入私有路径、密钥或实验日志
- 以 Megatron-Bridge 文档为源；整理完成后再决定是否保留指向原仓库的链接
