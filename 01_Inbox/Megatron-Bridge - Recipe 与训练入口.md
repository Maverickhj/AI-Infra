---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains:
  - megatron-bridge
  - training
aliases:
  - Recipe 与训练入口
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/recipe-usage.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/entry-points.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/README.md
ai_generated: true
reviewed: false
---

# Megatron-Bridge - Recipe 与训练入口

> [!todo]
> 待整理：从 `01_Inbox/` 整理到 `10_Projects/Megatron-Bridge/Notes/`；其中可直接复用的启动命令可沉淀到 `40_Runbooks/`。

## 来源

- [recipe-usage.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/recipe-usage.md)
- [entry-points.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/entry-points.md)
- [training/README.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/training/README.md)

## 待整理要点

- `recipes/<family>/` 与训练 recipe 的组织方式
- 训练脚本入口、`torch.distributed.run` 启动方式
- 单机与多机的启动差异、最小 smoke test 命令

## 整理时注意

- 只保留公开安全信息，不写入私有路径、密钥或实验日志
- 以 Megatron-Bridge 文档为源；整理完成后再决定是否保留指向原仓库的链接
