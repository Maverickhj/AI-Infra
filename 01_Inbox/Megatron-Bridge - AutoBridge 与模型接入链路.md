---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains:
  - megatron-bridge
  - model-integration
aliases:
  - AutoBridge 模型接入
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/adding-new-models.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/bridge-tech-details.md
ai_generated: true
reviewed: false
---

# Megatron-Bridge - AutoBridge 与模型接入链路

> [!todo]
> 待整理：从 `01_Inbox/` 整理到 `10_Projects/Megatron-Bridge/Notes/`；若提炼为通用接入方法论，再上移到 `20_Knowledge/`。

## 来源

- [adding-new-models.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/adding-new-models.md)
- [bridge-tech-details.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)

## 待整理要点

- `AutoBridge` 如何根据 HF 模型名或路径选择对应的 `<family>/bridge.py`
- 每个模型族涉及的 `bridge.py`、`config_mapping.py`、`param_mapping.py` 与 `recipes/<family>/`
- 新模型接入的最小改动面、测试路径与验证方式

## 整理时注意

- 只保留公开安全信息，不写入私有路径、密钥或实验日志
- 以 Megatron-Bridge 文档为源；整理完成后再决定是否保留指向原仓库的链接
