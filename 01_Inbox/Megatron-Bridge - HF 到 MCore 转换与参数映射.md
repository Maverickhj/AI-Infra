---
type: source
status: seed
created: 2026-08-15
updated: 2026-08-15
domains:
  - megatron-bridge
  - model-conversion
aliases:
  - HF 到 MCore 转换
  - 参数映射
source:
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/bridge-guide.md
  - https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/bridge-tech-details.md
ai_generated: true
reviewed: false
---

# Megatron-Bridge - HF 到 MCore 转换与参数映射

> [!todo]
> 待整理：从 `01_Inbox/` 整理到 `10_Projects/Megatron-Bridge/Notes/`；若形成可复用的转换知识，再上移到 `20_Knowledge/`。

## 来源

- [bridge-guide.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/bridge-guide.md)
- [bridge-tech-details.md](https://github.com/Maverickhj/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)

## 待整理要点

- HF checkpoint 到 Megatron/MCore checkpoint 的转换流程
- `config_mapping.py` 如何从 HF config 生成 Megatron config
- `param_mapping.py` 如何完成参数名映射与权重搬运
- 转换前后的校验方式与常见不匹配点

## 整理时注意

- 只保留公开安全信息，不写入私有路径、密钥或实验日志
- 以 Megatron-Bridge 文档为源；整理完成后再决定是否保留指向原仓库的链接
