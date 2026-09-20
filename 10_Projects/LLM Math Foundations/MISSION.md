---
type: project
status: active
created: 2026-09-05
updated: 2026-09-05
ai_generated: true
reviewed: false
---

# Mission: LLM Math Foundations

## Why

用户希望自底向上建立稳固的数学直觉：阅读 LLM 论文公式时，能理解数学对象、操作与成立条件，并手工推导基本公式。通过线性代数、微积分、概率论的连接，形成可迁移到新公式的理解。

## Success looks like

- 面对新公式，能标注符号、shape、求和或求导对象，并给出小规模例子。
- 能解释三个连接：导数作为局部线性映射；covariance 作为方向性波动；likelihood 如何连接 loss 与优化。
- 能独立推导简单复合函数的 Jacobian、线性投影的方差、least-squares gradient，并陈述所需假设。
- 能在间隔后的变式题中重新推导，区分定义、精确等式、近似和经验规则。

## Constraints

- 按概念依赖自底向上组织，不受现有 AdamW 或其他论文进度约束。
- 中文讲解、English-first 术语；保留标准数学符号，以低维例子、几何解释和手推建立理解。
- 每次只推进一个小单元，用独立 retrieval 检查；具体起点与节奏依据诊断，不预设学习时长。

## Out of scope

本轮 roadmap 暂不安排完整分析学或测度论课程、优化器收敛证明、LLM 训练复现。这是第一轮的教学取舍，后续可按需要扩展。
