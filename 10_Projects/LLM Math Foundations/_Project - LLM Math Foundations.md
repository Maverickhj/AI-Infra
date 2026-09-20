---
type: project
status: active
created: 2026-09-05
updated: 2026-09-20
domains: [mathematics, llm]
ai_generated: true
reviewed: false
---

# LLM Math Foundations

> [!warning]
> 本项目的 roadmap 为 AI 整理的教学设计，尚未完成人工核验；规划内容不代表已掌握。

从基础数学对象与定义出发，建立线性代数、微积分、概率论之间的三个连接，最终能够解释论文公式并手工推导基本结果。课程独立于论文阅读进度。

## 工作入口

- [[10_Projects/LLM Math Foundations/MISSION|Mission]]：学习目标与边界。
- [[10_Projects/LLM Math Foundations/ROADMAP|Roadmap]]：概念依赖、三个连接与阶段检索目标。
- [[10_Projects/LLM Math Foundations/RESOURCES|Resources]]：已核对的教材与课程入口。
- [[10_Projects/LLM Math Foundations/NOTES|Teaching notes]]：学习偏好与接续规则。
- [[10_Projects/LLM Math Foundations/exercises/01-linear-algebra/01.01-geometry-to-tensors/problem/readme|PyTorch practice — Geometry to tensors]]：12 道练习、API map、可运行 starter 与独立参考答案。

## 当前状态

2026-09-20：用户自述已学完 3Blue1Brown 的 *Essence of Linear Algebra*，现在通过 PyTorch 练习连接几何理解与计算。下一步先完成练习 01–03，用预测、代码与解释定位需补强的概念。视频学习完成不等于通过独立 retrieval；阶段掌握状态仍由 roadmap 与实际学习证据维护。

本机共享测试环境：`C:\Users\hyftu\.venvs\torch\Scripts\python.exe`（Python 3.12.14、PyTorch 2.14.0+cpu）。这是 Windows 本机环境，跨设备需自行提供 Python/PyTorch；详细运行命令见练习入口。

后续按需建立 `lessons/`、`reference/`、`learning-records/` 与 `assets/`；当前不创建空目录。课程使用 teach 工作流，每次只讲一个紧凑单元。

## 与现有课程的关系

[[10_Projects/LLMZeroToHero Paper Learning/_Project - LLMZeroToHero Paper Learning|LLMZeroToHero Paper Learning]] 是后续应用检验入口。本项目保留独立的学习状态；既有论文课程中的阅读或答疑不自动作为本项目的掌握证据。已有数学参考可以按需链接，当前不迁移或复制。
