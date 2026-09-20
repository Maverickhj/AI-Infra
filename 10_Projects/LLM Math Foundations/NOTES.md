---
type: project
status: active
created: 2026-09-05
updated: 2026-09-05
ai_generated: true
reviewed: false
---

# Teaching notes — LLM Math Foundations

## 用户已明确的目标与偏好

- 2026-09-05：希望自底向上建立线性代数、微积分、概率论的融合直觉，并能手工推导基本公式。
- 2026-09-13：描述数学领域内容时尽量使用 English；必要时可全英文表达，并在晦涩处补充中文说明。
- 无须遵守当前论文学习进度；数学内容按自身依赖组织。
- 本次请求是新建 teach 项目与 roadmap，三个连接为主轴。尚未要求开始首课。

## 教学约定

- 数学内容采用 English-first 表达，必要时使用全英文；在晦涩或容易误解的地方补充中文说明。首次出现的关键术语给简短中文解释，后续保持一致。
- 先解释数学对象与定义，再给直觉和推导；低维算例后推广一般 shape。不能以直觉替代成立条件。
- 学习状态以本项目 roadmap 与 learning records 为准；尚未诊断基础，不从其他项目进度推定掌握。
- 后续 lesson 使用 teach 的 HTML 小课形式；首次建课时建立共享 stylesheet，公式用 MathJax。Reference 提炼已教内容，按需维护 glossary；当前不生成占位课件或空目录。
- Mathematical foundations 保持 model-agnostic；LLM 应用放在阶段检验中。已有论文项目的参考资料优先链接，避免复制后形成两个版本。
- Markdown 使用 `$...$` / `$$...$$`；HTML 使用 `\(...\)` / `\[...\]`。每个公式注明对象、shape、范围与假设。
- 记录独立检索的实际表现与提示程度；首次答对和间隔后仍能推导分开记录。不提前创建学习成果记录。

## 下一次接续

读取 Mission、Roadmap 和本文件；如已有 learning records，先核对。由 P0 的短诊断确定首个小课，按需检查 P1/P2 起点。阶段编号不等于 lesson 编号，不一次生成整套课程。
