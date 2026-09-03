---
type: project
status: draft
created: 2026-09-03
updated: 2026-09-03
ai_generated: true
reviewed: false
---

# Handoff — AdamW 课程接续

> [!warning]
> 本文是 AI 整理的接续说明，不是新的学习掌握证明或技术事实来源。先核对当前文件与 Git 状态，再继续工作。

## 本次交接与下一步

当前会话的最后请求是：生成 handoff，并将它与 AI-Infra 工作区全部现有改动一起 commit、push 到 origin。本文先生成于系统临时目录；应用户要求，仓库中保留随提交分发的副本。提交与推送是否成功以 Git 记录和调用任务最终回复为准，不从本文预设成功。

下一教学接续点是 **Lesson 0014 的阅读答疑与 retrieval**，不是创建下一课。先读 [CURRICULUM.md](CURRICULUM.md)、[MISSION.md](MISSION.md)、[NOTES.md](NOTES.md)；进度、目标、术语规则和编号以它们为准。

## 最小接续阅读集

下列路径相对于本 handoff 的仓库副本所在课程目录。

- [0013 learning record](learning-records/0013-adamw-decoupled-weight-decay.md)：已有 retrieval 证据与系数换算纠错，不需再抄录本次手写答案。
- [Lesson 0014](lessons/0014-adamw-normalized-weight-decay-warm-restarts.html)：当前待学内容与末尾 recall gate。
- [0014 glossary](reference/0014-adamwr-glossary.html)：本课词汇和符号。
- [Math Foundations Index](reference/math-foundations-for-attention.html)：新增 prerequisite anchors 的入口。
- [RESOURCES.md](RESOURCES.md)：原论文与版本入口。
- [_Project - LLMZeroToHero Paper Learning.md](_Project%20-%20LLMZeroToHero%20Paper%20Learning.md)：项目导航；精确进度优先核对 curriculum 和 learning records。

这轮已形成 AdamW 两课、各课独立 glossary、L2 前置阅读与通用数学补充。具体改动直接看上述文件及本次提交 diff，不在 handoff 重复正文。

## 会话中还没有完成的事项

- Lesson 0014 没有用户最终复述证据，尚未创建同号 learning record。用户提问或阅读材料不能当作通过 recall gate。
- 旧课程的 Cross-Attention final retrieval 状态仍按 curriculum 保留，不因本次提交而自动改变。
- 对话抽取涉及 proportionality、reparameterization 与纯乘性收缩；原讨论为 ChatGPT 素材，不是一手证据。讨论入口为 [解释权重衰减缩放](chatgpt-conversation://6a956820-e144-83e9-9562-071927bd8378)。已整理内容由 Foundation、Lesson 与 glossary 分别持有。
- 未纳入那段对话提及的后续 weight-decay scaling 论文结论；如后续需要，重新查一手来源，不照抄聊天中的引用或推断。
- 默认继续中文、English-first 术语、按需数学支架和短 retrieval。持久化规则见 NOTES，不迁移旧课 glossary。

## 继续教学前的核验提醒

历史工作主要做了静态 HTML parsing、数学定界符计数、本地文件 / anchor 检查和 whitespace 检查；这些不能证明 MathJax 浏览器运行正常，也不是完整数学审稿。此前曾请求打开本机端口 8000 的课程页，但不要假定临时服务器在新会话中仍运行。

以下属于后续教学时应回原文核验的具体边界，本次 Git 归档不顺带修订课程：

- [Lesson 0013](lessons/0013-adamw-decoupled-weight-decay.html) 的 “Algorithm 2” 使用了 learning-rate-scaled decay 写法；Lesson 0014 使用论文的 schedule-multiplier 写法。跨课讲解前需明确两处 decay coefficient 的换算，不能把同名符号当作同一数值。核对原论文 Algorithm 2，避免将“先 adaptive step 再乘法收缩”误写成对旧参数计算的加法更新。
- [L2 前置阅读](reference/math-l2-regularization-motivation.md) 的 MAP 目标常数因子、bias–variance decomposition 的非线性边界和“Adam 改变先验”的表述仍需要独立技术复核；之前的格式修复不代表这些断言已验证。
- Learning record 记录已经表现出的理解；其中教师补充的边界不应全部反推成用户做过独立 retrieval。

## Suggested skills

下一 agent 应按实际任务读取并使用对应 SKILL.md（有 Skill 工具时可通过它加载）：

- **teach**：继续 stateful teaching、核对 mission 与 learning records、处理 Lesson 0014 recall gate。检查用户安装的 `~/.agents/skills/teach/SKILL.md` 是否存在；不存在时不要声称已调用。
- **handoff**：需要再次压缩会话时使用；默认输出系统临时目录，随仓库分发须有用户要求。
- 浏览器控制能力：仅在需要实际检查 HTML / MathJax 渲染时发现并使用当前可用工具，不把“打开请求已排队”说成视觉验收通过。

## Git 与范围

本次用户授权的是 **AI-Infra 当前工作区整体归档**，包括已有编辑器端口设置、课程文件、数学参考、图片和本 handoff；不包括相邻源码仓库，不自动授权后续 push、内容重构或历史改写。

新会话先运行 Git root / status 检查。若本次 push 失败，只报告或排查实际原因；不要 force push、自动 rebase 或覆盖远端。不要保存密钥、登录资料、完整聊天转录或临时附件。
