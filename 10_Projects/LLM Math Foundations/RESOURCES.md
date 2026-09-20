---
type: project
status: growing
created: 2026-09-05
updated: 2026-09-05
ai_generated: true
reviewed: false
---

# LLM Math Foundations Resources

## Knowledge

以下课程入口、相关先修说明与教材目录于 2026-09-05 打开核对。资源用于按主题选读；roadmap 的顺序和下述使用建议是本项目的 AI 教学设计，尚未完成人工核验，不是来源给出的统一课程要求。

- **R1**：[MIT OCW — 18.06 Linear Algebra, Spring 2010](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)，Gilbert Strang。用于 P1、C2 的线性映射、向量空间与矩阵基础；从课程讲义、视频和题目中选取当前单元所需部分。
- **R2**：[MIT OCW — 18.01SC Single Variable Calculus, Fall 2010](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)。用于 P2 的导数定义、局部近似、积分与 Taylor 基础；提供自学材料和例题。
- **R3**：[MIT OCW — 18.02SC Multivariable Calculus, Fall 2010](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/)。用于 C1 的多变量微分基础；按需要选择相关单元，不默认展开完整向量分析课程。
- **R4**：[MIT OCW — 6.041SC Probabilistic Systems Analysis and Applied Probability, Fall 2013](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/)。用于 P3、C2、C3 的概率建模基础，从条件概率、随机变量逐步进入联合分布与统计推断。
- **R5**：[Mathematics for Machine Learning — 作者网站](https://mml-book.github.io/)，Marc Peter Deisenroth、A. Aldo Faisal、Cheng Soon Ong，Cambridge University Press，2020。作为三门数学的连接读物：第 2–3 章线性代数与几何，第 4 章矩阵分解，第 5 章 vector calculus，第 6 章概率，第 7 章优化，第 9 章 linear regression。作者网站提供公开 PDF 与勘误入口。
- **R6**：[3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)，Grant Sanderson。用于 P1、C2 的几何直觉，重点查向量、span、基、linear transformation 与矩阵复合；建议看完后用 R1 的相关题目检验理解。
- **R7**：[3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus)，Grant Sanderson。用于 P2 的变化与累积直觉；导数、积分及二者联系难以想象时选看，再回 R2 做例题与手推。

### 入门选择与先修要求

建议先用 **R6 + R1** 建立线性代数的图像与计算能力；进入微积分时用 **R7 + R2**。R3、R4 在相应基础具备后使用；R5 用于把不同学科的概念连接起来。

| 课程 | 官方先修说明 | 本项目如何使用 |
|---|---|---|
| R1 · MIT 18.06 | [Syllabus](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/syllabus/) 列出 18.02 Multivariable Calculus | 本项目建议先选读向量、矩阵、子空间、投影等基础部分；遇到微分方程等应用时补先修。这是选读安排，不是声称官方无先修要求 |
| R2 · MIT 18.01SC | [Syllabus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/syllabus/) 要求高中代数与三角函数，微积分经历有帮助但非必需 | 作为微积分起点；先核对 P0 的函数、指数、对数和三角函数基础 |
| R3 · MIT 18.02SC | [Syllabus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/syllabus/) 要求 18.01 Single Variable Calculus | P2 后进入多变量微分，支持 C1；其他单元按 roadmap 需要选读 |
| R4 · MIT 6.041SC | [Syllabus](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/syllabus/) 要求 18.01 与 18.02，涉及极限、级数、chain rule 和积分工具 | 作为具备微积分基础后的概率论主课；离散概率部分可先选读，连续与多变量部分检查先修后推进 |

### 按问题查阅

| 当前卡点 | 先找哪里 | 查阅后做什么 |
|---|---|---|
| 向量、基、矩阵乘法没有几何图像 | R6 中 vectors、span、linear transformations 相关内容 | 用二维向量手算，并画出变换前后位置 |
| 子空间、投影、least squares 的推导跳步 | R1 的相关讲课、阅读材料与习题 | 补全一段推导，再做一道同主题题目 |
| 不理解 derivative、integral 或局部近似的含义 | 先 R7，再 R2 的对应单元 | 用一个简单函数解释变化或累积，并手推结果 |
| 单变量求导规则会背但不会用 | R2 的 worked examples 与 problem sets | 合上例题重推，再换一个函数练习 |
| 多变量 chain rule、gradient 或 Jacobian 的 shape 混乱 | R3 的多变量微分内容，配合 R5 第 5 章 | 标注输入输出维度，写出局部线性映射和复合顺序 |
| 条件概率、分布、expectation 或 variance 不清楚 | R4 的相关概念与例题 | 从一个小型离散分布算起，再推广连续情形 |
| covariance 的矩阵形式难以理解 | R4 的概率基础，配合 R5 第 6 章和 R1 的几何基础 | 计算二维随机向量的 covariance，再比较投影方差 |
| likelihood、loss 与优化联系不起来 | R5 第 6、7、9 章 | 写出概率假设、目标函数与求导对象，再逐步推导 |

推荐查阅节奏：**概念没有图像 → 直觉视频；推导跳步 → 讲义与 worked example；看懂却写不出 → 做题后合上答案重推。** 不要求先刷完整套课程；学习顺序仍由 [[10_Projects/LLM Math Foundations/ROADMAP|Roadmap]] 维护。

后续每个 lesson 应附上实际使用的具体课程单元或教材章节，说明“查什么、解决哪个疑问、做哪一个练习”。当前表格是主题导航；未定位到具体页面的内容，备课时再核对并补充，不编造课号或时间戳。

## Wisdom (Communities)

暂未选择社区，用户尚未表达加入社区的偏好。当前先通过手推与讨论建立基础；出现需要外部交流的问题时再选择适当渠道。

## Gaps

- 当前核对的是课程主页、先修说明与教材目录，尚未逐页审阅全部内容，也未逐段观看视频。每次备课需补充该课直接使用的具体章节或页面，并核对公式、假设和例题。
- P0 的基础代数诊断结果未知；若发现先修缺口，再补相应基础资源。
- Softmax / cross-entropy 的具体推导来源尚未定位到章节；在生成对应 lesson 前补齐。
- 资源以英文为主；中文讲解由课程提供，不把 AI 翻译或解释当作独立事实来源。
