# Curriculum: Paper–Lesson Scope

论文 → lesson / learning-record 编号范围的权威索引。新增、拆分、合并或重排 lesson 前，先改这里，再动文件。

## 编号规则

- `lesson` 与 `learning-record` 都用纯顺序整数，从 `0001` 递增；一个 `00NN` lesson 对应一个同号 `00NN` record。
- 一篇论文拆多课时占用连续整数；本表记录范围，文件名不再引入字母后缀。
- 桥接 lesson（挂在某篇主线论文下的单课，如 Transformer 主线里的 `0009`）归属主线，不单独出 learning-record。

## 映射表

| 论文 | 线路 | Lesson 范围 | Learning record 范围 | 状态 |
|---|---|---|---|---|
| Attention Is All You Need `1706.03762` | ★ 主线 | `0001–0012` | `0001–0008`、`0010–0012` | 已讲；`0012` Cross-Attention retrieval `[待核验]` |
| Adam: A Method for Stochastic Optimization `1412.6980` | 支线（挂在主线 `0009` 的优化器依赖） | `0009a–0009c` | `0009a–0009c` | 已讲 |
| Decoupled Weight Decay Regularization (AdamW) `1711.05101` | ★ 主线 | `0013–0014` | `0013–0014` | 0013 已讲；0014 已建，待 retrieval |
| RoFormer: Rotary Position Embedding (RoPE) `2104.09864` | ★ 主线 | `0015` 起（课数待定） | `0015` 起（课数待定） | 未开始 |

## 变更留痕

- 每次变动同步上表；不要在文件名或别处重复维护范围。
- 重大重排（顶号、合并、删除）时，在 `NOTES.md` 追加一条说明。
- 历史遗留：Adam 的 `0009a–0009c` 早于本规则建立，保持原样；新论文不再使用字母后缀。
- `2026-08-25`：`0013` 术语/记号按原文规范化——decay 率用 $\lambda$（Eq. 1），L2 penalty coefficient 用 $\lambda'$（Prop 1：$\lambda'=\lambda/\alpha$）；lesson 范围不变。
- `2026-08-26`：`0013` 新增前置阅读 `reference/math-l2-regularization-motivation.md`（model-agnostic 数学底座，由 `tmp` topic draft 升格，草稿已删除）；lesson 范围不变。
- `2026-08-31`：`0013` recall gate 通过并创建同号 learning record；`0014` 仍处于规划状态。
- `2026-08-31`：创建 `0014` normalized weight decay / warm restarts lesson 与同号 glossary；尚未创建 learning record。
