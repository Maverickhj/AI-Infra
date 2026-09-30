---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# P1：整模学习体验

输入：models.json、三篇完整 case、source-evidence.json。

输出：PLAN 定义的 Model Atlas、Sample Journey、Model Walkthrough、SFT/RL 场景页的第一版。允许合并为少量路线和标签页，不允许只有空路由。

行为验收：Qwen3 的 Q 宽2048、fused QKV4096、FFN gate/up各3072；Qwen2.5 的 QKV bias 打开/QK norm关闭；DeepSeek layer0–2 dense、后续MoE；MLA训练与decode两种图分开；点击步骤能定位source；所有没有采集的曲线区域明确not_run。

对已完成页面做用户路线测试，不只 snapshot 或组件 mount 测试。首轮不要求 GPU、Torch 或 Bridge 安装。读者为有一定基础的初学者，基本内容一页可跳过，核心仍按数学/shape→源码→演算→反例→验证精讲；浏览器测试增加“跳过基础速览直达整模核心”。
