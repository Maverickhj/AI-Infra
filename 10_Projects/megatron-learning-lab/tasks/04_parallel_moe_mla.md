---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# P4：并行与 MoE/MLA 的真实案例

不新建与完整模型无关的并行课。沿相同Qwen/DeepSeek模型和SFT/RL样本，叠加TP、DP、PP、SP、CP、EP。

每个 profile 只启用和验收当前 case 需要的能力，不要求环境支持全部 TP/CP/EP/FP8 组合。比较 logical shape、rank local layout、collective、源码分支和数值。TP合法性按该版本构建/分片逻辑校验，不写死“KV heads < TP一定不支持”。量化、TP>KVheads与特殊路由路径需要单独标签。

顺序：Qwen dense短序列TP/DP → PP/SP/CP packing → Qwen3 MoE EP → DeepSeek-V2-Lite导入/小实验 → DeepSeek-V3原结构或明确标注architecture-scaled实验。全量V3下载和运行不作为自动任务。

MLA训练图与decode缓存图分开；DeepSeek q_lora_rank=null与非null两条路径；shared expert独立/overlap路径；MoE辅助目标与router偏置；EP/ETP/EDP实际group关系均以源码及运行记录为准，不将EP机械乘进dense world-size公式。

最后独立运行性能profile：prefill/decode、forward/backward、通信overlap、显存峰值。数值对齐的重hook配置不当作性能基线。
