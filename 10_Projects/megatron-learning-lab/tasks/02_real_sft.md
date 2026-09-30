---
type: project
status: draft
created: 2026-09-30
updated: 2026-09-30
ai_generated: true
reviewed: false
---

# P2：真实 Qwen3 / Qwen2.5 SFT

先复用现有环境，按 profiles/minimum-compatibility.md 验收 bridge_sft。Bridge v0.6.2 的 recipe/runner 是参考，不是唯一允许运行的版本；需要时对实际 runner 建立薄适配器，检查接口与行为。不得凭旧版 CLI 假定或自行新建未经验证 trainer。

## 必交文件

`profiles/bridge-sft.resolved.json`：实际依赖与源码、必要接口/语义的检查记录、adapter、checkpoint、tokenizer、template、dtype/backend、并行和最终合并配置；不得包含“必须与参考版本相等”的准入规则。

`experiments/sft/`：源版本兼容的短序列运行脚本、预处理导出、受控数值捕获、checkpoint验证。

`runs/<run_id>/manifest.json` 与 trace：来源是 observed_bridge，带环境与版本，不缓存全部 logits。

## 必测

1. HF权重导入映射和关键参数校验；证明不是load_weights=False随机初始化冒充SFT。
2. 固定短输入，HF参考与Bridge比较selected-token logprob/hidden states。先统一权重、mask、布局、dtype/backend；容差在运行前声明并在报告解释，禁止比较完任意放宽。
3. assistant/last_turn/full、单次label shift、padding；packed两条样本不交叉attention，unpack后的对照正确。
4. loss sum/count的全局归一化；增加DP/CP前审查实际调度/finalize缩放，不凭公式手动再除一次。
5. 两个短训练step的loss/grad有限、预期参数更新；LoRA仅在独立profile中比较trainable/frozen集合。
6. 保存/恢复step、optimizer/scheduler等所需状态；HF导出和resume各自验收。

先短序列（建议128–512）、microbatch1、并行全1。具体global batch、checkpoint、样本重复、eval和save配置由小数据规模调整。不是从官方吞吐配方直接启动大数据训练。

无GPU时实现采集/配置/数据测试并标GPU步骤未执行，不能生成模拟loss通过验收。
