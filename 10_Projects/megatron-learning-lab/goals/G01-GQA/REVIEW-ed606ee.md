---
type: project
status: draft
created: 2026-10-02
updated: 2026-10-02
ai_generated: true
reviewed: false
---

# ed606ee 修复复核

审查提交：`ed606ee657097431e674785653da77dc6de19306`，父提交 `9d0c8faf5b379bcba8485efb7067ac3d26f2fc71`。修改范围8个文件。最新提交发现、内容读取与差异检查通过 GitHub 连接完成；本次没有远程写入。

## 结论

上一轮 F1/F2/F3 在代码审查层面均已修复，相关测试定义覆盖了修复行为。可以关闭这三个实现缺陷，进入下一阶段。不能将此结论扩展为“完整应用测试已独立复跑”或“核心精讲已全部完成”。

### F1：块级公式

[web/main.tsx](https://github.com/Maverickhj/AI-Infra/blob/ed606ee657097431e674785653da77dc6de19306/10_Projects/megatron-learning-lab/web/main.tsx) 的 Prose 已删除全局正则替换，只把 clean(text) 交给 Markdown renderer；动态公式和三篇课程正文改为规范多行 $$ 定界。避免在任意代码块中重新解释美元符号。

新增浏览器用例 Display formulas retain block semantics while symbols stay inline，断言当前步骤 .katex-display 数量，同时检查 inline math 和 .katex-error。静态检查结论：已修复。本轮没有执行真实 Markdown/KaTeX DOM 渲染。

### F2：当前语义步骤到源码

[web/data.ts](https://github.com/Maverickhj/AI-Infra/blob/ed606ee657097431e674785653da77dc6de19306/10_Projects/megatron-learning-lab/web/data.ts) 新增 sourceExcerptId，返回稳定的 excerpt ID，不再使用数组下标。Embedding→gpt-embedding、LM head→gpt-output-projection、loss→c-gpt-l790；MLA train/decode 分别选择 c-mla-l638 / c-mla-l321。C-MOE 仅在当前层为 MoE 时匹配。

[web/SourceExcerpts.tsx](https://github.com/Maverickhj/AI-Infra/blob/ed606ee657097431e674785653da77dc6de19306/10_Projects/megatron-learning-lab/web/SourceExcerpts.tsx) 无匹配时显示实现片段待补，不再用首项假冒当前实现。手动浏览非默认片段也有提示。父组件 key 包含 source/model/scenario/step/layer/mla，模式或层变化会重置默认选择。

新增两条浏览器路由用例：通过真实界面进入 Embedding/LM head/loss；Final norm 无匹配时拒绝错误默认；MLA train/decode 路由切换。静态审查与提取的映射逻辑复验通过。

边界：Final norm 仍没有专用实现片段，现有明确提示是符合上轮兜底要求的修复，不代表源码覆盖率已完整。Embedding 片段是调用入口而不是查表 kernel，注释如实说明。

### F3：head 的离散输入

[web/main.tsx Decoder](https://github.com/Maverickhj/AI-Infra/blob/ed606ee657097431e674785653da77dc6de19306/10_Projects/megatron-learning-lab/web/main.tsx) 使用 valueAsNumber、Number.isInteger 及范围检查；非法输入不更新 head 并设置 aria-invalid/错误提示；切换模型时把已有合法值裁剪至新范围。

浏览器测试新增 0、15、3、3.5、负数、越界、空值、溢出及模型切换。独立提取表达式检查了整数/小数/NaN/Infinity/范围及裁剪逻辑，结果通过；没有冒充实际 input/React 交互。

## 尚待补齐的执行证据

[implementation-report.json](https://github.com/Maverickhj/AI-Infra/blob/ed606ee657097431e674785653da77dc6de19306/10_Projects/megatron-learning-lab/implementation-report.json) 仍记录2026-09-30的9项 Chromium测试和45个片段，不是该修复提交的新报告。当前测试源码共13个 test 定义，其中新增4个定向回归；新增2个摘录使片段数量由45变为47。这些是代码/差异计数，不是运行成功计数。

针对当前 SHA 查询 GitHub Actions 返回 total_count=0。它只能说明此次查询没有可见 Actions 运行，不能证明开发者本地没有跑过测试。

下一轮 baseline 应重新运行当前测试并输出新报告（绑定 HEAD、源码指纹、命令、退出码、日志），同时更新项目汇总记录，或明确旧 implementation-report 是历史档案。不要把旧9项通过记录当成本次13项的证据。

## 本轮独立验证范围

完成23项提取逻辑检查，使用合成最小 model metadata 运行 sourceExcerptId/head 校验分支；详见交付包 review/logic-probes.json。这不是导入完整 React 工程，不是上游 Git blob 再验证。

审查容器到 codeload.github.com 的 DNS 解析失败（curl exit 6），也未具备该前端锁定依赖，因此未独立执行 npm build、完整32项项目Python测试、13项Chromium测试或截图检查。GitHub连接的静态读取成功，不与容器联网能力混为一谈。

没有将未安排的 GPU/SFT/RL 运行列为修复阻塞项。

## 后续 Goal

现有 Decoder 依然主要展示 GQA 高层路线、head/group 与shape，详细 QK→缩放→mask→softmax→V 的数值/源码逐步解释尚未实现。这是先前已指出的内容深化目标，不是本次修复回归。

首个目标 G01 因此限定为一个完整 GQA 精讲单元，加明确标为 reference 的确定性小张量演算。源代码图、教学数值和实际训练结果必须分开。不要直接把整个 PLAN.md 或真实 GPU 环境建设交给一个无限目标。