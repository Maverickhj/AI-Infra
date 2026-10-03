---
type: project
status: draft
created: 2026-10-02
updated: 2026-10-03
ai_generated: true
reviewed: false
---

# G01 进度

状态：ready_for_review。这是项目记录，不是 Codex 内部 Goal 状态。

## 当前起点

最新已审查提交为 ed606ee。三个历史问题在静态审查层面已修复，有针对性回归测试；本轮审查环境未重跑完整应用。先执行 baseline，不沿用旧版执行报告。

## 本轮记录模板

### checkpoint / 日期

- 目标与本次修改：
- 实际执行命令、退出状态与日志路径：
- 对照验收项：
- 新增证据、与前次相比的变化：
- 阻塞或下一步：

## 收尾

填写 ready_for_review 或 blocked 的理由及门槛报告路径。禁止只写“测试全绿”；不得把 G01 完成理解为已完成真实 SFT/RL。
### A / 2026-10-03：执行范围与基线

- 起点：9333d31，工作区干净。G01 取代旧 P0 范围；保留现有课程与全部旧测试。
- 修改范围：新增 experiments/gqa_reference.py、content/fixtures/gqa-reference.json、web/gqa/{compute.ts,GqaWalkthrough.tsx,steps.ts}、tests/{test_gqa_reference.py,gqa-numeric.mjs,browser/gqa.spec.ts}、content/cases/04_gqa_source_walkthrough.md；修改 web/{main.tsx,data.ts,SourceExcerpts.tsx,style.css}、content/source-snippets.json、research/source-evidence.json、package.json（仅 script）、必要的摘录覆盖测试与 implementation-report.json；更新本目标 PROGRESS/evidence/REVIEW。
- 顺序：A baseline → B 独立 CPU 数值和反例 → C 共用计算的页面与固定源码 → D 完整 final / verify。无结构搬迁、不改验收、不动训练栈、不 commit/push。
- baseline 已启动，记录实际命令：容器内 PATH=/data/cache/megatron-lab-node/bin:$PATH、PLAYWRIGHT_BROWSERS_PATH=/data/cache/megatron-lab-browsers，env -u PYTHONPATH uv run --no-project python -S tools/goal_gate.py baseline。尚待进程结束后记录结果。

### B / 2026-10-03：独立 CPU 参考与 TypeScript 对照

- A 已完成：runs/goal-g01/20261003T060256497404Z/report.json 为 baseline_passed；全部七条命令退出0，包含旧13项 Chromium 与四条新增回归。绑定 HEAD 9333d31 和报告内源码指纹。
- 新增完整 fixture、Python 标准库标量参考、五项 Python 数学/反例测试、生产 TS 模块和 npm run test:gqa（没有安装任何依赖）。RoPE 在固定 Core 的 rope_utils.py 核对为 split-half；Q/K norm 在旋转之前。
- 实跑：python -S -m unittest discover -s tests -p test_gqa_reference.py -v，5项通过；npm run test:gqa 退出0，对照1872个有限中间值和48个负无穷 mask 位置，max_abs_error=3.552713678800501e-15。容差未改变。
- 数值日志：runs/goal-g01/checkpoint-b/numeric.log。两种 variant 的漏缩放和错 KV group 均改变最终 residual，因果性/归一化/非法输入/全mask错误/16→8投影验证通过。
- 当前状态：ready_for_review。下一项 C：让页面直接调用已测模块，补齐子步骤源码与课程；尚未证明 G01 完成。真实 HF/Bridge/RL 为 not_run。


### C / 2026-10-03：课程、交互、源码和回归诊断

- 新增13个GQA子步骤、真实配置/教学reference区分、token/head/layer/substep深链接、split-half演算、probability矩阵、两种错误路径、完整fixture导出和独立课程符号表。页面直接调用已做数值对照的web/gqa/compute.ts。
- 新增C-LAYER/C-ROPE固定源码及C-ATTN的projection/RoPE/core边界。总29文件54摘录，全量Git blob与连续行校验已实跑通过。Final norm缺口保持；数学拆解不是TE kernel trace。
- 缓存下载先遇urllib TLS中断，curl容器同域仍失败；本地blob补齐B-DATASETUTIL，宿主网络只下载公开B-RUNNER，随后在容器执行完整校验。没有复制私钥或模型权重。
- 首次GQA浏览器加载失败由Node JSON import attribute引起，改为测试端fs读取；第二次源码断言暴露B-Q2没有显式qk_layernorm=false，修正证据边界并补C-ATTN条件入口，没有编造配置行。
- 全量浏览器回归出现新旧table-wrap选择器冲突，以及54片段循环达到30秒总时限。trace显示后段仍逐次成功而非卡住；新表格独立class，旧断言仍完整，源码查找限定dialog，固定课程Markdown/KaTeX和GQA组件缓存，避免无关重渲染。单独54片段已通过，正在复跑全部路线。
- 实跑Python完整51项通过；C-LAYER/C-ROPE静态阅读及新增连续行仅代表已读范围，AI解释待人工review。下一项D：保存当前测试证据，逐条自审，final和verify。


### D / 2026-10-03：完成审计，等待用户审阅

- final命令：env -u PYTHONPATH uv run --no-project python -S tools/goal_gate.py final，退出0。报告：runs/goal-g01/20261003T064146364998Z/report.json，status=ready_for_review。
- 九条必需命令全部passed/退出0：data、python-tests（51项）、handoff、snippets、build、browser（16项，skipped/flaky/unexpected均0）、diff、gqa-numeric、upstream-snippets（29文件54片段）。
- verify命令：env -u PYTHONPATH uv run --no-project python -S tools/goal_gate.py verify runs/goal-g01/20261003T064146364998Z/report.json，退出0，返回ready_for_review。
- 最终源码指纹：50ef897f45acc7a516e8a9a50adc90fded2d14e6fc54d7eec01dd974a137d360。HEAD保持9333d31，改动未commit/push。
- evidence.json七项均verified且真实文件SHA256已通过verify；其顶层verified_pending_final_gate记录冻结证据时点，最终状态由上述final/verify报告确定。PROGRESS不属于源码指纹或证据引用文件，收尾记录不使报告失效。
- 自审见REVIEW.md，仍ai_generated=true/reviewed=false。完整GQA参考教学已实现，等待用户审阅；Final norm专属片段为明确保留缺口。没有真实HF/Bridge/RL运行，没有GPU任务、权重下载、公开部署或后续SFT/RL扩展。
- 停止在G01。本轮没有剩余执行阻塞；后续修改源码或证据必须重新final/verify。
