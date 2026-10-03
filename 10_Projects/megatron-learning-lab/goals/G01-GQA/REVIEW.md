---
type: project
status: draft
created: 2026-10-03
updated: 2026-10-03
ai_generated: true
reviewed: false
---

# G01 逐项自审

这是AI自审，待用户审阅；不代替人类教学验收。源码基线为9333d31，尚未commit/push。本次目标只覆盖read_only与CPU教学参考，不进入SFT/RL。

## 验收对应与证据

| 验收项 | 自审结论与可检查证据 |
|---|---|
| G01-01 | 当前提交baseline的七条命令全部退出0；报告runs/goal-g01/20261003T060256497404Z/report.json包含源码指纹与13项浏览器标题。原公式/Embedding/MLA/head修复测试保留。当前16项浏览器全部通过，包含全部旧场景。旧implementation-report明确为2026-09-30历史记录。 |
| G01-02 | GqaWalkthrough/steps提供输入→RMSNorm→QKV/group→QK norm→RoPE→score→scale→mask→softmax→V→merge→projection→residual共13步，每步shape、公式、配置条件、验证办法。04课程实际挂接页面，符号表独立；整模上下文和层选择保留。Qwen2.5模型切换改变实际bias/norm计算。 |
| G01-03 | Python标准库参考独立实现，不读取TS或前端结果。页面直接导入compute.ts，数值测试也导入该文件；对照1872个有限值及48个mask负无穷位置，最大绝对误差3.552713678800501e-15，阈值未放宽。双实现验证因果边界、softmax行和、masked零、所有head的KV group、两类错误路径、16→8投影/residual、非法输入与全mask错误。 |
| G01-04 | sourceExcerptId扩展operator维度；源码组件key含model/scenario/layer/mode/head/token/operator。新增C-LAYER、C-ROPE，以及C-ATTN调用片段。29个文件54片段逐字与Git blob校验通过；浏览器逐步断言ID、范围、关键代码和固定SHA URL。真实实现、调用入口、数学等价参考、缺口四种情况分开。 |
| G01-05 | 浏览器实际操作13步、head/token与Qwen2.5切换，比较所有选中数组；重载深链接、键盘下一步、390px宽度、导出JSON、源码弹窗上下文变更和旧MLA页面均通过。桌面截图人工查看；移动宽度/表格/公式通过DOM行为断言，长截图为辅助证据，不作为数值证据。 |
| G01-06 | 真实配置为derived，教学fixture与导出为reference；完整权重、epsilon、theta、positions、split-half布局、mask约定与bias均显式记录。训练执行一直not_run。只在容器运行本地小张量测试，没有下载模型权重或运行GPU任务。 |
| G01-07 | git diff范围仅本项目；未改AGENTS、仓库设置、验收或Goal门槛，不删测试、不跳过断言、不改数值容差。旧离线测试只将查找范围限定dialog，全部54片段断言保留；类型/Markdown计算缓存减少无关重渲染。停止在G01，最终状态必须由final和verify确认。 |

## 数学核对

- B=1,S=4,H=8,n_q=4,n_kv=2,d=4，权重布局为逻辑[in,out]。mixed按group存Q,Q,K,V；不能按全Q/全K/全V拆。
- 输入RMSNorm按H轴，QK norm按d轴且在RoPE之前。V不做norm或RoPE。fixture声明Qwen3-style与Qwen2.5-style，而非实际Qwen checkpoint。
- split-half在d=4时配对(0,2)、(1,3)，位置0恒等且旋转保持每对平方和；独立解析测试验证。
- score按正确KV group点积，缩放除以sqrt(d)=2；causal对角线允许，未来负无穷，softmax沿key轴。全mask行明确报错而非产生成功NaN。
- 加权V后按Q head顺序合并为16维，再投影8维，加回原始X。漏缩放与错KV group两类反例在两种variant都改变最终residual；因果测试比较最终residual，不仅检查mask。
- 6位显示舍入不用于数值断言；导出保留有限float64精度，负无穷显式编码。误差阈值只属于本小fixture。

## 源码与界面核对

参考Core SHA a0f793dfa4e776d99a8aa63bed74e0fafc91b0db。新增摘录从本地该commit取出，校验完整blob后截取连续行；中文注释未写进原文。

RMSNorm/残差定位到TransformerLayer调用；QKV split与QK norm到SelfAttention；RoPE布局到_rotate_half，同时提供实际调用入口；score/scale/mask/softmax/V定位core_attention后端边界。数学拆解不意味着TE/FlashAttention物化完整score矩阵。

Qwen2.5的B-Q2片段明确证明add_qkv_bias=True，但没有显式qk_layernorm=False。页面说明norm关闭来自模型档案，并另提供C-ATTN模块存在性条件片段；没有把缺少的配置语句编进摘录。Final norm仍明确为专属片段待补，这是G01允许的已知缺口。

源码下载遇容器TLS失败后，使用已有Git blob和宿主网络下载的公开单文件；最终仍在容器全量校验。没有把本地当前HEAD不同文件冒充固定commit。

## 执行与复现

在现有Megatron-Bridge开发容器，项目目录 `/opt/AI-Infra/10_Projects/megatron-learning-lab`：

```bash
export PATH=/data/cache/megatron-lab-node/bin:$PATH
export PLAYWRIGHT_BROWSERS_PATH=/data/cache/megatron-lab-browsers
npm run dev
# 浏览器访问容器的5173端口；若没有端口映射，可在开发工具转发该端口。
npm run test:gqa
env -u PYTHONPATH uv run --no-project python -S tools/goal_gate.py final
# 用final打印的项目内report路径执行verify。
```

此轮复用了Node、Chromium和现有Python环境，未安装新训练依赖。baseline与checkpoint日志在runs/goal-g01，文件SHA256集中列于evidence.json。最终门槛会再执行当前源码的所有必需命令并绑定指纹；只有其ready_for_review且verify通过，才可报告目标完成。

## 保留限制

这是AI自审而非用户已审阅。教学数值不能证明BF16/GPU parity、kernel性能、真实权重或训练正确；packed/并行/dropout/额外attention bias路径不在fixture内。真实HF/Bridge/RL继续not_run。Final norm专属实现和后续SFT/RL新课程不在本次目标。
