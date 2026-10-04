---
type: experiment
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# G02 自审

## 行为与数学

审读 `web/sft/compute.ts`、`experiments/sft_data_reference.py`、fixture、课程和两类测试。两实现独立计算 data/attention/CE，TS 使用稳定 softmax，Python 用小范围直接 exp 与 fsum；容差预先固定 1e-11 + 1e-11*abs(reference)。首次 96 配置、3014 数值对照最大误差 1.4210854715202004e-14，通过；增加等宽 unpacked 对照后将由阶段 gate 重跑。

手算：多轮样本位置7的 assistant 标记预测“5”，位置8预测“。”，位置9预测 EOS，mask 均为1。assistant/last_turn/full 计数6/3/19。样本长11与23，按8对齐后有效累计[0,11,34]、物理累计[0,16,40]，unpacked 为[2,24]，同一有效目标与全局 sum/count 不变。末尾 ignore_index=-100；padding 无监督、无可见 key；零监督均值为 null。

实际故障用例覆盖重复 shift、input-mask 错位、前条 EOS 预测后条 BOS、全局三角 attention 泄漏、均值之均值。截断不补 EOS；last_turn 不回退到旧回复；空回复依声明的人工协议监督 EOS。单样本下两种平均确实相同，页面要求选两个样本才能演示错误。CE 列明确是预先计算的教学值，不冒充 backend 对 ignore_index 的输出。

## 固定源码

从固定 Bridge Git tree 获取实际路径并核对完整 Git blob（本地 checkout 的旧路径在该 commit 不存在，首次404已记录在过程输出；未用另一版本替代）。已审读 B-DIRECTSFT 100–113、B-SFTCOLLATE 107–163、B-CONVERSATION 861–886及1626–1651、B-PACK 159–217。新增原文分别覆盖 dataset委托、逐样本shift后pack、labels/mask同步移位、last_turn回退边界、有效/物理累计长度。B-STEP/B-LOSS沿用已固定片段。完整缓存校验已通过33文件/59片段；阶段 gate 再复验。

源码边界：只核验 direct-HF 路径，未声称全部 GPT-SFT/多模态/MTP 路径相同。authored fixture 有已知 message 边界，不模拟官方 template 的 fallback。没有 tokenizer 官方结果、HF/Bridge训练或CUDA实测。

## 界面与隔离

复用现有 SampleInspector 的 messages/模式选择；新增数据演算组件与深链接状态。已检查 Chromium 桌面截图：表格、边界、课程和源码按钮在原整模入口内，公式渲染完整。窄屏测试检查页面无水平溢出，表格可键盘聚焦滚动；不截掉列。导入器限制200KB，核对revision/template hash与token/mask长度，React按文本渲染；所有导入标 external_unverified，不执行上传代码。输入改变清除旧导入结果。

首次三条 G02 浏览器路径两条通过，一条因整段课程包含解释性“NaN”而失败；改为对实际数值摘要及表格同时断言无NaN/Infinity，没有删掉边界断言。旧 G01、公式、head、源码路由与离线全部片段测试均保留，由本轮完整 gate 验证；不复用旧绿色报告。

## 限制与接续

该因果上下文函数用于验证数据机制，不是完整 decoder。G03 必须复用 GQA 实现完成两层模型、CPU backward/update/resume。R01再验证真实官方 tokenizer/template。只覆盖 Chromium，Firefox/WebKit未运行。源码大bundle警告仍存在，后续G09评估按需加载；不改构建阈值掩盖。无commit/push或部署。

完整 gate 首次尝试：runs/program-v1/G02/20261004T152828520971Z/report.json。Python/build/source通过，18条浏览器通过，聚合离线59片段用例超出30秒；保留失败日志。将该用例按33个源码文件参数化，每片段原文/行号/注释、每文件离线与runtime缺失断言全部保留，单用例超时未增加。后续软件阶段继续增加源码时同样全部覆盖。
