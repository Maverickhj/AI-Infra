---
type: experiment
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# G04 自审

## 数学与完整模型

已审读 experiments/tp_dp_reference.py、G03的投影回调、六项CPU检查、rank布局、课程及浏览器测试。TP1/2均使用G03相同两层参数、三条原样本和tied/untied head；12组恢复logits和所有可训练参数梯度，atol=rtol=1e-10。最近实跑最大梯度误差1.9984014443252818e-15；DP全局sum/count梯度误差2.7755575615628914e-17。没有启用CUDA或分布式进程组。

QKV按完整Q,Q,K,V组切行，attention输出和FFN down按输入列切分并SUM。gate/up按同一FFN通道配对；错误rank顺序concat和漏SUM分别由重组/完整loss反例检查。选定输入、全局/局部输出、weight与gradient前2行4列来自实际CPU运算；未导出完整激活。tied词表参数gradient明确包括lookup与head路径。

词表CE执行MAX和两次SUM，未使用完整logits作为loss计算捷径。逻辑V27在TP2存储V28、额外槽位排除CE是明示教学约定，不推广到任意Core默认。自审修正过全局切片：现在global输出仍27类，只有局部rank显示补齐槽位；增加shape断言。手算行分片23与词表log(2)、梯度[.5,-.5]由独立小例核对；DP局部计数4/18、两次累积，对照单模型同四个microbatch，全局N22；均值之均值反例loss和gradient均改变。

## 固定来源

读取并核对Core固定commit的C-TPLINEAR、C-VOCABCE、C-FINALGRAD；完整blob校验通过。新增5个片段，全部归档共38文件/70片段。gather_output、allreduce_dgrad、SP和expert分支分别说明；num_tokens生产归约与DDP预缩放的关系保持边界，不从一处调用推断全配置。前后向说明为数学解释，源码调用不标kernel。

## 交互与失败修正

四条新增Chromium路径全部实跑通过：rank/group联动、各算子的输入/输出/梯度切片与CPU一致；层/hash链接刷新；DP错误均值和导出；固定源码离线语义路由；缺runtime时无代码冒充；TP3拒绝、TP1重置rank、390px窄屏、键盘展开课程和KaTeX。

已查看桌面和窄屏截图。参数表较宽可局部滚动，页面无水平溢出，group、范围、误差、字节口径及课程入口均可读。数据证据固定assistant/token7；上方mode改变时明确不改写CPU证据。TP和DP分别验证，逻辑group图不冒充4进程联合后端。

初次检查在源码修正期间因source hash不匹配拒绝旧证据；保留记录并重生成，未取消hash断言。随后精确label定位暴露select标签含选项文字，控件补充明确aria-label后全部通过，未增加timeout或删减断言。记录位于runs/program-v1/G04/preflight-*。

## 验收边界

当前完整门槛仍须执行通过后才更新STATE。理论704 bytes为S11/H8/float64/TP2时单次ring SUM每rank发送量，未含接收、全模型或其他collective，没有通信时间。新增package data:parallel复跑CPU生成，与现有Torch环境复用，无新依赖、权重下载或GPU训练。G04通过后继续G05；未宣称整个PROGRAM完成。

首次完整门槛报告 `runs/program-v1/G04/20261004T163054246764Z/report.json` 保留。82项Python中课程结构检查因标题缺少“数学符号”字面标识失败，原独立表内容完整；已统一标题并保留所有断言。TP完整对照另补token CE和scalar loss显式assert_close，避免仅凭logits/梯度间接推断loss对齐。重新跑完整门槛后再更新状态。

第二次完整门槛报告 `runs/program-v1/G04/20261004T163352229821Z/report.json` 保留：82 Python及构建通过，浏览器61通过、两条旧GQA长交互超过30秒。新增面板导致token/子步骤切换重复解析整章Markdown/KaTeX；现复用memo化CourseText，正文未变时不重新解析，GQA/SFT/decoder/TP课程内容和所有原断言保持。该性能修正扩展到四个课程渲染入口，专项旧GQA和完整回归均需重验；没有增加timeout或拆减旧GQA断言。

缓存正文后，原GQA三条与G04四条共7条专项浏览器用例全部通过；原先超时的两条分别19.2秒、25.4秒，仍沿用30秒时限。接着再次对当前快照执行完整门槛。
