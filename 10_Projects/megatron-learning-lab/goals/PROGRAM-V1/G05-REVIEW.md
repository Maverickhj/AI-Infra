---
type: experiment
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# G05 自审

## PP 依赖与成本口径

审读 sequence_reference.py、web/sequence/compute.ts、课程与测试。PP固定G03两层真实分配，PP1包括两层及embedding/head；PP2各一层，embedding首stage、final norm/head末stage。F/B每层1逻辑单位，embedding/head/通信成本忽略，保持跨stage依赖；不是wall time或GPU利用率。没有VPP、overlap或重计算假支持。

Python以DAG最长路径递归求时间，TS逐tick调度本地有序队列；PP1/2 × M1–8共16组逐项一致。独立手算PP2/M2为6单位、4 idle slots、峰值activation计数[2,1]；PP1/M2为8单位。每组检查任务唯一/完整、相邻stage F和反向B依赖、同stage无重叠、真实层归属、生命周期。删除任务/提前B反例确实失败。生命周期从F开始到对应B结束，是待反传状态计数，不转换成显存估计。

## SP / CP 与独立数值

SP输入取完整G03算出的对应层attention residual，逐token norm后沿sequence all-gather，成对gate/up列分片，down部分和SUM并reduce-scatter。两层×TP1/2与当前PyTorch完整decoder的norm/down对照；遗漏reduce反例失败。这里只验证此段forward布局，不冒充完整SP训练optimizer。

CP使用相同Layer0的QKV权重、QK norm与split-half RoPE，复用G02token与packing；Python完整softmax与TS分块max/exp-sum/加权V合并独立对照。ordinary全序列zigzag、THD逐document zigzag分别实现，CP1/2及padding1/8合法组合共7组；THD11/23不补齐时CP2拒绝。有效累计[0,11,34]与物理[0,16,40]同时保留；padding query不计算softmax。远端KV丢失与跨sample泄漏均产生显著错误。手算[0,log3]、V=[2,4]的输出3.5与极大logit10000稳定性通过。

落盘后 test:sequence 实跑25配置、5448数值，max_abs_error=4.440892098500626e-16，预设atol=rtol=1e-10未改。5项stdlib Python语义检查通过。临时草稿预检不是阶段验收，后续完整门槛绑定正式源码。

## 来源与前端

本地固定Core commit读取C-SCHEDULE、C-TPMAP、C-COREUTIL，blob与源码行逐字验证；新增8片段，总41完整文件/78片段。non-interleaved warmup/steady/cooldown、SP gather backward的TP/duplicate条件、THD TE索引与普通sequence zigzag、hybrid dispatch边界分开。CPU索引演算不声称调用TE kernel或通信后端。

G05四条Chromium路径通过：PP/M控制真正改变timeline与lifetime；SP显示对应CPU切片；THD非法配置、padding、position重置、remote KV、错误模式和导出；固定源码离线、键盘展开、KaTeX与窄屏。桌面/手机截图已查看，时间表与数值表局部滚动，页面无水平溢出。

集成初次旧GQA源码长流程超过30秒，trace已归档到preflight-gqa-timeout。此前正文memo仅避免重复解析，隐藏整章仍带来大量公式DOM。增加CourseDetails，只在展开时挂载完整正文，保留原生summary的鼠标/Enter/Space行为。此共享修正覆盖GQA/SFT/decoder/TP/sequence五处，没有删减内容/原测试或扩大时限；17条受影响浏览器路径全部通过，原GQA长路径分别11.5秒、14.3秒。

## 边界和下一步

参考时钟、CPU数组与分片布局分别验证，不是联合PP/SP/CP多进程backend。CP合并不测ring通信性能，SP不声称验证分布式反向，真实GPU/训练仍未运行。完整门槛通过后保存截图和源码清单，再标G05 validated并继续G06。本文不代替门槛结果。
