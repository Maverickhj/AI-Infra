---
type: experiment
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# G06 自审

## 完整模型与复用

实际审读 experiments/decoder_reference.py、web/decoder/compute.ts 和新增 family 实现。提取 attention/FFN 扩展点后立即实跑 G03、G04、G05：decoder 6组26790值、7项CPU；TP/DP 6项CPU与布局；sequence 25组5448值均通过，类型检查通过。没有新建孤立 MLP 流程。三种 family 共享 embedding、两层 residual、final norm、untied head、CE；Qwen复用原GQA，V2直接Q，V3低秩Q。tiny V3首层dense是明确缩小配置，未篡改全尺寸前三层dense档案。

## 数学与独立参考

当前 test:moe-mla 实际通过9组完整模型配置与4种EP/ETP布局，72033数值，max_abs_error=2.886579864025407e-15；atol=rtol=1e-10事先固定。Python以Torch矩阵/einsum/autograd，TS以独立数组循环和公共decoder主体实现。CPU八项检查覆盖手算、直接按token的expert oracle、重排恢复、padding、aux解析梯度、所有参数分片梯度、有限差分和真实SGD改变参数。冻结输入norm不变；bias不在SGD参数表中。

Qwen选择post-top-k softmax；V2固定pre-softmax不重归一化；V3 sigmoid+bias选组/选专家，但组合取原分数归一化再scale。shared不乘router权重。aux重选不带bias/group的top-k，不复用dispatch counts。B=1且一次归约，未声称覆盖变长多样本seq_aux或生产梯度scaler。V2辅助系数显式authored；原模型checkpoint没有加载。padding为MoE有效token过滤，不把loss mask当padding，不声称Core会自动清主路由概率。

最初bias手算因参考中计数误转float32产生2.235e-10偏差；已保持logits dtype，容差不变。生产路由fp32与参考float64的边界仍在fixture/课程中。bias preview按全局batch计数的符号更新，只展示下一次值；不在eval时写生产状态，不将每次点击当训练。

MLA先latent norm再up，positional key独立RoPE，标准RoPE/mscale=1。内容K展开与Q吸收相同，weighted latent再乘V up；逐prefix读取normalized latent+rotated Kpos cache也一致。wrong_norm、wrong_scale、missing_rope均显著破坏等价。训练激活与cache字段/UI分离，数组payload字节不称显存实测。

## 来源

新增B-DS2固定Bridge blob、C-ROUTER/C-MOEUTIL/C-DISPATCH/C-EXPERT固定Core原文；扩充C-MLA与C-FINALGRAD必要范围。46完整文件/98片段已逐字核验。辅助公式首次选段边界落到fused调用参数，人工审读后改到实际139–143行运算；完整性工具只验原文，不能代替语义审查。

读过Core的padding取反、aux单独scores、group top-(k/group_topk)和、bias无梯度更新及global-batch finalize调用；未把docstring中不一致的padding True含义当事实。MLA cache仅decode-only吸收，正常训练/混合prefill分支分开。片段明确高层实现，不冒充kernel或runtime lineage。

## 前端与回归

四条新Chromium路径与三条旧GQA路径共7项已通过：CPU数组、EP/ETP表、选token0与reload、padding、反例拒绝导出、bias与aux、V2/V3切换、训练/cache互斥导出、固定原文离线、KaTeX和键盘。首轮group整行预期多了一个字符，已改成逐单元格精确检查全部值，失败trace保存G06/preflight-group-expectation，没有删断言或增时限。

桌面和窄屏截图已实际查看，表格局部滚动、正文保留；长截图看到未聚焦skip link越界显现，补未聚焦opacity=0并保留focus显示，测试增加实际键盘跳至main。MLA训练表补选定query概率prefix。上述最终小改动将由完整阶段门槛覆盖，旧7项结果不冒充最终快照。

新源码按钮传明确片段ID，关闭后其他课程默认路由不被覆盖，手动切来源会清该ID。普通deep link状态含family/token/head/layer/padding/EP/ETP/fault；非法选项normalize。未改变原GQA默认片段或断言。

## 边界

没有真实checkpoint、GPU、NCCL、多进程MoE、任意world topology、EDP2、capacity dropping、MTP、FP8、YaRN或Flash MLA性能验证。专家group限EP1/2、ETP1/2、EDP1。数学/软件自审不代替完整门槛；只有门槛通过并保存当前源码清单、截图后将G06标validated，再继续G07。
