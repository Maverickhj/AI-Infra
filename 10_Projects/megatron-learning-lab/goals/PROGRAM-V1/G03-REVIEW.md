---
type: experiment
status: draft
created: 2026-10-04
updated: 2026-10-04
ai_generated: true
reviewed: false
---

# G03 自审

## 完整模型与独立验算

已审读 decoder_reference.py、web/decoder/compute.ts、课程、CPU检查、跨语言与浏览器测试。模型是明确 authored 的两层 Qwen3-style：V27/H8/F12、Q4/KV2/d4，无bias、split-half RoPE、QK norm、两次 attention/FFN residual、final norm、tied或同初始值的untied head、masked CE。G02原始样本整条输入，不只运行MLP。

Python与TS都调用各自原有GQA函数。Python只把同一运算图的sqrt/exp换为保留梯度的torch算子，并用detach视图做输入校验；没有用detach结果参与loss。共享函数扩展S=1–64，G01原1872值最大误差3.55e-15且原五项数学回归通过，新增变长边界测试将纳入完整门槛。

六组完整forward对照26790个数值，最大误差1.7763568394002505e-15，atol=rtol=1e-10。原测试容差未放宽。CPU六项测试已通过：CE=负目标logprob；untied head解析矩阵梯度；tied梯度=lookup+head；五个选定参数有限差分；冻结参数无grad且不变；未来输入不影响历史输出；零mask和非法ID拒绝；真实文件save/resume包含模型、momentum和RNG。prompt位置0直接mask=0，但activation梯度非零。

CPU exporter记录当前解释器/Torch、source hashes、参数前后、有限差分与resume。它不是HF/Bridge训练证据。第一次SGD更新按θ−0.03g逐项对照；原样本loss下降但不宣称能力改善。resume启用0.1 dropout，正确恢复后参数误差0，故意扰动RNG会失败。前向/有限差分关闭dropout，两种测试边界明确。

## 源码与前端

新增C-BLOCK/C-MLP的raw固定快照；GitHub contents API返回403，但本地Core确有固定commit，git ls-tree提供blob：block=0415035ffbe4c0e8e772e9dd3597195a777db4ce，mlp=f5659d21a6e59eb30ebc0739bafa67c754f87ba5，下载内容与之匹配。已读block373–402、692–700；MLP261–273、308–349；C-LAYER749–762、970–982。Final norm标调用入口，不伪装成kernel；MTP与非MTP stage条件分开。unfused GLU显式声明无clamp、offset0。

三条新增Chromium用户路线通过；子步骤实际数值逐项对照本次CPU输出。桌面与窄屏截图已查看：token/layer控制、输入输出切片、数学、源码按钮与滚动表可用，页面无水平溢出。CPU梯度卡固定标记multiturn/assistant/tied，不随其他选项假装重新训练。JSON只导出选定切片与小型CPU证据，backward/update导出slice=null避免把FFN数组误贴成梯度。

旧Final norm“待补”测试改为检查真实调用原文，再手动切到缺少该语义映射的B-Q3，继续验证缺口降级。全部离线源码断言仍保留。程序完整baseline与数值/source gate还需本次执行通过才更新STATE，本文不是通过票据。

## 文件范围和局限

除开工清单外，新增tests/decoder_cpu_checks.py，由标准库测试wrapper在已有Torch解释器真实执行；tsconfig仅增加allowImportingTsExtensions，以便同一TS模块同时供Vite与Node测试直接导入；package data步骤生成忽略目录content/generated/decoder-cpu.json。无新依赖、无模型下载、CUDA_VISIBLE_DEVICES为空。

教学权重不代表Qwen checkpoint；CPU优化器不是生产Megatron trainer；没有GPU/Bridge/RL实测。浏览器仅Chromium。G04必须在该完整模型上做TP/DP恢复与梯度对照。所有历史G01/G02证据保持原内容，G09再对最终快照跑全集。
