---
type: knowledge
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# 同一两层模型中的 TP 与 DP

> 本文包含 AI 生成的实质内容，尚未完成人工核验。所有数值为 authored float64 CPU reference simulation，没有多进程/NCCL/GPU 测量。

从 G03 的完整输入与同一组权重进入：embedding → 两层 GQA/FFN → final norm → head → masked CE。TP 切参数和激活；DP 对同一参数处理不同样本，再归约梯度。页面选定 token 7 和参数小切片，整模输出/所有可训练参数梯度则由 CPU 测试检查。参考固定 assistant 模式，TP1/2 × 三个样本 × tied/untied 共12种配置。

## 独立符号表（数学符号）

| 符号 | 含义与本例 |
|---|---|
| $S,H,F$ | 当前样本序列长度、隐藏维8、FFN通道12 |
| $n_q,n_{kv},d$ | Q头4、KV组2、每头维4 |
| $T,D$ | TP与DP度，各支持1或2 |
| $r$ | TP局部rank，$0\le r<T$ |
| $W$ | 物理参数存储为 $[\mathrm{out},\mathrm{in}]$，前向右乘 $W^\top$ |
| $X,Y,Z_r$ | 输入、完整输出、rank-local中间量或部分和 |
| $V,V_p$ | 逻辑词表27与补齐存储词表；TP2时$V_p=28$ |
| $z_j,y,m_t$ | 类别logit、目标类别、目标位置loss mask |
| $L_r,N_r$ | DP rank内累积token loss之和与有效目标数 |
| $N$ | 全部DP rank的有效目标数之和 |
| $b,P$ | dtype每元素字节数8；一次通信逻辑payload字节数 |
| $g_r$ | 当前DP rank对全局归一化loss的参数梯度贡献 |

## 列分片与 GQA 的组内排列

若 $W$ 沿输出维切成 $W_r$，每个rank读取同一 $X[S,H]$：

$$
Y_r=XW_r^\top,\qquad Y=\operatorname{concat}_r Y_r.
$$

最后的concat是逻辑恢复式，不表示实际层间一定通信。QKV在本例存储 $[32,8]$，每个KV组顺序为Q、Q、K、V，每头4维。因此TP2的rank0取行[0,16)，rank1取[16,32)，各自完整持有一组KV及对应两头Q。各头的因果attention可本地计算；CPU为复用G01图把逻辑头序列拼回Python列表，并不因此声称生产需要all-gather。

反向有 $dX=\sum_r dY_rW_r$，而 $dW_r=dY_r^\top X$ 保持本地。C-TPLINEAR的allreduce_dgrad分支与gather_output开关分别对应反向输入梯度归约和可选前向拼接；SP/专家通信等配置会改变路径，不能只凭类名判断一次运行。

## 成对 FFN 与行部分和

G03的gate_up参数全局存储为先12行gate、再12行up。TP2每个rank须同时取对应6通道：rank0取gate[0,6)、up[12,18)，rank1取gate[6,12)、up[18,24)。局部计算：

$$
Z_r=\operatorname{SiLU}(XW_{g,r}^\top)\odot XW_{u,r}^\top,\quad
Y=\sum_r Z_rW_{d,r}^\top.
$$

局部product为$[S,6]$，down权重$[8,6]$，局部部分输出均为$[S,8]$。gate/up之间没有gather，down之后SUM恢复隐藏维。注意：按rank直接concat成对权重得到gate0、up0、gate1、up1，与全局gate0、gate1、up0、up1不同。

可手算行分片：$X=[2,3]$，$W=[4,5]$；两个rank的输出分别8与15，SUM得到23。误用concat得到[8,15]，shape与数值都错误；漏reduce只得8。设输出梯度为1，两rank的输入梯度分别4与5，权重梯度分别2与3；无需再把输出梯度平均。

attention输出投影相同：每rank合并本地Q头后有8通道，乘$W_o[:,8r:8(r+1)]^\top$得到$[S,8]$部分和。C-TPLINEAR展示普通reduce、SP reduce-scatter、explicit_expert_comm三个分支；本阶段只执行普通内存内SUM，不把dense world size机械乘EP。

## 词表并行 CE

最终归一化激活复制到词表分片，每rank计算自己的logits。无label smoothing时不必先收集全部logits：

$$
a=\max_r\max_{j\in V_r}z_j,\qquad
u=\sum_r\sum_{j\in V_r}\exp(z_j-a),\qquad
z_y=\sum_r \mathbf 1[y\in V_r]z_y^{(r)},\qquad
\mathrm{CE}=a+\log u-z_y.
$$

这三步分别需要逻辑MAX、SUM、SUM。手算logits=[0,0]、两rank各一个类别、目标1：$a=0,u=2,z_y=0$，CE为$\log2$；梯度为[0.5,−0.5]。只用目标owner的分母则错误得到0。

本例27个类别为了等宽存储在TP2扩展到28，每rank14个槽位。第27槽位设$-\infty$并排除CE，head恢复逻辑27类。该策略是保持G03模型不变的明确参考约定，不是固定Core或任意配置都会排除padded vocabulary的声明。C-VOCABCE原文证明三次归约，没有单凭该片段证明外部传入logit如何屏蔽。把补齐槽位当真实类别会改变分母，测试必须失败。tied时embedding共享参数的梯度汇集查表与head路径。

## DP 累积与全局有效 token

梯度累积跨microbatch加loss sum和token count，再归约。目标为：

$$
L=\frac{\sum_r L_r}{N},\quad N=\sum_r N_r,\quad
g=\sum_r\nabla_\theta(L_r/N).
$$

本参考每个DP rank处理2个microbatch，有效目标数分别4和18。先本地归一化再简单平均会给两rank相同权重，偏离全局token均值。手算$N_0=1,L_0=1,N_1=3,L_1=9$：正确loss=10/4=2.5，均值之均值=(1+3)/2=2。梯度同样被错误加权。

C-FINALGRAD在num_tokens非空时从最后PP stage广播token数、在DP/CP组SUM、乘逆计数。生产梯度归约是否预缩放、AVG或SUM需结合DDP配置核验；本参考明确使用SUM，不能将公式再次除DP。全零监督不发起更新；G03已有拒绝路径。

## Group 与字节口径

本页PP=CP=1，用global_rank=dp_rank×T+tp_rank。T=D=2时，rank3的TP组[2,3]、DP组[1,3]。group图为逻辑成员关系，TP与DP数值用例分别验证，未启动4进程联合后端。

一次$[S,H]$ ring all-reduce的逻辑payload为$P=SHb$，每rank发送字节为：

$$
B_{\mathrm{send,rank}}=2\frac{T-1}{T}P.
$$

S=11、H=8、float64、T=2时P=704 bytes，每rank发送704 bytes。此值不包括接收、其他算子/梯度、协议开销或拓扑；不换算成毫秒、带宽或真实GPU利用率。

## 反例与验证边界

test:tp-dp实际跑完整模型forward/backward，固定atol=rtol=1e-10比较12种TP配置；检查权重正确重组、DP全局归约以及漏SUM、错误concat、错gate/up、额外词表槽位和均值之均值。前端读取当次data:parallel产生、带源码hash的小型CPU证据。rank、group、参数范围、层、样本与tied选项须联动，非法TP3须报错。这里的数值一致性不证明任何真实分布式性能或训练checkpoint正确性。
