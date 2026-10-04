---
type: knowledge
status: draft
created: 2026-10-05
updated: 2026-10-05
ai_generated: true
reviewed: false
---

# 两层模型的 PP、SP、CP 与 packed 序列

> 本文为尚未人工核验的 AI 草稿。使用同一组 authored 模型/数据的逻辑调度和 CPU 数值参考；没有启动流水线进程、TE attention 或测量 GPU 时间。

PP改变完整模型的层与训练任务归属；SP在TP组内改变部分算子的sequence布局；CP将attention上下文放到不同rank。三者的切分轴、依赖和通信不同，不能统一画成“长度除并行度”。本页逐项验证这些机制，不将分别正确的参考冒充已验证的联合训练backend。

## 独立数学符号表

| 符号 | 含义 |
|---|---|
| $P,M,s,m$ | PP stage数、microbatch数、stage编号、microbatch编号 |
| $F_{s,m},B_{s,m}$ | 在stage $s$上对microbatch $m$的forward/backward任务 |
| $a_e,b_e$ | 任务开始与结束的逻辑时刻，半开区间$[a_e,b_e)$ |
| $T,C$ | TP/SP组大小和CP组大小 |
| $H,F,d$ | 本例hidden8、FFN通道12、head维4 |
| $S,n_q,n_{kv}$ | 物理token行数、Q头4、KV组2 |
| $c_i,\tilde c_i$ | 第$i$个样本的有效/物理累计长度 |
| $q_t,k_j,v_j$ | 已完成QK norm和RoPE后的Q/K与未旋转V；位置按样本重置 |
| $z_{tj}$ | $q_t^\top k_j/\sqrt d$，只在合法key集合使用 |
| $\mu_r,u_r,w_r$ | rank局部最大分数、移位指数和、移位指数加权V之和 |
| $\mu,u$ | 合并后的全局最大值与分母 |
| $N_\mathrm{idle}$ | 未执行任务的stage×逻辑时间槽数量 |

## PP：层分配先于时间线

使用G03真实两层骨架。PP1持有embedding、L0、L1、final norm/head；PP2的stage0持有embedding+L0，stage1持有L1+final norm/head。microbatch ID贯穿所有stage的同一份输入和loss，不把“token个数”叫作microbatch个数。

每层F和B各耗1个逻辑单位，因此PP1的每次F/B耗2，PP2耗1；embedding/head成本忽略。通信耗时设0但依赖必须满足。没有VPP、overlap、重计算、optimizer overlap，也不声称近似真实硬件利用率。

non-interleaved 1F1B在stage $s$先warmup：

$$
W_s=\min(P-s-1,M).
$$

随后按本地顺序交替$F_{s,W_s+i}$与$B_{s,i}$，最后cooldown完成剩余$W_s$个B。全局开始时间还需遵循：

$$
a(F_{s,m})\ge b(F_{s-1,m}),\quad
a(B_{s,m})\ge b(F_{s,m}),\quad
a(B_{s,m})\ge b(B_{s+1,m}).
$$

边界stage忽略不存在的邻居，并确保同stage任务不重叠。固定Core源码的warmup计数、steady的send_forward_recv_backward、FIFO activation pop及cooldown给出真实实现入口；页面是这些依赖的逻辑事件模拟，不执行那些通信API。

### 手算 PP2 / M2

| 逻辑槽 | stage0 | stage1 |
|---|---|---|
| 0 | F0 | idle |
| 1 | F1 | F0 |
| 2 | idle | B0 |
| 3 | B0 | F1 |
| 4 | idle | B1 |
| 5 | B1 | idle |

共6单位、8个有效stage-slots、4个idle slots，逻辑bubble=4/(2×6)=1/3。stage0同时保留的待反传microbatch峰值为2，stage1为1。本页activation生命周期从F开始到对应B结束，表示待反传输入/状态的逻辑保留期；实际框架可能释放输出buffer、保存其他张量或重计算，不能从这个计数推断显存字节。

PP1同样M2共8单位，因为每个F/B包含两层。不能直接比较这两个逻辑数值就宣称真实加速比。

## SP：sequence gather 与 row reduce-scatter

固定同一完整多轮样本，S=20、assistant、两层之一。输入为真实G03参考图对应层的attention residual。TP2时每rank先持有不同的10行完整H=8，RMSNorm逐token独立：

$$
X_r[10,8]\to\mathrm{RMSNorm}(X_r)[10,8]
\xrightarrow{\mathrm{all\mbox{-}gather\ sequence}}
X[20,8].
$$

随后复用G04成对gate/up列分片，每rank得到$[20,6]$ product；down行分片产生$[20,8]$部分和。其SUM再沿sequence维reduce-scatter回$[10,8]$，重组应等于未切分FFN的$[20,8]$输出。

可手算两rank的同一位置部分和为2与5，归约后应是7；若只按行scatter各自部分和，某rank得到2或5而非7。将这一步除以TP会错误变成3.5。

C-TPMAP中gather的反向：后续是TP计算时reduce-scatter；若后续只是重复计算，则只scatter，该配置分支不能省略。row reduce-scatter的反向是gather。norm参数梯度等生产同步边界还需结合配置；本阶段数值对照验证这段布局与forward恢复，不声称实现分布式optimizer。

## CP：分片Q仍依赖完整历史KV

CP2时每rank拥有部分token的所有Q头和KV组。GQA仍为4个Q头共享2个KV组，head $h$映射$\lfloor h/2\rfloor$。本页使用Layer0同一权重、QK norm、split-half RoPE与position；保留padding标记，attention仅允许同一sample的有效历史key。

普通序列沿全物理长度切成$2C$等块：CP2把chunk0、3给rank0，chunk1、2给rank1，平衡因果attention中前后位置计算量。例长度8：rank0=[0,1,6,7]，rank1=[2,3,4,5]。查询7在rank0，但仍需rank1的历史KV。仅“本地causal attention”会丢失上下文。

本参考不模拟通信带宽或ring传输时间；将每个owner提供的合法KV贡献分块合并，等价性由独立完整softmax参考检验。

### 分块 softmax 如何合并

对一个有效query/head，owner $r$在本地合法key集合计算：

$$
\mu_r=\max_j z_{tj},\quad
u_r=\sum_j e^{z_{tj}-\mu_r},\quad
w_r=\sum_j e^{z_{tj}-\mu_r}v_j.
$$

全局$\mu=\max_r\mu_r$，则：

$$
u=\sum_r e^{\mu_r-\mu}u_r,\qquad
o_t=\frac{\sum_r e^{\mu_r-\mu}w_r}{u}.
$$

空局部集合的贡献为0；所有集合都为空的padding query不计算softmax，不制造NaN或伪造概率。手算两个rank各有一key，logit为0与$\log3$、V为2与4：分母权重1+3，输出(2+12)/4=3.5；平均两个局部归一化输出只得3，错误。丢掉远端key也改变结果。指数移位还须通过10000级logit的稳定性例子。

## THD：逐文档分片，不能忽略 padding 与边界

接G02两条长短样本，逻辑长度11与23，有效累计长度$c=[0,11,34]$。padding=8时物理长度16与24，$\tilde c=[0,16,40]$，总T=40；第二条sample从物理位置16开始，position重新从0计，padding位置不会成为query或key。

固定Core的THD非hybrid路径使用每文档zigzag与TE分片索引，要求每条物理文档长度整除$2C$。本页CPU参考实现同一索引数学，未调用TE kernel。例两个物理文档[0,4,8]、CP2：rank0=[0,3,4,7]，rank1=[1,2,5,6]；这与把全长8一次切分不同。

padding=1时11/23均不能被4整除，CP2配置必须报错，不能偷偷丢token或补齐却不更新metadata。CP1则可以读取未补齐布局。页面保留$c$与$\tilde c$两套metadata，不按本地行数重写为错误文档边界。控制输入与G02共用padding状态和batch构造。

Q的布局为$[T_\mathrm{local},n_q,d]$，KV为$[T_\mathrm{local},n_{kv},d]$；这和SP局部$[S/T,H]$的token-wise residual布局不同。THD展开后的microbatch尺寸1不代表只有一条样本。若把packed总长当成一条causal序列，第二sample会读取第一sample历史，跨样本泄漏反例必须被抓到。

## 验证范围

Python以任务DAG最长路径给出PP参考，前端以逐tick可执行任务模拟，16种PP/M配置逐项对照；另以独立手算时间轴检查依赖、无遗漏重复、activation生命周期与bubble。删除任务或提前backward的反例必须失败。

CP比较独立Python完整softmax与TypeScript分块统计合并，覆盖普通/THD、CP1/2、padding1/8的合法组合；错误远端KV、跨样本泄漏和非法物理长度均拒绝或产生明确数值偏差。SP的最终输出和norm与当前PyTorch完整decoder对应层对照，测试TP1/2及两层，并检查漏reduce反例。float64容差预先固定为atol=rtol=1e-10，不把通过当作真实PP/CP训练或性能证明。
