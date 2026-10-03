export const steps = [
  {
    id: "input",
    title: "Attention 输入",
    shape: "[1,4,8] → [1,4,8]",
    formula: "X_t\\in\\mathbb R^H",
    explanation:
      "当前层 residual stream。保留原始 X，最后加回 attention 输出。这里只计算一个 batch，数组省略 B=1 轴。",
    condition:
      "TP=PP=CP=1，无 packing，无 dropout；不同层复用同一教学 fixture，绝非该层激活。",
    verify: "修改未来 token，当前及更早 residual 应保持不变。",
    sourceId: "C-LAYER",
    excerptId: "gqa-input-norm",
    kind: "调用入口",
  },
  {
    id: "rms",
    title: "RMSNorm",
    shape: "[1,4,8] → [1,4,8]",
    formula:
      "U_{t,i}=\\gamma_i X_{t,i}/\\sqrt{H^{-1}\\sum_jX_{t,j}^2+\\epsilon}",
    explanation:
      "沿 hidden 维计算均方根，不减均值；gain 逐维相乘。输入 gain 与 Q/K head gain 不是同一权重。",
    condition:
      "epsilon=10⁻⁶；TE 可在 linear_qkv 内融合 norm，此处定位语义调用边界。",
    verify: "检查每个分量 U_i/gamma_i 与 X_i 的共同缩放因子；零输入保持有限。",
    sourceId: "C-LAYER",
    excerptId: "gqa-input-norm",
    kind: "调用入口",
  },
  {
    id: "qkv",
    title: "QKV / grouped split",
    shape: "[1,4,8] → [1,4,32] → Q[1,4,4,4], K/V[1,4,2,4]",
    formula: "M=UW_{QKV}+b,\\quad g(h)=\\lfloor h/(n_q/n_{kv})\\rfloor",
    explanation:
      "每组按 Q,Q,K,V 排列，先按 group 再拆 head。Q0/Q1 共用 KV0，Q2/Q3 共用 KV1。绝不能将前16列全部当作Q。",
    condition:
      "Qwen3-style 无 QKV bias；Qwen2.5-style 显式加 fixture bias。H=8，Q 总宽度16，不以 H/n_q 重算 d。",
    verify:
      "检查每组16列：[0:8]为两个Q，[8:12]为K，[12:16]为V；切换head检查group。",
    sourceId: "C-ATTN",
    excerptId: "c-attn-l1887",
    kind: "真实实现",
  },
  {
    id: "qknorm",
    title: "Q/K head norm",
    shape: "Q[1,4,4,4], K[1,4,2,4] → 同形",
    formula:
      "\\widehat Q=\\operatorname{RMSNorm}_d(Q),\\quad \\widehat K=\\operatorname{RMSNorm}_d(K)",
    explanation:
      "对每个 head 的最后一维单独归一化。V 不做此归一化。先 norm 再 RoPE。",
    condition:
      "Qwen3-style 开启，Qwen2.5-style 关闭；旁路时 qnorm=q、knorm=k。",
    verify:
      "切换模型，Qwen2.5-style 的前后数组必须逐项相同，Qwen3-style 通常不同。",
    sourceId: "C-ATTN",
    excerptId: "c-attn-l1920",
    kind: "真实实现",
  },
  {
    id: "rope",
    title: "RoPE",
    shape: "Q/K → 同形，V 不变",
    formula:
      "(a,b)\\mapsto(a\\cos\\phi-b\\sin\\phi,b\\cos\\phi+a\\sin\\phi),\\quad\\phi=p\\theta^{-2i/d}",
    explanation:
      "split-half 将维0与2、维1与3配对。position IDs=[0,1,2,3]，theta=10000，全维旋转；绝非相邻维旋转。",
    condition:
      "rotary_interleaved=false；上游也支持 interleaved/fused/dynamic 分支，本 fixture 只覆盖声明的路径。",
    verify: "位置0不变；每一对旋转前后平方和保持；Q/K使用各自token位置。",
    sourceId: "C-ROPE",
    excerptId: "gqa-rope-layout",
    kind: "真实实现",
  },
  {
    id: "scores",
    title: "QK 打分",
    shape: "Q[1,4,4,4] × K → [1,4,4,4] (token,head,key)",
    formula: "s_{t,h,j}=Q^R_{t,h}\\cdot K^R_{j,g(h)}",
    explanation:
      "固定 query token 和 head，与所有 key 做点积。列是 key 位置，不是 hidden 维。此时未来位置仍有原始分数。",
    condition:
      "下面矩阵是数学等价 reference，真实 core attention 可在 fused kernel 内完成；不是抓取的 score。",
    verify: "把四个乘积相加核对一个 score；故意读取另一 KV group，结果应不同。",
    sourceId: "C-ATTN",
    excerptId: "gqa-core-boundary",
    kind: "数学等价参考",
  },
  {
    id: "scale",
    title: "缩放",
    shape: "[1,4,4,4] → 同形",
    formula: "z_{t,h,j}=s_{t,h,j}/\\sqrt d",
    explanation:
      "d=4，所以除以2；不是除以H或head数。缩放会改变后续softmax的集中程度。",
    condition:
      "仅标准缩放教学路径，未启用额外 query-key scaling、attention bias 或 softmax 变体。",
    verify:
      "切换“漏缩放”反例，观察概率和最终residual确实变化，不仅改说明文字。",
    sourceId: "C-ATTN",
    excerptId: "gqa-core-boundary",
    kind: "数学等价参考",
  },
  {
    id: "mask",
    title: "Causal mask",
    shape: "[1,4,4,4] → 同形",
    formula:
      "\\widetilde z_{t,h,j}=\\begin{cases}z_{t,h,j}&j\\le t\\\\-\\infty&j>t\\end{cases}",
    explanation:
      "本fixture以key位置≤query位置为允许条件，对角线允许。mask用负无穷，不用乘零：零仍可能得到正概率。",
    condition:
      "不含padding/packed跨样本mask。全mask行在本教学实现报错，不能外推所有Megatron后端。",
    verify: "query=0时仅key0有效；改变未来token不能影响更早输出。",
    sourceId: "C-ATTN",
    excerptId: "gqa-core-boundary",
    kind: "数学等价参考",
  },
  {
    id: "softmax",
    title: "Softmax",
    shape: "[1,4,4,4] → 同形",
    formula:
      "P_j=\\frac{\\exp(\\widetilde z_j-m)}{\\sum_k\\exp(\\widetilde z_k-m)},\\quad m=\\max_k\\widetilde z_k",
    explanation:
      "沿key轴逐行归一化。先减最大值防止指数溢出；屏蔽位置exp(-∞)=0。",
    condition: "float64确定性CPU参考，dropout=0；没有测BF16/GPU误差。",
    verify: "每行和为1，所有未来位置概率为0；全mask行必须显式失败。",
    sourceId: "C-ATTN",
    excerptId: "gqa-core-boundary",
    kind: "数学等价参考",
  },
  {
    id: "value",
    title: "加权 V",
    shape: "P[1,4,4,4] × V → [1,4,4,4] (token,head,d)",
    formula: "O_{t,h,i}=\\sum_j P_{t,h,j}V_{j,g(h),i}",
    explanation:
      "同一概率行分别加权V的四个feature。读取的V group必须与打分使用的K group一致，输出轴现在是feature而非key。",
    condition: "每组共享K/V，但各Q head有自己的概率分布，因此不要求输出相同。",
    verify: "逐项核对一个weighted sum；query=0的head输出恰等于V[0,group]。",
    sourceId: "C-ATTN",
    excerptId: "gqa-core-boundary",
    kind: "数学等价参考",
  },
  {
    id: "merge",
    title: "合并 heads",
    shape: "[1,4,4,4] → [1,4,16]",
    formula: "C_t=\\operatorname{concat}_{h=0}^{n_q-1}O_{t,h}",
    explanation:
      "按Q head顺序拼接，不求和。4个head各4维合成16维；此处还不能与8维residual相加。",
    condition:
      "上游core后端输出布局与packed/dynamic分支相关；调用边界不等于总存在单独concat kernel。",
    verify: "C[t,4h+i]必须等于O[t,h,i]；改成平均heads会丢维度和信息。",
    sourceId: "C-ATTN",
    excerptId: "gqa-output-projection",
    kind: "调用入口",
  },
  {
    id: "projection",
    title: "输出投影",
    shape: "[1,4,16] × [16,8] → [1,4,8]",
    formula: "Y=CW_O+b_O",
    explanation:
      "将Q总宽度投影回hidden宽度。本fixture明确16→8，权重按逻辑[in,out]存储，上游Linear常以[out,in]存储。",
    condition: "output_bias=0；不启用output gate。真实Qwen3-0.6B为2048→1024。",
    verify: "选一个输出维，展开16项乘加；输出长度必须为8。",
    sourceId: "C-ATTN",
    excerptId: "gqa-output-projection",
    kind: "调用入口",
  },
  {
    id: "residual",
    title: "Residual add",
    shape: "[1,4,8] + [1,4,8] → [1,4,8]",
    formula: "R=X+Y",
    explanation:
      "加回最初保存的X，不是归一化后的U。此后进入本层FFN，层栈、final norm/head和SFT目标仍在整模路线中。",
    condition:
      "教学dropout=0；真实bias/dropout/add可融合，不能从这里推断真实训练随机数或精度。",
    verify: "逐项检查R-X=Y；未来token扰动测试在最终R上比较，不能只检查mask。",
    sourceId: "C-LAYER",
    excerptId: "gqa-residual",
    kind: "调用入口",
  },
] as const;
export type Substep = (typeof steps)[number]["id"];
export function stepFor(id: string) {
  return steps.find((s) => s.id === id) || steps[0];
}
