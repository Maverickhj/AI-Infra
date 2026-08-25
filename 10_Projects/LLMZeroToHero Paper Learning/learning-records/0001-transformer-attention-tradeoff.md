# 已建立：以全局 self-attention 换取层内并行

用户能够从 RNN 的 $h_t$ 必须等待 $h_{t-1}$ 的递推依赖，解释 Transformer 以全位置两两 attention score 的平方成本换取 layer 内直接连接与并行计算。后续课程需要保持三个边界：完整 self-attention 的单层复杂度为 $O(n^2 d)$，Encoder 可关注所有位置，而 Decoder self-attention 受 causal mask 限制且生成过程仍是自回归串行。

## Evidence

用户在课程 0001 的检索题中自主说明了“$O(n^2)$ 的矩阵计算”与 RNN 隐状态递推串行依赖之间的替换关系。
