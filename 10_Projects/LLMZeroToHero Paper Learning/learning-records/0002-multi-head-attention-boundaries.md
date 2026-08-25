# 已建立：Multi-Head Attention 的计算配比与子空间边界

用户已知道原始 Transformer 令每个 head 的 $d_k=d_v=d_{model}/h$，因而总 projection 与 attention 的理论计算量和单个 full-dimensional head 相近。后续需保留边界：不同 head 使用独立投影，因而模型能够学习不同 representation subspaces 与关注模式，但没有正交或不重叠的数学保证；实际 kernel 开销与物化的 attention-score memory 也不必完全相同。

## Evidence

用户在课程 0002 后准确说明“多个独立的低维投影”以近似不增加理论计算量的方式，使模型可以在不同 representation subspaces 和位置关系上进行 attention；其中“可以”正确保留了这是一种可学习能力、不是 head 互不重合的硬约束。
