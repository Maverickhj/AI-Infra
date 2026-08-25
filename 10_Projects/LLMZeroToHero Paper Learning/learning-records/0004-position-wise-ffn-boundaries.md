# 已建立：Position-wise FFN 的功能与信息边界

用户已理解：attention 汇聚上下文后，FFN 在每个 token position 上增加非线性变换容量；两次 Linear 中间的 ReLU 会改变 position-wise representation，输出随后成为下一层的输入。

需保留以下精确边界：

- FFN 将每个 token representation 从 $d_{model}$ 临时扩展到 $d_{ff}$，经 activation 后再投影回 $d_{model}$。它增加的是可学习的中间 features 与非线性函数容量，不会永久增加下一层看到的 feature width。
- 对输入 $Z\in\mathbb{R}^{n\times d_{model}}$，有 $\operatorname{FFN}(Z)_{i,:}=\operatorname{FFN}(Z_{i,:})$。给定 $Z$ 后，position $i$ 的 FFN 不直接读取 position $j$；跨 position communication 发生在 attention。
- 同一 layer 的所有 positions 共享 $W_1,b_1,W_2,b_2$，因此可以并行执行；不同 Transformer layers 使用不同的 FFN 参数。
- 输出仍为 $[n,d_{model}]$，因而可与 residual path 相加，并作为下一层的统一 representation interface。

## Evidence

用户准确指出 FFN 在每个 token position 上引入更多可学习的中间 features，并说明 Linear–ReLU–Linear 会改变逐位置 representation，且这种变化会传递到下一层。上述补充将“增加 features”限定为中间层扩展，并明确不发生 token-to-token mixing。
