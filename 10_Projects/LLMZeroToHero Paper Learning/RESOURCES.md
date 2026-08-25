# LLMZeroToHero 论文因果学习资源

## Knowledge

- [论文原文：Attention Is All You Need（Vaswani et al., arXiv:1706.03762）](https://arxiv.org/html/1706.03762v7)
  Transformer 主线的一手来源。用于核对问题设定、attention 公式、张量投影、复杂度比较、Encoder–Decoder attention 的 Q/K/V 来源与实验条件。
- [论文条目与版本记录（arXiv）](https://arxiv.org/abs/1706.03762)
  用于核对作者、版本、引用信息和原始 PDF 入口。
- [论文原文：Adam: A Method for Stochastic Optimization（Kingma & Ba, arXiv:1412.6980）](https://arxiv.org/abs/1412.6980)
  Adam 支线的一手来源。用于核对 Algorithm 1、moment estimates、bias correction、effective stepsize、原始 convergence assumptions 与 experiments。
- 本地阅读材料：`/Users/jingshao/projects/LLMZeroToHero/translated/00_foundations/1706.03762__Attention_Is_All_You_Need_(Transformer).zh.dual.pdf`
  中英对照阅读入口。该文件元数据显示由 BabelDOC AI 生成，因此只作辅助阅读，关键结论回查论文原文。
- [LLMZeroToHero 阅读路线](https://github.com/jingyaogong/LLMZeroToHero)
  用于选择后续论文的因果邻接关系；不作为论文技术结论的来源。

## Wisdom (Communities)

- 当前不引入社区讨论。第一阶段先以原始论文和可回忆的因果解释建立概念基线；当需要比较论文实现与主流框架时，再选择相应项目的官方讨论区或代码库。

## Gaps

- 首篇只处理论文提出的原始 Encoder–Decoder Transformer。与现代 decoder-only LLM 的对应关系、FlashAttention 等内核实现和 KV cache 的系统后果，留待后续课程建立证据链。
