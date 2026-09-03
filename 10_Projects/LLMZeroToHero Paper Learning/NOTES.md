# Teaching notes

- 用户偏好中文、按因果链学习论文，并把成熟结论落入 AI-Infra。
- 每篇课文只链接刚好够用的数学前置，不要求先学完通用数学课程。Math Foundation 讲 model-agnostic 的定义、性质、一般公式、成立条件与中性例子；Lesson 负责 architecture mapping、tensor axis、论文公式、超参数、证据与边界；Learning Record 记录用户自己的理解、修正与尚存疑问。
- 不在 Math Foundation 与 Lesson 中各保存一份相同的应用推导。若一般性质需要映射到论文机制，Foundation 给出原理，Lesson 用简短的 application mapping 完成实例化。
- 新内容只有在脱离当前论文后仍成立、可用 model-agnostic symbols 表达、且正文不依赖 architecture component、paper hyperparameter 或 benchmark result 时，才进入 Math Foundation。
- 当前从 `Attention Is All You Need` 开始；不要把首课的讲解误标为已掌握或已沉淀的正式知识。
- 术语规范：English-first；首次写作 `English（中文解释）`，后续优先复用 English；不把代码、公式、论文标题或可检索术语改写为纯中文。
- 术语页规则：从 AdamW Lesson `0013` 开始，每个新 lesson 都在 `reference/` 下维护一个同号、独立的 `<lesson-number>-<topic>-glossary.html`，只收录该课引入或重新精确定义的 terms、symbols 与易混边界；lesson 首次出现仍保留一句就地解释，并链接专属 glossary。旧课程 `0001–0012` 与现有 `attention-glossary.html` 暂不迁移，除非用户另行要求。
- 数学规范：所有数学变量、shape、等式和推导统一使用 LaTeX。HTML 课程使用 `\(...\)` 表示 inline math、`\[...\]` 表示 display math，并加载 MathJax；Markdown 笔记使用 `$...$` 与 `$$...$$`。不要再用 HTML `<sub>/<sup>`、Unicode 数学符号或纯文本伪公式拼写数学表达式。
- 编号与课程范围：lesson / learning-record 均纯顺序整数 +1，一篇论文拆多课占用连续号；论文 → 编号范围的权威映射维护在 `CURRICULUM.md`，新增/拆分/合并/重排时先更新该表再改文件。
