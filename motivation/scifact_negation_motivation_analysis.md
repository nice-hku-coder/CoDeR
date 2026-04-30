# SciFact Negation Motivation Analysis

## 1. 实验设置

本次分析使用如下数据集：

- benchmark: [CoDeR/data/processed/scifact_negation_benchmark_v1.jsonl](../data/processed/scifact_negation_benchmark_v1.jsonl)
- corpus: [CoDeR/data/processed/scifact_negation_corpus_v1.jsonl](../data/processed/scifact_negation_corpus_v1.jsonl)

评估方法为 BM25、BGE、Contriever 和 ConstraintEncoder，分析脚本为 [CoDeR/motivation/retrieval_failure_motivation.py](retrieval_failure_motivation.py)。

该 benchmark 共包含 88 个 negation 查询，每个查询的 topical relevant 文档数固定为 5，constraint_satisfying 文档是其中满足否定约束的子集。因为 topical set 已经被控制得非常紧，主题检索与约束规避的差异会在这个数据集上被放大。

## 2. 结果概览

| Method | Recall@5 | nDCG@5 | VR@1 | VR@3 | VR@5 | VR@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.9591 | 0.9712 | 0.2273 | 0.6818 | 0.7523 | 0.3932 |
| BGE | 0.6636 | 0.7373 | 0.0455 | 0.4432 | 0.4500 | 0.2773 |
| Contriever | 0.5455 | 0.6225 | 0.2159 | 0.3712 | 0.3455 | 0.2443 |
| ConstraintEncoder | 0.6068 | 0.6803 | 0.1705 | 0.4432 | 0.4045 | 0.2705 |

### 2.1 直接结论

- **BM25 的 topic-side 指标最好**，`Recall@5 = 0.9591`、`nDCG@5 = 0.9712`，说明在这个 SciFact negation 任务上，词面线索足以非常准确地命中主题相关文档。
- **但 BM25 的约束暴露最严重**，`VR@5 = 0.7523`，意味着 top-5 里有很高比例是违反 negation 约束的文档。它“找对主题”不等于“找对可用证据”。
- **BGE 在 `VR@1` 上最好**，只有 `0.0455`，说明它最不容易把明显违反约束的文档放到第一位；但它的 topic-side 指标明显落后于 BM25。
- **Contriever 的总体 topic-side 表现最弱**，`Recall@5 = 0.5455`、`nDCG@5 = 0.6225`，同时约束暴露也没有形成稳定优势，说明纯语义检索并不能自动解决 negation 类约束。
- **ConstraintEncoder 没有形成预期中的明显优势**。它相比 Contriever 的 topic-side 指标略有提升，`Recall@5 = 0.6068`、`nDCG@5 = 0.6803`，但仍落后于 BGE 和 BM25；在 constraint-side 上，`VR@1 = 0.1705` 只是中等水平，`VR@5 = 0.4045` 也没有优于 BGE 的 `0.4500` 太多，说明它学到了一些约束相关信号，但没有把这些信号稳定转化成 top-k 排序优势。

### 2.2 现象解读

1. **高 Recall / 高 nDCG 不代表低 violation exposure。**
2. **BM25 在这个数据集上是“更会找主题，但更容易把不该出现的证据也排上来”。**
3. **BGE 和 Contriever 的 dense 表示并没有显式建模 negation 约束，因此只能部分改变排序，不能保证约束满足。**
4. **ConstraintEncoder 虽然训练时见过 entailment / contradiction 监督，但它学到的更像是 NLI 判别边界，不是 SciFact negation 检索所需的“可检索约束表示”。**

## 3. 图像解读

### 3.1 Violation Rate@k 曲线

对应图为 [CoDeR/outputs/figures/motivation/violation_rate_at_k.png](../outputs/figures/motivation/violation_rate_at_k.png)。

- **BM25 的 VR@1 不是最高，但在 k 增大后迅速变差**，到 `VR@5 = 0.7523`，说明它在前几位中持续暴露 violating documents。
- **BGE 的 VR@1 最低**，说明它对 top-1 的约束保护最好，但到 top-5 之后优势明显减弱。
- **Contriever 在 top-5 位置的 violation 暴露低于 BM25 和 BGE**，但它的 topic retrieval 也更弱，属于“少暴露一些，但也少找对一些”的状态。

这条曲线反映的是：**越往后看，三种方法都在持续混入违反约束的文档**。因此如果后续 RAG 系统只靠 top-k 检索，不做 constraint-aware reranking 或过滤，生成阶段会很容易继承这些错误证据。

### 3.2 First Violating Rank 箱线图

对应图为 [CoDeR/outputs/figures/motivation/first_violating_rank_boxplot.png](../outputs/figures/motivation/first_violating_rank_boxplot.png)。

- 大多数查询的 first violating rank 都很靠前，说明 violating docs 不只是“偶尔混入”，而是**早期就出现**。
- BGE 的 top-1 保护最好，但这不意味着它能在 top-3 或 top-5 稳定压制 violation。
- BM25 和 Contriever 的分布更容易在前几名就碰到 violating docs，说明它们在 negation 任务上都缺少“约束优先”的排序偏置。

### 3.3 Category-level VR@5

对应图为 [CoDeR/outputs/figures/motivation/violation_rate_at_5_by_category.png](../outputs/figures/motivation/violation_rate_at_5_by_category.png)。

这个数据集只有 negation 类，所以主要看的是 negation 下的整体表现。结合当前 benchmark 的构造方式可以得到一个明确判断：**negation 类是当前检索失败的核心来源**，也是后续做 constraint-aware reranking 最值得优先处理的场景。

## 4. 为什么会这样

### 4.1 BM25 的强项和弱项

BM25 依赖词面重合，因此在 SciFact 这类包含明显疾病名、药物名、数值和实验措辞的查询里，能够非常有效地把 topical evidence 拉上来。

但它不会理解“not / no / without / does not”这类否定语义，因此只要 violating document 的词面和 query 共享足够多的 token，BM25 就会把它排得很靠前。

### 4.2 BGE / Contriever 的强项和弱项

这两个 dense retriever 能在一定程度上缓解纯词面匹配带来的噪声，因此 `VR@1` 或中低 rank 段会更好一些。

但它们仍然是在优化“语义相似”，不是在优化“满足约束”。对于 negation 任务来说，语义上非常接近的文档里往往恰好混有违反约束的证据，所以 dense 表示并不会天然解决问题。

### 4.3 ConstraintEncoder 为什么没有明显优势

ConstraintEncoder 的训练方式决定了它更像一个“句对判别器”而不是“检索式约束编码器”：

- 训练数据来自 [CoDeR/data/processed/train_triplets.jsonl](../data/processed/train_triplets.jsonl)，其构造逻辑是 premise / positive / hard_negative 三元组。
- 训练损失是 `MultipleNegativesRankingLoss`，本质上只推动 query 靠近 positive、远离 hard_negative，并没有显式监督“约束维度”和“主题维度”需要分离。
- 训练语料主要来自 SNLI 风格的 NLI 关系，而当前 benchmark 是 SciFact 风格的事实检索，二者在表达上有明显域差：NLI 关心句子蕴含关系，SciFact 关心证据检索与文献排序。

因此 ConstraintEncoder 学到的信号更接近“这对句子是不是相关 / 是否矛盾”，而不是“在一批候选证据里，哪些文档既相关又满足 negation 约束”。这就解释了为什么它能比 Contriever 稍好一些，但无法压过 BGE，更没有从根本上改变 violation exposure。

从 benchmark/corpus 的结构看，问题还会被进一步放大：

- `scifact_negation_benchmark_v1.jsonl` 里的 topical 文档数只有 5，而且其中只有一小部分是真正 satisfying。
- `scifact_negation_corpus_v1.jsonl` 里保留下来的文档大多都和 query 在词面上高度接近，特别是带有 `not / no / reduce / lower / less / without` 之类模式的文本，很容易形成高语义相似但约束相反的近邻。
- 这意味着一个只学“语义拉近”的 dense encoder，即使比 Contriever 更敏感于否定，也依然会把大量 violating docs 排到前面。

换句话说，ConstraintEncoder 在这个数据集上更像是在做“更聪明的相似度排序”，而不是“约束可满足性排序”。

## 5. 结论

在 SciFact negation benchmark 上，BM25、BGE 和 Contriever 都能检到主题相关证据，但都不能稳定地避免违反约束的文档进入 top-k。

其中 BM25 的主题检索能力最强，但 violation exposure 也最严重；BGE 在 top-1 的约束保护最好；Contriever 在这个数据集上并没有形成明显优势；ConstraintEncoder 介于 BGE 和 Contriever 之间，说明训练过的 NLI 风格约束信号确实带来了一些帮助，但距离“真正的 constraint-aware retrieval”还有明显差距。

因此，这个实验最重要的结论不是“哪个检索器更好”，而是：**对 negation 类查询，传统检索器的排序目标与实际可用证据之间存在系统性偏差**。这正是后续引入 constraint-aware disentangling 的动机来源。