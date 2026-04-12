# Motivation

## 1. 结论

传统检索器即使能检索到 topic-level relevance 文档，也会在 top-k 中频繁暴露 **constraint-violating** 文档。

4 个基线：BM25、BGE、Contriever、HyDE。

| Method | Recall@5 | nDCG@5 | VR@1 | VR@3 | VR@5 | VR@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3372 | 0.4435 | 0.4467 | 0.4122 | 0.3773 | 0.2107 |
| BGE | 0.6339 | 0.7694 | 0.4567 | 0.5133 | 0.3787 | 0.2333 |
| Contriever | 0.6333 | 0.7825 | 0.6667 | 0.4822 | 0.3813 | 0.2197 |
| HyDE | 0.6317 | 0.7832 | 0.5633 | 0.3333 | 0.3793 | 0.2470 |

- **BGE / Contriever / HyDE** 的 `Recall@5` 和 `nDCG@5` 都明显高于 BM25，说明 dense retriever 与 HyDE-style query expansion 的 topic retrieval 能力更强；但三者的 `VR@5` 都接近 `0.38`，说明 topic 命中强并不等于能避开违反约束的文档。
- **BM25** 的 topic-side 指标最低（`Recall@5 = 0.3372`, `nDCG@5 = 0.4435`），但 `VR@5 = 0.3773` 仍然不低，说明词法匹配既不够强，也没有显式的 constraint satisfaction 能力。
- **Contriever** 的 `VR@1 = 0.6667` 最高，说明它虽然有很强的语义排序能力，但最靠前结果非常容易暴露 constraint-violating evidence。
- **HyDE** 在 topic-side 上基本保持 Contriever 水平（`Recall@5 = 0.6317`, `nDCG@5 = 0.7832`），但 constraint-side 并没有稳定改善：它把 `VR@3` 从 Contriever 的 `0.4822` 降到 `0.3333`，说明 hypothetical document embedding 能在部分中间 rank 段缓解 violation exposure；但它的 `VR@1 = 0.5633` 仍显著高于 BM25/BGE，`VR@5 = 0.3793` 与 BGE/Contriever 几乎持平，`VR@10 = 0.2470` 反而是四个方法中最高。整体说明 HyDE 能保持 topic retrieval，但不能可靠地保留 negation、exclusion、numeric 等约束。

## 2. 两类指标

### Topic-side
- **Recall@5**：前 5 个结果中，覆盖了多少 `topical_relevant_doc_ids`
- **nDCG@5**：基于 `graded_relevance` 的排序质量

### Constraint-side
- **Violation Rate@k**：top-k 中 violating docs 的比例
- **First Violating Rank**：第一个 violating doc 出现的位置
- **Category-level VR@5**：按 negation / exclusion / numeric 统计的平均 VR@5

## 3. 指标定义

### 3.1 violating document

> violating doc = topical relevant doc - constraint satisfying doc

文档主题相关，但不满足 query 约束。

### 3.2 Recall@5

> Recall@5 = top-5 中命中的 topical relevant docs 数 / topical relevant docs 总数

把 `topical_relevant_doc_ids` 作为正例集合。

### 3.3 nDCG@5

> nDCG@5 衡量前 5 个结果的排序质量，使用 `graded_relevance` 计算。

### 3.4 Violation Rate@k

> VR@k = top-k 中 violating docs 的数量 / k

### 3.5 First Violating Rank

> 从前 max-k 个结果开始扫描，第一个 violating doc 出现的位置。

若前 max-k 内没有 violating doc，则记为 `max_k + 1`。

## 4. 各基线的检索分数计算

### 4.1 BM25

总分计算：
$$\text{score}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

逆文档频率：$\text{IDF}(q_i) = \ln(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5})$

词频饱和度：$\frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1}$

文档归一化长度：$1 - b + b \cdot \frac{|D|}{\text{avgdl}}$

其中：
- `N`：文档总数
- `n`：包含该 term 的文档数
- `freq`：term 在当前文档中的词频
- `dl`：当前文档长度
- `avgdl`：平均文档长度

最终对 query 中每个 term 累加 BM25 分数；然后对所有文档分数降序排序。

### 4.2 BGE / Contriever 分数

流程是：
1. 用句向量模型把所有文档编码成 embedding；
2. 对 query 编码成 embedding；
3. 归一化向量；
4. 计算点积后排序，等价于 **cosine similarity**。

即 BGE 会把 query 改写成：

```text
Represent this sentence for searching relevant passages: <query>
```

Contriever 不加前缀，直接编码 query。

### 4.3 HyDE 分数

当前 HyDE baseline 采用“保留 motivation 受控语料与标签、只对齐 HyDE query-side 表示构造”的方案：不切换到原仓库的 MS MARCO Faiss index，以保证 `topical_relevant_doc_ids` 和 `constraint_satisfying_doc_ids` 仍然可用于计算 violation exposure。

1. 使用 prompt 让 GPT-5 为 query 生成 hypothetical passage：

```text
Please write a passage to answer the question.
Question: <query>
Passage:
```

2. 默认生成 `n=8` 个 hypothetical documents；
3. 使用 Pyserini 的 `AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')` 对 corpus 文档编码；
4. 使用同一个 `AutoQueryEncoder` 对 `[query] + [hypothetical documents]` 逐条编码；
5. 对这些 embedding 求平均，得到 HyDE query vector；
6. 将平均后的 HyDE 向量与本地 motivation corpus 的 document embeddings 计算点积并排序。

HyDE 的检索向量不是单独的 query embedding，而是一个由 query 与生成文档共同形成的 dense representation。

## 5. 实验图解读

### 5.1 Violation Rate@k 曲线

`violation_rate_at_k.png` 显示：

- **Contriever 在 `k=1` 最严重**，`VR@1 = 0.6667`，即三分之二 query 的 top-1 结果就是 constraint-violating 文档。
- **HyDE 的 `VR@1 = 0.5633` 低于 Contriever，但仍明显高于 BM25/BGE**，说明 HyDE 的 hypothetical document 表示不能保证把最靠前结果从 violating document 拉开。
- **HyDE 在 `k=3` 明显改善**，`VR@3 = 0.3333`，低于 BM25、BGE、Contriever，是 HyDE 在本实验中最明显的 constraint-side 收益。
- **到 `k=5` 时各方法收敛到约 0.38**：BM25 `0.3773`、BGE `0.3787`、Contriever `0.3813`、HyDE `0.3793`。这说明只要扩大到 top-5，违反约束的 topical 文档几乎都会混入结果集。
- **到 `k=10` 时 HyDE 最高**，`VR@10 = 0.2470`，高于 BM25、BGE、Contriever，说明 HyDE 并没有在更大的候选集合中持续降低 violation exposure。

### 5.2 First Violating Rank 箱线图

`first_violating_rank_boxplot.png` 显示：

- 四个方法的 first violating rank 都偏低，说明 violating document 往往很早出现。
- Contriever 和 HyDE 的中位数约为 1，意味着不少 query 的第一条结果就是 violating document。
- BGE 与 BM25 的中位数约为 2，但它们同样无法稳定避免早期 violation。
- 图中虚线表示 top-10 内未出现 violation 的哨兵值 `max_k + 1 = 11`；多数箱体远低于该线，说明“top-10 内完全没有 violating document”并不是常态。

### 5.3 Category-level VR@5 柱状图

`violation_rate_at_5_by_category.png` 显示：

- **Negation 与 exclusion 是主要失败类型**。各方法在这两类上的 `VR@5` 普遍较高，约在 `0.47` 到 `0.60` 之间。
- **HyDE 在 negation 上没有改善**，图中 HyDE 的 negation VR@5 约为 `0.60`，与 BGE 接近，并高于 BM25/Contriever。
- **HyDE 在 exclusion 上有一定改善**，约为 `0.54`，低于 BM25/Contriever 的约 `0.60`，也接近或略高于 BGE。
- **Numeric 类整体较低**，图中除了 BM25 有少量 violation 外，BGE、Contriever、HyDE 基本接近 0。这说明当前 synthetic benchmark 中，数值约束对 dense retriever 的区分可能比 negation/exclusion 更容易，或者相关文档的 embedding 分布更容易被价格数值线索分开。
