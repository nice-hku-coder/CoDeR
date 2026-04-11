# Motivation

## 1. 结论

传统检索器即使能检索到主题相关文档，也会在 top-k 中频繁暴露 **constraint-violating** 文档。

3 个基线：BM25、BGE、Contriever。

| Method | Recall@5 | nDCG@5 | VR@5 |
|---|---:|---:|---:|
| BM25 | 0.4106 | 0.5135 | 0.4107 |
| BGE | 0.6339 | 0.7694 | 0.3787 |
| Contriever | 0.6333 | 0.7825 | 0.3813 |

- **BGE / Contriever** 的 `Recall@5` 和 `nDCG@5` 比较高，说明它们的 topic retrieval 能力最强；但它们的 `VR@5` 仍接近 0.38，说明topic 命中强并不等于能避开违反约束的文档
- **BM25** 的 topic 指标明显低于两个 dense retriever，同时 `VR@5 = 0.4107`，说明词法匹配在该任务上既不够强，也无法避免 violation exposure
- 从曲线图看，**Contriever** 在 `k=1` 时 violation 最严重（`VR@1 = 0.6667`）；从箱线图看，Contriever 的 first violating rank 中位数最低，说明 violating evidence 往往在最前面就出现
- 从类别柱状图看，问题主要集中在 **negation / exclusion**；numeric 类别整体较轻，且在 BGE / Contriever 上几乎不暴露 top-5 violation

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

## 5. 总结

- **Topic-side 指标不差**（尤其是 BGE / Contriever），说明问题不是“检索器完全不会检索”；
- **Constraint-side 指标仍然较差**，说明现有方法没有显式建模 constraint satisfaction；
- 即使换成更强的 dense retriever，violation exposure 依然存在，说明问题不只是 BM25 太弱，而是 retrieval objective 本身缺少 constraint awareness。
