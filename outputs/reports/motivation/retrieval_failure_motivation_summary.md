# Motivation

## 1. 结果

传统检索器即使能检索到“主题相关”的文档，也会在 top-k 中频繁暴露**违反约束**的证据文档。


| Method | vr@1 | vr@3 | vr@5 | vr@10 | First Violating Rank Mean | Median |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.4467 | 0.4578 | 0.4107 | 0.2410 | 3.7633 | 2 |
| BGE | 0.4567 | 0.5133 | 0.3787 | 0.2333 | 3.5433 | 2 |
| Contriever | 0.6667 | 0.4822 | 0.3813 | 0.2197 | 3.6200 | 1 |

| Method | Negation vr@5 | Exclusion vr@5 | Numeric vr@5 |
|---|---:|---:|---:|
| BM25 | 0.468 | 0.600 | 0.164 |
| BGE | 0.600 | 0.536 | 0.000 |
| Contriever | 0.544 | 0.600 | 0.000 |

---

## 2. 不足

### 2.1 证明了“检索暴露问题”，但还没有直接证明“RAG 生成因此变差”
当前脚本测的是检索阶段的 violation exposure，而不是最终回答质量。因此它更像：
- **retrieval-side motivation**，不是完整的 **end-to-end motivation**。

### 2.2 numeric 类别的说服力偏弱
在当前结果里：
- BGE numeric `vr@5 = 0.0`
- Contriever numeric `vr@5 = 0.0`
- BM25 numeric `vr@5 = 0.164`

这说明 numeric 约束上，问题并不像 negation / exclusion 那样稳定突出。

---

## 3. 指标

### 3.1 violating document
> violation_doc = topical relevant document - constraint satisfying document

这个文档和 query 的主题是相关的；但它**不满足约束条件**；因此它属于“相关但有害”的文档。

例如：
- query: “Find hotels that are **not dirty**”
- 一个介绍“dirty hotel”的文档可能仍然 topical relevant；
- 但它不满足 “not dirty” 这一约束；所以它会被算作 violating doc。

### 3.2 Violation Rate@k

> VR@k = top-k 结果中 violating docs 的数量 / k

### 3.3 First Violating Rank

> 从前 max_k 个结果开始扫描，第一个 violating doc 出现的位置就是 first violating rank。

如果在前 `max_k` 内都没有出现 violating doc，则返回：max_k + 1


### 3.4 Category-level VR@5
> 对每一类约束（negation / exclusion / numeric），分别统计该类 query 的平均 VR@5。

---

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

---

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

---

## 5. 三张图

### 5.1 `violation_rate_at_k.png`

结果：
- BM25: `vr@1=0.4467, vr@3=0.4578, vr@5=0.4107, vr@10=0.2410`
- BGE: `0.4567, 0.5133, 0.3787, 0.2333`
- Contriever: `0.6667, 0.4822, 0.3813, 0.2197`

解读：
- 在非常靠前的位置，violating evidence 占比很高；
- 随着 k 增大，比例下降，但 top-10 里依然不可忽略；
- 这说明错误证据不只是偶发插入，而是系统性暴露。

---

### 5.2 `first_violating_rank_boxplot.png`

结果：
- BM25: mean `3.7633`, median `2`
- BGE: mean `3.5433`, median `2`
- Contriever: mean `3.6200`, median `1`

解读：
- 中位数很小，说明至少一半以上 query 都会在很前面遇到 violating doc；
- Contriever 的中位数为 1，表示它更容易把 violating 文档直接排到第一。

---

### 5.3 `violation_rate_at_5_by_category.png`

结果：
- Negation：BM25 0.468 / BGE 0.600 / Contriever 0.544
- Exclusion：BM25 0.600 / BGE 0.536 / Contriever 0.600
- Numeric：BM25 0.164 / BGE 0.000 / Contriever 0.000

解读：
- 否定和排除类约束是最明显的失败来源；
- numeric 类别在 dense retriever 上并不突出
