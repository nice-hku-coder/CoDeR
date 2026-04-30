# Motivation

## 1. 结论

传统检索器即使能检索到 topic-level relevance 文档，也会在 top-k 中频繁暴露 **constraint-violating** 文档。

6 个方法：EncoderA、EncoderB、DualFusion、BM25、BGE、Contriever。

| Method | Recall@5 | nDCG@5 | VR@1 | VR@3 | VR@5 | VR@10 |
|---|---:|---:|---:|---:|---:|---:|
| EncoderA | 0.6364 | 0.7073 | 0.1591 | 0.4545 | 0.4318 | 0.2750 |
| EncoderB | 0.6068 | 0.6803 | 0.1705 | 0.4432 | 0.4045 | 0.2705 |
| DualFusion | 0.6364 | 0.7083 | 0.1591 | 0.4545 | 0.4318 | 0.2750 |
| BM25 | 0.9591 | 0.9712 | 0.2273 | 0.6818 | 0.7523 | 0.3932 |
| BGE | 0.6636 | 0.7373 | 0.0455 | 0.4432 | 0.4500 | 0.2773 |
| Contriever | 0.5455 | 0.6225 | 0.2159 | 0.3712 | 0.3455 | 0.2443 |

- **EncoderA / EncoderB / DualFusion** 的 `Recall@5` 和 `nDCG@5` 都明显低于 BM25，但高于或接近 Contriever，说明这条“约束相关”分支并没有把 topic-side 排名做强；其中 **DualFusion 与 EncoderA 几乎完全一致**，仅 `nDCG@5` 有极小提升，说明当前 `alpha/tau` 配置下融合几乎没有改变 top-k 排序。
- **BM25** 在这个 SciFact negation 子集上反而拿到了最强的 topic-side 指标（`Recall@5 = 0.9591`, `nDCG@5 = 0.9712`），但 `VR@5 = 0.7523` 也最高，说明它非常擅长命中主题相关文档，同时也最容易把违反约束的文档一起排到前面。
- **BGE** 是这组结果里最平衡的一条 dense baseline：topic-side 高于 Contriever，constraint-side 也显著优于 BM25，尤其 `VR@1 = 0.0455` 很低，说明它更不容易把 violating doc 顶到第一位。
- **Contriever** 的 topic-side 最弱（`Recall@5 = 0.5455`, `nDCG@5 = 0.6225`），但 violation exposure 仍然不低，说明语义检索强弱与是否遵守约束是两件不同的事。

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

### 4.2 EncoderA / EncoderB / DualFusion / BGE / Contriever 分数

流程是：
1. 用句向量模型把所有文档编码成 embedding；
2. 对 query 编码成 embedding；
3. 归一化向量；
4. 计算点积后排序，等价于 **cosine similarity**。

其中：
- **EncoderA** 使用 `sentence-transformers/all-MiniLM-L6-v2`，对应 topic encoder；
- **EncoderB** 使用 `outputs/checkpoints/constraint-encoder-v1`，对应 constraint encoder；
- **DualFusion** 先用 EncoderA 做候选召回，再按 EncoderA / EncoderB 的分数做融合，并用 `tau` 过滤，和 [eval_retrieval_metrics.py](../experiments/eval_retrieval_metrics.py) 的逻辑一致；
- **BGE** 会把 query 改写成：

```text
Represent this sentence for searching relevant passages: <query>
```

Contriever 不加前缀，直接编码 query。

### 4.3 为什么 DualFusion 没有明显优于 EncoderA

当前结果里，DualFusion 与 EncoderA 在 `Recall@5`、`VR@1`、`VR@3`、`VR@5`、`VR@10` 上完全一致，只在 `nDCG@5` 上有极小提升（`0.7083` vs `0.7073`）。这说明：

1. 候选集基本没有变，DualFusion 没有把更多新的文档推到 top-k；
2. `tau` 过滤和 `alpha` 融合在当前设置下对排序边界的影响很弱；
3. 这个 checkpoint 更像是“可作为约束分支使用的 encoder B”，但单独拿出来并不能自动形成一个更强的双路模型。

## 5. 实验图解读

### 5.1 Violation Rate@k 曲线

`violation_rate_at_k.png` 显示：

- **BM25 的 violation exposure 最重**，`VR@5 = 0.7523`、`VR@10 = 0.3932`，说明它虽然能强力命中 topic-side 文档，但几乎不区分约束是否被满足。
- **BGE 的 top-1 最干净**，`VR@1 = 0.0455`，这意味着它很少把 violating doc 顶到第一位；但随着 k 增大，`VR@5` 和 `VR@10` 仍然回到与其他 dense 方法相近的区间。
- **EncoderA / EncoderB / DualFusion 三条曲线几乎重合**，说明这次 dual fusion 没有改变主要排序结构。
- **Contriever 的 topic-side 反而最弱，但 violation exposure 并不低**，表明“语义更弱”不代表“更守约束”。

### 5.2 First Violating Rank 箱线图

`first_violating_rank_boxplot.png` 显示：

- 所有方法的 first violating rank 都偏低，说明 violating document 往往很早出现。
- **BGE 的分布最靠后一些**，和它极低的 `VR@1` 一致，说明它在 top-1 上更不容易踩到 violating doc。
- **BM25 的箱体更靠前且更分散**，和它较高的 VR 曲线一致，说明它对 constraint satisfaction 的区分能力最弱。
- **EncoderA / EncoderB / DualFusion** 的箱线图差异很小，进一步说明当前融合配置没有形成稳定优势。
- 图中虚线表示 top-10 内未出现 violation 的哨兵值 `max_k + 1 = 11`；多数箱体远低于该线，说明“top-10 内完全没有 violating document”并不是常态。

### 5.3 Category-level VR@5 柱状图

`violation_rate_at_5_by_category.png` 显示：

- **Negation 与 exclusion 仍然是主要失败类型**，但这次结果里 BM25 的 violation exposure 更高，说明词法匹配会更容易把“主题对但违反约束”的文档推上来。
- **BGE 在 negation 上的表现最好**，这与它更低的 `VR@1` 一致。
- **EncoderA / EncoderB / DualFusion 在三类上的柱状高度接近**，说明它们在类别级别上也没有被当前融合策略拉开。
- **Numeric 类整体仍然最低**，说明数值约束仍然更容易被当前 embedding 检索区分；但这并不代表模型真正理解了数值约束，只能说明在这个 benchmark 上它更少把 numeric violating doc 推入 top-5。
