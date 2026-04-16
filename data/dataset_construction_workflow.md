# 新数据集构建流程

## 目标

在保持现有输出格式不变的前提下，构建更可信的数据集：

- 语料文件：`retrieval_corpus_v1.jsonl` 风格
- 基准文件：`retrieval_benchmark_v1.jsonl` 风格

模型分工：

- 模型 A：从文档反推约束 + 受控改写 query
- 模型 B：独立审查与裁决

---

## 基础数据集选择

- numeric：fiqa
- exclusion：dbpedia-entity
- negation：scifact

---

## 模型与组件选型

1. 初检：BM25
2. 重排：cross-encoder
3. 模型 A：GLM 5.1
4. 模型 B：Claude Sonnet 4.6

说明：A/B 采用不同模型家族，降低同源偏差。

---

## 检索参数

统计基线：

- fiqa：57,638 docs；qrels 平均正样本 2.57/query
- dbpedia-entity：4,635,922 docs；qrels 平均正样本 35.74/query
- scifact：5,183 docs；qrels 平均正样本 1.13/query

建议参数：

| 数据集 | BM25 top-k | 种子近邻 top-k | 最终候选池大小 |
| --- | ---: | ---: | ---: |
| fiqa | 100 | 20 | 60 |
| dbpedia-entity | 150 | 8 | 70 |
| scifact | 200 | 30 | 80 |

统一约束：

1. 每个 query 仅使用最多 3 个种子文档做近邻扩展（按 qrels 分数和检索分排序）。
2. 候选池先取并集再去重，若超出目标大小，按综合分截断到“最终候选池大小”。
3. 若低于目标大小，用 BM25 后续文档补足。

---

## 输出字段定义

每条样本至少包含：

- `query_id`
- `query`
- `category`（numeric/exclusion/negation）
- `topical_relevant_doc_ids`
- `constraint_satisfying_doc_ids`
- `graded_relevance`（doc_id -> 0/1/2）

语义关系：

- `topical_relevant_doc_ids`：主题相关文档集合
- `constraint_satisfying_doc_ids`：主题相关且满足约束的子集
- 差集（topical - satisfying）：常见“主题相关但不满足约束”文档

---

## 流程

### 1. 扩大候选池

对每个原始 query：

1. BM25 初检：按上表取 top-k
2. 以 qrels 正样本为种子做文档近邻扩展：按上表取 top-k
3. 合并去重并截断：按上表得到最终候选池 `C(q)`

### 2. 主题过滤（得到 topical）

在 `C(q)` 上用 cross-encoder 重排并打 topical 分，保留高分文档，形成：

- `T(q) = topical_relevant_doc_ids`

### 3. 模型 A：从文档反推可分割约束

输入：原 query + `T(q)` 文档片段。  
输出（结构化 JSON）：

- 约束类型与约束定义
- satisfy / violate / uncertain 划分
- 每条判断对应证据句
- 置信度

### 4. 模型 A：受控改写 query

仅做最小编辑，不做自由重写：

1. 保留主题实体与核心词
2. 仅显式注入约束表达
3. 输出原 query、改写 query、保留词、新增约束词
4. 保证语义通顺

### 5. 模型 B：独立审查与裁决

模型 B 只负责审核：

1. 改写是否保持主题
2. 约束是否表达清楚
3. 文档标签与证据是否一致

低置信样本进入 uncertain 或丢弃，不强行打标。

### 6. 生成最终标签

- `topical_relevant_doc_ids`：审核通过的 topical 文档
- `constraint_satisfying_doc_ids`：topical 中满足约束的文档
- `graded_relevance`：
  - 2：主题相关且满足约束（高置信）
  - 1：主题相关但部分满足或不确定
  - 0：不相关或明确违反约束

---

## 质量门槛

1. 每条样本保留证据句（可回指原文）
2. `constraint_satisfying_doc_ids` 必须是 `topical_relevant_doc_ids` 子集
3. `graded_relevance` 只能取 0/1/2
4. 允许 uncertain，避免硬判导致噪声
