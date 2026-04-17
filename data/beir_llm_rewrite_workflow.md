# BEIR 子集 LLM 改写构建方案

## 目标

以真实 BEIR query/doc 为底座，用小规模 LLM 改写构建 numeric/exclusion/negation 三类数据，降低模板僵硬感，同时保持可控成本。

## 数据规模

1. fiqa 200 条（numeric）。
2. dbpedia-entity 200 条（exclusion）。
3. scifact 200 条（negation）。
4. 首轮共 600 条；若成本与通过率可接受，再扩到每类 300-500。

## 核心流程

1. Query 采样
   - 选长度适中的 query（过滤过短/过长）。
   - 记录 source_dataset/source_query_id。

2. Query 改写（LLM）
   - 按类别改写为约束 query：numeric/exclusion/negation。
   - 约束：保留主题实体与核心术语，不改变检索主题。

3. 候选文档检索
   - 对改写后 query 检索 top-12（qrels 命中文档优先，其余由 BM25 补齐）。
   - 从 top-12 选 5 条：优先保留 2 条 qrels 文档，再补 3 条语义去重文档。

4. 文档改写（LLM）
   - 将改写后 query + 5 条完整文档作为输入（不截断）。
   - 在保持主题相关的前提下：
     - 改写 2 条为“满足约束”；
     - 改写 1 条为“违反约束”；
     - 其余 2 条不改写，保留为 topical 文档。

5. 标签写入
   - topical_relevant_doc_ids：这 5 条文档全部纳入。
   - constraint_satisfying_doc_ids：2 条满足约束文档。
   - graded_relevance：
     - 2：满足约束的 2 条；
     - 1：未改写的 topical 2 条；
     - 0：违反约束的 1 条。
   - 候选池外文档默认视为 0（无需显式写入所有文档分数）。