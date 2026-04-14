我已经按你给的生成指导文件，把这版 SciFact negation benchmark 完整导出了；主输出严格保留了 query-level 的 `satisfying / violating / judged_irrelevant / graded_relevance` 结构，而不是 query-doc pair。整体 schema、字段语义和文件拆分都按你那份规范执行。

主文件在这里：
[scifact_negation_corpus.jsonl](sandbox:/mnt/data/scifact_negation_corpus.jsonl)
[scifact_negation_metadata.jsonl](sandbox:/mnt/data/scifact_negation_metadata.jsonl)

我另外附了两份便于你们抽检和复现的辅助文件：
[scifact_negation_audit.jsonl](sandbox:/mnt/data/scifact_negation_audit.jsonl)
[scifact_negation_stats.json](sandbox:/mnt/data/scifact_negation_stats.json)
[build_scifact_negation_dataset.py](sandbox:/mnt/data/build_scifact_negation_dataset.py)

这版产物的规模是：

* corpus：5183 篇文档
* semantic units：200 个
* final queries：600 条
* metadata 中每条 query 都有 satisfying、violating、judged_irrelevant 和 graded_relevance

需要你特别注意的一点是：这版已经是“完整可跑”的 benchmark 文件，但 `satisfying_doc_ids / violating_doc_ids` 这层不是直接继承原始 SciFact gold，而是我基于 full corpus 做的 **abstract-level weakly supervised re-judging**。也就是说，它现在更适合当你们的 **v0 / machine-constructed benchmark**：结构完整、能直接进评测脚本、也有 audit 轨迹；但如果后面要进论文主结果，最稳的做法还是对 audit 里一部分单元做人工复核，再把修正写回 metadata。

下一步最值当的是先抽检 `scifact_negation_audit.jsonl` 里几十个 semantic units，优先看 evidence sentence 和 doc label 是否符合你们想要的严格标准。
