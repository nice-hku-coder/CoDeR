# Alpha Sweep (tau=0.2)

- Benchmark: `/home/xingkun/ant/data/processed/retrieval_benchmark_v1.jsonl`
- Topic model: `/data/xingkun/local_model/Llama-3.2-3B-Instruct`
- Constraint model: `/home/xingkun/ant/outputs/checkpoints/constraint-encoder-v1`

| alpha | Recall@10 | Recall@100 | NDCG@10 | CCR@10 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.8483 | 1.0000 | 0.8200 | 0.2667 |
| 0.05 | 0.8483 | 1.0000 | 0.8200 | 0.2667 |
| 0.10 | 0.8483 | 1.0000 | 0.8200 | 0.2667 |
| 0.15 | 0.8483 | 1.0000 | 0.8207 | 0.2667 |
| 0.20 | 0.8483 | 1.0000 | 0.8203 | 0.2667 |
| 0.25 | 0.8483 | 1.0000 | 0.8203 | 0.2667 |
| 0.30 | 0.8483 | 1.0000 | 0.8203 | 0.2667 |
| 0.35 | 0.8483 | 1.0000 | 0.8210 | 0.2667 |
| 0.40 | 0.8483 | 1.0000 | 0.8210 | 0.2667 |
| 0.45 | 0.8483 | 1.0000 | 0.8226 | 0.2667 |
| 0.50 | 0.8483 | 1.0000 | 0.8226 | 0.2667 |
| 0.55 | 0.8483 | 1.0000 | 0.8226 | 0.2667 |
| 0.60 | 0.8483 | 1.0000 | 0.8226 | 0.2667 |
| 0.65 | 0.8483 | 1.0000 | 0.8226 | 0.2667 |
| 0.70 | 0.8661 | 1.0000 | 0.8299 | 0.2667 |
| 0.75 | 0.8661 | 1.0000 | 0.8293 | 0.2667 |
| 0.80 | 0.8511 | 1.0000 | 0.8179 | 0.2667 |
| 0.85 | 0.8339 | 1.0000 | 0.8070 | 0.2667 |
| 0.90 | 0.7744 | 1.0000 | 0.7615 | 0.2540 |
| 0.95 | 0.7533 | 1.0000 | 0.7329 | 0.2413 |
| 1.00 | 0.7156 | 1.0000 | 0.6863 | 0.2187 |
| Vanilla (no polarity encoder, no tau filter) | 0.6617 | 1.0000 | 0.6533 | 0.1987 |

- Unique metric tuples: **12**
- Best by CCR@10: alpha=0.00, CCR@10=0.2667


<!-- 这份 `alpha_sweep_tau_0.2.md` 的数据来源是：

1. **主数据源 JSON**：`/home/xingkun/ant/outputs/reports/alpha_sweep_tau_0.2.json`  
2. **再格式化成 Markdown 表**（就是你看到的 1-33 行）

我核对过，表中每一行 `alpha` 对应 JSON 里的 `results` 数组（`recall@10 / recall@100 / ndcg@10 / ccr@10` 一一对应）。

---

### 具体怎么算出来的（口径）

固定配置：

- benchmark: `data/processed/retrieval_benchmark_v1.jsonl`
- topic model: `/data/xingkun/local_model/Llama-3.2-3B-Instruct`
- constraint model: `outputs/checkpoints/constraint-encoder-v1`
- `retrieve_k = 54`
- `tau = 0.2`
- `alpha = 0.00 ~ 1.00`（步长 0.05，共 21 个）

对每个 alpha：

- `final_score = alpha * topic_score + (1-alpha) * constraint_score`
- 只保留 `constraint_score >= tau` 的候选（若空则回退）
- 计算整体指标：`Recall@10`、`Recall@100`、`NDCG@10`、`CCR@10`

然后写入 JSON 的 `results`，再渲染成 md 表。

---

### 你表里最后两行注释是怎么来的

- `Unique metric tuples: 12`：  
  对 21 个 alpha 的四元组指标去重后，剩 12 种唯一组合。

- `Best by CCR@10: alpha=0.00, CCR@10=0.2667`：  
  直接取 JSON 里 `best_by_ccr@10`（同分时是第一个出现的 alpha）。

---

### “Vanilla (no polarity encoder, no tau filter)” 那行

这一行不是 tau=0.2 sweep 的某一行本体，而是额外附加的**纯 topic baseline**。  
数值 `0.6617 / 1.0000 / 0.6533 / 0.1987` 与这些来源一致：

- `alpha_sweep_tau_0.0.json` 中 `alpha=1.0` 那一行
- 以及 `retrieval_metrics_local_dual.json` 里的 `overall.vanilla`

也就是“无约束编码器影响、无阈值过滤”的基准线。 -->


<!-- 
cd /home/xingkun/ant && python3 - <<'PY'
import json
from pathlib import Path
import numpy as np

from experiments.common import load_sentence_encoder, read_jsonl
from experiments.metrics import ccr_at_k, ndcg_at_k, recall_at_k

root = Path("/home/xingkun/ant")
benchmark_file = root / "data/processed/retrieval_benchmark_v1.jsonl"
corpus_file = root / "data/processed/retrieval_corpus_v1.jsonl"
topic_model_name = "/data/xingkun/local_model/Llama-3.2-3B-Instruct"
constraint_model_name = str(root / "outputs/checkpoints/constraint-encoder-v1")
retrieve_k = 54
tau = 0.2
alphas = [round(i * 0.05, 2) for i in range(21)]

benchmark = read_jsonl(benchmark_file)
corpus = read_jsonl(corpus_file)
doc_ids = [d["doc_id"] for d in corpus]
doc_texts = [d["text"] for d in corpus]

topic_model = load_sentence_encoder(topic_model_name)
constraint_model = load_sentence_encoder(constraint_model_name)
topic_doc_emb = topic_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
constraint_doc_emb = constraint_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

def m(v): return float(np.mean(v)) if v else 0.0

rows = []
vanilla_vals = {"recall@10": [], "recall@100": [], "ndcg@10": [], "ccr@10": []}

for ai, alpha in enumerate(alphas, start=1):
    dual_vals = {"recall@10": [], "recall@100": [], "ndcg@10": [], "ccr@10": []}
    for item in benchmark:
        query = item["query"]
        topical_relevant = item["topical_relevant_doc_ids"]
        constraint_positive = item["constraint_satisfying_doc_ids"]
        graded_rel = {k: float(v) for k, v in item["graded_relevance"].items()}

        q_topic = topic_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_constraint = constraint_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        topic_scores = np.dot(topic_doc_emb, q_topic)
        constraint_scores = np.dot(constraint_doc_emb, q_constraint)

        vanilla_idx = np.argsort(-topic_scores)
        vanilla_rank = [doc_ids[i] for i in vanilla_idx]

        retrieve_idx = vanilla_idx[:retrieve_k]
        final_scores = alpha * topic_scores[retrieve_idx] + (1.0 - alpha) * constraint_scores[retrieve_idx]
        keep_local = [j for j, i in enumerate(retrieve_idx) if float(constraint_scores[i]) >= tau]

        if keep_local:
            kept_idx = retrieve_idx[keep_local]
            kept_scores = final_scores[keep_local]
            order = np.argsort(-kept_scores)
            dual_idx = kept_idx[order]
        else:
            dual_idx = retrieve_idx

        dual_rank = [doc_ids[i] for i in dual_idx] + [doc_ids[i] for i in vanilla_idx if i not in set(dual_idx)]

        if ai == 1:
            vanilla_vals["recall@10"].append(recall_at_k(vanilla_rank, topical_relevant, 10))
            vanilla_vals["recall@100"].append(recall_at_k(vanilla_rank, topical_relevant, 100))
            vanilla_vals["ndcg@10"].append(ndcg_at_k(vanilla_rank, graded_rel, 10))
            vanilla_vals["ccr@10"].append(ccr_at_k(vanilla_rank, constraint_positive, 10))

        dual_vals["recall@10"].append(recall_at_k(dual_rank, topical_relevant, 10))
        dual_vals["recall@100"].append(recall_at_k(dual_rank, topical_relevant, 100))
        dual_vals["ndcg@10"].append(ndcg_at_k(dual_rank, graded_rel, 10))
        dual_vals["ccr@10"].append(ccr_at_k(dual_rank, constraint_positive, 10))

    rows.append({
        "alpha": alpha, "tau": tau,
        "recall@10": m(dual_vals["recall@10"]),
        "recall@100": m(dual_vals["recall@100"]),
        "ndcg@10": m(dual_vals["ndcg@10"]),
        "ccr@10": m(dual_vals["ccr@10"]),
    })

payload = {
    "benchmark_file": str(benchmark_file),
    "topic_model": topic_model_name,
    "constraint_model": constraint_model_name,
    "retrieve_k": retrieve_k,
    "tau": tau,
    "alphas": alphas,
    "results": rows,
    "best_by_ccr@10": max(rows, key=lambda x: x["ccr@10"]),
    "best_by_ndcg@10": max(rows, key=lambda x: x["ndcg@10"]),
}
vanilla = {
    "recall@10": m(vanilla_vals["recall@10"]),
    "recall@100": m(vanilla_vals["recall@100"]),
    "ndcg@10": m(vanilla_vals["ndcg@10"]),
    "ccr@10": m(vanilla_vals["ccr@10"]),
}

json_out = root / "outputs/reports/alpha_sweep_tau_0.2.json"
json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

unique = len({(round(r["recall@10"],4), round(r["recall@100"],4), round(r["ndcg@10"],4), round(r["ccr@10"],4)) for r in rows})
best = payload["best_by_ccr@10"]

lines = [
    "# Alpha Sweep (tau=0.2)",
    "",
    f"- Benchmark: `{benchmark_file}`",
    f"- Topic model: `{topic_model_name}`",
    f"- Constraint model: `{constraint_model_name}`",
    "",
    "| alpha | Recall@10 | Recall@100 | NDCG@10 | CCR@10 |",
    "| ---: | ---: | ---: | ---: | ---: |",
]
for r in rows:
    lines.append(f"| {r['alpha']:.2f} | {r['recall@10']:.4f} | {r['recall@100']:.4f} | {r['ndcg@10']:.4f} | {r['ccr@10']:.4f} |")
lines += [
    f"| Vanilla (no polarity encoder, no tau filter) | {vanilla['recall@10']:.4f} | {vanilla['recall@100']:.4f} | {vanilla['ndcg@10']:.4f} | {vanilla['ccr@10']:.4f} |",
    "",
    f"- Unique metric tuples: **{unique}**",
    f"- Best by CCR@10: alpha={best['alpha']:.2f}, CCR@10={best['ccr@10']:.4f}",
]
md_out = root / "outputs/reports/alpha_sweep_tau_0.2.md"
md_out.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved: {json_out}")
print(f"Saved: {md_out}")
PY
 -->
