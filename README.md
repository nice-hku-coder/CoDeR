# CoDeR

Constraint-Aware Disentangled Retrieval for RAG.

## 项目简介

CoDeR 是本文论文的主代码库，目标是研究并实现一种 **Constraint-Aware / Dual-View Retrieval** 框架，用于缓解 RAG 在否定、排除、数值约束等场景下的检索违约束问题。

核心思路：

- `Topic Encoder` 负责主题召回
- `Constraint Encoder` 负责约束一致性判断
- 在线阶段采用 `Retrieve -> Filter/Rerank -> Generate` 流程

## 常用流程

### 1. 构建数据

```bash
python experiments/build_triplets.py
python experiments/build_constraint_benchmark.py \
  --output-file data/processed/constraint_benchmark_v1.jsonl \
  --num-negation 100 \
  --num-exclusion 100 \
  --num-numeric 100
python experiments/build_retrieval_benchmark.py
```

### 2. 训练 Constraint Encoder

```bash
python encoder_trainer/train_constraint_encoder.py
```

训练参数位于：

- `encoder_trainer/trainer_config.py`

### 3. 离线评估 Constraint Encoder

```bash
python experiments/eval_constraint_encoder.py
```

默认输出：

- `outputs/reports/eval_constraint_encoder/constraint_eval_report.json`

### 4. 检索评测与 baseline

```bash
python experiments/eval_retrieval_metrics.py \
  --report-file outputs/reports/retrieval_metrics_local_dual.json

python experiments/run_bm25_baseline.py
python experiments/run_cross_encoder_baseline.py
python experiments/rag_e2e_eval.py \
  --retrieval-report outputs/reports/retrieval_metrics_local_dual.json \
  --mode dual \
  --report-file outputs/reports/rag_e2e_eval/rag_e2e_proxy_dual.json
```
