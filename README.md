# CoDeR

Constraint-Aware Disentangled Retrieval for RAG.

## 项目简介

CoDeR 是本文论文的主代码库，目标是研究并实现一种 **Constraint-Aware / Dual-View Retrieval** 框架，用于缓解 RAG 在否定、排除、数值约束等场景下的检索违约束问题。


## 环境准备

### 1. 创建 conda 环境

```bash
conda create -n coder python=3.10 -y
conda activate coder
```

### 2. 安装项目依赖

```bash
cd CoDeR
pip install -r requirements.txt
```

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

### 5. Motivation 分析

默认脚本会使用本地 `models` 目录中的 BGE / Contriever 模型；如果要同时评估 HyDE，需要显式开启 `--enable-hyde` 并传入 GPT-5 API key。

```bash
python motivation/retrieval_failure_motivation.py \
  --enable-hyde \
  --hyde-api-key YOUR_API_KEY
```

默认输出：

- `outputs/reports/motivation/retrieval_failure_summary.json`
- `outputs/reports/motivation/hyde_generations.jsonl`（启用 HyDE 时）
- `outputs/figures/motivation/violation_rate_at_k.png`
- `outputs/figures/motivation/first_violating_rank_boxplot.png`
- `outputs/figures/motivation/violation_rate_at_5_by_category.png`
