#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for:
# CoDeR/data/build_beir_llm_rewrite_dataset.py
# Now includes automatic index building when missing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${CODER_ROOT}/.." && pwd)"

BEIR_ROOT="${BEIR_ROOT:-${WORKSPACE_ROOT}/beir}"
INDEX_ROOT="${INDEX_ROOT:-${CODER_ROOT}/data/indexes/pyserini}"
TMP_INPUT_ROOT="${TMP_INPUT_ROOT:-${CODER_ROOT}/data/tmp/pyserini_input}"

BASE_URL="${BASE_URL:-https://api.vectorengine.ai/v1}"
MODEL="${MODEL:-gpt-5.4}"
QRELS_SPLIT="${QRELS_SPLIT:-test}"
QUERIES_PER_DATASET="${QUERIES_PER_DATASET:-200}"
TOP_K="${TOP_K:-12}"
FINAL_DOCS="${FINAL_DOCS:-5}"
QRELS_PRIORITY_COUNT="${QRELS_PRIORITY_COUNT:-2}"
MIN_QUERY_WORDS="${MIN_QUERY_WORDS:-4}"
MAX_QUERY_WORDS="${MAX_QUERY_WORDS:-20}"
SEED="${SEED:-42}"
INDEX_THREADS="${INDEX_THREADS:-8}"
FORCE_REBUILD_INDEX="${FORCE_REBUILD_INDEX:-0}"

# API key source priority:
# 1) first positional arg
# 2) env VECTORENGINE_API_KEY
API_KEY="${1:-${VECTORENGINE_API_KEY:-}}"

format_duration() {
  local total_sec="${1:-0}"
  if [[ "${total_sec}" -lt 0 ]]; then
    total_sec=0
  fi
  local h=$((total_sec / 3600))
  local m=$(((total_sec % 3600) / 60))
  local s=$((total_sec % 60))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${s}"
}

progress_bar() {
  local done="${1:-0}"
  local total="${2:-1}"
  local label="${3:-stage}"
  local start_ts="${4:-0}"
  local now elapsed eta width filled empty

  now="$(date +%s)"
  elapsed=$((now - start_ts))
  width=26
  if [[ "${total}" -le 0 ]]; then
    total=1
  fi

  filled=$((done * width / total))
  if [[ "${filled}" -gt "${width}" ]]; then
    filled="${width}"
  fi
  empty=$((width - filled))

  if [[ "${done}" -gt 0 && "${done}" -lt "${total}" ]]; then
    eta=$((elapsed * (total - done) / done))
  elif [[ "${done}" -ge "${total}" ]]; then
    eta=0
  else
    eta=-1
  fi

  local bar_fill bar_empty eta_text
  bar_fill="$(printf '%*s' "${filled}" '' | tr ' ' '#')"
  bar_empty="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  if [[ "${eta}" -lt 0 ]]; then
    eta_text="--:--:--"
  else
    eta_text="$(format_duration "${eta}")"
  fi

  printf "\r[PROGRESS] %-8s [%s%s] %3d/%-3d elapsed=%s eta=%s" \
    "${label}" "${bar_fill}" "${bar_empty}" "${done}" "${total}" \
    "$(format_duration "${elapsed}")" "${eta_text}"

  if [[ "${done}" -ge "${total}" ]]; then
    printf "\n"
  fi
}

if [[ -n "${CODER_PYTHON:-}" ]]; then
  PYTHON_BIN="${CODER_PYTHON}"
elif [[ -x "/d/Anaconda3/envs/coder/python.exe" ]]; then
  PYTHON_BIN="/d/Anaconda3/envs/coder/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[ERROR] Python not found. Set CODER_PYTHON explicitly." >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  echo "[ERROR] Missing API key." >&2
  echo "Usage: bash scripts/run_beir_llm_rewrite_dataset.sh <API_KEY>" >&2
  echo "Or export VECTORENGINE_API_KEY first." >&2
  exit 1
fi

datasets=(fiqa dbpedia-entity scifact)
pipeline_start_ts="$(date +%s)"
pipeline_total_stages=2
pipeline_done=0

"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["pyserini", "openai"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("[ERROR] Missing Python deps: " + ", ".join(missing))
print("[INFO] Python deps check passed")
PY

if ! command -v java >/dev/null 2>&1; then
  echo "[ERROR] Java not found in PATH. Pyserini indexing requires Java." >&2
  exit 1
fi

echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] BEIR_ROOT: ${BEIR_ROOT}"
echo "[INFO] INDEX_ROOT: ${INDEX_ROOT}"
echo "[INFO] TMP_INPUT_ROOT: ${TMP_INPUT_ROOT}"
echo "[INFO] MODEL: ${MODEL}"
echo "[INFO] BASE_URL: ${BASE_URL}"
echo "[INFO] QUERIES_PER_DATASET: ${QUERIES_PER_DATASET}"
echo "[INFO] TOP_K: ${TOP_K}, FINAL_DOCS: ${FINAL_DOCS}"
echo "[INFO] FORCE_REBUILD_INDEX: ${FORCE_REBUILD_INDEX}, INDEX_THREADS: ${INDEX_THREADS}"

progress_bar "${pipeline_done}" "${pipeline_total_stages}" "pipeline" "${pipeline_start_ts}"

export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${INDEX_ROOT}" "${TMP_INPUT_ROOT}"

index_total="${#datasets[@]}"
index_done=0
index_start_ts="$(date +%s)"
progress_bar "${index_done}" "${index_total}" "index" "${index_start_ts}"

for ds in "${datasets[@]}"; do
  src_corpus="${BEIR_ROOT}/${ds}/corpus.jsonl"
  prep_dir="${TMP_INPUT_ROOT}/${ds}"
  prep_file="${prep_dir}/docs.jsonl"
  out_index="${INDEX_ROOT}/${ds}"

  if [[ ! -f "${src_corpus}" ]]; then
    echo "[ERROR] Missing corpus file: ${src_corpus}" >&2
    exit 1
  fi

  if [[ "${FORCE_REBUILD_INDEX}" != "1" ]] && [[ -d "${out_index}" ]] && compgen -G "${out_index}/segments*" >/dev/null; then
    echo "[INFO] Reusing existing index for ${ds}: ${out_index}"
    index_done=$((index_done + 1))
    progress_bar "${index_done}" "${index_total}" "index" "${index_start_ts}"
    continue
  fi

  mkdir -p "${prep_dir}"

  echo "[INFO] Preparing JsonCollection for ${ds}"
  "${PYTHON_BIN}" - "${src_corpus}" "${prep_file}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)

cnt = 0
with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        did = str(obj.get("_id", "")).strip()
        if not did:
            continue
        title = str(obj.get("title", "")).strip()
        text = str(obj.get("text", "")).strip()
        contents = (title + "\n" + text).strip() if title else text
        rec = {"id": did, "contents": contents}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cnt += 1
print(f"[INFO] Prepared {cnt} docs -> {out}")
PY

  echo "[INFO] Building index for ${ds}"
  "${PYTHON_BIN}" -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${prep_dir}" \
    --index "${out_index}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${INDEX_THREADS}" \
    --storePositions --storeDocvectors --storeRaw

  index_done=$((index_done + 1))
  progress_bar "${index_done}" "${index_total}" "index" "${index_start_ts}"
done

pipeline_done=$((pipeline_done + 1))
progress_bar "${pipeline_done}" "${pipeline_total_stages}" "pipeline" "${pipeline_start_ts}"

echo "[INFO] Starting dataset generation stage (this may take a while due to LLM calls)..."

"${PYTHON_BIN}" "${CODER_ROOT}/data/build_beir_llm_rewrite_dataset.py" \
  --beir-root "${BEIR_ROOT}" \
  --index-root "${INDEX_ROOT}" \
  --qrels-split "${QRELS_SPLIT}" \
  --queries-per-dataset "${QUERIES_PER_DATASET}" \
  --min-query-words "${MIN_QUERY_WORDS}" \
  --max-query-words "${MAX_QUERY_WORDS}" \
  --top-k "${TOP_K}" \
  --final-docs "${FINAL_DOCS}" \
  --qrels-priority-count "${QRELS_PRIORITY_COUNT}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --api-key "${API_KEY}"

pipeline_done=$((pipeline_done + 1))
progress_bar "${pipeline_done}" "${pipeline_total_stages}" "pipeline" "${pipeline_start_ts}"

echo "[DONE] LLM rewrite dataset pipeline finished in $(format_duration $(( $(date +%s) - pipeline_start_ts )))"
