#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to a full SFT checkpoint directory}"
: "${BENCH_ROOT:=${REPO_ROOT}/data/VA-Judger-Bench}"
: "${OUTPUT_ROOT:=${REPO_ROOT}/outputs/evaluation/$(basename "${MODEL_PATH}")}"
: "${CONDA_ENV:=vllm}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${BATCH_SIZE:=16}"
: "${MAX_NEW_TOKENS:=2048}"
: "${VLLM_MAX_MODEL_LEN:=24576}"
: "${VLLM_MAX_NUM_SEQS:=16}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.9}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not a directory: ${MODEL_PATH}" >&2
  exit 1
fi
for split in easy indomain outdomain; do
  if [[ ! -f "${BENCH_ROOT}/${split}/data.jsonl" ]]; then
    echo "ERROR: BENCH_ROOT is missing ${split}/data.jsonl: ${BENCH_ROOT}" >&2
    exit 1
  fi
done

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  for split in easy indomain outdomain; do
    test -f "${BENCH_ROOT}/${split}/data.jsonl"
    echo "${split}: $(wc -l < "${BENCH_ROOT}/${split}/data.jsonl") cases"
  done
  echo "Validation passed; evaluation would use CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

if ! python -c 'from swift import InferStats, RequestConfig; from swift.infer_engine import InferRequest, VllmEngine' \
    >/dev/null 2>&1; then
  echo "ERROR: the installed ms-swift package does not provide the required inference APIs." >&2
  echo "Activate the evaluation environment and run: pip install -r requirements-eval.txt" >&2
  exit 1
fi

export USE_AUDIO_IN_VIDEO=true
export ENABLE_AUDIO_OUTPUT=0
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-602112}"

IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE="${#DEVICES[@]}"
mkdir -p "${OUTPUT_ROOT}"

for split in easy indomain outdomain; do
  dataset="${BENCH_ROOT}/${split}/data.jsonl"
  split_root="${BENCH_ROOT}/${split}"
  split_output="${OUTPUT_ROOT}/${split}"
  mkdir -p "${split_output}"
  pids=()
  for ((worker = 0; worker < WORLD_SIZE; worker++)); do
    CUDA_VISIBLE_DEVICES="${DEVICES[$worker]}" \
      python scripts/evaluate.py \
        --model "${MODEL_PATH}" \
        --dataset "${dataset}" \
        --split-root "${split_root}" \
        --results "${split_output}/results.worker${worker}.jsonl" \
        --summary "${split_output}/summary.worker${worker}.json" \
        --worker "${worker}" \
        --world-size "${WORLD_SIZE}" \
        --batch-size "${BATCH_SIZE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --resume \
        > "${split_output}/worker${worker}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if (( failed )); then
    echo "ERROR: ${split} evaluation failed; inspect ${split_output}/worker*.log" >&2
    exit 1
  fi
  : > "${split_output}/results.jsonl"
  for ((worker = 0; worker < WORLD_SIZE; worker++)); do
    cat "${split_output}/results.worker${worker}.jsonl" >> "${split_output}/results.jsonl"
  done
  python scripts/evaluate.py \
    --results "${split_output}/results.jsonl" \
    --summary "${split_output}/summary.json" \
    --eval-only
done

python - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = {name: json.loads((root / name / "summary.json").read_text()) for name in ("easy", "indomain", "outdomain")}
total = sum(item["total"] for item in splits.values())
parsed = sum(item["parsed"] for item in splits.values())
correct = sum(item["correct"] for item in splits.values())
summary = {
    "splits": splits,
    "overall": {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "all_cases_accuracy": correct / total if total else None,
        "parsed_only_accuracy": correct / parsed if parsed else None,
        "parse_rate": parsed / total if total else None,
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
