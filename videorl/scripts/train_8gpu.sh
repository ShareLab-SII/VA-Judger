#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${WEIGHTS_ROOT:=${REPO_ROOT}/weights}"
: "${CONDA_SH:=}"
: "${QWEN_CONDA_ENV:=vllm}"
: "${LTX_CONDA_ENV:=ltx2}"
: "${MS_SWIFT_ROOT:=}"
: "${QWEN_GPU:=0}"
: "${LTX_GPUS:=1,2,3,4,5,6,7}"
: "${QWEN3OMNI_REWARD_PORT:=8112}"
: "${QWEN3OMNI_REWARD_SERVER:=127.0.0.1}"
: "${QWEN3OMNI_REWARD_MODEL:=${WEIGHTS_ROOT}/VA-Judger/VA-Judger}"
: "${LTX_MODEL_PATH:=${WEIGHTS_ROOT}/LTX-2/ltx-2-19b-dev.safetensors}"
: "${GEMMA_MODEL_PATH:=${WEIGHTS_ROOT}/LTX-2/gemma}"
: "${PAIR_VIDEO_PROMPT_TRAIN_DATASET:=${REPO_ROOT}/JavisBench-mini.csv}"
: "${PAIR_VIDEO_PROMPT_TEST_DATASET:=${REPO_ROOT}/dataset/vggsound/test_metadata_arena.jsonl}"
: "${OUTPUT_DIR:=outputs}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=6112}"
: "${LTX_DISTRIBUTED_TIMEOUT_MINUTES:=60}"
: "${LTX_DEBUG_TIMING:=1}"

IFS=',' read -r -a LTX_GPU_ARRAY <<< "${LTX_GPUS}"
LTX_NPROC="${#LTX_GPU_ARRAY[@]}"

if [[ "${LTX_NPROC}" -lt 1 ]]; then
  echo "ERROR: LTX_GPUS is empty" >&2
  exit 1
fi
if [[ ! -d "${QWEN3OMNI_REWARD_MODEL}" ]]; then
  echo "ERROR: QWEN3OMNI_REWARD_MODEL is not a directory: ${QWEN3OMNI_REWARD_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${LTX_MODEL_PATH}" ]]; then
  echo "ERROR: LTX_MODEL_PATH is not a file: ${LTX_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${GEMMA_MODEL_PATH}" ]]; then
  echo "ERROR: GEMMA_MODEL_PATH is not a directory: ${GEMMA_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PAIR_VIDEO_PROMPT_TRAIN_DATASET}" ]]; then
  echo "ERROR: PAIR_VIDEO_PROMPT_TRAIN_DATASET is not a file: ${PAIR_VIDEO_PROMPT_TRAIN_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${PAIR_VIDEO_PROMPT_TEST_DATASET}" ]]; then
  echo "ERROR: PAIR_VIDEO_PROMPT_TEST_DATASET is not a file: ${PAIR_VIDEO_PROMPT_TEST_DATASET}" >&2
  exit 1
fi

RUN_DIR="${OUTPUT_DIR}/qwen3omni_ltx2_av_quality_split_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_DIR}"
QWEN_LOG="${RUN_DIR}/qwen3omni_reward_server.log"
LTX_LOG="${RUN_DIR}/ltx2_grpo_train.log"
QWEN3OMNI_INFER_DUMP_PATH="${RUN_DIR}/qwen3omni_infer_dump.jsonl"
QWEN3OMNI_REWARD_RUN_ID="$(basename "${RUN_DIR}")-$$"

echo "Run dir: ${RUN_DIR}"
echo "Qwen3-Omni reward model: ${QWEN3OMNI_REWARD_MODEL}"
echo "Qwen3-Omni reward GPU: ${QWEN_GPU}, port: ${QWEN3OMNI_REWARD_PORT}"
echo "Qwen3-Omni prompt mode: dimension_scores"
echo "Reward weights: overall=${QWEN3OMNI_OVERALL_REWARD_WEIGHT:-0.3333333333333333}, audio=${QWEN3OMNI_AUDIO_REWARD_WEIGHT:-0.3333333333333333}, video=${QWEN3OMNI_VIDEO_REWARD_WEIGHT:-0.3333333333333333}"
echo "LTX GPUs: ${LTX_GPUS} (${LTX_NPROC} processes)"

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Validation passed; this node would run ${LTX_NPROC} LTX ranks and one vLLM reward server."
  exit 0
fi

if [[ -z "${CONDA_SH}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available; install Miniconda or set CONDA_SH." >&2
    exit 1
  fi
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"

if python - "${QWEN3OMNI_REWARD_SERVER}" "${QWEN3OMNI_REWARD_PORT}" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2).read()
except Exception:
    sys.exit(1)
sys.exit(0)
PY
then
  echo "ERROR: ${QWEN3OMNI_REWARD_SERVER}:${QWEN3OMNI_REWARD_PORT} already has an HTTP service." >&2
  exit 1
fi

(
  set -euo pipefail
  source "${CONDA_SH}"
  conda activate "${QWEN_CONDA_ENV}"
  export CUDA_VISIBLE_DEVICES="${QWEN_GPU}"
  export QWEN3OMNI_REWARD_MODEL
  export QWEN3OMNI_REWARD_PORT
  export QWEN3OMNI_REWARD_SERVER
  export QWEN3OMNI_REWARD_RUN_ID
  export QWEN3OMNI_REWARD_PROMPT_MODE=dimension_scores
  export QWEN3OMNI_DUMP_INFER="${QWEN3OMNI_DUMP_INFER:-1}"
  export QWEN3OMNI_INFER_DUMP_PATH
  export MS_SWIFT_ROOT
  if [[ -n "${MS_SWIFT_ROOT}" ]]; then
    export PYTHONPATH="${REPO_ROOT}:${MS_SWIFT_ROOT}:${PYTHONPATH:-}"
  else
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  fi
  export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-true}"
  export ENABLE_AUDIO_OUTPUT="${ENABLE_AUDIO_OUTPUT:-0}"
  export MAX_PIXELS="${MAX_PIXELS:-1003520}"
  export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-602112}"
  export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"
  export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
  export QWEN3OMNI_MAX_NEW_TOKENS="${QWEN3OMNI_MAX_NEW_TOKENS:-2048}"
  export QWEN3OMNI_TEMPERATURE="${QWEN3OMNI_TEMPERATURE:-0.0}"
  export QWEN3OMNI_TOP_P="${QWEN3OMNI_TOP_P:-1.0}"
  bash flow_grpo/server/run_remote_qwen3omni_reward.sh
) > "${QWEN_LOG}" 2>&1 &
QWEN_PID=$!
echo "${QWEN_PID}" > "${RUN_DIR}/qwen3omni_reward_server.pid"
echo "Started Qwen3-Omni reward server PID=${QWEN_PID}; log=${QWEN_LOG}"

cleanup() {
  if kill -0 "${QWEN_PID}" >/dev/null 2>&1; then
    kill "${QWEN_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Waiting for reward server health..."
SERVER_READY=0
for _ in $(seq 1 180); do
  if python - "${QWEN3OMNI_REWARD_SERVER}" "${QWEN3OMNI_REWARD_PORT}" "${QWEN3OMNI_REWARD_RUN_ID}" <<'PY'
import json
import sys
import urllib.request

host, port, expected_run_id = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
        data = json.loads(response.read().decode("utf-8"))
    sys.exit(0 if data.get("status") == "healthy" and data.get("run_id") == expected_run_id else 1)
except Exception:
    sys.exit(1)
PY
  then
    SERVER_READY=1
    break
  fi
  if ! kill -0 "${QWEN_PID}" >/dev/null 2>&1; then
    echo "ERROR: reward server exited. See ${QWEN_LOG}" >&2
    exit 1
  fi
  sleep 5
done
if [[ "${SERVER_READY}" != "1" ]]; then
  echo "ERROR: reward server did not become healthy. See ${QWEN_LOG}" >&2
  exit 1
fi

conda activate "${LTX_CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export LTX_DISTRIBUTED_TIMEOUT_MINUTES
export LTX_DEBUG_TIMING
export LTX_MODEL_PATH
export GEMMA_MODEL_PATH
export PAIR_VIDEO_PROMPT_TRAIN_DATASET
export PAIR_VIDEO_PROMPT_TEST_DATASET
export QWEN3OMNI_REWARD_SERVER
export QWEN3OMNI_REWARD_PORT
export QWEN3OMNI_REWARD_RUN_ID
export OUTPUT_DIR

CUDA_VISIBLE_DEVICES="${LTX_GPUS}" torchrun \
  --nnodes 1 \
  --nproc_per_node "${LTX_NPROC}" \
  --node_rank 0 \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py --config config/nft.py:ltx2_qwen3omni_av_quality_split_reward \
  2>&1 | tee "${LTX_LOG}"
