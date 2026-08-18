#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${WEIGHTS_ROOT:=${REPO_ROOT}/weights}"
: "${CONDA_SH:=}"
: "${QWEN_CONDA_ENV:=vllm}"
: "${LTX_CONDA_ENV:=ltx2}"
: "${MS_SWIFT_ROOT:=}"

: "${PET_NNODES:?PET_NNODES is required}"
: "${PET_NODE_RANK:?PET_NODE_RANK is required}"
: "${PET_NPROC_PER_NODE:?PET_NPROC_PER_NODE is required}"
: "${TRAIN_JOB_ID:?TRAIN_JOB_ID is required to namespace this launch}"
: "${RUNNING_ROUND:?RUNNING_ROUND is required to namespace this launch}"

EXPECTED_NNODES=4
EXPECTED_GPUS_PER_NODE=8
TRAIN_NNODES=3
REWARD_NODE_RANK=3

if [[ "${PET_NNODES}" -ne "${EXPECTED_NNODES}" ]]; then
  echo "ERROR: expected PET_NNODES=${EXPECTED_NNODES}, got ${PET_NNODES}" >&2
  exit 1
fi
if [[ "${PET_NPROC_PER_NODE}" -ne "${EXPECTED_GPUS_PER_NODE}" ]]; then
  echo "ERROR: expected PET_NPROC_PER_NODE=${EXPECTED_GPUS_PER_NODE}, got ${PET_NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ "${PET_NODE_RANK}" -lt 0 || "${PET_NODE_RANK}" -ge "${EXPECTED_NNODES}" ]]; then
  echo "ERROR: PET_NODE_RANK must be in [0, 3], got ${PET_NODE_RANK}" >&2
  exit 1
fi

sanitize() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9_.-' '_'
}

JOB_ID="$(sanitize "${TRAIN_JOB_ID}")"
ROUND_ID="$(sanitize "${RUNNING_ROUND}")"
RENDEZVOUS_ID="$(sanitize "${MASTER_PORT:-${PET_MASTER_PORT:-port0}}")"
LAUNCH_GENERATION="$(sanitize "${PET_LAUNCH_GENERATION:-${JOB_ID}_${ROUND_ID}_${RENDEZVOUS_ID}}")"
: "${OUTPUT_ROOT:=${REPO_ROOT}/outputs}"
RUN_DIR="${OUTPUT_ROOT}/qwen3omni_avsplit_4node_${LAUNCH_GENERATION}"
LOG_DIR="${RUN_DIR}/logs"
COORD_DIR="${RUN_DIR}/coord"
mkdir -p "${LOG_DIR}" "${COORD_DIR}"

HOST_NAME="$(hostname -f 2>/dev/null || hostname)"
printf '%s\n' "${HOST_NAME}" > "${COORD_DIR}/node_${PET_NODE_RANK}.host.tmp"
mv "${COORD_DIR}/node_${PET_NODE_RANK}.host.tmp" "${COORD_DIR}/node_${PET_NODE_RANK}.host"

: "${QWEN3OMNI_REWARD_MODEL:=${WEIGHTS_ROOT}/VA-Judger/VA-Judger}"
: "${LTX_MODEL_PATH:=${WEIGHTS_ROOT}/LTX-2/ltx-2-19b-dev.safetensors}"
: "${GEMMA_MODEL_PATH:=${WEIGHTS_ROOT}/LTX-2/gemma}"
: "${PAIR_VIDEO_PROMPT_TRAIN_DATASET:=${REPO_ROOT}/JavisBench-mini.csv}"
: "${PAIR_VIDEO_PROMPT_TEST_DATASET:=${REPO_ROOT}/dataset/vggsound/test_metadata_arena.jsonl}"
: "${REWARD_PORT_BASE:=8112}"
: "${REWARD_SERVER_COUNT:=4}"
: "${REWARD_TP_SIZE:=2}"
: "${REWARD_READY_TIMEOUT:=1800}"

atomic_marker() {
  local path="$1"
  local content="${2:-ok}"
  local temp_path="${path}.tmp.${PET_NODE_RANK}.$$"
  printf '%s\n' "${content}" > "${temp_path}"
  mv "${temp_path}" "${path}"
}

wait_for_file() {
  local path="$1"
  local timeout="$2"
  local failure_path="${3:-}"
  local waited=0
  while [[ ! -s "${path}" ]]; do
    if [[ -n "${failure_path}" && -e "${failure_path}" ]]; then
      echo "ERROR: peer node reported failure: $(<"${failure_path}")" >&2
      return 1
    fi
    if [[ "${waited}" -ge "${timeout}" ]]; then
      echo "ERROR: timed out waiting for ${path}" >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

TRAINING_FAILED_MARKER="${COORD_DIR}/training_failed_${LAUNCH_GENERATION}"
REWARD_FAILED_MARKER="${COORD_DIR}/reward_failed_${LAUNCH_GENERATION}"
PREFLIGHT_SUCCEEDED=0
report_preflight_exit() {
  local status="$?"
  if [[ "${PREFLIGHT_SUCCEEDED}" != "1" && "${status}" -ne 0 ]]; then
    if [[ "${PET_NODE_RANK}" -lt "${TRAIN_NNODES}" ]]; then
      atomic_marker "${TRAINING_FAILED_MARKER}" "preflight node=${PET_NODE_RANK}, exit=${status}"
    else
      atomic_marker "${REWARD_FAILED_MARKER}" "reward preflight node=${PET_NODE_RANK}, exit=${status}"
    fi
  fi
}

NODE_CLAIM="${COORD_DIR}/launch_${LAUNCH_GENERATION}_rank${PET_NODE_RANK}.claim"
if ! (set -o noclobber; printf '%s\n' "${HOST_NAME}:$$" > "${NODE_CLAIM}") 2>/dev/null; then
  echo "ERROR: launch generation ${LAUNCH_GENERATION}, PET rank ${PET_NODE_RANK} was already claimed." >&2
  echo "Set PET_LAUNCH_GENERATION to a new value before retrying this exact platform run." >&2
  exit 1
fi
trap report_preflight_exit EXIT

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_MEMORY_MIB < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  if [[ "${#GPU_MEMORY_MIB[@]}" -lt "${EXPECTED_GPUS_PER_NODE}" ]]; then
    echo "ERROR: node has ${#GPU_MEMORY_MIB[@]} visible GPUs; expected at least ${EXPECTED_GPUS_PER_NODE}" >&2
    exit 1
  fi
  if [[ "${SKIP_GPU_MEMORY_CHECK:-0}" != "1" ]]; then
    for index in $(seq 0 $((EXPECTED_GPUS_PER_NODE - 1))); do
      memory="${GPU_MEMORY_MIB[$index]//[[:space:]]/}"
      if [[ "${memory}" -lt 45000 ]]; then
        echo "ERROR: GPU ${index} reports ${memory} MiB, below the required 45000 MiB." >&2
        echo "Set SKIP_GPU_MEMORY_CHECK=1 only if this check is known to be wrong." >&2
        exit 1
      fi
    done
  fi
fi

if [[ ! -d "${QWEN3OMNI_REWARD_MODEL}" ]]; then
  echo "ERROR: reward model directory not found: ${QWEN3OMNI_REWARD_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${LTX_MODEL_PATH}" ]]; then
  echo "ERROR: LTX model not found: ${LTX_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${GEMMA_MODEL_PATH}" ]]; then
  echo "ERROR: Gemma directory not found: ${GEMMA_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PAIR_VIDEO_PROMPT_TRAIN_DATASET}" || ! -f "${PAIR_VIDEO_PROMPT_TEST_DATASET}" ]]; then
  echo "ERROR: train/test dataset path is missing." >&2
  exit 1
fi
if [[ $((REWARD_SERVER_COUNT * REWARD_TP_SIZE)) -ne "${EXPECTED_GPUS_PER_NODE}" ]]; then
  echo "ERROR: REWARD_SERVER_COUNT * REWARD_TP_SIZE must equal ${EXPECTED_GPUS_PER_NODE}" >&2
  exit 1
fi
PREFLIGHT_SUCCEEDED=1
trap - EXIT

reward_node_main() {
  local advertised_host="${REWARD_HOST:-${HOST_NAME}}"
  local reward_generation="${LAUNCH_GENERATION}"
  local reward_succeeded=0
  atomic_marker "${COORD_DIR}/reward_host" "${advertised_host}"
  atomic_marker "${COORD_DIR}/reward_generation" "${reward_generation}"

  local -a server_pids=()
  local -a server_logs=()
  local -a reward_urls=()
  local tail_pid=""

  cleanup_reward() {
    local status="$?"
    if [[ -n "${tail_pid}" ]]; then
      kill "${tail_pid}" >/dev/null 2>&1 || true
    fi
    for pid in "${server_pids[@]:-}"; do
      kill -- "-${pid}" >/dev/null 2>&1 || kill "${pid}" >/dev/null 2>&1 || true
    done
    if [[ "${reward_succeeded}" != "1" && "${status}" -ne 0 ]]; then
      atomic_marker "${REWARD_FAILED_MARKER}" "reward runtime exit=${status}"
    fi
  }
  trap cleanup_reward EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  echo "Role: reward node (PET_NODE_RANK=${PET_NODE_RANK}, host=${advertised_host})"
  echo "Starting ${REWARD_SERVER_COUNT} reward servers with TP=${REWARD_TP_SIZE}"

  for server_index in $(seq 0 $((REWARD_SERVER_COUNT - 1))); do
    local first_gpu=$((server_index * REWARD_TP_SIZE))
    local second_gpu=$((first_gpu + 1))
    local port=$((REWARD_PORT_BASE + server_index))
    local run_id="${reward_generation}-reward${server_index}"
    local log_path="${LOG_DIR}/reward_server_${server_index}_gpu${first_gpu}-${second_gpu}_port${port}.log"
    local dump_path="${RUN_DIR}/qwen3omni_infer_server_${server_index}.jsonl"
    server_logs+=("${log_path}")
    reward_urls+=("http://${advertised_host}:${port}")

    setsid env \
      CUDA_VISIBLE_DEVICES="${first_gpu},${second_gpu}" \
      CONDA_ENV="${QWEN_CONDA_ENV}" \
      CONDA_SH="${CONDA_SH}" \
      MS_SWIFT_ROOT="${MS_SWIFT_ROOT}" \
      QWEN3OMNI_REWARD_MODEL="${QWEN3OMNI_REWARD_MODEL}" \
      QWEN3OMNI_REWARD_PORT="${port}" \
      QWEN3OMNI_REWARD_RUN_ID="${run_id}" \
      QWEN3OMNI_REWARD_PROMPT_MODE="dimension_scores" \
      QWEN3OMNI_INFER_DUMP_PATH="${dump_path}" \
      QWEN3OMNI_DUMP_INFER="${QWEN3OMNI_DUMP_INFER:-1}" \
      QWEN3OMNI_MAX_NEW_TOKENS="${QWEN3OMNI_MAX_NEW_TOKENS:-2048}" \
      QWEN3OMNI_TEMPERATURE="${QWEN3OMNI_TEMPERATURE:-0.0}" \
      QWEN3OMNI_TOP_P="${QWEN3OMNI_TOP_P:-1.0}" \
      VLLM_TENSOR_PARALLEL_SIZE="${REWARD_TP_SIZE}" \
      VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}" \
      VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}" \
      VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}" \
      bash flow_grpo/server/run_remote_qwen3omni_reward.sh \
      > "${log_path}" 2>&1 &
    server_pids+=("$!")
  done

  tail -n +1 -F "${server_logs[@]}" &
  tail_pid="$!"

  local ready_count=0
  local waited=0
  while [[ "${ready_count}" -lt "${REWARD_SERVER_COUNT}" ]]; do
    ready_count=0
    for server_index in $(seq 0 $((REWARD_SERVER_COUNT - 1))); do
      local port=$((REWARD_PORT_BASE + server_index))
      local expected_run_id="${reward_generation}-reward${server_index}"
      if python - "${advertised_host}" "${port}" "${expected_run_id}" <<'PY'
import json
import sys
import urllib.request

host, port, expected_run_id = sys.argv[1:]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ok = payload.get("status") == "healthy" and payload.get("run_id") == expected_run_id
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
      then
        ready_count=$((ready_count + 1))
      fi
    done
    if [[ "${ready_count}" -eq "${REWARD_SERVER_COUNT}" ]]; then
      break
    fi
    for pid in "${server_pids[@]}"; do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        echo "ERROR: reward server process ${pid} exited; inspect ${LOG_DIR}/reward_server_*.log" >&2
        exit 1
      fi
    done
    if [[ "${waited}" -ge "${REWARD_READY_TIMEOUT}" ]]; then
      echo "ERROR: reward servers were not ready after ${REWARD_READY_TIMEOUT}s" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done

  local urls_csv
  urls_csv="$(IFS=,; echo "${reward_urls[*]}")"
  atomic_marker "${COORD_DIR}/reward_urls" "${urls_csv}"
  atomic_marker "${COORD_DIR}/reward_ready" "${reward_generation}"
  echo "All reward servers are ready: ${urls_csv}"

  while true; do
    if [[ -e "${TRAINING_FAILED_MARKER}" ]]; then
      echo "A training node reported failure." >&2
      exit 1
    fi
    local done_count
    done_count="$(compgen -G "${COORD_DIR}/train_node_*_${LAUNCH_GENERATION}.done" | wc -l || true)"
    if [[ "${done_count}" -ge "${TRAIN_NNODES}" ]]; then
      echo "All training nodes completed successfully."
      reward_succeeded=1
      return 0
    fi
    for pid in "${server_pids[@]}"; do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        echo "ERROR: reward server process ${pid} exited during training." >&2
        exit 1
      fi
    done
    sleep 10
  done
}

training_node_main() {
  local train_node_rank="${PET_NODE_RANK}"
  local training_succeeded=0
  report_training_exit() {
    local status="$?"
    if [[ "${training_succeeded}" != "1" && "${status}" -ne 0 ]]; then
      atomic_marker "${COORD_DIR}/train_node_${train_node_rank}_${LAUNCH_GENERATION}.failed" "exit=${status}"
      atomic_marker "${TRAINING_FAILED_MARKER}" "node=${train_node_rank}, exit=${status}"
    fi
  }
  trap report_training_exit EXIT

  if [[ "${train_node_rank}" -ge "${TRAIN_NNODES}" ]]; then
    echo "ERROR: invalid training node rank ${train_node_rank}" >&2
    exit 1
  fi

  echo "Role: training node ${train_node_rank}/${TRAIN_NNODES} (host=${HOST_NAME})"
  wait_for_file "${COORD_DIR}/reward_generation" "${REWARD_READY_TIMEOUT}" "${REWARD_FAILED_MARKER}"
  wait_for_file "${COORD_DIR}/reward_urls" "${REWARD_READY_TIMEOUT}" "${REWARD_FAILED_MARKER}"
  wait_for_file "${COORD_DIR}/reward_ready" "${REWARD_READY_TIMEOUT}" "${REWARD_FAILED_MARKER}"
  local reward_generation
  reward_generation="$(<"${COORD_DIR}/reward_generation")"
  local reward_ready_generation
  reward_ready_generation="$(<"${COORD_DIR}/reward_ready")"
  if [[ "${reward_generation}" != "${reward_ready_generation}" ]]; then
    echo "ERROR: reward coordination generation mismatch." >&2
    return 1
  fi
  export QWEN3OMNI_REWARD_URLS
  QWEN3OMNI_REWARD_URLS="$(<"${COORD_DIR}/reward_urls")"
  export QWEN3OMNI_REWARD_RUN_ID="${reward_generation}"

  python - "${QWEN3OMNI_REWARD_URLS}" "${reward_generation}" <<'PY'
import json
import sys
import urllib.request

urls = sys.argv[1].split(",")
generation = sys.argv[2]
for index, base_url in enumerate(urls):
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    expected_run_id = f"{generation}-reward{index}"
    if payload.get("status") != "healthy" or payload.get("run_id") != expected_run_id:
        raise RuntimeError(f"unhealthy reward server: {base_url}: {payload}")
print(f"Verified reward servers: {sys.argv[1]}")
PY

  export LTX_MODEL_PATH GEMMA_MODEL_PATH
  export PAIR_VIDEO_PROMPT_TRAIN_DATASET PAIR_VIDEO_PROMPT_TEST_DATASET
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  export TOKENIZERS_PARALLELISM=false
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export LTX_DISTRIBUTED_TIMEOUT_MINUTES="${LTX_DISTRIBUTED_TIMEOUT_MINUTES:-120}"
  export LTX_DEBUG_TIMING="${LTX_DEBUG_TIMING:-1}"

  export LTX_FSDP_SHARDING_STRATEGY="${LTX_FSDP_SHARDING_STRATEGY:-FULL_SHARD}"
  export LTX_FSDP_BACKWARD_PREFETCH="${LTX_FSDP_BACKWARD_PREFETCH:-NONE}"
  export LTX_FSDP_NUM_REPLICATE=1
  export LTX_FSDP_NUM_SHARD="${LTX_FSDP_NUM_SHARD:-$((TRAIN_NNODES * EXPECTED_GPUS_PER_NODE))}"
  export LTX_FSDP_USE_DEVICE_MESH=0
  export LTX_FSDP_CPU_OFFLOAD="${LTX_FSDP_CPU_OFFLOAD:-0}"
  export LTX_TRAIN_SAMPLE_CPU_OFFLOAD="${LTX_TRAIN_SAMPLE_CPU_OFFLOAD:-1}"

  export QWEN3OMNI_PAIR_GROUP_SIZE="${QWEN3OMNI_PAIR_GROUP_SIZE:-8}"
  export QWEN3OMNI_PROMPT_RANK_GROUP_SIZE="${QWEN3OMNI_PROMPT_RANK_GROUP_SIZE:-4}"
  export QWEN3OMNI_TRAIN_MICRO_BSZ="${QWEN3OMNI_TRAIN_MICRO_BSZ:-1}"
  export QWEN3OMNI_NUM_BATCHES_PER_EPOCH="${QWEN3OMNI_NUM_BATCHES_PER_EPOCH:-1}"
  export QWEN3OMNI_GRAD_ACCUM="${QWEN3OMNI_GRAD_ACCUM:-1}"
  export QWEN3OMNI_REWARD_PAIR_BATCH="${QWEN3OMNI_REWARD_PAIR_BATCH:-8}"
  export OUTPUT_DIR="${RUN_DIR}/training_outputs"

  local train_master_addr="${TRAIN_MASTER_ADDR:-${MASTER_ADDR:-${PET_MASTER_ADDR:-}}}"
  local train_master_port="${TRAIN_MASTER_PORT:-${MASTER_PORT:-${PET_MASTER_PORT:-}}}"
  if [[ -z "${train_master_addr}" || -z "${train_master_port}" ]]; then
    echo "ERROR: MASTER_ADDR/MASTER_PORT (or PET equivalents) are required." >&2
    atomic_marker "${TRAINING_FAILED_MARKER}" "missing rendezvous address"
    exit 1
  fi

  local log_path="${LOG_DIR}/train_node_${train_node_rank}.log"
  echo "Training rendezvous: ${train_master_addr}:${train_master_port}"
  echo "Reward URLs: ${QWEN3OMNI_REWARD_URLS}"
  echo "FSDP: ${LTX_FSDP_SHARDING_STRATEGY} across $((TRAIN_NNODES * EXPECTED_GPUS_PER_NODE)) training ranks, backward_prefetch=${LTX_FSDP_BACKWARD_PREFETCH}"
  echo "Memory: synchronized FSDP gradient accumulation, sample_cpu_offload=${LTX_TRAIN_SAMPLE_CPU_OFFLOAD}, parameter_cpu_offload=${LTX_FSDP_CPU_OFFLOAD}"
  echo "Rollout grouping: ${QWEN3OMNI_PROMPT_RANK_GROUP_SIZE} ranks/prompt, $((QWEN3OMNI_PAIR_GROUP_SIZE / QWEN3OMNI_PROMPT_RANK_GROUP_SIZE)) videos/rank"
  echo "Training log: ${log_path}"

  unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
  if [[ -z "${CONDA_SH}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "ERROR: conda is not available; install Miniconda or set CONDA_SH." >&2
      exit 1
    fi
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${LTX_CONDA_ENV}"
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes "${TRAIN_NNODES}" \
    --nproc_per_node "${EXPECTED_GPUS_PER_NODE}" \
    --node_rank "${train_node_rank}" \
    --master_addr "${train_master_addr}" \
    --master_port "${train_master_port}" \
    scripts/train.py \
    --config config/nft.py:ltx2_qwen3omni_av_quality_split_reward \
    2>&1 | tee "${log_path}"
  local train_status="${PIPESTATUS[0]}"
  set -e

  if [[ "${train_status}" -eq 0 ]]; then
    atomic_marker "${COORD_DIR}/train_node_${train_node_rank}_${LAUNCH_GENERATION}.done" "success"
    training_succeeded=1
    trap - EXIT
    return 0
  fi
  atomic_marker "${COORD_DIR}/train_node_${train_node_rank}_${LAUNCH_GENERATION}.failed" "exit=${train_status}"
  atomic_marker "${TRAINING_FAILED_MARKER}" "node=${train_node_rank}, exit=${train_status}"
  return "${train_status}"
}

echo "Shared run directory: ${RUN_DIR}"
if [[ "${PET_LAUNCH_VALIDATE_ONLY:-0}" == "1" ]]; then
  if [[ "${PET_NODE_RANK}" -eq "${REWARD_NODE_RANK}" ]]; then
    echo "Validation passed; this node would run ${REWARD_SERVER_COUNT} TP=${REWARD_TP_SIZE} reward servers."
  else
    echo "Validation passed; this node would be training node ${PET_NODE_RANK}/${TRAIN_NNODES}."
  fi
  exit 0
fi

if [[ "${PET_NODE_RANK}" -eq "${REWARD_NODE_RANK}" ]]; then
  reward_node_main
else
  training_node_main
fi
