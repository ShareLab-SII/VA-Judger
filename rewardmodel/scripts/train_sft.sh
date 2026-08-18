#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the original Qwen3-Omni-30B-A3B-Instruct directory}"
: "${VAPREF_ROOT:=${REPO_ROOT}/data/VAPref-10K}"
: "${TRAIN_JSONL:=${VAPREF_ROOT}/train.jsonl}"
: "${OUTPUT_DIR:=${REPO_ROOT}/outputs/sft}"
: "${CONDA_ENV:=swift}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NPROC_PER_NODE:=8}"

# Conservative defaults for full-parameter SFT.
: "${DEEPSPEED:=zero3_offload}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=16}"
: "${MAX_LENGTH:=24576}"
: "${LEARNING_RATE:=5e-6}"
: "${MAX_STEPS:=2000}"
: "${SAVE_STEPS:=100}"
: "${SAVE_TOTAL_LIMIT:=20}"
: "${ATTN_IMPL:=sdpa}"
: "${PADDING_FREE:=false}"
: "${DATASET_NUM_PROC:=4}"
: "${DATALOADER_NUM_WORKERS:=2}"
: "${FREEZE_VIT:=true}"
: "${FREEZE_ALIGNER:=true}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not a directory: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "ERROR: TRAIN_JSONL is not a file: ${TRAIN_JSONL}" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "ERROR: the ms-swift CLI is unavailable in Conda env ${CONDA_ENV}." >&2
  echo "Activate the training environment and run: pip install -r requirements-train.txt" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RESOLVED_JSONL="${OUTPUT_DIR}/train.resolved.jsonl"
python scripts/resolve_dataset.py --input "${TRAIN_JSONL}" --output "${RESOLVED_JSONL}"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-602112}"
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-true}"
export ENABLE_AUDIO_OUTPUT=0

echo "Model: ${MODEL_PATH}"
echo "Dataset: ${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") rows)"
echo "Output: ${OUTPUT_DIR}"
echo "Memory defaults: deepspeed=${DEEPSPEED}, micro_batch=${PER_DEVICE_TRAIN_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, gradient_checkpointing=true"

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Validation passed; continuous SFT would run on ${NPROC_PER_NODE} GPUs."
  exit 0
fi

swift sft \
  --model "${MODEL_PATH}" \
  --model_type qwen3_omni_moe \
  --check_model false \
  --dataset "${RESOLVED_JSONL}" \
  --dataset_shuffle true \
  --split_dataset_ratio 0 \
  --load_from_cache_file true \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl "${ATTN_IMPL}" \
  --experts_impl eager \
  --freeze_vit "${FREEZE_VIT}" \
  --freeze_aligner "${FREEZE_ALIGNER}" \
  --padding_free "${PADDING_FREE}" \
  --gradient_checkpointing true \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps 5 \
  --max_length "${MAX_LENGTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --add_version false \
  --create_checkpoint_symlink false \
  --warmup_ratio 0.05 \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --deepspeed "${DEEPSPEED}" \
  "${@}"
