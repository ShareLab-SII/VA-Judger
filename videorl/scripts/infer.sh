#!/usr/bin/env bash
# Merge a VA-Judger training LoRA and generate one audio-video sample with
# two-stage 2x spatial upsampling.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${WEIGHTS_ROOT:=${REPO_ROOT}/weights}"
: "${CONDA_SH:=}"
: "${CONDA_ENV:=ltx2}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${BASE_MODEL_PATH:=${WEIGHTS_ROOT}/LTX-2/ltx-2-19b-dev.safetensors}"
: "${LORA_DIR:=${WEIGHTS_ROOT}/VA-Judger/LTX-RL-VA-Judger/lora}"
: "${GEMMA_PATH:=${WEIGHTS_ROOT}/LTX-2/gemma}"
: "${SPATIAL_UPSAMPLER_PATH:=${WEIGHTS_ROOT}/LTX-2/latent_upsampler/diffusion_pytorch_model.safetensors}"
: "${DISTILLED_LORA_PATH:=${WEIGHTS_ROOT}/LTX-2/ltx-2-19b-distilled-lora-384.safetensors}"
: "${PROMPT_FILE:=${REPO_ROOT}/examples/santa_prompt.txt}"
: "${OUTPUT_HEIGHT:=1024}"
: "${OUTPUT_WIDTH:=1536}"
: "${NUM_FRAMES:=121}"
: "${FRAME_RATE:=24}"
: "${NUM_INFERENCE_STEPS:=40}"
: "${VIDEO_GUIDANCE_SCALE:=3.0}"
: "${AUDIO_GUIDANCE_SCALE:=7.0}"
: "${DISTILLED_LORA_STRENGTH:=0.8}"
: "${SEED:=42}"
: "${MASTER_PORT:=6211}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/outputs/inference_${RUN_TIMESTAMP}}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-${WORK_DIR}/merged/va_judger_ltx2.safetensors}"
PROMPT_JSONL="${WORK_DIR}/prompt.jsonl"
OUTPUT_DIR="${WORK_DIR}/generated"
LOG_FILE="${WORK_DIR}/inference.log"

mkdir -p "${WORK_DIR}/merged" "${OUTPUT_DIR}"

for required in \
  "${BASE_MODEL_PATH}" \
  "${LORA_DIR}/adapter_config.json" \
  "${LORA_DIR}/adapter_model.safetensors" \
  "${GEMMA_PATH}" \
  "${SPATIAL_UPSAMPLER_PATH}" \
  "${DISTILLED_LORA_PATH}" \
  "${PROMPT_FILE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: missing required path: ${required}" >&2
    exit 1
  fi
done

if [[ -z "${CONDA_SH}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available; install Miniconda or set CONDA_SH." >&2
    exit 1
  fi
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda initialization script not found: ${CONDA_SH}" >&2
  exit 1
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "Validation passed; LoRA merge and two-stage inference inputs are available."
  exit 0
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${MERGED_MODEL_PATH}" || "${OVERWRITE_MERGED:-0}" == "1" ]]; then
  python scripts/merge_lora.py \
    --checkpoint-path "${BASE_MODEL_PATH}" \
    --lora-dir "${LORA_DIR}" \
    --output-path "${MERGED_MODEL_PATH}" \
    --dtype bf16
else
  echo "Reusing merged checkpoint: ${MERGED_MODEL_PATH}"
fi

python - "${PROMPT_FILE}" "${PROMPT_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

prompt = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
if not prompt:
    raise SystemExit("prompt file is empty")
record = {"set": "example", "sample_id": "santa", "prompt_av": prompt}
Path(sys.argv[2]).write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "Stage 1: $((OUTPUT_WIDTH / 2))x$((OUTPUT_HEIGHT / 2)); final: ${OUTPUT_WIDTH}x${OUTPUT_HEIGHT}"
echo "Output directory: ${OUTPUT_DIR}"

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  --master_addr 127.0.0.1 \
  --master_port "${MASTER_PORT}" \
  scripts/infer_two_stage.py \
    --model-path "${MERGED_MODEL_PATH}" \
    --gemma-path "${GEMMA_PATH}" \
    --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
    --distilled-lora-path "${DISTILLED_LORA_PATH}" \
    --distilled-lora-strength "${DISTILLED_LORA_STRENGTH}" \
    --input-jsonl "${PROMPT_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --prompt-key prompt_av \
    --constant-seed \
    --seed "${SEED}" \
    --height "${OUTPUT_HEIGHT}" \
    --width "${OUTPUT_WIDTH}" \
    --num-frames "${NUM_FRAMES}" \
    --frame-rate "${FRAME_RATE}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --video-guidance-scale "${VIDEO_GUIDANCE_SCALE}" \
    --audio-guidance-scale "${AUDIO_GUIDANCE_SCALE}" \
    --streaming-prefetch-count "${STREAMING_PREFETCH_COUNT:-2}" \
    --max-batch-size 1 \
    --save-wav \
    2>&1 | tee "${LOG_FILE}"

echo "Done: ${OUTPUT_DIR}/000000_example_santa.mp4"
