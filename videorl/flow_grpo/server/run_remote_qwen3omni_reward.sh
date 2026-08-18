#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

: "${WEIGHTS_ROOT:=${REPO_ROOT}/weights}"
: "${CONDA_SH:=}"
: "${CONDA_ENV:=vllm}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${QWEN3OMNI_REWARD_PORT:=8100}"
: "${QWEN3OMNI_REWARD_MODEL:=${WEIGHTS_ROOT}/VA-Judger/VA-Judger}"
: "${MS_SWIFT_ROOT:=}"

if [[ -z "${CONDA_SH}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
if [[ -n "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES
export QWEN3OMNI_REWARD_MODEL
export QWEN3OMNI_REWARD_PORT
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
if [[ -z "${VLLM_LIMIT_MM_PER_PROMPT:-}" ]]; then
  export VLLM_LIMIT_MM_PER_PROMPT='{"video": 2, "audio": 2}'
fi

python flow_grpo/server/qwen3omni_pair_reward_server.py \
  --model "${QWEN3OMNI_REWARD_MODEL}" \
  --port "${QWEN3OMNI_REWARD_PORT}" \
  "$@"
