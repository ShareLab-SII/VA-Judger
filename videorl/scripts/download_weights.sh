#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${REPO_ROOT}/weights}"

if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo "ERROR: Hugging Face CLI not found. Run: pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "${WEIGHTS_ROOT}/VA-Judger" "${WEIGHTS_ROOT}/LTX-2"

echo "Downloading VA-Judger LoRA and reward model..."
"${HF[@]}" YinmingHuang/VA-Judger \
  --include "LTX-RL-VA-Judger/*" "VA-Judger/*" \
  --local-dir "${WEIGHTS_ROOT}/VA-Judger"

echo "Downloading LTX-2 base model, distilled LoRA, and spatial upsampler..."
"${HF[@]}" Lightricks/LTX-2 \
  ltx-2-19b-dev.safetensors \
  ltx-2-19b-distilled-lora-384.safetensors \
  latent_upsampler/diffusion_pytorch_model.safetensors \
  --local-dir "${WEIGHTS_ROOT}/LTX-2"

echo "Downloading the Gemma 3 text encoder..."
"${HF[@]}" google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir "${WEIGHTS_ROOT}/LTX-2/gemma"

echo "Weights are ready under ${WEIGHTS_ROOT}"
