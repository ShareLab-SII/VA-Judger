<h2 align="center">VA-Judger</h2>

<h4 align="center">Dimension-aware Reinforcement Learning for Joint Audio-Video Generation</h4>

<p align="center">
  <a href="https://huggingface.co/ShareLab-SII/VA-Judger"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-VA--Judger-ffc107" alt="Hugging Face"/>
</p>

VA-Judger is a compact release of our reinforcement-learning pipeline for
LTX-2 text-to-audio-video generation. A fine-tuned Qwen3-Omni reward model
compares every pair of rollouts on five dimensions. The resulting prompt,
audio, video, and cross-modal scores are routed to the corresponding LTX-2
branches during GRPO training.

This repository contains launchers for:

- 8 GPUs: 7 LTX-2 training workers + 1 vLLM reward-model GPU.
- 32 GPUs: 24 LTX-2 training workers + 8 vLLM reward-model GPUs.
- Single-prompt LoRA inference with two-stage 2x spatial upsampling.

No machine-specific absolute paths are required. All scripts use paths below
`weights/` by default and can be overridden with environment variables.

## Repository layout

```text
videorl/
├── config/                     # LTX-2 GRPO configuration
├── dataset/                    # Prompt metadata
├── examples/santa_prompt.txt  # Default inference prompt
├── flow_grpo/                  # Reward client, routing, and vLLM server
├── ltx_v2/                     # Required LTX-2 modules
└── scripts/
    ├── download_weights.sh
    ├── train_8gpu.sh
    ├── train_32gpu.sh
    ├── train.py
    ├── infer.sh
    ├── infer_two_stage.py
    └── merge_lora.py
```

## Installation

### Prerequisites

- Linux, NVIDIA GPUs, and a working CUDA driver.
- Conda or Miniconda.
- Python 3.11.
- `ffmpeg` available in `PATH`.
- For 32-GPU training, four 8-GPU nodes with a shared project/output
  filesystem and unrestricted TCP connectivity between nodes.

Training/inference and reward serving are intentionally installed in separate
environments because vLLM often requires a specific PyTorch build.

### LTX-2 training and inference environment

```bash
conda create -n ltx2 python=3.11 -y
conda activate ltx2
pip install --upgrade pip
pip install -r requirements.txt
```

If your cluster requires a specific CUDA build of PyTorch, install the matching
`torch`, `torchvision`, and `torchaudio` wheels first, then install
`requirements.txt`. On Hopper GPUs, a compatible FlashAttention 3 build can be
installed for additional attention speed; otherwise the code uses the
available PyTorch/xFormers attention backend.

### Qwen3-Omni vLLM reward environment

```bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install --upgrade pip
pip install -r requirements-reward.txt
```

The reward server imports `swift.infer_engine.VllmEngine`. Installing
`ms-swift` from PyPI is sufficient. To use a source checkout instead, install
it with `pip install -e /path/to/ms-swift` or set
`MS_SWIFT_ROOT=/path/to/ms-swift` when launching.

Verify both environments before downloading large checkpoints:

```bash
conda run -n ltx2 python -c "import torch, peft; print(torch.__version__)"
conda run -n vllm python -c "import swift, vllm; print(vllm.__version__)"
ffmpeg -version
```

## Model checkpoints

First accept the model terms for the gated LTX-2/Gemma repositories and log in
with a Hugging Face read token:

```bash
conda activate ltx2
hf auth login
bash scripts/download_weights.sh
```

The script downloads:

| Component | Source | Default local path |
| --- | --- | --- |
| VA-Judger training LoRA | [LTX-RL-VA-Judger](https://huggingface.co/ShareLab-SII/VA-Judger/tree/main/LTX-RL-VA-Judger) | `weights/VA-Judger/LTX-RL-VA-Judger/lora` |
| Qwen3-Omni reward model | [VA-Judger](https://huggingface.co/ShareLab-SII/VA-Judger/tree/main/VA-Judger) | `weights/VA-Judger/VA-Judger` |
| LTX-2 19B dev model | [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) | `weights/LTX-2/ltx-2-19b-dev.safetensors` |
| LTX-2 distilled refinement LoRA | [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) | `weights/LTX-2/ltx-2-19b-distilled-lora-384.safetensors` |
| 2x spatial upsampler | [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2/tree/main/latent_upsampler) | `weights/LTX-2/latent_upsampler/diffusion_pytorch_model.safetensors` |
| Gemma 3 text encoder | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) | `weights/LTX-2/gemma` |

The reward-model upload may temporarily be incomplete. Training can start only
after all 13 model shards referenced by `model.safetensors.index.json` are
present. LoRA inference does not load the reward model and can be used as soon
as the LTX-2 and LoRA assets are available.

To store weights elsewhere, set one root once:

```bash
WEIGHTS_ROOT=/data/models/va-judger bash scripts/download_weights.sh
WEIGHTS_ROOT=/data/models/va-judger bash scripts/infer.sh
```

Individual paths can also be overridden with `LTX_MODEL_PATH`,
`GEMMA_MODEL_PATH`, `QWEN3OMNI_REWARD_MODEL`, `LORA_DIR`,
`SPATIAL_UPSAMPLER_PATH`, and `DISTILLED_LORA_PATH`.

## Training

The supplied metadata files contain prompts only. Replace
`PAIR_VIDEO_PROMPT_TRAIN_DATASET` and `PAIR_VIDEO_PROMPT_TEST_DATASET` if using
your own JSONL/CSV prompt set.

### 8 GPUs: 7 training + 1 reward

GPU 0 serves VA-Judger through vLLM. GPUs 1-7 execute the same LTX-2 rollout,
dimension decoding, reward routing, and FSDP/GRPO training logic used by the
32-GPU launcher.

```bash
bash scripts/train_8gpu.sh
```

Simple background launch:

```bash
nohup bash scripts/train_8gpu.sh > train_8gpu.log 2>&1 &
```

Validate paths and the 7+1 split without loading either model:

```bash
VALIDATE_ONLY=1 bash scripts/train_8gpu.sh
```

Check the process and logs with:

```bash
tail -f train_8gpu.log
jobs -l
```

The default GPU split can be changed without editing the script:

```bash
QWEN_GPU=7 LTX_GPUS=0,1,2,3,4,5,6 bash scripts/train_8gpu.sh
```

### 32 GPUs: 24 training + 8 reward

The launcher assumes four nodes with eight GPUs per node:

| Node rank | Role | GPU use |
| --- | --- | --- |
| 0-2 | LTX-2 FSDP/GRPO | 8 GPUs per node, 24 ranks total |
| 3 | VA-Judger vLLM | 4 servers, TP=2, 8 GPUs total |

All nodes must use the same `TRAIN_JOB_ID`, `RUNNING_ROUND`, `MASTER_ADDR`, and
`MASTER_PORT`. `PET_NODE_RANK` is the only value that changes per node.

Run the following on every node, setting `PET_NODE_RANK` to `0`, `1`, `2`, or
`3` respectively:

```bash
export PET_NNODES=4
export PET_NPROC_PER_NODE=8
export PET_NODE_RANK=0
export TRAIN_JOB_ID=va_judger_run
export RUNNING_ROUND=0
export MASTER_ADDR=<node-0-host-or-ip>
export MASTER_PORT=23456

bash scripts/train_32gpu.sh
```

To keep each node launcher in the background after exporting the variables
above:

```bash
nohup bash scripts/train_32gpu.sh > "train_32gpu_node_${PET_NODE_RANK}.log" 2>&1 &
```

For a scheduler, map its node-rank variables to `PET_NODE_RANK` and execute the
same script once per node. The launcher starts reward servers on node 3,
publishes their health-checked URLs through the shared output directory, and
then starts a 24-rank `torchrun` job on nodes 0-2.

Use a unique generation identifier when retrying the same scheduler job:

```bash
PET_LAUNCH_GENERATION=retry_01 bash scripts/train_32gpu.sh
```

### 32-GPU memory-saving defaults

The 32-GPU launcher is conservative by default:

- `LTX_FSDP_SHARDING_STRATEGY=FULL_SHARD`: shards parameters, gradients, and
  optimizer state across all 24 training ranks.
- `LTX_ACTIVATION_CHECKPOINTING=1`: recomputes transformer activations during
  backward instead of retaining all of them.
- `LTX_TRAIN_SAMPLE_CPU_OFFLOAD=1`: moves stored rollout tensors to CPU between
  sampling and optimization.
- `LTX_FSDP_BACKWARD_PREFETCH=NONE`: avoids the extra peak memory from
  prefetching the next FSDP shard.
- `QWEN3OMNI_PROMPT_RANK_GROUP_SIZE=4`: spreads each group of eight rollouts
  over four ranks, so each rank generates only two videos.
- `QWEN3OMNI_TRAIN_MICRO_BSZ=1`: keeps the per-rank optimization microbatch at
  one.
- Reward node: four vLLM servers with `REWARD_TP_SIZE=2` and
  `VLLM_GPU_MEMORY_UTILIZATION=0.88`.

For an out-of-memory error, keep the above defaults and reduce reward-side
pressure first:

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.82 \
VLLM_MAX_NUM_SEQS=4 \
QWEN3OMNI_REWARD_PAIR_BATCH=4 \
bash scripts/train_32gpu.sh
```

If training ranks still run out of memory, parameter offload is the strongest
fallback, but it is substantially slower and requires enough host RAM:

```bash
LTX_FSDP_CPU_OFFLOAD=1 bash scripts/train_32gpu.sh
```

### Faster settings when H200 memory is sufficient

Change one group at a time and watch peak memory. These switches preserve the
reward decoding and branch-routing logic:

```bash
LTX_TRAIN_SAMPLE_CPU_OFFLOAD=0 \
LTX_ACTIVATION_CHECKPOINTING=0 \
LTX_FSDP_BACKWARD_PREFETCH=BACKWARD_PRE \
bash scripts/train_32gpu.sh
```

- Disabling sample CPU offload removes CPU/GPU transfer stalls.
- Disabling activation checkpointing avoids recomputation during backward.
- `BACKWARD_PRE` overlaps the next all-gather with computation, at the cost of
  a higher memory peak.

If there is still substantial headroom, test the following independently:

```bash
# Less aggressive sharding; potentially faster, but much more memory-hungry.
LTX_FSDP_SHARDING_STRATEGY=SHARD_GRAD_OP bash scripts/train_32gpu.sh

# More rollouts per rank; may improve utilization but increases rollout memory.
QWEN3OMNI_PROMPT_RANK_GROUP_SIZE=2 bash scripts/train_32gpu.sh

# Higher reward-server concurrency.
VLLM_GPU_MEMORY_UTILIZATION=0.93 \
VLLM_MAX_NUM_SEQS=16 \
QWEN3OMNI_REWARD_PAIR_BATCH=16 \
bash scripts/train_32gpu.sh
```

`QWEN3OMNI_PAIR_GROUP_SIZE` should normally remain `8`: changing it changes the
number of candidates and pairwise comparisons in each GRPO group. Reducing
`LTX_ROLLOUT_NUM_STEPS` or `LTX_TIMESTEP_FRACTION` also speeds training, but
changes the sampling/optimization approximation and is therefore not a
memory-only optimization.

### Common training variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `QWEN3OMNI_PAIR_GROUP_SIZE` | `8` | Candidates per prompt |
| `QWEN3OMNI_PROMPT_RANK_GROUP_SIZE` | `1` (8 GPU), `4` (32 GPU) | Ranks sharing one prompt group |
| `LTX_RESOLUTION_HEIGHT/WIDTH` | `512/768` | Rollout resolution |
| `LTX_ROLLOUT_NUM_STEPS` | `20` | Denoising steps per training rollout |
| `LTX_TIMESTEP_FRACTION` | `0.4` | Fraction of rollout timesteps used for policy updates |
| `LTX_ACTIVATION_CHECKPOINTING` | `1` | Activation checkpointing switch |
| `LTX_TRAIN_SAMPLE_CPU_OFFLOAD` | `0` (8 GPU), `1` (32 GPU) | Store rollout tensors on CPU |
| `LTX_FSDP_CPU_OFFLOAD` | `0` | Offload FSDP parameters to CPU |
| `LTX_RL_LR` | `3e-5` | LoRA learning rate |
| `LTX_RESUME_FROM` | empty | Checkpoint directory to resume |
| `WANDB_MODE` | `offline` | Weights & Biases mode |

## Reward decoding and routing

For each pair of videos, VA-Judger returns 1-10 scores for:

- A: prompt matching.
- B: audio-video consistency.
- C: audio quality.
- D: video quality.
- E: completeness and coherence.

The decoded normalized rewards are:

```text
overall = (A + B + E) / 30
audio   = C / 10
video   = D / 10
```

For the default group size of eight, all 28 unordered pairs are evaluated.
Each candidate receives its mean score across the seven comparisons that
contain it. `overall` is routed to the shared/synchronization branch, `audio`
to the audio branch, and `video` to the video branch. If a response cannot be
parsed, that comparison receives the neutral score `0.5` rather than crashing
the training job.

## Inference

The default command uses the released VA-Judger LoRA and the prompt in
`examples/santa_prompt.txt`:

```bash
bash scripts/infer.sh
```

The launcher:

1. Merges `weights/VA-Judger/LTX-RL-VA-Judger/lora` into the LTX-2 dev model.
2. Generates a 512x768 first-stage audio-video sample.
3. Applies 2x spatial upsampling and distilled refinement.
4. Writes a final 1024x1536 MP4 with audio and a separate WAV file under
   `outputs/inference_<timestamp>/generated/`.

Run inference in the background with:

```bash
nohup bash scripts/infer.sh > infer.log 2>&1 &
```

Use a LoRA produced by training:

```bash
# Default 8-GPU output layout; adjust the checkpoint number as needed.
LORA_DIR=outputs/checkpoints/va_judger_ltx2/checkpoint-451/lora \
bash scripts/infer.sh
```

For the 32-GPU launcher, checkpoints are under
`outputs/qwen3omni_avsplit_4node_<generation>/training_outputs/checkpoints/`.

Use another prompt or output shape:

```bash
PROMPT_FILE=/path/to/prompt.txt \
OUTPUT_HEIGHT=1024 \
OUTPUT_WIDTH=1536 \
NUM_FRAMES=121 \
SEED=123 \
bash scripts/infer.sh
```

Set `OVERWRITE_MERGED=1` to rebuild an existing merged checkpoint. Set
`VALIDATE_ONLY=1` to verify all required paths without loading models:

```bash
VALIDATE_ONLY=1 bash scripts/infer.sh
```

## Acknowledgements

This release builds on [OmniNFT](https://github.com/zghhui/OmniNFT),
[LTX-2](https://github.com/Lightricks/LTX-2),
[ms-swift](https://github.com/modelscope/ms-swift), and
[vLLM](https://github.com/vllm-project/vllm). Please follow the licenses and
model-use terms of all upstream projects and checkpoints.

## License

The Hugging Face model repository is marked Apache-2.0. Before redistributing
this source bundle, retain the license and notices required by the vendored
LTX-2 components and verify that the included prompt metadata is redistributable.
