# VA-Judger Reward Model

Training and evaluation code for the VA-Judger audio-video preference model.
Given a caption and two generated audio/video candidates, the model compares
prompt alignment, audio-video consistency, audio quality, video quality, and
content completeness, then predicts either `video 1 is better` or
`video 2 is better`.

This directory is self-contained at the source-code level. Model checkpoints
and datasets are supplied at runtime through environment variables or the
ignored `weights/` and `data/` directories. No machine-specific path is
required by the scripts.

## Contents

```text
rewardmodel/
├── scripts/
│   ├── train_sft.sh          # Continuous multi-GPU SFT
│   ├── evaluate.sh           # Multi-GPU data-parallel benchmark launcher
│   ├── evaluate.py           # Inference and accuracy calculation
│   ├── resolve_dataset.py    # Resolve portable training-video paths
│   └── prepare_vapref.py     # Export a portable training dataset
├── environment-train.yml
├── environment-eval.yml
├── requirements-train.txt
├── requirements-eval.txt
└── README.md
```

## Requirements

- Linux.
- An NVIDIA GPU and driver compatible with the PyTorch/vLLM CUDA build.
- Conda or Miniconda.
- `ffmpeg`, required for audio/video decoding.
- Enough local storage for Qwen3-Omni, the selected checkpoint, datasets, and
  generated outputs.

Training and evaluation should use separate environments. Full-parameter SFT
uses DeepSpeed, while evaluation uses vLLM; forcing both dependency stacks
into one environment makes CUDA and PyTorch conflicts more likely.

The supplied dependency files pin MS-Swift 4.4.2, vLLM 0.19.0, and DeepSpeed
0.19.1. The
launcher uses MS-Swift's `InferRequest` and `VllmEngine`; vLLM remains the
underlying inference engine. Both packages are therefore required for the
current evaluator.

## Environment setup

Run all commands below from the `rewardmodel/` directory.

### Evaluation environment

Create the environment from the supplied file:

```bash
conda env create -f environment-eval.yml
conda activate vllm
```

Equivalent manual setup:

```bash
conda create -n vllm python=3.12 ffmpeg -c conda-forge -y
conda activate vllm
python -m pip install --upgrade pip
python -m pip install -r requirements-eval.txt
```

If the configured package mirror does not provide MS-Swift, install from the
official Python Package Index explicitly:

```bash
python -m pip install --index-url https://pypi.org/simple \
  -r requirements-eval.txt
```

Verify the imports before allocating GPUs:

```bash
python - <<'PY'
import torch
import vllm
import swift
from swift import InferStats, RequestConfig
from swift.infer_engine import InferRequest, VllmEngine

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("vLLM:", vllm.__version__)
print("MS-Swift:", swift.__version__)
print("Evaluation imports: OK")
PY
```

If `torch.cuda.is_available()` is false on a GPU machine, do not start the
benchmark. Install a PyTorch/vLLM build compatible with the host driver and
CUDA runtime first. Do not mix packages from unrelated CUDA builds.

### Training environment

Create and activate the independent SFT environment:

```bash
conda env create -f environment-train.yml
conda activate swift
```

Equivalent manual setup:

```bash
conda create -n swift python=3.12 ffmpeg -c conda-forge -y
conda activate swift
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

If a private package mirror is incomplete, use the official index:

```bash
python -m pip install --index-url https://pypi.org/simple \
  -r requirements-train.txt
```

Verify it with:

```bash
swift --version
python -c 'import torch, deepspeed, swift; print(torch.__version__, swift.__version__)'
```

DeepSpeed compilation depends on the local CUDA toolchain. If installation
fails, confirm that the CUDA toolkit used to build extensions is compatible
with the installed PyTorch build.

## Model preparation

The base model is Qwen3-Omni-30B-A3B-Instruct. Download it with the Hugging
Face CLI or place an existing local copy under `weights/`:

```bash
conda activate swift
python -m pip install --upgrade huggingface_hub
hf auth login
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir weights/Qwen3-Omni-30B-A3B-Instruct
```

For training, `MODEL_PATH` points to the base-model directory. For evaluation,
it points to a full SFT checkpoint. A full checkpoint must contain the model
configuration, tokenizer/processor files, the Safetensors index, and every
shard referenced by that index.

Examples in this README use repository-relative paths, but absolute paths are
also accepted through environment variables:

```bash
export MODEL_PATH=weights/Qwen3-Omni-30B-A3B-Instruct
```

## Dataset preparation

### VAPref-10K training data

The reward-model training data are not included in this repository. Prepare
your own pairwise audio-video preference data with this layout:

```text
data/VAPref-10K/
├── train.jsonl
└── videos/
    ├── example_1.mp4
    └── example_2.mp4
```

Each JSONL record contains only `messages` and `videos`:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "... <video> ... <video>"},
    {"role": "assistant", "content": "... <answer>video 1 is better</answer>"}
  ],
  "videos": ["videos/example_1.mp4", "videos/example_2.mp4"]
}
```

The `videos` list must contain exactly two paths relative to `train.jsonl`.
The user message must contain two `<video>` placeholders in the same order as
the paths, and the assistant message must contain the pairwise preference
target. The training launcher resolves the relative media paths into an
ignored runtime JSONL without modifying the source dataset.

To convert an existing messages/videos JSONL and its media files into the
portable layout:

```bash
python scripts/prepare_vapref.py \
  --input /path/to/source_train.jsonl \
  --data-root /path/to/source_data_root \
  --output-root data/VAPref-10K
```

The exporter validates every record, copies referenced videos into `videos/`,
and writes relative paths. Users are responsible for preparing and licensing
their own training data.

### VA-Judger-Bench evaluation data

The benchmark directory must contain three splits:

```text
data/VA-Judger-Bench/
├── easy/
│   ├── data.jsonl
│   └── videos/
├── indomain/
│   ├── data.jsonl
│   └── videos/
└── outdomain/
    ├── data.jsonl
    └── videos/
```

Each record requires:

```json
{
  "case_id": "unique-case-id",
  "text_prompt": "generation prompt",
  "video_1_relative_path": "videos/candidate_1.mp4",
  "video_2_relative_path": "videos/candidate_2.mp4",
  "human_preference_answer": "video 1 is better"
}
```

`dataset_summary.json` is not required. The launcher validates the three
split-level `data.jsonl` files directly.

Datasets may live elsewhere and be linked into the repository:

```bash
mkdir -p data
ln -s /path/to/VAPref-10K data/VAPref-10K
ln -s /path/to/VA-Judger-Bench data/VA-Judger-Bench
```

The `data/` directory is ignored by Git.

## Validate paths before running

Training validation checks the model and dataset paths and generates the
resolved runtime JSONL, but does not launch SFT:

```bash
conda activate swift
MODEL_PATH=weights/Qwen3-Omni-30B-A3B-Instruct \
VALIDATE_ONLY=1 \
bash scripts/train_sft.sh
```

Evaluation validation checks the model directory and all three benchmark
JSONLs without loading the model onto a GPU:

```bash
conda activate vllm
MODEL_PATH=weights/VA-Judger/VA-Judger \
BENCH_ROOT=data/VA-Judger-Bench \
VALIDATE_ONLY=1 \
bash scripts/evaluate.sh
```

## SFT

Start the default eight-GPU full-parameter SFT run:

```bash
conda activate swift
MODEL_PATH=weights/Qwen3-Omni-30B-A3B-Instruct \
bash scripts/train_sft.sh
```

The default configuration uses:

- Eight data-parallel workers.
- BF16 full-parameter SFT.
- DeepSpeed ZeRO-3 parameter and optimizer CPU offload.
- Gradient checkpointing.
- Frozen visual/audio towers and aligner.
- Per-device batch size 1 and gradient accumulation 16.
- Maximum sequence length 24,576.
- At most 12 sampled video frames and 128 video tokens.
- A checkpoint every 100 optimizer steps.
- No in-training evaluation.

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
GRADIENT_ACCUMULATION_STEPS=32 \
MAX_STEPS=1000 \
SAVE_STEPS=50 \
OUTPUT_DIR=outputs/my_sft \
MODEL_PATH=weights/Qwen3-Omni-30B-A3B-Instruct \
bash scripts/train_sft.sh
```

Additional arguments are forwarded to `swift sft`:

```bash
MODEL_PATH=weights/Qwen3-Omni-30B-A3B-Instruct \
bash scripts/train_sft.sh \
  --resume_from_checkpoint outputs/sft/checkpoint-500
```

If training runs out of memory, reduce `MAX_LENGTH`,
`VIDEO_MAX_TOKEN_NUM`, or `FPS_MAX_FRAMES`. Parameter CPU offload requires
substantial host RAM and is slower than keeping parameters on GPU.

## Evaluation

Activate the evaluation environment and run all three splits:

```bash
conda activate vllm
MODEL_PATH=weights/VA-Judger/VA-Judger \
BENCH_ROOT=data/VA-Judger-Bench \
bash scripts/evaluate.sh
```

The launcher treats every visible GPU as an independent data-parallel worker.
Each worker loads a complete TP=1 model replica and evaluates a disjoint shard
of the split. Using more GPUs improves throughput but does not reduce the
memory required on each GPU.

Use a selected set of GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=weights/VA-Judger/VA-Judger \
BENCH_ROOT=data/VA-Judger-Bench \
bash scripts/evaluate.sh
```

Useful evaluation controls:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | One independent worker per listed GPU |
| `BATCH_SIZE` | `16` | Number of benchmark requests submitted per batch |
| `MAX_NEW_TOKENS` | `2048` | Maximum generated tokens per request |
| `VLLM_MAX_MODEL_LEN` | `24576` | Model context limit |
| `VLLM_MAX_NUM_SEQS` | `16` | Maximum vLLM request concurrency |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of each GPU reserved by vLLM |
| `OUTPUT_ROOT` | `outputs/evaluation/<checkpoint>` | Result directory |
| `CONDA_ENV` | `vllm` | Conda environment activated by the launcher |

For a partially occupied GPU, lower the memory reservation and concurrency:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_GPU_MEMORY_UTILIZATION=0.5 \
VLLM_MAX_NUM_SEQS=4 \
BATCH_SIZE=4 \
MODEL_PATH=weights/VA-Judger/VA-Judger \
BENCH_ROOT=data/VA-Judger-Bench \
bash scripts/evaluate.sh
```

vLLM requires the requested reservation to be smaller than the free memory at
startup. If a GPU has 80 GiB free out of 140 GiB, for example, the utilization
must be below approximately `0.57`. Prefer releasing unrelated GPU processes
instead of using an extremely small KV-cache budget.

Evaluation uses greedy decoding and writes:

```text
outputs/evaluation/<checkpoint>/
├── easy/
│   ├── results.worker0.jsonl
│   ├── results.jsonl
│   ├── summary.json
│   └── worker0.log
├── indomain/
├── outdomain/
└── summary.json
```

Reported metrics are:

- `all_cases_accuracy`: correct predictions divided by all cases; failures and
  unparseable completions count as incorrect.
- `parsed_only_accuracy`: accuracy among parseable completions.
- `parse_rate`: parseable completions divided by all cases.

Worker outputs are resumable. Re-running the same command skips case IDs
already present in the corresponding worker file. Use a new `OUTPUT_ROOT` when
changing the number of GPUs, decoding settings, or checkpoint; otherwise old
and new results may be mixed.

## Troubleshooting

### `ModuleNotFoundError: No module named 'swift'`

The evaluator uses MS-Swift to construct Qwen3-Omni audio/video requests and
uses vLLM as the backend. Install both dependencies in the evaluation
environment:

```bash
conda activate vllm
python -m pip install -r requirements-eval.txt
```

The launcher intentionally does not search for a neighboring MS-Swift source
checkout. This keeps the public repository independent of the developer's
filesystem layout.

### `No matching distribution found for ms-swift`

The active pip mirror may not synchronize MS-Swift. Confirm the configured
index and retry against the official Python Package Index:

```bash
python -m pip config list
python -m pip install --index-url https://pypi.org/simple \
  -r requirements-eval.txt
```

MS-Swift can also be installed from its official source repository when a
source build is preferred:

```bash
python -m pip install \
  'git+https://github.com/modelscope/ms-swift.git@v4.4.2'
python -m pip install vllm==0.19.0
```

### `Free memory ... is less than desired GPU memory utilization`

Another process is using the GPU, or the reservation is too high. Stop the
unrelated process or lower `VLLM_GPU_MEMORY_UTILIZATION`. The value must be
below `free_memory / total_memory` at engine startup.

### `BENCH_ROOT is missing <split>/data.jsonl`

Point `BENCH_ROOT` at the directory containing `easy/`, `indomain/`, and
`outdomain/`, not at one individual split.

### A worker fails after evaluation starts

Inspect the exact log named by the launcher:

```bash
tail -n 200 outputs/evaluation/<checkpoint>/<split>/worker0.log
```

Search all worker logs for the first exception:

```bash
rg -n -i 'traceback|error|exception|out of memory|killed' \
  outputs/evaluation/<checkpoint>/<split>/worker*.log
```

Warnings about deprecated `torch_dtype` or an unknown environment variable
are not necessarily fatal. Diagnose the final traceback or the first explicit
`ERROR` emitted by the vLLM engine.

## Releasing this directory

Before publishing, verify that the release does not contain models, datasets,
logs, caches, generated results, or paths from the development machine. Remove
Python caches with:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
```

The provided `.gitignore` excludes `data/`, `weights/`, `outputs/`, logs, and
Python cache files. Dataset and model licenses must be reviewed separately;
the code license does not grant redistribution rights for model weights or
included media.

## Acknowledgements

This implementation builds on
[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct),
[MS-Swift](https://github.com/modelscope/ms-swift), and
[vLLM](https://github.com/vllm-project/vllm). Follow the licenses and release
terms of all upstream projects, models, datasets, and media files.
