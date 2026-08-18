# VA-Judger 本次对话工作记录

更新日期：2026-08-05

本文记录本次对话中完成的代码调查、开源目录整理、数据整理、训练与推理脚本、评测适配、运行命令、验证结果和排障结论。它是开发交接记录；面向公开用户的使用说明分别见 `videorl/README.md` 和 `rewardmodel/README.md`。

## 1. 最终交付概览

VA-Judger 被整理为两个相互独立的部分：

```text
VA-Judger/
├── videorl/       # LTX-2 强化学习：8 卡、32 卡和 LoRA 音视频推理
├── rewardmodel/   # Qwen3-Omni reward model：连续 SFT 和独立评测
└── WORKLOG.md     # 本文
```

本机项目根目录为：

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/czxs253130297/VA-Judger
```

路径迁移遵循以下规则：

- 使用当前根目录 `/inspire/qb-ilm/project/semantic-visual-tokenizer/czxs253130297`。
- 不再使用旧目录 `/inspire/hdd/project/autoregressive-video-generation/czxs253130297`。
- Conda 环境只使用当前根目录下的 `miniconda3`，不使用已经弃用的 `anaconda3`。
- 可开源脚本和 README 不依赖本机绝对路径；本地模型和数据位置通过环境变量传入。
- 原始模型权重和已有 checkpoint 没有移动或修改。

## 2. OmniNFT / VideoRL 调查结论

对原 OmniNFT 训练代码的调查确认：

- 存在 32 卡训练逻辑。
- 存在与 32 卡版本核心训练、vLLM rollout、reward 解码和 reward routing 基本一致的 8 卡版本。
- 8 卡布局为 `7 + 1`：GPU 0 运行 Qwen3-Omni vLLM reward server，GPU 1-7 运行 7 个 LTX-2/FSDP/GRPO rank。
- 32 卡布局为 `24 + 8`：4 个 8 卡节点中，前 3 个节点共 24 卡训练 LTX-2；第 4 个节点用 8 卡运行 4 个 TP=2 的 reward server。
- 两种规模共用 `scripts/train.py`、相同的 rollout、dimension score 解码、成对比较和分支 reward 路由逻辑，启动脚本只负责资源拓扑与运行参数。

## 3. `videorl` 开源精简版

### 3.1 已整理文件

```text
videorl/
├── config/
│   ├── base.py
│   └── nft.py
├── examples/
│   └── santa_prompt.txt
├── flow_grpo/
│   ├── ema.py
│   ├── fsdp_utils.py
│   ├── qwen3omni_pair_scorer.py
│   ├── rewards.py
│   └── stat_tracking.py
├── scripts/
│   ├── download_weights.sh
│   ├── train.py
│   ├── train_8gpu.sh
│   ├── train_32gpu.sh
│   ├── merge_lora.py
│   ├── infer_two_stage.py
│   └── infer.sh
├── JavisBench-mini.csv
├── requirements.txt
├── requirements-reward.txt
├── README.md
└── .gitignore
```

README 参考 OmniNFT 的组织方式重新编写，包含环境配置、权重下载、8/32 卡训练、推理、显存优化、reward 解码及开源注意事项。

### 3.2 权重布局

默认使用相对路径：

| 内容 | Hugging Face 来源 | 默认位置 |
| --- | --- | --- |
| VA-Judger 训练 LoRA | `YinmingHuang/VA-Judger/LTX-RL-VA-Judger` | `weights/VA-Judger/LTX-RL-VA-Judger/lora` |
| Qwen3-Omni reward model | `YinmingHuang/VA-Judger/VA-Judger` | `weights/VA-Judger/VA-Judger` |
| LTX-2 19B dev | `Lightricks/LTX-2` | `weights/LTX-2/ltx-2-19b-dev.safetensors` |
| distilled refinement LoRA | `Lightricks/LTX-2` | `weights/LTX-2/ltx-2-19b-distilled-lora-384.safetensors` |
| 2x spatial upsampler | `Lightricks/LTX-2` | `weights/LTX-2/latent_upsampler/diffusion_pytorch_model.safetensors` |
| Gemma 3 text encoder | `google/gemma-3-12b-it-qat-q4_0-unquantized` | `weights/LTX-2/gemma` |

下载命令：

```bash
cd /inspire/qb-ilm/project/semantic-visual-tokenizer/czxs253130297/VA-Judger/videorl
conda activate ltx2
hf auth login
bash scripts/download_weights.sh
```

reward model 上传尚未完成时，RL 训练不能完整启动；普通 LoRA 推理不加载 reward model，因此 LTX-2 和 LoRA 权重完整后即可推理。

### 3.3 8 卡训练

前台：

```bash
cd /inspire/qb-ilm/project/semantic-visual-tokenizer/czxs253130297/VA-Judger/videorl
bash scripts/train_8gpu.sh
```

简单后台启动：

```bash
nohup bash scripts/train_8gpu.sh > train_8gpu.log 2>&1 &
```

只检查路径和 7+1 拓扑，不加载模型：

```bash
VALIDATE_ONLY=1 bash scripts/train_8gpu.sh
```

如需交换 reward 与训练 GPU：

```bash
QWEN_GPU=7 LTX_GPUS=0,1,2,3,4,5,6 bash scripts/train_8gpu.sh
```

### 3.4 32 卡训练

四个节点都设置相同的 job、round、master 地址和端口；每个节点仅修改 `PET_NODE_RANK=0/1/2/3`：

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

后台启动：

```bash
nohup bash scripts/train_32gpu.sh > "train_32gpu_node_${PET_NODE_RANK}.log" 2>&1 &
```

重复提交同一个调度任务时，使用新的 generation 避免旧 coordination marker 冲突：

```bash
PET_LAUNCH_GENERATION=retry_01 bash scripts/train_32gpu.sh
```

### 3.5 32 卡显存配置

默认节省显存参数：

- `LTX_FSDP_SHARDING_STRATEGY=FULL_SHARD`
- `LTX_ACTIVATION_CHECKPOINTING=1`
- `LTX_TRAIN_SAMPLE_CPU_OFFLOAD=1`
- `LTX_FSDP_BACKWARD_PREFETCH=NONE`
- `QWEN3OMNI_PROMPT_RANK_GROUP_SIZE=4`
- `QWEN3OMNI_TRAIN_MICRO_BSZ=1`
- reward 节点默认 4 个 server，每个 `TP=2`
- `VLLM_GPU_MEMORY_UTILIZATION=0.88`

reward 侧 OOM 时可先降低：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.82 \
VLLM_MAX_NUM_SEQS=4 \
QWEN3OMNI_REWARD_PAIR_BATCH=4 \
bash scripts/train_32gpu.sh
```

训练侧仍 OOM 时可启用参数 CPU offload，但速度会明显下降且需要足够主机内存：

```bash
LTX_FSDP_CPU_OFFLOAD=1 bash scripts/train_32gpu.sh
```

H200 显存充足时可依次测试以下加速项：

```bash
LTX_TRAIN_SAMPLE_CPU_OFFLOAD=0 \
LTX_ACTIVATION_CHECKPOINTING=0 \
LTX_FSDP_BACKWARD_PREFETCH=BACKWARD_PRE \
bash scripts/train_32gpu.sh
```

还可分别测试 `LTX_FSDP_SHARDING_STRATEGY=SHARD_GRAD_OP`、减小 `QWEN3OMNI_PROMPT_RANK_GROUP_SIZE`，或增加 vLLM 并发。`QWEN3OMNI_PAIR_GROUP_SIZE=8` 通常不要改，因为它会改变候选数及成对比较数，而不只是改变性能。

### 3.6 Dimension score 解码逻辑

Reward model 对每对视频生成五个 1-10 分的维度：

- A：prompt matching
- B：audio-video consistency
- C：audio quality
- D：video quality
- E：completeness and coherence

归一化为三个训练 reward：

```text
overall = (A + B + E) / 30
audio   = C / 10
video   = D / 10
```

默认一个 prompt 生成 8 个候选，对全部 `C(8, 2) = 28` 个无序视频对进行判断。每个候选参与 7 次比较，其最终分数是这 7 次比较所得分数的均值：

- `overall` 路由到 shared/synchronization 分支。
- `audio` 路由到 audio 分支。
- `video` 路由到 video 分支。
- 无法解析的某次比较使用中性值 `0.5`，避免单次格式异常直接终止训练。

### 3.7 LoRA 单样例推理

推理流程为：合并 LoRA → 生成 512×768 第一阶段音视频 → 2x 空间上采样与 distilled refinement → 输出 1024×1536 MP4 和独立 WAV。

默认 LoRA 已改为发布目录中的 checkpoint-451 对应 LoRA；开源下载后的默认目录是：

```text
weights/VA-Judger/LTX-RL-VA-Judger/lora
```

训练产生的本地 checkpoint 可显式指定：

```bash
LORA_DIR=outputs/checkpoints/va_judger_ltx2/checkpoint-451/lora \
bash scripts/infer.sh
```

前台及后台命令：

```bash
bash scripts/infer.sh
nohup bash scripts/infer.sh > infer.log 2>&1 &
```

默认推理样例使用 `examples/santa_prompt.txt`，内容为本次对话指定的 Santa workshop 音视频提示词。这里只包含单 prompt 普通文本到音视频生成，不包含批量推理。

### 3.8 8 卡日志报错结论

检查 `videorl/train_8gpu.log` 后，LTX-2 已完成加载并实际生成过 rollout 视频。最终错误不是 dimension score 解析错误，而是 reward server 在训练途中不可访问：

```text
http://127.0.0.1:8112/predict
Connection refused
```

即 GPU 1-7 的训练 rank 仍在向 reward server 请求评分，但 GPU 0 上的 vLLM reward 服务已退出。应优先查看该 run 目录中的：

```text
outputs/qwen3omni_ltx2_av_quality_split_<timestamp>/qwen3omni_reward_server.log
```

主训练日志里的多 rank traceback 是 reward 服务退出后的连锁失败，而不是 7 个独立根因。

## 4. GPU 显存占用但看不到进程的排障方法

本次对话只提供了排查命令，没有在目标设备上执行。若 `nvidia-smi` 的 process 表和普通 `ps` 看不到 GPU 0 占用，可依次执行：

```bash
nvidia-smi -q -d PIDS
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
sudo fuser -v /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm
sudo lsof /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm
```

检查容器或其他 PID namespace：

```bash
docker ps
sudo docker top <container-id>
sudo crictl ps
sudo crictl inspect <container-id>
```

直接扫描所有进程打开的 NVIDIA 设备文件：

```bash
sudo find /proc/[0-9]*/fd -lname '/dev/nvidia*' -printf '%p -> %l\n' 2>/dev/null
```

路径中的 `/proc/<PID>/fd/...` 可以反推出真实 PID，再查看：

```bash
sudo tr '\0' ' ' < /proc/<PID>/cmdline
sudo cat /proc/<PID>/cgroup
sudo nsenter -t <PID> -p -m ps -ef
```

也可以检查 CUDA MPS：

```bash
ps -ef | grep -i mps
echo get_server_list | nvidia-cuda-mps-control
```

若设备显存满但所有 PID/设备句柄均为空，可能是驱动上下文残留。确认该卡没有其他用户任务后才考虑：

```bash
sudo nvidia-smi --gpu-reset -i 0
```

GPU reset 会影响使用同一设备的所有任务；若 reset 被拒绝或设备关联 display/NVLink fabric，需由管理员重启相关持久化服务、驱动或节点。

## 5. Reward model 原始数据和训练逻辑调查

在 `ms-swift` 中确认当前整理最完整的是 no-CoT 数据：

| 类型 | 原始 JSONL | 样本数 |
| --- | --- | ---: |
| Easy | `public/hym/data/final_train/train_easy_nocot_final.jsonl` | 4,390 |
| Hard | `public/hym/data/final_train/train_hard_nocot_final.jsonl` | 4,436 |

补充调查统计：

- CoT hard 可比较样本 8,673 条。
- 人类与 Gemini 判断一致 5,082 条。
- 清理视频缺失与格式异常后得到 4,887 条。
- no-CoT hard 最终保留 4,436 条。

原有脚本属于 curriculum/分阶段训练思路；最终开源训练按用户要求改成从原始 `Qwen3-Omni-30B-A3B-Instruct` 初始化的一次连续 SFT，并将训练和评测完全拆开。公开文档中统一称为 SFT，不使用 “hard-only SFT” 名称。

## 6. VAPref-10K 数据整理

最终训练数据位置：

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/data/VAPref-10K
├── train.jsonl
└── videos/
```

完成内容：

- 从 `train_hard_nocot_final.jsonl` 导出 4,436 条训练记录。
- 只保留训练实际需要的 `messages` 和 `videos` 两个顶层 key。
- 每条消息严格包含 `system`、`user`、`assistant` 三个 role。
- 对话内容与原始 JSONL 逐条一致，没有改写训练文本。
- 视频统一复制到 `videos/`，共 8,872 个唯一文件。
- 文件名使用逻辑源路径 SHA-256 的前 20 位生成稳定名称，避免重名。
- JSONL 中视频路径全部为 `videos/<hash>.mp4` 相对路径，目录整体移动后仍可使用。
- 验证无缺失视频、无重复引用，JSONL 中无 `/inspire`、`/home` 或 `/root` 绝对路径。
- 原始引用视频总大小 37,087,828,742 bytes，约 34.54 GiB；磁盘占用约 35 GiB。
- `train.jsonl` 当前大小 20,695,132 bytes。

提供了可复现整理脚本：

- `rewardmodel/scripts/prepare_vapref.py`：一次性导出、校验和复制数据；已有目标文件大小一致时可复用。
- `rewardmodel/scripts/resolve_dataset.py`：训练启动时把便携相对视频路径解析为临时绝对路径，同时仍只保留 `messages`、`videos` schema。

目录名 VAPref-10K 表示约 10K 个视频；当前是 4,436 个视频对、8,872 个唯一视频，不是 10K 条 pair。

## 7. VA-Judger-Bench 整理与适配

已将：

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/czxs253130297/Qwen3-Omni/VA-Judger-Bench-Release
```

完整复制到：

```text
/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/data/VA-Judger-Bench-Release
```

验证结果：

- 共 2,259 个文件。
- 总大小 13,277,257,596 bytes。
- 源目录与目标目录的相对路径和文件大小映射完全一致。
- easy 400 条、indomain 250 条、outdomain 500 条，共 1,150 个 pair。
- 三个 split 引用的视频全部存在。

评测数据字段为：

```text
case_id
text_prompt
video_1_relative_path
video_2_relative_path
human_preference_answer
reason
```

原 benchmark README 没有声明可再分发许可证。正式开源数据前仍需确认数据和视频的授权，不应把代码许可证自动视为数据许可证。

## 8. `rewardmodel` 开源精简版

### 8.1 文件结构

```text
rewardmodel/
├── scripts/
│   ├── prepare_vapref.py
│   ├── resolve_dataset.py
│   ├── train_sft.sh
│   ├── evaluate.py
│   └── evaluate.sh
├── requirements-train.txt
├── requirements-eval.txt
├── README.md
└── .gitignore
```

### 8.2 连续 SFT

训练特性：

- 从原始 `Qwen/Qwen3-Omni-30B-A3B-Instruct` 初始化。
- 一次连续执行 `swift sft`，中间只保存 checkpoint，不插入 vLLM 评测。
- 8 个 data-parallel worker。
- full-parameter SFT。
- BF16。
- 默认 DeepSpeed ZeRO-3 参数和 optimizer CPU offload。
- gradient checkpointing。
- 冻结视觉/音频 tower、aligner、talker 和 waveform decoder，训练 thinker/LLM。
- microbatch 1，gradient accumulation 16；8 卡 effective global batch 为 128。
- 最多 12 帧、128 video tokens、24,576 sequence length。
- 默认 2,000 steps，每 100 steps 保存，最多保留 20 个 checkpoint。
- 不切 validation split，不进行训练内 eval。
- 额外命令行参数原样传给 `swift sft`，支持 `--resume_from_checkpoint`。
- `VALIDATE_ONLY=1` 可只检查模型/数据并解析路径。

本机前台启动：

```bash
cd /inspire/qb-ilm/project/semantic-visual-tokenizer/czxs253130297/VA-Judger/rewardmodel

MODEL_PATH=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/weight/Qwen3Omni/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/QwenOmni/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695 \
VAPREF_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/data/VAPref-10K \
bash scripts/train_sft.sh
```

本机后台启动：

```bash
nohup env \
MODEL_PATH=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/weight/Qwen3Omni/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/QwenOmni/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695 \
VAPREF_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/data/VAPref-10K \
bash scripts/train_sft.sh > train_sft.log 2>&1 &
```

常用覆盖：

```bash
MAX_STEPS=1000 SAVE_STEPS=50 OUTPUT_DIR=outputs/my_sft bash scripts/train_sft.sh
bash scripts/train_sft.sh --resume_from_checkpoint outputs/sft/checkpoint-500
```

显存仍不足时，优先降低 `MAX_LENGTH`、`VIDEO_MAX_TOKEN_NUM` 或 `FPS_MAX_FRAMES`。显存足够时，可关闭 CPU offload 并使用兼容的 FlashAttention：

```bash
DEEPSPEED=zero3 \
ATTN_IMPL=flash_attn \
PADDING_FREE=true \
bash scripts/train_sft.sh
```

### 8.3 独立评测

`evaluate.py` 已适配新的 VA-Judger-Bench schema，并具有以下行为：

- 使用与训练一致的 no-CoT 五维判断 prompt。
- 每条输入包含 caption 和两个 `<video>`。
- 使用 `swift.infer_engine.VllmEngine`、`qwen3_omni_moe`、TP=1、BF16。
- greedy decoding：`temperature=0`、`top_p=1`、`top_k=-1`。
- 支持 worker/world-size 数据切分和断点续跑。
- 首选解析 `<answer>video 1/2 is better</answer>`，也支持纯文本 fallback。
- 分别输出 `all_cases_accuracy`、`parsed_only_accuracy` 和 `parse_rate`。

`evaluate.sh` 默认让 8 张可见 GPU 各运行一个独立 TP=1 vLLM worker。easy、indomain、outdomain 依次评测，每个 split 内跨 GPU 并行，最后合并 worker 输出并生成各 split 和 overall 汇总。

本机命令：

```bash
MODEL_PATH=outputs/sft/checkpoint-1000 \
BENCH_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/data/VA-Judger-Bench-Release \
bash scripts/evaluate.sh
```

使用两卡评测：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=outputs/sft/checkpoint-1000 \
BENCH_ROOT=/inspire/qb-ilm/project/semantic-visual-tokenizer/public/hym/data/VA-Judger-Bench-Release \
bash scripts/evaluate.sh
```

指标含义：

- `all_cases_accuracy = correct / total`，无法解析和推理失败都算错。
- `parsed_only_accuracy = correct / parsed`。
- `parse_rate = parsed / total`。

## 9. 环境建议

为避免 vLLM 与训练 PyTorch/CUDA 版本相互约束，三个用途建议分环境：

```bash
# LTX-2 RL 与生成
conda create -n ltx2 python=3.11 -y
conda activate ltx2
pip install -r videorl/requirements.txt

# Qwen3-Omni vLLM reward serving / benchmark evaluation
conda create -n vllm python=3.11 -y
conda activate vllm
pip install -r videorl/requirements-reward.txt

# Reward model SFT
conda create -n swift python=3.11 -y
conda activate swift
pip install -r rewardmodel/requirements-train.txt
```

如集群需要特定 CUDA 版本，应先安装对应的 PyTorch wheel，再安装 requirements。系统还需能直接调用 `ffmpeg`。

## 10. 已完成验证与未执行项

已完成：

- `videorl` 和 `rewardmodel` 文件存在性核对。
- Shell 脚本语法检查。
- Python 文件编译检查。
- VAPref-10K 的记录数、字段、role、相对路径和视频存在性检查。
- 新旧训练数据逐条 `messages` 一致性检查。
- VA-Judger-Bench 源/目标完整复制及全部视频引用检查。
- reward-model 训练 `VALIDATE_ONLY=1`，使用本机原始 Qwen3-Omni checkpoint 通过。
- benchmark 评测 `VALIDATE_ONLY=1` 通过。
- 评测 summary 的 synthetic test：2 条总样本、1 条可解析且正确时，all-cases accuracy=0.5、parsed-only accuracy=1.0、parse rate=0.5。
- 开源源码内机器绝对路径扫描；运行所需本地路径通过环境变量传入。

未执行：

- 没有真正启动新的 reward model SFT。
- 没有使用新 checkpoint 跑完整 VA-Judger-Bench。
- 没有启动 32 卡多机训练。
- 本次整理不代表数据和视频的再分发许可证已经确认。

## 11. 推荐后续发布前检查

1. 补齐 Hugging Face 上 `VA-Judger` reward model 的所有 shard，并验证 index 中引用文件齐全。
2. 在目标机器分别运行 `VALIDATE_ONLY=1`，然后做一次短步数 SFT、一次单 prompt LoRA inference 和每个 benchmark split 的少量样本 smoke test。
3. 确认 VAPref-10K、VA-Judger-Bench 及其中视频的公开授权和 attribution。
4. 检查 vendored LTX-2/OmniNFT/ms-swift 代码的 LICENSE、NOTICE 和上游保留要求。
5. 发布前移除日志、临时输出、checkpoint、缓存和任何不应进入 Git 的大型二进制文件。
6. 使用全新目录按 README 从零下载权重并运行一次，以验证开源说明不依赖本机隐式环境。
