"""Minimal VA-Judger LTX-2 GRPO configuration."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import ml_collections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BASE_SPEC = importlib.util.spec_from_file_location("va_judger_base", Path(__file__).with_name("base.py"))
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("could not load config/base.py")
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_MODULE)


TARGET_MODULES = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
    "video_to_audio_attn.to_q",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "video_to_audio_attn.to_out.0",
]


def get_config(name: str) -> ml_collections.ConfigDict:
    return globals()[name]()


def ltx2_qwen3omni_av_quality_split_reward() -> ml_collections.ConfigDict:
    """Three-branch pairwise Qwen3-Omni dimension-score reward."""
    config = _BASE_MODULE.get_config()
    output_dir = os.environ.get("OUTPUT_DIR", "outputs")

    config.mixed_precision = "bf16"
    config.activation_checkpointing = os.environ.get("LTX_ACTIVATION_CHECKPOINTING", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    config.base_model = "ltx"
    config.pretrained.model = os.environ.get(
        "LTX_MODEL_PATH",
        str(PROJECT_ROOT / "weights/LTX-2/ltx-2-19b-dev.safetensors"),
    )
    config.gemma_root = os.environ.get(
        "GEMMA_MODEL_PATH",
        str(PROJECT_ROOT / "weights/LTX-2/gemma"),
    )
    config.train_dataset = os.environ.get(
        "PAIR_VIDEO_PROMPT_TRAIN_DATASET",
        str(PROJECT_ROOT / "JavisBench-mini.csv"),
    )
    config.test_dataset = os.environ.get(
        "PAIR_VIDEO_PROMPT_TEST_DATASET",
        str(PROJECT_ROOT / "dataset/vggsound/test_metadata_arena.jsonl"),
    )
    config.extra_train_jsonls = [
        os.environ.get(
            "PAIR_VIDEO_PROMPT_EXTRA_TRAIN_DATASET",
            str(PROJECT_ROOT / "dataset/vggsound/train_metadata_20k_sample1k_seed42.jsonl"),
        )
    ]
    config.extra_train_repeat = 1
    config.resume_from = os.environ.get("LTX_RESUME_FROM", "")

    config.run_name = f"{output_dir}/logs/va_judger_ltx2"
    config.save_dir = f"{output_dir}/checkpoints/va_judger_ltx2"
    config.save_freq = int(os.environ.get("LTX_SAVE_FREQ", "25"))
    config.eval_freq = int(os.environ.get("LTX_EVAL_FREQ", "50"))

    config.reward_fn = ml_collections.ConfigDict(
        {
            "qwen3omni_pair_overall_reward": float(
                os.environ.get("QWEN3OMNI_OVERALL_REWARD_WEIGHT", "0.3333333333333333")
            ),
            "qwen3omni_pair_audio_reward": float(
                os.environ.get("QWEN3OMNI_AUDIO_REWARD_WEIGHT", "0.3333333333333333")
            ),
            "qwen3omni_pair_video_reward": float(
                os.environ.get("QWEN3OMNI_VIDEO_REWARD_WEIGHT", "0.3333333333333333")
            ),
        }
    )
    config.reward_route = ml_collections.ConfigDict()
    config.reward_route.video_keys = ["qwen3omni_pair_video_reward"]
    config.reward_route.audio_keys = ["qwen3omni_pair_audio_reward"]
    config.reward_route.sync_keys = ["qwen3omni_pair_overall_reward"]

    group_size = int(os.environ.get("QWEN3OMNI_PAIR_GROUP_SIZE", "8"))
    prompt_rank_group_size = int(os.environ.get("QWEN3OMNI_PROMPT_RANK_GROUP_SIZE", "1"))
    if prompt_rank_group_size <= 0 or group_size % prompt_rank_group_size != 0:
        raise ValueError(
            "QWEN3OMNI_PROMPT_RANK_GROUP_SIZE must be positive and divide "
            f"QWEN3OMNI_PAIR_GROUP_SIZE, got {prompt_rank_group_size=} {group_size=}"
        )

    config.sample.num_steps = int(os.environ.get("LTX_ROLLOUT_NUM_STEPS", "20"))
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.noise_level = 0.7
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.prompt_rank_group_size = prompt_rank_group_size
    config.sample.local_prompt_grouping = prompt_rank_group_size == 1
    config.sample.num_image_per_prompt = group_size
    config.sample.train_batch_size = group_size // prompt_rank_group_size
    config.sample.test_batch_size = 1
    config.sample.num_batches_per_epoch = int(os.environ.get("QWEN3OMNI_NUM_BATCHES_PER_EPOCH", "1"))

    config.train.batch_size = int(os.environ.get("QWEN3OMNI_TRAIN_MICRO_BSZ", "1"))
    config.train.gradient_accumulation_steps = int(os.environ.get("QWEN3OMNI_GRAD_ACCUM", "1"))
    config.train.timestep_fraction = float(os.environ.get("LTX_TIMESTEP_FRACTION", "0.4"))
    config.train.learning_rate = float(os.environ.get("LTX_RL_LR", "3e-5"))
    config.train.adv_mode = "branch_aware"
    config.train.ema = False
    config.train.use_fsdp = True
    config.train.cpu_offload_samples = os.environ.get("LTX_TRAIN_SAMPLE_CPU_OFFLOAD", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    config.train.ca_kv_scale_a2v = [
        {"blocks": [0], "dir": "a2v", "scale": 0.0},
        {"blocks": ["1-10"], "dir": "a2v", "scale": 0.1},
        {"blocks": ["40-47"], "dir": "a2v", "scale": 0.3},
    ]
    config.train.attn_sync_weight_max = float(os.environ.get("ATTN_SYNC_WEIGHT_MAX", "1.5"))
    config.train.attn_sync_warmup_steps = int(os.environ.get("ATTN_SYNC_WARMUP_STEPS", "400"))

    config.resolution_height = int(os.environ.get("LTX_RESOLUTION_HEIGHT", "512"))
    config.resolution_width = int(os.environ.get("LTX_RESOLUTION_WIDTH", "768"))
    config.decay_type = 1
    config.beta = 1.0
    config.train.beta = 0.0001
    config.use_lora = True
    config.target_modules = TARGET_MODULES

    if "LTX_FSDP_SHARDING_STRATEGY" in os.environ:
        config.train.fsdp_sharding_strategy = os.environ["LTX_FSDP_SHARDING_STRATEGY"]
        config.train.fsdp_backward_prefetch = os.environ.get("LTX_FSDP_BACKWARD_PREFETCH", "BACKWARD_POST")
        config.train.fsdp_num_replicate = int(os.environ.get("LTX_FSDP_NUM_REPLICATE", "1"))
        config.train.fsdp_num_shard = int(os.environ.get("LTX_FSDP_NUM_SHARD", "1"))
        config.train.fsdp_use_device_mesh = os.environ.get("LTX_FSDP_USE_DEVICE_MESH", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        config.train.fsdp_cpu_offload = os.environ.get("LTX_FSDP_CPU_OFFLOAD", "0").lower() in {
            "1",
            "true",
            "yes",
        }

    return config
