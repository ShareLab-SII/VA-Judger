"""Data-parallel LTX-2 two-stage (super-resolution) inference from a JSONL prompt file.

Stage 1 generates video at half resolution with CFG guidance, then Stage 2 upsamples
2x and refines with distilled LoRA. Each torchrun rank loads one pipeline replica.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio

_LTX_V2_ROOT = Path(__file__).resolve().parents[1] / "ltx_v2"
for _pkg in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
    _src = str(_LTX_V2_ROOT / _pkg / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LTX-2 two-stage data-parallel JSONL inference")
    parser.add_argument("--model-path", required=True, help="Merged LTX-2 checkpoint .safetensors")
    parser.add_argument("--gemma-path", required=True, help="Local Gemma text encoder directory")
    parser.add_argument("--spatial-upsampler-path", required=True, help="Spatial upsampler .safetensors")
    parser.add_argument("--distilled-lora-path", required=True, help="Distilled LoRA for stage 2 refinement")
    parser.add_argument("--distilled-lora-strength", type=float, default=0.8)
    parser.add_argument("--input-jsonl", required=True, help="Input metadata jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for generated mp4/wav files")
    parser.add_argument("--prompt-key", default="prompt_av", help="Prompt field to use")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-sample seed is base + sample index")
    parser.add_argument(
        "--constant-seed",
        action="store_true",
        help="Use exactly --seed for every sample instead of adding the sample index.",
    )
    parser.add_argument("--height", type=int, default=None, help="Final output height (stage 2, divisible by 64)")
    parser.add_argument("--width", type=int, default=None, help="Final output width (stage 2, divisible by 64)")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Stage 1 denoising steps")
    parser.add_argument("--video-guidance-scale", type=float, default=None)
    parser.add_argument("--audio-guidance-scale", type=float, default=None)
    parser.add_argument("--video-stg-scale", type=float, default=None)
    parser.add_argument("--audio-stg-scale", type=float, default=None)
    parser.add_argument("--video-rescale-scale", type=float, default=None)
    parser.add_argument("--audio-rescale-scale", type=float, default=None)
    parser.add_argument("--a2v-guidance-scale", type=float, default=None)
    parser.add_argument("--v2a-guidance-scale", type=float, default=None)
    parser.add_argument("--save-wav", action="store_true", help="Also save a separate wav next to the mp4")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument(
        "--manifest-tag",
        default="",
        help="Optional manifest subdirectory, useful for concurrent index-range jobs.",
    )
    parser.add_argument(
        "--streaming-prefetch-count",
        type=int,
        default=2,
        help="Layer-stream transformer: keep at most 1+N layers on GPU (default: 2).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Max batch per transformer forward; keep 1 for lowest peak VRAM.",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "fp8-cast"),
        default="none",
        help="Weight quantization policy (default: none).",
    )
    return parser.parse_args()


def get_rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def read_jsonl(path: Path, prompt_key: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                item = {"prompt": str(item)}
            prompt = extract_prompt(item, prompt_key)
            if prompt:
                item = dict(item)
                item["_index"] = idx
                item["_prompt"] = prompt
                samples.append(item)
    return samples


def extract_prompt(item: dict[str, Any], prompt_key: str) -> str:
    keys = [prompt_key, "prompt_av", "prompt", "text", "caption", "prompt_v", "prompt_a"]
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def safe_part(text: str, default: str) -> str:
    text = str(text or default)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    text = text.strip("._")
    return text or default


def output_stem(sample: dict[str, Any]) -> str:
    idx = int(sample["_index"])
    set_name = safe_part(sample.get("set", ""), "set")
    sample_id = safe_part(sample.get("sample_id", ""), f"{idx:06d}")
    return f"{idx:06d}_{set_name}_{sample_id}"


def save_wav(path: Path, audio: Audio) -> None:
    wav = audio.waveform
    if wav is None:
        return
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.shape[0] > wav.shape[1]:
        wav = wav.transpose(0, 1)
    torchaudio.save(str(path), wav.detach().cpu(), sample_rate=int(audio.sampling_rate))


def build_guider_params(args: argparse.Namespace, params, modality: str) -> MultiModalGuiderParams:
    if modality == "video":
        guider = params.video_guider_params
        return MultiModalGuiderParams(
            cfg_scale=float(args.video_guidance_scale if args.video_guidance_scale is not None else guider.cfg_scale),
            stg_scale=float(args.video_stg_scale if args.video_stg_scale is not None else guider.stg_scale),
            rescale_scale=float(args.video_rescale_scale if args.video_rescale_scale is not None else guider.rescale_scale),
            modality_scale=float(args.a2v_guidance_scale if args.a2v_guidance_scale is not None else guider.modality_scale),
            skip_step=int(guider.skip_step),
            stg_blocks=list(guider.stg_blocks),
        )
    guider = params.audio_guider_params
    return MultiModalGuiderParams(
        cfg_scale=float(args.audio_guidance_scale if args.audio_guidance_scale is not None else guider.cfg_scale),
        stg_scale=float(args.audio_stg_scale if args.audio_stg_scale is not None else guider.stg_scale),
        rescale_scale=float(args.audio_rescale_scale if args.audio_rescale_scale is not None else guider.rescale_scale),
        modality_scale=float(args.v2a_guidance_scale if args.v2a_guidance_scale is not None else guider.modality_scale),
        skip_step=int(guider.skip_step),
        stg_blocks=list(guider.stg_blocks),
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = get_rank_info()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - rank {rank} - %(levelname)s - %(message)s",
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_tag = safe_part(args.manifest_tag, "") if args.manifest_tag else ""
    result_dir = output_dir / "manifests"
    if manifest_tag:
        result_dir = result_dir / manifest_tag
    result_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(input_path, args.prompt_key)
    selected = samples[args.start_index : args.end_index]
    rank_samples = [sample for i, sample in enumerate(selected) if i % world_size == rank]

    params = detect_params(args.model_path)
    height = int(args.height or params.stage_2_height)
    width = int(args.width or params.stage_2_width)
    num_frames = int(args.num_frames or params.num_frames)
    frame_rate = float(args.frame_rate or params.frame_rate)
    num_inference_steps = int(args.num_inference_steps or params.num_inference_steps)

    quantization = None if args.quantization == "none" else QuantizationPolicy.fp8_cast()

    logging.info(
        "world_size=%s local_rank=%s total=%s selected=%s this_rank=%s output=%s "
        "two_stage final_res=%sx%s stage1_res=%sx%s steps=%s "
        "streaming_prefetch=%s max_batch_size=%s quantization=%s",
        world_size,
        local_rank,
        len(samples),
        len(selected),
        len(rank_samples),
        output_dir,
        height,
        width,
        height // 2,
        width // 2,
        num_inference_steps,
        args.streaming_prefetch_count,
        args.max_batch_size,
        args.quantization,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=args.distilled_lora_path,
            strength=float(args.distilled_lora_strength),
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    logging.info("loading two-stage pipeline: %s", args.model_path)
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.model_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_path,
        loras=(),
        device=device,
        quantization=quantization,
    )

    video_guider_params = build_guider_params(args, params, "video")
    audio_guider_params = build_guider_params(args, params, "audio")
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    manifest_path = result_dir / f"rank_{rank:03d}.jsonl"
    with manifest_path.open("a", encoding="utf-8") as mf:
        for pos, sample in enumerate(rank_samples, start=1):
            stem = output_stem(sample)
            video_path = output_dir / f"{stem}.mp4"
            wav_path = output_dir / f"{stem}.wav"
            seed = int(args.seed if args.constant_seed else args.seed + int(sample["_index"]))

            record = {
                "index": int(sample["_index"]),
                "sample_id": sample.get("sample_id"),
                "set": sample.get("set"),
                "prompt_key": args.prompt_key,
                "seed": seed,
                "video_path": str(video_path),
                "pipeline": "two_stage",
                "final_height": height,
                "final_width": width,
            }

            if video_path.exists() and not args.overwrite:
                record["status"] = "skipped_exists"
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                mf.flush()
                logging.info("skip existing %s (%s/%s)", video_path.name, pos, len(rank_samples))
                continue

            try:
                logging.info("generating %s (%s/%s)", video_path.name, pos, len(rank_samples))
                # The pipeline returns a lazy video iterator, so keep inference mode
                # active through encode_video() as well as the initial pipeline call.
                with torch.inference_mode():
                    video, audio = pipeline(
                        prompt=sample["_prompt"],
                        negative_prompt=args.negative_prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        frame_rate=frame_rate,
                        num_inference_steps=num_inference_steps,
                        video_guider_params=video_guider_params,
                        audio_guider_params=audio_guider_params,
                        images=[],
                        tiling_config=tiling_config,
                        streaming_prefetch_count=args.streaming_prefetch_count,
                        max_batch_size=args.max_batch_size,
                    )
                    encode_video(
                        video=video,
                        fps=int(round(frame_rate)),
                        audio=audio,
                        output_path=str(video_path),
                        video_chunks_number=video_chunks_number,
                    )
                    if args.save_wav and audio is not None:
                        save_wav(wav_path, audio)
                        record["wav_path"] = str(wav_path)
                record["status"] = "ok"
            except Exception as exc:
                logging.exception("failed %s", stem)
                record["status"] = "failed"
                record["error"] = repr(exc)
            finally:
                cleanup_memory()

            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            mf.flush()

    logging.info("done")


if __name__ == "__main__":
    main()
