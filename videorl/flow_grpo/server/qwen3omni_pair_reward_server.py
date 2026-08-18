from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ms_swift_root = os.environ.get("MS_SWIFT_ROOT", "").strip()
if _ms_swift_root:
    MS_SWIFT_ROOT = Path(_ms_swift_root).expanduser().resolve()
    MS_SWIFT_QWEN_DIR = MS_SWIFT_ROOT / "examples/train/grpo/qwen3_omni_multitalker"
    for path in (str(MS_SWIFT_ROOT), str(MS_SWIFT_QWEN_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

try:
    from infer_pair_video_batch import MULTITALKER_SYSTEM_PROMPT, USER_CONTENT_TEMPLATE
except Exception:
    MULTITALKER_SYSTEM_PROMPT = """Given a caption and two generated audio/video files based on this caption, evaluate which video is better.

Compare the two videos on the following fixed dimensions:

(A) Better match to the prompt (characters, content, or speech)
(B) Better audio-visual consistency (e.g., lip sync, sound matches the image)
(C) Higher audio quality
(D) Higher video quality
(E) More complete and coherent content

Finally, in the <answer> tag, output exactly one of the following strings based on the total scores:
video 1 is better
or
video 2 is better

No additional text or quotation marks are allowed in the <answer> section. Write the answer tag on one line, for example:
<answer>video 1 is better</answer>"""
    USER_CONTENT_TEMPLATE = """Text Caption: {caption}

Video 1:
<video>

Video 2:
<video>"""

DIMENSION_SCORE_SYSTEM_PROMPT = """Given a caption and two generated audio/video files based on this caption, evaluate which video is better.

compare the two videos on the following fixed dimensions:

(A) Better match to the prompt (characters, content, or speech)
(B) Better audio-visual consistency (e.g., lip sync, sound matches the image)
(C) Higher audio quality
(D) Higher video quality
(E) More complete and coherent content

For each dimension, provide:
- A score between 1-10 for both videos (e.g., video 1: 8/10, video 2: 6/10)
- A concise rationale explaining the score

Then compute the total score for each video by summing all dimension scores.

Use the following output format exactly:

<think>
(A) Better match to the prompt: video 1 (.../10) - ...; video 2 (.../10) - ...
(B) Audio-visual consistency: video 1 (.../10) - ...; video 2 (.../10) - ...
(C) Audio quality: video 1 (.../10) - ...; video 2 (.../10) - ...
(D) Video quality: video 1 (.../10) - ...; video 2 (.../10) - ...
(E) Completeness and coherence: video 1 (.../10) - ...; video 2 (.../10) - ...

Total score:
video 1: ... = ...
video 2: ... = ...
</think>

Finally, in the <answer> tag, output exactly one of the following strings based on the total scores:
video 1 is better
or
video 2 is better

No additional text or quotation marks are allowed in the <answer> section. Write the answer tag on one line, for example:
<answer>video 1 is better</answer>"""

AV_QUALITY_SPLIT_SYSTEM_PROMPT = """Given a caption and two generated audio/video files based on this caption, compare which video is better.

Use the following fixed dimensions:

(A) Better match to the prompt (characters, content, or speech)
(B) Better audio-visual consistency (e.g., lip sync, sound matches the image)
(C) Higher audio quality
(D) Higher video quality
(E) More complete and coherent content

Return three independent pairwise decisions:

1. overall: compare only dimensions (A), (B), and (E), with equal weight across those three dimensions.
2. audio: compare only dimension (C).
3. video: compare only dimension (D).

For each decision, output exactly one of:
video 1 is better
or
video 2 is better

Use this exact final format:
<overall>video 1 is better</overall>
<audio>video 1 is better</audio>
<video>video 1 is better</video>

No additional text is allowed after these tags."""

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_CHOICE_RE = re.compile(r"video\s+([12])\s+is\s+better", re.IGNORECASE)
_DIMENSION_TAGS = ("overall", "audio", "video")
_SCORE_DIMENSIONS = ("A", "B", "C", "D", "E")
_SCORE_DIMENSION_RE = re.compile(r"\(\s*([A-E])\s*\)", re.IGNORECASE)
_VIDEO_SCORE_RE_TEMPLATE = r"video\s*{video}\s*[:(]?\s*(\d+(?:\.\d+)?)\s*/\s*10"


class PairItem(BaseModel):
    prompt: str
    video_1: str
    video_2: str
    id: str | None = None
    group_index: int | None = None
    group_size: int | None = None
    group_pair_count: int | None = None
    pair_index_in_group: int | None = None
    sample_index_1: int | None = None
    sample_index_2: int | None = None


class PredictRequest(BaseModel):
    pairs: list[PairItem]


def parse_choice(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        inner = re.sub(r"\s+", " ", matches[-1].group(1).strip().lower())
        if inner in ("1", "video 1 is better"):
            return 1
        if inner in ("2", "video 2 is better"):
            return 2
        choice = _CHOICE_RE.search(inner)
        if choice:
            return int(choice.group(1))
        return None
    raw = text.strip().lower()
    choice = _CHOICE_RE.search(raw)
    if choice:
        return int(choice.group(1))
    for line in reversed(raw.splitlines()[-12:]):
        line = re.sub(r"\s+", " ", line.strip())
        if line == "video 1 is better":
            return 1
        if line == "video 2 is better":
            return 2
    return None


def parse_dimension_choices(text: object) -> dict[str, int | None]:
    choices: dict[str, int | None] = {}
    if not isinstance(text, str):
        return {tag: None for tag in _DIMENSION_TAGS}
    for tag in _DIMENSION_TAGS:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
        choices[tag] = parse_choice(match.group(1)) if match else None
    return choices


def parse_dimension_scores(text: object) -> dict[str, dict[str, float]] | None:
    if not isinstance(text, str):
        return None
    matches = list(_SCORE_DIMENSION_RE.finditer(text))
    scores: dict[str, dict[str, float]] = {}
    for index, match in enumerate(matches):
        dimension = match.group(1).upper()
        if dimension not in _SCORE_DIMENSIONS or dimension in scores:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end():end]
        dimension_scores: dict[str, float] = {}
        for video_id in ("1", "2"):
            score_match = re.search(
                _VIDEO_SCORE_RE_TEMPLATE.format(video=video_id),
                segment,
                flags=re.IGNORECASE,
            )
            if score_match is None:
                return None
            value = float(score_match.group(1))
            if not 0.0 <= value <= 10.0:
                return None
            dimension_scores[video_id] = value
        scores[dimension] = dimension_scores
    if any(dimension not in scores for dimension in _SCORE_DIMENSIONS):
        return None
    return scores


def response_text(resp: Any) -> str:
    if getattr(resp, "choices", None):
        choice = resp.choices[0]
        if getattr(choice, "message", None) is not None:
            content = choice.message.content
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype={name}")
    return mapping[key]


class Qwen3OmniPairRewardService:
    def __init__(self, args: argparse.Namespace):
        os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "0")
        os.environ.setdefault("USE_AUDIO_IN_VIDEO", "true")
        os.environ.setdefault("MAX_PIXELS", args.max_pixels)
        os.environ.setdefault("VIDEO_MAX_PIXELS", args.video_max_pixels)

        from swift import RequestConfig
        from swift.infer_engine import InferRequest, VllmEngine

        self.InferRequest = InferRequest
        self.RequestConfig = RequestConfig
        limit_mm = json.loads(args.vllm_limit_mm_per_prompt) if args.vllm_limit_mm_per_prompt else None
        self.engine = VllmEngine(
            args.model,
            model_type=args.model_type,
            torch_dtype=parse_torch_dtype(args.torch_dtype),
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            limit_mm_per_prompt=limit_mm,
            enforce_eager=args.vllm_enforce_eager,
            seed=args.seed,
        )
        self.request_config = RequestConfig(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            stream=False,
        )
        self.lock = threading.Lock()
        self.dump_infer = args.dump_infer
        self.dump_path = Path(args.dump_path).expanduser() if args.dump_path else None
        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        self._request_seq = 0
        self.prompt_mode = args.prompt_mode

    def _build_messages(self, pair: PairItem) -> list[dict[str, str]]:
        if self.prompt_mode == "dimension_scores":
            system_prompt = DIMENSION_SCORE_SYSTEM_PROMPT
        elif self.prompt_mode == "av_quality_split":
            system_prompt = AV_QUALITY_SPLIT_SYSTEM_PROMPT
        else:
            system_prompt = MULTITALKER_SYSTEM_PROMPT
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_CONTENT_TEMPLATE.format(caption=pair.prompt)},
        ]

    def _build_request(self, pair: PairItem, messages: list[dict[str, str]]):
        return self.InferRequest(messages=messages, videos=[pair.video_1, pair.video_2])

    def _dump_record(self, record: dict[str, Any]) -> None:
        if not self.dump_infer:
            return
        line = json.dumps(record, ensure_ascii=False)
        print(f"QWEN3OMNI_INFER_DUMP {line}", flush=True)
        if self.dump_path is not None:
            with self.dump_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _record_from_pair(self, pair: PairItem, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "id": pair.id,
            "group_index": pair.group_index,
            "group_size": pair.group_size,
            "group_pair_count": pair.group_pair_count,
            "pair_index_in_group": pair.pair_index_in_group,
            "sample_index_1": pair.sample_index_1,
            "sample_index_2": pair.sample_index_2,
            "prompt": pair.prompt,
            "messages": messages,
            "video_1": pair.video_1,
            "video_2": pair.video_2,
        }

    def _next_request_seq(self) -> int:
        self._request_seq += 1
        return self._request_seq

    def predict(self, pairs: list[PairItem]) -> list[dict[str, Any]]:
        requests = []
        records: list[dict[str, Any]] = []
        for pair in pairs:
            if not Path(pair.video_1).is_file():
                raise FileNotFoundError(f"video_1 not found: {pair.video_1}")
            if not Path(pair.video_2).is_file():
                raise FileNotFoundError(f"video_2 not found: {pair.video_2}")
            messages = self._build_messages(pair)
            requests.append(self._build_request(pair, messages))
            records.append(self._record_from_pair(pair, messages))

        with self.lock:
            request_seq = self._next_request_seq()
            responses = self.engine.infer(requests, self.request_config, use_tqdm=False)

        out = []
        for batch_index, (rec, resp) in enumerate(zip(records, responses, strict=True)):
            text = response_text(resp)
            dimension_scores = None
            if self.prompt_mode == "dimension_scores":
                dimension_scores = parse_dimension_scores(text)
                choice = parse_choice(text)
                choices = {}
                component_scores = {}
                if dimension_scores is not None:
                    normalized = {
                        "overall": {
                            video_id: sum(dimension_scores[dimension][video_id] for dimension in ("A", "B", "E"))
                            / 30.0
                            for video_id in ("1", "2")
                        },
                        "audio": {video_id: dimension_scores["C"][video_id] / 10.0 for video_id in ("1", "2")},
                        "video": {video_id: dimension_scores["D"][video_id] / 10.0 for video_id in ("1", "2")},
                    }
                    for name, scores in normalized.items():
                        is_tie = scores["1"] == scores["2"]
                        component_scores[name] = {
                            "score_1": scores["1"],
                            "score_2": scores["2"],
                            "tie": is_tie,
                        }
                        choices[name] = 0 if is_tie else (1 if scores["1"] > scores["2"] else 2)
                else:
                    component_scores = {
                        name: {"score_1": 0.5, "score_2": 0.5, "tie": True}
                        for name in _DIMENSION_TAGS
                    }
            elif self.prompt_mode == "av_quality_split":
                choices = parse_dimension_choices(text)
                choice = choices.get("overall") or parse_choice(text)
                component_scores = {}
                for name, component_choice in choices.items():
                    if component_choice == 1:
                        component_scores[name] = {"score_1": 1.0, "score_2": 0.0, "tie": False}
                    elif component_choice == 2:
                        component_scores[name] = {"score_1": 0.0, "score_2": 1.0, "tie": False}
                    else:
                        component_scores[name] = {"score_1": 0.5, "score_2": 0.5, "tie": True}
            else:
                choices = {}
                choice = parse_choice(text)
                component_scores = {}
            if choice == 1:
                score_1, score_2, tie = 1.0, 0.0, False
            elif choice == 2:
                score_1, score_2, tie = 0.0, 1.0, False
            else:
                score_1, score_2, tie = 0.5, 0.5, True
            rec.update(
                {
                    "choice": choice or 0,
                    "score_1": score_1,
                    "score_2": score_2,
                    "tie": tie,
                    "choices": choices,
                    "component_scores": component_scores,
                    "dimension_scores": dimension_scores,
                    "completion": text,
                }
            )
            self._dump_record(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_seq": request_seq,
                    "batch_index": batch_index,
                    **rec,
                }
            )
            out.append(rec)
        return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-Omni pairwise video reward server")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "QWEN3OMNI_REWARD_MODEL",
            str(PROJECT_ROOT / "weights/VA-Judger/VA-Judger"),
        ),
    )
    parser.add_argument("--model_type", default=os.environ.get("QWEN3OMNI_MODEL_TYPE", "qwen3_omni_moe"))
    parser.add_argument("--host", default=os.environ.get("QWEN3OMNI_REWARD_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN3OMNI_REWARD_PORT", "8100")))
    parser.add_argument("--torch_dtype", default=os.environ.get("QWEN3OMNI_REWARD_DTYPE", "bfloat16"))
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9")))
    parser.add_argument("--vllm_max_model_len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "24576")))
    parser.add_argument("--vllm_max_num_seqs", type=int, default=int(os.environ.get("VLLM_MAX_NUM_SEQS", "8")))
    parser.add_argument("--vllm_limit_mm_per_prompt", default=os.environ.get("VLLM_LIMIT_MM_PER_PROMPT", '{"video": 2, "audio": 2}'))
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=int(os.environ.get("QWEN3OMNI_MAX_NEW_TOKENS", "2048")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("QWEN3OMNI_TEMPERATURE", "0.75")))
    parser.add_argument("--top_p", type=float, default=float(os.environ.get("QWEN3OMNI_TOP_P", "0.92")))
    parser.add_argument("--top_k", type=int, default=int(os.environ.get("QWEN3OMNI_TOP_K", "32")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("QWEN3OMNI_SEED", "42")))
    parser.add_argument(
        "--prompt_mode",
        choices=("default", "dimension_scores", "av_quality_split"),
        default=os.environ.get("QWEN3OMNI_REWARD_PROMPT_MODE", "default"),
    )
    parser.add_argument("--max_pixels", default=os.environ.get("MAX_PIXELS", "1003520"))
    parser.add_argument("--video_max_pixels", default=os.environ.get("VIDEO_MAX_PIXELS", "602112"))
    parser.add_argument("--run_id", default=os.environ.get("QWEN3OMNI_REWARD_RUN_ID", ""))
    parser.add_argument(
        "--dump_infer",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("QWEN3OMNI_DUMP_INFER", "1").lower() not in ("0", "false", "no"),
        help="Dump full Qwen3-Omni infer prompts and completions as JSONL.",
    )
    parser.add_argument(
        "--dump_path",
        default=os.environ.get("QWEN3OMNI_INFER_DUMP_PATH", ""),
        help="Optional JSONL path for full Qwen3-Omni infer prompt/completion dumps.",
    )
    return parser


app = FastAPI(title="Qwen3-Omni Pairwise Reward API")
model_service: Qwen3OmniPairRewardService | None = None
server_args: argparse.Namespace | None = None


@app.post("/predict")
async def predict(request: PredictRequest):
    if not request.pairs:
        return {"status": "success", "results": []}
    if model_service is None:
        raise HTTPException(status_code=503, detail="reward model is not initialized")
    try:
        return {"status": "success", "results": model_service.predict(request.pairs)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.get("/health")
async def health():
    if server_args is None:
        raise HTTPException(status_code=503, detail="reward model is not initialized")
    return {
        "status": "healthy",
        "model": server_args.model,
        "run_id": server_args.run_id,
        "pid": os.getpid(),
        "gpu": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }


def main() -> None:
    global model_service, server_args

    mp.freeze_support()
    args = build_argparser().parse_args()
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"Qwen3-Omni reward model directory not found: {args.model}")

    server_args = args
    model_service = Qwen3OmniPairRewardService(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
