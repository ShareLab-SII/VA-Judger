#!/usr/bin/env python3
"""Run VA-Judger inference and score one VA-Judger-Bench split."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


SYSTEM_PROMPT = """Given a caption and two generated audio/video files based on this caption, evaluate which video is better.

Compare the two videos on the following fixed dimensions:

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

ANSWER_RE = re.compile(r"<answer>\s*(video\s+[12]\s+is\s+better)\s*</answer>", re.I | re.S)
CHOICE_RE = re.compile(r"video\s+([12])\s+is\s+better", re.I)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "case_id",
                "text_prompt",
                "video_1_relative_path",
                "video_2_relative_path",
                "human_preference_answer",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            yield row


def normalize_choice(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = CHOICE_RE.search(value.strip())
    return int(match.group(1)) if match else None


def parse_completion(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    tagged = list(ANSWER_RE.finditer(value))
    if tagged:
        return normalize_choice(tagged[-1].group(1))
    return normalize_choice(value)


def response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices and getattr(choices[0], "message", None) is not None:
        content = choices[0].message.content
        return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row.get("case_id")) for row in iter_result_rows(path) if row.get("case_id") is not None}


def iter_result_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def score(results: Path, summary_path: Path) -> dict:
    total = parsed = correct = errors = 0
    for row in iter_result_rows(results):
        total += 1
        errors += int(bool(row.get("error")))
        gold = normalize_choice(row.get("gold"))
        prediction = parse_completion(row.get("completion"))
        if prediction is not None and gold is not None:
            parsed += 1
            correct += int(prediction == gold)
    summary = {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "errors": errors,
        "all_cases_accuracy": correct / total if total else None,
        "parsed_only_accuracy": correct / parsed if parsed else None,
        "parse_rate": parsed / total if total else None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    results = args.results.expanduser().resolve()
    summary = args.summary.expanduser().resolve()
    if args.eval_only:
        score(results, summary)
        return
    if args.model is None or args.dataset is None or args.split_root is None:
        parser.error("--model, --dataset, and --split-root are required for inference")
    if args.worker < 0 or args.worker >= args.world_size:
        parser.error("--worker must be in [0, --world-size)")

    model = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    split_root = args.split_root.expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"model directory not found: {model}")
    rows = [row for index, row in enumerate(iter_jsonl(dataset)) if index % args.world_size == args.worker]
    done = completed_ids(results) if args.resume else set()
    rows = [row for row in rows if str(row["case_id"]) not in done]
    results.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("USE_AUDIO_IN_VIDEO", "true")
    os.environ.setdefault("ENABLE_AUDIO_OUTPUT", "0")
    os.environ.setdefault("MAX_PIXELS", "1003520")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "602112")

    import torch
    from swift import InferStats, RequestConfig
    from swift.infer_engine import InferRequest, VllmEngine

    engine = VllmEngine(
        str(model),
        model_type="qwen3_omni_moe",
        torch_dtype=torch.bfloat16,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"video": 2, "audio": 2},
        seed=42,
    )
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=42,
        stream=False,
    )
    metric = InferStats()
    mode = "a" if args.resume and results.is_file() else "w"
    with results.open(mode, encoding="utf-8") as output:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            requests = []
            records = []
            for row in batch:
                video_1 = (split_root / row["video_1_relative_path"]).resolve()
                video_2 = (split_root / row["video_2_relative_path"]).resolve()
                if not video_1.is_file() or not video_2.is_file():
                    raise FileNotFoundError(f"missing benchmark videos for {row['case_id']}")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Text Caption: {row['text_prompt']}\n\n"
                            "Audio/video 1:\n<video>\n\nAudio/video 2:\n<video>"
                        ),
                    },
                ]
                requests.append(InferRequest(messages=messages, videos=[str(video_1), str(video_2)]))
                records.append(
                    {
                        "case_id": row["case_id"],
                        "gold": row["human_preference_answer"],
                    }
                )
            try:
                responses = engine.infer(requests, request_config, metrics=[metric], use_tqdm=False)
                for record, response in zip(records, responses):
                    record["completion"] = response_text(response)
            except Exception as error:
                for record in records:
                    record["error"] = repr(error)
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"worker {args.worker}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
