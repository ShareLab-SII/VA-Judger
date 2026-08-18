#!/usr/bin/env python3
"""Export the SFT subset to a portable messages/videos-only JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def resolve_video(value: str, data_root: Path, train_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("pair_video/"):
        return data_root / path
    train_candidate = train_root / path
    if train_candidate.is_file():
        return train_candidate
    return data_root / path


def portable_name(source: Path, data_root: Path) -> str:
    try:
        logical_path = source.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        logical_path = source.name
    digest = hashlib.sha256(logical_path.encode("utf-8")).hexdigest()[:20]
    return f"{digest}{source.suffix.lower()}"


def validate_messages(messages: object, line_no: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_no}: messages must be a non-empty list")
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"line {line_no}: invalid message")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"line {line_no}: invalid role/content")
        cleaned.append({"role": role, "content": content})
    if [message["role"] for message in cleaned] != ["system", "user", "assistant"]:
        raise ValueError(f"line {line_no}: expected system/user/assistant messages")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    source_jsonl = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    train_root = data_root / "final_train"
    videos_root = output_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    output_rows: list[str] = []
    destinations: dict[str, Path] = {}
    copied = reused = 0
    total_bytes = 0

    for line_no, row in iter_jsonl(source_jsonl):
        messages = validate_messages(row.get("messages"), line_no)
        videos = row.get("videos")
        if not isinstance(videos, list) or len(videos) != 2 or not all(isinstance(v, str) for v in videos):
            raise ValueError(f"line {line_no}: videos must contain exactly two paths")

        portable_videos = []
        for value in videos:
            source = resolve_video(value, data_root, train_root).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"line {line_no}: missing video: {source}")
            name = portable_name(source, data_root)
            previous = destinations.get(name)
            if previous is not None and previous != source:
                raise RuntimeError(f"hash collision: {previous} and {source} -> {name}")
            destinations[name] = source
            destination = videos_root / name
            source_size = source.stat().st_size
            if destination.is_file() and destination.stat().st_size == source_size:
                reused += 1
            else:
                shutil.copy2(source, destination)
                copied += 1
            total_bytes += source_size
            portable_videos.append(f"videos/{name}")

        output_rows.append(
            json.dumps({"messages": messages, "videos": portable_videos}, ensure_ascii=False) + "\n"
        )

    temp_jsonl = output_root / "train.jsonl.tmp"
    temp_jsonl.write_text("".join(output_rows), encoding="utf-8")
    temp_jsonl.replace(output_root / "train.jsonl")
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "video_references": len(output_rows) * 2,
                "unique_videos": len(destinations),
                "copied": copied,
                "reused": reused,
                "bytes": total_bytes,
                "output": str(output_root / "train.jsonl"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
