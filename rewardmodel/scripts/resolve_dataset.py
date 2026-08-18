#!/usr/bin/env python3
"""Resolve portable dataset video paths for ms-swift without changing its schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"messages", "videos"}:
                raise ValueError(f"{source}:{line_no}: expected only messages and videos")
            videos = row.get("videos")
            if not isinstance(videos, list) or len(videos) != 2:
                raise ValueError(f"{source}:{line_no}: expected exactly two videos")
            resolved = []
            for value in videos:
                path = Path(value)
                if not path.is_absolute():
                    path = source.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"{source}:{line_no}: missing video {path}")
                resolved.append(str(path))
            rows.append(json.dumps({"messages": row["messages"], "videos": resolved}, ensure_ascii=False) + "\n")

    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text("".join(rows), encoding="utf-8")
    temp.replace(destination)
    print(f"Resolved {len(rows)} rows: {source} -> {destination}")


if __name__ == "__main__":
    main()
