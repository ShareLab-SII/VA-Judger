from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from typing import Any

import requests
import torch.distributed as dist


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_CHOICE_RE = re.compile(r"video\s+([12])\s+is\s+better", re.IGNORECASE)
_COMPONENTS = ("overall", "audio", "video")
_EXPECTED_GROUP_SIZE_KEY = "_qwen3omni_expected_group_size"
_SPLIT_CACHE: dict[tuple[Any, ...], dict[str, list[float]]] = {}
_RANK_GROUP_CACHE: dict[tuple[int, int], tuple[Any, int]] = {}


def _reward_base_urls() -> list[str]:
    raw_urls = os.environ.get("QWEN3OMNI_REWARD_URLS", "")
    if raw_urls.strip():
        urls = [item.strip().rstrip("/") for item in raw_urls.split(",") if item.strip()]
    else:
        host = os.environ.get("QWEN3OMNI_REWARD_SERVER", "127.0.0.1")
        port = os.environ.get("QWEN3OMNI_REWARD_PORT", "8100")
        urls = [f"http://{host}:{port}"]
    if not urls:
        raise ValueError("QWEN3OMNI_REWARD_URLS does not contain any usable URL")
    return [url[:-8] if url.endswith("/predict") else url for url in urls]


def _distributed_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _prompt_rank_group_size() -> int:
    return int(os.environ.get("QWEN3OMNI_PROMPT_RANK_GROUP_SIZE", "1"))


def _reward_server_index() -> int:
    rank_group_size = max(_prompt_rank_group_size(), 1)
    return _distributed_rank() // rank_group_size


def parse_qwen3omni_choice(text: object) -> int | None:
    """Return 1/2 for a parsed choice, or None for an unparsed/tie answer."""
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


def _post_pairs(pairs: list[dict[str, Any]], timeout: int, max_retries: int) -> list[dict[str, Any]]:
    base_urls = _reward_base_urls()
    start_index = _reward_server_index() % len(base_urls)

    last_err: str | None = None
    attempt_count = max(max_retries + 1, len(base_urls))
    for attempt in range(attempt_count):
        base_url = base_urls[(start_index + attempt) % len(base_urls)]
        url = f"{base_url}/predict"
        try:
            resp = requests.post(url, json={"pairs": pairs}, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    results = data.get("results", [])
                    if len(results) != len(pairs):
                        raise RuntimeError(f"server returned {len(results)} results for {len(pairs)} pairs")
                    return results
                last_err = f"url={url}, response={data}"
            else:
                last_err = f"url={url}, http={resp.status_code}, resp={resp.text[:500]}"
        except Exception as exc:
            last_err = f"url={url}, error={exc!r}"
        if attempt + 1 < attempt_count:
            time.sleep(2)
    raise RuntimeError(f"Qwen3-Omni reward request failed after {attempt_count} attempts: {last_err}")


def _normalise_inputs(images, prompts, metadata):
    if not isinstance(images, (list, tuple)):
        images = [images]
    if not isinstance(prompts, (list, tuple)):
        prompts = [prompts] * len(images)
    if metadata is None:
        metadata = [{} for _ in images]
    elif isinstance(metadata, dict):
        metadata = [metadata for _ in images]
    elif len(metadata) == 1 and len(images) > 1:
        metadata = [metadata[0] for _ in images]

    if len(images) != len(prompts) or len(images) != len(metadata):
        raise ValueError(
            f"qwen3omni_pair_reward length mismatch: images={len(images)}, "
            f"prompts={len(prompts)}, metadata={len(metadata)}"
        )
    return list(images), list(prompts), list(metadata)


def _prompt_for_item(prompt: str, md: Any) -> str:
    if isinstance(md, dict):
        for key in ("prompt_av", "caption_en", "prompt", "prompt_v"):
            value = md.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(prompt).strip()


def _component_score(result: dict[str, Any], component: str, side: str) -> float:
    component_scores = result.get("component_scores")
    if isinstance(component_scores, dict):
        scores = component_scores.get(component)
        if isinstance(scores, dict):
            value = scores.get(side)
            if value is not None:
                return float(value)

    if component == "overall":
        value = result.get(side)
        if value is not None:
            return float(value)
        choice = result.get("choice")
    else:
        choices = result.get("choices")
        choice = choices.get(component) if isinstance(choices, dict) else None

    if choice == 1:
        return 1.0 if side == "score_1" else 0.0
    if choice == 2:
        return 0.0 if side == "score_1" else 1.0
    return 0.5


def _score_components_local(images, prompts, metadata) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Run pairwise Qwen3-Omni once and return win-rate scores per component."""
    timeout = int(os.environ.get("QWEN3OMNI_REWARD_TIMEOUT", "900"))
    max_retries = int(os.environ.get("QWEN3OMNI_REWARD_RETRIES", "2"))
    batch_pairs = int(os.environ.get("QWEN3OMNI_REWARD_PAIR_BATCH", "8"))

    images, prompts, metadata = _normalise_inputs(images, prompts, metadata)
    prompt_texts = [_prompt_for_item(prompts[i], metadata[i]) for i in range(len(images))]

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, prompt in enumerate(prompt_texts):
        groups[prompt].append(idx)

    wins = {component: [0.0 for _ in images] for component in _COMPONENTS}
    counts = [0 for _ in images]
    pair_payloads: list[dict[str, Any]] = []
    pair_indices: list[tuple[int, int]] = []

    for group_idx, (prompt, idxs) in enumerate(groups.items()):
        if len(idxs) < 2:
            continue
        pair_count = len(idxs) * (len(idxs) - 1) // 2
        pair_idx = 0
        for pos_i, i in enumerate(idxs):
            for j in idxs[pos_i + 1 :]:
                pair_payloads.append(
                    {
                        "id": f"group{group_idx:04d}_pair{pair_idx:04d}_{i}_{j}",
                        "prompt": prompt,
                        "video_1": str(images[i]),
                        "video_2": str(images[j]),
                        "group_index": group_idx,
                        "group_size": len(idxs),
                        "group_pair_count": pair_count,
                        "pair_index_in_group": pair_idx,
                        "sample_index_1": i,
                        "sample_index_2": j,
                    }
                )
                pair_indices.append((i, j))
                pair_idx += 1

    for start in range(0, len(pair_payloads), batch_pairs):
        payload_chunk = pair_payloads[start : start + batch_pairs]
        index_chunk = pair_indices[start : start + batch_pairs]
        results = _post_pairs(payload_chunk, timeout=timeout, max_retries=max_retries)
        for (i, j), result in zip(index_chunk, results, strict=True):
            for component in _COMPONENTS:
                wins[component][i] += _component_score(result, component, "score_1")
                wins[component][j] += _component_score(result, component, "score_2")
            counts[i] += 1
            counts[j] += 1

    scores = {
        component: [(wins[component][i] / counts[i]) if counts[i] else 0.5 for i in range(len(images))]
        for component in _COMPONENTS
    }
    details = {
        "qwen3omni_pair_counts": counts,
        "qwen3omni_pair_wins": wins["overall"],
        "qwen3omni_pair_wins_overall": wins["overall"],
        "qwen3omni_pair_wins_audio": wins["audio"],
        "qwen3omni_pair_wins_video": wins["video"],
    }
    return scores, details


def _get_rank_group(rank_group_size: int):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("cross-rank Qwen3-Omni grouping requires initialized torch.distributed")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size % rank_group_size != 0:
        raise ValueError(f"rank group size must divide world size, got {rank_group_size=} {world_size=}")
    cache_key = (world_size, rank_group_size)
    if cache_key not in _RANK_GROUP_CACHE:
        current_group = None
        current_start = None
        for start in range(0, world_size, rank_group_size):
            ranks = list(range(start, start + rank_group_size))
            process_group = dist.new_group(ranks=ranks)
            if rank in ranks:
                current_group = process_group
                current_start = start
        if current_group is None or current_start is None:
            raise RuntimeError(f"rank {rank} was not assigned to a Qwen3-Omni rank group")
        _RANK_GROUP_CACHE[cache_key] = (current_group, current_start)
    return _RANK_GROUP_CACHE[cache_key]


def _slice_details(details: dict[str, Any], start: int, end: int, total_size: int) -> dict[str, Any]:
    local_details = {}
    for key, value in details.items():
        if isinstance(value, list) and len(value) == total_size:
            local_details[key] = value[start:end]
        else:
            local_details[key] = value
    return local_details


def _score_components(images, prompts, metadata) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Score locally or gather one prompt across a configured rank group."""
    images, prompts, metadata = _normalise_inputs(images, prompts, metadata)
    rank_group_size = _prompt_rank_group_size()
    if rank_group_size <= 1:
        return _score_components_local(images, prompts, metadata)

    process_group, group_start_rank = _get_rank_group(rank_group_size)
    rank = dist.get_rank()
    group_rank = rank - group_start_rank
    gathered_inputs: list[Any] = [None] * rank_group_size
    dist.all_gather_object(
        gathered_inputs,
        {"images": images, "prompts": prompts, "metadata": metadata},
        group=process_group,
    )

    local_count = len(images)
    if any(len(item["images"]) != local_count for item in gathered_inputs):
        raise RuntimeError("all ranks in a Qwen3-Omni prompt group must contribute the same number of videos")
    all_images = [image for item in gathered_inputs for image in item["images"]]
    all_prompts = [prompt for item in gathered_inputs for prompt in item["prompts"]]
    all_metadata = [meta for item in gathered_inputs for meta in item["metadata"]]
    declared_group_sizes = {
        int(meta[_EXPECTED_GROUP_SIZE_KEY])
        for meta in all_metadata
        if isinstance(meta, dict) and meta.get(_EXPECTED_GROUP_SIZE_KEY) is not None
    }
    if len(declared_group_sizes) > 1:
        raise RuntimeError(f"inconsistent declared Qwen3-Omni group sizes: {sorted(declared_group_sizes)}")
    expected_group_size = (
        declared_group_sizes.pop()
        if declared_group_sizes
        else int(os.environ.get("QWEN3OMNI_PAIR_GROUP_SIZE", str(len(all_images))))
    )
    if len(all_images) != expected_group_size:
        raise RuntimeError(
            f"Qwen3-Omni prompt group produced {len(all_images)} videos, expected {expected_group_size}"
        )
    prompt_texts = {
        _prompt_for_item(all_prompts[index], all_metadata[index]) for index in range(len(all_images))
    }
    if len(prompt_texts) != 1:
        raise RuntimeError(f"rank group {group_start_rank}-{group_start_rank + rank_group_size - 1} has mixed prompts")

    result_box: list[Any] = [None]
    if group_rank == 0:
        try:
            scores, details = _score_components_local(all_images, all_prompts, all_metadata)
            result_box[0] = {"ok": True, "scores": scores, "details": details}
        except Exception as exc:
            result_box[0] = {"ok": False, "error": repr(exc)}
    dist.broadcast_object_list(result_box, src=group_start_rank, group=process_group)
    result = result_box[0]
    if not result["ok"]:
        raise RuntimeError(f"Qwen3-Omni rank-group scoring failed: {result['error']}")

    local_start = group_rank * local_count
    local_end = local_start + local_count
    local_scores = {
        component: values[local_start:local_end] for component, values in result["scores"].items()
    }
    local_details = _slice_details(result["details"], local_start, local_end, len(all_images))
    return local_scores, local_details


def _cache_key(images, prompts, metadata) -> tuple[Any, ...]:
    images, prompts, metadata = _normalise_inputs(images, prompts, metadata)
    metadata_repr = []
    for item in metadata:
        if isinstance(item, dict):
            metadata_repr.append(tuple(sorted((str(k), str(v)) for k, v in item.items())))
        else:
            metadata_repr.append(str(item))
    return (
        tuple(_reward_base_urls()),
        _reward_server_index() % len(_reward_base_urls()),
        _prompt_rank_group_size(),
        os.environ.get("QWEN3OMNI_REWARD_RUN_ID", ""),
        tuple(str(x) for x in images),
        tuple(str(x) for x in prompts),
        tuple(metadata_repr),
    )


def qwen3omni_pair_component_reward(component: str):
    if component not in _COMPONENTS:
        raise ValueError(f"unsupported qwen3omni component reward: {component}")

    def _fn(images, prompts, metadata):
        key = _cache_key(images, prompts, metadata)
        if key not in _SPLIT_CACHE:
            _SPLIT_CACHE.clear()
            _SPLIT_CACHE[key] = _score_components(images, prompts, metadata)[0]
        return _SPLIT_CACHE[key][component], {}

    return _fn


def qwen3omni_pair_reward(device=None):
    """Return each sample's mean overall pairwise win rate."""

    def _fn(images, prompts, metadata):
        scores, details = _score_components(images, prompts, metadata)
        return scores["overall"], details

    return _fn


def qwen3omni_pair_overall_reward(device=None):
    return qwen3omni_pair_component_reward("overall")


def qwen3omni_pair_audio_reward(device=None):
    return qwen3omni_pair_component_reward("audio")


def qwen3omni_pair_video_reward(device=None):
    return qwen3omni_pair_component_reward("video")
