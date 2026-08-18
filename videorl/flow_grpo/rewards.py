"""Reward registry used by the minimal VA-Judger training configuration."""

from __future__ import annotations

from flow_grpo.qwen3omni_pair_scorer import (
    qwen3omni_pair_audio_reward,
    qwen3omni_pair_overall_reward,
    qwen3omni_pair_reward,
    qwen3omni_pair_video_reward,
)


def multi_score(device, score_dict):
    """Build the configured weighted Qwen3-Omni pairwise reward function."""
    factories = {
        "qwen3omni_pair_reward": qwen3omni_pair_reward,
        "qwen3omni_pair_overall_reward": qwen3omni_pair_overall_reward,
        "qwen3omni_pair_audio_reward": qwen3omni_pair_audio_reward,
        "qwen3omni_pair_video_reward": qwen3omni_pair_video_reward,
    }
    unknown = set(score_dict) - set(factories)
    if unknown:
        raise ValueError(f"unsupported rewards in minimal VA-Judger build: {sorted(unknown)}")

    score_fns = {name: factories[name](device) for name in score_dict}

    def _fn(images, prompts, metadata, only_strict=True):
        del only_strict
        total_scores = None
        score_details = {}
        for name, weight in score_dict.items():
            scores, details = score_fns[name](images, prompts, metadata)
            scores = [float(weight) * float(score) for score in scores]
            score_details[name] = scores
            score_details.update(details)
            if total_scores is None:
                total_scores = scores
            else:
                total_scores = [left + right for left, right in zip(total_scores, scores, strict=True)]
        score_details["avg"] = total_scores or []
        return score_details, {}

    return _fn
