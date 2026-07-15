from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from typing import Any, MutableMapping


CACHE_STAGES = (
    "parsed_artifact",
    "preprocessed_input",
    "embedding",
    "decode",
    "simulation",
    "evaluation",
)


def initialize_stage_caches(state: MutableMapping[str, Any]) -> None:
    caches = state.setdefault("stage_caches", {})
    for stage in CACHE_STAGES:
        caches.setdefault(stage, {})


def stage_cache(state: MutableMapping[str, Any], stage: str) -> dict[str, Any]:
    if stage not in CACHE_STAGES:
        raise KeyError(f"unknown cache stage: {stage}")
    initialize_stage_caches(state)
    return state["stage_caches"][stage]


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=_json_default, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def cache_put(cache: dict[str, Any], key: str, value: Any, *, maximum_entries: int = 128) -> None:
    cache[key] = value
    while len(cache) > maximum_entries:
        cache.pop(next(iter(cache)))


def clear_inference_caches(state: MutableMapping[str, Any]) -> None:
    initialize_stage_caches(state)
    for stage in ("preprocessed_input", "embedding", "decode", "simulation", "evaluation"):
        state["stage_caches"][stage].clear()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)

