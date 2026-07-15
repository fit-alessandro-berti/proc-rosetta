from __future__ import annotations

from typing import Any, MutableMapping

from proc_rosetta_ui.ui_types import WorkspaceArtifact


DEFAULTS: dict[str, Any] = {
    "workspace": {},
    "active_checkpoint_path": None,
    "active_checkpoint_identifier": None,
    "evaluation_results": [],
    "reference_gallery": [],
    "reference_gallery_checkpoint": None,
    "selected_artifact_id": None,
}


def initialize_state(state: MutableMapping[str, Any]) -> None:
    for key, value in DEFAULTS.items():
        if key not in state:
            state[key] = value.copy() if isinstance(value, (dict, list)) else value
    from proc_rosetta_ui.cache_service import initialize_stage_caches

    initialize_stage_caches(state)


def workspace(state: MutableMapping[str, Any]) -> dict[str, WorkspaceArtifact]:
    initialize_state(state)
    return state["workspace"]


def reset_inference_for_checkpoint(
    state: MutableMapping[str, Any], checkpoint_identifier: str
) -> None:
    initialize_state(state)
    previous = state.get("active_checkpoint_identifier")
    state["active_checkpoint_identifier"] = checkpoint_identifier
    if previous is None or previous == checkpoint_identifier:
        return
    for item in workspace(state).values():
        if item.encoding and item.encoding.checkpoint_identifier != checkpoint_identifier:
            item.state = "checkpoint mismatch"
