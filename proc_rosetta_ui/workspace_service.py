from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from hashlib import sha256
from typing import MutableMapping

from proc_rosetta.artifact_io import ArtifactParseSettings, ParsedArtifact, parse_artifact
from proc_rosetta.inference import ArtifactEncodingResult
from proc_rosetta.visualization_data import cosine_similarity_matrix
from proc_rosetta_ui.ui_types import WorkspaceArtifact
from proc_rosetta_ui.cache_service import cache_key, cache_put


def add_uploaded_artifact(
    workspace: MutableMapping[str, WorkspaceArtifact],
    data: bytes,
    filename: str,
    settings: ArtifactParseSettings,
    parsed_cache: dict[str, object] | None = None,
) -> WorkspaceArtifact:
    content_hash = sha256(data).hexdigest()
    duplicate = next(
        (
            item
            for item in workspace.values()
            if item.parsed.content_hash == content_hash
            and item.parsed.display_name == filename
        ),
        None,
    )
    if duplicate is not None:
        return duplicate
    key = cache_key(content_hash, filename, asdict(settings))
    parsed = parsed_cache.get(key) if parsed_cache is not None else None
    if parsed is None:
        parsed = parse_artifact(data, filename=filename, settings=settings)
        if parsed_cache is not None:
            cache_put(parsed_cache, key, parsed)
    assert isinstance(parsed, ParsedArtifact)
    item = WorkspaceArtifact(parsed=parsed, warnings=list(parsed.warnings))
    workspace[item.artifact_id] = item
    return item


def set_process_group(item: WorkspaceArtifact, group: str) -> None:
    item.process_group = group.strip()
    if item.encoding is not None:
        item.encoding.process_group = item.process_group
    item.touch()


def workspace_rows(items: Iterable[WorkspaceArtifact]) -> list[dict[str, object]]:
    return [item.table_row() for item in items]


def completed_encodings(items: Iterable[WorkspaceArtifact]) -> list[ArtifactEncodingResult]:
    return [
        item.encoding
        for item in items
        if item.encoding is not None and item.encoding.mu and not item.encoding.errors
    ]


def pairwise_similarity_rows(items: Iterable[WorkspaceArtifact]) -> list[dict[str, object]]:
    encodings = completed_encodings(items)
    similarities = cosine_similarity_matrix(encodings)
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(encodings):
        for right_index in range(left_index + 1, len(encodings)):
            right = encodings[right_index]
            rows.append(
                {
                    "left": left.artifact_name,
                    "right": right.artifact_name,
                    "left_modality": left.modality.label,
                    "right_modality": right.modality.label,
                    "cosine_similarity": float(similarities[left_index, right_index]),
                }
            )
    return sorted(rows, key=lambda row: row["cosine_similarity"], reverse=True)
