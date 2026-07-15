from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from proc_rosetta.artifact_io import PreprocessingSettings
from proc_rosetta.inference import LoadedCheckpoint, encode_artifact, prepare_artifact_for_model
from proc_rosetta_ui.ui_types import WorkspaceArtifact
from proc_rosetta_ui.cache_service import cache_key, cache_put


def encode_workspace_items(
    items: Iterable[WorkspaceArtifact],
    checkpoint: LoadedCheckpoint,
    settings: PreprocessingSettings,
    on_update: Callable[[WorkspaceArtifact], None] | None = None,
    preprocessed_cache: dict[str, object] | None = None,
    embedding_cache: dict[str, object] | None = None,
) -> Iterator[WorkspaceArtifact]:
    """Encode sequentially and yield each completed workspace item immediately."""

    for item in items:
        item.state = "canonicalizing"
        item.touch()
        if on_update:
            on_update(item)
        prepared_key = cache_key(
            item.parsed.content_hash,
            checkpoint.metadata.identifier,
            settings,
        )
        cached_prepared = preprocessed_cache.get(prepared_key) if preprocessed_cache else None
        if cached_prepared is None:
            item.prepared = prepare_artifact_for_model(item.parsed, checkpoint.model, settings)
            if preprocessed_cache is not None:
                cache_put(preprocessed_cache, prepared_key, item.prepared)
        else:
            item.prepared = cached_prepared
        item.warnings = list(item.prepared.warnings)
        item.errors = list(item.prepared.errors)
        if not item.prepared.ready:
            item.state = "input limitation"
            item.touch()
            if on_update:
                on_update(item)
            yield item
            continue
        item.state = "encoding"
        item.touch()
        if on_update:
            on_update(item)
        embedding_key = cache_key(prepared_key, checkpoint.metadata.identifier, "deterministic_mu")
        cached_encoding = embedding_cache.get(embedding_key) if embedding_cache else None
        if cached_encoding is None:
            item.encoding = encode_artifact(item.prepared, checkpoint)
            if embedding_cache is not None:
                cache_put(embedding_cache, embedding_key, item.encoding)
        else:
            item.encoding = cached_encoding
        item.encoding.process_group = item.process_group
        item.warnings = list(item.encoding.warnings)
        item.errors = list(item.encoding.errors)
        item.state = "embedding ready" if not item.errors else "encoding failed"
        item.touch()
        if on_update:
            on_update(item)
        yield item
