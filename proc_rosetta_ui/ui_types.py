from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from proc_rosetta.artifact_io import ParsedArtifact, PreparedArtifact
from proc_rosetta.inference import ArtifactEncodingResult, DecodeResult


@dataclass
class WorkspaceArtifact:
    parsed: ParsedArtifact
    process_group: str = ""
    state: str = "parsed"
    prepared: PreparedArtifact | None = None
    encoding: ArtifactEncodingResult | None = None
    decodes: list[DecodeResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def artifact_id(self) -> str:
        return self.parsed.artifact_id

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def table_row(self) -> dict[str, object]:
        metadata = self.parsed.source_metadata
        size = (
            metadata.get("total_events")
            or metadata.get("tree_size")
            or metadata.get("nodes")
            or 0
        )
        labels = metadata.get("distinct_activities")
        if labels is None and self.parsed.graph is not None:
            labels = metadata.get("visible_transitions", 0)
        latest = self.decodes[-1] if self.decodes else None
        return {
            "Artifact": self.parsed.display_name,
            "Group": self.process_group or "—",
            "Modality": self.parsed.modality.label,
            "Size": int(size),
            "Labels": int(labels or 0),
            "State": self.state,
            "Encoded": bool(self.encoding and self.encoding.mu and not self.encoding.errors),
            "Decoded": latest is not None,
            "Validation": "—" if latest is None else ("valid" if latest.successful else "failed"),
        }

