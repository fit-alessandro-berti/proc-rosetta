from __future__ import annotations

from io import BytesIO, StringIO
import csv
import json
from pathlib import Path
import tempfile
import zipfile
from typing import Any, Iterable

import numpy as np

from proc_rosetta.inference import (
    ArtifactEncodingResult,
    DecodeResult,
    simulate_decoded_behavior,
    validate_decoded_tree,
)
from proc_rosetta.pm4py_bridge import to_pm4py_tree
from proc_rosetta.visualization_data import cosine_similarity_matrix, project_pca
from proc_rosetta_ui.ui_types import WorkspaceArtifact


def embedding_json(encoding: ArtifactEncodingResult) -> bytes:
    return json.dumps(encoding.to_dict(), indent=2, sort_keys=True).encode("utf-8")


def embedding_csv(encoding: ArtifactEncodingResult) -> bytes:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "artifact_id",
            "artifact_name",
            "process_group",
            "modality",
            "checkpoint",
            "dimension",
            "mu",
            "logvar",
            "preprocessing_settings",
            "canonical_mapping",
            "encoding_runtime_seconds",
            "warnings",
        ]
    )
    for index, (mu, logvar) in enumerate(zip(encoding.mu, encoding.logvar)):
        writer.writerow(
            [
                encoding.artifact_id,
                encoding.artifact_name,
                encoding.process_group,
                encoding.modality.value,
                encoding.checkpoint_identifier,
                index,
                mu,
                logvar,
                json.dumps(encoding.preprocessing_metadata, sort_keys=True),
                json.dumps(encoding.canonical_mapping, sort_keys=True),
                encoding.embedding_seconds,
                json.dumps(encoding.warnings),
            ]
        )
    return stream.getvalue().encode("utf-8")


def embedding_numpy(encoding: ArtifactEncodingResult) -> bytes:
    stream = BytesIO()
    np.savez(
        stream,
        mu=np.asarray(encoding.mu, dtype=np.float32),
        logvar=np.asarray(encoding.logvar, dtype=np.float32),
        metadata=np.asarray(json.dumps(encoding.to_dict(), sort_keys=True)),
    )
    return stream.getvalue()


def process_tree_ptml(result: DecodeResult, *, restore_labels: bool = True) -> bytes:
    tree = result.restored_tree if restore_labels else result.tree
    if tree is None or not result.grammar_valid:
        raise ValueError("a grammar-valid decoded tree is required for PTML export")
    import pm4py

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "decoded.ptml"
        pm4py.write_ptml(to_pm4py_tree(tree), str(path))
        content = path.read_bytes()
        if any(modality.value == "petri_net" for modality in result.source_modalities):
            content = _annotate_xml(
                content,
                "PNML-derived ProcRosetta decode. Source transition labels were not used by the "
                "external encoder; canonical decoded labels do not preserve source semantics.",
            )
        return content


def process_tree_report(result: DecodeResult) -> bytes:
    payload = {
        "source_artifact_ids": result.source_artifact_ids,
        "source_modalities": [item.value for item in result.source_modalities],
        "latent_source": result.latent_source,
        "token_ids": result.token_ids,
        "token_names": result.token_names,
        "canonical_tree": None if result.tree is None else result.tree.to_dict(),
        "restored_tree": None if result.restored_tree is None else result.restored_tree.to_dict(),
        "human_readable_tree": None
        if result.restored_tree is None
        else str(result.restored_tree),
        "restored_label_mapping": result.restored_label_mapping,
        "unmapped_labels": result.unmapped_labels,
        "validation": validate_decoded_tree(result),
        "warnings": result.warnings,
        "errors": result.errors,
        "decode_seconds": result.decode_seconds,
        "decoder_configuration": result.decoder_configuration,
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def petri_net_pnml(result: DecodeResult) -> bytes:
    if result.petri_net is None:
        raise ValueError("a Petri-convertible decoded tree is required for PNML export")
    import pm4py

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "decoded-derived.pnml"
        pm4py.write_pnml(
            result.petri_net.net,
            result.petri_net.initial_marking,
            result.petri_net.final_marking,
            str(path),
        )
        note = "Derived deterministically from a ProcRosetta-decoded process tree; not independently generated."
        if any(modality.value == "petri_net" for modality in result.source_modalities):
            note += (
                " The source PNML transition labels were not used by the external encoder and "
                "canonical decoded labels do not preserve source semantics."
            )
        return _annotate_xml(path.read_bytes(), note)


def simulated_log_xes(
    result: DecodeResult,
    *,
    num_traces: int = 100,
    random_seed: int = 13,
) -> bytes:
    traces = simulate_decoded_behavior(
        result,
        num_traces=num_traces,
        random_seed=random_seed,
    )
    from pm4py.objects.log.obj import Event, EventLog, Trace
    import pm4py

    log = EventLog()
    for index, values in enumerate(traces):
        trace = Trace(attributes={"concept:name": f"simulated-{index}"})
        for value in values:
            trace.append(Event({"concept:name": value}))
        log.append(trace)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "simulated.xes"
        pm4py.write_xes(log, str(path))
        return _annotate_xml(
            path.read_bytes(),
            f"ProcRosetta simulated behavior; source={','.join(result.source_artifact_ids)}; "
            f"traces={num_traces}; random_seed={random_seed}.",
        )


def complete_workspace_zip(
    items: Iterable[WorkspaceArtifact],
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
    application_configuration: dict[str, Any] | None = None,
    evaluation_results: dict[str, Any] | None = None,
) -> bytes:
    stream = BytesIO()
    manifest: dict[str, Any] = {
        "checkpoint": checkpoint_metadata or {},
        "application_configuration": application_configuration or {},
        "evaluation_results": evaluation_results or {},
        "artifacts": [],
    }
    item_list = list(items)
    encodings = [item.encoding for item in item_list if item.encoding is not None and item.encoding.mu]
    if encodings:
        names = [encoding.artifact_name for encoding in encodings]
        similarities = cosine_similarity_matrix(encodings)
        manifest["pairwise_cosine_similarity"] = {
            "artifacts": names,
            "matrix": similarities.tolist(),
        }
        groups = {item.artifact_id: item.process_group for item in item_list}
        projection = project_pca(encodings, groups)
        manifest["pca"] = {
            "coordinates": projection.rows,
            "explained_variance": projection.explained_variance,
            "meaningful": projection.meaningful,
        }
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in item_list:
            artifact_data: dict[str, Any] = {
                "artifact_id": item.artifact_id,
                "display_name": item.parsed.display_name,
                "modality": item.parsed.modality.value,
                "process_group": item.process_group,
                "state": item.state,
                "source_metadata": item.parsed.source_metadata,
                "warnings": item.warnings,
                "errors": item.errors,
            }
            if item.encoding is not None:
                artifact_data["encoding"] = item.encoding.to_dict()
                archive.writestr(
                    f"embeddings/{item.artifact_id}.json",
                    embedding_json(item.encoding),
                )
            decode_rows = []
            for index, result in enumerate(item.decodes):
                report = process_tree_report(result)
                decode_rows.append(json.loads(report))
                archive.writestr(f"decodes/{item.artifact_id}-{index}.json", report)
                if result.grammar_valid:
                    archive.writestr(
                        f"decodes/{item.artifact_id}-{index}.ptml",
                        process_tree_ptml(result),
                    )
                if result.petri_convertible:
                    archive.writestr(
                        f"decodes/{item.artifact_id}-{index}-derived.pnml",
                        petri_net_pnml(result),
                    )
                if result.grammar_valid:
                    archive.writestr(
                        f"simulations/{item.artifact_id}-{index}.xes",
                        simulated_log_xes(result, num_traces=100),
                    )
            artifact_data["decodes"] = decode_rows
            manifest["artifacts"].append(artifact_data)
        archive.writestr("workspace.json", json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return stream.getvalue()


def _annotate_xml(content: bytes, note: str) -> bytes:
    comment = f"<!-- {note.replace('--', '—')} -->\n".encode("utf-8")
    if content.startswith(b"<?xml") and b"?>" in content:
        end = content.index(b"?>") + 2
        return content[:end] + b"\n" + comment + content[end:].lstrip(b"\r\n")
    return comment + content
