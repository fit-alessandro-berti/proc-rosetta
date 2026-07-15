from pathlib import Path
from dataclasses import replace

import pytest

from proc_rosetta.artifact_io import (
    ArtifactModality,
    PreprocessingSettings,
    TraceSelectionStrategy,
    parse_artifact,
    select_traces,
)
from proc_rosetta.inference import (
    PETRI_LABEL_WARNING,
    build_decode_result,
    decode_latent_iter,
    encode_artifact,
    list_trusted_checkpoints,
    load_trusted_checkpoint,
    prepare_artifact_for_model,
    interpolate_latents,
    sample_latent_distribution,
    validate_decoded_tree,
)
from proc_rosetta.evaluation_iterators import (
    cross_modal_retrieval_iter,
    decode_quality_iter,
    discovery_comparison_iter,
    embedding_baseline_iter,
    neural_loss_iter,
)
from proc_rosetta.data import write_samples_jsonl
from proc_rosetta.synthetic import SyntheticConfig, generate_samples
from proc_rosetta.reference_gallery import build_reference_gallery_iter
from proc_rosetta_ui.ui_types import WorkspaceArtifact
from proc_rosetta_ui.cache_service import CACHE_STAGES, cache_key, cache_put, stage_cache
from proc_rosetta_ui.export_service import petri_net_pnml, process_tree_ptml
from proc_rosetta_ui.workspace_service import bundled_artifact_paths


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
FILES = ROOT / "scripts" / "files"


def studio_checkpoint():
    paths = list_trusted_checkpoints(CHECKPOINT_DIR)
    if not paths:
        pytest.skip("repository checkpoint is unavailable")
    return load_trusted_checkpoint(paths[0], trusted_directory=CHECKPOINT_DIR, device="cpu")


def test_trace_selection_is_reproducible_and_bounded():
    traces = tuple((f"A{index % 3}", str(index)) for index in range(20))
    settings = PreprocessingSettings(
        max_traces=7,
        trace_selection_strategy=TraceSelectionStrategy.SEEDED_RANDOM,
        random_seed=41,
    )

    assert select_traces(traces, settings) == select_traces(traces, settings)
    assert len(select_traces(traces, settings)) == 7


def test_all_external_modalities_parse_prepare_and_encode():
    checkpoint = studio_checkpoint()
    expected_suffixes = {
        ".xes": ArtifactModality.EVENT_LOG,
        ".ptml": ArtifactModality.PROCESS_TREE,
        ".pnml": ArtifactModality.PETRI_NET,
    }
    paths = bundled_artifact_paths()

    assert len(paths) == 9
    assert {path.parent for path in paths} == {FILES}
    assert {suffix: sum(path.suffix == suffix for path in paths) for suffix in expected_suffixes} == {
        suffix: 3 for suffix in expected_suffixes
    }
    for path in paths:
        modality = expected_suffixes[path.suffix]
        parsed = parse_artifact(path)
        prepared = prepare_artifact_for_model(parsed, checkpoint.model)
        encoded = encode_artifact(prepared, checkpoint)

        assert parsed.modality is modality
        assert prepared.ready
        assert len(encoded.mu) == checkpoint.metadata.latent_dimension
        assert len(encoded.logvar) == checkpoint.metadata.latent_dimension
        assert not encoded.errors
        if modality is ArtifactModality.PETRI_NET:
            assert prepared.model_input_summary["visible_labels_used_by_encoder"] is False
            assert any("does not use visible transition labels" in warning for warning in encoded.warnings)


def test_progressive_decode_and_exports_keep_validation_stages_separate():
    checkpoint = studio_checkpoint()
    tokenizer = checkpoint.model.tree_tokenizer
    token_names = ["<bos>", "SEQ", "ARITY_2", "A0", "A1", "<eos>"]
    token_ids = [tokenizer.token_to_id[name] for name in token_names]
    result = build_decode_result(
        checkpoint.model,
        [0.0] * checkpoint.metadata.latent_dimension,
        source_artifact_ids=["example"],
        source_modalities=[ArtifactModality.EVENT_LOG],
        latent_source="test",
        token_ids=token_ids,
        canonical_mapping={"Register": "A0", "Approve": "A1"},
    )

    assert validate_decoded_tree(result) == {
        "eos_emitted": True,
        "grammar_valid": True,
        "arity_valid": True,
        "vocabulary_valid": True,
        "petri_convertible": True,
        "label_restoration_complete": True,
        "length_limit_reached": False,
    }
    ptml = process_tree_ptml(result)
    pnml = petri_net_pnml(result)
    assert b"Register" in ptml
    assert b"pnml" in pnml.lower()
    assert parse_artifact(ptml, filename="roundtrip.ptml").tree is not None
    assert parse_artifact(pnml, filename="roundtrip.pnml").graph is not None

    steps = list(
        decode_latent_iter(
            checkpoint.model,
            [0.0] * checkpoint.metadata.latent_dimension,
            max_length=12,
        )
    )
    assert steps
    assert steps[0].current_prefix[0] == "<bos>"
    assert all(step.valid_next_tokens for step in steps)


def test_pnml_decode_never_restores_source_labels():
    checkpoint = studio_checkpoint()
    tokenizer = checkpoint.model.tree_tokenizer
    result = build_decode_result(
        checkpoint.model,
        [0.0] * checkpoint.metadata.latent_dimension,
        source_artifact_ids=["petri"],
        source_modalities=[ArtifactModality.PETRI_NET],
        latent_source="petri_net_mean",
        token_ids=[tokenizer.bos_id, tokenizer.token_to_id["A0"], tokenizer.eos_id],
        canonical_mapping={"Misleading source label": "A0"},
    )

    assert result.restored_label_mapping == {}
    assert result.restored_tree == result.tree
    assert PETRI_LABEL_WARNING in result.warnings


def test_trusted_checkpoint_loader_rejects_paths_outside_root(tmp_path):
    untrusted = tmp_path / "model.pt"
    untrusted.write_bytes(b"not a checkpoint")
    with pytest.raises(PermissionError):
        load_trusted_checkpoint(untrusted, trusted_directory=CHECKPOINT_DIR, device="cpu")


def test_stochastic_sampling_and_interpolation_are_reproducible():
    mu = [0.0, 1.0, 2.0]
    logvar = [-1.0, 0.0, 1.0]
    assert sample_latent_distribution(mu, logvar, random_seed=7) == sample_latent_distribution(
        mu, logvar, random_seed=7
    )
    assert interpolate_latents([0.0, 2.0], [2.0, 4.0], 0.25) == [0.5, 2.5]


def test_reference_gallery_yields_all_modalities_progressively():
    checkpoint = studio_checkpoint()
    updates = list(
        build_reference_gallery_iter(
            checkpoint,
            count=1,
            seed=3,
            traces_per_sample=3,
        )
    )

    assert len(updates) == 1
    assert {entry.modality for entry in updates[0].entries} == set(ArtifactModality)
    assert all(entry.encoding.dimension == checkpoint.metadata.latent_dimension for entry in updates[0].entries)


def test_workspace_cross_modal_retrieval_covers_six_directions():
    checkpoint = studio_checkpoint()
    items = []
    for filename in ("running-example.xes", "running-example.ptml", "running-example.pnml"):
        parsed = parse_artifact(FILES / filename)
        encoding = encode_artifact(prepare_artifact_for_model(parsed, checkpoint.model), checkpoint)
        items.append(
            WorkspaceArtifact(
                parsed=parsed,
                process_group="running-example",
                encoding=encoding,
            )
        )

    updates = list(cross_modal_retrieval_iter(items))
    assert len(updates) == 6
    assert all(update.result["available"] for update in updates)
    assert all(update.result["top1_accuracy"] == 1.0 for update in updates)
    decode_updates = list(
        decode_quality_iter(items, checkpoint, max_length=32, simulated_traces=3)
    )
    assert len(decode_updates) == 4
    assert decode_updates[-1].result["latent_source"] == "fused_mean"


def test_embedding_baselines_progress_one_method_at_a_time():
    checkpoint = studio_checkpoint()
    first = parse_artifact(FILES / "running-example.xes")
    second = replace(first, artifact_id="second-log", display_name="second.xes")
    items = []
    for parsed in (first, second):
        encoding = encode_artifact(prepare_artifact_for_model(parsed, checkpoint.model), checkpoint)
        items.append(WorkspaceArtifact(parsed=parsed, encoding=encoding))

    updates = list(embedding_baseline_iter(items))
    assert len(updates) == 10
    assert updates[-1].completed == updates[-1].total
    assert {update.result["method"] for update in updates} >= {
        "Activity counts",
        "ProcRosetta event-log means",
    }


def test_stage_caches_are_independent_and_bounded():
    state = {}
    caches = {stage: stage_cache(state, stage) for stage in CACHE_STAGES}
    key = cache_key("artifact", {"setting": 1})
    cache_put(caches["embedding"], key, [1.0, 2.0], maximum_entries=2)
    cache_put(caches["embedding"], "second", 2, maximum_entries=2)
    cache_put(caches["embedding"], "third", 3, maximum_entries=2)

    assert key not in caches["embedding"]
    assert len(caches["embedding"]) == 2
    assert not caches["decode"]


def test_neural_loss_iterator_yields_batch_updates(tmp_path):
    checkpoint = studio_checkpoint()
    sample_path = tmp_path / "samples.jsonl"
    samples = generate_samples(
        1,
        SyntheticConfig(
            generator="isolated",
            max_depth=2,
            max_activities=4,
            traces_per_sample=2,
            curriculum_phase=1,
        ),
        seed=5,
    )
    write_samples_jsonl(sample_path, samples)

    updates = list(
        neural_loss_iter(
            checkpoint,
            str(sample_path),
            batch_size=1,
            max_batches=1,
        )
    )
    assert len(updates) == 1
    assert updates[0].result["running_loss"] > 0


def test_discovery_comparison_progresses_by_method():
    checkpoint = studio_checkpoint()
    parsed = parse_artifact(FILES / "running-example.xes")
    encoding = encode_artifact(prepare_artifact_for_model(parsed, checkpoint.model), checkpoint)
    item = WorkspaceArtifact(parsed=parsed, encoding=encoding)

    updates = list(
        discovery_comparison_iter(
            [item],
            checkpoint,
            max_length=32,
            exact_conformance=False,
        )
    )
    assert [update.result["method"] for update in updates] == [
        "ProcRosetta",
        "Inductive Miner",
    ]


def test_analytical_pages_render_with_a_populated_multimodal_workspace():
    from streamlit.testing.v1 import AppTest

    checkpoint = studio_checkpoint()
    workspace = {}
    for filename in ("running-example.xes", "running-example.ptml", "running-example.pnml"):
        parsed = parse_artifact(FILES / filename)
        encoding = encode_artifact(prepare_artifact_for_model(parsed, checkpoint.model), checkpoint)
        item = WorkspaceArtifact(
            parsed=parsed,
            process_group="running-example",
            encoding=encoding,
            state="embedding ready",
        )
        workspace[item.artifact_id] = item

    for page in ("pages/01_workspace.py", "pages/03_latent_explorer.py", "pages/04_evaluation.py"):
        app = AppTest.from_file(str(ROOT / page), default_timeout=30)
        app.session_state["workspace"] = workspace
        app.run()
        assert not app.exception, (page, [exception.value for exception in app.exception])

    log_item = next(
        item for item in workspace.values() if item.parsed.modality is ArtifactModality.EVENT_LOG
    )
    tokenizer = checkpoint.model.tree_tokenizer
    decoded = build_decode_result(
        checkpoint.model,
        log_item.encoding.mu,
        source_artifact_ids=[log_item.artifact_id],
        source_modalities=[ArtifactModality.EVENT_LOG],
        latent_source="event_log_mean",
        token_ids=[
            tokenizer.bos_id,
            tokenizer.token_to_id["SEQ"],
            tokenizer.token_to_id["ARITY_2"],
            tokenizer.token_to_id["A0"],
            tokenizer.token_to_id["A1"],
            tokenizer.eos_id,
        ],
        canonical_mapping=log_item.encoding.canonical_mapping,
    )
    translation = AppTest.from_file(str(ROOT / "pages/02_translation.py"), default_timeout=30)
    translation.session_state["workspace"] = workspace
    translation.session_state["latest_decode_result"] = decoded
    translation.session_state["latest_decode_alternatives"] = [decoded]
    translation.session_state["latest_decode_source_ids"] = [log_item.artifact_id]
    translation.run()
    assert not translation.exception, [exception.value for exception in translation.exception]


def test_workspace_upload_automatically_builds_an_embedding():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "pages/01_workspace.py"), default_timeout=30).run()
    app.get("file_uploader")[0].upload(
        "uploaded-example.xes",
        (FILES / "running-example.xes").read_bytes(),
        "text/xml",
    ).run()

    assert not app.exception, [exception.value for exception in app.exception]
    imported = list(app.session_state["workspace"].values())
    assert len(imported) == 1
    assert imported[0].state == "embedding ready"
    assert imported[0].encoding is not None
    assert imported[0].encoding.mu
    assert any("imported and embedded successfully" in message.value for message in app.success)


def test_workspace_imports_and_groups_all_nine_bundled_artifacts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "pages/01_workspace.py"), default_timeout=30).run()
    next(widget for widget in app.selectbox if widget.label == "Bundled artifacts").select(
        "All 9 artifacts"
    ).run()
    next(widget for widget in app.button if widget.label == "Import bundled selection").click().run()

    assert not app.exception, [exception.value for exception in app.exception]
    imported = list(app.session_state["workspace"].values())
    assert len(imported) == 9
    assert all(item.state == "embedding ready" for item in imported)
    assert all(item.encoding is not None and item.encoding.mu for item in imported)
    assert {item.process_group for item in imported} == {
        "receipt",
        "roadtraffic100traces",
        "running-example",
    }
