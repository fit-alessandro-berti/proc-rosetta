import json
import csv
import pytest
import torch

from proc_rosetta.cli import main
from proc_rosetta.cli import (
    build_parser,
    checkpoint_for_selection,
    split_counts_from_args,
    synthetic_config_from_args,
)
from proc_rosetta.data import (
    SplitCounts,
    multiprocessing_worker_count,
    recreate_data_splits,
)
from proc_rosetta.devices import default_device
from proc_rosetta.synthetic import SyntheticConfig


def test_default_sample_and_train_values_match_recommended_run():
    parser = build_parser()

    sample_args = parser.parse_args(["sample"])
    train_args = parser.parse_args(["train"])
    test_args = parser.parse_args(["test"])
    split_counts = split_counts_from_args(sample_args)

    assert split_counts.training == 16384
    assert split_counts.validation == 2048
    assert split_counts.test == 2048
    rows_per_family = 4
    assert split_counts.training // rows_per_family == 4096
    assert split_counts.validation // rows_per_family == 512
    assert split_counts.test // rows_per_family == 512
    assert sample_args.max_depth == 8
    assert sample_args.max_activities == 30
    assert sample_args.max_arity == 3
    assert sample_args.traces_per_sample == 128
    assert sample_args.curriculum_phase == 3
    assert sample_args.operator_probabilities is None
    assert sample_args.root_operator_probabilities is None
    sample_config = synthetic_config_from_args(sample_args)
    assert sample_config.root_operator_probabilities == {
        "seq": 0.7,
        "xor": 0.1,
        "and": 0.1,
        "loop": 0.1,
    }
    assert not sample_args.multiprocessing
    assert not sample_args.quiet
    assert train_args.epochs == 100
    assert train_args.batch_size == 128
    assert train_args.latent_dim == 96
    assert train_args.device == default_device()
    assert train_args.hidden_dim == 192
    assert train_args.semantic_latent_mode == "deterministic"
    assert train_args.dropout is None
    assert train_args.trace_encoder_dropout == 0.20
    assert train_args.decoder_dropout == 0.20
    assert train_args.projection_dropout == 0.20
    assert train_args.weight_decay == 5e-4
    assert train_args.label_smoothing == 0.04
    assert train_args.early_stopping_patience == 6
    assert train_args.activity_remap_probability == 0.5
    assert not train_args.resume
    assert test_args.device == default_device()
    assert not test_args.quiet
    assert test_args.conformance_method == "token_based_replay"
    assert test_args.checkpoint_selection == "best"


def test_sample_operator_probability_overrides_are_parsed_independently():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sample",
            "--operator-probabilities",
            "seq=0.1,xor=0.2,and=0.3,loop=0.4",
            "--root-operator-probabilities",
            "sequence=0.7,xor=0.1,and=0.1,loop=0.1",
        ]
    )

    config = synthetic_config_from_args(args)

    assert config.operator_probabilities == {
        "seq": 0.1,
        "xor": 0.2,
        "and": 0.3,
        "loop": 0.4,
    }
    assert config.root_operator_probabilities == {
        "seq": 0.7,
        "xor": 0.1,
        "and": 0.1,
        "loop": 0.1,
    }


@pytest.mark.parametrize(("cpu_count", "expected"), [(16, 15), (2, 1), (1, 1), (None, 1)])
def test_multiprocessing_worker_count_reserves_one_core(monkeypatch, cpu_count, expected):
    monkeypatch.setattr("proc_rosetta.data.os.cpu_count", lambda: cpu_count)

    assert multiprocessing_worker_count() == expected


def test_sample_multiprocessing_flag_is_forwarded(monkeypatch, capsys):
    calls = {}

    def recreate(**kwargs):
        calls.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("proc_rosetta.cli.recreate_data_splits", recreate)

    assert main(["sample", "--multiprocessing", "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert calls["use_multiprocessing"] is True
    assert calls["show_progress"] is False


def test_checkpoint_selection_resolves_best_and_latest_paths():
    latest = "checkpoints/proc_rosetta.pt"

    assert checkpoint_for_selection(latest, "best").name == "proc_rosetta.best.pt"
    assert checkpoint_for_selection(latest, "latest").name == "proc_rosetta.pt"
    assert checkpoint_for_selection(
        "checkpoints/proc_rosetta.best_trace.pt", "best"
    ).name == "proc_rosetta.best_trace.pt"


def test_unsupported_kl_regularizer_is_not_exposed_by_cli():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--kl-weight", "0.1"])


def test_test_quiet_disables_progress_output(monkeypatch, capsys):
    calls = {}

    def read_samples(path, show_progress=False):
        calls["read_progress"] = show_progress
        return [object()]

    def report(**kwargs):
        calls["report_progress"] = kwargs["show_progress"]
        calls["conformance_method"] = kwargs["conformance_method"]
        calls["checkpoint_path"] = kwargs["checkpoint_path"]
        return {"split": "test"}

    monkeypatch.setattr("proc_rosetta.cli.read_samples_jsonl", read_samples)
    monkeypatch.setattr("proc_rosetta.cli.rich_test_report", report)

    assert main(["test", "--quiet", "--json"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"split": "test"}
    assert captured.err == ""
    assert calls == {
        "read_progress": False,
        "report_progress": False,
        "conformance_method": "token_based_replay",
        "checkpoint_path": checkpoint_for_selection(
            "checkpoints/proc_rosetta.pt", "best"
        ),
    }


def test_sample_progress_reports_generated_triplets(tmp_path, monkeypatch):
    updates: list[int] = []

    class RecordingProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def update(self, count: int = 1) -> None:
            updates.append(count)

    def progress_bar(total, enabled, **kwargs):
        assert enabled
        assert kwargs["unit"] == "triplet"
        return RecordingProgress()

    monkeypatch.setattr("proc_rosetta.data.progress_bar", progress_bar)
    recreate_data_splits(
        tmp_path / "data",
        counts=SplitCounts(training=2, validation=1, test=1),
        config=SyntheticConfig(
            generator="isolated",
            max_depth=2,
            max_activities=4,
            traces_per_sample=1,
        ),
        show_progress=True,
    )

    assert sum(updates) == 4


def test_sample_cli_recreates_data_splits(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stale = data_dir / "stale.txt"
    stale.write_text("remove me", encoding="utf-8")

    exit_code = main(
        [
            "sample",
            "--data-dir",
            str(data_dir),
            "--train-count",
            "2",
            "--validation-count",
            "1",
            "--test-count",
            "1",
            "--class-coverage-mode",
            "best_effort",
            "--seed",
            "5",
            "--max-depth",
            "2",
            "--traces-per-sample",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "flattened row-count flags are deprecated" in captured.err
    metadata = json.loads(captured.out.strip().splitlines()[-1])
    assert not stale.exists()
    assert (data_dir / "training" / "samples.jsonl").exists()
    assert (data_dir / "validation" / "samples.jsonl").exists()
    assert (data_dir / "test" / "samples.jsonl").exists()
    assert metadata["splits"]["training"]["statistics"]["count"] == 2
    assert metadata["exact_behavior_signatures_disjoint"] is True
    assert metadata["synthetic_config"]["logs"]["log_views_per_behavior"] == 2
    coverage = metadata["splits"]["training"]["class_coverage"]
    assert coverage["mode"] == "best_effort"
    assert not coverage["meets_minimum"]
    assert coverage["deficits_by_motif"]


def test_strict_sample_coverage_hits_each_motif_and_representation(tmp_path, capsys):
    data_dir = tmp_path / "covered-data"

    assert main(
        [
            "sample",
            "--data-dir",
            str(data_dir),
            "--train-families",
            "4",
            "--validation-families",
            "4",
            "--test-families",
            "4",
            "--min-families-per-motif",
            "1",
            "--max-depth",
            "2",
            "--traces-per-sample",
            "2",
        ]
    ) == 0

    metadata = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    for split in ("training", "validation", "test"):
        coverage = metadata["splits"][split]["class_coverage"]
        assert coverage["meets_minimum"]
        assert coverage["exact_quota_match"]
        assert not coverage["representation_slot_deficits_by_motif"]
        assert set(coverage["actual_family_counts_by_motif"].values()) == {1}
        assert all(
            len(representation_counts) == 2
            for representation_counts in coverage["motif_representation_counts"].values()
        )


def test_strict_sample_coverage_rejects_infeasible_counts_before_replacing_data(
    tmp_path,
):
    data_dir = tmp_path / "existing-data"
    data_dir.mkdir()
    marker = data_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="divisible|cannot provide at least"):
        main(
            [
                "sample",
                "--data-dir",
                str(data_dir),
                "--train-count",
                "2",
                "--validation-count",
                "2",
                "--test-count",
                "2",
            ]
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_train_and_test_cli_smoke(tmp_path, capsys):
    data_dir = tmp_path / "data"
    checkpoint = tmp_path / "checkpoints" / "model.pt"
    metrics_csv = tmp_path / "checkpoints" / "metrics.csv"
    assert main(
        [
            "sample",
            "--data-dir",
            str(data_dir),
            "--train-count",
            "2",
            "--validation-count",
            "1",
            "--test-count",
            "1",
            "--class-coverage-mode",
            "best_effort",
            "--max-depth",
            "2",
            "--max-activities",
            "4",
            "--traces-per-sample",
            "2",
        ]
    ) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "train",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--metrics-csv",
            str(metrics_csv),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "8",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[train] Starting epoch 1/1" in captured.err
    assert "unique behavior families" in captured.err
    assert "Restored best validation-loss weights" in captured.err
    row = json.loads(captured.out.strip().splitlines()[-1])
    assert row["epoch"] == 1
    assert row["training"]["loss"] > 0
    assert row["validation"]["loss"] > 0
    assert abs(
        row["generalization_gap"]["loss"] - (row["validation"]["loss"] - row["training"]["loss"])
    ) < 1e-5
    assert "learning_rate" in row
    assert "epoch_seconds" in row
    assert checkpoint.exists()
    epoch_one_directory = checkpoint.parent / "00001"
    epoch_one_checkpoint = epoch_one_directory / checkpoint.name
    epoch_one_metrics = epoch_one_directory / metrics_csv.name
    assert epoch_one_checkpoint.exists()
    assert epoch_one_metrics.exists()
    assert torch.load(
        epoch_one_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["epoch"] == 1
    assert checkpoint.with_name("model.best.pt").exists()
    for objective in ("loss", "trace", "edit", "latent"):
        assert checkpoint.with_name(f"model.best_{objective}.pt").exists()
    assert metrics_csv.exists()
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["training_loss"]
    assert rows[0]["validation_loss"]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["version"] == 6
    assert saved["model_architecture"] == "proc-rosetta-latent-transformer-v6"
    assert saved["tree_normalization_version"] == "pm4py-fold-v1"
    assert saved["semantic_latent_stochastic"] is False
    assert saved["loss_weights"]["kl"] == 0.0
    assert "optimizer_state_dict" in saved
    assert "scheduler_state_dict" in saved
    assert "rng_state" in saved
    assert "training_loader_state" in saved

    # Version-2 checkpoints, including the repository's interrupted run, have
    # weights and history but no exact optimizer/RNG continuation state.
    for key in (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "training_loader_state",
    ):
        saved.pop(key)
    saved["version"] = 2
    torch.save(saved, checkpoint)

    assert main(
        [
            "train",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--metrics-csv",
            str(metrics_csv),
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "8",
            "--scheduled-sampling-max",
            "0",
            "--resume",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "Applying resume runtime-policy overrides" in captured.err
    assert "scheduled_sampling_max" in captured.err
    assert "Legacy checkpoint has no optimizer" in captured.err
    assert "Starting epoch 2/2" in captured.err
    assert json.loads(captured.out.strip().splitlines()[-1])["epoch"] == 2
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epoch"] for row in rows] == ["1", "2"]
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert len(resumed["history"]) == 2
    assert resumed["history"][-1]["resume_policy_overrides"] == {
        "scheduled_sampling_max": {
            "checkpoint": 0.075,
            "requested": 0.0,
        }
    }
    assert "optimizer_state_dict" in resumed
    assert (checkpoint.parent / "00002" / checkpoint.name).exists()
    with (checkpoint.parent / "00002" / metrics_csv.name).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert [row["epoch"] for row in csv.DictReader(handle)] == ["1", "2"]
    assert torch.load(
        epoch_one_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["version"] == 6

    assert main(
        [
            "train",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--metrics-csv",
            str(metrics_csv),
            "--epochs",
            "3",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "8",
            "--scheduled-sampling-max",
            "0",
            "--resume",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "Restored optimizer, scheduler, RNG, and data-loader state" in captured.err
    assert "Starting epoch 3/3" in captured.err
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["epoch"] for row in rows] == ["1", "2", "3"]
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed["ema_state_dict"]["updates"] > 0
    assert resumed["history"][-1]["validation_ema"] is not None
    assert set(resumed["history"][-1]["objective_candidate_weights"]) == {
        "loss",
        "trace",
        "edit",
        "latent",
    }
    epoch_three_checkpoint = checkpoint.parent / "00003" / checkpoint.name
    assert epoch_three_checkpoint.exists()

    assert main(
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(epoch_three_checkpoint),
            "--checkpoint-selection",
            "latest",
            "--batch-size",
            "1",
            "--max-decode-length",
            "32",
            "--petri-embedding-dim",
            "8",
            "--petri-num-walks",
            "2",
            "--petri-walk-length",
            "6",
            "--petri-epochs",
            "2",
            "--json",
        ]
    ) == 0
    captured = capsys.readouterr()
    test_row = json.loads(captured.out.strip().splitlines()[-1])
    assert "[test] Plan: 1 sample" in captured.err
    assert "2 token-replay conformance evaluations" in captured.err
    assert "[test] [4/8] Running 2 token-replay evaluations" in captured.err
    assert "Discovery replays" in captured.err
    assert test_row["split"] == "test"
    assert test_row["loss_metrics"]["loss"] > 0
    assert "cross_modal_retrieval" in test_row
    assert "proc_rosetta_fused_mu" in test_row["embedding_methods"]
    assert "trace_directly_follows" in test_row["embedding_methods"]
    assert "pm4py_colonna_petri_node2vec" in test_row["embedding_methods"]
    assert "method_comparisons_against_proc_rosetta_fused_mu" in test_row
    assert "decode_quality" in test_row
    assert "proc_rosetta_fused_mu" in test_row["decode_quality"]["methods"]
    assert "valid_tree_rate" in test_row["decode_quality"]["methods"]["proc_rosetta_fused_mu"]
    assert "discovery_quality" in test_row
    assert test_row["discovery_quality"]["conformance_method"] == "token_based_replay"
    assert "proc_rosetta_trace_mu" in test_row["discovery_quality"]["methods"]
    assert "inductive_miner" in test_row["discovery_quality"]["methods"]
    assert "mean_f1" in test_row["discovery_quality"]["methods"]["inductive_miner"]

    assert main(
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-selection",
            "latest",
            "--batch-size",
            "1",
            "--max-decode-length",
            "32",
            "--petri-embedding-dim",
            "8",
            "--petri-num-walks",
            "2",
            "--petri-walk-length",
            "6",
            "--petri-epochs",
            "2",
            "--conformance-method",
            "footprints",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "ProcRosetta Test Report" in captured.out
    assert "Decode quality" in captured.out
    assert "Process discovery quality" in captured.out
    assert "Footprint conformance" in captured.out
    assert "directly on the discovered process trees (not Petri nets)" in captured.out
    assert "pm4py Petri Node2Vec vs ProcRosetta fused" in captured.out
    assert "Agreement against ProcRosetta fused encoding" in captured.out
