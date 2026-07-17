import json
import csv
import pytest

from proc_rosetta.cli import main
from proc_rosetta.cli import build_parser, split_counts_from_args
from proc_rosetta.data import SplitCounts, recreate_data_splits
from proc_rosetta.devices import default_device
from proc_rosetta.synthetic import SyntheticConfig


def test_default_sample_and_train_values_match_recommended_run():
    parser = build_parser()

    sample_args = parser.parse_args(["sample"])
    train_args = parser.parse_args(["train"])
    test_args = parser.parse_args(["test"])
    split_counts = split_counts_from_args(sample_args)

    assert split_counts.training == 8192
    assert split_counts.validation == 1024
    assert split_counts.test == 1024
    rows_per_family = 2
    active_motifs = 4
    assert split_counts.training // rows_per_family // active_motifs == 1024
    assert split_counts.validation // rows_per_family // active_motifs == 128
    assert split_counts.test // rows_per_family // active_motifs == 128
    assert sample_args.max_depth == 8
    assert sample_args.max_activities == 30
    assert sample_args.max_arity == 3
    assert sample_args.traces_per_sample == 128
    assert sample_args.curriculum_phase == 3
    assert not sample_args.quiet
    assert train_args.epochs == 100
    assert train_args.batch_size == 32
    assert train_args.latent_dim == 256
    assert train_args.device == default_device()
    assert train_args.hidden_dim == 96
    assert train_args.dropout == 0.25
    assert train_args.weight_decay == 1e-3
    assert train_args.label_smoothing == 0.08
    assert train_args.early_stopping_patience == 4
    assert train_args.activity_remap_probability == 0.5
    assert test_args.device == default_device()
    assert not test_args.quiet


def test_test_quiet_disables_progress_output(monkeypatch, capsys):
    calls = {}

    def read_samples(path, show_progress=False):
        calls["read_progress"] = show_progress
        return [object()]

    def report(**kwargs):
        calls["report_progress"] = kwargs["show_progress"]
        return {"split": "test"}

    monkeypatch.setattr("proc_rosetta.cli.read_samples_jsonl", read_samples)
    monkeypatch.setattr("proc_rosetta.cli.rich_test_report", report)

    assert main(["test", "--quiet", "--json"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"split": "test"}
    assert captured.err == ""
    assert calls == {"read_progress": False, "report_progress": False}


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
    metadata = json.loads(captured.out.strip().splitlines()[-1])
    assert not stale.exists()
    assert (data_dir / "training" / "samples.jsonl").exists()
    assert (data_dir / "validation" / "samples.jsonl").exists()
    assert (data_dir / "test" / "samples.jsonl").exists()
    assert metadata["splits"]["training"]["statistics"]["count"] == 2
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
            "--train-count",
            "8",
            "--validation-count",
            "8",
            "--test-count",
            "8",
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

    with pytest.raises(ValueError, match="cannot provide at least"):
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
    assert checkpoint.with_name("model.best.pt").exists()
    assert metrics_csv.exists()
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["training_loss"]
    assert rows[0]["validation_loss"]

    assert main(
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "1",
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
    assert "2 discovery replay evaluations" in captured.err
    assert "[test] [4/8] Running 2 discovery replay evaluations" in captured.err
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
            "--batch-size",
            "1",
            "--petri-embedding-dim",
            "8",
            "--petri-num-walks",
            "2",
            "--petri-walk-length",
            "6",
            "--petri-epochs",
            "2",
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "ProcRosetta Test Report" in captured.out
    assert "Decode quality" in captured.out
    assert "Process discovery quality" in captured.out
    assert "pm4py Petri Node2Vec vs ProcRosetta fused" in captured.out
    assert "Agreement against ProcRosetta fused encoding" in captured.out
