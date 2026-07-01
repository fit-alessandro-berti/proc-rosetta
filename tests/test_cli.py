import json
import csv

from proc_rosetta.cli import main
from proc_rosetta.cli import build_parser, split_counts_from_args


def test_default_sample_and_train_values_match_recommended_run():
    parser = build_parser()

    sample_args = parser.parse_args(["sample"])
    train_args = parser.parse_args(["train"])
    split_counts = split_counts_from_args(sample_args)

    assert split_counts.training == 2000
    assert split_counts.validation == 256
    assert split_counts.test == 256
    assert sample_args.traces_per_sample == 16
    assert train_args.epochs == 20
    assert train_args.batch_size == 32
    assert train_args.dropout == 0.15
    assert train_args.weight_decay == 1e-4
    assert train_args.label_smoothing == 0.05
    assert train_args.early_stopping_patience == 5


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
    assert "pm4py Petri Node2Vec vs ProcRosetta fused" in captured.out
    assert "Agreement against ProcRosetta fused encoding" in captured.out
