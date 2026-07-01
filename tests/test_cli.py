import json

from proc_rosetta.cli import main


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
    row = json.loads(captured.out.strip().splitlines()[-1])
    assert row["epoch"] == 1
    assert row["training"]["loss"] > 0
    assert row["validation"]["loss"] > 0
    assert checkpoint.exists()

    assert main(
        [
            "test",
            "--data-dir",
            str(data_dir),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "1",
        ]
    ) == 0
    captured = capsys.readouterr()
    test_row = json.loads(captured.out.strip().splitlines()[-1])
    assert test_row["split"] == "test"
    assert test_row["loss"] > 0
