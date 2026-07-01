import json

from proc_rosetta.cli import main


def test_sample_cli_outputs_json_lines(capsys):
    exit_code = main(
        [
            "sample",
            "--count",
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
    row = json.loads(captured.out.strip().splitlines()[-1])
    assert "tree" in row
    assert "traces" in row
    assert "petri_graph" in row


def test_train_cli_smoke(capsys):
    exit_code = main(
        [
            "train",
            "--samples",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--max-depth",
            "2",
            "--max-activities",
            "4",
            "--hidden-dim",
            "16",
            "--latent-dim",
            "8",
            "--traces-per-sample",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    row = json.loads(captured.out.strip().splitlines()[-1])
    assert row["epoch"] == 1
    assert row["loss"] > 0
