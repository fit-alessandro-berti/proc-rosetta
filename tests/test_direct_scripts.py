import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sample_script_runs_without_install_bootstrap():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sample.py"),
            "--count",
            "1",
            "--max-depth",
            "2",
            "--traces-per-sample",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert "tree" in row
    assert "petri_graph" in row


def test_train_script_runs_without_install_bootstrap():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
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
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row["epoch"] == 1
    assert row["loss"] > 0
