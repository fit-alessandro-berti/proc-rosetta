import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sample_script_runs_without_install_bootstrap():
    data_dir = ROOT / ".tmp-test-data-script"
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "sample.py"),
            "--data-dir",
            str(data_dir),
            "--train-count",
            "1",
            "--validation-count",
            "1",
            "--test-count",
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

    metadata = json.loads(result.stdout.strip().splitlines()[-1])
    assert metadata["splits"]["training"]["statistics"]["count"] == 1
    assert (data_dir / "training" / "samples.jsonl").exists()
    import shutil

    shutil.rmtree(data_dir)


def test_train_script_runs_without_install_bootstrap():
    data_dir = ROOT / ".tmp-test-data-train-script"
    checkpoint = ROOT / ".tmp-test-checkpoints" / "model.pt"
    metrics_csv = ROOT / ".tmp-test-checkpoints" / "metrics.csv"
    import shutil

    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(checkpoint.parent, ignore_errors=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "sample.py"),
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
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
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
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    row = json.loads(result.stdout.strip().splitlines()[-1])
    assert row["epoch"] == 1
    assert row["training"]["loss"] > 0
    assert row["validation"]["loss"] > 0
    assert "generalization_gap" in row
    assert checkpoint.exists()
    assert metrics_csv.exists()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "test.py"),
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
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    test_row = json.loads(result.stdout.strip().splitlines()[-1])
    assert test_row["split"] == "test"
    assert test_row["loss_metrics"]["loss"] > 0
    assert "behavioral_distance_summary" in test_row
    assert "pm4py_colonna_petri_node2vec" in test_row["embedding_methods"]
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(checkpoint.parent, ignore_errors=True)
