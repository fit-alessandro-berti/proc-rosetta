from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


def _checkpoint_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.glob("*.pt") if path.is_file() and path.stat().st_size > 0
    )


def _copy_local(source: Path, output: Path) -> bool:
    if not source.is_dir() or not _checkpoint_files(source):
        return False
    shutil.copytree(source, output, dirs_exist_ok=True)
    return True


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "proc-rosetta-docker-build/1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha256 and actual.casefold() != expected_sha256.casefold():
        raise RuntimeError(
            f"checkpoint archive SHA-256 mismatch: expected {expected_sha256}, received {actual}"
        )


def _copy_archive(archive: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="proc-rosetta-checkpoints-") as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive, mode="r:gz") as bundle:
            bundle.extractall(extracted, filter="data")
        checkpoints = sorted(
            path for path in extracted.rglob("*.pt") if path.is_file() and path.stat().st_size > 0
        )
        if not checkpoints:
            raise RuntimeError("downloaded archive contains no non-empty .pt checkpoints")
        for checkpoint in checkpoints:
            shutil.copy2(checkpoint, output / checkpoint.name)
        for metrics in extracted.rglob("training_metrics.csv"):
            if metrics.is_file():
                shutil.copy2(metrics, output / metrics.name)
                break


def prepare(local: Path, output: Path, url: str, expected_sha256: str = "") -> str:
    output.mkdir(parents=True, exist_ok=True)
    if _copy_local(local, output):
        source = f"local directory {local}"
    else:
        with tempfile.TemporaryDirectory(prefix="proc-rosetta-download-") as temporary:
            archive = Path(temporary) / "checkpoint_rosetta_latest.tar.gz"
            _download(url, archive, expected_sha256.strip())
            _copy_archive(archive, output)
        source = url
    checkpoints = _checkpoint_files(output)
    if not checkpoints:
        raise RuntimeError("checkpoint preparation produced no usable .pt files")
    return f"Prepared {len(checkpoints)} checkpoint(s) from {source}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", default="")
    arguments = parser.parse_args()
    print(prepare(arguments.local, arguments.output, arguments.url, arguments.sha256))


if __name__ == "__main__":
    main()
