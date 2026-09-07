"""Record raw-map file checksums and the Python environment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


INPUT_SUFFIXES = {".xlsx", ".xlsm", ".csv"}
CHUNK_SIZE = 1024 * 1024
PACKAGES = ("numpy", "pandas", "scipy", "matplotlib", "openpyxl", "scikit-learn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in INPUT_SUFFIXES
            and not path.name.startswith("~$")
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def build_manifest(root: Path, output: Path) -> int:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Write the manifest outside the raw-data directory")
    files = input_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pipeline_order",
                "relative_path",
                "size_bytes",
                "modified_utc",
                "sha256",
            ],
        )
        writer.writeheader()
        for index, path in enumerate(files, start=1):
            stat = path.stat()
            writer.writerow(
                {
                    "pipeline_order": index,
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": sha256(path),
                }
            )
    return len(files)


def write_environment(path: Path) -> None:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "packages": packages,
        "entry_point": "00_make_input_manifest.py",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--environment-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.input_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")
    count = build_manifest(root, args.output_csv.expanduser().resolve())
    if args.environment_json is not None:
        write_environment(args.environment_json.expanduser().resolve())
    print(f"Wrote {count} entries to {args.output_csv}")


if __name__ == "__main__":
    main()
