from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_manifest_module():
    path = ROOT / "scripts" / "00_make_input_manifest.py"
    spec = importlib.util.spec_from_file_location("input_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ManifestTests(unittest.TestCase):
    def test_manifest_is_ordered_and_hashed(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            (raw / "b.xlsx").write_bytes(b"second")
            (raw / "A.xlsm").write_bytes(b"first")
            (raw / "c.csv").write_bytes(b"third")
            (raw / "ignored.txt").write_bytes(b"ignored")
            output = root / "manifest.csv"

            count = module.build_manifest(raw, output)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(count, 3)
            self.assertEqual([row["relative_path"] for row in rows], ["A.xlsm", "b.xlsx", "c.csv"])
            self.assertEqual(rows[0]["sha256"], hashlib.sha256(b"first").hexdigest())
            self.assertEqual(rows[1]["pipeline_order"], "2")


if __name__ == "__main__":
    unittest.main()
