from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_core_math import load_script


class RawInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = load_script("01_extract_motif_coordinates.py", "raw_input_analysis")

    def write_workbook(self, file, sheets):
        with pd.ExcelWriter(file) as writer:
            for name, values in sheets.items():
                pd.DataFrame(values).to_excel(writer, sheet_name=name, header=False, index=False)

    def test_single_sheet_workbook(self):
        raw = np.arange(400, dtype=float).reshape(20, 20) + 0.25
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Current": raw})
            records, log = self.primary.extract_raw_map(file)
        self.assertEqual(len(records), 1, log)
        np.testing.assert_array_equal(records[0].data, raw)

    def test_second_sheet_is_not_read(self):
        raw = np.arange(400, dtype=float).reshape(20, 20)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Current": raw, "Unused": -raw * 100})
            original_parse = pd.ExcelFile.parse
            calls = []

            def parse(workbook, *args, **kwargs):
                calls.append(kwargs["sheet_name"])
                return original_parse(workbook, *args, **kwargs)

            with patch.object(pd.ExcelFile, "parse", parse):
                records, log = self.primary.extract_raw_map(file)
        self.assertEqual(len(records), 1, log)
        self.assertEqual(calls, ["Current"])
        np.testing.assert_array_equal(records[0].data, raw)

    def test_explicit_sheet_name(self):
        raw = np.arange(400, dtype=float).reshape(20, 20)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Info": ["metadata"], "Current": raw})
            with patch.object(self.primary, "RAW_SHEET", "Current"):
                records, log = self.primary.extract_raw_map(file)
        self.assertEqual(len(records), 1, log)
        self.assertEqual(records[0].sheet, "Current")
        np.testing.assert_array_equal(records[0].data, raw)

    def test_invalid_first_sheet_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Info": ["metadata"], "Current": np.ones((20, 20))})
            records, log = self.primary.extract_raw_map(file)
        self.assertEqual(records, [])
        self.assertEqual(log[0]["status"], "read_error")

    def test_csv_unit_conversion(self):
        raw = np.arange(400, dtype=float).reshape(20, 20)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.csv"
            np.savetxt(file, raw, delimiter=",")
            with patch.object(self.primary, "NUMERIC_INPUT_MULTIPLIER", 1e-12):
                records, log = self.primary.extract_raw_map(file)
        self.assertEqual(len(records), 1, log)
        np.testing.assert_allclose(records[0].data, raw * 1e-12, atol=0)

    def test_excel_explicit_units_override_numeric_multiplier(self):
        raw = np.full((20, 20), "3pA", dtype=object)
        raw[0, :3] = ["3 pA", "0.003nA", 3]
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Current": raw})
            with patch.object(self.primary, "NUMERIC_INPUT_MULTIPLIER", 1e-12):
                records, log = self.primary.extract_raw_map(file)
        self.assertEqual(len(records), 1, log)
        np.testing.assert_allclose(records[0].data, 3e-12, rtol=1e-14, atol=0)

    def test_excel_unknown_unit_is_not_silently_filled(self):
        raw = np.full((20, 20), "3pA", dtype=object)
        raw[5, 5] = "3 unknown"
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Current": raw})
            records, log = self.primary.extract_raw_map(file)
        self.assertEqual(records, [])
        self.assertEqual(log[0]["status"], "read_error")

    def test_floor_uses_unmodified_finite_raw_pixels(self):
        y, x = np.indices((20, 20))
        raw = 8.0 + x * 0.7 + y * 0.4 + np.sin(x)
        raw[:6, :6] = np.nan
        original = raw.copy()
        corrected, metrics = self.primary.prepare_current_map(raw)
        self.assertAlmostEqual(metrics["raw_current_floor_p05"], np.nanpercentile(raw, 5))
        np.testing.assert_array_equal(raw, original)
        self.assertTrue(np.isfinite(corrected).all())
        self.assertFalse(np.shares_memory(corrected, raw))

    def test_plane_removal(self):
        y, x = np.indices((32, 32))
        raw = (50 + 0.2 * x - 0.3 * y) * 1e-12
        corrected, _ = self.primary.prepare_current_map(raw)
        np.testing.assert_allclose(corrected, 0, atol=1e-24)

    def test_plane_background_does_not_change_corrected_motif(self):
        raw = np.random.default_rng(4).normal(size=(32, 32))
        y, x = np.indices(raw.shape)
        first, _ = self.primary.prepare_current_map(raw)
        second, _ = self.primary.prepare_current_map(raw + 100 + x * 2 - y * 0.5)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_background_none_preserves_raw_values(self):
        raw = np.arange(400, dtype=float).reshape(20, 20)
        corrected, _ = self.primary.prepare_current_map(raw, "none")
        np.testing.assert_array_equal(corrected, raw)
        self.assertFalse(np.shares_memory(corrected, raw))

    def test_invalid_map_is_rejected(self):
        for raw in (np.full((20, 20), np.nan), np.full((20, 20), np.inf)):
            with self.assertRaises(ValueError):
                self.primary.prepare_current_map(raw)

    def test_excel_missing_pixels_are_preserved_until_preprocessing(self):
        raw = np.arange(400, dtype=float).reshape(20, 20)
        raw[2:5, 2:5] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "current.xlsx"
            self.write_workbook(file, {"Current": raw})
            records, log = self.primary.extract_raw_map(file)
        self.assertEqual(len(records), 1, log)
        np.testing.assert_array_equal(records[0].data, raw)
        _, metrics = self.primary.prepare_current_map(records[0].data)
        self.assertAlmostEqual(metrics["raw_current_floor_p05"], np.nanpercentile(raw, 5))

    def test_summary_contains_calculated_quantities(self):
        coordinates = pd.DataFrame({"name": ["a", "b"]})
        summary = pd.DataFrame({"metric": ["R_orthogonal_norm"], "mean": [2.0]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.primary.write_report(root, coordinates, summary, {})
            report = (root / "analysis_summary.txt").read_text(encoding="utf-8")
        self.assertIn("Processed maps: 2", report)
        self.assertIn("R_orthogonal_norm", report)

    def test_acquisition_names_include_relative_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("a", "b"):
                (root / folder).mkdir()
                np.savetxt(root / folder / "current.csv", np.ones((20, 20)), delimiter=",")
            with patch.object(self.primary, "INPUT_ROOT_FOR_PROVENANCE", root):
                records, _ = self.primary.extract_all_images(self.primary.list_input_files(root))
        self.assertEqual([r.name for r in records], ["a/current.csv", "b/current.csv"])


if __name__ == "__main__":
    unittest.main()
