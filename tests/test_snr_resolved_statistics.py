from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "05_compute_snr_resolved_statistics.py"
    spec = importlib.util.spec_from_file_location("snr_resolved_statistics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_supplementary_module():
    path = ROOT / "scripts" / "04_compute_supplementary_source_data.py"
    spec = importlib.util.spec_from_file_location("supplementary_source_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SnrResolvedStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.supplementary = load_supplementary_module()

    def test_plus_convention_shell_definitions(self):
        expected_sqrt3 = (
            (-2, 1),
            (-1, -1),
            (-1, 2),
            (1, -2),
            (1, 1),
            (2, -1),
        )
        expected_noise = (
            (-3, 1),
            (-3, 2),
            (-2, -1),
            (-2, 3),
            (-1, -2),
            (-1, 3),
            (1, -3),
            (1, 2),
            (2, -3),
            (2, 1),
            (3, -2),
            (3, -1),
        )
        self.assertEqual(self.module.SQRT3_SHELL_HK, expected_sqrt3)
        self.assertTrue(all(self.module.q2_120(*mode) == 3 for mode in expected_sqrt3))
        noise = self.module.local_noise_indices()
        self.assertEqual(noise, expected_noise)
        self.assertTrue(all(self.module.q2_120(*mode) == 7 for mode in noise))

        self.assertEqual(tuple(self.supplementary.SQRT3_SHELL_HK), expected_sqrt3)
        self.assertEqual(tuple(self.supplementary.local_noise_hks()), expected_noise)
        self.assertTrue(
            all(self.supplementary.q2_120(*mode) == 3 for mode in expected_sqrt3)
        )

    def test_direct_dft_recovers_integer_mode(self):
        n = 32
        u = np.arange(n) / n
        v = np.arange(n) / n
        uu, vv = np.meshgrid(u, v)
        image = np.cos(2 * np.pi * (uu - vv))
        coefficients = self.module.direct_dft_coefficients(image, [(1, -1), (0, 1)])
        self.assertAlmostEqual(abs(coefficients[0]), 0.5, places=12)
        self.assertAlmostEqual(abs(coefficients[1]), 0.0, places=12)

    def test_threshold_is_inclusive_and_join_is_order_independent(self):
        readout = pd.DataFrame(
            {
                "name": ["a", "b", "c"],
                "array_index": [0, 1, 2],
                "w_rel_primary": [0.1, 0.2, 0.3],
                "eta_rel_logamp": [0.3, np.nan, 0.1],
                "raw_current_floor_log10_abs_p05": [-10.0, -9.5, -9.0],
            }
        )
        snr = pd.DataFrame(
            {
                "array_index": [2, 0, 1],
                "name": ["c", "a", "b"],
                "sqrt3_amp_snr": [6.0, 4.9, 5.0],
                "sqrt3_resolved": [True, False, True],
            }
        )
        merged = self.module.merge_readout_and_snr(readout, snr, threshold=5.0)
        self.assertEqual(merged["name"].tolist(), ["c", "a", "b"])
        self.assertEqual(merged["analysis_keep"].tolist(), [True, False, True])
        self.assertAlmostEqual(float(merged.loc[merged["name"] == "b", "w"].iloc[0]), 0.2)

    def test_nonfinite_snr_is_not_resolved(self):
        readout = pd.DataFrame(
            {
                "name": ["a"],
                "array_index": [0],
                "w_rel_primary": [0.1],
                "eta_rel_logamp": [0.2],
                "raw_current_floor_log10_abs_p05": [-10.0],
            }
        )
        snr = pd.DataFrame(
            {
                "array_index": [0],
                "name": ["a"],
                "sqrt3_amp_snr": [np.inf],
                "sqrt3_resolved": [False],
            }
        )
        merged = self.module.merge_readout_and_snr(readout, snr)
        self.assertFalse(bool(merged.loc[0, "sqrt3_resolved"]))
        self.assertFalse(bool(merged.loc[0, "analysis_keep"]))

    def test_reference_columns_are_required(self):
        readout = pd.DataFrame(
            {
                "name": ["a"],
                "array_index": [0],
                "w_rel_logamp_model": [0.1],
                "eta_rel_logamp": [0.2],
                "raw_current_floor_log10_abs_p05": [-10.0],
            }
        )
        snr = pd.DataFrame(
            {
                "array_index": [0],
                "name": ["a"],
                "sqrt3_amp_snr": [7.0],
                "sqrt3_resolved": [True],
            }
        )
        with self.assertRaisesRegex(KeyError, "w_rel_primary"):
            self.module.merge_readout_and_snr(readout, snr)

    def test_fractional_array_index_is_rejected(self):
        readout = pd.DataFrame(
            {
                "name": ["a"],
                "array_index": [0.5],
                "w_rel_primary": [0.1],
                "eta_rel_logamp": [0.2],
                "raw_current_floor_log10_abs_p05": [-10.0],
            }
        )
        snr = pd.DataFrame(
            {
                "array_index": [0],
                "name": ["a"],
                "sqrt3_amp_snr": [7.0],
                "sqrt3_resolved": [True],
            }
        )
        with self.assertRaisesRegex(ValueError, "integer"):
            self.module.merge_readout_and_snr(readout, snr)

    def test_array_index_mismatch_is_rejected(self):
        readout = pd.DataFrame(
            {
                "name": ["a"],
                "array_index": [1],
                "w_rel_primary": [0.1],
                "eta_rel_logamp": [0.2],
                "raw_current_floor_log10_abs_p05": [-10.0],
            }
        )
        snr = pd.DataFrame(
            {
                "array_index": [0],
                "name": ["a"],
                "sqrt3_amp_snr": [7.0],
                "sqrt3_resolved": [True],
            }
        )
        with self.assertRaisesRegex(ValueError, "array_index"):
            self.module.merge_readout_and_snr(readout, snr)

    def test_missing_names_are_rejected_before_string_conversion(self):
        readout = pd.DataFrame(
            {
                "name": [np.nan],
                "array_index": [0],
                "w_rel_primary": [0.1],
                "eta_rel_logamp": [0.2],
                "raw_current_floor_log10_abs_p05": [-10.0],
            }
        )
        snr = pd.DataFrame(
            {
                "array_index": [0],
                "name": [np.nan],
                "sqrt3_amp_snr": [7.0],
                "sqrt3_resolved": [True],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing or empty names"):
            self.module.merge_readout_and_snr(readout, snr)


if __name__ == "__main__":
    unittest.main()
