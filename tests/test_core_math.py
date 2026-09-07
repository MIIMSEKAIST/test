from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CoreMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = load_script("01_extract_motif_coordinates.py", "primary_analysis")
        cls.readout = load_script("03_compute_readout_attenuation.py", "readout_analysis")

    def test_robust_zscore(self):
        values = np.array([[0.0, 1.0], [2.0, 3.0]])
        normalized, info = self.primary.robust_zscore(values)
        self.assertAlmostEqual(float(np.median(normalized)), 0.0)
        self.assertAlmostEqual(info["scale"], 1.4826)

    def test_contrast_coordinates_are_baseline_invariant(self):
        first = self.primary.motif_coordinates_from_sites(1.0, 2.0, -1.0)
        shifted = self.primary.motif_coordinates_from_sites(8.0, 9.0, 6.0)
        self.assertAlmostEqual(first["c_AH"], shifted["c_AH"])
        self.assertAlmostEqual(first["c_beta_signed"], shifted["c_beta_signed"])
        self.assertAlmostEqual(first["theta_beta_over_AH_deg"], shifted["theta_beta_over_AH_deg"])

    def test_contrast_coordinate_definition(self):
        result = self.primary.motif_coordinates_from_sites(1.0, 3.0, 0.0)
        self.assertAlmostEqual(result["c_AH"], 4.0 / math.sqrt(6.0))
        self.assertAlmostEqual(result["c_beta_signed"], 2.0 / math.sqrt(2.0))

    def test_reciprocal_shell_metric(self):
        self.assertEqual(self.readout.q2_120(1, 0), 1)
        self.assertEqual(self.readout.q2_120(1, 1), 3)
        self.assertEqual(self.readout.q2_120(2, 0), 4)
        shells = self.readout.generate_shell_indices()
        self.assertEqual(len(shells["G1"]), 6)
        self.assertEqual(len(shells["sqrt3G1"]), 6)

    def test_explicit_current_unit_overrides_numeric_multiplier(self):
        old = self.primary.NUMERIC_INPUT_MULTIPLIER
        try:
            self.primary.NUMERIC_INPUT_MULTIPLIER = 1e-12
            self.assertTrue(np.isclose(self.primary.parse_numeric_cell(3.0), 3e-12, rtol=1e-15, atol=0.0))
            self.assertTrue(np.isclose(self.primary.parse_numeric_cell("3 pA"), 3e-12, rtol=1e-15, atol=0.0))
            self.assertTrue(np.isclose(self.primary.parse_numeric_cell("2 nA"), 2e-9, rtol=1e-15, atol=0.0))
        finally:
            self.primary.NUMERIC_INPUT_MULTIPLIER = old


if __name__ == "__main__":
    unittest.main()
