from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_primary_module():
    path = ROOT / "scripts" / "01_extract_motif_coordinates.py"
    spec = importlib.util.spec_from_file_location("primary_fixed_registry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedEnsembleRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = load_primary_module()

    def test_production_defaults_use_fixed_ensemble_registry(self):
        self.assertEqual(self.primary.SITE_COORDINATE_MODE, "ensemble_mean_fixed")
        self.assertFalse(self.primary.REFINE_A_SITE_LOCAL_AFTER_PICK)

    def test_calibration_search_receives_only_the_ensemble_mean(self):
        maps = np.stack(
            [
                np.full((4, 4), 1.0),
                np.full((4, 4), 3.0),
                np.full((4, 4), 8.0),
            ],
            axis=0,
        )
        expected_mean = np.mean(maps, axis=0)
        captured = {}

        def fake_search(search_maps):
            captured["maps"] = np.asarray(search_maps).copy()
            return (
                {
                    "basis": "test_basis",
                    "origin": (0.25, 0.5),
                    "positions_role": {
                        "B": (0.0, 0.0),
                        "A": (2.0 / 3.0, 2.0 / 3.0),
                        "H": (1.0 / 3.0, 1.0 / 3.0),
                    },
                },
                pd.DataFrame([{"score": 1.0}]),
            )

        original = self.primary.grid_search_site_origin_vectorized
        try:
            self.primary.grid_search_site_origin_vectorized = fake_search
            choice, search_df, ensemble_mean = self.primary.site_choice_from_phase_aligned_ensemble_mean(maps)
        finally:
            self.primary.grid_search_site_origin_vectorized = original

        np.testing.assert_allclose(ensemble_mean, expected_mean)
        self.assertEqual(captured["maps"].shape, (1, 4, 4))
        np.testing.assert_allclose(captured["maps"][0], expected_mean)
        self.assertEqual(choice["site_coordinate_mode"], "ensemble_mean_fixed")
        self.assertTrue(choice["site_positions_fixed_across_acquisitions"])
        self.assertEqual(int(search_df.loc[0, "n_images_in_ensemble_mean"]), 3)

    def test_every_acquisition_receives_identical_coordinates(self):
        choice = {
            "positions_role": {
                "B": (0.0, 0.0),
                "A": (2.0 / 3.0, 2.0 / 3.0),
                "H": (1.0 / 3.0, 1.0 / 3.0),
                "geoA": (2.0 / 3.0, 2.0 / 3.0),
                "geoB": (0.0, 0.0),
                "geoH": (1.0 / 3.0, 1.0 / 3.0),
            }
        }
        positions = self.primary.fixed_registry_positions_for_acquisitions(choice, 531)
        reference = tuple(positions[0][site] for site in ("B", "A", "H"))
        self.assertEqual(len(positions), 531)
        self.assertTrue(all(tuple(row[site] for site in ("B", "A", "H")) == reference for row in positions))
        self.assertIsNot(positions[0], positions[1])


if __name__ == "__main__":
    unittest.main()
