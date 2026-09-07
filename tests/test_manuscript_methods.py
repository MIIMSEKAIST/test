from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from test_core_math import load_script


class ManuscriptMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = load_script("01_extract_motif_coordinates.py", "primary_hex120")
        cls.readout = load_script("03_compute_readout_attenuation.py", "readout_coherent")
        cls.reconstruction = load_script("02_validate_quantile_reconstruction.py", "reconstruction_hex120")

    def setUp(self):
        shells = self.readout.generate_shell_indices()
        self.modes = [hk for key in ("G1", "sqrt3G1", "2G1") for hk in shells[key]]
        self.hs, self.ks = np.array(self.modes).T
        self.q2 = np.array([self.readout.q2_120(*hk) for hk in self.modes])
        self.reference = np.ones(len(self.modes), dtype=complex)

    def test_compact_and_spaced_units(self):
        cases = {"3pA": 3e-12, "3 pA": 3e-12, "2nA": 2e-9, "2 nA": 2e-9,
                 "-4.5fA": -4.5e-15, "1.2e-3nA": 1.2e-12, "2µA": 2e-6,
                 "2μA": 2e-6, "1.2 x 10^-12 A": 1.2e-12, "10⁻¹²A": 1e-12,
                 "1,234pA": 1234e-12}
        for text, expected in cases.items():
            with self.subTest(text=text):
                np.testing.assert_allclose(self.primary.parse_numeric_cell(text), expected, rtol=1e-14, atol=0)

    def test_unknown_units_and_embedded_labels_are_rejected(self):
        for text in ("3 unknown", "3V", "current 3 pA", "3pA text", "1,2pA", "3MA"):
            with self.subTest(text=text):
                self.assertTrue(np.isnan(self.primary.parse_numeric_cell(text)))

    def test_mask_uses_120_degree_cartesian_distance(self):
        self.assertEqual(self.primary.SITE_MASK_DISTANCE_METRIC, "hex120")
        n, sigma = 32, self.primary.SITE_AVERAGE_SIGMA_FRAC
        u, v = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
        pos = (0.97, 0.02)
        distances = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                du, dv = u - pos[0] + a, v - pos[1] + b
                x, y = du - dv / 2, math.sqrt(3) * dv / 2
                distances.append(x * x + y * y)
        d2 = np.min(distances, axis=0)
        expected = np.exp(-d2 / (2 * sigma**2))
        expected /= expected.sum()
        actual = self.primary.periodic_gaussian_weight(n, pos)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        reconstructed_d2 = self.reconstruction._periodic_distance2_uv_grid(u, v, *pos)
        np.testing.assert_allclose(reconstructed_d2, d2, atol=1e-15)

    def test_fixed_registry_does_not_enable_soft_site_correction(self):
        self.assertEqual(self.primary.SITE_COORDINATE_MODE, "ensemble_mean_fixed")
        self.assertFalse(self.primary.USE_SITE_PICKING_UNCERTAINTY_CORRECTION)
        self.assertFalse(self.primary.USE_SOFT_SITE_COORDINATES_AS_PRIMARY)
        self.assertFalse(self.primary.REFINE_A_SITE_LOCAL_AFTER_PICK)

    def test_mask_is_periodic(self):
        first = self.primary.periodic_gaussian_weight(32, (0.97, 0.02))
        shifted = self.primary.periodic_gaussian_weight(32, (1.97, -0.98))
        np.testing.assert_allclose(first, shifted, atol=1e-15)

    def test_vectorized_and_scalar_masks_agree(self):
        maps = np.random.default_rng(81).normal(size=(3, 32, 32))
        pu, pv = np.array([0.97, 0.21, 0.5]), np.array([0.02, 0.83, 0.5])
        values = self.primary._site_values_vectorized_for_positions(maps, pu, pv, np)
        expected = [[self.primary.site_mean(m, (u, v)) for m in maps] for u, v in zip(pu, pv)]
        np.testing.assert_allclose(values, expected, atol=2e-6)

    def test_reference_averages_complex_coefficients(self):
        values = np.stack([self.reference, self.reference * 1j])
        ref = self.readout.coherent_reference(values, np.array([0, 1]), np.ones(2))
        np.testing.assert_allclose(ref, (1 + 1j) / 2)
        self.assertFalse(np.allclose(np.abs(ref), np.mean(np.abs(values), axis=0)))

    def test_leave_one_out_excludes_target_even_in_small_ensembles(self):
        theta = np.array([1.0, 2.0, 3.0, np.nan])
        for i in range(3):
            indices, weights = self.readout.neighbor_indices(theta, i, 51, None, None, 15)
            self.assertNotIn(i, indices)
            self.assertNotIn(3, indices)
            self.assertEqual(len(indices), 2)
            np.testing.assert_array_equal(weights, np.ones(2))
        with self.assertRaises(ValueError):
            self.readout.neighbor_indices(np.array([1.0]), 0, 51, None, None, 15)

    def test_w_recovers_known_isotropic_attenuation(self):
        expected = 0.23
        observed = 2 * np.exp(-expected * (self.q2 - 1)).astype(complex)
        result, _ = self.readout.readout_from_coherent_reference(observed, self.reference, self.hs, self.ks)
        self.assertAlmostEqual(result["w_rel_primary"], expected, places=13)
        self.assertAlmostEqual(result["eta_rel_logamp"], 0, places=13)

    def test_shell_amplitudes_are_arithmetic_means(self):
        observed = self.reference.copy()
        observed[self.q2 == 3] = [1, 2, 9, 1, 2, 9]
        result, _ = self.readout.readout_from_coherent_reference(observed, self.reference, self.hs, self.ks)
        self.assertAlmostEqual(result["w_rel_primary"], -0.5 * math.log(4), places=13)

    def test_eta_recovers_three_direction_model(self):
        kc, ks = 0.12, -0.08
        observed = self.reference.copy()
        for j, hk in enumerate(self.modes):
            if self.q2[j] == 3:
                gx, gy = self.readout.reciprocal_cart_120(*hk)
                angle = math.atan2(gy, gx)
                observed[j] = np.exp(0.2 - 1.5 * (kc * np.cos(2 * angle) + ks * np.sin(2 * angle)))
        result, rows = self.readout.fit_sqrt3_anisotropy(observed, self.reference, self.hs, self.ks)
        self.assertEqual(len(rows), 3)
        self.assertEqual(result["eta_fit_residual_dof"], 0)
        self.assertAlmostEqual(result["kappa_c"], kc, places=13)
        self.assertAlmostEqual(result["kappa_s"], ks, places=13)
        self.assertAlmostEqual(result["eta_rel_logamp"], math.hypot(kc, ks) / 2, places=13)
        phi = math.radians(result["eta_rel_logamp_orientation_deg"])
        self.assertAlmostEqual(result["eta_rel_logamp"] * math.cos(2 * phi), kc / 2, places=13)
        self.assertAlmostEqual(result["eta_rel_logamp"] * math.sin(2 * phi), ks / 2, places=13)
        observed[self.q2 != 3] *= np.linspace(1e-8, 1e8, np.count_nonzero(self.q2 != 3))
        changed, _ = self.readout.fit_sqrt3_anisotropy(observed, self.reference, self.hs, self.ks)
        self.assertEqual(changed["eta_rel_logamp"], result["eta_rel_logamp"])

    def test_mode_order_and_common_gain_do_not_change_results(self):
        observed = np.exp(-0.13 * self.q2).astype(complex)
        a, _ = self.readout.readout_from_coherent_reference(observed, self.reference, self.hs, self.ks)
        order = np.random.default_rng(19).permutation(len(self.hs))
        b, _ = self.readout.readout_from_coherent_reference(observed[order] * 7, self.reference[order] * 7, self.hs[order], self.ks[order])
        for key in ("w_rel_primary", "eta_rel_logamp"):
            self.assertAlmostEqual(a[key], b[key], places=13)

    def test_cancelled_reference_is_not_replaced_by_a_positive_floor(self):
        reference = self.reference.copy()
        reference[self.q2 == 3] = 0
        result, _ = self.readout.readout_from_coherent_reference(self.reference, reference, self.hs, self.ks)
        self.assertTrue(np.isnan(result["w_rel_primary"]))
        self.assertTrue(np.isnan(result["eta_rel_logamp"]))
        self.assertFalse(result["w_fit_ok"])
        self.assertFalse(result["eta_fit_ok"])

    def test_file_pipeline_uses_coherent_mean_and_three_directions(self):
        n, count = 32, 5
        maps = []
        for i in range(count):
            spectrum = np.zeros((n, n), dtype=complex)
            for h, k in self.modes:
                q2 = self.readout.q2_120(h, k)
                gx, gy = self.readout.reciprocal_cart_120(h, k)
                phi = math.atan2(gy, gx)
                amp = np.exp(-0.02 * i * q2 - 0.03 * i * np.cos(2 * phi))
                spectrum[k % n, h % n] = amp * np.exp(1j * (0.07 * i * h + 0.03 * i * k))
            maps.append(np.fft.ifft2(spectrum).real * n * n)
        maps = np.asarray(maps)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = np.array([f"map_{i}" for i in range(count)])
            np.savez(root / "motif_analysis_arrays.npz", aligned_norm=maps, names=names)
            pd.DataFrame({"name": names, "theta_beta_over_AH_deg_norm": np.arange(count)}).to_csv(root / "orthogonal_motif_coordinates_site_currents.csv", index=False)
            paths = self.readout.compute_template_relative(root, skip_complex_fit=True, no_plots=False, skip_excel=True)
            table = pd.read_csv(paths["per_image"])
            directions = pd.read_csv(paths["out_dir"] / "sqrt3_directional_fit.csv")
            neighbors = pd.read_csv(paths["out_dir"] / "template_neighbor_table_long.csv")
            self.assertEqual(len(directions), count * 3)
            self.assertTrue((neighbors["array_index"] != neighbors["neighbor_array_index"]).all())
            self.assertTrue(table["readout_estimator"].eq(self.readout.READOUT_ESTIMATOR).all())
            self.assertNotIn("logamp_residual", pd.read_csv(paths["correlations"])["y_metric"].tolist())
            self.assertTrue(list((paths["out_dir"] / "preview_png").glob("*.png")))
            transforms = np.fft.fft2(maps, axes=(-2, -1)) / (n * n)
            for i in range(count):
                ref = np.mean(np.delete(transforms, i, axis=0), axis=0)
                means = []
                for ft in (transforms[i], ref):
                    means.append([np.mean([abs(ft[k % n, h % n]) for h, k in self.modes if self.readout.q2_120(h, k) == shell]) for shell in (1, 3)])
                expected = -0.5 * (math.log(means[0][1] / means[0][0]) - math.log(means[1][1] / means[1][0]))
                self.assertAlmostEqual(table.loc[i, "w_rel_primary"], expected, places=12)


if __name__ == "__main__":
    unittest.main()
