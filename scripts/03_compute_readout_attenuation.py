# -*- coding: utf-8 -*-
"""Template-relative higher-order attenuation analysis for 120-degree graphite cells.

For each folded unit cell, the first and sqrt(3) reciprocal-lattice shells are
compared with a local theta-conditioned leave-one-out reference.  The script
exports the relative isotropic descriptor w and anisotropic descriptor eta."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:  # pragma: no cover
    spearmanr = None
    pearsonr = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

DEFAULT_MAP_KEY = "aligned_norm"
DEFAULT_OUT_FOLDER = "readout_attenuation_120deg"
SAFE_EPS = 1e-300
READOUT_ESTIMATOR = "coherent_mean_sqrt3_v1"

# =============================================================================
# General helpers
# =============================================================================


def _decode_name(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def corrcoef2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 3:
        return np.nan
    aa = aa[m] - np.nanmean(aa[m])
    bb = bb[m] - np.nanmean(bb[m])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den <= 0:
        return np.nan
    return float(np.dot(aa, bb) / den)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    x = x[m]
    w = w[m]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w) / np.sum(w)
    return float(np.interp(0.5, cw, x))


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    x = x[m]
    w = w[m]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w) / np.sum(w)
    return float(np.interp(float(q), cw, x))


def robust_log_geomean(vals: Sequence[float]) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    return float(np.exp(np.nanmedian(np.log(np.maximum(v, SAFE_EPS)))))


def find_col(df: pd.DataFrame, candidates: Sequence[str], label: str, required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        cl = str(c).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    if required:
        raise KeyError(f"Could not find {label}. Tried {candidates}. Available columns include: {list(df.columns)[:80]}")
    return None


def stat_corr(x, y) -> Tuple[float, float, float, float, int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    n = int(m.sum())
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan, n
    if spearmanr is not None:
        sr = spearmanr(xx[m], yy[m])
        rho_s, p_s = float(sr.correlation), float(sr.pvalue)
    else:
        rx = pd.Series(xx[m]).rank().to_numpy()
        ry = pd.Series(yy[m]).rank().to_numpy()
        rho_s = float(np.corrcoef(rx, ry)[0, 1])
        p_s = np.nan
    if pearsonr is not None:
        pr = pearsonr(xx[m], yy[m])
        rho_p, p_p = float(pr.statistic), float(pr.pvalue)
    else:
        rho_p = float(np.corrcoef(xx[m], yy[m])[0, 1])
        p_p = np.nan
    return rho_s, p_s, rho_p, p_p, n

# =============================================================================
# 120° reciprocal helpers
# =============================================================================


def q2_120(h: int, k: int) -> int:
    return int(h * h + h * k + k * k)


def reciprocal_cart_120(h: float, k: float) -> Tuple[float, float]:
    # direct angle = 120°. a1=(1,0), a2=(-1/2,sqrt(3)/2)
    return float(h), float((2.0 * k + h) / math.sqrt(3.0))


def g1_norm() -> float:
    gx, gy = reciprocal_cart_120(1, 0)
    return math.sqrt(gx * gx + gy * gy)


def generate_shell_indices(max_abs_index: int = 4) -> Dict[str, List[Tuple[int, int]]]:
    targets = {"G1": 1, "sqrt3G1": 3, "2G1": 4}
    shells = {k: [] for k in targets}
    for h in range(-max_abs_index, max_abs_index + 1):
        for k in range(-max_abs_index, max_abs_index + 1):
            if h == 0 and k == 0:
                continue
            q2 = q2_120(h, k)
            for shell, tq in targets.items():
                if q2 == tq:
                    shells[shell].append((h, k))
    for shell in shells:
        shells[shell] = sorted(shells[shell], key=lambda p: math.atan2(reciprocal_cart_120(*p)[1], reciprocal_cart_120(*p)[0]))
    return shells


def shell_indices_table(shells: Dict[str, List[Tuple[int, int]]]) -> pd.DataFrame:
    g1 = g1_norm()
    rows = []
    for shell, inds in shells.items():
        for h, k in inds:
            gx, gy = reciprocal_cart_120(h, k)
            rows.append({
                "shell": shell,
                "h": h,
                "k": k,
                "q2_120": q2_120(h, k),
                "Gx_normG1": gx / g1,
                "Gy_normG1": gy / g1,
                "G_norm_over_G1": math.sqrt(gx * gx + gy * gy) / g1,
                "angle_deg": math.degrees(math.atan2(gy, gx)),
            })
    return pd.DataFrame(rows)


def fft_coeff(unit_map: np.ndarray, h: int, k: int) -> complex:
    arr = np.asarray(unit_map, dtype=float)
    arr = arr - np.nanmean(arr)
    F = np.fft.fft2(np.nan_to_num(arr, nan=0.0)) / arr.size
    ny, nx = arr.shape
    return complex(F[k % ny, h % nx])


def compute_all_coeffs(maps: np.ndarray, names: Sequence[str], shells: Dict[str, List[Tuple[int, int]]]) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], np.ndarray]]:
    modes = []
    for shell in ["G1", "sqrt3G1", "2G1"]:
        for h, k in shells[shell]:
            modes.append((shell, h, k, q2_120(h, k)))
    coeff_by_mode: Dict[Tuple[int, int], List[complex]] = {(h, k): [] for _, h, k, _ in modes}
    rows = []
    for i, (nm, m) in enumerate(zip(names, maps)):
        for shell, h, k, q2 in modes:
            c = fft_coeff(m, h, k)
            coeff_by_mode[(h, k)].append(c)
            amp = abs(c)
            power = amp * amp
            gx, gy = reciprocal_cart_120(h, k)
            rows.append({
                "name": nm,
                "array_index": i,
                "shell": shell,
                "h": h,
                "k": k,
                "q2_120": q2,
                "Gx_normG1": gx / g1_norm(),
                "Gy_normG1": gy / g1_norm(),
                "coeff_real": c.real,
                "coeff_imag": c.imag,
                "coeff_abs": amp,
                "coeff_power": power,
                "log_coeff_abs": math.log(max(amp, SAFE_EPS)),
                "log10_power": math.log10(max(power, SAFE_EPS)),
            })
    coeff_arr = {mk: np.asarray(v, dtype=complex) for mk, v in coeff_by_mode.items()}
    return pd.DataFrame(rows), coeff_arr

# =============================================================================
# Template/reference helpers
# =============================================================================


def neighbor_indices(theta: np.ndarray, i: int, template_n: int, theta_window_deg: Optional[float], theta_sigma_deg: Optional[float], min_template_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select finite-theta neighbors without ever including the target image."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or not 0 <= i < len(theta) or not np.isfinite(theta[i]):
        raise ValueError("The target must have a finite theta value")
    if template_n < 1 or min_template_n < 1:
        raise ValueError("Reference neighborhood sizes must be positive")
    for value in (theta_window_deg, theta_sigma_deg):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("Theta window and sigma must be finite and positive")
    candidates = np.flatnonzero(np.isfinite(theta) & (np.arange(len(theta)) != i))
    if not len(candidates):
        raise ValueError("At least one other finite-theta image is required")
    delta = np.abs(theta - theta[i])
    order = candidates[np.argsort(delta[candidates], kind="stable")]
    if theta_window_deg is not None:
        idx = order[delta[order] <= theta_window_deg]
        if len(idx) < min_template_n:
            idx = order[:min_template_n]
        idx = idx[:template_n]
    else:
        idx = order[:template_n]
    weights = np.ones(len(idx), dtype=float)
    if theta_sigma_deg is not None:
        log_weights = -0.5 * (delta[idx] / theta_sigma_deg) ** 2
        weights = np.exp(log_weights - np.max(log_weights))
    return idx.astype(int), weights


def weighted_complex_mean(z: np.ndarray, w: np.ndarray) -> complex:
    z = np.asarray(z, dtype=complex)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(z.real) & np.isfinite(z.imag) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return complex(np.nan, np.nan)
    return complex(np.sum(w[m] * z[m]) / np.sum(w[m]))


def positive_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    valid = np.isfinite(values) & (values > 0)
    np.log(values, out=out, where=valid)
    return out


def shell_log_ratio_from_mode_df(mode_df: pd.DataFrame) -> pd.Series:
    """Log ratio of arithmetic mean amplitudes, not median log amplitudes."""
    pivot = mode_df.groupby(["array_index", "shell"])["coeff_abs"].mean().unstack()
    return pd.Series(
        positive_log(pivot["sqrt3G1"]) - positive_log(pivot["G1"]),
        index=pivot.index,
    )


# =============================================================================
# Fit functions
# =============================================================================


def complex_fit_phase_aligned(
    obs: np.ndarray,
    templ: np.ndarray,
    hs: np.ndarray,
    ks: np.ndarray,
    q2s: np.ndarray,
    shells: np.ndarray,
    w_grid: np.ndarray,
    du_grid: np.ndarray,
    dv_grid: np.ndarray,
) -> Dict[str, float]:
    """Diagnostic complex fit with optional phase-shift grid search.

    F_obs ~= C * exp[-w(q2-1)] * exp[-2pi i(h du+k dv)] * F_template
    Residuals are shell-balanced by observed shell amplitude scale.
    """
    obs = np.asarray(obs, dtype=complex)
    templ = np.asarray(templ, dtype=complex)
    hs = np.asarray(hs, dtype=float)
    ks = np.asarray(ks, dtype=float)
    q2s = np.asarray(q2s, dtype=float)
    shells = np.asarray(shells, dtype=object)

    finite = np.isfinite(obs.real) & np.isfinite(obs.imag) & np.isfinite(templ.real) & np.isfinite(templ.imag) & (np.abs(templ) > 0)
    if finite.sum() < 6:
        return {"fit_ok": False, "message": "too_few_modes"}

    obs = obs[finite]
    templ = templ[finite]
    hs = hs[finite]
    ks = ks[finite]
    q2s = q2s[finite]
    shells = shells[finite]

    # Shell-balanced weights.
    weights = np.zeros(obs.size, dtype=float)
    for sh in np.unique(shells):
        mask = shells == sh
        scale = np.nanmedian(np.abs(obs[mask]))
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nanmedian(np.abs(obs))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        weights[mask] = 1.0 / (scale * scale * max(1, mask.sum()))
    weights = weights / np.nanmean(weights)

    best = {"score": np.inf, "w": np.nan, "du": np.nan, "dv": np.nan, "C": np.nan + 1j * np.nan}
    sqrtw = np.sqrt(weights)
    yw = obs * sqrtw
    denom_y = np.sum(np.abs(yw) ** 2)
    if denom_y <= 0:
        denom_y = 1.0

    for du in du_grid:
        for dv in dv_grid:
            phase = np.exp(-2j * np.pi * (hs * du + ks * dv))
            t0 = templ * phase
            for w in w_grid:
                decay = np.exp(-float(w) * (q2s - 1.0))
                x = t0 * decay
                xw = x * sqrtw
                den = np.sum(np.abs(xw) ** 2)
                if den <= 0 or not np.isfinite(den):
                    continue
                C = np.sum(np.conj(xw) * yw) / den
                resid = yw - C * xw
                score = float(np.sum(np.abs(resid) ** 2) / denom_y)
                if np.isfinite(score) and score < best["score"]:
                    best = {"score": score, "w": float(w), "du": float(du), "dv": float(dv), "C": C}

    C = best["C"]
    return {
        "fit_ok": bool(np.isfinite(best["score"])),
        "message": "ok" if np.isfinite(best["score"]) else "no_finite_score",
        "w_rel_complex_phase_aligned": float(best["w"]),
        "phase_shift_u_frac": float(best["du"]),
        "phase_shift_v_frac": float(best["dv"]),
        "complex_scale_real": float(np.real(C)) if np.isfinite(np.real(C)) else np.nan,
        "complex_scale_imag": float(np.imag(C)) if np.isfinite(np.imag(C)) else np.nan,
        "complex_scale_abs": float(abs(C)) if np.isfinite(abs(C)) else np.nan,
        "complex_fit_residual_fraction": float(best["score"]),
        "complex_fit_w_at_lower_bound": bool(np.isfinite(best["w"]) and abs(best["w"] - np.min(w_grid)) < 1e-12),
        "complex_fit_w_at_upper_bound": bool(np.isfinite(best["w"]) and abs(best["w"] - np.max(w_grid)) < 1e-12),
    }


def coherent_reference(coefficients: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average complex coefficients before taking their magnitudes."""
    selected = np.asarray(coefficients, dtype=complex)[indices]
    weights = np.asarray(weights, dtype=float)
    if selected.ndim != 2 or len(selected) == 0 or weights.shape != (len(selected),):
        raise ValueError("Invalid reference coefficient array or weights")
    if not np.isfinite(selected).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Reference coefficients and weights must be finite")
    return np.average(selected, axis=0, weights=weights)


def fit_sqrt3_anisotropy(observed: np.ndarray, reference: np.ndarray, hs: np.ndarray, ks: np.ndarray) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    """Fit centered directional log ratios to -3/2 * (kappa_c cos(2 phi) + kappa_s sin(2 phi))."""
    observed, reference = np.asarray(observed, dtype=complex), np.asarray(reference, dtype=complex)
    hs, ks = np.asarray(hs), np.asarray(ks)
    if observed.ndim != 1 or not (observed.shape == reference.shape == hs.shape == ks.shape):
        raise ValueError("Mode arrays must have matching one-dimensional shapes")
    lookup = {}
    for j, (h, k) in enumerate(zip(hs, ks)):
        if h != int(h) or k != int(k):
            raise ValueError("Fourier indices must be integers")
        if (int(h), int(k)) in lookup:
            raise ValueError("Duplicate Fourier mode")
        lookup[(int(h), int(k))] = j
    expected = set(generate_shell_indices()["sqrt3G1"])
    if not expected.issubset(lookup):
        raise ValueError("All six sqrt(3) modes are required")
    directions = sorted((h, k) for h, k in expected if h > 0 or (h == 0 and k > 0))
    rows = []
    for h, k in directions:
        pair = [lookup[(h, k)], lookup[(-h, -k)]]
        obs_amp = float(np.mean(np.abs(observed[pair])))
        ref_amp = float(np.mean(np.abs(reference[pair])))
        gx, gy = reciprocal_cart_120(h, k)
        delta = float(positive_log(np.array(obs_amp)) - positive_log(np.array(ref_amp)))
        rows.append({"h": h, "k": k, "angle_deg": math.degrees(math.atan2(gy, gx)),
                     "obs_pair_amp": obs_amp, "reference_pair_amp": ref_amp, "delta": delta})
    delta = np.array([row["delta"] for row in rows])
    result = {
        "eta_fit_ok": False, "eta_fit_message": "nonpositive_or_nonfinite_direction_amplitude",
        "eta_rel_logamp": np.nan, "eta_rel_logamp_orientation_deg": np.nan,
        "kappa_c": np.nan, "kappa_s": np.nan,
        "logamp_model_a_traceless": np.nan, "logamp_model_b_traceless": np.nan,
        "eta_fit_shell_q2": 3, "eta_fit_n_directions": 3, "eta_fit_residual_dof": 0,
    }
    if not np.isfinite(delta).all():
        return result, rows
    angles = np.radians([row["angle_deg"] for row in rows])
    design = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)])
    centered = delta - delta.mean()
    kappa, _, rank, _ = np.linalg.lstsq(-1.5 * design, centered, rcond=None)
    if rank != 2:
        raise ValueError("The sqrt(3) directions do not span the anisotropy model")
    fitted = -1.5 * design @ kappa
    eta = 0.5 * float(np.linalg.norm(kappa))
    orientation = (0.5 * math.degrees(math.atan2(kappa[1], kappa[0]))) % 180.0 if eta > 0 else 0.0
    for row, y, pred in zip(rows, centered, fitted):
        row.update({"delta_centered": float(y), "delta_fitted": float(pred), "residual": float(y - pred)})
    result.update({
        "eta_fit_ok": True, "eta_fit_message": "ok",
        "eta_rel_logamp": eta, "eta_rel_logamp_orientation_deg": orientation,
        "kappa_c": float(kappa[0]), "kappa_s": float(kappa[1]),
        "logamp_model_a_traceless": float(kappa[0] / 2),
        "logamp_model_b_traceless": float(kappa[1] / 2),
        "eta_delta_mean": float(delta.mean()),
    })
    return result, rows


def readout_from_coherent_reference(observed: np.ndarray, reference: np.ndarray, hs: np.ndarray, ks: np.ndarray) -> Tuple[Dict[str, object], List[Dict[str, float]]]:
    """Compute manuscript w and eta from one observed and one reference spectrum."""
    observed, reference = np.asarray(observed, dtype=complex), np.asarray(reference, dtype=complex)
    eta, directions = fit_sqrt3_anisotropy(observed, reference, hs, ks)
    q2 = np.array([q2_120(int(h), int(k)) for h, k in zip(hs, ks)])
    if np.count_nonzero(q2 == 1) != 6:
        raise ValueError("All six first-order modes are required")
    obs1, obs3 = [float(np.mean(np.abs(observed[q2 == shell]))) for shell in (1, 3)]
    ref1, ref3 = [float(np.mean(np.abs(reference[q2 == shell]))) for shell in (1, 3)]
    obs_log = float(positive_log(np.array(obs3)) - positive_log(np.array(obs1)))
    ref_log = float(positive_log(np.array(ref3)) - positive_log(np.array(ref1)))
    w = -0.5 * (obs_log - ref_log)
    return {
        "readout_estimator": READOUT_ESTIMATOR,
        "w_fit_ok": bool(np.isfinite(w)), "w_rel_primary": w,
        "obs_log_sqrt3_over_G1": obs_log, "local_ref_log_sqrt3_over_G1": ref_log,
        "G1_obs_amp_mean": obs1, "sqrt3G1_obs_amp_mean": obs3,
        "G1_local_ref_amp_mean": ref1, "sqrt3G1_local_ref_amp_mean": ref3,
        **eta,
    }, directions


# =============================================================================
# Main computation
# =============================================================================


def load_inputs(analysis_dir: Path, map_key: str) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    npz_path = analysis_dir / "motif_analysis_arrays.npz"
    coord_path = analysis_dir / "orthogonal_motif_coordinates_site_currents.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not coord_path.exists():
        raise FileNotFoundError(coord_path)
    npz = np.load(npz_path, allow_pickle=True)
    if map_key not in npz.files:
        raise KeyError(f"map_key {map_key!r} not in {npz_path}. Available keys: {npz.files}")
    maps = np.asarray(npz[map_key], dtype=float)
    names = [_decode_name(x) for x in npz["names"]] if "names" in npz.files else [f"image_{i:04d}" for i in range(maps.shape[0])]
    coord = pd.read_csv(coord_path)
    if "name" not in coord.columns:
        raise KeyError("coord CSV must contain 'name'")
    coord = coord.copy()
    coord["name"] = coord["name"].astype(str)
    # Reorder coord to npz names.
    idx = pd.DataFrame({"name": names, "array_index": np.arange(len(names), dtype=int)})
    merged = idx.merge(coord, on="name", how="left", suffixes=("", "_coord"))
    if merged["theta_beta_over_AH_deg_norm"].isna().all():
        raise RuntimeError("Failed to merge coord rows to NPZ names; theta column is all NaN")
    return maps, names, merged


def compute_template_relative(
    analysis_dir: Path,
    map_key: str = DEFAULT_MAP_KEY,
    out_folder: str = DEFAULT_OUT_FOLDER,
    template_n: int = 51,
    min_template_n: int = 15,
    theta_window_deg: Optional[float] = None,
    theta_sigma_deg: Optional[float] = None,
    w_grid_min: float = -3.0,
    w_grid_max: float = 3.0,
    w_grid_step: float = 0.01,
    phase_search_half_width: float = 0.04,
    phase_search_steps: int = 7,
    skip_complex_fit: bool = False,
    no_plots: bool = False,
    skip_excel: bool = False,
) -> Dict[str, Path]:
    analysis_dir = Path(analysis_dir)
    out_dir = analysis_dir / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview_png"
    preview_dir.mkdir(exist_ok=True)

    maps, names, coord = load_inputs(analysis_dir, map_key)
    theta_col = find_col(coord, ["theta_beta_over_AH_deg_norm", "theta_deg"], "theta")
    cB_col = find_col(coord, ["abs_c_beta_norm", "c_B", "cB"], "c_B", required=False)
    cAH_col = find_col(coord, ["c_AH_norm", "cAH"], "c_AH", required=False)
    floor_col = find_col(coord, ["raw_current_floor_log10_abs_p05", "log10_I_floor"], "I_floor", required=False)

    theta = pd.to_numeric(coord[theta_col], errors="coerce").to_numpy(dtype=float)
    shells = generate_shell_indices()
    shell_df = shell_indices_table(shells)
    shell_df.to_csv(out_dir / "shell_indices_used_120deg.csv", index=False, encoding="utf-8-sig")

    mode_df, coeff_by_mode = compute_all_coeffs(maps, names, shells)
    mode_df.to_csv(out_dir / "exact_fft_coefficients_all_images_long.csv", index=False, encoding="utf-8-sig")

    # Per-image shell summaries.
    shell_summ_rows = []
    for (idx, shell), g in mode_df.groupby(["array_index", "shell"]):
        amps = g["coeff_abs"].to_numpy(dtype=float)
        shell_summ_rows.append({
            "array_index": int(idx),
            "name": names[int(idx)],
            "shell": shell,
            "amp_geomean": robust_log_geomean(amps),
            "log_amp_median": float(np.nanmedian(np.log(np.maximum(amps, SAFE_EPS)))),
            "amp_median": float(np.nanmedian(amps)),
            "amp_mean": float(np.nanmean(amps)),
            "amp_cv": float(np.nanstd(amps) / np.nanmean(amps)) if np.nanmean(amps) > 0 else np.nan,
        })
    shell_summary = pd.DataFrame(shell_summ_rows)
    shell_summary.to_csv(out_dir / "shell_summary_per_image_long.csv", index=False, encoding="utf-8-sig")
    log_ratio_series = shell_log_ratio_from_mode_df(mode_df)
    obs_log_ratio = log_ratio_series.reindex(np.arange(len(names))).to_numpy(dtype=float)

    # Precompute per-mode log amps matrix in mode order G1+sqrt3+2G1.
    mode_order: List[Tuple[str, int, int, int]] = []
    for sh in ["G1", "sqrt3G1", "2G1"]:
        for h, k in shells[sh]:
            mode_order.append((sh, h, k, q2_120(h, k)))
    n = len(names)
    M = len(mode_order)
    coeff_mat = np.empty((n, M), dtype=complex)
    logamp_mat = np.empty((n, M), dtype=float)
    hs = np.empty(M, dtype=int)
    ks = np.empty(M, dtype=int)
    q2s = np.empty(M, dtype=int)
    shell_labels = np.empty(M, dtype=object)
    for j, (sh, h, k, q2) in enumerate(mode_order):
        arr = coeff_by_mode[(h, k)]
        coeff_mat[:, j] = arr
        logamp_mat[:, j] = positive_log(np.abs(arr))
        hs[j] = h
        ks[j] = k
        q2s[j] = q2
        shell_labels[j] = sh

    w_grid = np.arange(w_grid_min, w_grid_max + 0.5 * w_grid_step, w_grid_step, dtype=float)
    if phase_search_steps <= 1 or phase_search_half_width <= 0:
        du_grid = np.array([0.0])
        dv_grid = np.array([0.0])
    else:
        du_grid = np.linspace(-phase_search_half_width, phase_search_half_width, phase_search_steps)
        dv_grid = np.linspace(-phase_search_half_width, phase_search_half_width, phase_search_steps)

    result_rows = []
    mode_rows = []
    neigh_rows = []
    direction_rows = []

    for i in range(n):
        neigh, wt = neighbor_indices(theta, i, template_n, theta_window_deg, theta_sigma_deg, min_template_n)
        wt_sum = float(np.sum(wt))
        wt_norm = wt / wt_sum if wt_sum > 0 else np.ones_like(wt) / len(wt)

        template_coeff = coherent_reference(coeff_mat, neigh, wt_norm)
        readout, directions = readout_from_coherent_reference(coeff_mat[i], template_coeff, hs, ks)
        w_primary = readout["w_rel_primary"]
        ref_log_ratio = readout["local_ref_log_sqrt3_over_G1"]
        ref_logamp_mode = positive_log(np.abs(template_coeff))
        rel_log_att = -(positive_log(np.abs(coeff_mat[i])) - ref_logamp_mode)
        template_map = np.average(maps[neigh], axis=0, weights=wt_norm)
        t_corr = corrcoef2(maps[i], template_map)
        for direction in directions:
            direction_rows.append({"name": names[i], "array_index": i, **direction})

        if skip_complex_fit:
            cfit = {"fit_ok": False, "message": "skipped"}
        else:
            cfit = complex_fit_phase_aligned(coeff_mat[i, :12], template_coeff[:12], hs[:12], ks[:12], q2s[:12], shell_labels[:12], w_grid, du_grid, dv_grid)

        row = {
            "name": names[i],
            "array_index": i,
            "theta_deg": theta[i],
            "template_n": int(len(neigh)),
            "template_theta_mean_deg": weighted_mean(theta[neigh], wt_norm),
            "template_theta_median_deg": weighted_median(theta[neigh], wt_norm),
            "template_theta_min_deg": float(np.nanmin(theta[neigh])) if len(neigh) else np.nan,
            "template_theta_max_deg": float(np.nanmax(theta[neigh])) if len(neigh) else np.nan,
            "template_theta_abs_delta_median_deg": weighted_median(np.abs(theta[neigh] - theta[i]), wt_norm),
            "template_theta_abs_delta_max_deg": float(np.nanmax(np.abs(theta[neigh] - theta[i]))) if len(neigh) else np.nan,
            "template_map_corr_to_image": t_corr,
            "obs_log_sqrt3_over_G1": obs_log_ratio[i],
            "local_ref_log_sqrt3_over_G1": ref_log_ratio,
            "w_rel_primary": w_primary,
            "template_relative_readout_blur_w_rel": w_primary,
            "w_rel_signed_interpretation": ("undefined" if not np.isfinite(w_primary) else
                "more_attenuated_than_local_theta_reference" if w_primary > 0 else
                "sharper_than_local_theta_reference" if w_primary < 0 else "same_as_local_theta_reference"),
            "sqrt3_over_G1_obs_amp_ratio": float(np.exp(obs_log_ratio[i])) if np.isfinite(obs_log_ratio[i]) else np.nan,
            "sqrt3_over_G1_local_ref_amp_ratio": float(np.exp(ref_log_ratio)) if np.isfinite(ref_log_ratio) else np.nan,
            "G1_obs_amp_geomean": robust_log_geomean(np.abs(coeff_mat[i, shell_labels == "G1"])),
            "sqrt3G1_obs_amp_geomean": robust_log_geomean(np.abs(coeff_mat[i, shell_labels == "sqrt3G1"])),
            "G1_local_ref_amp_geomean": robust_log_geomean(np.abs(template_coeff[shell_labels == "G1"])),
            "sqrt3G1_local_ref_amp_geomean": robust_log_geomean(np.abs(template_coeff[shell_labels == "sqrt3G1"])),
            **{f"complex_{k}": v for k, v in cfit.items()},
            **readout,
        }
        # Merge useful coord columns.
        for c in [theta_col, cB_col, cAH_col, floor_col, "R_orthogonal_norm", "I_B_norm", "I_A_norm", "I_H_norm"]:
            if c and c in coord.columns:
                row[c] = coord.loc[i, c]
        result_rows.append(row)

        for rank, (j, wj) in enumerate(zip(neigh, wt_norm), start=1):
            neigh_rows.append({
                "name": names[i],
                "array_index": i,
                "neighbor_rank": rank,
                "neighbor_array_index": int(j),
                "neighbor_name": names[int(j)],
                "theta_deg": theta[i],
                "neighbor_theta_deg": theta[int(j)],
                "theta_abs_delta_deg": abs(theta[int(j)] - theta[i]),
                "template_weight": float(wj),
            })
        for j, (sh, h, k, q2) in enumerate(mode_order):
            mode_rows.append({
                "name": names[i],
                "array_index": i,
                "shell": sh,
                "h": h,
                "k": k,
                "q2_120": q2,
                "obs_coeff_real": coeff_mat[i, j].real,
                "obs_coeff_imag": coeff_mat[i, j].imag,
                "obs_coeff_abs": abs(coeff_mat[i, j]),
                "obs_log_amp": logamp_mat[i, j],
                "local_ref_log_amp": ref_logamp_mode[j],
                "local_ref_amp": float(np.exp(ref_logamp_mode[j])) if np.isfinite(ref_logamp_mode[j]) else np.nan,
                "relative_log_attenuation_positive_weaker": rel_log_att[j],
                "local_ref_coeff_real": template_coeff[j].real,
                "local_ref_coeff_imag": template_coeff[j].imag,
                "used_for_eta": bool(q2 == 3),
            })

    res = pd.DataFrame(result_rows)
    mode_out = pd.DataFrame(mode_rows)
    neigh_out = pd.DataFrame(neigh_rows)
    direction_out = pd.DataFrame(direction_rows)

    # Derived diagnostics.
    res["complex_fit_boundary_flag"] = res.get("complex_complex_fit_w_at_lower_bound", False).astype(bool) | res.get("complex_complex_fit_w_at_upper_bound", False).astype(bool) if "complex_complex_fit_w_at_lower_bound" in res else False
    if "complex_w_rel_complex_phase_aligned" in res:
        res["delta_complex_minus_primary_w"] = res["complex_w_rel_complex_phase_aligned"] - res["w_rel_primary"]

    # Correlations.
    corr_pairs = []
    y_candidates = []
    if cB_col: y_candidates.append(("c_B", cB_col))
    if floor_col: y_candidates.append(("log10_I_floor", floor_col))
    if cAH_col: y_candidates.append(("c_AH", cAH_col))
    y_candidates.append(("theta", "theta_deg"))
    y_candidates += [
        ("template_map_corr", "template_map_corr_to_image"),
        ("complex_residual", "complex_complex_fit_residual_fraction"),
    ]
    x_candidates = [
        ("w_rel_primary", "w_rel_primary"),
        ("eta_rel_logamp", "eta_rel_logamp"),
        ("w_rel_complex_phase_aligned", "complex_w_rel_complex_phase_aligned"),
    ]
    for xlab, xcol in x_candidates:
        if xcol not in res.columns:
            continue
        for ylab, ycol in y_candidates:
            if ycol not in res.columns:
                continue
            rho_s, p_s, rho_p, p_p, nn = stat_corr(res[xcol], res[ycol])
            corr_pairs.append({
                "x_metric": xlab,
                "x_column": xcol,
                "y_metric": ylab,
                "y_column": ycol,
                "n": nn,
                "spearman_rho": rho_s,
                "spearman_p": p_s,
                "pearson_r": rho_p,
                "pearson_p": p_p,
                "abs_spearman_rho": abs(rho_s) if np.isfinite(rho_s) else np.nan,
            })
    corr_df = pd.DataFrame(corr_pairs).sort_values("abs_spearman_rho", ascending=False)

    # Summary.
    summary_rows = []
    for col in [
        "w_rel_primary", "eta_rel_logamp", "complex_w_rel_complex_phase_aligned",
        "template_map_corr_to_image", "complex_complex_fit_residual_fraction",
        "obs_log_sqrt3_over_G1", "local_ref_log_sqrt3_over_G1",
    ]:
        if col not in res.columns:
            continue
        v = pd.to_numeric(res[col], errors="coerce").to_numpy(dtype=float)
        vv = v[np.isfinite(v)]
        if vv.size:
            summary_rows.append({
                "metric": col,
                "n": int(vv.size),
                "mean": float(np.mean(vv)),
                "median": float(np.median(vv)),
                "std": float(np.std(vv, ddof=1)) if vv.size > 1 else np.nan,
                "p05": float(np.percentile(vv, 5)),
                "p25": float(np.percentile(vv, 25)),
                "p75": float(np.percentile(vv, 75)),
                "p95": float(np.percentile(vv, 95)),
                "min": float(np.min(vv)),
                "max": float(np.max(vv)),
            })
    # Boundary counts.
    if "complex_complex_fit_w_at_lower_bound" in res.columns:
        summary_rows.append({"metric": "complex_fit_lower_bound_count", "n": len(res), "mean": float(res["complex_complex_fit_w_at_lower_bound"].sum())})
        summary_rows.append({"metric": "complex_fit_upper_bound_count", "n": len(res), "mean": float(res["complex_complex_fit_w_at_upper_bound"].sum())})
    summary = pd.DataFrame(summary_rows)

    # Save.
    res.to_csv(out_dir / "readout_parameters.csv", index=False, encoding="utf-8-sig")
    direction_out.to_csv(out_dir / "sqrt3_directional_fit.csv", index=False, encoding="utf-8-sig")
    mode_out.to_csv(out_dir / "readout_modes.csv", index=False, encoding="utf-8-sig")
    neigh_out.to_csv(out_dir / "template_neighbor_table_long.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(
        out_dir / "readout_all_image_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(out_dir / "template_relative_diagnostic_summary.csv", index=False, encoding="utf-8-sig")

    # Excel workbook.
    if not skip_excel:
        try:
            with pd.ExcelWriter(out_dir / "readout_attenuation_results.xlsx", engine="openpyxl") as writer:
                res.to_excel(writer, sheet_name="per_image", index=False)
                direction_out.to_excel(writer, sheet_name="sqrt3_directions", index=False)
                corr_df.to_excel(writer, sheet_name="correlations", index=False)
                summary.to_excel(writer, sheet_name="summary", index=False)
                shell_df.to_excel(writer, sheet_name="shell_indices", index=False)
                # Avoid too-large workbook sheets.
                mode_out.head(100000).to_excel(writer, sheet_name="mode_table_head", index=False)
                neigh_out.head(100000).to_excel(writer, sheet_name="neighbors_head", index=False)
        except Exception as exc:
            warnings.warn(f"Excel workbook skipped/failed: {exc}")

    # Plots.
    if not no_plots and plt is not None:
        try:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.hist(res["w_rel_primary"].dropna(), bins=40, alpha=0.75, edgecolor="black")
            ax.axvline(res["w_rel_primary"].median(), linestyle="--", linewidth=1.2)
            ax.set_xlabel("template-relative attenuation, w_rel primary")
            ax.set_ylabel("count")
            ax.set_title("Primary local-theta shell-ratio attenuation")
            fig.tight_layout()
            fig.savefig(preview_dir / "preview_w_rel_primary_histogram.png", dpi=200)
            plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Plot skipped: {exc}")
        try:
            if cB_col:
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                ax.scatter(res["w_rel_primary"], res[cB_col], s=20, alpha=0.65, edgecolors="none")
                rho, p, _, _, nn = stat_corr(res["w_rel_primary"], res[cB_col])
                ax.text(0.03, 0.97, f"Spearman ρ={rho:.3g}\nN={nn}", transform=ax.transAxes, va="top")
                ax.set_xlabel("w_rel primary")
                ax.set_ylabel("c_B = |c_beta|")
                ax.set_title("Relative attenuation vs c_B")
                fig.tight_layout()
                fig.savefig(preview_dir / "preview_w_rel_primary_vs_cB.png", dpi=200)
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Plot skipped: {exc}")
        try:
            if floor_col:
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                ax.scatter(res["w_rel_primary"], res[floor_col], s=20, alpha=0.65, edgecolors="none")
                rho, p, _, _, nn = stat_corr(res["w_rel_primary"], res[floor_col])
                ax.text(0.03, 0.97, f"Spearman ρ={rho:.3g}\nN={nn}", transform=ax.transAxes, va="top")
                ax.set_xlabel("w_rel primary")
                ax.set_ylabel("log10 |I_floor|")
                ax.set_title("Relative attenuation vs current floor")
                fig.tight_layout()
                fig.savefig(preview_dir / "preview_w_rel_primary_vs_floor.png", dpi=200)
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Plot skipped: {exc}")
        try:
            if "complex_w_rel_complex_phase_aligned" in res.columns:
                fig, ax = plt.subplots(figsize=(5.2, 4.2))
                ax.scatter(res["w_rel_primary"], res["complex_w_rel_complex_phase_aligned"], s=20, alpha=0.55, edgecolors="none")
                ax.axline((0, 0), slope=1, linestyle="--", linewidth=1)
                ax.set_xlabel("w_rel primary, shell-ratio")
                ax.set_ylabel("w_rel complex phase-aligned diagnostic")
                ax.set_title("Primary vs diagnostic complex fit")
                fig.tight_layout()
                fig.savefig(preview_dir / "preview_primary_vs_complex_w.png", dpi=200)
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Plot skipped: {exc}")
    readme = f"""# Template-relative higher-order attenuation analysis

This folder was generated by `03_compute_readout_attenuation.py`.

## Primary metric

`w_rel_primary` is the primary descriptor. It is computed from exact folded-cell Fourier coefficients as:

`w_rel_primary = -0.5 * [log(A_sqrt3/A_G1)_image - log(A_sqrt3/A_G1)_reference]`.

The reference is the coherent arithmetic mean of the other nearest-theta spectra
(default: up to 51 neighbors). Each shell amplitude is the arithmetic mean of
its six coefficient magnitudes. Optional theta-sigma weights apply to the complex
mean, never to individual log ratios.

`eta_rel_logamp` uses only the three independent sqrt(3) directions. Their centered
log-amplitude residuals are fitted to `-3/2 * (kappa_c cos(2 phi) + kappa_s sin(2 phi))`;
`eta = 0.5 * sqrt(kappa_c**2 + kappa_s**2)`. The 2G1 shell is diagnostic only.
Three directions determine the directional mean and two anisotropy components
exactly, leaving no residual degrees of freedom for a goodness-of-fit test.

Positive values mean the image has weaker √3G1 content relative to G1 than the local θ-conditioned reference. Negative values mean it is sharper than the local reference. This is a relative descriptor, not an absolute readout width.

## Estimator

The shell-ratio attenuation is the primary metric. A phase-aligned complex fit is retained as a diagnostic.

## Key files

- `readout_parameters.csv`: per-image metrics.
- `readout_modes.csv`: mode-level exact FFT coefficients and local references.
- `sqrt3_directional_fit.csv`: the three directional log ratios and anisotropy fit.
- `readout_all_image_diagnostics.csv`: all-image correlations used only as estimator diagnostics. The selected tests are produced after the SNR selection by `05_compute_snr_resolved_statistics.py`.
- `template_relative_diagnostic_summary.csv`: distribution summaries and boundary counts.
- `preview_png/`: quick-look plots.

## Settings

```json
{json.dumps({
    'analysis_dir': analysis_dir.name,
    'map_key': map_key,
    'template_n': template_n,
    'min_template_n': min_template_n,
    'theta_window_deg': theta_window_deg,
    'theta_sigma_deg': theta_sigma_deg,
    'reference_method': 'arithmetic_complex_mean' if theta_sigma_deg is None else 'gaussian_weighted_complex_mean',
    'readout_estimator': READOUT_ESTIMATOR,
    'shell_amplitude_statistic': 'arithmetic_mean_abs',
    'eta_shell_q2': 3,
    'eta_n_directions': 3,
    'w_grid_min': w_grid_min,
    'w_grid_max': w_grid_max,
    'w_grid_step': w_grid_step,
    'phase_search_half_width': phase_search_half_width,
    'phase_search_steps': phase_search_steps,
    'skip_complex_fit': skip_complex_fit,
}, indent=2, ensure_ascii=False)}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "out_dir": out_dir,
        "per_image": out_dir / "readout_parameters.csv",
        "correlations": out_dir / "readout_all_image_diagnostics.csv",
        "summary": out_dir / "template_relative_diagnostic_summary.csv",
    }

# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Template-relative higher-order attenuation analysis, 120° convention.", allow_abbrev=False)
    parser.add_argument("--analysis-dir", "--analysis_dir", dest="analysis_dir", required=True)
    parser.add_argument("--map-key", "--map_key", dest="map_key", default=DEFAULT_MAP_KEY)
    parser.add_argument("--out-folder", "--out_folder", dest="out_folder", default=DEFAULT_OUT_FOLDER)
    parser.add_argument("--template-n", "--template_n", dest="template_n", type=int, default=51)
    parser.add_argument("--min-template-n", "--min_template_n", dest="min_template_n", type=int, default=15)
    parser.add_argument("--theta-window-deg", "--theta_window_deg", dest="theta_window_deg", type=float, default=None)
    parser.add_argument("--theta-sigma-deg", "--theta_sigma_deg", dest="theta_sigma_deg", type=float, default=None)
    parser.add_argument("--w-grid-min", "--w_grid_min", dest="w_grid_min", type=float, default=-3.0)
    parser.add_argument("--w-grid-max", "--w_grid_max", dest="w_grid_max", type=float, default=3.0)
    parser.add_argument("--w-grid-step", "--w_grid_step", dest="w_grid_step", type=float, default=0.01)
    parser.add_argument("--phase-search-half-width", "--phase_search_half_width", dest="phase_search_half_width", type=float, default=0.04)
    parser.add_argument("--phase-search-steps", "--phase_search_steps", dest="phase_search_steps", type=int, default=7)
    parser.add_argument("--skip-complex-fit", "--skip_complex_fit", dest="skip_complex_fit", action="store_true")
    parser.add_argument("--no-plots", "--no_plots", dest="no_plots", action="store_true")
    parser.add_argument("--skip-excel", "--skip_excel", dest="skip_excel", action="store_true")
    args = parser.parse_args()

    paths = compute_template_relative(
        analysis_dir=Path(args.analysis_dir),
        map_key=args.map_key,
        out_folder=args.out_folder,
        template_n=args.template_n,
        min_template_n=args.min_template_n,
        theta_window_deg=args.theta_window_deg,
        theta_sigma_deg=args.theta_sigma_deg,
        w_grid_min=args.w_grid_min,
        w_grid_max=args.w_grid_max,
        w_grid_step=args.w_grid_step,
        phase_search_half_width=args.phase_search_half_width,
        phase_search_steps=args.phase_search_steps,
        skip_complex_fit=args.skip_complex_fit,
        no_plots=args.no_plots,
        skip_excel=args.skip_excel,
    )
    print("=== DONE: template-relative attenuation ===")
    for k, p in paths.items():
        print(f"{k}: {p}")

    try:
        res = pd.read_csv(paths["per_image"])
        print("\nPrimary w_rel summary:")
        print(res["w_rel_primary"].describe().to_string())
        corr = pd.read_csv(paths["correlations"])
        print("\nTop correlations:")
        print(corr.head(10).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
