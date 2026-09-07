# -*- coding: utf-8 -*-
"""Export the numerical source data and preview panels for Figures 4 and 5.

Inputs are the primary-analysis table, aligned unit-cell arrays, the
template-relative readout descriptors from 03_compute_readout_attenuation.py,
and the resolved-image selection from 05_compute_snr_resolved_statistics.py."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = None
    pearsonr = None

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_OUT_FOLDER_NAME = "fig4_fig5_template_readout_source_data_120deg"
DEFAULT_MAP_KEY = "aligned_norm"
DEFAULT_ANGLE_DEG = 120.0
DEFAULT_FFT_CROP_Q = 5.0
DEFAULT_DRIVER_ARROW_SCALE = "auto"  # "auto" or a number as string
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_RANDOM_SEED = 2027
DEFAULT_SNR_THRESHOLD = 5.0

# Shell names used throughout the output.
SHELL_TARGET_Q2 = {
    "G1": 1.0,
    "sqrt3G1": 3.0,
    "2G1": 4.0,
}

# =============================================================================
# Basic utilities
# =============================================================================


def safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clean_sheet_name(name: str, max_len: int = 31) -> str:
    s = "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in str(name))
    s = s.strip() or "sheet"
    return s[:max_len]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def find_template_readout_csv(analysis_dir: Path) -> Path:
    candidates = [
        analysis_dir / "readout_attenuation_120deg" / "readout_parameters.csv",
        analysis_dir / "readout_parameters.csv",
    ]
    p = find_first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            "Could not find a readout-parameter table. Use --readout-csv to specify it."
        )
    return p


def choose_col(df: pd.DataFrame, candidates: Sequence[str], label: str, required: bool = True) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"Could not find {label}. Tried {candidates}. Available columns: {list(df.columns)}")
    return None


def parse_boolean_column(series: pd.Series, label: str) -> pd.Series:
    """Parse a CSV boolean column without treating non-empty strings as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    parsed = normalized.map({"true": True, "1": True, "false": False, "0": False})
    if parsed.isna().any():
        bad = ", ".join(sorted(normalized.loc[parsed.isna()].unique())[:3])
        raise ValueError(f"Could not parse {label} as Boolean; examples: {bad}")
    return parsed.astype(bool)


def finite_mask(*arrs) -> np.ndarray:
    if not arrs:
        return np.array([], dtype=bool)
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        aa = np.asarray(a, dtype=float)
        m &= np.isfinite(aa)
    return m


def robust_geomean_positive(vals: np.ndarray, eps: float = 1e-300) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(np.maximum(v, eps)))))


def robust_log_median(vals: np.ndarray, eps: float = 1e-300) -> float:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    return float(np.median(np.log(np.maximum(v, eps))))


def zero_mean(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr - np.nanmean(arr)

# =============================================================================
# 120° reciprocal shell geometry
# =============================================================================


def reciprocal_cart_from_hk(h, k, angle_deg: float = DEFAULT_ANGLE_DEG) -> Tuple[np.ndarray, np.ndarray]:
    """Cartesian reciprocal coordinates for direct cell a1=(1,0), a2=(cos alpha, sin alpha).

    A^{-T} [h,k] gives coordinates up to a common 2pi factor:
      Gx = h
      Gy = (k - h cos(alpha)) / sin(alpha)
    """
    alpha = math.radians(float(angle_deg))
    h_arr = np.asarray(h, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    if abs(sin_a) < 1e-12:
        raise ValueError("angle_deg too close to 0 or 180 degrees")
    gx = h_arr
    gy = (k_arr - h_arr * cos_a) / sin_a
    return gx, gy


def q2_metric(h, k, angle_deg: float = DEFAULT_ANGLE_DEG) -> float:
    gx, gy = reciprocal_cart_from_hk(h, k, angle_deg)
    # Normalize so that the G1 shell has q^2=1.
    # For an oblique cell, raw |G|^2 for G1 is 1/sin^2(alpha). Multiplying by sin^2 gives q^2.
    alpha = math.radians(float(angle_deg))
    return float((gx * gx + gy * gy) * (math.sin(alpha) ** 2))


def enumerate_shell_indices(angle_deg: float = DEFAULT_ANGLE_DEG, max_index: int = 6, tol: float = 1e-8) -> pd.DataFrame:
    rows = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            if h == 0 and k == 0:
                continue
            q2 = q2_metric(h, k, angle_deg)
            gx, gy = reciprocal_cart_from_hk(h, k, angle_deg)
            for shell, target in SHELL_TARGET_Q2.items():
                if abs(q2 - target) < tol:
                    rows.append({
                        "shell": shell,
                        "h": int(h),
                        "k": int(k),
                        "q2": float(q2),
                        "Gx_cart": float(gx),
                        "Gy_cart": float(gy),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No shell indices found; check angle_deg/tolerance.")
    # Normalize reciprocal coordinates by G1 magnitude for easier plotting.
    g1 = df.loc[df["shell"] == "G1"].copy()
    g1_norm = float(np.median(np.sqrt(g1["Gx_cart"] ** 2 + g1["Gy_cart"] ** 2)))
    df["Gx_normG1"] = df["Gx_cart"] / g1_norm
    df["Gy_normG1"] = df["Gy_cart"] / g1_norm
    df = df.sort_values(["shell", "h", "k"]).reset_index(drop=True)
    return df


def fft_index_for_mode(n: int, h: int, k: int) -> Tuple[int, int]:
    """Return unshifted FFT array index for mode (h,k), where rows are k/v, cols are h/u."""
    return int(k % n), int(h % n)


def shifted_grid_hk(n: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.fft.fftshift(np.fft.fftfreq(n) * n).astype(int)
    H, K = np.meshgrid(idx, idx, indexing="xy")
    return H, K


def fft2_unitcell(unit_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unshifted complex FFT, shifted complex FFT, and shifted power. Normalized by N pixels."""
    arr = zero_mean(np.asarray(unit_map, dtype=float))
    n_pix = arr.size
    F = np.fft.fft2(arr) / max(1, n_pix)
    F_shift = np.fft.fftshift(F)
    P_shift = np.abs(F_shift) ** 2
    return F, F_shift, P_shift


def extract_shell_from_fft(F_unshifted: np.ndarray, shell_df: pd.DataFrame) -> pd.DataFrame:
    n = F_unshifted.shape[0]
    rows = []
    for _, r in shell_df.iterrows():
        h = int(r["h"]); k = int(r["k"])
        iy, ix = fft_index_for_mode(n, h, k)
        coeff = complex(F_unshifted[iy, ix])
        rows.append({
            **r.to_dict(),
            "coeff_real": coeff.real,
            "coeff_imag": coeff.imag,
            "amplitude": abs(coeff),
            "power": abs(coeff) ** 2,
            "log10_power": math.log10(max(abs(coeff) ** 2, 1e-300)),
        })
    return pd.DataFrame(rows)

# =============================================================================
# Statistics
# =============================================================================


def correlation_summary(x, y, x_name: str, y_name: str, bootstrap_n: int = DEFAULT_BOOTSTRAP_N, seed: int = DEFAULT_RANDOM_SEED) -> Dict[str, object]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    xx = x[m]
    yy = y[m]
    out = {
        "x": x_name,
        "y": y_name,
        "n": int(xx.size),
        "method": "spearman",
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "spearman_ci95_low": np.nan,
        "spearman_ci95_high": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
    }
    if xx.size < 4:
        return out
    if spearmanr is not None:
        res = spearmanr(xx, yy)
        out["spearman_rho"] = float(res.correlation)
        out["spearman_p"] = float(res.pvalue)
    else:
        # Rank-based fallback.
        rx = pd.Series(xx).rank().to_numpy(float)
        ry = pd.Series(yy).rank().to_numpy(float)
        out["spearman_rho"] = float(np.corrcoef(rx, ry)[0, 1])
    if pearsonr is not None:
        pr = pearsonr(xx, yy)
        out["pearson_r"] = float(pr.statistic)
        out["pearson_p"] = float(pr.pvalue)
    else:
        out["pearson_r"] = float(np.corrcoef(xx, yy)[0, 1])
    if bootstrap_n and bootstrap_n > 0 and xx.size >= 5:
        rng = np.random.default_rng(seed)
        vals = []
        idx = np.arange(xx.size)
        for _ in range(int(bootstrap_n)):
            ii = rng.choice(idx, size=idx.size, replace=True)
            if spearmanr is not None:
                rb = spearmanr(xx[ii], yy[ii]).correlation
            else:
                rb = np.corrcoef(pd.Series(xx[ii]).rank(), pd.Series(yy[ii]).rank())[0, 1]
            vals.append(rb)
        vals = np.asarray(vals, dtype=float)
        out["spearman_ci95_low"] = float(np.nanpercentile(vals, 2.5))
        out["spearman_ci95_high"] = float(np.nanpercentile(vals, 97.5))
    return out


def center_predictor_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    return X - mu[None, :], mu


def linear_residualize_to_mean(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y corrected to mean(X), beta, predictor means.

    y_corr = y - (X - mean(X)) @ beta
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    beta = np.full(X.shape[1], np.nan, dtype=float)
    mu = np.nanmean(X, axis=0)
    y_corr = np.full_like(y, np.nan, dtype=float)
    if m.sum() < X.shape[1] + 2:
        return y_corr, beta, mu
    Xc = X[m] - mu[None, :]
    yy = y[m] - np.nanmean(y[m])
    # Fit no intercept because predictors are centered and y is centered.
    beta_fit, *_ = np.linalg.lstsq(Xc, yy, rcond=None)
    beta[:] = beta_fit
    y_corr[m] = y[m] - (X[m] - mu[None, :]) @ beta_fit
    return y_corr, beta, mu


def theta_from_cAH_cB(cAH, cB) -> np.ndarray:
    cAH = np.asarray(cAH, dtype=float)
    cB = np.asarray(cB, dtype=float)
    return np.degrees(np.arctan2(np.maximum(cB, 0), cAH))


def driver_arrow_from_univariate(df: pd.DataFrame, driver_col: str, cAH_col: str, cB_col: str, label: str) -> Dict[str, object]:
    x = pd.to_numeric(df[driver_col], errors="coerce").to_numpy(float)
    y1 = pd.to_numeric(df[cAH_col], errors="coerce").to_numpy(float)
    y2 = pd.to_numeric(df[cB_col], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    if m.sum() < 4 or np.nanstd(x[m]) == 0:
        return {"driver": label, "n": int(m.sum()), "dx_1sd": np.nan, "dy_1sd": np.nan, "magnitude_1sd": np.nan, "angle_deg": np.nan}
    xc = x[m] - np.nanmean(x[m])
    sd = float(np.nanstd(x[m], ddof=1))
    b1 = float(np.dot(xc, y1[m] - np.nanmean(y1[m])) / np.dot(xc, xc))
    b2 = float(np.dot(xc, y2[m] - np.nanmean(y2[m])) / np.dot(xc, xc))
    dx = b1 * sd
    dy = b2 * sd
    return {
        "driver": label,
        "n": int(m.sum()),
        "driver_col": driver_col,
        "driver_sd": sd,
        "slope_cAH_per_driver": b1,
        "slope_cB_per_driver": b2,
        "dx_1sd": dx,
        "dy_1sd": dy,
        "magnitude_1sd": float(math.sqrt(dx * dx + dy * dy)),
        "angle_deg": float(math.degrees(math.atan2(dy, dx))),
    }


def driver_arrow_from_pc1(df: pd.DataFrame, predictor_cols: Sequence[str], cAH_col: str, cB_col: str, label: str) -> Dict[str, object]:
    X = df[list(predictor_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y1 = pd.to_numeric(df[cAH_col], errors="coerce").to_numpy(float)
    y2 = pd.to_numeric(df[cB_col], errors="coerce").to_numpy(float)
    m = np.isfinite(y1) & np.isfinite(y2) & np.all(np.isfinite(X), axis=1)
    if m.sum() < len(predictor_cols) + 3:
        return {"driver": label, "n": int(m.sum()), "dx_1sd": np.nan, "dy_1sd": np.nan}
    X0 = X[m]
    Xs = (X0 - X0.mean(axis=0)) / (X0.std(axis=0, ddof=1) + 1e-12)
    # PC1 sign is arbitrary. Choose sign so that it is positively correlated with w if w is in predictors.
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    if "w" in label.lower() or any("w" == c.lower() for c in predictor_cols):
        # Sign relative to first predictor.
        if np.corrcoef(pc1, Xs[:, 0])[0, 1] < 0:
            pc1 = -pc1
            Vt[0] = -Vt[0]
    sd = float(np.nanstd(pc1, ddof=1))
    xc = pc1 - pc1.mean()
    b1 = float(np.dot(xc, y1[m] - np.nanmean(y1[m])) / np.dot(xc, xc))
    b2 = float(np.dot(xc, y2[m] - np.nanmean(y2[m])) / np.dot(xc, xc))
    dx = b1 * sd
    dy = b2 * sd
    out = {
        "driver": label,
        "n": int(m.sum()),
        "predictor_cols": ";".join(predictor_cols),
        "pc1_explained_fraction": float((S[0] ** 2) / np.sum(S ** 2)) if np.sum(S ** 2) > 0 else np.nan,
        "dx_1sd": dx,
        "dy_1sd": dy,
        "magnitude_1sd": float(math.sqrt(dx * dx + dy * dy)),
        "angle_deg": float(math.degrees(math.atan2(dy, dx))),
    }
    for c, loading in zip(predictor_cols, Vt[0]):
        out[f"pc1_loading_{c}"] = float(loading)
    return out

# =============================================================================
# Data preparation
# =============================================================================


def load_master_table(
    analysis_dir: Path,
    readout_csv: Optional[Path],
    coord_csv: Optional[Path],
    args,
) -> Tuple[pd.DataFrame, Dict[str, str], Path, Optional[Path]]:
    if readout_csv is None:
        readout_csv = find_template_readout_csv(analysis_dir)
    readout_csv = Path(readout_csv)
    df = pd.read_csv(readout_csv)

    # Optional coordinate merge if needed or requested.
    if coord_csv is None:
        coord_csv_p = analysis_dir / "orthogonal_motif_coordinates_site_currents.csv"
    else:
        coord_csv_p = Path(coord_csv)
    if coord_csv_p.exists():
        coord = pd.read_csv(coord_csv_p)
        if "name" in df.columns and "name" in coord.columns:
            needed = [
                "name", "short_name", "c_AH_norm", "abs_c_beta_norm", "c_beta_signed_norm",
                "theta_beta_over_AH_deg_norm", "raw_current_floor_log10_abs_p05",
                "I_B_norm", "I_A_norm", "I_H_norm", "R_orthogonal_norm",
            ]
            merge_cols = [c for c in needed if c in coord.columns and c not in df.columns]
            if merge_cols:
                df = df.merge(coord[["name"] + merge_cols], on="name", how="left")

    # Select columns. Explicit args override defaults.
    w_col = args.w_col or choose_col(df, ["w_rel_primary"], "w/readout descriptor")
    eta_col = args.eta_col or choose_col(df, ["eta_rel_logamp"], "eta/aniso descriptor")
    eta_orientation_col = args.eta_orientation_col or choose_col(df, ["eta_rel_logamp_orientation_deg", "template_relative_anisotropy_orientation_deg", "anisotropic_orientation_deg"], "eta orientation", required=False)
    theta_col = args.theta_col or choose_col(df, ["theta_beta_over_AH_deg_norm", "theta_deg"], "theta")
    cAH_col = args.cAH_col or choose_col(df, ["c_AH_norm", "c_AH"], "c_AH")
    cB_col = args.cB_col or choose_col(df, ["abs_c_beta_norm", "c_B", "abs_c_beta"], "c_B")
    floor_col = args.floor_col or choose_col(df, ["raw_current_floor_log10_abs_p05"], "log10 I_floor")

    # Signed beta: prefer existing, otherwise reconstruct from I_B/I_A, otherwise assume positive cB.
    cbeta_signed_col = args.cbeta_signed_col if args.cbeta_signed_col else None
    if cbeta_signed_col and cbeta_signed_col not in df.columns:
        raise KeyError(f"Requested cbeta_signed_col={cbeta_signed_col!r} not found")
    if cbeta_signed_col is None:
        if "c_beta_signed_norm" in df.columns:
            cbeta_signed_col = "c_beta_signed_norm"
        elif {"I_B_norm", "I_A_norm"}.issubset(df.columns):
            df["c_beta_signed_norm_reconstructed"] = (pd.to_numeric(df["I_B_norm"], errors="coerce") - pd.to_numeric(df["I_A_norm"], errors="coerce")) / math.sqrt(2.0)
            cbeta_signed_col = "c_beta_signed_norm_reconstructed"
        else:
            # Role convention makes c_beta positive in many pipeline outputs.
            df["c_beta_signed_norm_assumed_positive"] = pd.to_numeric(df[cB_col], errors="coerce")
            cbeta_signed_col = "c_beta_signed_norm_assumed_positive"

    # Derived analysis columns with standard names.
    df["w"] = pd.to_numeric(df[w_col], errors="coerce")
    df["eta"] = pd.to_numeric(df[eta_col], errors="coerce") if eta_col else np.nan
    df["theta_deg"] = pd.to_numeric(df[theta_col], errors="coerce")
    df["c_AH"] = pd.to_numeric(df[cAH_col], errors="coerce")
    df["c_B"] = pd.to_numeric(df[cB_col], errors="coerce").abs()
    df["c_beta_signed"] = pd.to_numeric(df[cbeta_signed_col], errors="coerce")
    df["log10_I_floor"] = pd.to_numeric(df[floor_col], errors="coerce")

    # Apply the selected population: sqrt(3)-resolved images with finite
    # readout descriptors.  The selection table is joined by acquisition name
    # and array index; row order is never used as an identifier.
    selection_csv_p: Optional[Path] = Path(
        args.snr_selection_csv
        or analysis_dir / "snr_resolved_readout" / "snr_resolved_readout_per_image.csv"
    )
    if selection_csv_p:
        if not selection_csv_p.exists():
            raise FileNotFoundError(f"SNR selection table not found: {selection_csv_p}")
        selection = pd.read_csv(selection_csv_p)
        required = {
            "name",
            "array_index",
            "sqrt3_amp_snr",
            "sqrt3_resolved",
            "analysis_keep",
            "exclusion_reason",
        }
        missing = sorted(required - set(selection.columns))
        if missing:
            raise KeyError(f"SNR selection table is missing columns: {missing}")
        if not {"name", "array_index"}.issubset(df.columns):
            raise KeyError(
                "Readout table must contain 'name' and 'array_index' when an SNR selection is used"
            )
        df["name"] = df["name"].astype(str)
        selection["name"] = selection["name"].astype(str)
        for table, label in ((df, "readout"), (selection, "selection")):
            numeric_index = pd.to_numeric(table["array_index"], errors="raise")
            index_values = numeric_index.to_numpy(dtype=float)
            if not np.isfinite(index_values).all() or not np.equal(
                index_values, np.floor(index_values)
            ).all():
                raise ValueError(f"{label} array_index must contain finite integers")
            table["array_index"] = numeric_index.astype(int)
        key = ["name", "array_index"]
        if (
            df["name"].duplicated().any()
            or selection["name"].duplicated().any()
            or df["array_index"].duplicated().any()
            or selection["array_index"].duplicated().any()
        ):
            raise ValueError("Readout and SNR selection identifiers must be unique")
        if set(map(tuple, df[key].to_numpy())) != set(map(tuple, selection[key].to_numpy())):
            raise ValueError("Readout and SNR selection tables contain different identifiers")
        payload = {
            "sqrt3_amp_snr",
            "sqrt3_resolved",
            "snr_analysis_keep",
            "snr_exclusion_reason",
        }
        overlap = payload & set(df.columns)
        if overlap:
            raise ValueError(f"SNR selection columns already occur in the readout table: {sorted(overlap)}")
        selection = selection[
            [
                "name",
                "array_index",
                "sqrt3_amp_snr",
                "sqrt3_resolved",
                "analysis_keep",
                "exclusion_reason",
            ]
        ].rename(
            columns={
                "analysis_keep": "snr_analysis_keep",
                "exclusion_reason": "snr_exclusion_reason",
            }
        )
        selection["sqrt3_resolved"] = parse_boolean_column(
            selection["sqrt3_resolved"], "sqrt3_resolved"
        )
        selection["snr_analysis_keep"] = parse_boolean_column(
            selection["snr_analysis_keep"], "analysis_keep"
        )
        selection["sqrt3_amp_snr"] = pd.to_numeric(
            selection["sqrt3_amp_snr"], errors="coerce"
        )
        snr_values = selection["sqrt3_amp_snr"].to_numpy(dtype=float)
        numeric_resolved = np.isfinite(snr_values) & (
            snr_values >= args.snr_threshold
        )
        if not np.array_equal(
            selection["sqrt3_resolved"].to_numpy(dtype=bool), numeric_resolved
        ):
            raise ValueError(
                "sqrt3_resolved flags disagree with the numeric SNR threshold"
            )
        if not np.array_equal(
            selection["snr_analysis_keep"].to_numpy(dtype=bool), numeric_resolved
        ):
            raise ValueError(
                "analysis_keep must contain exactly the finite SNR-resolved population"
            )
        df = df.merge(selection, on=key, how="left", validate="one_to_one")
    else:
        df["sqrt3_amp_snr"] = np.nan
        df["sqrt3_resolved"] = True
        df["snr_analysis_keep"] = True
        df["snr_exclusion_reason"] = ""

    if eta_orientation_col is not None and eta_orientation_col in df.columns:
        phi = np.deg2rad(pd.to_numeric(df[eta_orientation_col], errors="coerce").to_numpy(float))
        eta = df["eta"].to_numpy(float)
        df["eta_cos2phi"] = eta * np.cos(2.0 * phi)
        df["eta_sin2phi"] = eta * np.sin(2.0 * phi)
    else:
        df["eta_cos2phi"] = np.nan
        df["eta_sin2phi"] = np.nan

    # Optional filtering.
    df["fig45_keep"] = True
    reasons = []
    for i, r in df.iterrows():
        rs = []
        if not bool(r["snr_analysis_keep"]):
            reason = str(r["snr_exclusion_reason"]).strip()
            rs.append(reason or "not_snr_resolved")
        if not np.isfinite(r["w"]):
            rs.append("w_not_finite")
        if args.w_min is not None and np.isfinite(r["w"]) and r["w"] < float(args.w_min):
            rs.append(f"w_lt_{args.w_min}")
        if args.w_max is not None and np.isfinite(r["w"]) and r["w"] > float(args.w_max):
            rs.append(f"w_gt_{args.w_max}")
        reasons.append(";".join(rs))
    df["fig45_filter_reason"] = reasons
    df.loc[df["fig45_filter_reason"].astype(str) != "", "fig45_keep"] = False
    if not args.include_nonfinite_core:
        core = ["w", "theta_deg", "c_AH", "c_B", "c_beta_signed", "log10_I_floor"]
        core_finite = np.all(np.isfinite(df[core].to_numpy(float)), axis=1)
        df.loc[~core_finite, "fig45_keep"] = False
        df.loc[~core_finite, "fig45_filter_reason"] = df.loc[~core_finite, "fig45_filter_reason"].astype(str).replace("", "nonfinite_core")

    cols = {
        "w_col_input": w_col,
        "eta_col_input": eta_col or "",
        "eta_orientation_col_input": eta_orientation_col or "",
        "theta_col_input": theta_col,
        "cAH_col_input": cAH_col,
        "cB_col_input": cB_col,
        "cbeta_signed_col_input": cbeta_signed_col,
        "floor_col_input": floor_col,
        "snr_selection_csv": selection_csv_p.name if selection_csv_p else "",
    }
    return df, cols, readout_csv, coord_csv_p if coord_csv_p.exists() else None

# =============================================================================
# Fig 4 computations
# =============================================================================


def compute_fig4_fft_outputs(analysis_dir: Path, out_dir: Path, map_key: str, angle_deg: float, fft_crop_q: float) -> Dict[str, pd.DataFrame]:
    arrays_path = analysis_dir / "motif_analysis_arrays.npz"
    if not arrays_path.exists():
        warnings.warn(f"motif_analysis_arrays.npz not found at {arrays_path}; skipping Fig. 4a,b FFT outputs.")
        return {}
    data = np.load(arrays_path, allow_pickle=True)
    if map_key not in data:
        warnings.warn(f"map_key={map_key!r} not found in {arrays_path}; available={list(data.keys())}; skipping Fig. 4a,b.")
        return {}
    maps = np.asarray(data[map_key], dtype=float)
    if maps.ndim != 3:
        warnings.warn(f"Expected {map_key} to have shape (n, grid, grid); got {maps.shape}; skipping Fig. 4a,b.")
        return {}
    mean_map = np.nanmean(maps, axis=0)
    n = mean_map.shape[0]
    F, F_shift, P_shift = fft2_unitcell(mean_map)
    H, K = shifted_grid_hk(n)
    gx, gy = reciprocal_cart_from_hk(H, K, angle_deg)
    g1_shell = enumerate_shell_indices(angle_deg)
    g1_norm = np.median(np.sqrt(g1_shell.loc[g1_shell["shell"] == "G1", "Gx_cart"] ** 2 + g1_shell.loc[g1_shell["shell"] == "G1", "Gy_cart"] ** 2))
    GxN = gx / g1_norm
    GyN = gy / g1_norm
    q_radius = np.sqrt(GxN ** 2 + GyN ** 2)
    logP = np.log10(np.maximum(P_shift, 1e-300))
    mask_crop = q_radius <= float(fft_crop_q)
    fig4a = pd.DataFrame({
        "h": H[mask_crop].ravel(),
        "k": K[mask_crop].ravel(),
        "Gx_normG1": GxN[mask_crop].ravel(),
        "Gy_normG1": GyN[mask_crop].ravel(),
        "q_normG1": q_radius[mask_crop].ravel(),
        "log10_power": logP[mask_crop].ravel(),
        "power": P_shift[mask_crop].ravel(),
    })
    fig4a.to_csv(out_dir / "fig4a_ensemble_mean_fft_log10_power_xyz.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_map).to_csv(out_dir / "fig4a_ensemble_mean_unitcell_matrix.csv", index=False, header=False, encoding="utf-8-sig")

    shell_df = enumerate_shell_indices(angle_deg)
    shell_df.to_csv(out_dir / "fig4_shell_indices_120deg.csv", index=False, encoding="utf-8-sig")
    markers = shell_df[shell_df["shell"].isin(["G1", "sqrt3G1", "2G1"])].copy()
    markers.to_csv(out_dir / "fig4a_shell_marker_points_for_fft_overlay.csv", index=False, encoding="utf-8-sig")

    coeff_df = extract_shell_from_fft(F, shell_df)
    coeff_df.to_csv(out_dir / "fig4b_ensemble_mean_shell_coefficients_long.csv", index=False, encoding="utf-8-sig")
    shell_rows = []
    for shell, g in coeff_df.groupby("shell"):
        amp = g["amplitude"].to_numpy(float)
        power = g["power"].to_numpy(float)
        shell_rows.append({
            "category": shell,
            "kind": "expected_shell_peak",
            "n_modes": int(len(g)),
            "amp_geomean": robust_geomean_positive(amp),
            "amp_mean": float(np.nanmean(amp)),
            "amp_median": float(np.nanmedian(amp)),
            "power_geomean": robust_geomean_positive(power),
            "power_mean": float(np.nanmean(power)),
            "power_median": float(np.nanmedian(power)),
            "log10_power_mean": float(np.log10(max(np.nanmean(power), 1e-300))),
            "log10_power_median": float(np.log10(max(np.nanmedian(power), 1e-300))),
        })

    # Off-shell background/noise diagnostic from low-q integer modes excluding expected shells.
    expected = {(int(r.h), int(r.k)) for r in shell_df.itertuples()}
    off_rows = []
    max_hk = min(8, n // 2 - 1)
    for h in range(-max_hk, max_hk + 1):
        for k in range(-max_hk, max_hk + 1):
            if h == 0 and k == 0:
                continue
            if (h, k) in expected:
                continue
            q2 = q2_metric(h, k, angle_deg)
            if q2 <= 12.0:
                iy, ix = fft_index_for_mode(n, h, k)
                c = F[iy, ix]
                gx0, gy0 = reciprocal_cart_from_hk(h, k, angle_deg)
                off_rows.append({
                    "h": h, "k": k, "q2": q2, "Gx_cart": float(gx0), "Gy_cart": float(gy0),
                    "amplitude": abs(c), "power": abs(c) ** 2,
                })
    off_df = pd.DataFrame(off_rows)
    if not off_df.empty:
        off_df.to_csv(out_dir / "fig4b_off_shell_background_modes_long.csv", index=False, encoding="utf-8-sig")
        amp = off_df["amplitude"].to_numpy(float)
        power = off_df["power"].to_numpy(float)
        shell_rows.append({
            "category": "off_shell_background_q2_le_12",
            "kind": "off_shell_background",
            "n_modes": int(len(off_df)),
            "amp_geomean": robust_geomean_positive(amp),
            "amp_mean": float(np.nanmean(amp)),
            "amp_median": float(np.nanmedian(amp)),
            "power_geomean": robust_geomean_positive(power),
            "power_mean": float(np.nanmean(power)),
            "power_median": float(np.nanmedian(power)),
            "log10_power_mean": float(np.log10(max(np.nanmean(power), 1e-300))),
            "log10_power_median": float(np.log10(max(np.nanmedian(power), 1e-300))),
        })
    summary = pd.DataFrame(shell_rows)
    # Add ratios relative to G1.
    try:
        g1_power = float(summary.loc[summary["category"] == "G1", "power_mean"].iloc[0])
        g1_amp = float(summary.loc[summary["category"] == "G1", "amp_geomean"].iloc[0])
        summary["power_ratio_to_G1_power_mean"] = summary["power_mean"] / g1_power
        summary["amp_ratio_to_G1_amp_geomean"] = summary["amp_geomean"] / g1_amp
    except Exception:
        summary["power_ratio_to_G1_power_mean"] = np.nan
        summary["amp_ratio_to_G1_amp_geomean"] = np.nan
    summary.to_csv(out_dir / "fig4b_shell_power_summary.csv", index=False, encoding="utf-8-sig")
    return {"fig4a_fft_xyz": fig4a, "fig4_markers": markers, "fig4b_shell_summary": summary, "fig4b_shell_coefficients": coeff_df}

# =============================================================================
# Fig 4c and Fig 5 computations
# =============================================================================


def compute_histogram_data(values: np.ndarray, bins: int = 30) -> pd.DataFrame:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return pd.DataFrame()
    counts, edges = np.histogram(v, bins=bins)
    density, _ = np.histogram(v, bins=edges, density=True)
    return pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "bin_center": 0.5 * (edges[:-1] + edges[1:]),
        "count": counts,
        "density": density,
    })


def compute_fig4c(master: pd.DataFrame, out_dir: Path, args) -> Dict[str, pd.DataFrame]:
    df = master.loc[master["fig45_keep"]].copy()
    rows = []
    for param, col, desc in [
        ("w", "w", "theta-conditioned template-relative higher-order attenuation descriptor"),
        ("eta", "eta", "template-relative anisotropic log-amplitude descriptor"),
    ]:
        if col not in df:
            continue
        tmp = df[["name", col]].copy() if "name" in df.columns else df[[col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["parameter"] = param
        tmp["description"] = desc
        rows.append(tmp)
        hist = compute_histogram_data(tmp["value"].to_numpy(float), bins=args.hist_bins)
        if not hist.empty:
            hist["parameter"] = param
            hist.to_csv(out_dir / f"fig4c_{param}_histogram.csv", index=False, encoding="utf-8-sig")
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    long.to_csv(out_dir / "fig4c_readout_parameter_values_long.csv", index=False, encoding="utf-8-sig")
    summ_rows = []
    for param, g in long.groupby("parameter") if not long.empty else []:
        v = pd.to_numeric(g["value"], errors="coerce").dropna().to_numpy(float)
        summ_rows.append({
            "parameter": param,
            "n": int(v.size),
            "mean": float(np.mean(v)) if v.size else np.nan,
            "median": float(np.median(v)) if v.size else np.nan,
            "std": float(np.std(v, ddof=1)) if v.size > 1 else np.nan,
            "p05": float(np.percentile(v, 5)) if v.size else np.nan,
            "p25": float(np.percentile(v, 25)) if v.size else np.nan,
            "p75": float(np.percentile(v, 75)) if v.size else np.nan,
            "p95": float(np.percentile(v, 95)) if v.size else np.nan,
        })
    summary = pd.DataFrame(summ_rows)
    summary.to_csv(out_dir / "fig4c_readout_parameter_distribution_summary.csv", index=False, encoding="utf-8-sig")
    return {"fig4c_values": long, "fig4c_summary": summary}


def compute_fig5(master: pd.DataFrame, out_dir: Path, args) -> Dict[str, pd.DataFrame]:
    df = master.loc[master["fig45_keep"]].copy().reset_index(drop=True)
    readout_df = master.loc[master["snr_analysis_keep"]].copy().reset_index(drop=True)
    outputs: Dict[str, pd.DataFrame] = {}

    # 5a
    fig5a_cols = [c for c in ["name", "w", "c_B", "c_AH", "theta_deg", "log10_I_floor"] if c in df.columns]
    fig5a = df[fig5a_cols].copy()
    fig5a.to_csv(out_dir / "fig5a_w_vs_cB_points.csv", index=False, encoding="utf-8-sig")
    corr_rows = [
        correlation_summary(df["w"], df["c_B"], "w", "c_B", args.bootstrap_n, args.seed),
        correlation_summary(df["w"], df["c_AH"], "w", "c_AH", args.bootstrap_n, args.seed),
    ]
    fig5a_corr = pd.DataFrame(corr_rows)
    fig5a_corr.to_csv(out_dir / "fig5a_w_vs_cB_correlations.csv", index=False, encoding="utf-8-sig")
    outputs["fig5a_points"] = fig5a
    outputs["fig5a_correlations"] = fig5a_corr

    # 5b
    rows = []
    for param, col in [("w", "w"), ("eta", "eta")]:
        if col not in readout_df:
            continue
        tmp = readout_df[["name", "log10_I_floor", col]].copy() if "name" in readout_df.columns else readout_df[["log10_I_floor", col]].copy()
        tmp = tmp.loc[np.isfinite(tmp[["log10_I_floor", col]].to_numpy(float)).all(axis=1)]
        tmp = tmp.rename(columns={col: "parameter_value"})
        tmp["parameter"] = param
        rows.append(tmp)
    fig5b = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fig5b.to_csv(out_dir / "fig5b_readout_vs_floor_points_long.csv", index=False, encoding="utf-8-sig")
    corr_rows = []
    for param, col in [("w", "w"), ("eta", "eta")]:
        corr_rows.append(correlation_summary(readout_df["log10_I_floor"], readout_df[col], "log10_I_floor", param, args.bootstrap_n, args.seed))
    fig5b_corr = pd.DataFrame(corr_rows)
    fig5b_corr.to_csv(out_dir / "fig5b_readout_vs_floor_correlations.csv", index=False, encoding="utf-8-sig")
    outputs["fig5b_points_long"] = fig5b
    outputs["fig5b_correlations"] = fig5b_corr

    # 5c corrections.
    wX = df[["w"]].to_numpy(float)
    full_cols = ["w"]
    if np.isfinite(df["eta_cos2phi"].to_numpy(float)).sum() >= 10 and np.isfinite(df["eta_sin2phi"].to_numpy(float)).sum() >= 10:
        full_cols += ["eta_cos2phi", "eta_sin2phi"]
    elif np.isfinite(df["eta"].to_numpy(float)).sum() >= 10:
        full_cols += ["eta"]
    fullX = df[full_cols].to_numpy(float)
    floorX = df[["log10_I_floor"]].to_numpy(float)

    cAH = df["c_AH"].to_numpy(float)
    cbeta = df["c_beta_signed"].to_numpy(float)

    cAH_w, beta_cAH_w, mu_w = linear_residualize_to_mean(cAH, wX)
    cbeta_w, beta_cbeta_w, _ = linear_residualize_to_mean(cbeta, wX)
    cAH_full, beta_cAH_full, mu_full = linear_residualize_to_mean(cAH, fullX)
    cbeta_full, beta_cbeta_full, _ = linear_residualize_to_mean(cbeta, fullX)
    cAH_floor, beta_cAH_floor, mu_floor = linear_residualize_to_mean(cAH, floorX)
    cbeta_floor, beta_cbeta_floor, _ = linear_residualize_to_mean(cbeta, floorX)

    fig5c = df[["name", "w", "eta", "c_AH", "c_B", "c_beta_signed", "theta_deg", "log10_I_floor"]].copy() if "name" in df.columns else df[["w", "eta", "c_AH", "c_B", "c_beta_signed", "theta_deg", "log10_I_floor"]].copy()
    fig5c["c_AH_w_corrected"] = cAH_w
    fig5c["c_B_w_corrected"] = np.abs(cbeta_w)
    fig5c["theta_w_corrected_deg"] = theta_from_cAH_cB(cAH_w, np.abs(cbeta_w))
    fig5c["c_AH_full_readout_corrected"] = cAH_full
    fig5c["c_B_full_readout_corrected"] = np.abs(cbeta_full)
    fig5c["theta_full_readout_corrected_deg"] = theta_from_cAH_cB(cAH_full, np.abs(cbeta_full))
    fig5c["c_AH_transmission_corrected"] = cAH_floor
    fig5c["c_B_transmission_corrected"] = np.abs(cbeta_floor)
    fig5c["theta_transmission_corrected_deg"] = theta_from_cAH_cB(cAH_floor, np.abs(cbeta_floor))
    fig5c.to_csv(out_dir / "fig5c_theta_floor_correction_points.csv", index=False, encoding="utf-8-sig")

    theta_variants = [
        ("original", "theta_deg"),
        ("w_corrected", "theta_w_corrected_deg"),
        ("full_readout_corrected", "theta_full_readout_corrected_deg"),
    ]
    corr_rows = []
    orig_abs = None
    for label, col in theta_variants:
        cr = correlation_summary(fig5c["log10_I_floor"], fig5c[col], "log10_I_floor", label, args.bootstrap_n, args.seed)
        cr["theta_variant"] = label
        cr["rho"] = cr["spearman_rho"]
        cr["abs_rho"] = abs(cr["spearman_rho"]) if np.isfinite(cr["spearman_rho"]) else np.nan
        if label == "original":
            orig_abs = cr["abs_rho"]
            cr["delta_abs_rho_vs_original"] = 0.0
            cr["percent_abs_rho_reduction_vs_original"] = 0.0
        else:
            cr["delta_abs_rho_vs_original"] = cr["abs_rho"] - orig_abs if orig_abs is not None else np.nan
            cr["percent_abs_rho_reduction_vs_original"] = 100.0 * (orig_abs - cr["abs_rho"]) / orig_abs if orig_abs and np.isfinite(orig_abs) and orig_abs != 0 else np.nan
        corr_rows.append(cr)
    fig5c_corr = pd.DataFrame(corr_rows)
    fig5c_corr.to_csv(out_dir / "fig5c_theta_floor_correction_correlations.csv", index=False, encoding="utf-8-sig")
    outputs["fig5c_points"] = fig5c
    outputs["fig5c_correlations"] = fig5c_corr

    # Regression coefficients used in corrections.
    coef_rows = []
    for target, beta, predictors in [
        ("c_AH_w", beta_cAH_w, ["w"]),
        ("c_beta_signed_w", beta_cbeta_w, ["w"]),
        ("c_AH_full_readout", beta_cAH_full, full_cols),
        ("c_beta_signed_full_readout", beta_cbeta_full, full_cols),
        ("c_AH_transmission", beta_cAH_floor, ["log10_I_floor"]),
        ("c_beta_signed_transmission", beta_cbeta_floor, ["log10_I_floor"]),
    ]:
        for pred, b in zip(predictors, beta):
            coef_rows.append({"target": target, "predictor": pred, "coefficient": b})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out_dir / "fig5c_fig5d_correction_regression_coefficients.csv", index=False, encoding="utf-8-sig")
    outputs["correction_coefficients"] = coef_df

    # 5d point clouds.
    point_rows = []
    for series, xcol, ycol in [
        ("original", "c_AH", "c_B"),
        ("w_readout_corrected_to_mean_w", "c_AH_w_corrected", "c_B_w_corrected"),
        ("full_readout_corrected_to_mean_readout", "c_AH_full_readout_corrected", "c_B_full_readout_corrected"),
        ("transmission_corrected_to_mean_floor", "c_AH_transmission_corrected", "c_B_transmission_corrected"),
    ]:
        tmp = fig5c[["name", xcol, ycol]].copy() if "name" in fig5c.columns else fig5c[[xcol, ycol]].copy()
        tmp = tmp.rename(columns={xcol: "c_AH_plot", ycol: "c_B_plot"})
        tmp["series"] = series
        point_rows.append(tmp)
    fig5d_points = pd.concat(point_rows, ignore_index=True)
    fig5d_points.to_csv(out_dir / "fig5d_cAH_cB_points_long.csv", index=False, encoding="utf-8-sig")
    outputs["fig5d_points_long"] = fig5d_points

    # 5d correction vectors.
    vec_rows = []
    for corr_type, xend, yend in [
        ("original_to_w_readout_corrected", "c_AH_w_corrected", "c_B_w_corrected"),
        ("original_to_full_readout_corrected", "c_AH_full_readout_corrected", "c_B_full_readout_corrected"),
        ("original_to_transmission_corrected", "c_AH_transmission_corrected", "c_B_transmission_corrected"),
    ]:
        base_cols = ["name", "c_AH", "c_B", xend, yend]
        tmp = fig5c[base_cols].copy() if "name" in fig5c.columns else fig5c[["c_AH", "c_B", xend, yend]].copy()
        tmp = tmp.rename(columns={"c_AH": "x_start_c_AH", "c_B": "y_start_c_B", xend: "x_end_c_AH", yend: "y_end_c_B"})
        tmp["dx_c_AH"] = tmp["x_end_c_AH"] - tmp["x_start_c_AH"]
        tmp["dy_c_B"] = tmp["y_end_c_B"] - tmp["y_start_c_B"]
        tmp["vector_magnitude"] = np.sqrt(tmp["dx_c_AH"] ** 2 + tmp["dy_c_B"] ** 2)
        tmp["vector_angle_deg"] = np.degrees(np.arctan2(tmp["dy_c_B"], tmp["dx_c_AH"]))
        tmp["correction_type"] = corr_type
        vec_rows.append(tmp)
    fig5d_vectors = pd.concat(vec_rows, ignore_index=True)
    fig5d_vectors.to_csv(out_dir / "fig5d_correction_vectors_long.csv", index=False, encoding="utf-8-sig")
    outputs["fig5d_vectors_long"] = fig5d_vectors

    # 5d driver arrows.
    arrows = []
    arrows.append(driver_arrow_from_univariate(df, "w", "c_AH", "c_B", "w"))
    arrows.append(driver_arrow_from_univariate(df, "log10_I_floor", "c_AH", "c_B", "log10_I_floor"))
    if len(full_cols) > 1:
        arrows.append(driver_arrow_from_pc1(df, full_cols, "c_AH", "c_B", "full_readout_PC1"))
    arrows_df = pd.DataFrame(arrows)
    center_x = float(np.nanmedian(df["c_AH"].to_numpy(float)))
    center_y = float(np.nanmedian(df["c_B"].to_numpy(float)))
    arrows_df["x_start"] = center_x
    arrows_df["y_start"] = center_y
    arrows_df["x_end_1sd"] = arrows_df["x_start"] + arrows_df["dx_1sd"]
    arrows_df["y_end_1sd"] = arrows_df["y_start"] + arrows_df["dy_1sd"]
    arrows_df.to_csv(out_dir / "fig5d_driver_signature_arrows.csv", index=False, encoding="utf-8-sig")

    # Scaled arrows for display.
    scale_arg = str(args.driver_arrow_scale)
    if scale_arg.lower() == "auto":
        x_range = np.nanpercentile(df["c_AH"], 95) - np.nanpercentile(df["c_AH"], 5)
        y_range = np.nanpercentile(df["c_B"], 95) - np.nanpercentile(df["c_B"], 5)
        target_len = 0.22 * min(x_range, y_range)
        max_len = float(np.nanmax(arrows_df["magnitude_1sd"].to_numpy(float)))
        scale = target_len / max_len if max_len > 0 and np.isfinite(max_len) else 1.0
        scale = float(np.clip(scale, 1.0, 8.0))
    else:
        scale = float(scale_arg)
    scaled = arrows_df.copy()
    scaled["display_scale"] = scale
    scaled["dx_display"] = scaled["dx_1sd"] * scale
    scaled["dy_display"] = scaled["dy_1sd"] * scale
    scaled["x_end_display"] = scaled["x_start"] + scaled["dx_display"]
    scaled["y_end_display"] = scaled["y_start"] + scaled["dy_display"]
    scaled.to_csv(out_dir / "fig5d_driver_signature_arrows_scaled_for_display.csv", index=False, encoding="utf-8-sig")
    outputs["fig5d_driver_arrows"] = arrows_df
    outputs["fig5d_driver_arrows_scaled"] = scaled

    return outputs

# =============================================================================
# Preview plots
# =============================================================================


def maybe_savefig(path: Path):
    if plt is None:
        return
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_preview_plots(outputs: Dict[str, pd.DataFrame], out_dir: Path, preview_dir: Path):
    if plt is None:
        warnings.warn("matplotlib not available; preview plots skipped.")
        return
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Fig4a
    if "fig4a_fft_xyz" in outputs and "fig4_markers" in outputs:
        df = outputs["fig4a_fft_xyz"]
        mk = outputs["fig4_markers"]
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        sc = ax.scatter(df["Gx_normG1"], df["Gy_normG1"], c=df["log10_power"], s=7, marker="s")
        for shell, marker, label in [("G1", "o", "G1"), ("sqrt3G1", "^", "√3G1"), ("2G1", "s", "2G1")]:
            g = mk[mk["shell"] == shell]
            if g.empty:
                continue
            ax.scatter(g["Gx_normG1"], g["Gy_normG1"], marker=marker, s=60, facecolors="none", edgecolors="white" if shell=="sqrt3G1" else "cyan", linewidths=1.4, label=label)
        ax.set_aspect("equal")
        ax.set_xlabel("Gx / |G1|")
        ax.set_ylabel("Gy / |G1|")
        ax.set_title("Fig. 4a source: ensemble-mean unit-cell FFT")
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="log10 power")
        maybe_savefig(preview_dir / "preview_fig4a_ensemble_mean_fft.png")

    # Fig4b
    if "fig4b_shell_summary" in outputs:
        s = outputs["fig4b_shell_summary"].copy()
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        order = [x for x in ["G1", "sqrt3G1", "2G1", "off_shell_background_q2_le_12"] if x in set(s["category"])]
        g = s.set_index("category").loc[order].reset_index()
        ax.bar(np.arange(len(g)), g["power_mean"])
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels(g["category"], rotation=25, ha="right")
        ax.set_ylabel("mean peak/background power")
        ax.set_title("Fig. 4b source: shell power summary")
        maybe_savefig(preview_dir / "preview_fig4b_shell_power_barplot.png")

    # Fig4c
    if "fig4c_values" in outputs and not outputs["fig4c_values"].empty:
        vals = outputs["fig4c_values"]
        fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
        w = vals.loc[vals["parameter"] == "w", "value"].dropna().to_numpy(float)
        e = vals.loc[vals["parameter"] == "eta", "value"].dropna().to_numpy(float)
        if w.size:
            ax1.hist(w, bins=30, alpha=0.65, label="w", edgecolor="black")
            ax1.axvline(np.nanmedian(w), ls="--", lw=1.2)
        ax1.set_xlabel("w / template-relative attenuation")
        ax1.set_ylabel("count for w")
        ax2 = ax1.twiny()
        if e.size:
            ax2.hist(e, bins=30, alpha=0.35, label="η", edgecolor="black")
            ax2.set_xlabel("η / anisotropy descriptor")
        ax1.set_title("Fig. 4c source: readout descriptor distributions")
        maybe_savefig(preview_dir / "preview_fig4c_w_eta_histograms.png")

    # Fig5a
    if "fig5a_points" in outputs:
        d = outputs["fig5a_points"]
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.scatter(d["w"], d["c_B"], s=18, alpha=0.65)
        ax.set_xlabel("w / template-relative attenuation")
        ax.set_ylabel("c_B = |c_beta|")
        ax.set_title("Fig. 5a source: w vs c_B")
        maybe_savefig(preview_dir / "preview_fig5a_w_vs_cB.png")

    # Fig5b
    if "fig5b_points_long" in outputs:
        d = outputs["fig5b_points_long"]
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for param, g in d.groupby("parameter"):
            ax.scatter(g["log10_I_floor"], g["parameter_value"], s=18, alpha=0.6, label=param)
        ax.set_xlabel("log10 |I_floor|")
        ax.set_ylabel("readout descriptor value")
        ax.legend()
        ax.set_title("Fig. 5b source: readout descriptors vs current floor")
        maybe_savefig(preview_dir / "preview_fig5b_readout_vs_floor.png")

    # Fig5c
    if "fig5c_correlations" in outputs:
        d = outputs["fig5c_correlations"].copy()
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.bar(np.arange(len(d)), d["abs_rho"])
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["theta_variant"], rotation=25, ha="right")
        ax.set_ylabel("|Spearman ρ(θ, I_floor)|")
        ax.set_title("Fig. 5c source: residual θ–I_floor coupling")
        maybe_savefig(preview_dir / "preview_fig5c_theta_floor_absrho.png")

    # Fig5d arrows
    if "fig5d_points_long" in outputs and "fig5d_driver_arrows_scaled" in outputs:
        pts = outputs["fig5d_points_long"]
        arr = outputs["fig5d_driver_arrows_scaled"]
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        orig = pts[pts["series"] == "original"]
        ax.scatter(orig["c_AH_plot"], orig["c_B_plot"], s=12, alpha=0.35, label="original")
        for _, r in arr.iterrows():
            ax.arrow(r["x_start"], r["y_start"], r["dx_display"], r["dy_display"], head_width=0.04, length_includes_head=True, linewidth=2, alpha=0.9, label=str(r["driver"]))
        ax.set_xlabel("c_AH")
        ax.set_ylabel("c_B = |c_beta|")
        ax.legend(fontsize=8)
        ax.set_title("Fig. 5d source: driver signature arrows")
        maybe_savefig(preview_dir / "preview_fig5d_cAH_cB_driver_arrows.png")

    # Fig5d delta vector space
    if "fig5d_vectors_long" in outputs:
        v = outputs["fig5d_vectors_long"]
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        for ct, g in v.groupby("correction_type"):
            # Subsample for preview if huge.
            gg = g.sample(min(len(g), 800), random_state=1) if len(g) > 800 else g
            ax.scatter(gg["dx_c_AH"], gg["dy_c_B"], s=10, alpha=0.4, label=ct)
        ax.axhline(0, lw=0.8, color="0.5")
        ax.axvline(0, lw=0.8, color="0.5")
        ax.set_xlabel("Δc_AH")
        ax.set_ylabel("Δc_B")
        ax.legend(fontsize=7)
        ax.set_title("Fig. 5d diagnostic: correction-vector space")
        maybe_savefig(preview_dir / "preview_fig5d_delta_vector_space.png")

# =============================================================================
# Documentation/export
# =============================================================================


def write_index_and_readme(out_dir: Path, outputs: Dict[str, pd.DataFrame], settings: Dict[str, object]):
    index_rows = []
    descriptions = {
        "fig4a_fft_xyz": "Fig. 4a ensemble-mean folded-unit-cell FFT log10 power map; plot X=Gx_normG1, Y=Gy_normG1, Z=log10_power.",
        "fig4_markers": "Fig. 4a marker positions for G1, sqrt3G1 and 2G1 shells in the 120° convention.",
        "fig4b_shell_summary": "Fig. 4b shell-averaged power and amplitude at G1, sqrt3G1, 2G1 and off-shell background modes.",
        "fig4c_values": "Fig. 4c raw values for w and eta histograms.",
        "fig4c_summary": "Fig. 4c distribution summary statistics for w and eta.",
        "fig5a_points": "Fig. 5a point data for w versus c_B.",
        "fig5a_correlations": "Fig. 5a correlation summary.",
        "fig5b_points_long": "Fig. 5b long-format readout descriptor versus log10 I_floor data.",
        "fig5b_correlations": "Fig. 5b correlation summary.",
        "fig5c_points": "Fig. 5c theta and corrected theta values before/after readout correction.",
        "fig5c_correlations": "Fig. 5c Spearman theta-I_floor correlations before/after correction.",
        "fig5d_points_long": "Fig. 5d c_AH-c_B point clouds: original, readout-corrected and transmission-corrected.",
        "fig5d_vectors_long": "Fig. 5d individual correction vectors in Δc_AH-Δc_B space.",
        "fig5d_driver_arrows": "Fig. 5d unscaled 1-SD driver signature arrows.",
        "fig5d_driver_arrows_scaled": "Fig. 5d display-scaled driver signature arrows.",
    }
    file_map = {
        "fig4a_fft_xyz": "fig4a_ensemble_mean_fft_log10_power_xyz.csv",
        "fig4_markers": "fig4a_shell_marker_points_for_fft_overlay.csv",
        "fig4b_shell_summary": "fig4b_shell_power_summary.csv",
        "fig4c_values": "fig4c_readout_parameter_values_long.csv",
        "fig4c_summary": "fig4c_readout_parameter_distribution_summary.csv",
        "fig5a_points": "fig5a_w_vs_cB_points.csv",
        "fig5a_correlations": "fig5a_w_vs_cB_correlations.csv",
        "fig5b_points_long": "fig5b_readout_vs_floor_points_long.csv",
        "fig5b_correlations": "fig5b_readout_vs_floor_correlations.csv",
        "fig5c_points": "fig5c_theta_floor_correction_points.csv",
        "fig5c_correlations": "fig5c_theta_floor_correction_correlations.csv",
        "fig5d_points_long": "fig5d_cAH_cB_points_long.csv",
        "fig5d_vectors_long": "fig5d_correction_vectors_long.csv",
        "fig5d_driver_arrows": "fig5d_driver_signature_arrows.csv",
        "fig5d_driver_arrows_scaled": "fig5d_driver_signature_arrows_scaled_for_display.csv",
    }
    for key, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            index_rows.append({
                "key": key,
                "file": file_map.get(key, ""),
                "n_rows": int(len(df)),
                "description": descriptions.get(key, ""),
            })
    pd.DataFrame(index_rows).to_csv(out_dir / "fig4_fig5_source_data_index.csv", index=False, encoding="utf-8-sig")
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Fig. 4 / Fig. 5 source data from template-relative readout analysis")
    lines.append("")
    lines.append("This folder contains source data for plotting Fig. 4 and Fig. 5 using the theta-conditioned template-relative folded-unit-cell FFT analysis.")
    lines.append("")
    lines.append("## Important interpretation")
    lines.append("")
    lines.append("The default `w` column is `w_rel_primary`, a theta-conditioned template-relative higher-order attenuation descriptor. It should be described as a relative/apparent readout-related spectral attenuation descriptor, not as an absolute real-space readout width unless further calibrated.")
    lines.append("")
    n_all = int(settings["n_fig4ab_maps"])
    n_selected = int(settings["n_fig4c_fig5_maps"])
    snr_threshold = float(settings["snr_threshold"])
    lines.append(
        f"Figure 4a/b uses the ensemble of all {n_all} maps. Figure 4c and "
        f"Figure 5 use the per-image selection recorded in "
        f"`fig4_fig5_filter_audit.csv`; {n_selected} maps satisfy the inclusive "
        f"sqrt(3) SNR >= {snr_threshold:g} rule."
    )
    lines.append("")
    lines.append("## 120° reciprocal convention")
    lines.append("")
    lines.append("Shells are indexed using q² = h² + h k + k². G1, sqrt3G1 and 2G1 correspond to q²=1, 3 and 4, respectively.")
    lines.append("The separate SNR selection uses q² = h² + h k + k², matching the readout-shell convention; its exact modes are recorded in `analysis/snr_resolved_readout/settings.json`.")
    lines.append("")
    lines.append("## Recommended plotting files")
    lines.append("")
    lines.append("- Fig. 4a: fig4a_ensemble_mean_fft_log10_power_xyz.csv + fig4a_shell_marker_points_for_fft_overlay.csv")
    lines.append("- Fig. 4b: fig4b_shell_power_summary.csv")
    lines.append("- Fig. 4c: fig4c_readout_parameter_values_long.csv")
    lines.append("- Fig. 5a: fig5a_w_vs_cB_points.csv")
    lines.append("- Fig. 5b: fig5b_readout_vs_floor_points_long.csv")
    lines.append("- Fig. 5c: fig5c_theta_floor_correction_correlations.csv")
    lines.append("- Fig. 5d: fig5d_cAH_cB_points_long.csv and fig5d_driver_signature_arrows_scaled_for_display.csv")
    (out_dir / "README_Fig4_Fig5_source_data.md").write_text("\n".join(lines), encoding="utf-8")


def write_excel(out_dir: Path, outputs: Dict[str, pd.DataFrame], skip_excel: bool):
    if skip_excel:
        return
    xlsx = out_dir / "Fig4_Fig5_source_data.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            for key, df in outputs.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(writer, sheet_name=clean_sheet_name(key), index=False)
    except Exception as exc:
        warnings.warn(f"Could not write Excel workbook {xlsx}: {exc}")

# =============================================================================
# Main
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Fig. 4/Fig. 5 source data from template-relative readout analysis.",
        allow_abbrev=False,
    )
    p.add_argument("--analysis-dir", "--analysis_dir", dest="analysis_dir", required=True)
    p.add_argument("--readout-csv", "--readout_csv", dest="readout_csv", type=str, default=None)
    p.add_argument(
        "--snr-selection-csv",
        "--snr_selection_csv",
        dest="snr_selection_csv",
        type=str,
        default=None,
        help=(
            "Per-image selection written by 05_compute_snr_resolved_statistics.py; "
            "defaults to ANALYSIS_DIR/snr_resolved_readout/snr_resolved_readout_per_image.csv"
        ),
    )
    p.add_argument("--coord-csv", "--coord_csv", dest="coord_csv", type=str, default=None)
    p.add_argument("--out-dir", "--out_dir", dest="out_dir", type=str, default=None)
    p.add_argument("--out-folder", "--out_folder", dest="out_folder", type=str, default=DEFAULT_OUT_FOLDER_NAME)
    p.add_argument("--map-key", "--map_key", dest="map_key", type=str, default=DEFAULT_MAP_KEY)
    p.add_argument("--angle-deg", "--angle_deg", dest="angle_deg", type=float, default=DEFAULT_ANGLE_DEG)
    p.add_argument("--fft-crop-q", "--fft_crop_q", dest="fft_crop_q", type=float, default=DEFAULT_FFT_CROP_Q)
    p.add_argument("--hist-bins", "--hist_bins", dest="hist_bins", type=int, default=30)
    p.add_argument("--bootstrap-n", "--bootstrap_n", dest="bootstrap_n", type=int, default=DEFAULT_BOOTSTRAP_N)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument(
        "--snr-threshold",
        "--snr_threshold",
        dest="snr_threshold",
        type=float,
        default=DEFAULT_SNR_THRESHOLD,
    )
    p.add_argument("--driver-arrow-scale", "--driver_arrow_scale", dest="driver_arrow_scale", type=str, default=DEFAULT_DRIVER_ARROW_SCALE)
    p.add_argument("--skip-excel", "--skip_excel", dest="skip_excel", action="store_true")
    p.add_argument("--no-plots", "--no_plots", dest="no_plots", action="store_true")

    # Optional column overrides.
    p.add_argument("--w-col", "--w_col", dest="w_col", type=str, default=None)
    p.add_argument("--eta-col", "--eta_col", dest="eta_col", type=str, default=None)
    p.add_argument("--eta-orientation-col", "--eta_orientation_col", dest="eta_orientation_col", type=str, default=None)
    p.add_argument("--theta-col", "--theta_col", dest="theta_col", type=str, default=None)
    p.add_argument("--cAH-col", "--cAH_col", dest="cAH_col", type=str, default=None)
    p.add_argument("--cB-col", "--cB_col", dest="cB_col", type=str, default=None)
    p.add_argument("--cbeta-signed-col", "--cbeta_signed_col", dest="cbeta_signed_col", type=str, default=None)
    p.add_argument("--floor-col", "--floor_col", dest="floor_col", type=str, default=None)

    # Optional filtering.
    p.add_argument("--w-min", "--w_min", dest="w_min", type=float, default=None)
    p.add_argument("--w-max", "--w_max", dest="w_max", type=float, default=None)
    p.add_argument("--include-nonfinite-core", "--include_nonfinite_core", dest="include_nonfinite_core", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    analysis_dir = Path(args.analysis_dir)
    readout_csv = Path(args.readout_csv) if args.readout_csv else None
    coord_csv = Path(args.coord_csv) if args.coord_csv else None
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / args.out_folder
    ensure_dir(out_dir)
    preview_dir = ensure_dir(out_dir / "preview_png")

    master, colinfo, readout_csv_used, coord_csv_used = load_master_table(analysis_dir, readout_csv, coord_csv, args)
    master.to_csv(out_dir / "fig4_fig5_master_table.csv", index=False, encoding="utf-8-sig")
    master[[c for c in ["name", "array_index", "w", "eta", "theta_deg", "c_AH", "c_B", "log10_I_floor", "sqrt3_amp_snr", "sqrt3_resolved", "fig45_keep", "fig45_filter_reason"] if c in master.columns]].to_csv(
        out_dir / "fig4_fig5_filter_audit.csv", index=False, encoding="utf-8-sig"
    )
    filter_summary = pd.DataFrame([{
        "n_input": int(len(master)),
        "n_retained_fig45_keep": int(master["fig45_keep"].sum()),
        "w_min": args.w_min,
        "w_max": args.w_max,
        "readout_csv": readout_csv_used.name,
        "coord_csv": coord_csv_used.name if coord_csv_used else "",
        "n_sqrt3_resolved": int(master["sqrt3_resolved"].sum()),
        "snr_threshold": args.snr_threshold,
        **colinfo,
    }])
    filter_summary.to_csv(out_dir / "fig4_fig5_filter_summary.csv", index=False, encoding="utf-8-sig")

    outputs: Dict[str, pd.DataFrame] = {}
    outputs.update(compute_fig4_fft_outputs(analysis_dir, out_dir, args.map_key, args.angle_deg, args.fft_crop_q))
    outputs.update(compute_fig4c(master, out_dir, args))
    outputs.update(compute_fig5(master, out_dir, args))

    write_index_and_readme(out_dir, outputs, settings={
        "analysis_dir": analysis_dir.name,
        "readout_csv": readout_csv_used.name,
        "coord_csv": coord_csv_used.name if coord_csv_used else "",
        "out_dir": out_dir.name,
        "map_key": args.map_key,
        "angle_deg": args.angle_deg,
        "w_col": colinfo.get("w_col_input"),
        "eta_col": colinfo.get("eta_col_input"),
        "theta_col": colinfo.get("theta_col_input"),
        "cAH_col": colinfo.get("cAH_col_input"),
        "cB_col": colinfo.get("cB_col_input"),
        "floor_col": colinfo.get("floor_col_input"),
        "snr_selection_csv": colinfo.get("snr_selection_csv"),
        "snr_threshold": args.snr_threshold,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "n_input": int(len(master)),
        "n_retained": int(master["fig45_keep"].sum()),
        "n_fig4ab_maps": int(len(master)),
        "n_fig4c_fig5_maps": int(master["fig45_keep"].sum()),
        "n_fig5b_snr_population": int(master["snr_analysis_keep"].sum()),
    })
    write_excel(out_dir, outputs, args.skip_excel)
    if not args.no_plots:
        make_preview_plots(outputs, out_dir, preview_dir)

    print("=== DONE: Fig. 4/Fig. 5 template-relative readout source data ===")
    print(f"analysis_dir: {analysis_dir}")
    print(f"readout_csv: {readout_csv_used}")
    print(f"coord_csv: {coord_csv_used}")
    print(f"rows input: {len(master)}")
    print(f"rows retained: {int(master['fig45_keep'].sum())}")
    print(f"output: {out_dir}")
    if "fig5a_correlations" in outputs:
        print("\nFig. 5a correlations:")
        print(outputs["fig5a_correlations"].to_string(index=False))
    if "fig5c_correlations" in outputs:
        print("\nFig. 5c theta-floor correlations:")
        cols = ["theta_variant", "n", "rho", "spearman_ci95_low", "spearman_ci95_high", "abs_rho", "percent_abs_rho_reduction_vs_original"]
        print(outputs["fig5c_correlations"][[c for c in cols if c in outputs["fig5c_correlations"].columns]].to_string(index=False))
    if "fig5d_driver_arrows" in outputs:
        print("\nFig. 5d driver arrows, unscaled 1 SD:")
        cols = ["driver", "n", "dx_1sd", "dy_1sd", "magnitude_1sd", "angle_deg"]
        print(outputs["fig5d_driver_arrows"][[c for c in cols if c in outputs["fig5d_driver_arrows"].columns]].to_string(index=False))


if __name__ == "__main__":
    main()
