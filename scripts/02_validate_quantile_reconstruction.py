# -*- coding: utf-8 -*-
"""Leave-quantile-out validation of the empirical unit-cell reconstruction.

The input is the output directory created by 01_extract_motif_coordinates.py.
The analysis learns the Fourier-domain kernel without the target theta window
and reports full-map and site-excluded agreement statistics."""

from __future__ import annotations

import argparse

import itertools
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# User settings
# =============================================================================

# Folder produced by the main same-condition motif-analysis code.
ANALYSIS_OUT_DIR = Path(".")

OUTPUT_SUBDIR = "physics_forward_model_validation"

# Quantile windows. Use a local window around each theta percentile.
QUANTILE_PERCENTILES = (5, 25, 50, 75, 95)
QUANTILE_WINDOW_FRACTION = 0.06
QUANTILE_WINDOW_MIN_N = 15
QUANTILE_WINDOW_MAX_N = 31

# Learned-kernel settings.
KERNEL_TIKHONOV_RELATIVE = 2.0e-3
APPLY_KERNEL_LOW_PASS = True
KERNEL_LOW_PASS_SIGMA_CYCLES = 14.0
FIT_SINGLE_SCALE_ON_TRAIN = True

# Masked-agreement settings: exclude pixels near B/A/H source masks.
RUN_MASKED_AGREEMENT = True
SITE_MASK_SIGMA_FRAC = 0.055
SITE_MASK_EXCLUDE_WEIGHT = 0.20

# Leave-quantile-out null controls.
RUN_INTENSITY_SHUFFLE_NULL = True
RUN_ROLE_PERMUTATION_NULL = True
RUN_FIXED_GLOBAL_INTENSITY_NULL = True
N_INTENSITY_SHUFFLES = 100
NULL_RANDOM_SEED = 20260519

# Repeated train/test validation.
RUN_REPEATED_TRAIN_TEST = True
N_REPEATED_SPLITS = 30
TEST_FRACTION = 0.25
STRATIFY_BY_THETA = True
N_THETA_STRATA = 5
REPEATED_SPLIT_SEED0 = 20260601
REPEATED_QUANTILE_WINDOW_FRACTION = 0.18
REPEATED_QUANTILE_WINDOW_MIN_N = 9
REPEATED_QUANTILE_WINDOW_MAX_N = 31

# Plot/export settings.
GRAPHITE_DISPLAY_ANGLE_DEG = 120.0
FIG_DPI = 250
CMAP = "RdBu_r"

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT6 = math.sqrt(6.0)

# =============================================================================
# Geometry and basic helpers
# =============================================================================


def uv_to_graphite60(u, v, angle_deg: float = GRAPHITE_DISPLAY_ANGLE_DEG):
    ang = math.radians(angle_deg)
    return np.asarray(u) + math.cos(ang) * np.asarray(v), math.sin(ang) * np.asarray(v)


def unit_cell_edges(n: int) -> Tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n + 1)
    v = np.linspace(0.0, 1.0, n + 1)
    Ue, Ve = np.meshgrid(u, v)
    return uv_to_graphite60(Ue, Ve)


def unit_cell_centers(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = (np.arange(n) + 0.5) / n
    v = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, v)
    X, Y = uv_to_graphite60(U, V)
    return U, V, X, Y


def zero_mean(m: np.ndarray) -> np.ndarray:
    z = np.asarray(m, dtype=float)
    return z - np.nanmean(z)


def normalized_corr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if mask is not None:
        aa = aa[np.asarray(mask, dtype=bool)]
        bb = bb[np.asarray(mask, dtype=bool)]
    else:
        aa = aa.ravel()
        bb = bb.ravel()
    finite = np.isfinite(aa) & np.isfinite(bb)
    if finite.sum() < 3:
        return float("nan")
    aa = aa[finite] - np.nanmean(aa[finite])
    bb = bb[finite] - np.nanmean(bb[finite])
    if np.nanstd(aa) <= 0 or np.nanstd(bb) <= 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def agreement(measured: np.ndarray, predicted: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    m = zero_mean(measured)
    p = zero_mean(predicted)
    if mask is not None:
        m = m[np.asarray(mask, dtype=bool)]
        p = p[np.asarray(mask, dtype=bool)]
    finite = np.isfinite(m) & np.isfinite(p)
    if finite.sum() < 3:
        return float("nan")
    m = m[finite]
    p = p[finite]
    denom = float(np.sqrt(np.nanmean(m ** 2)))
    if denom <= 0:
        return float("nan")
    err = float(np.sqrt(np.nanmean((m - p) ** 2)))
    return float(1.0 - err / denom)


def relative_rmse(measured: np.ndarray, predicted: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = agreement(measured, predicted, mask=mask)
    return float(1.0 - a) if np.isfinite(a) else float("nan")


def save_xyz_map(arr: np.ndarray, path: Path) -> None:
    n = arr.shape[0]
    U, V, X, Y = unit_cell_centers(n)
    pd.DataFrame({
        "u_center": U.ravel(),
        "v_center": V.ravel(),
        "x_graphite60": X.ravel(),
        "y_graphite60": Y.ravel(),
        "z": np.asarray(arr, dtype=float).ravel(),
    }).to_csv(path, index=False, encoding="utf-8-sig")


def save_matrix_and_xyz(arr: np.ndarray, out_dir: Path, prefix: str) -> Tuple[Path, Path]:
    mat_path = out_dir / f"{prefix}_matrix_uv.csv"
    xyz_path = out_dir / f"{prefix}_graphite60_xyz.csv"
    pd.DataFrame(arr).to_csv(mat_path, index=False, header=False, encoding="utf-8-sig")
    save_xyz_map(arr, xyz_path)
    return mat_path, xyz_path

# =============================================================================
# Loading analysis outputs
# =============================================================================


def _load_array_key(arrays: np.lib.npyio.NpzFile, keys: Sequence[str]) -> np.ndarray:
    for k in keys:
        if k in arrays.files:
            return np.asarray(arrays[k])
    raise KeyError(f"None of these array keys were found: {keys}. Available: {arrays.files}")


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_required_outputs(out_dir: Path) -> Dict[str, object]:
    out_dir = Path(out_dir)
    arrays_path = out_dir / "motif_analysis_arrays.npz"
    coord_path = out_dir / "orthogonal_motif_coordinates_site_currents.csv"
    site_path = out_dir / "per_image_BAH_site_positions.csv"

    missing = [str(p) for p in [arrays_path, coord_path, site_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required analysis result files:\n" + "\n".join(missing))

    arrays = np.load(arrays_path, allow_pickle=True)
    maps = _load_array_key(arrays, ["aligned_norm", "aligned_maps", "folded_norm", "unit_maps", "maps"])
    names = _load_array_key(arrays, ["names", "image_names", "short_names"]).astype(str)
    maps = np.asarray(maps, dtype=float)

    coord_df = pd.read_csv(coord_path)
    site_df = pd.read_csv(site_path)
    if "role" in site_df.columns and "site" not in site_df.columns:
        site_df = site_df.rename(columns={"role": "site"})

    required_site = {"name", "site", "u", "v"}
    if not required_site.issubset(site_df.columns):
        raise ValueError(f"{site_path} must contain columns {required_site}. Found {site_df.columns.tolist()}")
    if "name" not in coord_df.columns:
        raise ValueError(f"{coord_path} must contain a name column.")
    if maps.shape[0] != len(names):
        raise ValueError(f"maps count {maps.shape[0]} and names count {len(names)} differ.")
    if maps.ndim != 3 or maps.shape[1] != maps.shape[2]:
        raise ValueError(f"maps must have shape (N, n, n). Found {maps.shape}")

    return {"maps": maps, "names": names, "coord_df": coord_df, "site_df": site_df}


def coord_rows_for_names(coord_df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    df = coord_df.copy()
    df["name"] = df["name"].astype(str)
    by_name = df.set_index("name", drop=False)
    rows = []
    for nm in map(str, names):
        if nm in by_name.index:
            row = by_name.loc[nm]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append(row)
            continue
        hit = df[df["name"].apply(lambda x: (x in nm) or (nm in x))]
        if hit.empty:
            raise KeyError(f"Could not match image name {nm} in coord_df.")
        rows.append(hit.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def get_theta(coord_rows: pd.DataFrame) -> np.ndarray:
    col = _find_col(coord_rows, ["theta_beta_over_AH_deg_norm", "theta_deg", "theta"])
    if col is None:
        raise ValueError("Could not find theta column in orthogonal_motif_coordinates_site_currents.csv")
    return pd.to_numeric(coord_rows[col], errors="coerce").to_numpy(float)


def get_site_intensity_columns(coord_rows: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    b = _find_col(coord_rows, ["I_B_norm", "B_norm", "I_B", "B"])
    a = _find_col(coord_rows, ["I_A_norm", "A_norm", "I_A", "A"])
    h = _find_col(coord_rows, ["I_H_norm", "H_norm", "I_H", "H"])
    return b, a, h


def get_site_intensities(coord_rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bcol, acol, hcol = get_site_intensity_columns(coord_rows)
    if bcol and acol and hcol:
        IB = pd.to_numeric(coord_rows[bcol], errors="coerce").to_numpy(float)
        IA = pd.to_numeric(coord_rows[acol], errors="coerce").to_numpy(float)
        IH = pd.to_numeric(coord_rows[hcol], errors="coerce").to_numpy(float)
        mean = np.nanmean(np.vstack([IB, IA, IH]), axis=0)
        return IB - mean, IA - mean, IH - mean

    ccol = _find_col(coord_rows, ["c_AH_norm", "c_AH"])
    betacol = _find_col(coord_rows, ["abs_c_beta_norm", "abs_c_beta", "c_beta_abs"])
    if ccol is None or betacol is None:
        raise ValueError("Could not find site intensities or c_AH/abs_c_beta columns.")
    cAH = pd.to_numeric(coord_rows[ccol], errors="coerce").to_numpy(float)
    cb = pd.to_numeric(coord_rows[betacol], errors="coerce").to_numpy(float)
    IH = -(SQRT6 / 3.0) * cAH
    IA = (SQRT6 / 6.0) * cAH - (SQRT2 / 2.0) * cb
    IB = (SQRT6 / 6.0) * cAH + (SQRT2 / 2.0) * cb
    return IB, IA, IH


def site_positions_for_names(site_df: pd.DataFrame, names: Sequence[str]) -> Dict[str, np.ndarray]:
    sdf = site_df.copy()
    sdf["name"] = sdf["name"].astype(str)
    sdf["site"] = sdf["site"].astype(str).str.upper()
    out: Dict[str, List[Tuple[float, float]]] = {"B": [], "A": [], "H": []}
    for nm in map(str, names):
        rows_nm = sdf[sdf["name"] == nm]
        if rows_nm.empty:
            rows_nm = sdf[sdf["name"].apply(lambda x: (x in nm) or (nm in x))]
        for role in ["B", "A", "H"]:
            r = rows_nm[rows_nm["site"] == role]
            if r.empty:
                raise KeyError(f"Missing {role} site position for {nm}")
            out[role].append((float(r.iloc[0]["u"]) % 1.0, float(r.iloc[0]["v"]) % 1.0))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}

# =============================================================================
# Site masks for masked agreement
# =============================================================================


def _periodic_distance2_uv_grid(U: np.ndarray, V: np.ndarray, u0: float, v0: float) -> np.ndarray:
    best = None
    for su in (-1, 0, 1):
        for sv in (-1, 0, 1):
            du = U - (float(u0) + su)
            dv = V - (float(v0) + sv)
            x, y = uv_to_graphite60(du, dv)
            d2 = x * x + y * y
            best = d2 if best is None else np.minimum(best, d2)
    return best


def site_weight_map_for_index(n: int, positions: Dict[str, np.ndarray], idx: int, sigma: float = SITE_MASK_SIGMA_FRAC) -> np.ndarray:
    U, V, _, _ = unit_cell_centers(n)
    w = np.zeros((n, n), dtype=float)
    for role in ["B", "A", "H"]:
        u0, v0 = positions[role][int(idx)]
        d2 = _periodic_distance2_uv_grid(U, V, u0, v0)
        w = np.maximum(w, np.exp(-0.5 * d2 / (float(sigma) ** 2)))
    return w


def evaluation_mask_outside_sites(n: int, positions: Dict[str, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    if not RUN_MASKED_AGREEMENT:
        return np.ones((n, n), dtype=bool)
    acc = np.zeros((n, n), dtype=float)
    idxs = list(map(int, indices))
    if len(idxs) == 0:
        return np.ones((n, n), dtype=bool)
    for j in idxs:
        acc += site_weight_map_for_index(n, positions, j)
    acc /= float(len(idxs))
    # True means evaluate. Exclude pixels repeatedly close to any B/A/H site.
    return acc < float(SITE_MASK_EXCLUDE_WEIGHT)

# =============================================================================
# Fourier-domain kernel learning and rendering
# =============================================================================


def fft_freq_mesh(n: int) -> Tuple[np.ndarray, np.ndarray]:
    freq = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(freq, freq)
    return KX, KY


def structure_factor_fft(n: int, positions: Dict[str, Tuple[float, float]], intensities: Dict[str, float]) -> np.ndarray:
    KX, KY = fft_freq_mesh(n)
    S = np.zeros((n, n), dtype=np.complex128)
    for role in ["B", "A", "H"]:
        u, v = positions[role]
        I = float(intensities[role])
        S += I * np.exp(-2j * np.pi * (KX * float(u) + KY * float(v)))
    return S


def estimate_kernel_fft(
    maps: np.ndarray,
    positions: Dict[str, np.ndarray],
    IB: np.ndarray,
    IA: np.ndarray,
    IH: np.ndarray,
    fit_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    maps = np.asarray(maps, dtype=float)
    n = maps.shape[1]
    numerator = np.zeros((n, n), dtype=np.complex128)
    denom = np.zeros((n, n), dtype=np.float64)
    rows = []

    for j in map(int, fit_indices):
        if not (np.isfinite(IB[j]) and np.isfinite(IA[j]) and np.isfinite(IH[j])):
            continue
        m = zero_mean(maps[j])
        if not np.all(np.isfinite(m)):
            continue
        pos_j = {role: tuple(positions[role][j]) for role in ["B", "A", "H"]}
        int_j = {"B": float(IB[j]), "A": float(IA[j]), "H": float(IH[j])}
        S = structure_factor_fft(n, pos_j, int_j)
        M = np.fft.fft2(m)
        numerator += np.conj(S) * M
        denom += np.abs(S) ** 2
        rows.append({"array_index": int(j), "IB": IB[j], "IA": IA[j], "IH": IH[j]})

    positive = denom[denom > 0]
    if positive.size == 0:
        raise RuntimeError("No valid train images available for kernel estimation.")
    reg = KERNEL_TIKHONOV_RELATIVE * float(np.nanmedian(positive))
    K_fft = numerator / (denom + reg)
    K_fft[0, 0] = 0.0

    if APPLY_KERNEL_LOW_PASS and KERNEL_LOW_PASS_SIGMA_CYCLES is not None and KERNEL_LOW_PASS_SIGMA_CYCLES > 0:
        KX, KY = fft_freq_mesh(n)
        filt = np.exp(-0.5 * (KX ** 2 + KY ** 2) / float(KERNEL_LOW_PASS_SIGMA_CYCLES) ** 2)
        filt[0, 0] = 1.0
        K_fft *= filt

    kernel = zero_mean(np.fft.ifft2(K_fft).real)
    diag = pd.DataFrame(rows)
    diag["n_kernel_fit_images"] = len(rows)
    diag["kernel_tikhonov_reg"] = reg
    diag["kernel_tikhonov_relative"] = KERNEL_TIKHONOV_RELATIVE
    diag["apply_kernel_low_pass"] = APPLY_KERNEL_LOW_PASS
    diag["kernel_low_pass_sigma_cycles"] = KERNEL_LOW_PASS_SIGMA_CYCLES if APPLY_KERNEL_LOW_PASS else np.nan
    return K_fft, kernel, diag


def render_from_kernel_fft(n: int, K_fft: np.ndarray, positions_j: Dict[str, Tuple[float, float]], intensities_j: Dict[str, float]) -> np.ndarray:
    S = structure_factor_fft(n, positions_j, intensities_j)
    return zero_mean(np.fft.ifft2(K_fft * S).real)


def predict_maps_for_indices(
    maps: np.ndarray,
    K_fft: np.ndarray,
    positions: Dict[str, np.ndarray],
    IB: np.ndarray,
    IA: np.ndarray,
    IH: np.ndarray,
    indices: Sequence[int],
    scale: float = 1.0,
) -> np.ndarray:
    n = maps.shape[1]
    preds = []
    for j in map(int, indices):
        pos_j = {role: tuple(positions[role][j]) for role in ["B", "A", "H"]}
        int_j = {"B": float(IB[j]), "A": float(IA[j]), "H": float(IH[j])}
        preds.append(scale * render_from_kernel_fft(n, K_fft, pos_j, int_j))
    return np.stack(preds, axis=0)


def fit_train_scale(maps: np.ndarray, preds_train: np.ndarray, train_idx: Sequence[int]) -> float:
    num = 0.0
    den = 0.0
    for pred, j in zip(preds_train, map(int, train_idx)):
        m = zero_mean(maps[j])
        p = zero_mean(pred)
        mask = np.isfinite(m) & np.isfinite(p)
        num += float(np.nansum(m[mask] * p[mask]))
        den += float(np.nansum(p[mask] * p[mask]))
    return float(num / den) if den > 0 else 1.0

# =============================================================================
# Quantile windows and splits
# =============================================================================


def quantile_windows(indices: Sequence[int], theta: np.ndarray, fraction: float, min_n: int, max_n: int) -> Dict[int, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    valid = indices[np.isfinite(theta[indices])]
    order = valid[np.argsort(theta[valid])]
    n = order.size
    if n == 0:
        raise RuntimeError("No finite theta values in supplied indices.")
    window_n = int(round(fraction * n))
    window_n = max(int(min_n), min(int(max_n), window_n, n))
    windows = {}
    for q in QUANTILE_PERCENTILES:
        center = int(round((q / 100.0) * (n - 1)))
        lo = max(0, center - window_n // 2)
        hi = min(n, lo + window_n)
        lo = max(0, hi - window_n)
        windows[int(q)] = order[lo:hi]
    return windows


def stratified_train_test_split(theta: np.ndarray, test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    valid = np.where(np.isfinite(theta))[0]
    if valid.size < 10:
        raise RuntimeError("Too few finite theta values for train/test split.")

    if not STRATIFY_BY_THETA:
        shuffled = rng.permutation(valid)
        n_test = max(1, int(round(test_fraction * valid.size)))
        test_idx = np.sort(shuffled[:n_test])
        train_idx = np.sort(shuffled[n_test:])
        rows = [{"array_index": int(i), "theta_deg": theta[i], "theta_stratum": 0, "split": "test" if i in set(test_idx) else "train"} for i in valid]
        return train_idx, test_idx, pd.DataFrame(rows)

    order = valid[np.argsort(theta[valid])]
    train_list: List[int] = []
    test_list: List[int] = []
    rows = []
    chunks = np.array_split(order, N_THETA_STRATA)
    for s, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=int)
        if chunk.size == 0:
            continue
        perm = rng.permutation(chunk)
        n_test_s = max(1, int(round(test_fraction * chunk.size)))
        if chunk.size - n_test_s < 1:
            n_test_s = max(1, chunk.size - 1)
        test_s = set(map(int, perm[:n_test_s]))
        for i in chunk:
            split = "test" if int(i) in test_s else "train"
            rows.append({"array_index": int(i), "theta_deg": float(theta[i]), "theta_stratum": int(s), "split": split})
            (test_list if split == "test" else train_list).append(int(i))
    return np.array(sorted(train_list), dtype=int), np.array(sorted(test_list), dtype=int), pd.DataFrame(rows)

# =============================================================================
# Metrics for averaged maps
# =============================================================================


def map_metrics(measured: np.ndarray, predicted: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    return {
        "agreement": agreement(measured, predicted, mask=mask),
        "relative_rmse": relative_rmse(measured, predicted, mask=mask),
        "correlation": normalized_corr(measured, predicted, mask=mask),
    }


def save_quantile_maps(measured: np.ndarray, predicted: np.ndarray, out_dir: Path, prefix: str) -> Dict[str, str]:
    residual = zero_mean(measured - predicted)
    m_mat, m_xyz = save_matrix_and_xyz(measured, out_dir, f"measured_{prefix}")
    p_mat, p_xyz = save_matrix_and_xyz(predicted, out_dir, f"predicted_{prefix}")
    r_mat, r_xyz = save_matrix_and_xyz(residual, out_dir, f"residual_{prefix}")
    relative = lambda path: path.relative_to(out_dir.parent).as_posix()
    return {
        "measured_matrix_path": relative(m_mat),
        "predicted_matrix_path": relative(p_mat),
        "residual_matrix_path": relative(r_mat),
        "measured_graphite60_xyz": relative(m_xyz),
        "predicted_graphite60_xyz": relative(p_xyz),
        "residual_graphite60_xyz": relative(r_xyz),
    }


def plot_quantile_figure(summary_df: pd.DataFrame, out_dir: Path, filename: str, title: str) -> None:
    if summary_df.empty:
        return
    ncols = len(summary_df)
    fig, axes = plt.subplots(3, ncols, figsize=(3.2 * ncols, 7.8), constrained_layout=True)
    if ncols == 1:
        axes = axes[:, None]
    all_maps = []
    for _, row in summary_df.iterrows():
        for key in ["measured_matrix_path", "predicted_matrix_path", "residual_matrix_path"]:
            all_maps.append(pd.read_csv(out_dir / row[key], header=None).to_numpy(float))
    vmax = float(np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in all_maps])), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    Xe, Ye = unit_cell_edges(all_maps[0].shape[0])

    for c, (_, row) in enumerate(summary_df.iterrows()):
        maps_row = [
            pd.read_csv(out_dir / row["measured_matrix_path"], header=None).to_numpy(float),
            pd.read_csv(out_dir / row["predicted_matrix_path"], header=None).to_numpy(float),
            pd.read_csv(out_dir / row["residual_matrix_path"], header=None).to_numpy(float),
        ]
        titles = [
            f"Measured p{int(row['percentile']):02d}\nθ={row['theta_mean_deg']:.1f}°, n={int(row['n_eval'])}",
            f"Predicted\nagree={row['agreement']:.3f}, r={row['correlation']:.3f}",
            f"Residual\nmasked agree={row.get('agreement_masked', np.nan):.3f}",
        ]
        for r in range(3):
            ax = axes[r, c]
            im = ax.pcolormesh(Xe, Ye, maps_row[r], shading="auto", cmap=CMAP, vmin=-vmax, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[r], fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_dir / filename, dpi=FIG_DPI)
    plt.close(fig)

# =============================================================================
# Leave-quantile-out validation
# =============================================================================


def run_leave_quantile_out(
    maps: np.ndarray,
    names: np.ndarray,
    theta: np.ndarray,
    IB: np.ndarray,
    IA: np.ndarray,
    IH: np.ndarray,
    positions: Dict[str, np.ndarray],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lqo_dir = out_dir / "leave_quantile_out"
    map_dir = lqo_dir / "lqo_quantile_map_data"
    kernel_dir = lqo_dir / "kernels"
    lqo_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    all_idx = np.where(np.isfinite(theta))[0]
    windows = quantile_windows(all_idx, theta, QUANTILE_WINDOW_FRACTION, QUANTILE_WINDOW_MIN_N, QUANTILE_WINDOW_MAX_N)
    n = maps.shape[1]
    rng = np.random.default_rng(NULL_RANDOM_SEED)
    q_rows = []
    null_rows = []
    global_mean_int = {
        "B": float(np.nanmean(IB[all_idx])),
        "A": float(np.nanmean(IA[all_idx])),
        "H": float(np.nanmean(IH[all_idx])),
    }

    for q, eval_idx in windows.items():
        eval_idx = np.asarray(eval_idx, dtype=int)
        train_idx = np.array(sorted(set(map(int, all_idx)) - set(map(int, eval_idx))), dtype=int)
        K_fft, kernel, diag = estimate_kernel_fft(maps, positions, IB, IA, IH, train_idx)
        diag.to_csv(kernel_dir / f"kernel_fit_diag_excluding_p{q:02d}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(kernel).to_csv(kernel_dir / f"kernel_excluding_p{q:02d}.csv", index=False, header=False, encoding="utf-8-sig")
        save_xyz_map(kernel, kernel_dir / f"kernel_excluding_p{q:02d}_graphite60_xyz.csv")

        preds_train_unscaled = predict_maps_for_indices(maps, K_fft, positions, IB, IA, IH, train_idx, scale=1.0)
        scale = fit_train_scale(maps, preds_train_unscaled, train_idx) if FIT_SINGLE_SCALE_ON_TRAIN else 1.0
        preds_eval = predict_maps_for_indices(maps, K_fft, positions, IB, IA, IH, eval_idx, scale=scale)
        measured_q = zero_mean(np.nanmean(maps[eval_idx], axis=0))
        predicted_q = zero_mean(np.nanmean(preds_eval, axis=0))
        eval_mask = evaluation_mask_outside_sites(n, positions, eval_idx)

        metrics = map_metrics(measured_q, predicted_q)
        metrics_masked = map_metrics(measured_q, predicted_q, mask=eval_mask)
        paths = save_quantile_maps(measured_q, predicted_q, map_dir, f"lqo_p{q:02d}")
        pd.DataFrame(eval_mask.astype(int)).to_csv(map_dir / f"eval_mask_outside_sites_p{q:02d}_matrix_uv.csv", index=False, header=False, encoding="utf-8-sig")
        save_xyz_map(eval_mask.astype(float), map_dir / f"eval_mask_outside_sites_p{q:02d}_graphite60_xyz.csv")

        q_rows.append({
            "validation": "leave_quantile_out",
            "percentile": int(q),
            "n_eval": int(len(eval_idx)),
            "n_train": int(len(train_idx)),
            "theta_mean_deg": float(np.nanmean(theta[eval_idx])),
            "theta_median_deg": float(np.nanmedian(theta[eval_idx])),
            "theta_min_deg": float(np.nanmin(theta[eval_idx])),
            "theta_max_deg": float(np.nanmax(theta[eval_idx])),
            "train_scale_factor": float(scale),
            "agreement": metrics["agreement"],
            "relative_rmse": metrics["relative_rmse"],
            "correlation": metrics["correlation"],
            "agreement_masked": metrics_masked["agreement"],
            "relative_rmse_masked": metrics_masked["relative_rmse"],
            "correlation_masked": metrics_masked["correlation"],
            "masked_evaluation_fraction": float(np.mean(eval_mask)),
            "eval_window_names": ";".join(str(names[j]) for j in eval_idx),
            **paths,
        })

        # Null 1: shuffle intensity triplets within the evaluation window.
        if RUN_INTENSITY_SHUFFLE_NULL:
            for rep in range(int(N_INTENSITY_SHUFFLES)):
                perm = rng.permutation(eval_idx)
                IB_sh = IB.copy(); IA_sh = IA.copy(); IH_sh = IH.copy()
                IB_sh[eval_idx] = IB[perm]
                IA_sh[eval_idx] = IA[perm]
                IH_sh[eval_idx] = IH[perm]
                pred_sh = predict_maps_for_indices(maps, K_fft, positions, IB_sh, IA_sh, IH_sh, eval_idx, scale=scale)
                pred_q = zero_mean(np.nanmean(pred_sh, axis=0))
                met = map_metrics(measured_q, pred_q)
                met_m = map_metrics(measured_q, pred_q, mask=eval_mask)
                null_rows.append({
                    "null_type": "site_intensity_shuffle_within_quantile",
                    "replicate": int(rep),
                    "percentile": int(q),
                    "agreement": met["agreement"],
                    "relative_rmse": met["relative_rmse"],
                    "correlation": met["correlation"],
                    "agreement_masked": met_m["agreement"],
                    "relative_rmse_masked": met_m["relative_rmse"],
                    "correlation_masked": met_m["correlation"],
                })

        # Null 2: role permutation. Keep positions fixed but assign intensities to wrong roles.
        if RUN_ROLE_PERMUTATION_NULL:
            role_arrays = {"B": IB, "A": IA, "H": IH}
            for perm in itertools.permutations(["B", "A", "H"], 3):
                if perm == ("B", "A", "H"):
                    continue
                # predicted role B receives original intensity of perm[0], etc.
                pred_perm = predict_maps_for_indices(
                    maps, K_fft, positions,
                    role_arrays[perm[0]], role_arrays[perm[1]], role_arrays[perm[2]],
                    eval_idx,
                    scale=scale,
                )
                pred_q = zero_mean(np.nanmean(pred_perm, axis=0))
                met = map_metrics(measured_q, pred_q)
                met_m = map_metrics(measured_q, pred_q, mask=eval_mask)
                null_rows.append({
                    "null_type": "role_permutation",
                    "replicate": -1,
                    "role_assignment_for_BAH": f"B<-{perm[0]};A<-{perm[1]};H<-{perm[2]}",
                    "percentile": int(q),
                    "agreement": met["agreement"],
                    "relative_rmse": met["relative_rmse"],
                    "correlation": met["correlation"],
                    "agreement_masked": met_m["agreement"],
                    "relative_rmse_masked": met_m["relative_rmse"],
                    "correlation_masked": met_m["correlation"],
                })

        # Null 3: same global mean intensity for all quantiles.
        if RUN_FIXED_GLOBAL_INTENSITY_NULL:
            # Create temporary intensity arrays where all eval images use global mean triplet.
            IB_fix = IB.copy(); IA_fix = IA.copy(); IH_fix = IH.copy()
            IB_fix[eval_idx] = global_mean_int["B"]
            IA_fix[eval_idx] = global_mean_int["A"]
            IH_fix[eval_idx] = global_mean_int["H"]
            pred_fix = predict_maps_for_indices(maps, K_fft, positions, IB_fix, IA_fix, IH_fix, eval_idx, scale=scale)
            pred_q = zero_mean(np.nanmean(pred_fix, axis=0))
            met = map_metrics(measured_q, pred_q)
            met_m = map_metrics(measured_q, pred_q, mask=eval_mask)
            null_rows.append({
                "null_type": "fixed_global_mean_site_intensity",
                "replicate": -1,
                "percentile": int(q),
                "agreement": met["agreement"],
                "relative_rmse": met["relative_rmse"],
                "correlation": met["correlation"],
                "agreement_masked": met_m["agreement"],
                "relative_rmse_masked": met_m["relative_rmse"],
                "correlation_masked": met_m["correlation"],
            })

    qdf = pd.DataFrame(q_rows)
    qdf.to_csv(lqo_dir / "leave_quantile_out_summary.csv", index=False, encoding="utf-8-sig")
    null_df = pd.DataFrame(null_rows)
    if not null_df.empty:
        null_df.to_csv(lqo_dir / "leave_quantile_out_null_metrics.csv", index=False, encoding="utf-8-sig")
        # Summarize nulls by percentile and type.
        summary_rows = []
        for q in QUANTILE_PERCENTILES:
            true = qdf[qdf["percentile"] == q].iloc[0]
            for nt, nr in null_df[null_df["percentile"] == q].groupby("null_type"):
                summary_rows.append({
                    "percentile": int(q),
                    "null_type": str(nt),
                    "true_agreement": float(true["agreement"]),
                    "null_agreement_mean": float(nr["agreement"].mean()),
                    "null_agreement_p05": float(nr["agreement"].quantile(0.05)),
                    "null_agreement_p50": float(nr["agreement"].quantile(0.50)),
                    "null_agreement_p95": float(nr["agreement"].quantile(0.95)),
                    "true_agreement_minus_null_mean": float(true["agreement"] - nr["agreement"].mean()),
                    "true_agreement_masked": float(true["agreement_masked"]),
                    "null_agreement_masked_mean": float(nr["agreement_masked"].mean()),
                    "null_agreement_masked_p05": float(nr["agreement_masked"].quantile(0.05)),
                    "null_agreement_masked_p95": float(nr["agreement_masked"].quantile(0.95)),
                    "true_agreement_masked_minus_null_mean": float(true["agreement_masked"] - nr["agreement_masked"].mean()),
                })
        pd.DataFrame(summary_rows).to_csv(lqo_dir / "leave_quantile_out_null_summary.csv", index=False, encoding="utf-8-sig")

    plot_quantile_figure(qdf, lqo_dir, "figure_leave_quantile_out_measured_vs_predicted.png", "Leave-quantile-out validation: measured vs held-out predicted quantile maps")
    plot_lqo_null_summary(lqo_dir)
    return qdf, null_df


def plot_lqo_null_summary(lqo_dir: Path) -> None:
    true_path = lqo_dir / "leave_quantile_out_summary.csv"
    null_path = lqo_dir / "leave_quantile_out_null_summary.csv"
    if not true_path.exists() or not null_path.exists():
        return
    qdf = pd.read_csv(true_path)
    ns = pd.read_csv(null_path)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(qdf))
    ax.plot(x, qdf["agreement"], marker="o", label="True LQO prediction")
    for nt, grp in ns.groupby("null_type"):
        grp = grp.set_index("percentile").loc[list(qdf["percentile"])].reset_index()
        ax.plot(x, grp["null_agreement_mean"], marker="s", linestyle="--", label=f"Null mean: {nt}")
        ax.fill_between(x, grp["null_agreement_p05"], grp["null_agreement_p95"], alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p{int(q):02d}" for q in qdf["percentile"]])
    ax.set_ylabel("Agreement")
    ax.set_xlabel("Theta quantile")
    ax.set_title("Leave-quantile-out agreement vs null controls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(lqo_dir / "figure_leave_quantile_out_agreement_vs_nulls.png", dpi=FIG_DPI)
    plt.close(fig)
    # Raw data for Origin.
    rows = []
    for _, r in qdf.iterrows():
        rows.append({"series": "true", "percentile": int(r["percentile"]), "agreement": float(r["agreement"]), "agreement_p05": np.nan, "agreement_p95": np.nan})
    for _, r in ns.iterrows():
        rows.append({"series": str(r["null_type"]), "percentile": int(r["percentile"]), "agreement": float(r["null_agreement_mean"]), "agreement_p05": float(r["null_agreement_p05"]), "agreement_p95": float(r["null_agreement_p95"])})
    pd.DataFrame(rows).to_csv(lqo_dir / "plot_data_leave_quantile_out_agreement_vs_nulls.csv", index=False, encoding="utf-8-sig")

# =============================================================================
# Repeated train/test validation
# =============================================================================


def run_one_train_test_split(
    maps: np.ndarray,
    names: np.ndarray,
    theta: np.ndarray,
    IB: np.ndarray,
    IA: np.ndarray,
    IH: np.ndarray,
    positions: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, float]:
    train_idx, test_idx, _ = stratified_train_test_split(theta, TEST_FRACTION, seed)
    K_fft, _, _ = estimate_kernel_fft(maps, positions, IB, IA, IH, train_idx)
    preds_train = predict_maps_for_indices(maps, K_fft, positions, IB, IA, IH, train_idx, scale=1.0)
    scale = fit_train_scale(maps, preds_train, train_idx) if FIT_SINGLE_SCALE_ON_TRAIN else 1.0
    preds_test = predict_maps_for_indices(maps, K_fft, positions, IB, IA, IH, test_idx, scale=scale)
    test_local = {int(j): i for i, j in enumerate(map(int, test_idx))}

    # Image-level metrics.
    image_ag = []
    image_corr = []
    image_ag_masked = []
    for li, j in enumerate(map(int, test_idx)):
        m = zero_mean(maps[j])
        p = zero_mean(preds_test[li])
        image_ag.append(agreement(m, p))
        image_corr.append(normalized_corr(m, p))
        mask = evaluation_mask_outside_sites(maps.shape[1], positions, [j])
        image_ag_masked.append(agreement(m, p, mask=mask))

    # Quantile averaged metrics over test images only.
    windows = quantile_windows(test_idx, theta, REPEATED_QUANTILE_WINDOW_FRACTION, REPEATED_QUANTILE_WINDOW_MIN_N, REPEATED_QUANTILE_WINDOW_MAX_N)
    q_ag = []
    q_corr = []
    q_ag_masked = []
    for _, idxs in windows.items():
        idxs = np.asarray(idxs, dtype=int)
        loc = [test_local[int(j)] for j in idxs]
        measured_q = zero_mean(np.nanmean(maps[idxs], axis=0))
        pred_q = zero_mean(np.nanmean(preds_test[loc], axis=0))
        mask = evaluation_mask_outside_sites(maps.shape[1], positions, idxs)
        q_ag.append(agreement(measured_q, pred_q))
        q_corr.append(normalized_corr(measured_q, pred_q))
        q_ag_masked.append(agreement(measured_q, pred_q, mask=mask))

    return {
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_scale": float(scale),
        "test_image_agreement_mean": float(np.nanmean(image_ag)),
        "test_image_agreement_median": float(np.nanmedian(image_ag)),
        "test_image_correlation_mean": float(np.nanmean(image_corr)),
        "test_image_agreement_masked_mean": float(np.nanmean(image_ag_masked)),
        "test_quantile_agreement_mean": float(np.nanmean(q_ag)),
        "test_quantile_agreement_median": float(np.nanmedian(q_ag)),
        "test_quantile_correlation_mean": float(np.nanmean(q_corr)),
        "test_quantile_agreement_masked_mean": float(np.nanmean(q_ag_masked)),
    }


def run_repeated_train_test(
    maps: np.ndarray,
    names: np.ndarray,
    theta: np.ndarray,
    IB: np.ndarray,
    IA: np.ndarray,
    IH: np.ndarray,
    positions: Dict[str, np.ndarray],
    out_dir: Path,
) -> pd.DataFrame:
    rep_dir = out_dir / "repeated_train_test"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(int(N_REPEATED_SPLITS)):
        seed = int(REPEATED_SPLIT_SEED0 + rep)
        rows.append(run_one_train_test_split(maps, names, theta, IB, IA, IH, positions, seed))
    df = pd.DataFrame(rows)
    df.to_csv(rep_dir / "repeated_train_test_split_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = [c for c in df.columns if c not in {"seed", "n_train", "n_test"}]
    summary_rows = []
    for c in metrics:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        summary_rows.append({
            "metric": c,
            "n_splits": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
            "ci95_low_percentile": float(vals.quantile(0.025)) if vals.size > 0 else np.nan,
            "ci95_high_percentile": float(vals.quantile(0.975)) if vals.size > 0 else np.nan,
        })
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(rep_dir / "repeated_train_test_aggregate_summary.csv", index=False, encoding="utf-8-sig")

    # Plot raw distributions of split-level metrics.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(df["test_quantile_agreement_mean"].dropna(), bins=12, alpha=0.7, label="Test quantile agreement")
    ax.hist(df["test_image_agreement_mean"].dropna(), bins=12, alpha=0.5, label="Test image agreement")
    ax.set_xlabel("Agreement")
    ax.set_ylabel("Count")
    ax.set_title("Repeated stratified train/test validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rep_dir / "figure_repeated_train_test_agreement_distribution.png", dpi=FIG_DPI)
    plt.close(fig)
    pd.DataFrame({
        "test_quantile_agreement_mean": df["test_quantile_agreement_mean"],
        "test_image_agreement_mean": df["test_image_agreement_mean"],
        "test_quantile_agreement_masked_mean": df["test_quantile_agreement_masked_mean"],
        "test_image_agreement_masked_mean": df["test_image_agreement_masked_mean"],
    }).to_csv(rep_dir / "plot_data_repeated_train_test_agreement_raw_distribution.csv", index=False, encoding="utf-8-sig")
    return df

# =============================================================================
# Overall report
# =============================================================================


def write_overall_summary(out_dir: Path, lqo_df: pd.DataFrame, null_df: pd.DataFrame, rep_df: Optional[pd.DataFrame]) -> None:
    rows = []
    rows.append({"metric": "lqo_quantile_agreement_mean", "value": float(lqo_df["agreement"].mean())})
    rows.append({"metric": "lqo_quantile_agreement_median", "value": float(lqo_df["agreement"].median())})
    rows.append({"metric": "lqo_quantile_correlation_mean", "value": float(lqo_df["correlation"].mean())})
    rows.append({"metric": "lqo_quantile_agreement_masked_mean", "value": float(lqo_df["agreement_masked"].mean())})
    rows.append({"metric": "lqo_quantile_correlation_masked_mean", "value": float(lqo_df["correlation_masked"].mean())})

    if not null_df.empty:
        true_by_q = lqo_df.set_index("percentile")
        for nt, nr in null_df.groupby("null_type"):
            diffs = []
            diffs_m = []
            for q, g in nr.groupby("percentile"):
                if q in true_by_q.index:
                    diffs.append(float(true_by_q.loc[q, "agreement"] - g["agreement"].mean()))
                    diffs_m.append(float(true_by_q.loc[q, "agreement_masked"] - g["agreement_masked"].mean()))
            rows.append({"metric": f"true_minus_{nt}_agreement_mean", "value": float(np.nanmean(diffs)) if diffs else np.nan})
            rows.append({"metric": f"true_minus_{nt}_agreement_masked_mean", "value": float(np.nanmean(diffs_m)) if diffs_m else np.nan})

    if rep_df is not None and not rep_df.empty:
        for c in ["test_quantile_agreement_mean", "test_image_agreement_mean", "test_quantile_agreement_masked_mean", "test_image_agreement_masked_mean"]:
            rows.append({"metric": f"repeated_{c}_mean", "value": float(rep_df[c].mean())})
            rows.append({"metric": f"repeated_{c}_p025", "value": float(rep_df[c].quantile(0.025))})
            rows.append({"metric": f"repeated_{c}_p975", "value": float(rep_df[c].quantile(0.975))})

    odf = pd.DataFrame(rows)
    odf.to_csv(out_dir / "validation_overall_summary.csv", index=False, encoding="utf-8-sig")

    def val(metric: str) -> str:
        hit = odf[odf["metric"] == metric]
        return "n/a" if hit.empty else f"{float(hit.iloc[0]['value']):.4f}"

    md = []
    md.append("# Physics forward-model validation summary\n")
    md.append("## Primary circularity control: leave-quantile-out\n")
    md.append(f"- Mean LQO quantile agreement: **{val('lqo_quantile_agreement_mean')}**")
    md.append(f"- Mean LQO quantile correlation: **{val('lqo_quantile_correlation_mean')}**")
    md.append(f"- Mean masked LQO agreement outside B/A/H site masks: **{val('lqo_quantile_agreement_masked_mean')}**")
    md.append("\n## Null-control improvement\n")
    for nt in ["site_intensity_shuffle_within_quantile", "role_permutation", "fixed_global_mean_site_intensity"]:
        md.append(f"- True minus {nt} agreement mean: **{val(f'true_minus_{nt}_agreement_mean')}**")
        md.append(f"- True minus {nt} masked agreement mean: **{val(f'true_minus_{nt}_agreement_masked_mean')}**")
    if rep_df is not None and not rep_df.empty:
        md.append("\n## Repeated train/test generalization\n")
        md.append(f"- Repeated test quantile agreement mean: **{val('repeated_test_quantile_agreement_mean_mean')}**")
        md.append(f"- Repeated test quantile agreement 95% interval: **{val('repeated_test_quantile_agreement_mean_p025')}–{val('repeated_test_quantile_agreement_mean_p975')}**")
        md.append(f"- Repeated test image agreement mean: **{val('repeated_test_image_agreement_mean_mean')}**")
        md.append(f"- Repeated masked test quantile agreement mean: **{val('repeated_test_quantile_agreement_masked_mean_mean')}**")
    md.append("\n## Interpretation guide\n")
    md.append("- Leave-quantile-out is the main defense against circularity: the full maps in the predicted quantile are excluded from kernel learning.")
    md.append("- Masked agreement tests whether prediction works outside the B/A/H site masks used for intensity extraction.")
    md.append("- Shuffle and role-permutation nulls test whether the correct B/A/H intensity/role information is required.")
    md.append("- Repeated train/test evaluates generalization to unseen images and split robustness.")
    (out_dir / "validation_overall_summary.md").write_text("\n".join(md), encoding="utf-8")

# =============================================================================
# Main pipeline
# =============================================================================


def run_physics_forward_validation(analysis_out_dir: Path = ANALYSIS_OUT_DIR) -> None:
    analysis_out_dir = Path(analysis_out_dir)
    out_dir = analysis_out_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_required_outputs(analysis_out_dir)
    maps: np.ndarray = loaded["maps"]  # type: ignore
    names: np.ndarray = loaded["names"]  # type: ignore
    coord_df: pd.DataFrame = loaded["coord_df"]  # type: ignore
    site_df: pd.DataFrame = loaded["site_df"]  # type: ignore

    coord_rows = coord_rows_for_names(coord_df, names)
    theta = get_theta(coord_rows)
    IB, IA, IH = get_site_intensities(coord_rows)
    positions = site_positions_for_names(site_df, names)

    # Store canonical input table for traceability.
    input_rows = pd.DataFrame({
        "array_index": np.arange(len(names), dtype=int),
        "name": names.astype(str),
        "theta_deg": theta,
        "I_B_zero_mean": IB,
        "I_A_zero_mean": IA,
        "I_H_zero_mean": IH,
        "B_u": positions["B"][:, 0], "B_v": positions["B"][:, 1],
        "A_u": positions["A"][:, 0], "A_v": positions["A"][:, 1],
        "H_u": positions["H"][:, 0], "H_v": positions["H"][:, 1],
    })
    input_rows.to_csv(out_dir / "validation_input_image_table.csv", index=False, encoding="utf-8-sig")

    lqo_df, null_df = run_leave_quantile_out(maps, names, theta, IB, IA, IH, positions, out_dir)
    rep_df = run_repeated_train_test(maps, names, theta, IB, IA, IH, positions, out_dir) if RUN_REPEATED_TRAIN_TEST else None
    write_overall_summary(out_dir, lqo_df, null_df, rep_df)

    print("\n=== Physics forward-model validation DONE ===")
    print(f"analysis folder: {analysis_out_dir}")
    print(f"output folder:   {out_dir}")
    print("Key outputs:")
    print("  - validation_overall_summary.md")
    print("  - validation_overall_summary.csv")
    print("  - leave_quantile_out/leave_quantile_out_summary.csv")
    print("  - leave_quantile_out/leave_quantile_out_null_summary.csv")
    print("  - leave_quantile_out/figure_leave_quantile_out_measured_vs_predicted.png")
    if RUN_REPEATED_TRAIN_TEST:
        print("  - repeated_train_test/repeated_train_test_aggregate_summary.csv")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--analysis-dir", required=True, help="Primary-analysis output directory")
    args = parser.parse_args(argv)
    run_physics_forward_validation(Path(args.analysis_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
