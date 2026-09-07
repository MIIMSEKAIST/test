# -*- coding: utf-8 -*-
"""Primary analysis for atomic-resolution graphite C-AFM maps.

Read one raw current map per file, subtract a fitted background plane, and
fold the registered lattice onto a 64 x 64 grid. A/B/H positions are selected
once on the phase-aligned ensemble mean and fixed across acquisitions.
Current-floor statistics use the uncorrected raw map."""

from __future__ import annotations

import argparse

import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import gaussian_kde, spearmanr, kruskal

# =============================================================================
# USER SETTINGS
# =============================================================================

path = "."
INPUT_ROOT_FOR_PROVENANCE = Path(path).resolve()

# Search settings
SEARCH_RECURSIVELY = True   # robust default: also find files in nested acquisition folders
INPUT_SUFFIXES = {".xlsx", ".xlsm", ".csv"}
OUTPUT_FOLDER_NAME = "analysis_results"

# Physical/scan settings. Change if your images are not 2.5 x 2.5 nm^2.
SCAN_SIZE_X_NM = 2.5
SCAN_SIZE_Y_NM = 2.5
GRAPHITE_LATTICE_A_NM = 0.246  # graphite/graphene in-plane lattice constant
USE_GRAPHITE_LATTICE_CONSTRAINT = True
LATTICE_RECIPROCAL_MAG_TOL = 0.45  # fractional tolerance around expected FFT radius

# Unit-cell folding / motif settings
UNIT_CELL_GRID = 64
SITE_AVERAGE_SIGMA_FRAC = 0.055  # Gaussian site-average width in fractional u/v units
ORIGIN_GRID_SEARCH_N = 32        # 32 -> origin step 1/32
ALIGN_UNIT_CELLS = True
ALIGN_ITERATIONS = 3

# Display geometry (analysis uses fractional coordinates).
DISPLAY_UNIT_CELL_MAPS_AS_GRAPHITE_PARALLELOGRAM = True
GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG = 120.0

# A/B/H registry: calibrated once, then fixed across acquisitions.
SITE_COORDINATE_MODE = "ensemble_mean_fixed"

# Optional FFT-phase diagnostics (not used in the default analysis).
FFT_PHASE_ANCHOR_FOLDING = False
FFT_PHASE_OFFSET_SIGN = 1.0
FFT_PHASE_PREPROCESS_SIGMA_FRAC = 16.0

# Fixed-B-vertex diagnostic mode.
FFT_FIXED_SITE_BASIS_NAME = "B_vertex_diag_13_13"
FFT_FIXED_ROLE_ASSIGNMENT = {"B_key": "B", "A_key": "A", "H_key": "H"}
DISABLE_CIRCULAR_ALIGNMENT_FOR_FFT_FIXED_SITES = True

# Bases for orientation-family diagnostics.
FFT_ORIENTATION_60_BASIS_NAME = "graphite_hollow_13_13__23_23"
FFT_ORIENTATION_120_BASIS_NAME = "graphite_hollow_13_23__23_13"
FFT_ORIENTATION_ANGLE_THRESHOLD_DEG = 90.0

# Manual-coordinate diagnostic defaults, in fractional u/v coordinates.
MANUAL_ABH_SITES_BY_ORIENTATION = {
    "fft60": {
        "basis": "manual_fft60_diag_default",
        "B": (0.0, 0.0),
        "H": (1.0 / 3.0, 1.0 / 3.0),
        "A": (2.0 / 3.0, 2.0 / 3.0),
    },
    "fft120": {
        "basis": "manual_fft120_cross_default",
        "B": (0.0, 0.0),
        "H": (1.0 / 3.0, 2.0 / 3.0),
        "A": (2.0 / 3.0, 1.0 / 3.0),
    },
}

# Optional per-image registration diagnostics.
INTENSITY_PER_IMAGE_SITE_TOP_N = 25
INTENSITY_PER_IMAGE_SAVE_ALL_CANDIDATES = False

# GPU / vectorized acceleration for the expensive A/B/H site search.
# Requires CuPy + a CUDA-capable NVIDIA GPU.  If CuPy/CUDA is unavailable,
# the same vectorized code automatically falls back to NumPy CPU.
USE_GPU_ACCELERATION_FOR_SITE_SEARCH = False
GPU_SITE_SEARCH_BATCH_CANDIDATES = 512
GPU_SITE_SEARCH_DTYPE = "float32"  # "float32" is much faster and enough for site-search scoring
GPU_SITE_SEARCH_VERBOSE = True

# Optional score-weighted site-picking diagnostics.
USE_SITE_PICKING_UNCERTAINTY_CORRECTION = False
SITE_PRIMARY_POLICY = "hard_all_candidates"
USE_SOFT_SITE_COORDINATES_AS_PRIMARY = SITE_PRIMARY_POLICY == "soft_all_candidates"
SITE_SOFT_TOP_N = 25
SITE_SOFT_SCORE_TEMPERATURE = 0.10
SITE_SOFT_MIN_REL_WEIGHT = 1.0e-4
SITE_AMBIGUITY_NEFF_CAUTION = 3.0
SITE_AMBIGUITY_THETA_IQR_CAUTION_DEG = 15.0
SITE_SCORE_GAP_CAUTION = 0.05

# Site-search score: total B-H contrast plus ordering penalties.
INTENSITY_SITE_SCORE_MODE = "theta_neutral_total_contrast"
SITE_SCORE_SATURATION_SCALE = 0.75
SAVE_LEGACY_ABH_BALANCE_SCORE_FOR_DIAGNOSTIC = True

# Local A-site refinement is disabled for the fixed ensemble registry.
REFINE_A_SITE_LOCAL_AFTER_PICK = False
A_REFINEMENT_RADIUS_FRAC = 0.045        # about 3 pixels on a 64x64 unit-cell grid
A_REFINEMENT_GRID_STEPS = 11            # odd number recommended
A_REFINEMENT_SIGMA_SHARP_FRAC = 0.032   # sharper than SITE_AVERAGE_SIGMA_FRAC
A_REFINEMENT_SIGMA_BROAD_FRAC = 0.090   # local background estimate for peakness
A_REFINEMENT_PRIOR_WEIGHT = 0.020       # discourages unnecessary A motion
A_REFINEMENT_ORDER_PENALTY_WEIGHT = 8.0 # reject A choices outside B>A>H
A_REFINEMENT_INTENSITY_WEIGHT = 0.15    # small; main term is local peakness
A_REFINEMENT_MIN_SCORE_IMPROVEMENT = 1.0e-4
A_REFINEMENT_MAX_THETA_CHANGE_DEG = 25.0 # keep larger shifts only as diagnostics
A_REFINEMENT_ACCEPT_IF_ORDER_IMPROVES = True

# Diagnostics for detecting artificial theta pile-up near ABH prototype.
THETA_PROTOTYPE_DEG = 30.0
THETA_PROTOTYPE_WINDOWS_DEG = (1.0, 2.0, 5.0)

# Orientation overrides for manual diagnostic mode.
MANUAL_FFT_ORIENTATION_OVERRIDES = {
    # "example_file_name_substring": "fft60",
}
RUN_INTENSITY_SITE_GRID_SEARCH_DIAGNOSTIC = True

# Two atomic candidates and one geometric hollow per cell.
USE_BAH_ROLE_CONVENTION = True
ROLE_ASSIGNMENT_REFERENCE = "normalized"  # "normalized" or "raw"
SITE_GEOMETRY_MODEL = "graphite_single_hollow"
SITE_MASK_DISTANCE_METRIC = "hex120"       # "hex120" or "rect"; hex120 uses du^2+dv^2-du*dv

# Numeric input conversion to amperes; set by --current-unit.
NUMERIC_INPUT_MULTIPLIER = 1.0
PARSE_TEXT_UNITS = True
MIN_IMAGE_HEIGHT = 16
MIN_IMAGE_WIDTH = 16

RAW_SHEET = 0  # zero-based index or exact sheet name; other sheets are ignored
BACKGROUND = "plane"  # "plane" or "none"

# Current-floor analysis settings.
# The primary current floor is a robust lower-percentile baseline from raw map.
CURRENT_FLOOR_PERCENTILE = 5.0
CURRENT_FLOOR_BIN_COUNT = 3  # low/mid/high floor bins
CURRENT_FLOOR_CORRELATION_BOOTSTRAP_N = 500
CURRENT_FLOOR_BOOTSTRAP_RANDOM_SEED = 19
CURRENT_FLOOR_PRIMARY_METRIC = "raw_current_floor_log10_abs_p05"
CURRENT_FLOOR_BOOTSTRAP_PRIMARY_ONLY = True
CURRENT_FLOOR_EFFECT_SIZE_CAUTION_ABS_RHO = 0.35
CURRENT_FLOOR_EFFECT_SIZE_STRONG_ABS_RHO = 0.50
SAFE_LOG10_EPS = 1e-300

# Compare normalized, background-corrected, and raw amplitudes.
ENABLE_NORMALIZATION_FLOOR_CONTROLS = True
NORMALIZATION_FLOOR_BOOTSTRAP_N = 500
NORMALIZATION_FLOOR_RANDOM_SEED = 2026

# Normalization. robust_z is primary; rank is computed as a robustness check.
PRIMARY_NORM = "robust_z"
COMPUTE_RANK_ROBUSTNESS = True
ROBUST_EPS = 1e-15

# Bootstrap/statistics settings
BOOTSTRAP_SITE_CI = True
BOOTSTRAP_N = 300
BOOTSTRAP_RANDOM_SEED = 7
GROUP_BOOTSTRAP_N = 2000

# Soft membership uses physical prototype angles and a common radial scale.
COMPUTE_SOFT_MEMBERSHIP = True
SOFT_PROTOTYPE_MODE = "physical_limits"
SOFT_SCALE_MODE = "common_radial_95pct"
SOFT_SIGMA = 0.45
SOFT_AMBIGUITY_MARGIN = 0.15

# Whole-map statistics
RUN_WHOLE_MAP_PCA_MDS = True
RUN_GMM_BIC_IF_SKLEARN_AVAILABLE = False

# Plot settings
LABEL_POINTS_IN_SCATTER = True
MAX_LABELLED_POINTS = 40
FIG_DPI = 220

# Plotting-table export.
EXPORT_ORIGIN_CSV_FOR_EACH_FIGURE = False  # legacy figure-object exporter; usually leave False
ORIGIN_PLOT_DATA_FOLDER_NAME = "origin_plot_data"

EXPORT_ORIGIN_SOURCE_DATA = True
ORIGIN_SOURCE_DATA_FOLDER_NAME = "origin_source_data"
ORIGIN_SOURCE_DATA_EXCEL_NAME = "origin_source_plot_data.xlsx"
ORIGIN_SOURCE_COMPILE_XLSX = True
ORIGIN_SOURCE_UNITCELL_MAX_LONG_ROWS = 1000000

# =============================================================================
# Validation / manuscript-number settings
# =============================================================================

# AFM slow-scan drift can shear/stretch the apparent lattice. Therefore the
# validation below is drift-aware: lattice angle is reported as a distortion
# diagnostic, but is not used as a hard pass/fail criterion by default.
DRIFT_AWARE_LATTICE_VALIDATION = True
LATTICE_LENGTH_MIN_NM = 0.16
LATTICE_LENGTH_MAX_NM = 0.36
RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION = 0.65
MIN_LATTICE_LENGTH_PASS_FRACTION = 0.70
MIN_RECIPROCAL_MAG_PASS_FRACTION = 0.70

# Unit-cell folding/alignment sanity checks.
MIN_UNITCELL_BIN_COVERAGE_FRACTION = 0.50
MIN_SITE_ORDERING_FRACTION = 0.80  # fraction with I_B > I_A > I_H
MIN_NORMALIZATION_ROBUSTNESS_CORR = 0.60

# Unit-cell capture validation.  These are drift-aware reliability checks, not
# crystallographic pass/fail tests.  The split-half test folds alternating
# real-space unit cells independently; high correlation means the apparent
# lattice coordinates produce reproducible unit-cell motifs across repeated
# cells in the same image.
MIN_SPLIT_HALF_COVERAGE_FRACTION = 0.35
MIN_SPLIT_HALF_CORR_MEDIAN = 0.25
MIN_SPLIT_HALF_CORR_STRONG = 0.50
MIN_QUANTILE_WINDOW_PAIRWISE_CORR = 0.20

# Motif interpretation thresholds. These are descriptive, not hard labels.
THETA_AH_DOMINANT_MAX_DEG = 20.0
THETA_BETA_DOMINANT_MIN_DEG = 40.0

# Quantile-ordered theta visualization/statistics for manuscript Fig. 3b/3c.
# The script reports these percentiles explicitly and makes both nearest-image
# representatives and local quantile-window averaged unit-cell maps.
THETA_QUANTILE_PERCENTILES = (5, 25, 50, 75, 95)
THETA_QUANTILE_AVERAGE_WINDOW_FRACTION = 0.05  # use about 5% of images around each quantile
THETA_QUANTILE_AVERAGE_MIN_N = 5
THETA_QUANTILE_AVERAGE_MAX_N = 31

# If a motif coordinate is strongly correlated with lattice distortion metrics,
# report a caution because scan distortion may be contributing to the descriptor.
ARTIFACT_SPEARMAN_CAUTION_ABS_RHO = 0.50
ARTIFACT_SPEARMAN_STRONG_ABS_RHO = 0.70

# =============================================================================
# Utility functions
# =============================================================================


def ensure_out_dir(input_path: Path) -> Path:
    if input_path.is_dir():
        base = input_path
    else:
        base = input_path.parent
    out = base / OUTPUT_FOLDER_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_input_files(input_path: Path) -> List[Path]:
    """List raw-map files in relative-path order."""
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")
    candidates = input_path.rglob("*") if SEARCH_RECURSIVELY else input_path.glob("*")
    output_path = (input_path / OUTPUT_FOLDER_NAME).resolve()
    return sorted(
        (f for f in candidates if f.is_file()
         and f.suffix.lower() in INPUT_SUFFIXES
         and not f.name.startswith("~$")
         and not f.resolve().is_relative_to(output_path)),
        key=lambda f: f.relative_to(input_path).as_posix().casefold(),
    )


def _provenance_path(file_path: Path) -> str:
    """Return an input-root-relative path without recording a workstation path."""
    resolved = Path(file_path).resolve()
    try:
        return resolved.relative_to(INPUT_ROOT_FOR_PROVENANCE).as_posix()
    except ValueError:
        return resolved.name


_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
    "−": "-", "–": "-", "—": "-",
})


def parse_numeric_cell(x) -> float:
    """Parse a current value in amperes; explicit units override numeric scaling."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        value = float(x) * NUMERIC_INPUT_MULTIPLIER
        return value if math.isfinite(value) else np.nan
    if not isinstance(x, str) or not x.strip():
        return np.nan

    text = x.strip().replace("−", "-").replace("×", "x").replace("⋅", "x")
    text = re.sub(r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
                  lambda m: "10^" + m.group(1).translate(_SUPERSCRIPT_MAP), text)
    # Allow thousands separators, but reject malformed comma-separated numbers.
    if "," in text:
        if not re.match(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s|[eEfpnumµμAa]|$)", text):
            return np.nan
        text = text.replace(",", "")

    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    power = rf"(?:{number}\s*[x*]\s*)?10\s*\^\s*[+-]?\d+"
    match = re.fullmatch(rf"(?P<value>{power}|{number})\s*(?P<unit>[fpnumµμ]?[Aa])?", text)
    if match is None:
        return np.nan
    unit = match.group("unit")
    if unit is not None and not PARSE_TEXT_UNITS:
        return np.nan
    factors = {"a": 1.0, "fa": 1e-15, "pa": 1e-12, "na": 1e-9,
               "ua": 1e-6, "µa": 1e-6, "μa": 1e-6, "ma": 1e-3}
    multiplier = factors[unit.lower()] if unit else NUMERIC_INPUT_MULTIPLIER
    value_text = match.group("value")
    try:
        if "^" in value_text:
            power_match = re.fullmatch(
                rf"(?:(?P<coefficient>{number})\s*[x*]\s*)?10\s*\^\s*(?P<exponent>[+-]?\d+)",
                value_text,
            )
            coefficient = float(power_match.group("coefficient") or "1")
            value = coefficient * 10.0 ** int(power_match.group("exponent"))
        else:
            value = float(value_text)
        value *= multiplier
        return value if math.isfinite(value) else np.nan
    except (ValueError, OverflowError):
        return np.nan


def robust_zscore(a: np.ndarray, eps: float = ROBUST_EPS) -> Tuple[np.ndarray, Dict[str, float]]:
    arr = np.asarray(a, dtype=float)
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    z = (arr - med) / scale
    return z, {"median": med, "mad": mad, "scale": scale}


def rank_normalize(a: np.ndarray) -> np.ndarray:
    """Rank normalize to approximately [-1, 1], preserving only relative motif order."""
    arr = np.asarray(a, dtype=float)
    flat = arr.ravel()
    valid = np.isfinite(flat)
    out = np.full_like(flat, np.nan, dtype=float)
    vals = flat[valid]
    if vals.size == 0:
        return np.zeros_like(arr)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(vals.size, dtype=float)
    if vals.size > 1:
        ranks = 2.0 * (ranks / (vals.size - 1.0)) - 1.0
    else:
        ranks[:] = 0.0
    out[valid] = ranks
    out[~valid] = np.nanmedian(ranks)
    return out.reshape(arr.shape)


def fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    a = np.array(arr, dtype=float, copy=True)
    if not np.isnan(a).any():
        return a
    if np.all(np.isnan(a)):
        return np.zeros_like(a)
    mask = np.isnan(a)
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    a[mask] = a[tuple(idx[:, mask])]
    return a


def remove_linear_plane(img: np.ndarray) -> np.ndarray:
    """Subtract a least-squares plane fitted to finite pixels."""
    z = np.asarray(img, dtype=float)
    ny, nx = z.shape
    yy, xx = np.indices(z.shape)
    valid = np.isfinite(z)
    if valid.sum() < 10:
        return np.nan_to_num(z, nan=0.0)
    A = np.column_stack([xx[valid].ravel(), yy[valid].ravel(), np.ones(valid.sum())])
    b = z[valid].ravel()
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        plane = coef[0] * xx + coef[1] * yy + coef[2]
        return z - plane
    except Exception:
        return z - np.nanmedian(z)


# =============================================================================
# Excel image extraction
# =============================================================================



@dataclass
class ImageRecord:
    name: str
    file: str
    sheet: str
    data: np.ndarray  # raw current in amperes


def _sheet_block_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Parse a current matrix without inferring headers or coordinate axes."""
    num = df.map(parse_numeric_cell).to_numpy(dtype=float)
    nonempty = df.notna().to_numpy()
    if np.any(nonempty & ~np.isfinite(num)):
        raise ValueError("Excel input must be a current matrix without headers or coordinate columns")
    return num


def extract_raw_map(file_path: Path) -> Tuple[List[ImageRecord], List[Dict[str, object]]]:
    """Read one raw map. Excel uses only RAW_SHEET; CSV is a numeric matrix."""
    source = _provenance_path(file_path)
    sheet = ""
    try:
        if file_path.suffix.lower() == ".csv":
            block = np.loadtxt(file_path, delimiter=",", ndmin=2) * NUMERIC_INPUT_MULTIPLIER
        else:
            with pd.ExcelFile(file_path, engine="openpyxl") as workbook:
                sheet = workbook.sheet_names[RAW_SHEET] if isinstance(RAW_SHEET, int) else RAW_SHEET
                frame = workbook.parse(sheet_name=sheet, header=None)
                block = _sheet_block_from_dataframe(frame)
        if block is None or block.ndim != 2 or block.shape[0] < MIN_IMAGE_HEIGHT or block.shape[1] < MIN_IMAGE_WIDTH:
            raise ValueError("The selected input must contain a current matrix of at least 16 x 16 pixels")
        if np.isinf(block).any() or np.count_nonzero(np.isfinite(block)) < 10:
            raise ValueError("The current map contains infinite values or too few finite pixels")
        record = ImageRecord(name=source, file=source, sheet=sheet, data=block)
        return [record], [{
            "file": source, "sheet": sheet, "status": "raw_map_ok",
            "height": block.shape[0], "width": block.shape[1],
            "finite_fraction": float(np.isfinite(block).mean()),
        }]
    except Exception as exc:
        return [], [{"file": source, "sheet": sheet, "status": "read_error", "message": str(exc)}]


def prepare_current_map(raw: np.ndarray, background: str = "plane") -> Tuple[np.ndarray, Dict[str, float]]:
    """Measure the raw floor, then prepare a separate map for motif analysis."""
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or np.isinf(values).any() or np.count_nonzero(np.isfinite(values)) < 10:
        raise ValueError("Expected a 2D map with finite current samples")
    floor = compute_current_floor_metrics(values, prefix="raw_current")
    if background == "plane":
        corrected = remove_linear_plane(values)
    elif background == "none":
        corrected = values.copy()
    else:
        raise ValueError(f"Unknown background correction: {background}")
    return fill_nan_nearest(corrected), floor


def extract_all_images(files: Sequence[Path]) -> Tuple[List[ImageRecord], pd.DataFrame]:
    all_records: List[ImageRecord] = []
    logs: List[Dict[str, object]] = []
    for f in files:
        recs, lgs = extract_raw_map(f)
        all_records.extend(recs)
        logs.extend(lgs)
    return all_records, pd.DataFrame(logs)


# =============================================================================
# Lattice detection and unit-cell folding
# =============================================================================


def fft_preprocess(img: np.ndarray) -> np.ndarray:
    z, _ = robust_zscore(img)
    z = remove_linear_plane(z)
    # High-pass to reduce slow background/flattening residuals.
    sigma = max(2.0, min(z.shape) / 16.0)
    z_hp = z - ndimage.gaussian_filter(z, sigma=sigma, mode="nearest")
    z_hp, _ = robust_zscore(z_hp)
    wy = np.hanning(z_hp.shape[0])
    wx = np.hanning(z_hp.shape[1])
    return np.nan_to_num(z_hp, nan=0.0) * wy[:, None] * wx[None, :]


def expected_reciprocal_cycles_per_nm() -> float:
    # Primitive reciprocal vector magnitude for a 2D hexagonal lattice.
    return 2.0 / (math.sqrt(3.0) * GRAPHITE_LATTICE_A_NM)


def find_lattice_vectors_fft(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Find two graphite reciprocal lattice vectors in rad/pixel using constrained FFT peaks."""
    ny, nx = img.shape
    prep = fft_preprocess(img)
    F = np.fft.fftshift(np.fft.fft2(prep))
    power = np.abs(F) ** 2
    cy, cx = ny // 2, nx // 2
    power[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0

    yidx = np.arange(ny) - cy
    xidx = np.arange(nx) - cx
    KY_idx, KX_idx = np.meshgrid(yidx, xidx, indexing="ij")
    fx_nm = KX_idx / SCAN_SIZE_X_NM
    fy_nm = KY_idx / SCAN_SIZE_Y_NM
    mag_nm = np.sqrt(fx_nm ** 2 + fy_nm ** 2)

    exp_mag = expected_reciprocal_cycles_per_nm()
    if USE_GRAPHITE_LATTICE_CONSTRAINT:
        lo = exp_mag * (1.0 - LATTICE_RECIPROCAL_MAG_TOL)
        hi = exp_mag * (1.0 + LATTICE_RECIPROCAL_MAG_TOL)
        band = (mag_nm >= lo) & (mag_nm <= hi)
    else:
        # Use a broad non-center band.
        band = mag_nm > (0.08 * max(nx / SCAN_SIZE_X_NM, ny / SCAN_SIZE_Y_NM))

    # Local maxima in the band.
    local_max = power == ndimage.maximum_filter(power, size=5, mode="nearest")
    mask = band & local_max & np.isfinite(power)
    ys, xs = np.where(mask)
    if ys.size < 6:
        # Broaden constraint if needed.
        band = (mag_nm >= exp_mag * 0.35) & (mag_nm <= exp_mag * 1.85)
        mask = band & local_max & np.isfinite(power)
        ys, xs = np.where(mask)
    if ys.size < 2:
        raise RuntimeError("FFT lattice peak detection failed: not enough peaks in graphite band.")

    vals = power[ys, xs]
    order = np.argsort(vals)[::-1]
    max_candidates = min(80, order.size)
    candidates = []
    for idx in order[:max_candidates]:
        y, x = int(ys[idx]), int(xs[idx])
        kx_i = float(KX_idx[y, x])
        ky_i = float(KY_idx[y, x])
        if abs(kx_i) < 1 and abs(ky_i) < 1:
            continue
        candidates.append({
            "kx_idx": kx_i,
            "ky_idx": ky_i,
            "power": float(power[y, x]),
            "mag_nm": float(mag_nm[y, x]),
            "y": y,
            "x": x,
        })

    if len(candidates) < 2:
        raise RuntimeError("FFT lattice peak detection failed after candidate filtering.")

    best = None
    best_score = -np.inf
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            v1_nm = np.array([c1["kx_idx"] / SCAN_SIZE_X_NM, c1["ky_idx"] / SCAN_SIZE_Y_NM], dtype=float)
            v2_nm = np.array([c2["kx_idx"] / SCAN_SIZE_X_NM, c2["ky_idx"] / SCAN_SIZE_Y_NM], dtype=float)
            n1, n2 = np.linalg.norm(v1_nm), np.linalg.norm(v2_nm)
            if n1 == 0 or n2 == 0:
                continue
            cosang = float(np.clip(np.dot(v1_nm, v2_nm) / (n1 * n2), -1.0, 1.0))
            angle = math.degrees(math.acos(cosang))
            # reject nearly opposite duplicates and nearly collinear pairs
            if angle < 35 or angle > 145:
                continue
            angle_penalty = min(abs(angle - 60.0), abs(angle - 120.0)) / 30.0
            mag_penalty = (abs(n1 - exp_mag) + abs(n2 - exp_mag)) / max(exp_mag, 1e-9)
            ratio_penalty = abs(math.log((n1 + 1e-9) / (n2 + 1e-9)))
            pscore = math.log(c1["power"] + 1.0) + math.log(c2["power"] + 1.0)
            score = pscore - 3.0 * angle_penalty - 3.0 * mag_penalty - 2.0 * ratio_penalty
            if score > best_score:
                best_score = score
                best = (c1, c2, angle, n1, n2)

    if best is None:
        raise RuntimeError("FFT lattice peak detection failed: no suitable non-collinear peak pair.")

    c1, c2, angle, n1, n2 = best
    G_raw_selected = np.array([
        [2.0 * np.pi * c1["kx_idx"] / nx, 2.0 * np.pi * c1["ky_idx"] / ny],
        [2.0 * np.pi * c2["kx_idx"] / nx, 2.0 * np.pi * c2["ky_idx"] / ny],
    ], dtype=float)
    G, canonical_diag = canonicalize_reciprocal_basis(G_raw_selected)

    # Real-space sanity check.
    dx = SCAN_SIZE_X_NM / nx
    dy = SCAN_SIZE_Y_NM / ny
    G_nm = G.copy()
    G_nm[:, 0] /= dx
    G_nm[:, 1] /= dy
    try:
        A_nm = 2.0 * np.pi * np.linalg.inv(G_nm)
        a1 = float(np.linalg.norm(A_nm[:, 0]))
        a2 = float(np.linalg.norm(A_nm[:, 1]))
        aang = float(math.degrees(math.acos(np.clip(np.dot(A_nm[:, 0], A_nm[:, 1]) / (a1 * a2), -1.0, 1.0))))
        area = float(abs(np.linalg.det(A_nm)))
    except Exception:
        a1 = a2 = aang = area = np.nan

    diag = {
        "fft_peak1_kx_idx": c1["kx_idx"],
        "fft_peak1_ky_idx": c1["ky_idx"],
        "fft_peak2_kx_idx": c2["kx_idx"],
        "fft_peak2_ky_idx": c2["ky_idx"],
        "fft_pair_angle_deg_reciprocal": float(angle),
        "fft_peak1_mag_cycles_per_nm": float(n1),
        "fft_peak2_mag_cycles_per_nm": float(n2),
        "expected_mag_cycles_per_nm": float(exp_mag),
        "lattice_a1_nm": a1,
        "lattice_a2_nm": a2,
        "lattice_angle_deg": aang,
        "lattice_area_nm2": area,
        "fft_pair_score": float(best_score),
        **canonical_diag,
    }
    return G, diag, power


def canonicalize_reciprocal_basis(G: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Choose a deterministic sign/order convention for the two FFT G vectors.

    FFT peak pairs have sign/order ambiguity.  For fixed A/B/H fractional offsets
    to be meaningful, we canonicalize the pair to a ~60 degree reciprocal angle
    and positive handedness.  This uses only FFT geometry, not image intensity at
    A/B/H sites.
    """
    G = np.asarray(G, dtype=float)
    cand = []
    for swap in (False, True):
        B = G[[1, 0], :].copy() if swap else G.copy()
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                C = B.copy()
                C[0] *= s1
                C[1] *= s2
                n1 = np.linalg.norm(C[0])
                n2 = np.linalg.norm(C[1])
                if n1 <= 0 or n2 <= 0:
                    continue
                angle = math.degrees(math.acos(np.clip(np.dot(C[0], C[1]) / (n1 * n2), -1.0, 1.0)))
                det = float(np.linalg.det(C))
                # Prefer 60 deg, right-handed, and a reproducible orientation of G1.
                a1 = math.degrees(math.atan2(C[0, 1], C[0, 0]))
                orient_pen = abs(((a1 + 180.0) % 360.0) - 180.0) / 180.0
                score = -abs(angle - 60.0) - (0.0 if det > 0 else 100.0) - 0.01 * orient_pen
                cand.append((score, C, angle, det, swap, s1, s2, a1))
    if not cand:
        return G, {"fft_basis_canonicalized": False}
    cand.sort(key=lambda x: x[0], reverse=True)
    _, C, angle, det, swap, s1, s2, a1 = cand[0]
    return C, {
        "fft_basis_canonicalized": True,
        "fft_canonical_angle_deg": float(angle),
        "fft_canonical_det": float(det),
        "fft_canonical_swapped": bool(swap),
        "fft_canonical_sign1": float(s1),
        "fft_canonical_sign2": float(s2),
        "fft_canonical_g1_angle_deg": float(a1),
    }


def fft_phase_preprocess_for_site_anchor(img: np.ndarray) -> np.ndarray:
    """Preprocess image for complex FFT phase extraction without windowing."""
    z, _ = robust_zscore(img)
    z = remove_linear_plane(z)
    sigma = max(2.0, min(z.shape) / max(FFT_PHASE_PREPROCESS_SIGMA_FRAC, 1.0))
    z = z - ndimage.gaussian_filter(z, sigma=sigma, mode="nearest")
    z, _ = robust_zscore(z)
    return np.nan_to_num(z, nan=0.0)


def estimate_fft_phase_offsets(img: np.ndarray, G: np.ndarray) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """Return fractional phase offsets from complex coefficients at G1/G2.

    For an image component cos(G·r + phi), the complex coefficient at +G has
    phase phi.  Adding phi/2π to the fractional coordinate puts that Fourier
    maximum at u=0.  The sign is configurable because FFT sign conventions can be
    inverted by different data conventions.
    """
    arr = fft_phase_preprocess_for_site_anchor(img)
    yy, xx = np.indices(arr.shape)
    offsets = []
    phases = []
    amps = []
    for k in range(2):
        phase_arg = G[k, 0] * xx + G[k, 1] * yy
        coeff = np.sum(arr * np.exp(-1j * phase_arg))
        phi = float(np.angle(coeff))
        amp = float(np.abs(coeff))
        off = float((FFT_PHASE_OFFSET_SIGN * phi / (2.0 * np.pi)) % 1.0)
        offsets.append(off)
        phases.append(phi)
        amps.append(amp)
    return (offsets[0], offsets[1]), {
        "fft_phase_offset_u_frac": offsets[0],
        "fft_phase_offset_v_frac": offsets[1],
        "fft_phase_phi1_rad": phases[0],
        "fft_phase_phi2_rad": phases[1],
        "fft_phase_amp1": amps[0],
        "fft_phase_amp2": amps[1],
        "fft_phase_offset_sign": float(FFT_PHASE_OFFSET_SIGN),
    }


def fixed_fft_site_choice() -> Dict[str, object]:
    if FFT_FIXED_SITE_BASIS_NAME not in FIXED_ABH_SITE_BASES:
        raise ValueError(f"Unknown FFT_FIXED_SITE_BASIS_NAME={FFT_FIXED_SITE_BASIS_NAME!r}")
    base = FIXED_ABH_SITE_BASES[FFT_FIXED_SITE_BASIS_NAME]
    positions_role = {"B": base["B"], "A": base["A"], "H": base["H"], "geoA": base["A"], "geoB": base["B"], "geoH": base["H"]}
    return {
        "basis": FFT_FIXED_SITE_BASIS_NAME,
        "origin": base["B"],
        "site_geometry_model": "fft_phase_fixed_B_vertex",
        "origin_definition": "B_vertex_from_fft_phase",
        "positions_role": positions_role,
        "positions_geo": {"geoA": base["A"], "geoB": base["B"], "H": base["H"]},
        "role": FFT_FIXED_ROLE_ASSIGNMENT.copy(),
        "score": np.nan,
        "median_delta_BA": np.nan,
        "median_delta_AH": np.nan,
        "derived_from": base.get("derived_from", ""),
    }


def fold_image_to_unit_cell(img: np.ndarray, G: np.ndarray, grid: int = UNIT_CELL_GRID, phase_offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Fold pixels to fractional unit-cell coordinates using two reciprocal vectors."""
    arr = np.asarray(img, dtype=float)
    ny, nx = arr.shape
    yy, xx = np.indices(arr.shape)
    u = (G[0, 0] * xx + G[0, 1] * yy) / (2.0 * np.pi) + float(phase_offset[0])
    v = (G[1, 0] * xx + G[1, 1] * yy) / (2.0 * np.pi) + float(phase_offset[1])
    u = np.mod(u, 1.0)
    v = np.mod(v, 1.0)
    iu = np.floor(u * grid).astype(int) % grid
    iv = np.floor(v * grid).astype(int) % grid
    # array convention: rows -> v, columns -> u
    accum = np.zeros((grid, grid), dtype=float)
    count = np.zeros((grid, grid), dtype=float)
    vals = np.nan_to_num(arr, nan=np.nanmedian(arr))
    np.add.at(accum, (iv.ravel(), iu.ravel()), vals.ravel())
    np.add.at(count, (iv.ravel(), iu.ravel()), 1.0)
    out = np.divide(accum, count, out=np.full_like(accum, np.nan), where=count > 0)
    out = fill_nan_nearest(out)
    return out, count


def fold_image_to_unit_cell_masked(img: np.ndarray, G: np.ndarray, mask: np.ndarray, grid: int = UNIT_CELL_GRID, phase_offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Fold only masked pixels to fractional unit-cell coordinates.

    This is used for split-half validation: the same apparent lattice is used to
    fold two independent subsets of real-space unit cells.  If the lattice
    registration is meaningful, the two folded motifs should be correlated.
    """
    arr = np.asarray(img, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    ny, nx = arr.shape
    yy, xx = np.indices(arr.shape)
    u = (G[0, 0] * xx + G[0, 1] * yy) / (2.0 * np.pi) + float(phase_offset[0])
    v = (G[1, 0] * xx + G[1, 1] * yy) / (2.0 * np.pi) + float(phase_offset[1])
    u_mod = np.mod(u, 1.0)
    v_mod = np.mod(v, 1.0)
    iu = np.floor(u_mod * grid).astype(int) % grid
    iv = np.floor(v_mod * grid).astype(int) % grid
    accum = np.zeros((grid, grid), dtype=float)
    count = np.zeros((grid, grid), dtype=float)
    vals = np.nan_to_num(arr, nan=np.nanmedian(arr))
    if not np.any(mask):
        return np.full((grid, grid), np.nan), count
    np.add.at(accum, (iv[mask].ravel(), iu[mask].ravel()), vals[mask].ravel())
    np.add.at(count, (iv[mask].ravel(), iu[mask].ravel()), 1.0)
    out = np.divide(accum, count, out=np.full_like(accum, np.nan), where=count > 0)
    out = fill_nan_nearest(out)
    return out, count


def unitcell_split_half_reliability(img: np.ndarray, G: np.ndarray, grid: int = UNIT_CELL_GRID, phase_offset: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, float]:
    """Validate that repeated unit cells fold to a reproducible motif.

    The image is split into two subsets using the parity of unwrapped lattice-cell
    indices, approximately separating alternating apparent unit cells.  Each half
    is folded independently and robust-normalized; their correlation is a direct
    reliability check for unit-cell capture.

    Interpretation:
      * high correlation: many repeated cells carry the same folded motif phase;
      * low/negative correlation: noisy image, non-affine drift, wrong FFT peak,
        or insufficient per-bin coverage.  This should be inspected rather than
        used as an automatic hard rejection.
    """
    arr = np.asarray(img, dtype=float)
    yy, xx = np.indices(arr.shape)
    u_unwrapped = (G[0, 0] * xx + G[0, 1] * yy) / (2.0 * np.pi)
    v_unwrapped = (G[1, 0] * xx + G[1, 1] * yy) / (2.0 * np.pi)
    # Split by apparent unit-cell index, not by pixel coordinates, so each half
    # samples the full image and the full unit-cell area as evenly as possible.
    cell_parity = (np.floor(u_unwrapped).astype(np.int64) + np.floor(v_unwrapped).astype(np.int64)) & 1
    valid = np.isfinite(arr)
    mask0 = valid & (cell_parity == 0)
    mask1 = valid & (cell_parity == 1)
    out: Dict[str, float] = {
        "unitcell_split_half_npix_0": float(np.sum(mask0)),
        "unitcell_split_half_npix_1": float(np.sum(mask1)),
    }
    try:
        m0, c0 = fold_image_to_unit_cell_masked(arr, G, mask0, grid=grid, phase_offset=phase_offset)
        m1, c1 = fold_image_to_unit_cell_masked(arr, G, mask1, grid=grid, phase_offset=phase_offset)
        z0, _ = robust_zscore(m0)
        z1, _ = robust_zscore(m1)
        corr = corrcoef2(z0, z1)
        cov0 = float(np.mean(c0 > 0))
        cov1 = float(np.mean(c1 > 0))
        # Compare each half to the full folded result as an additional stability
        # indicator.  These values are often higher than split-half corr because
        # the full map contains both halves.
        full, cfull = fold_image_to_unit_cell(arr, G, grid=grid, phase_offset=phase_offset)
        zfull, _ = robust_zscore(full)
        out.update({
            "unitcell_split_half_corr": float(corr),
            "unitcell_split_half_coverage_0": cov0,
            "unitcell_split_half_coverage_1": cov1,
            "unitcell_split_half_min_coverage": float(min(cov0, cov1)),
            "unitcell_split_half_corr_to_full_0": float(corrcoef2(z0, zfull)),
            "unitcell_split_half_corr_to_full_1": float(corrcoef2(z1, zfull)),
            "unitcell_split_half_full_coverage": float(np.mean(cfull > 0)),
        })
    except Exception:
        out.update({
            "unitcell_split_half_corr": np.nan,
            "unitcell_split_half_coverage_0": np.nan,
            "unitcell_split_half_coverage_1": np.nan,
            "unitcell_split_half_min_coverage": np.nan,
            "unitcell_split_half_corr_to_full_0": np.nan,
            "unitcell_split_half_corr_to_full_1": np.nan,
            "unitcell_split_half_full_coverage": np.nan,
        })
    return out


def roll2(a: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
    return np.roll(np.roll(a, int(shift[0]), axis=0), int(shift[1]), axis=1)


def corrcoef2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    aa = aa - np.nanmean(aa)
    bb = bb - np.nanmean(bb)
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den <= 0:
        return 0.0
    return float(np.dot(aa, bb) / den)


def best_integer_shift_to_ref(img: np.ndarray, ref: np.ndarray) -> Tuple[int, int]:
    """Return integer circular shift for img that maximizes correlation with ref."""
    f_ref = np.fft.fft2(ref - np.mean(ref))
    f_img = np.fft.fft2(img - np.mean(img))
    corr = np.fft.ifft2(f_ref * np.conj(f_img)).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    sy, sx = int(peak[0]), int(peak[1])
    if sy > img.shape[0] // 2:
        sy -= img.shape[0]
    if sx > img.shape[1] // 2:
        sx -= img.shape[1]
    # Try both signs to avoid correlation convention mistakes.
    cand1 = (sy, sx)
    cand2 = (-sy, -sx)
    if corrcoef2(roll2(img, cand2), ref) > corrcoef2(roll2(img, cand1), ref):
        return cand2
    return cand1


def align_unit_cell_maps(maps: np.ndarray, iterations: int = ALIGN_ITERATIONS) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Iterative circular alignment of unit-cell maps."""
    arr = np.array(maps, dtype=float, copy=True)
    n = arr.shape[0]
    # Start from first map after zero-mean normalization.
    ref = arr[0] - np.mean(arr[0])
    cumulative = [(0, 0) for _ in range(n)]
    aligned = arr.copy()
    for _ in range(max(1, iterations)):
        new_aligned = []
        for i in range(n):
            sh = best_integer_shift_to_ref(aligned[i], ref)
            new_map = roll2(aligned[i], sh)
            cumulative[i] = (cumulative[i][0] + sh[0], cumulative[i][1] + sh[1])
            new_aligned.append(new_map)
        aligned = np.stack(new_aligned, axis=0)
        ref = np.mean(aligned, axis=0)
        ref = ref - np.mean(ref)
    return aligned, cumulative


# =============================================================================
# Site positions and motif coordinates
# =============================================================================


# Graphite-correct site model.
#
# The unit cell contains two atom-site candidates and one hollow-site candidate.
# A single H candidate avoids treating a neighboring atom site as a hollow. The
# grid-search origin is the
# geometric hollow candidate, and the two atom candidates are placed at fractional
# offsets from that hollow. Two offset families are included because FFT/lattice
# basis conventions can rotate or relabel the fractional coordinates.
CANDIDATE_BASES = {
    # name: {"atom_offsets": (geoA_offset, geoB_offset), "hollow_offset": H_offset}
    "graphite_hollow_13_23__23_13": {
        "atom_offsets": ((1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0)),
        "hollow_offset": (0.0, 0.0),
    },
    "graphite_hollow_13_13__23_23": {
        "atom_offsets": ((1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0)),
        "hollow_offset": (0.0, 0.0),
    },
}

# Fixed B-vertex site bases used by SITE_COORDINATE_MODE="fft_phase_fixed".
# Coordinates are fractional (u, v) positions in the FFT-phase-corrected folded
# unit cell.  B is deliberately at the unit-cell vertex (0, 0).  Choose the basis
# that matches the FFT convention you want to use for the manuscript figures.
FIXED_ABH_SITE_BASES = {
    # B -> H -> A along the main diagonal in (u, v).
    # Equivalent to H-origin basis graphite_hollow_13_13__23_23 with B=geoB.
    "B_vertex_diag_13_13": {
        "B": (0.0, 0.0),
        "H": (1.0 / 3.0, 1.0 / 3.0),
        "A": (2.0 / 3.0, 2.0 / 3.0),
        "derived_from": "graphite_hollow_13_13__23_23",
    },
    # B -> H/A on the cross diagonal in (u, v).
    # Equivalent to H-origin basis graphite_hollow_13_23__23_13 with B=geoB.
    "B_vertex_cross_13_23": {
        "B": (0.0, 0.0),
        "H": (1.0 / 3.0, 2.0 / 3.0),
        "A": (2.0 / 3.0, 1.0 / 3.0),
        "derived_from": "graphite_hollow_13_23__23_13",
    },
}


def mod1_pair(p: Tuple[float, float]) -> Tuple[float, float]:
    return (float(p[0] % 1.0), float(p[1] % 1.0))


def positions_from_origin_basis(origin: Tuple[float, float], basis_name: str) -> Dict[str, Tuple[float, float]]:
    basis = CANDIDATE_BASES[basis_name]
    atom_offsets = basis["atom_offsets"]
    h_off = basis["hollow_offset"]
    ou, ov = origin
    geoA = mod1_pair((ou + atom_offsets[0][0], ov + atom_offsets[0][1]))
    geoB = mod1_pair((ou + atom_offsets[1][0], ov + atom_offsets[1][1]))
    geoH = mod1_pair((ou + h_off[0], ov + h_off[1]))
    return {"geoA": geoA, "geoB": geoB, "H": geoH}


def periodic_gaussian_weight(grid: int, pos: Tuple[float, float], sigma_frac: float = SITE_AVERAGE_SIGMA_FRAC) -> np.ndarray:
    """Gaussian site mask with periodic wrapping.

    SITE_MASK_DISTANCE_METRIC="hex120" uses the canonical oblique graphite metric
    d^2 = du^2 + dv^2 - du*dv.  We explicitly minimize over neighboring periodic
    images, which is more accurate near unit-cell boundaries than wrapping du and
    dv independently once.  Use SITE_MASK_DISTANCE_METRIC="rect" to reproduce the
    previous rectangular fractional-coordinate mask.
    """
    u0, v0 = pos
    coords = (np.arange(grid) + 0.5) / grid
    U, V = np.meshgrid(coords, coords, indexing="xy")
    du0 = ((U - u0 + 0.5) % 1.0) - 0.5
    dv0 = ((V - v0 + 0.5) % 1.0) - 0.5
    best = np.full_like(du0, np.inf, dtype=float)
    for su in (-1.0, 0.0, 1.0):
        for sv in (-1.0, 0.0, 1.0):
            du = du0 + su
            dv = dv0 + sv
            if SITE_MASK_DISTANCE_METRIC == "hex120":
                d2 = du ** 2 + dv ** 2 - du * dv
            else:
                d2 = du ** 2 + dv ** 2
            best = np.minimum(best, d2)
    w = np.exp(-best / (2.0 * sigma_frac ** 2))
    s = float(w.sum())
    if s <= 0:
        return np.ones((grid, grid)) / (grid * grid)
    return w / s


def site_mean(unit_map: np.ndarray, pos: Tuple[float, float], sigma_frac: float = SITE_AVERAGE_SIGMA_FRAC) -> float:
    w = periodic_gaussian_weight(unit_map.shape[0], pos, sigma_frac)
    return float(np.sum(w * unit_map))


def extract_candidate_site_values(maps: np.ndarray, pos: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    vals: Dict[str, List[float]] = {"geoA": [], "geoB": [], "H": []}
    for m in maps:
        for k in vals:
            vals[k].append(site_mean(m, pos[k]))
    return {k: np.asarray(v, dtype=float) for k, v in vals.items()}


def choose_role_assignment(vals: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Assign B/A roles by global median brightness; H is fixed geometric hollow."""
    med = {k: float(np.nanmedian(v)) for k, v in vals.items()}
    if USE_BAH_ROLE_CONVENTION:
        if med["geoA"] >= med["geoB"]:
            B_key, A_key = "geoA", "geoB"
        else:
            B_key, A_key = "geoB", "geoA"
    else:
        # Deterministic fallback; still not a crystallographic beta assignment.
        B_key, A_key = "geoB", "geoA"
    return {"B_key": B_key, "A_key": A_key, "H_key": "H"}



def site_pick_score_from_deltas(
    d_ba,
    d_ah,
    ordered=None,
    stability=0.0,
    mode: Optional[str] = None,
):
    """Theta-bias-controlled A/B/H candidate score.

    d_ba = I_B - I_A and d_ah = I_A - I_H.  The legacy score used
    +0.5*min(d_ba, d_ah), which is a hidden prior toward d_ba ~= d_ah
    and therefore theta ~= 30 deg.  The recommended score below uses only
    total B-H visibility and ordering penalties, so for a fixed B-H contrast it
    does not prefer AH-like, ABH-like, or B-like motifs.
    """
    mode = mode or INTENSITY_SITE_SCORE_MODE
    xp = np
    d_ba_arr = np.asarray(d_ba) if not np.isscalar(d_ba) else d_ba
    d_ah_arr = np.asarray(d_ah) if not np.isscalar(d_ah) else d_ah

    if mode == "legacy_ABH_balance":
        positive = xp.maximum(0.0, d_ba_arr) + xp.maximum(0.0, d_ah_arr) + 0.5 * xp.maximum(0.0, xp.minimum(d_ba_arr, d_ah_arr))
        neg_penalty = 4.0 * (xp.maximum(0.0, -d_ba_arr) + xp.maximum(0.0, -d_ah_arr))
        score = positive - neg_penalty
    elif mode == "theta_neutral_saturated":
        total = xp.maximum(0.0, d_ba_arr + d_ah_arr)
        scale = max(float(SITE_SCORE_SATURATION_SCALE), 1.0e-9)
        visibility = xp.tanh(total / scale)
        # Require both gaps to be positive but do not reward equality of the gaps.
        neg_penalty = 4.0 * (xp.maximum(0.0, -d_ba_arr) + xp.maximum(0.0, -d_ah_arr))
        weak_order_penalty = 0.35 * (xp.maximum(0.0, -d_ba_arr) + xp.maximum(0.0, -d_ah_arr))
        score = visibility - neg_penalty - weak_order_penalty
    else:  # "theta_neutral_total_contrast"
        # For positive ordering, this is simply I_B - I_H.  It is independent of
        # where A falls between B and H, so it does not impose theta ~= 30 deg.
        visibility = d_ba_arr + d_ah_arr
        neg_penalty = 5.0 * (xp.maximum(0.0, -d_ba_arr) + xp.maximum(0.0, -d_ah_arr))
        score = visibility - neg_penalty

    if ordered is not None:
        score = score + 0.10 * np.asarray(ordered, dtype=float)
    if stability is not None:
        score = score + stability
    return score


def theta_from_deltas_deg(d_ba, d_ah):
    """Candidate theta from Delta_BA and Delta_AH without reusing site masks."""
    d_ba = np.asarray(d_ba, dtype=float)
    d_ah = np.asarray(d_ah, dtype=float)
    c_AH = (d_ba + 2.0 * d_ah) / math.sqrt(6.0)
    c_beta_abs = np.abs(d_ba) / math.sqrt(2.0)
    return np.degrees(np.arctan2(c_beta_abs, np.maximum(c_AH, 1.0e-12)))

def score_candidate_origin(vals: Dict[str, np.ndarray], role: Dict[str, str]) -> float:
    H_vals = vals["H"]
    B_vals = vals[role["B_key"]]
    A_vals = vals[role["A_key"]]
    d_ba = float(np.nanmedian(B_vals - A_vals))
    d_ah = float(np.nanmedian(A_vals - H_vals))
    # Want bright > intermediate > geometric hollow in the intensity-role convention,
    # but do not reward Delta_BA ~= Delta_AH because that biases theta toward 30 deg.
    ordering_fraction = float(np.nanmean((B_vals > A_vals) & (A_vals > H_vals)))
    stability = -0.10 * (float(np.nanstd(B_vals - A_vals)) + float(np.nanstd(A_vals - H_vals)))
    return float(site_pick_score_from_deltas(d_ba, d_ah, ordered=ordering_fraction, stability=stability))



# -----------------------------------------------------------------------------
# Vectorized / GPU-accelerated site-search helpers
# -----------------------------------------------------------------------------

_GPU_SITE_SEARCH_STATUS: Dict[str, object] = {"backend": "not_initialized", "message": ""}


def _get_site_search_backend():
    """Return (xp, use_gpu, message) for site-search vectorization."""
    if not USE_GPU_ACCELERATION_FOR_SITE_SEARCH:
        _GPU_SITE_SEARCH_STATUS.update({"backend": "numpy_cpu", "message": "GPU disabled by setting."})
        return np, False, "GPU disabled by setting."
    try:
        import cupy as cp  # type: ignore
        ndev = int(cp.cuda.runtime.getDeviceCount())
        if ndev <= 0:
            raise RuntimeError("No CUDA device reported by CuPy")
        # Touch the current device so driver problems fail here, not mid-analysis.
        dev = cp.cuda.Device()
        dev.use()
        _GPU_SITE_SEARCH_STATUS.update({"backend": "cupy_gpu", "message": f"Using CuPy on CUDA device {int(dev.id)}."})
        return cp, True, f"Using CuPy on CUDA device {int(dev.id)}."
    except Exception as exc:
        msg = f"CuPy/CUDA unavailable; using NumPy CPU vectorized site search. Reason: {exc}"
        _GPU_SITE_SEARCH_STATUS.update({"backend": "numpy_cpu", "message": msg})
        return np, False, msg


def _asnumpy_maybe(x):
    """Convert CuPy array to NumPy if needed."""
    try:
        import cupy as cp  # type: ignore
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def build_site_candidate_table() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Build all graphite-triplet candidate positions for the origin grid."""
    rows: List[Dict[str, object]] = []
    arrs: Dict[str, List[float]] = {"geoA_u": [], "geoA_v": [], "geoB_u": [], "geoB_v": [], "H_u": [], "H_v": []}
    grid = np.arange(ORIGIN_GRID_SEARCH_N, dtype=float) / ORIGIN_GRID_SEARCH_N
    for basis in CANDIDATE_BASES:
        for ou in grid:
            for ov in grid:
                pos = positions_from_origin_basis((float(ou), float(ov)), basis)
                idx = len(rows)
                rows.append({
                    "candidate_index": idx,
                    "basis": basis,
                    "site_geometry_model": SITE_GEOMETRY_MODEL,
                    "origin_definition": "single_geometric_hollow",
                    "origin_u": float(ou),
                    "origin_v": float(ov),
                    "geoA_u": pos["geoA"][0], "geoA_v": pos["geoA"][1],
                    "geoB_u": pos["geoB"][0], "geoB_v": pos["geoB"][1],
                    "geoH_u": pos["H"][0], "geoH_v": pos["H"][1],
                })
                arrs["geoA_u"].append(pos["geoA"][0]); arrs["geoA_v"].append(pos["geoA"][1])
                arrs["geoB_u"].append(pos["geoB"][0]); arrs["geoB_v"].append(pos["geoB"][1])
                arrs["H_u"].append(pos["H"][0]); arrs["H_v"].append(pos["H"][1])
    return pd.DataFrame(rows), {k: np.asarray(v, dtype=np.float32) for k, v in arrs.items()}


def _site_values_vectorized_for_positions(
    maps_ref: np.ndarray,
    pos_u: np.ndarray,
    pos_v: np.ndarray,
    xp,
    batch_size: int = GPU_SITE_SEARCH_BATCH_CANDIDATES,
) -> np.ndarray:
    """Gaussian-weighted site means for many candidate positions and many maps.

    Returns a NumPy array with shape (n_candidates, n_maps).  Heavy pixel-weight
    construction and matrix multiplication run on CuPy when available.
    """
    maps_np = np.asarray(maps_ref, dtype=np.float32)
    if maps_np.ndim == 2:
        maps_np = maps_np[None, :, :]
    n_maps, grid_h, grid_w = maps_np.shape
    if grid_h != grid_w:
        raise ValueError("Folded unit-cell maps must be square for site-search acceleration.")
    grid = int(grid_h)
    n_cand = int(len(pos_u))
    dtype = np.float32 if GPU_SITE_SEARCH_DTYPE == "float32" else np.float64

    coords = ((np.arange(grid, dtype=dtype) + dtype(0.5)) / dtype(grid)).astype(dtype)
    U, V = np.meshgrid(coords, coords, indexing="xy")
    pix_u_np = U.reshape(-1)
    pix_v_np = V.reshape(-1)
    maps_flat_np = maps_np.reshape(n_maps, -1)
    # NaNs are rare, but a single NaN would poison dot products.  Replace by map median.
    if not np.isfinite(maps_flat_np).all():
        med = np.nanmedian(maps_flat_np, axis=1)
        inds = np.where(~np.isfinite(maps_flat_np))
        maps_flat_np = maps_flat_np.copy()
        maps_flat_np[inds] = np.take(med, inds[0])

    pix_u = xp.asarray(pix_u_np, dtype=dtype)
    pix_v = xp.asarray(pix_v_np, dtype=dtype)
    maps_flat_T = xp.asarray(maps_flat_np.T, dtype=dtype)  # pixels x maps
    out_chunks: List[np.ndarray] = []
    sigma2 = dtype(2.0 * SITE_AVERAGE_SIGMA_FRAC ** 2)

    for start in range(0, n_cand, int(batch_size)):
        stop = min(n_cand, start + int(batch_size))
        pu = xp.asarray(pos_u[start:stop], dtype=dtype)[:, None]
        pv = xp.asarray(pos_v[start:stop], dtype=dtype)[:, None]
        du0 = ((pix_u[None, :] - pu + dtype(0.5)) % dtype(1.0)) - dtype(0.5)
        dv0 = ((pix_v[None, :] - pv + dtype(0.5)) % dtype(1.0)) - dtype(0.5)
        best = xp.full(du0.shape, xp.inf, dtype=dtype)
        for su in (-1.0, 0.0, 1.0):
            for sv in (-1.0, 0.0, 1.0):
                du = du0 + dtype(su)
                dv = dv0 + dtype(sv)
                if SITE_MASK_DISTANCE_METRIC == "hex120":
                    d2 = du * du + dv * dv - du * dv
                else:
                    d2 = du * du + dv * dv
                best = xp.minimum(best, d2)
        W = xp.exp(-best / sigma2)
        W = W / (xp.sum(W, axis=1, keepdims=True) + dtype(1e-30))
        vals = W @ maps_flat_T  # candidates x maps
        out_chunks.append(_asnumpy_maybe(vals).astype(np.float64, copy=False))
        # Release batch temporaries promptly when using CuPy.
        del pu, pv, du0, dv0, best, W, vals
    return np.vstack(out_chunks)


def _choice_from_candidate_row(row: pd.Series, role: Dict[str, str]) -> Dict[str, object]:
    pos_geo = {
        "geoA": (float(row["geoA_u"]), float(row["geoA_v"])),
        "geoB": (float(row["geoB_u"]), float(row["geoB_v"])),
        "H": (float(row["geoH_u"]), float(row["geoH_v"])),
    }
    positions_role = {
        "B": pos_geo[role["B_key"]],
        "A": pos_geo[role["A_key"]],
        "H": pos_geo["H"],
        "geoA": pos_geo["geoA"],
        "geoB": pos_geo["geoB"],
        "geoH": pos_geo["H"],
    }
    return {
        "basis": row["basis"],
        "origin": (float(row["origin_u"]), float(row["origin_v"])),
        "positions_geo": pos_geo,
        "positions_role": positions_role,
        "role": role,
    }


def grid_search_site_origin_vectorized(maps_ref: np.ndarray) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Global site-origin search using vectorized CPU/GPU evaluation."""
    maps_np = np.asarray(maps_ref, dtype=float)
    if maps_np.ndim == 2:
        maps_np = maps_np[None, :, :]
    cand_df, arr = build_site_candidate_table()
    xp, use_gpu, msg = _get_site_search_backend()
    if GPU_SITE_SEARCH_VERBOSE:
        print(f"[site-search] {msg}")
    vals_geoA = _site_values_vectorized_for_positions(maps_np, arr["geoA_u"], arr["geoA_v"], xp)
    vals_geoB = _site_values_vectorized_for_positions(maps_np, arr["geoB_u"], arr["geoB_v"], xp)
    vals_H = _site_values_vectorized_for_positions(maps_np, arr["H_u"], arr["H_v"], xp)

    medA = np.nanmedian(vals_geoA, axis=1)
    medB = np.nanmedian(vals_geoB, axis=1)
    B_is_geoA = medA >= medB
    B_vals = np.where(B_is_geoA[:, None], vals_geoA, vals_geoB)
    A_vals = np.where(B_is_geoA[:, None], vals_geoB, vals_geoA)
    d_ba = np.nanmedian(B_vals - A_vals, axis=1)
    d_ah = np.nanmedian(A_vals - vals_H, axis=1)
    ordering_fraction = np.nanmean((B_vals > A_vals) & (A_vals > vals_H), axis=1)
    stability = -0.10 * (np.nanstd(B_vals - A_vals, axis=1) + np.nanstd(A_vals - vals_H, axis=1))
    score = site_pick_score_from_deltas(d_ba, d_ah, ordered=ordering_fraction, stability=stability)
    legacy_score = site_pick_score_from_deltas(d_ba, d_ah, ordered=ordering_fraction, stability=stability, mode="legacy_ABH_balance")

    out = cand_df.copy()
    out["score"] = score
    out["site_pick_score_mode"] = INTENSITY_SITE_SCORE_MODE
    if SAVE_LEGACY_ABH_BALANCE_SCORE_FOR_DIAGNOSTIC:
        out["legacy_ABH_balance_score"] = legacy_score
    out["B_key"] = np.where(B_is_geoA, "geoA", "geoB")
    out["A_key"] = np.where(B_is_geoA, "geoB", "geoA")
    out["H_key"] = "H"
    out["median_I_B"] = np.nanmedian(B_vals, axis=1)
    out["median_I_A"] = np.nanmedian(A_vals, axis=1)
    out["median_I_H"] = np.nanmedian(vals_H, axis=1)
    out["median_delta_BA"] = d_ba
    out["median_delta_AH"] = d_ah
    out["B_gt_A_gt_H_fraction"] = ordering_fraction
    out["site_search_backend"] = _GPU_SITE_SEARCH_STATUS.get("backend", "unknown")
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    best_row = out.iloc[0]
    role = {"B_key": str(best_row["B_key"]), "A_key": str(best_row["A_key"]), "H_key": "H"}
    best = _choice_from_candidate_row(best_row, role)
    best.update(best_row.to_dict())
    return best, out


def site_choice_from_phase_aligned_ensemble_mean(
    phase_aligned_maps: np.ndarray,
) -> Tuple[Dict[str, object], pd.DataFrame, np.ndarray]:
    """Calibrate one A/B/H registry from the phase-aligned ensemble mean.

    The arithmetic ensemble mean is formed first, so the origin, graphite
    triplet geometry, and A/B role assignment are selected exactly once.  The
    returned registry is intended to be applied unchanged to every acquisition;
    no individual map participates in a separate site-position optimization.
    """
    maps_np = np.asarray(phase_aligned_maps, dtype=float)
    if maps_np.ndim != 3:
        raise ValueError("phase_aligned_maps must have shape (n_images, grid, grid)")
    if maps_np.shape[0] < 1:
        raise ValueError("phase_aligned_maps must contain at least one image")

    ensemble_mean = np.nanmean(maps_np, axis=0)
    if not np.isfinite(ensemble_mean).any():
        raise ValueError("The phase-aligned ensemble mean contains no finite values")

    choice, search_df = grid_search_site_origin_vectorized(ensemble_mean[None, :, :])
    choice = dict(choice)
    choice.update({
        "image_name": "phase_aligned_ensemble_mean",
        "fft_orientation_label": "all_acquisitions_common_frame",
        "n_images_in_orientation_group": int(maps_np.shape[0]),
        "site_geometry_model": SITE_GEOMETRY_MODEL,
        "origin_definition": "phase_aligned_ensemble_mean_grid_search_geometric_hollow",
        "site_coordinate_mode": "ensemble_mean_fixed",
        "site_registry_scope": "single_fixed_registry_all_acquisitions",
        "site_registry_calibration_source": "phase_aligned_ensemble_arithmetic_mean",
        "site_positions_fixed_across_acquisitions": True,
    })

    search_df = search_df.copy()
    search_df["site_coordinate_mode"] = "ensemble_mean_fixed"
    search_df["site_registry_scope"] = "single_fixed_registry_all_acquisitions"
    search_df["site_registry_calibration_source"] = "phase_aligned_ensemble_arithmetic_mean"
    search_df["n_images_in_ensemble_mean"] = int(maps_np.shape[0])
    search_df["diagnostic_only_not_used_for_site_coordinates"] = False
    return choice, search_df, ensemble_mean


def fixed_registry_positions_for_acquisitions(
    site_choice: Dict[str, object],
    n_acquisitions: int,
) -> List[Dict[str, Tuple[float, float]]]:
    """Repeat one registry and verify that no acquisition-specific positions enter."""
    if n_acquisitions < 1:
        raise ValueError("n_acquisitions must be positive")
    positions = site_choice.get("positions_role")
    if not isinstance(positions, dict):
        raise ValueError("site_choice does not contain a positions_role mapping")
    required = ("B", "A", "H")
    if any(site not in positions for site in required):
        raise ValueError("positions_role must contain B, A, and H")

    registry = {
        key: (float(value[0]), float(value[1]))
        for key, value in positions.items()
    }
    repeated = [dict(registry) for _ in range(int(n_acquisitions))]
    reference = tuple(registry[site] for site in required)
    if any(tuple(row[site] for site in required) != reference for row in repeated):
        raise RuntimeError("Fixed A/B/H registry invariant failed")
    return repeated


def grid_search_site_origin_all_maps_per_image_vectorized(
    maps_ref: np.ndarray,
    names: Sequence[str],
    fft_orientation_labels: Optional[Sequence[str]] = None,
    top_n: int = INTENSITY_PER_IMAGE_SITE_TOP_N,
) -> Tuple[Dict[str, Dict[str, object]], pd.DataFrame]:
    """Independent per-image A/B/H detection, vectorized across all images.

    This is the GPU-accelerated replacement for calling
    grid_search_site_origin_single_map() in a Python loop.  It evaluates all
    origin/basis candidates for all images in a few batched matrix multiplications.
    """
    maps_np = np.asarray(maps_ref, dtype=float)
    if maps_np.ndim != 3:
        raise ValueError("maps_ref must have shape (n_images, grid, grid)")
    n_maps = maps_np.shape[0]
    cand_df, arr = build_site_candidate_table()
    xp, use_gpu, msg = _get_site_search_backend()
    if GPU_SITE_SEARCH_VERBOSE:
        print(f"[site-search] {msg}")
        print(f"[site-search] evaluating {len(cand_df)} A/B/H candidates for {n_maps} images")
    vals_geoA = _site_values_vectorized_for_positions(maps_np, arr["geoA_u"], arr["geoA_v"], xp)
    vals_geoB = _site_values_vectorized_for_positions(maps_np, arr["geoB_u"], arr["geoB_v"], xp)
    vals_H = _site_values_vectorized_for_positions(maps_np, arr["H_u"], arr["H_v"], xp)

    # Per-image role assignment: brighter atom-site -> B, other atom-site -> A.
    B_is_geoA = vals_geoA >= vals_geoB  # candidates x images
    B_vals = np.where(B_is_geoA, vals_geoA, vals_geoB)
    A_vals = np.where(B_is_geoA, vals_geoB, vals_geoA)
    d_ba = B_vals - A_vals
    d_ah = A_vals - vals_H
    ordered = (B_vals > A_vals) & (A_vals > vals_H)
    # Single-image version of score_candidate_origin. d_ba is nonnegative by
    # role construction; d_ah determines whether A is truly intermediate above H.
    # Use theta-neutral score by default to avoid a hidden ABH-prototype prior.
    score = site_pick_score_from_deltas(d_ba, d_ah, ordered=ordered.astype(float))
    legacy_score = site_pick_score_from_deltas(d_ba, d_ah, ordered=ordered.astype(float), mode="legacy_ABH_balance")
    candidate_theta_deg = theta_from_deltas_deg(d_ba, d_ah)

    best_idx = np.nanargmax(score, axis=0)  # one candidate per image
    choices: Dict[str, Dict[str, object]] = {}
    top_rows: List[pd.DataFrame] = []
    labels = list(fft_orientation_labels) if fft_orientation_labels is not None else ["unknown"] * n_maps

    for j in range(n_maps):
        nm = str(names[j])
        bi = int(best_idx[j])

        # Candidate subset for uncertainty-aware soft estimate.  This is computed
        # before choosing the hard marker position so ambiguous images carry their
        # site-picking uncertainty into the reported theta.
        k_soft = max(1, int(SITE_SOFT_TOP_N if SITE_SOFT_TOP_N is not None else top_n or 1))
        soft_idx = np.argpartition(score[:, j], -min(k_soft, score.shape[0]))[-min(k_soft, score.shape[0]):]
        soft_idx = soft_idx[np.argsort(score[soft_idx, j])[::-1]]
        soft_summary = _soft_candidate_motif_summary(
            score[:, j],
            B_vals[:, j],
            A_vals[:, j],
            vals_H[:, j],
            soft_idx,
        ) if USE_SITE_PICKING_UNCERTAINTY_CORRECTION else {}

        base_row = cand_df.iloc[bi]
        role = {"B_key": "geoA" if bool(B_is_geoA[bi, j]) else "geoB", "A_key": "geoB" if bool(B_is_geoA[bi, j]) else "geoA", "H_key": "H"}
        ch = _choice_from_candidate_row(base_row, role)
        ch.update({
            "image_name": nm,
            "image_index": int(j),
            "fft_orientation_label": labels[j] if j < len(labels) else "unknown",
            "n_images_in_orientation_group": 1,
            "site_geometry_model": "intensity_per_image_graphite_triplet_gpu_vectorized" if use_gpu else "intensity_per_image_graphite_triplet_cpu_vectorized",
            "origin_definition": "per_image_intensity_grid_search_geometric_hollow",
            "site_coordinate_mode": "intensity_per_image_grid_search",
            "candidate_index": bi,
            "score": float(score[bi, j]),
            "median_I_B": float(B_vals[bi, j]),
            "median_I_A": float(A_vals[bi, j]),
            "median_I_H": float(vals_H[bi, j]),
            "median_delta_BA": float(d_ba[bi, j]),
            "median_delta_AH": float(d_ah[bi, j]),
            "B_gt_A_gt_H_fraction": float(ordered[bi, j]),
            "site_search_backend": _GPU_SITE_SEARCH_STATUS.get("backend", "unknown"),
            "site_pick_score_mode": INTENSITY_SITE_SCORE_MODE,
            "candidate_theta_deg": float(candidate_theta_deg[bi, j]),
            "legacy_ABH_balance_score": float(legacy_score[bi, j]) if SAVE_LEGACY_ABH_BALANCE_SCORE_FOR_DIAGNOSTIC else np.nan,
            "site_uncertainty_correction_enabled": bool(USE_SITE_PICKING_UNCERTAINTY_CORRECTION),
            "soft_coordinates_used_as_primary": bool(USE_SOFT_SITE_COORDINATES_AS_PRIMARY and USE_SITE_PICKING_UNCERTAINTY_CORRECTION),
        })
        ch.update(soft_summary)
        choices[nm] = ch

        # Save top candidates for this image so marker failures can be inspected.
        if INTENSITY_PER_IMAGE_SAVE_ALL_CANDIDATES:
            k_idx = np.argsort(score[:, j])[::-1]
        else:
            k = max(1, int(top_n or 1))
            k_idx = np.argpartition(score[:, j], -min(k, score.shape[0]))[-min(k, score.shape[0]):]
            k_idx = k_idx[np.argsort(score[k_idx, j])[::-1]]
        td = cand_df.iloc[k_idx].copy()
        td.insert(0, "name", nm)
        td.insert(1, "short_name", short_name(nm))
        td["site_coordinate_mode"] = "intensity_per_image_grid_search"
        td["per_image_intensity_detected"] = True
        td["candidate_rank_in_image"] = np.arange(1, len(td) + 1, dtype=int)
        td["score"] = score[k_idx, j]
        td["site_pick_score_mode"] = INTENSITY_SITE_SCORE_MODE
        td["candidate_theta_deg"] = candidate_theta_deg[k_idx, j]
        if SAVE_LEGACY_ABH_BALANCE_SCORE_FOR_DIAGNOSTIC:
            td["legacy_ABH_balance_score"] = legacy_score[k_idx, j]
        td["B_key"] = np.where(B_is_geoA[k_idx, j], "geoA", "geoB")
        td["A_key"] = np.where(B_is_geoA[k_idx, j], "geoB", "geoA")
        td["H_key"] = "H"
        td["median_I_B"] = B_vals[k_idx, j]
        td["median_I_A"] = A_vals[k_idx, j]
        td["median_I_H"] = vals_H[k_idx, j]
        td["median_delta_BA"] = d_ba[k_idx, j]
        td["median_delta_AH"] = d_ah[k_idx, j]
        td["B_gt_A_gt_H_fraction"] = ordered[k_idx, j].astype(float)
        td["fft_orientation_label"] = labels[j] if j < len(labels) else "unknown"
        td["site_search_backend"] = _GPU_SITE_SEARCH_STATUS.get("backend", "unknown")
        if USE_SITE_PICKING_UNCERTAINTY_CORRECTION:
            # Same weights as the soft summary, restricted to saved top candidates.
            best_score_j = float(np.nanmax(score[:, j]))
            rel_w = np.exp((score[k_idx, j] - best_score_j) / max(float(SITE_SOFT_SCORE_TEMPERATURE), 1.0e-9))
            rel_w = np.where(rel_w >= float(SITE_SOFT_MIN_REL_WEIGHT), rel_w, 0.0)
            sw = rel_w / rel_w.sum() if rel_w.sum() > 0 else np.zeros_like(rel_w)
            td["soft_candidate_weight_for_theta"] = sw
        top_rows.append(td)

    return choices, pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()


def grid_search_site_origin(maps_ref: np.ndarray) -> Tuple[Dict[str, object], pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    best_score = -np.inf
    grid = np.arange(ORIGIN_GRID_SEARCH_N, dtype=float) / ORIGIN_GRID_SEARCH_N
    for basis in CANDIDATE_BASES:
        for ou in grid:
            for ov in grid:
                pos = positions_from_origin_basis((float(ou), float(ov)), basis)
                vals = extract_candidate_site_values(maps_ref, pos)
                role = choose_role_assignment(vals)
                score = score_candidate_origin(vals, role)
                H_vals = vals["H"]
                B_vals = vals[role["B_key"]]
                A_vals = vals[role["A_key"]]
                row = {
                    "basis": basis,
                    "site_geometry_model": SITE_GEOMETRY_MODEL,
                    "origin_definition": "single_geometric_hollow",
                    "origin_u": float(ou),
                    "origin_v": float(ov),
                    "score": float(score),
                    "B_key": role["B_key"],
                    "A_key": role["A_key"],
                    "H_key": role["H_key"],
                    "median_I_B": float(np.nanmedian(B_vals)),
                    "median_I_A": float(np.nanmedian(A_vals)),
                    "median_I_H": float(np.nanmedian(H_vals)),
                    "median_delta_BA": float(np.nanmedian(B_vals - A_vals)),
                    "median_delta_AH": float(np.nanmedian(A_vals - H_vals)),
                    "B_gt_A_gt_H_fraction": float(np.nanmean((B_vals > A_vals) & (A_vals > H_vals))),
                }
                rows.append(row)
                if score > best_score:
                    best_score = score
                    best = {"basis": basis, "origin": (float(ou), float(ov)), "positions_geo": pos, "role": role, **row}
    if best is None:
        raise RuntimeError("Site origin grid search failed.")

    pos_geo = best["positions_geo"]
    role = best["role"]
    best["positions_role"] = {
        "B": pos_geo[role["B_key"]],
        "A": pos_geo[role["A_key"]],
        "H": pos_geo["H"],
        "geoA": pos_geo["geoA"],
        "geoB": pos_geo["geoB"],
        "geoH": pos_geo["H"],
    }
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def grid_search_site_origin_single_map(
    unit_map: np.ndarray,
    image_name: str = "",
    top_n: int = INTENSITY_PER_IMAGE_SITE_TOP_N,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Find A/B/H sites from one folded unit-cell map by intensity.

    This intentionally ignores FFT 60/120 grouping.  It tests both graphite
    triplet geometries over the full unit-cell origin grid.  H is the geometric
    hollow candidate; B and A are the brighter and intermediate atom candidates.
    The best candidate maximizes the B>A>H ordering score for this image.
    """
    m = np.asarray(unit_map, dtype=float)
    if m.ndim != 2:
        raise ValueError("unit_map must be a 2D folded unit-cell map")
    choice, df = grid_search_site_origin(m[None, :, :])
    df = df.copy()
    df.insert(0, "name", image_name)
    df.insert(1, "short_name", short_name(image_name) if image_name else "")
    df["site_coordinate_mode"] = "intensity_per_image_grid_search"
    df["per_image_intensity_detected"] = True
    df["candidate_rank_in_image"] = np.arange(1, len(df) + 1, dtype=int)
    if not INTENSITY_PER_IMAGE_SAVE_ALL_CANDIDATES and top_n is not None and top_n > 0:
        df = df.head(int(top_n)).copy()
    choice = dict(choice)
    choice["image_name"] = image_name
    choice["n_images_in_orientation_group"] = 1
    choice["site_geometry_model"] = "intensity_per_image_graphite_triplet"
    choice["origin_definition"] = "per_image_intensity_grid_search_geometric_hollow"
    choice["site_coordinate_mode"] = "intensity_per_image_grid_search"
    return choice, df


def fft_orientation_label_from_angle(angle_deg: float) -> str:
    """Classify image into 60deg or 120deg FFT geometry family.

    This uses only the raw reciprocal-peak angle reported by the FFT peak pair,
    before site intensities are sampled.  It therefore cannot push theta toward a
    B-dominant solution the way per-image intensity-based site fitting can.
    """
    try:
        a = float(angle_deg)
    except Exception:
        return "unknown"
    if not np.isfinite(a):
        return "unknown"
    return "fft60" if a < FFT_ORIENTATION_ANGLE_THRESHOLD_DEG else "fft120"


def basis_for_fft_orientation_label(label: str) -> str:
    if label == "fft60":
        return FFT_ORIENTATION_60_BASIS_NAME
    if label == "fft120":
        return FFT_ORIENTATION_120_BASIS_NAME
    return FFT_ORIENTATION_60_BASIS_NAME


def apply_manual_orientation_overrides(names: Sequence[str], labels: Sequence[str]) -> List[str]:
    out = list(labels)
    if not MANUAL_FFT_ORIENTATION_OVERRIDES:
        return out
    for i, nm in enumerate(names):
        for key, val in MANUAL_FFT_ORIENTATION_OVERRIDES.items():
            if str(key) in str(nm) and val in ("fft60", "fft120"):
                out[i] = val
                break
    return out


def manual_fixed_site_choice_for_orientation(label: str, n_images: int = 0) -> Dict[str, object]:
    """Return fixed A/B/H site positions for an FFT orientation group.

    This deliberately avoids any intensity-driven site-origin fitting.  It is the
    recommended mode when automatic fitting pushes theta toward B- or AH-dominant
    artificial distributions.
    """
    lab = label if label in MANUAL_ABH_SITES_BY_ORIENTATION else "fft60"
    spec = MANUAL_ABH_SITES_BY_ORIENTATION[lab]
    positions_role = {
        "B": mod1_pair(tuple(spec["B"])),
        "A": mod1_pair(tuple(spec["A"])),
        "H": mod1_pair(tuple(spec["H"])),
    }
    # geoA/geoB/geoH are retained for compatibility with existing output tables.
    positions_role["geoA"] = positions_role["A"]
    positions_role["geoB"] = positions_role["B"]
    positions_role["geoH"] = positions_role["H"]
    return {
        "fft_orientation_label": lab,
        "n_images_in_orientation_group": int(n_images),
        "basis": str(spec.get("basis", f"manual_{lab}")),
        "site_geometry_model": "manual_fixed_ABH_by_fft_orientation",
        "origin_definition": "manual_fixed_site_coordinates_not_intensity_fitted",
        "origin": positions_role["H"],
        "positions_role": positions_role,
        "positions_geo": {"geoA": positions_role["A"], "geoB": positions_role["B"], "H": positions_role["H"]},
        "role": {"B_key": "B_manual", "A_key": "A_manual", "H_key": "H_manual"},
        "score": np.nan,
        "median_delta_BA": np.nan,
        "median_delta_AH": np.nan,
    }


def grid_search_site_origin_for_bases(
    maps_ref: np.ndarray,
    allowed_bases: Sequence[str],
    group_label: str = "",
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """One global site-origin calibration restricted to a basis family."""
    rows: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    best_score = -np.inf
    bases = [b for b in allowed_bases if b in CANDIDATE_BASES]
    if not bases:
        bases = list(CANDIDATE_BASES.keys())
    grid = np.arange(ORIGIN_GRID_SEARCH_N, dtype=float) / ORIGIN_GRID_SEARCH_N
    for basis in bases:
        for ou in grid:
            for ov in grid:
                pos = positions_from_origin_basis((float(ou), float(ov)), basis)
                vals = extract_candidate_site_values(maps_ref, pos)
                role = choose_role_assignment(vals)
                score = score_candidate_origin(vals, role)
                H_vals = vals["H"]
                B_vals = vals[role["B_key"]]
                A_vals = vals[role["A_key"]]
                row = {
                    "fft_orientation_label": group_label,
                    "basis": basis,
                    "site_geometry_model": SITE_GEOMETRY_MODEL,
                    "origin_definition": "single_geometric_hollow_group_global",
                    "origin_u": float(ou),
                    "origin_v": float(ov),
                    "score": float(score),
                    "B_key": role["B_key"],
                    "A_key": role["A_key"],
                    "H_key": role["H_key"],
                    "median_I_B": float(np.nanmedian(B_vals)),
                    "median_I_A": float(np.nanmedian(A_vals)),
                    "median_I_H": float(np.nanmedian(H_vals)),
                    "median_delta_BA": float(np.nanmedian(B_vals - A_vals)),
                    "median_delta_AH": float(np.nanmedian(A_vals - H_vals)),
                    "B_gt_A_gt_H_fraction": float(np.nanmean((B_vals > A_vals) & (A_vals > H_vals))),
                }
                rows.append(row)
                if score > best_score:
                    best_score = score
                    best = {"basis": basis, "origin": (float(ou), float(ov)), "positions_geo": pos, "role": role, **row}
    if best is None:
        raise RuntimeError(f"Site origin grid search failed for group {group_label}.")
    pos_geo = best["positions_geo"]
    role = best["role"]
    best["positions_role"] = {
        "B": pos_geo[role["B_key"]],
        "A": pos_geo[role["A_key"]],
        "H": pos_geo["H"],
        "geoA": pos_geo["geoA"],
        "geoB": pos_geo["geoB"],
        "geoH": pos_geo["H"],
    }
    return best, pd.DataFrame(rows).sort_values("score", ascending=False)


def extract_role_site_intensities(map_arr: np.ndarray, positions_role: Dict[str, Optional[Tuple[float, float]]]) -> Dict[str, float]:
    IB = site_mean(map_arr, positions_role["B"])
    IA = site_mean(map_arr, positions_role["A"])
    IH = site_mean(map_arr, positions_role["H"])
    return {"I_A": IA, "I_B": IB, "I_H": IH}


def motif_coordinates_from_sites(IA: float, IB: float, IH: float) -> Dict[str, float]:
    # These projections are invariant to adding a constant baseline to all three
    # site intensities because the basis coefficients sum to zero.
    c_AH = (IA + IB - 2.0 * IH) / math.sqrt(6.0)
    c_beta = (IB - IA) / math.sqrt(2.0)
    abs_c_beta = abs(c_beta)
    R = math.sqrt(c_AH ** 2 + c_beta ** 2)
    theta = math.degrees(math.atan2(abs_c_beta, c_AH)) if np.isfinite(c_AH) and np.isfinite(abs_c_beta) else np.nan
    delta_BA = IB - IA
    delta_AH = IA - IH
    R_flat = math.sqrt(delta_BA ** 2 + delta_AH ** 2)
    theta_flat = math.degrees(math.atan2(delta_AH, delta_BA)) if np.isfinite(delta_BA) and np.isfinite(delta_AH) else np.nan
    return {
        "c_AH": c_AH,
        "c_beta_signed": c_beta,
        "abs_c_beta": abs_c_beta,
        "R_orthogonal": R,
        "theta_beta_over_AH_deg": theta,
        "delta_BA": delta_BA,
        "delta_AH": delta_AH,
        "R_flat_BA_AH": R_flat,
        "theta_AH_over_BA_deg": theta_flat,
    }


def _periodic_delta_frac(a: float, b: float) -> float:
    return float(((float(a) - float(b) + 0.5) % 1.0) - 0.5)


def _hex120_distance_frac(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    du = _periodic_delta_frac(p[0], q[0])
    dv = _periodic_delta_frac(p[1], q[1])
    d2 = du * du + dv * dv - du * dv
    return float(math.sqrt(max(0.0, d2)))


def refine_A_site_local(
    map_arr: np.ndarray,
    positions_role: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, object]]:
    """Refine only the A marker while keeping B and H fixed.

    This is intended for the empirically common case where B/H are visually
    reliable but A is slightly displaced by residual drift or local scan
    distortion.  The local score deliberately avoids an ABH-balance reward; it
    uses local A-site peakness plus B>A>H ordering and a small displacement prior.
    """
    pos0 = {k: tuple(v) for k, v in positions_role.items()}
    if not REFINE_A_SITE_LOCAL_AFTER_PICK:
        return pos0, {"A_refinement_enabled": False, "A_refinement_accepted": False}

    A0 = tuple(pos0["A"])
    B = tuple(pos0["B"])
    H = tuple(pos0["H"])
    radius = float(A_REFINEMENT_RADIUS_FRAC)
    nsteps = int(A_REFINEMENT_GRID_STEPS)
    if nsteps < 3 or radius <= 0:
        return pos0, {"A_refinement_enabled": True, "A_refinement_accepted": False, "A_refinement_reason": "invalid_settings"}

    # Fixed B/H intensities; use the normal site-average width for final ordering.
    IB = site_mean(map_arr, B, SITE_AVERAGE_SIGMA_FRAC)
    IH = site_mean(map_arr, H, SITE_AVERAGE_SIGMA_FRAC)
    IA0 = site_mean(map_arr, A0, SITE_AVERAGE_SIGMA_FRAC)
    coords0 = motif_coordinates_from_sites(IA0, IB, IH)
    theta0 = coords0.get("theta_beta_over_AH_deg", np.nan)

    offsets = np.linspace(-radius, radius, nsteps)
    cand_rows: List[Dict[str, float]] = []
    best_row: Optional[Dict[str, float]] = None
    best_score = -np.inf
    origin_score = np.nan

    for du in offsets:
        for dv in offsets:
            d_hex = math.sqrt(max(0.0, du * du + dv * dv - du * dv))
            if d_hex > radius * 1.0001:
                continue
            A = ((A0[0] + float(du)) % 1.0, (A0[1] + float(dv)) % 1.0)
            IA = site_mean(map_arr, A, SITE_AVERAGE_SIGMA_FRAC)
            IA_sharp = site_mean(map_arr, A, A_REFINEMENT_SIGMA_SHARP_FRAC)
            IA_broad = site_mean(map_arr, A, A_REFINEMENT_SIGMA_BROAD_FRAC)
            peakness = IA_sharp - IA_broad
            d_ba = IB - IA
            d_ah = IA - IH
            order_pen = max(0.0, -d_ba) ** 2 + max(0.0, -d_ah) ** 2
            prior_pen = (d_hex / max(radius, 1.0e-12)) ** 2
            # Theta-neutral: no min(dBA,dAH), no midpoint reward.
            score = (
                peakness
                + float(A_REFINEMENT_INTENSITY_WEIGHT) * IA
                - float(A_REFINEMENT_ORDER_PENALTY_WEIGHT) * order_pen
                - float(A_REFINEMENT_PRIOR_WEIGHT) * prior_pen
            )
            coords = motif_coordinates_from_sites(IA, IB, IH)
            row = {
                "A_u": A[0], "A_v": A[1],
                "du_from_initial": float(du), "dv_from_initial": float(dv),
                "shift_hex120_frac": float(d_hex),
                "I_A": float(IA), "I_A_sharp": float(IA_sharp), "I_A_broad": float(IA_broad),
                "A_peakness": float(peakness),
                "delta_BA": float(d_ba), "delta_AH": float(d_ah),
                "theta_deg": float(coords.get("theta_beta_over_AH_deg", np.nan)),
                "score": float(score),
                "ordered": float((d_ba > 0) and (d_ah > 0)),
            }
            cand_rows.append(row)
            if abs(du) < 1.0e-12 and abs(dv) < 1.0e-12:
                origin_score = float(score)
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_row = row

    if best_row is None:
        return pos0, {"A_refinement_enabled": True, "A_refinement_accepted": False, "A_refinement_reason": "no_candidates"}
    if not np.isfinite(origin_score):
        # Compute original-score fallback from the selected candidate nearest A0.
        origin_score = float(min(cand_rows, key=lambda r: r["shift_hex120_frac"])["score"])

    theta_best = float(best_row.get("theta_deg", np.nan))
    theta_change = theta_best - theta0 if np.isfinite(theta_best) and np.isfinite(theta0) else np.nan
    orig_ordered = bool((IB - IA0 > 0) and (IA0 - IH > 0))
    best_ordered = bool(best_row.get("ordered", 0.0) > 0.5)
    score_improvement = float(best_score - origin_score)

    accept = bool(score_improvement >= float(A_REFINEMENT_MIN_SCORE_IMPROVEMENT))
    if A_REFINEMENT_ACCEPT_IF_ORDER_IMPROVES and (not orig_ordered) and best_ordered:
        accept = True
    if np.isfinite(theta_change) and abs(theta_change) > float(A_REFINEMENT_MAX_THETA_CHANGE_DEG):
        # Very large theta jumps are more likely a bad A choice; keep original but flag.
        accept = False

    pos_ref = dict(pos0)
    if accept:
        pos_ref["A"] = (float(best_row["A_u"]), float(best_row["A_v"]))
        # Preserve geoA/geoB/H geometry for diagnostics; only role A is locally refined.

    # Top local candidates help diagnose ambiguous A choices.
    cand_sorted = sorted(cand_rows, key=lambda r: r["score"], reverse=True)
    top_scores = [r["score"] for r in cand_sorted[:5]]
    diag: Dict[str, object] = {
        "A_refinement_enabled": True,
        "A_refinement_accepted": bool(accept),
        "A_refinement_reason": "accepted" if accept else "kept_initial",
        "A_initial_u": A0[0], "A_initial_v": A0[1],
        "A_refined_u": pos_ref["A"][0], "A_refined_v": pos_ref["A"][1],
        "A_refinement_shift_hex120_frac": _hex120_distance_frac(pos_ref["A"], A0),
        "A_refinement_score_initial": origin_score,
        "A_refinement_score_best": best_score,
        "A_refinement_score_improvement": score_improvement,
        "A_refinement_I_A_initial": IA0,
        "A_refinement_I_A_best": best_row.get("I_A", np.nan),
        "A_refinement_delta_BA_initial": IB - IA0,
        "A_refinement_delta_AH_initial": IA0 - IH,
        "A_refinement_delta_BA_best": best_row.get("delta_BA", np.nan),
        "A_refinement_delta_AH_best": best_row.get("delta_AH", np.nan),
        "A_refinement_theta_initial_deg": theta0,
        "A_refinement_theta_best_deg": theta_best,
        "A_refinement_theta_change_deg": theta_change,
        "A_refinement_initial_ordered": bool(orig_ordered),
        "A_refinement_best_ordered": bool(best_ordered),
        "A_refinement_top_score_gap_1_2": float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else np.nan,
        "A_refinement_n_candidates": int(len(cand_rows)),
    }
    return pos_ref, diag


def _weighted_mean_and_sd(vals: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan, np.nan
    v = vals[mask]
    w = weights[mask]
    w = w / np.sum(w)
    mu = float(np.sum(w * v))
    var = float(np.sum(w * (v - mu) ** 2))
    return mu, math.sqrt(max(0.0, var))


def _weighted_quantile(vals: np.ndarray, weights: np.ndarray, q: float) -> float:
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    v = vals[mask]
    w = weights[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w) / np.sum(w)
    return float(np.interp(float(q), cw, v))


def _soft_candidate_motif_summary(
    score_vec: np.ndarray,
    B_vec: np.ndarray,
    A_vec: np.ndarray,
    H_vec: np.ndarray,
    candidate_indices: np.ndarray,
) -> Dict[str, float]:
    """Score-weighted motif estimate over plausible A/B/H candidates.

    This is a correction for the site-picking bias of a hard argmax.  It keeps
    the best candidate for visualization, but reports the primary normalized
    coordinates as an uncertainty-aware weighted estimate when enabled.
    """
    idx = np.asarray(candidate_indices, dtype=int)
    if idx.size == 0:
        return {}
    s = np.asarray(score_vec[idx], dtype=float)
    B = np.asarray(B_vec[idx], dtype=float)
    A = np.asarray(A_vec[idx], dtype=float)
    H = np.asarray(H_vec[idx], dtype=float)
    finite = np.isfinite(s) & np.isfinite(B) & np.isfinite(A) & np.isfinite(H)
    if not np.any(finite):
        return {}
    idx = idx[finite]
    s = s[finite]
    B = B[finite]
    A = A[finite]
    H = H[finite]

    best = float(np.nanmax(s))
    temp = max(float(SITE_SOFT_SCORE_TEMPERATURE), 1.0e-9)
    rel = np.exp((s - best) / temp)
    if SITE_SOFT_MIN_REL_WEIGHT is not None and SITE_SOFT_MIN_REL_WEIGHT > 0:
        keep = rel >= float(SITE_SOFT_MIN_REL_WEIGHT)
        if np.any(keep):
            idx, s, B, A, H, rel = idx[keep], s[keep], B[keep], A[keep], H[keep], rel[keep]
    w = rel / np.sum(rel)

    # Weighted site intensities, then physically consistent motif coordinates.
    IB = float(np.sum(w * B))
    IA = float(np.sum(w * A))
    IH = float(np.sum(w * H))
    coords = motif_coordinates_from_sites(IA, IB, IH)

    # Candidate-level coordinate spread for uncertainty/ambiguity diagnostics.
    cAH_c = (A + B - 2.0 * H) / math.sqrt(6.0)
    cb_c = (B - A) / math.sqrt(2.0)
    acb_c = np.abs(cb_c)
    theta_c = np.degrees(np.arctan2(acb_c, cAH_c))
    R_c = np.sqrt(cAH_c ** 2 + cb_c ** 2)
    dBA_c = B - A
    dAH_c = A - H

    theta_mu, theta_sd = _weighted_mean_and_sd(theta_c, w)
    cAH_mu, cAH_sd = _weighted_mean_and_sd(cAH_c, w)
    acb_mu, acb_sd = _weighted_mean_and_sd(acb_c, w)
    R_mu, R_sd = _weighted_mean_and_sd(R_c, w)
    dBA_mu, dBA_sd = _weighted_mean_and_sd(dBA_c, w)
    dAH_mu, dAH_sd = _weighted_mean_and_sd(dAH_c, w)

    score_sorted = np.sort(s)[::-1]
    score_gap12 = float(score_sorted[0] - score_sorted[1]) if score_sorted.size > 1 else np.nan
    neff = float(1.0 / np.sum(w ** 2))
    entropy = float(-np.sum(w * np.log(np.maximum(w, 1e-300))))
    theta_q05 = _weighted_quantile(theta_c, w, 0.05)
    theta_q25 = _weighted_quantile(theta_c, w, 0.25)
    theta_q50 = _weighted_quantile(theta_c, w, 0.50)
    theta_q75 = _weighted_quantile(theta_c, w, 0.75)
    theta_q95 = _weighted_quantile(theta_c, w, 0.95)
    theta_iqr = theta_q75 - theta_q25 if np.isfinite(theta_q75) and np.isfinite(theta_q25) else np.nan

    out = {
        "site_soft_candidate_count": int(idx.size),
        "site_soft_neff": neff,
        "site_soft_entropy": entropy,
        "site_soft_score_gap_1_2": score_gap12,
        "site_soft_score_best": best,
        "site_soft_score_temperature": float(SITE_SOFT_SCORE_TEMPERATURE),
        "site_soft_theta_q05": theta_q05,
        "site_soft_theta_q25": theta_q25,
        "site_soft_theta_q50": theta_q50,
        "site_soft_theta_q75": theta_q75,
        "site_soft_theta_q95": theta_q95,
        "site_soft_theta_iqr": theta_iqr,
        "site_soft_theta_sd": theta_sd,
        "site_soft_c_AH_sd": cAH_sd,
        "site_soft_abs_c_beta_sd": acb_sd,
        "site_soft_R_sd": R_sd,
        "site_soft_delta_BA_sd": dBA_sd,
        "site_soft_delta_AH_sd": dAH_sd,
        "site_soft_ambiguous": bool(
            (np.isfinite(neff) and neff >= SITE_AMBIGUITY_NEFF_CAUTION)
            or (np.isfinite(theta_iqr) and theta_iqr >= SITE_AMBIGUITY_THETA_IQR_CAUTION_DEG)
            or (np.isfinite(score_gap12) and score_gap12 <= SITE_SCORE_GAP_CAUTION)
        ),
        "soft_I_B_norm": IB,
        "soft_I_A_norm": IA,
        "soft_I_H_norm": IH,
        "soft_c_AH_norm_direct": coords["c_AH"],
        "soft_c_beta_signed_norm_direct": coords["c_beta_signed"],
        "soft_abs_c_beta_norm_direct": coords["abs_c_beta"],
        "soft_R_orthogonal_norm_direct": coords["R_orthogonal"],
        "soft_theta_beta_over_AH_deg_norm_direct": coords["theta_beta_over_AH_deg"],
        "soft_delta_BA_norm_direct": coords["delta_BA"],
        "soft_delta_AH_norm_direct": coords["delta_AH"],
        "soft_R_flat_BA_AH_norm_direct": coords["R_flat_BA_AH"],
        "soft_theta_AH_over_BA_deg_norm_direct": coords["theta_AH_over_BA_deg"],
        # weighted candidate means are diagnostic, direct site-weighted values above are primary
        "site_soft_theta_weighted_candidate_mean": theta_mu,
        "site_soft_c_AH_weighted_candidate_mean": cAH_mu,
        "site_soft_abs_c_beta_weighted_candidate_mean": acb_mu,
        "site_soft_R_weighted_candidate_mean": R_mu,
        "site_soft_delta_BA_weighted_candidate_mean": dBA_mu,
        "site_soft_delta_AH_weighted_candidate_mean": dAH_mu,
    }
    return out


# =============================================================================
# Bootstrap uncertainty
# =============================================================================


def weighted_bootstrap_means(map_arr: np.ndarray, pos: Tuple[float, float], nboot: int, rng: np.random.Generator) -> np.ndarray:
    w = periodic_gaussian_weight(map_arr.shape[0], pos)
    flat_w = w.ravel()
    flat_w = flat_w / flat_w.sum()
    values = map_arr.ravel()
    n_eff = int(max(20, min(values.size, round(1.0 / np.sum(flat_w ** 2)))))
    idx = np.arange(values.size)
    out = np.empty(nboot, dtype=float)
    for b in range(nboot):
        sampled = rng.choice(idx, size=n_eff, replace=True, p=flat_w)
        out[b] = float(np.nanmean(values[sampled]))
    return out


def bootstrap_site_coordinate_ci(name: str, map_arr: np.ndarray, positions_role: Dict[str, Optional[Tuple[float, float]]], rng: np.random.Generator) -> Dict[str, float]:
    if not BOOTSTRAP_SITE_CI or BOOTSTRAP_N <= 0:
        return {"name": name}
    IBs = weighted_bootstrap_means(map_arr, positions_role["B"], BOOTSTRAP_N, rng)
    IAs = weighted_bootstrap_means(map_arr, positions_role["A"], BOOTSTRAP_N, rng)
    IHs = weighted_bootstrap_means(map_arr, positions_role["H"], BOOTSTRAP_N, rng)

    cAH = (IAs + IBs - 2.0 * IHs) / math.sqrt(6.0)
    cb = (IBs - IAs) / math.sqrt(2.0)
    acb = np.abs(cb)
    dBA = IBs - IAs
    dAH = IAs - IHs
    R = np.sqrt(cAH ** 2 + cb ** 2)
    theta = np.degrees(np.arctan2(acb, cAH))
    out = {"name": name}
    for key, vals in {
        "I_A_norm": IAs, "I_B_norm": IBs, "I_H_norm": IHs,
        "delta_BA_norm": dBA, "delta_AH_norm": dAH,
        "c_AH_norm": cAH, "c_beta_signed_norm": cb, "abs_c_beta_norm": acb,
        "R_orthogonal_norm": R, "theta_beta_over_AH_deg_norm": theta,
    }.items():
        out[f"{key}_ci95_low"] = float(np.nanpercentile(vals, 2.5))
        out[f"{key}_ci95_high"] = float(np.nanpercentile(vals, 97.5))
        out[f"{key}_boot_sd"] = float(np.nanstd(vals))
    return out


def bootstrap_group_summary(df: pd.DataFrame, cols: Sequence[str], nboot: int = GROUP_BOOTSTRAP_N) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED + 101)
    rows = []
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    for col in cols:
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        boots = np.empty(nboot, dtype=float)
        for b in range(nboot):
            boots[b] = float(np.nanmean(rng.choice(vals, size=vals.size, replace=True)))
        rows.append({
            "metric": col,
            "n": int(vals.size),
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan,
            "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
            "mean_ci95_low": float(np.nanpercentile(boots, 2.5)),
            "mean_ci95_high": float(np.nanpercentile(boots, 97.5)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Whole-map statistics
# =============================================================================


def pca_from_maps(maps: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    n = maps.shape[0]
    X = maps.reshape(n, -1)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    denom = max(1, n - 1)
    var = (S ** 2) / denom
    ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    scores = U * S[None, :]
    expl = pd.DataFrame({
        "PC": np.arange(1, len(ratio) + 1),
        "explained_variance_ratio": ratio,
        "cumulative": np.cumsum(ratio),
        "singular_value": S,
    })
    score_df = pd.DataFrame(scores[:, :min(8, scores.shape[1])], columns=[f"PC{k}" for k in range(1, min(8, scores.shape[1]) + 1)])
    return expl, score_df, Vt


def pairwise_corr_distance(maps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = maps.shape[0]
    X = maps.reshape(n, -1)
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0] = 1.0
    Xn = X / norms[:, None]
    corr = np.clip(Xn @ Xn.T, -1.0, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return corr, dist


def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    pos_vals = np.maximum(vals[:n_components], 0.0)
    return vecs[:, :n_components] * np.sqrt(pos_vals)[None, :]


def physical_soft_membership_prototypes() -> Dict[str, np.ndarray]:
    """Prototype directions in orthogonal (c_AH, |c_beta|) coordinates.

    The directions are derived from idealized site-intensity limits:
      AH/contact-registry: A=B>H -> theta=0 deg.
      beta-polarized/B-only-like: B>A≈H -> theta=60 deg.
      ABH hybrid: B>A>H with approximately even B-A and A-H spacing -> theta=30 deg.

    These prototypes are descriptive reference directions, not hard classes.
    """
    return {
        "AH_contact_registry": np.array([1.0, 0.0], dtype=float),
        "ABH_hybrid": np.array([math.sqrt(3.0) / 2.0, 0.5], dtype=float),
        "beta_polarized": np.array([0.5, math.sqrt(3.0) / 2.0], dtype=float),
    }


def soft_membership_prototype_table(radial_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for label, p in physical_soft_membership_prototypes().items():
        rows.append({
            "prototype": label,
            "mode": SOFT_PROTOTYPE_MODE,
            "c_AH_scaled": float(p[0]),
            "abs_c_beta_scaled": float(p[1]),
            "theta_deg": float(np.degrees(np.arctan2(p[1], p[0]))),
            "radial_scale_used": float(radial_scale),
            "interpretation": {
                "AH_contact_registry": "A≈B above geometric hollow; contact-registry/atom-vs-hollow endpoint direction.",
                "ABH_hybrid": "B>A>H with both coordinates finite; evenly spaced A/B/H role intensities.",
                "beta_polarized": "B-only-like sublattice-polarized limit B>A≈H; c_AH remains positive.",
            }.get(label, ""),
        })
    return pd.DataFrame(rows)


def soft_membership(df: pd.DataFrame, xcol: str = "c_AH_norm", ycol: str = "abs_c_beta_norm") -> pd.DataFrame:
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    # Use one common radial scale so motif angles and physical-limit prototype
    # directions are preserved. Independent x/y scaling would move the prototype
    # directions and can make beta-only appear at the unphysical (0,1) limit.
    r = np.sqrt(x ** 2 + y ** 2)
    s = np.nanpercentile(r[np.isfinite(r)], 95) if np.isfinite(r).any() else 1.0
    s = s if np.isfinite(s) and s > 0 else 1.0
    X = np.column_stack([x / s, y / s])
    protos = physical_soft_membership_prototypes()
    d2 = np.column_stack([np.sum((X - p[None, :]) ** 2, axis=1) for p in protos.values()])
    logits = -d2 / (2.0 * SOFT_SIGMA ** 2)
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P = P / P.sum(axis=1, keepdims=True)
    out = pd.DataFrame({"name": df["name"].values})
    for idx, k in enumerate(protos.keys()):
        out[f"P_{k}"] = P[:, idx]
    order = np.argsort(P, axis=1)[:, ::-1]
    cols = list(protos.keys())
    out["soft_label_top"] = [cols[i] for i in order[:, 0]]
    out["soft_label_second"] = [cols[i] for i in order[:, 1]]
    out["soft_margin_top_minus_second"] = P[np.arange(len(P)), order[:, 0]] - P[np.arange(len(P)), order[:, 1]]
    out["soft_entropy"] = -np.sum(P * np.log(P + 1e-12), axis=1)
    out["soft_ambiguous"] = out["soft_margin_top_minus_second"] < SOFT_AMBIGUITY_MARGIN
    out["soft_scale_radial_95pct"] = s
    out["soft_scale_mode"] = SOFT_SCALE_MODE
    out["soft_prototype_mode"] = SOFT_PROTOTYPE_MODE
    return out


# =============================================================================
# Plot functions
# =============================================================================




def _safe_filename_component(text_value: object, max_len: int = 80) -> str:
    s = str(text_value)
    s = re.sub(r"[^0-9A-Za-z가-힣_.+-]+", "_", s).strip("_")
    if not s:
        s = "unnamed"
    return s[:max_len]


def _export_current_figure_data_for_origin(fig: plt.Figure, image_path: Path) -> None:
    """Export numeric data behind a Matplotlib figure for re-plotting in Origin.

    The function is intentionally generic and is called by savefig().  It exports
    the objects that are most commonly used in this script:
      - scatter/PathCollection: x, y, optional color c, marker size
      - line/reference/trend: x, y
      - histogram/bar rectangles: center, height, left/right/bottom/top
      - imshow: 2D matrix
      - pcolormesh/QuadMesh: z matrix, x/y edge matrices, long-format xyz centers

    KDE contours are usually stored by Matplotlib as polygon collections without
    their original evaluation grid. For those plots, the exported scatter CSV is
    the safest Origin input; Origin can regenerate density contours from x/y.
    """
    if not globals().get("EXPORT_ORIGIN_CSV_FOR_EACH_FIGURE", False):
        return
    try:
        image_path = Path(image_path)
        origin_root = image_path.parent / globals().get("ORIGIN_PLOT_DATA_FOLDER_NAME", "origin_plot_data")
        fig_dir = origin_root / image_path.stem
        fig_dir.mkdir(parents=True, exist_ok=True)

        index_rows: List[Dict[str, object]] = []
        axes_rows: List[Dict[str, object]] = []

        for ax_i, ax in enumerate(fig.axes):
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            title = ax.get_title()
            try:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
            except Exception:
                xlim = (np.nan, np.nan)
                ylim = (np.nan, np.nan)
            axes_rows.append({
                "figure": image_path.name,
                "axis_index": ax_i,
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "xlim_min": xlim[0],
                "xlim_max": xlim[1],
                "ylim_min": ylim[0],
                "ylim_max": ylim[1],
            })

            # imshow images.
            for im_i, im in enumerate(ax.get_images()):
                try:
                    arr = np.asarray(im.get_array(), dtype=float)
                    if arr.size == 0:
                        continue
                    fname = f"ax{ax_i:02d}_image{im_i:02d}_{_safe_filename_component(title)}.csv"
                    pd.DataFrame(arr).to_csv(fig_dir / fname, index=False, header=False, encoding="utf-8-sig")
                    index_rows.append({
                        "figure": image_path.name,
                        "axis_index": ax_i,
                        "data_type": "image_2d_matrix",
                        "file": fname,
                        "title": title,
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                        "n_rows": int(arr.shape[0]) if arr.ndim else 1,
                        "n_cols": int(arr.shape[1]) if arr.ndim > 1 else 1,
                    })
                except Exception:
                    continue

            # pcolormesh / QuadMesh heatmaps, including graphite 60° unit-cell plots.
            for qm_i, qm in enumerate(ax.collections):
                try:
                    if qm.__class__.__name__ != "QuadMesh" or not hasattr(qm, "get_coordinates"):
                        continue
                    coords = np.asarray(qm.get_coordinates(), dtype=float)
                    if coords.ndim != 3 or coords.shape[-1] != 2:
                        continue
                    z = np.asarray(qm.get_array(), dtype=float)
                    target_shape = (coords.shape[0] - 1, coords.shape[1] - 1)
                    if z.size == int(np.prod(target_shape)):
                        z2 = z.reshape(target_shape)
                    elif z.shape == target_shape:
                        z2 = z
                    else:
                        continue
                    stem = f"ax{ax_i:02d}_quadmesh{qm_i:02d}_{_safe_filename_component(title)}"
                    z_file = f"{stem}_z.csv"
                    x_file = f"{stem}_x_edges.csv"
                    y_file = f"{stem}_y_edges.csv"
                    xyz_file = f"{stem}_xyz_centers.csv"

                    x_edges = coords[:, :, 0]
                    y_edges = coords[:, :, 1]
                    pd.DataFrame(z2).to_csv(fig_dir / z_file, index=False, header=False, encoding="utf-8-sig")
                    pd.DataFrame(x_edges).to_csv(fig_dir / x_file, index=False, header=False, encoding="utf-8-sig")
                    pd.DataFrame(y_edges).to_csv(fig_dir / y_file, index=False, header=False, encoding="utf-8-sig")

                    x_centers = 0.25 * (x_edges[:-1, :-1] + x_edges[1:, :-1] + x_edges[:-1, 1:] + x_edges[1:, 1:])
                    y_centers = 0.25 * (y_edges[:-1, :-1] + y_edges[1:, :-1] + y_edges[:-1, 1:] + y_edges[1:, 1:])
                    pd.DataFrame({"x": x_centers.ravel(), "y": y_centers.ravel(), "z": z2.ravel()}).to_csv(
                        fig_dir / xyz_file, index=False, encoding="utf-8-sig"
                    )
                    index_rows.append({
                        "figure": image_path.name,
                        "axis_index": ax_i,
                        "data_type": "quadmesh_2d",
                        "file": z_file,
                        "x_edges_file": x_file,
                        "y_edges_file": y_file,
                        "xyz_centers_file": xyz_file,
                        "title": title,
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                        "n_rows": int(z2.shape[0]),
                        "n_cols": int(z2.shape[1]),
                    })
                except Exception:
                    continue

            # Lines including reference/trend lines.
            for line_i, line in enumerate(ax.get_lines()):
                try:
                    x = np.asarray(line.get_xdata(), dtype=float)
                    y = np.asarray(line.get_ydata(), dtype=float)
                    if x.size == 0 or y.size == 0:
                        continue
                    n = min(x.size, y.size)
                    label = line.get_label()
                    fname = f"ax{ax_i:02d}_line{line_i:02d}_{_safe_filename_component(label)}.csv"
                    pd.DataFrame({"x": x[:n], "y": y[:n]}).to_csv(fig_dir / fname, index=False, encoding="utf-8-sig")
                    index_rows.append({
                        "figure": image_path.name,
                        "axis_index": ax_i,
                        "data_type": "line_xy",
                        "file": fname,
                        "label": label,
                        "title": title,
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                        "n_points": int(n),
                    })
                except Exception:
                    continue

            # Scatter points: only true PathCollections, to avoid KDE contour polygons.
            for col_i, col in enumerate(ax.collections):
                try:
                    if col.__class__.__name__ != "PathCollection" or not hasattr(col, "get_offsets"):
                        continue
                    offsets = np.asarray(col.get_offsets(), dtype=float)
                    if offsets.ndim != 2 or offsets.shape[1] < 2 or offsets.shape[0] < 1:
                        continue
                    finite_offsets = np.isfinite(offsets[:, 0]) & np.isfinite(offsets[:, 1])
                    if finite_offsets.sum() < 1:
                        continue
                    data: Dict[str, object] = {"x": offsets[:, 0], "y": offsets[:, 1]}
                    try:
                        arr = col.get_array()
                        if arr is not None:
                            arr = np.asarray(arr, dtype=float)
                            if arr.size == offsets.shape[0]:
                                data["c"] = arr
                    except Exception:
                        pass
                    try:
                        sizes = np.asarray(col.get_sizes(), dtype=float)
                        if sizes.size == offsets.shape[0]:
                            data["marker_size"] = sizes
                    except Exception:
                        pass
                    label = col.get_label()
                    fname = f"ax{ax_i:02d}_scatter{col_i:02d}_{_safe_filename_component(label)}.csv"
                    pd.DataFrame(data).to_csv(fig_dir / fname, index=False, encoding="utf-8-sig")
                    index_rows.append({
                        "figure": image_path.name,
                        "axis_index": ax_i,
                        "data_type": "scatter_xy",
                        "file": fname,
                        "label": label,
                        "title": title,
                        "xlabel": xlabel,
                        "ylabel": ylabel,
                        "n_points": int(offsets.shape[0]),
                        "has_color_value_c": "c" in data,
                    })
                except Exception:
                    continue

            # Histograms and bar-like rectangles.
            bar_rows = []
            for patch_i, patch in enumerate(ax.patches):
                try:
                    if patch is ax.patch:
                        continue
                    if not all(hasattr(patch, attr) for attr in ["get_x", "get_y", "get_width", "get_height"]):
                        continue
                    x0 = float(patch.get_x())
                    y0 = float(patch.get_y())
                    w = float(patch.get_width())
                    h = float(patch.get_height())
                    if not np.all(np.isfinite([x0, y0, w, h])):
                        continue
                    if w == 0 and h == 0:
                        continue
                    bar_rows.append({
                        "patch_index": patch_i,
                        "left": x0,
                        "right": x0 + w,
                        "center": x0 + 0.5 * w,
                        "width": w,
                        "bottom": y0,
                        "height": h,
                        "top": y0 + h,
                    })
                except Exception:
                    continue
            if bar_rows:
                fname = f"ax{ax_i:02d}_bars_hist_rectangles.csv"
                pd.DataFrame(bar_rows).to_csv(fig_dir / fname, index=False, encoding="utf-8-sig")
                index_rows.append({
                    "figure": image_path.name,
                    "axis_index": ax_i,
                    "data_type": "bars_or_histogram_rectangles",
                    "file": fname,
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "n_bars": len(bar_rows),
                })

        pd.DataFrame(axes_rows).to_csv(fig_dir / "axes_metadata.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(index_rows).to_csv(fig_dir / "figure_data_index.csv", index=False, encoding="utf-8-sig")

        if index_rows:
            origin_root.mkdir(parents=True, exist_ok=True)
            global_index = origin_root / "origin_plot_data_index.csv"
            idx_df = pd.DataFrame(index_rows)
            idx_df.insert(0, "figure_stem", image_path.stem)
            idx_df.insert(1, "figure_data_folder", str(fig_dir.name))
            if global_index.exists():
                try:
                    old = pd.read_csv(global_index)
                    idx_df = pd.concat([old, idx_df], ignore_index=True)
                except Exception:
                    pass
            idx_df.drop_duplicates(subset=["figure_stem", "file", "axis_index", "data_type"], keep="last").to_csv(
                global_index, index=False, encoding="utf-8-sig"
            )
    except Exception as exc:
        warnings.warn(f"Origin CSV export failed for {image_path.name}: {exc}")


def savefig(path: Path) -> None:
    try:
        plt.tight_layout()
    except Exception:
        pass
    fig = plt.gcf()
    _export_current_figure_data_for_origin(fig, Path(path))
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()



def _ensure_raw_distribution_dir(out_dir: Path) -> Path:
    """Directory for raw, pre-binned distributions used by histogram figures."""
    d = Path(out_dir) / "plot_raw_distribution_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_raw_distribution_table(
    out_dir: Path,
    figure_stem: str,
    data: pd.DataFrame,
    description: str,
    value_column: str = "value",
    suggested_bins: Optional[int] = None,
) -> None:
    """Save the actual distribution values behind a histogram.

    This is different from exporting the drawn histogram rectangles.  Origin users
    should load the *_raw_distribution.csv file and choose their own binning.
    A suggested-bin count table is also written only as a convenience.
    """
    if data is None or data.empty:
        return
    raw_dir = _ensure_raw_distribution_dir(out_dir)
    clean = data.copy()
    clean["source_figure"] = f"{figure_stem}.png"
    clean["description"] = description

    root_csv = Path(out_dir) / f"plot_data_{figure_stem}_raw_distribution.csv"
    sub_csv = raw_dir / f"{figure_stem}_raw_distribution.csv"
    clean.to_csv(root_csv, index=False, encoding="utf-8-sig")
    clean.to_csv(sub_csv, index=False, encoding="utf-8-sig")

    # Optional suggested bins, using the same default style as the figure. This is
    # not the primary data; it is only a reproducibility aid.
    if suggested_bins is not None and value_column in clean.columns:
        vals = clean[value_column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            try:
                counts, edges = np.histogram(vals, bins=suggested_bins)
                bin_df = pd.DataFrame({
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "bin_center": 0.5 * (edges[:-1] + edges[1:]),
                    "count": counts,
                    "suggested_bins_only_not_primary_data": True,
                })
                bin_df.to_csv(Path(out_dir) / f"plot_data_{figure_stem}_suggested_bins.csv", index=False, encoding="utf-8-sig")
                bin_df.to_csv(raw_dir / f"{figure_stem}_suggested_bins.csv", index=False, encoding="utf-8-sig")
            except Exception:
                pass

    # Maintain an index so the user can quickly find the raw data for every
    # histogram-like figure.
    idx_path = raw_dir / "plot_raw_distribution_data_index.csv"
    idx_row = pd.DataFrame([{
        "source_figure": f"{figure_stem}.png",
        "raw_distribution_file": sub_csv.name,
        "root_level_copy": root_csv.name,
        "description": description,
        "n_rows": int(clean.shape[0]),
        "value_column": value_column,
        "note": "Load the raw_distribution_file in Origin and choose binning manually.",
    }])
    if idx_path.exists():
        try:
            old = pd.read_csv(idx_path)
            idx_row = pd.concat([old, idx_row], ignore_index=True)
            idx_row = idx_row.drop_duplicates(subset=["source_figure", "raw_distribution_file"], keep="last")
        except Exception:
            pass
    idx_row.to_csv(idx_path, index=False, encoding="utf-8-sig")


def graphite_uv_to_xy(u: np.ndarray | float, v: np.ndarray | float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert fractional coordinates to the configured oblique display cell."""
    ang = math.radians(float(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG))
    x = np.asarray(u) + np.asarray(v) * math.cos(ang)
    y = np.asarray(v) * math.sin(ang)
    return x, y


def graphite_grid_edges(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return pcolormesh edge coordinates for an n x n fractional map."""
    e = np.linspace(0.0, 1.0, int(n) + 1)
    Ue, Ve = np.meshgrid(e, e)
    Xe, Ye = graphite_uv_to_xy(Ue, Ve)
    return Xe, Ye


def graphite_grid_centers(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return center u/v and graphite-display x/y coordinates for an n x n map."""
    c = (np.arange(int(n), dtype=float) + 0.5) / float(n)
    Uc, Vc = np.meshgrid(c, c)
    Xc, Yc = graphite_uv_to_xy(Uc, Vc)
    return Uc, Vc, Xc, Yc


def plot_unit_cell_map_on_axis(
    ax,
    unit_map: np.ndarray,
    *,
    cmap: str = "RdBu_r",
    vmin: float = -2.5,
    vmax: float = 2.5,
    show_axes: bool = False,
):
    """Plot a folded unit-cell map as the configured primitive parallelogram."""
    arr = np.asarray(unit_map, dtype=float)
    if DISPLAY_UNIT_CELL_MAPS_AS_GRAPHITE_PARALLELOGRAM:
        Xe, Ye = graphite_grid_edges(arr.shape[0])
        artist = ax.pcolormesh(Xe, Ye, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        # Draw primitive-cell boundary so it is visually clear this is not a square.
        bx = [0.0, 1.0, 1.0 + math.cos(math.radians(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)), math.cos(math.radians(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)), 0.0]
        by = [0.0, 0.0, math.sin(math.radians(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)), math.sin(math.radians(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)), 0.0]
        ax.plot(bx, by, color="0.15", lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.04, 1.0 + math.cos(math.radians(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)) + 0.04)
        ax.set_ylim(-0.04, math.sin(math.radians(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)) + 0.04)
        if show_axes:
            angle = float(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG)
            ax.set_xlabel(f"x = u + v cos({angle:g}°)", fontsize=7)
            ax.set_ylabel(f"y = v sin({angle:g}°)", fontsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        return artist
    artist = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return artist


def plot_site_marker(ax, u: float, v: float, marker: str, size: float, label: Optional[str] = None) -> None:
    """Overlay one B/A/H marker in the same coordinate system as the unit-cell map."""
    if DISPLAY_UNIT_CELL_MAPS_AS_GRAPHITE_PARALLELOGRAM:
        x, y = graphite_uv_to_xy(float(u), float(v))
        ax.scatter([x], [y], marker=marker, s=size, facecolors="none", edgecolors="black", linewidths=1.0, zorder=5)
        if label is not None:
            ax.text(float(x) + 0.018, float(y) + 0.018, label, fontsize=7, color="black", weight="bold", zorder=6)
    else:
        # In square fallback mode the caller may set limits in pixel units; here
        # use fractional axes, so this branch is mainly for debugging.
        ax.scatter([float(u)], [float(v)], marker=marker, s=size, facecolors="none", edgecolors="black", linewidths=1.0, zorder=5)
        if label is not None:
            ax.text(float(u) + 0.02, float(v) + 0.02, label, fontsize=7, color="black", weight="bold", zorder=6)


def plot_unit_cell_gallery(maps: np.ndarray, names: Sequence[str], positions_role: object, out_dir: Path) -> None:
    """Plot gallery using either one global or per-image B/A/H positions."""
    n = len(names)
    cols = min(6, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    variable = isinstance(positions_role, (list, tuple))
    for i, (m, name) in enumerate(zip(maps, names)):
        ax = axes.ravel()[i]
        im = plot_unit_cell_map_on_axis(ax, m, show_axes=False)
        ax.set_title(name, fontsize=7)
        pos_i = positions_role[i] if variable else positions_role
        for label, marker, size in [("B", "o", 28), ("A", "s", 28), ("H", "^", 25)]:
            u, v = pos_i[label]
            plot_site_marker(ax, float(u), float(v), marker, size, label=label)
        plt.colorbar(im, ax=ax, shrink=0.65)
    fig.suptitle("Aligned normalized unit-cell maps with graphite-correct B/A/H overlay", fontsize=13)
    savefig(out_dir / "aligned_unit_cells_with_BAH_overlay.png")


def plot_motif_scatter_kde(df: pd.DataFrame, out_dir: Path, prefix: str = "normalized") -> None:
    x = df["c_AH_norm"].to_numpy(dtype=float)
    y = df["abs_c_beta_norm"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 5:
        xx = x[finite]
        yy = y[finite]
        pad_x = 0.15 * (np.nanmax(xx) - np.nanmin(xx) + 1e-9)
        pad_y = 0.15 * (np.nanmax(yy) - np.nanmin(yy) + 1e-9)
        xi = np.linspace(np.nanmin(xx) - pad_x, np.nanmax(xx) + pad_x, 160)
        yi = np.linspace(np.nanmin(yy) - pad_y, np.nanmax(yy) + pad_y, 160)
        Xg, Yg = np.meshgrid(xi, yi)
        try:
            kde = gaussian_kde(np.vstack([xx, yy]))
            Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contourf(Xg, Yg, Z, levels=16, alpha=0.55)
            ax.contour(Xg, Yg, Z, levels=8, linewidths=0.7, alpha=0.8)
        except Exception:
            pass
    ax.scatter(x, y, s=50, edgecolors="black", linewidths=0.6)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$c_{AH}=(I_A+I_B-2I_H)/\sqrt{6}$  (atom-vs-hollow/contact-registry)")
    ax.set_ylabel(r"$|c_{\beta}|=|I_B-I_A|/\sqrt{2}$  (sublattice-polarization magnitude)")
    ax.set_title("Orthogonal motif-coordinate space: same-condition ensemble")
    # Overlay physical-limit prototype directions used for soft membership.
    rr = np.sqrt(x ** 2 + y ** 2)
    rscale = np.nanpercentile(rr[np.isfinite(rr)], 95) if np.isfinite(rr).any() else 1.0
    rscale = rscale if np.isfinite(rscale) and rscale > 0 else 1.0
    for label, proto in physical_soft_membership_prototypes().items():
        px, py = proto * rscale
        ax.plot([0, px], [0, py], ls="--", lw=0.8, color="0.35")
        ax.scatter([px], [py], marker="*", s=90, edgecolors="black", linewidths=0.6)
        ax.annotate(label.replace("_", "\n"), (px, py), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.text(0.02, 0.98, "continuous motif-state space\nphysical prototypes are soft references", transform=ax.transAxes, va="top", ha="left", fontsize=9)
    if LABEL_POINTS_IN_SCATTER and len(df) <= MAX_LABELLED_POINTS:
        for _, r in df.iterrows():
            ax.annotate(str(r["short_name"]), (r["c_AH_norm"], r["abs_c_beta_norm"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    savefig(out_dir / f"{prefix}_orthogonal_motif_coordinate_scatter_kde.png")


def plot_R_theta(df: pd.DataFrame, out_dir: Path, prefix: str = "normalized") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df["theta_beta_over_AH_deg_norm"], df["R_orthogonal_norm"], s=55, edgecolors="black", linewidths=0.6)
    ax.set_xlabel(r"motif angle $\theta=\tan^{-1}(|c_{\beta}|/c_{AH})$ from contact-registry axis (deg)")
    ax.set_ylabel(r"motif amplitude $R=\sqrt{c_{AH}^2+c_{\beta}^2}$")
    ax.set_title("Motif amplitude and motif angle after unit-cell normalization")
    for deg in [0, 45, 90]:
        ax.axvline(deg, color="0.55", ls="--", lw=0.8)
    ax.text(0.02, 0.96,
            "low θ: atom-vs-hollow/contact-registry dominated\n"
            "high θ: sublattice-polarization dominated\n"
            "intermediate: A/B/H hybrid motif",
            transform=ax.transAxes, va="top", ha="left", fontsize=9)
    if LABEL_POINTS_IN_SCATTER and len(df) <= MAX_LABELLED_POINTS:
        for _, r in df.iterrows():
            ax.annotate(str(r["short_name"]), (r["theta_beta_over_AH_deg_norm"], r["R_orthogonal_norm"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    savefig(out_dir / f"{prefix}_motif_amplitude_R_vs_angle_theta.png")


def plot_theta_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    base_cols = [c for c in ["name", "short_name", "theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm"] if c in df.columns]
    raw_df = df[base_cols].copy() if base_cols else pd.DataFrame()
    if "theta_beta_over_AH_deg_norm" in raw_df.columns:
        raw_df = raw_df.rename(columns={"theta_beta_over_AH_deg_norm": "value"})
        raw_df["metric"] = "motif_angle_theta_deg"
        try:
            raw_df["theta_regime"] = raw_df["value"].map(theta_regime_label)
        except Exception:
            pass
    vals = df["theta_beta_over_AH_deg_norm"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    suggested_bins = min(12, max(4, len(vals) // 2)) if len(vals) else None
    if not raw_df.empty:
        _save_raw_distribution_table(
            out_dir,
            "motif_angle_theta_histogram",
            raw_df,
            "Raw per-image motif angle theta values before histogram binning.",
            value_column="value",
            suggested_bins=suggested_bins,
        )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(vals, bins=suggested_bins, alpha=0.75, edgecolor="black")
    ax.set_xlabel(r"motif angle $\theta$ (deg)")
    ax.set_ylabel("number of images")
    ax.set_title("Distribution of motif shape coordinate")
    for deg in [0, 45, 90]:
        ax.axvline(deg, color="0.55", ls="--", lw=0.8)
    savefig(out_dir / "motif_angle_theta_histogram.png")


def plot_site_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    site_rows = []
    id_cols = [c for c in ["name", "short_name"] if c in df.columns]
    for site, col, desc in [("A", "I_A_norm", "A intermediate atom-site"), ("B", "I_B_norm", "B bright atom-site"), ("H", "I_H_norm", "H geometric hollow/low site")]:
        if col in df.columns:
            tmp = df[id_cols + [col]].copy() if id_cols else df[[col]].copy()
            tmp = tmp.rename(columns={col: "value"})
            tmp["site"] = site
            tmp["site_description"] = desc
            tmp["metric"] = "site_averaged_normalized_current"
            site_rows.append(tmp)
    if site_rows:
        site_raw = pd.concat(site_rows, ignore_index=True)
        _save_raw_distribution_table(
            out_dir,
            "site_conditioned_intensity_histograms",
            site_raw,
            "Raw site-averaged normalized current values for A/B/H histograms before binning.",
            value_column="value",
            suggested_bins=12,
        )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(df["I_A_norm"], bins=12, alpha=0.55, label="A intermediate", edgecolor="black")
    ax.hist(df["I_B_norm"], bins=12, alpha=0.55, label="B bright", edgecolor="black")
    ax.hist(df["I_H_norm"], bins=12, alpha=0.55, label="H low/hollow", edgecolor="black")
    ax.set_xlabel("site-averaged normalized current")
    ax.set_ylabel("count")
    ax.set_title("Site-conditioned intensity distributions")
    ax.legend(fontsize=8)
    savefig(out_dir / "site_conditioned_intensity_histograms.png")

    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    data = [df["I_B_norm"].to_numpy(), df["I_A_norm"].to_numpy(), df["I_H_norm"].to_numpy()]
    ax.boxplot(data, tick_labels=["B bright", "A intermediate", "H low"], showmeans=True)
    ax.set_ylabel("site-averaged normalized current")
    ax.set_title("B/A/H site intensity statistics")
    savefig(out_dir / "site_intensity_boxplot.png")


def theta_regime_label(theta_deg: float) -> str:
    """Descriptive theta regime label; not a hard physical class."""
    if not np.isfinite(theta_deg):
        return "NA"
    if theta_deg < THETA_AH_DOMINANT_MAX_DEG:
        return "AH-like/contact-registry-dominant tail"
    if theta_deg > THETA_BETA_DOMINANT_MIN_DEG:
        return "B-like/sublattice-polarized tail"
    return "ABH-like hybrid regime"


def compute_theta_quantile_table(
    maps: np.ndarray,
    df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Compute manuscript theta percentiles and representative unit-cell maps.

    For each requested percentile, this records:
      - exact theta percentile value from the ensemble,
      - nearest individual image/map to that percentile,
      - local-window averaged map around that quantile in theta-sorted order.

    The local average is useful for Figure 3c because it suppresses single-image
    noise while preserving the smooth motif progression along theta.
    """
    if "theta_beta_over_AH_deg_norm" not in df.columns:
        qdf = pd.DataFrame()
        return qdf, []

    theta_all = df["theta_beta_over_AH_deg_norm"].to_numpy(dtype=float)
    finite_mask = np.isfinite(theta_all)
    finite_positions = np.where(finite_mask)[0]
    if finite_positions.size == 0:
        qdf = pd.DataFrame()
        return qdf, []

    order = finite_positions[np.argsort(theta_all[finite_positions])]
    n_valid = int(order.size)
    window_n = int(round(n_valid * THETA_QUANTILE_AVERAGE_WINDOW_FRACTION))
    window_n = max(THETA_QUANTILE_AVERAGE_MIN_N, window_n)
    window_n = min(THETA_QUANTILE_AVERAGE_MAX_N, window_n, n_valid)

    rows = []
    avg_maps: List[np.ndarray] = []
    for pct in THETA_QUANTILE_PERCENTILES:
        pct_float = float(pct)
        target_theta = float(np.nanpercentile(theta_all[finite_mask], pct_float))
        nearest_local = int(np.nanargmin(np.abs(theta_all[finite_positions] - target_theta)))
        nearest_idx = int(finite_positions[nearest_local])

        # Window centered on quantile rank in theta-sorted order.
        rank_pos = int(round((n_valid - 1) * pct_float / 100.0))
        half = window_n // 2
        lo = max(0, rank_pos - half)
        hi = min(n_valid, lo + window_n)
        lo = max(0, hi - window_n)
        window_indices = order[lo:hi]

        avg_map = np.nanmean(maps[window_indices], axis=0)
        avg_maps.append(avg_map)

        # Quantile-window coherence: if the local window is coherent, the
        # average map is a meaningful percentile motif rather than an average of
        # unrelated shapes.  The representative-to-average correlation tells how
        # typical the nearest real image is for that percentile window.
        try:
            wcorr, _ = pairwise_corr_distance(maps[window_indices])
            tri = wcorr[np.triu_indices_from(wcorr, k=1)]
            window_pairwise_corr_median = float(np.nanmedian(tri)) if tri.size else np.nan
            window_pairwise_corr_mean = float(np.nanmean(tri)) if tri.size else np.nan
        except Exception:
            window_pairwise_corr_median = np.nan
            window_pairwise_corr_mean = np.nan
        try:
            representative_to_average_corr = float(corrcoef2(maps[nearest_idx], avg_map))
        except Exception:
            representative_to_average_corr = np.nan

        nearest_row = df.iloc[nearest_idx]
        window_theta = theta_all[window_indices]
        rows.append({
            "percentile": int(pct),
            "theta_percentile_deg": target_theta,
            "theta_regime_at_percentile": theta_regime_label(target_theta),
            "nearest_index": nearest_idx,
            "nearest_name": str(nearest_row.get("name", nearest_idx)),
            "nearest_short_name": str(nearest_row.get("short_name", nearest_idx)),
            "nearest_theta_deg": float(theta_all[nearest_idx]),
            "nearest_theta_error_deg": float(theta_all[nearest_idx] - target_theta),
            "window_n": int(len(window_indices)),
            "window_rank_min": int(lo),
            "window_rank_max": int(hi - 1),
            "window_theta_mean_deg": float(np.nanmean(window_theta)),
            "window_theta_median_deg": float(np.nanmedian(window_theta)),
            "window_theta_min_deg": float(np.nanmin(window_theta)),
            "window_theta_max_deg": float(np.nanmax(window_theta)),
            "window_pairwise_corr_median": window_pairwise_corr_median,
            "window_pairwise_corr_mean": window_pairwise_corr_mean,
            "representative_to_window_average_corr": representative_to_average_corr,
            "window_names": ";".join(str(df.iloc[int(i)].get("name", i)) for i in window_indices),
            "window_indices_json": json.dumps([int(i) for i in window_indices]),
        })

    qdf = pd.DataFrame(rows)
    if out_dir is not None:
        qdf.to_csv(out_dir / "theta_quantiles_5_25_50_75_95.csv", index=False, encoding="utf-8-sig")

        # Compact summary for Figure 3b text/caption.
        theta_vals = theta_all[finite_mask]
        summary = {
            "N_theta_valid": int(theta_vals.size),
            "theta_mean_deg": float(np.nanmean(theta_vals)),
            "theta_median_deg": float(np.nanmedian(theta_vals)),
            "theta_std_deg": float(np.nanstd(theta_vals, ddof=1)) if theta_vals.size > 1 else np.nan,
            "theta_AH_like_tail_fraction_lt_20deg": float(np.mean(theta_vals < THETA_AH_DOMINANT_MAX_DEG)),
            "theta_hybrid_fraction_20_to_70deg": float(np.mean((theta_vals >= THETA_AH_DOMINANT_MAX_DEG) & (theta_vals <= THETA_BETA_DOMINANT_MIN_DEG))),
            "theta_B_like_tail_fraction_gt_70deg": float(np.mean(theta_vals > THETA_BETA_DOMINANT_MIN_DEG)),
            "theta_combined_tail_fraction": float(np.mean((theta_vals < THETA_AH_DOMINANT_MAX_DEG) | (theta_vals > THETA_BETA_DOMINANT_MIN_DEG))),
        }
        for pct in THETA_QUANTILE_PERCENTILES:
            summary[f"theta_p{int(pct):02d}_deg"] = float(np.nanpercentile(theta_vals, pct))
        pd.DataFrame([summary]).to_csv(out_dir / "theta_distribution_summary.csv", index=False, encoding="utf-8-sig")

    return qdf, avg_maps


def _plot_quantile_map_strip(
    maps_to_plot: Sequence[np.ndarray],
    qdf: pd.DataFrame,
    out_path: Path,
    title: str,
    map_label: str,
) -> None:
    if qdf.empty or len(maps_to_plot) == 0:
        return
    fig, axes = plt.subplots(1, len(maps_to_plot), figsize=(2.65 * len(maps_to_plot), 2.9))
    if len(maps_to_plot) == 1:
        axes = [axes]
    for ax, unit_map, (_, row) in zip(axes, maps_to_plot, qdf.iterrows()):
        im = plot_unit_cell_map_on_axis(ax, unit_map, show_axes=False)
        pct = int(row["percentile"])
        theta_pct = float(row["theta_percentile_deg"])
        if map_label == "average":
            subtitle = f"{pct}th pct\nθ={theta_pct:.1f}°\nwindow n={int(row['window_n'])}"
        else:
            subtitle = f"{pct}th pct\n{row['nearest_short_name']}\nθ={float(row['nearest_theta_deg']):.1f}°"
        ax.set_title(subtitle, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.65)
    fig.suptitle(title, fontsize=12)
    savefig(out_path)


def save_theta_quantile_maps_as_2d_data(
    avg_maps: Sequence[np.ndarray],
    qdf: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save each theta-quantile averaged unit-cell map as reusable 2D data."""
    if qdf.empty or len(avg_maps) == 0:
        return

    map_dir = out_dir / "theta_quantile_avg_map_data"
    map_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    metadata_rows = []
    for unit_map, (_, row) in zip(avg_maps, qdf.iterrows()):
        pct = int(row["percentile"])
        key = f"theta_p{pct:02d}"
        arr = np.asarray(unit_map, dtype=float)
        arrays[key] = arr
        df_map = pd.DataFrame(arr)
        csv_path = map_dir / f"theta_quantile_avg_map_p{pct:02d}.csv"
        df_map.to_csv(csv_path, index=False, header=False, encoding="utf-8-sig")

        # Origin/Matlab/Python long-format data for the true graphite 60°
        # parallelogram display.  Use x_graphite60/y_graphite60 as X/Y and z as
        # the color value.  The matrix CSV above remains the raw fractional
        # (u, v) grid for reproducibility.
        Uc, Vc, Xc, Yc = graphite_grid_centers(arr.shape[0])
        xyz_path = map_dir / f"theta_quantile_avg_map_p{pct:02d}_graphite60_xyz.csv"
        pd.DataFrame({
            "u_center": Uc.ravel(),
            "v_center": Vc.ravel(),
            "x_graphite60": Xc.ravel(),
            "y_graphite60": Yc.ravel(),
            "z": arr.ravel(),
        }).to_csv(xyz_path, index=False, encoding="utf-8-sig")
        Xe, Ye = graphite_grid_edges(arr.shape[0])
        xedge_path = map_dir / f"theta_quantile_avg_map_p{pct:02d}_x_edges_graphite60.csv"
        yedge_path = map_dir / f"theta_quantile_avg_map_p{pct:02d}_y_edges_graphite60.csv"
        pd.DataFrame(Xe).to_csv(xedge_path, index=False, header=False, encoding="utf-8-sig")
        pd.DataFrame(Ye).to_csv(yedge_path, index=False, header=False, encoding="utf-8-sig")

        metadata_rows.append({
            "percentile": pct,
            "theta_percentile_deg": float(row["theta_percentile_deg"]),
            "theta_regime_at_percentile": row.get("theta_regime_at_percentile", ""),
            "window_n": int(row.get("window_n", 0)),
            "window_theta_mean_deg": float(row.get("window_theta_mean_deg", np.nan)),
            "window_theta_median_deg": float(row.get("window_theta_median_deg", np.nan)),
            "window_theta_min_deg": float(row.get("window_theta_min_deg", np.nan)),
            "window_theta_max_deg": float(row.get("window_theta_max_deg", np.nan)),
            "window_pairwise_corr_median": float(row.get("window_pairwise_corr_median", np.nan)),
            "window_pairwise_corr_mean": float(row.get("window_pairwise_corr_mean", np.nan)),
            "representative_to_window_average_corr": float(row.get("representative_to_window_average_corr", np.nan)),
            "nearest_name": row.get("nearest_name", ""),
            "nearest_theta_deg": float(row.get("nearest_theta_deg", np.nan)),
            "csv_path": str(csv_path.name),
            "graphite60_xyz_path": str(xyz_path.name),
            "graphite60_x_edges_path": str(xedge_path.name),
            "graphite60_y_edges_path": str(yedge_path.name),
            "display_geometry": f"graphite_{GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG:g}deg_parallelogram" if DISPLAY_UNIT_CELL_MAPS_AS_GRAPHITE_PARALLELOGRAM else "square_uv_grid",
            "n_rows": int(arr.shape[0]),
            "n_cols": int(arr.shape[1]),
        })

    np.savez_compressed(map_dir / "theta_quantile_avg_maps.npz", **arrays)

    with pd.ExcelWriter(map_dir / "theta_quantile_avg_maps.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="metadata", index=False)
        for unit_map, (_, row) in zip(avg_maps, qdf.iterrows()):
            pct = int(row["percentile"])
            pd.DataFrame(np.asarray(unit_map, dtype=float)).to_excel(
                writer, sheet_name=f"theta_p{pct:02d}", index=False, header=False
            )

    pd.DataFrame(metadata_rows).to_csv(
        map_dir / "theta_quantile_avg_map_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_quantile_unit_cells(maps: np.ndarray, df: pd.DataFrame, out_dir: Path) -> None:
    qdf, avg_maps = compute_theta_quantile_table(maps, df, out_dir)
    if qdf.empty:
        return

    # Save reusable 2D data so the percentile-averaged maps can be re-visualized.
    save_theta_quantile_maps_as_2d_data(avg_maps, qdf, out_dir)

    # Figure 3c-style output: local-window averaged unit cells at theta percentiles.
    _plot_quantile_map_strip(
        avg_maps,
        qdf,
        out_dir / "quantile_ordered_unit_cells_by_theta.png",
        "Quantile-window averaged unit cells along motif angle θ",
        "average",
    )

    # Also keep nearest-image representatives for traceability/debugging.
    representative_maps = [maps[int(row["nearest_index"])] for _, row in qdf.iterrows()]
    _plot_quantile_map_strip(
        representative_maps,
        qdf,
        out_dir / "quantile_representative_unit_cells_by_theta.png",
        "Nearest representative unit cells at motif-angle θ percentiles",
        "representative",
    )

def plot_pairwise_heatmap(corr: np.ndarray, names: Sequence[str], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("Whole-unit-cell motif correlation matrix")
    n = len(names)
    if n <= 35:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([short_name(x) for x in names], rotation=90, fontsize=5)
        ax.set_yticklabels([short_name(x) for x in names], fontsize=5)
    plt.colorbar(im, ax=ax, label="correlation")
    savefig(out_dir / "whole_map_pairwise_correlation_heatmap.png")


def plot_pca(expl: pd.DataFrame, score_df: pd.DataFrame, df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    k = min(10, len(expl))
    ax.bar(expl["PC"].iloc[:k], 100.0 * expl["explained_variance_ratio"].iloc[:k])
    ax.plot(expl["PC"].iloc[:k], 100.0 * expl["cumulative"].iloc[:k], marker="o")
    ax.set_xlabel("PC")
    ax.set_ylabel("explained variance (%)")
    ax.set_title("PCA of aligned normalized unit-cell maps")
    savefig(out_dir / "whole_map_pca_explained_variance.png")

    if score_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6.2, 5.1))
        sc = ax.scatter(score_df["PC1"], score_df["PC2"], c=df["theta_beta_over_AH_deg_norm"], s=55, edgecolors="black", linewidths=0.6)
        ax.set_xlabel(f"PC1 score ({100*expl['explained_variance_ratio'].iloc[0]:.1f}%)")
        ax.set_ylabel(f"PC2 score ({100*expl['explained_variance_ratio'].iloc[1]:.1f}%)")
        ax.set_title("Whole-map PCA scores colored by motif angle θ")
        plt.colorbar(sc, ax=ax, label="θ (deg)")
        if LABEL_POINTS_IN_SCATTER and len(df) <= MAX_LABELLED_POINTS:
            for i, r in df.iterrows():
                ax.annotate(str(r["short_name"]), (score_df.loc[i, "PC1"], score_df.loc[i, "PC2"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        savefig(out_dir / "whole_map_pca_scores_pc1_pc2.png")


def plot_mds(mds: np.ndarray, df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    sc = ax.scatter(mds[:, 0], mds[:, 1], c=df["theta_beta_over_AH_deg_norm"], s=55, edgecolors="black", linewidths=0.6)
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.set_title("Motif-shape embedding from whole-map correlation distance")
    plt.colorbar(sc, ax=ax, label="θ (deg)")
    if LABEL_POINTS_IN_SCATTER and len(df) <= MAX_LABELLED_POINTS:
        for i, r in df.iterrows():
            ax.annotate(str(r["short_name"]), (mds[i, 0], mds[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    savefig(out_dir / "whole_map_classical_mds_embedding.png")


def plot_soft_membership(mem: pd.DataFrame, df: pd.DataFrame, out_dir: Path) -> None:
    merged = mem.merge(df[["name", "theta_beta_over_AH_deg_norm", "short_name"]], on="name", how="left")
    merged = merged.sort_values("theta_beta_over_AH_deg_norm")
    prob_cols = [c for c in mem.columns if c.startswith("P_")]
    x = np.arange(len(merged))
    bottom = np.zeros(len(merged))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.32 * len(merged)), 4.6))
    for c in prob_cols:
        vals = merged[c].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=c.replace("P_", ""))
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("soft membership probability")
    ax.set_xlabel("images ordered by motif angle θ")
    ax.set_title("Soft membership to reference motif directions")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["short_name"], rotation=90, fontsize=6)
    ax.legend(fontsize=8, loc="upper right")
    savefig(out_dir / "soft_membership_probabilities.png")


def plot_gmm_bic(gmm_df: pd.DataFrame, out_dir: Path) -> None:
    if gmm_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(gmm_df["k"], gmm_df["BIC"], marker="o", label="BIC")
    ax.plot(gmm_df["k"], gmm_df["AIC"], marker="s", label="AIC")
    ax.set_xlabel("number of Gaussian components")
    ax.set_ylabel("information criterion, lower is better")
    ax.set_title("Optional GMM model selection in motif-coordinate space")
    ax.legend()
    savefig(out_dir / "gmm_bic_aic_optional_cluster_check.png")


def short_name(name: str) -> str:
    return str(name).replace("::Sheet1", "").replace("::", ":")



# =============================================================================
# Origin source-data exporter
# =============================================================================

_ORIGIN_SOURCE_INDEX_ROWS: List[Dict[str, object]] = []


def _origin_source_enabled() -> bool:
    return bool(globals().get("EXPORT_ORIGIN_SOURCE_DATA", True))


def _origin_source_root(out_dir: Path) -> Path:
    root = Path(out_dir) / globals().get("ORIGIN_SOURCE_DATA_FOLDER_NAME", "origin_source_data")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _origin_safe_sheet_name(text_value: object, used: Optional[set] = None, max_len: int = 31) -> str:
    """Make a valid, unique Excel sheet name for the Origin source workbook."""
    s = str(text_value)
    s = re.sub(r"[\\/\?\*\[\]:]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    if not s:
        s = "sheet"
    s = s[:max_len]
    if used is None:
        return s
    base = s[:max_len]
    i = 1
    while s in used:
        suffix = f"_{i}"
        s = base[:max_len - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s


def _origin_select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    have = [c for c in columns if c in df.columns]
    return df[have].copy() if have else pd.DataFrame()


def _reset_origin_source_data(out_dir: Path) -> None:
    """Start a fresh Origin source-data index for the current run."""
    global _ORIGIN_SOURCE_INDEX_ROWS
    _ORIGIN_SOURCE_INDEX_ROWS = []
    root = _origin_source_root(out_dir)
    for fname in ["origin_source_data_index.csv", globals().get("ORIGIN_SOURCE_DATA_EXCEL_NAME", "origin_source_plot_data.xlsx"), "README_origin_source_data.md"]:
        try:
            f = root / fname
            if f.exists():
                f.unlink()
        except Exception:
            pass


def _save_origin_source_table(
    out_dir: Path,
    figure_stem: str,
    table_name: str,
    data: pd.DataFrame,
    *,
    plot_type: str,
    x_columns: Sequence[str] = (),
    y_columns: Sequence[str] = (),
    z_columns: Sequence[str] = (),
    label_columns: Sequence[str] = (),
    description: str = "",
    origin_action: str = "",
) -> Optional[Path]:
    """Save the actual source data behind a plot in an Origin-friendly form.

    Unlike the legacy Matplotlib artist exporter, this function writes the raw
    x/y/z values from the analysis DataFrames.  These CSV files are the files to
    import into Origin for re-plotting.
    """
    if not _origin_source_enabled():
        return None
    if data is None or data.empty:
        return None

    root = _origin_source_root(out_dir)
    fig_dir = root / _safe_filename_component(figure_stem, max_len=90)
    fig_dir.mkdir(parents=True, exist_ok=True)

    clean = data.copy()
    # Avoid Excel/Origin trouble with list/dict objects.
    for c in clean.columns:
        if clean[c].dtype == object:
            clean[c] = clean[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v)

    csv_name = f"{_safe_filename_component(table_name, max_len=90)}.csv"
    csv_path = fig_dir / csv_name
    clean.to_csv(csv_path, index=False, encoding="utf-8-sig")

    _ORIGIN_SOURCE_INDEX_ROWS.append({
        "figure_stem": figure_stem,
        "table_name": table_name,
        "plot_type": plot_type,
        "csv_file": str(csv_path.relative_to(root)),
        "n_rows": int(clean.shape[0]),
        "n_columns": int(clean.shape[1]),
        "x_columns": ";".join(x_columns),
        "y_columns": ";".join(y_columns),
        "z_columns": ";".join(z_columns),
        "label_columns": ";".join(label_columns),
        "description": description,
        "origin_action": origin_action,
    })
    return csv_path


def _flush_origin_source_index(out_dir: Path) -> pd.DataFrame:
    root = _origin_source_root(out_dir)
    idx = pd.DataFrame(_ORIGIN_SOURCE_INDEX_ROWS)
    idx.to_csv(root / "origin_source_data_index.csv", index=False, encoding="utf-8-sig")
    return idx


def _compile_origin_source_workbook(out_dir: Path, index_df: pd.DataFrame) -> None:
    """Compile all small Origin source CSVs into one Excel workbook."""
    if not bool(globals().get("ORIGIN_SOURCE_COMPILE_XLSX", True)):
        return
    if index_df is None or index_df.empty:
        return
    root = _origin_source_root(out_dir)
    xlsx_path = root / globals().get("ORIGIN_SOURCE_DATA_EXCEL_NAME", "origin_source_plot_data.xlsx")
    used: set = set()
    max_excel_rows = 1048576 - 1  # leave room for header row
    max_excel_cols = 16384

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        index_df.to_excel(writer, sheet_name="_index", index=False)
        for _, r in index_df.iterrows():
            rel = Path(str(r.get("csv_file", "")))
            csv_path = root / rel
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            sheet_base = f"{r.get('figure_stem', '')}_{r.get('table_name', '')}"
            sheet_name = _origin_safe_sheet_name(sheet_base, used=used)
            if df.shape[0] > max_excel_rows or df.shape[1] > max_excel_cols:
                note = pd.DataFrame([{
                    "csv_file": str(rel),
                    "n_rows": df.shape[0],
                    "n_columns": df.shape[1],
                    "note": "This table is larger than Excel's sheet limit. Import the CSV directly into Origin.",
                }])
                note.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def _write_origin_source_readme(out_dir: Path, index_df: pd.DataFrame) -> None:
    root = _origin_source_root(out_dir)
    lines = []
    lines.append("# Origin source data")
    lines.append("")
    lines.append("Use this folder for Origin re-plotting. These files contain the raw analysis data used for each figure, not Matplotlib-rendered objects.")
    lines.append("")
    lines.append("Recommended workflow in Origin:")
    lines.append("1. Open `origin_source_plot_data.xlsx`, or import the corresponding CSV listed in `origin_source_data_index.csv`.")
    lines.append("2. Use the `x_columns`, `y_columns`, and `z_columns` columns in the index to assign X/Y/Z designations.")
    lines.append("3. For scatter plots, plot the listed X and Y columns directly. For histograms and boxplots, use the raw long-format `value` column and choose binning/grouping in Origin.")
    lines.append("4. For unit-cell maps, use `x_graphite60`, `y_graphite60`, and `z_norm` or `z` as XYZ data for a contour/colormap plot.")
    lines.append("")
    lines.append("Do not use the legacy `origin_plot_data` folder for manuscript re-plotting unless you specifically need Matplotlib artist geometry.")
    lines.append("")
    if index_df is not None and not index_df.empty:
        lines.append("## Tables")
        for _, r in index_df.iterrows():
            lines.append(f"- `{r['csv_file']}`: {r['plot_type']}; X={r['x_columns']}; Y={r['y_columns']}; Z={r['z_columns']}; {r['description']}")
    (root / "README_origin_source_data.md").write_text("\n".join(lines), encoding="utf-8")


def _origin_current_floor_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    floor_metric = CURRENT_FLOOR_PRIMARY_METRIC
    if floor_metric not in out.columns:
        return out
    valid = np.isfinite(pd.to_numeric(out[floor_metric], errors="coerce").to_numpy(dtype=float))
    labels = ["low_floor", "mid_floor", "high_floor"] if CURRENT_FLOOR_BIN_COUNT == 3 else [f"bin_{i+1}" for i in range(CURRENT_FLOOR_BIN_COUNT)]
    try:
        out.loc[valid, "current_floor_bin_for_origin"] = pd.qcut(out.loc[valid, floor_metric], q=CURRENT_FLOOR_BIN_COUNT, labels=labels, duplicates="drop")
    except Exception:
        try:
            out.loc[valid, "current_floor_bin_for_origin"] = pd.cut(out.loc[valid, floor_metric], bins=CURRENT_FLOOR_BIN_COUNT, labels=labels[:CURRENT_FLOOR_BIN_COUNT])
        except Exception:
            out["current_floor_bin_for_origin"] = "NA"
    return out


def _unitcell_maps_long_dataframe(maps: np.ndarray, names: Sequence[str], z_col: str = "z_norm") -> pd.DataFrame:
    maps = np.asarray(maps, dtype=float)
    if maps.ndim != 3:
        return pd.DataFrame()
    n, grid_y, grid_x = maps.shape
    if grid_y != grid_x:
        return pd.DataFrame()
    Uc, Vc, Xc, Yc = graphite_grid_centers(grid_y)
    base = pd.DataFrame({
        "u_center": Uc.ravel(),
        "v_center": Vc.ravel(),
        "x_graphite60": Xc.ravel(),
        "y_graphite60": Yc.ravel(),
    })
    parts = []
    for i in range(n):
        tmp = base.copy()
        tmp.insert(0, "image_index", int(i))
        tmp.insert(1, "name", str(names[i]))
        tmp.insert(2, "short_name", short_name(str(names[i])))
        tmp[z_col] = maps[i].ravel()
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def export_origin_source_plot_data(
    out_dir: Path,
    *,
    coord_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    lattice_df: pd.DataFrame,
    origin_search_df: pd.DataFrame,
    mem_df: pd.DataFrame,
    pca_expl: pd.DataFrame,
    pca_scores: pd.DataFrame,
    corr: Optional[np.ndarray],
    mds: Optional[np.ndarray],
    gmm_df: pd.DataFrame,
    aligned_norm: np.ndarray,
    names: Sequence[str],
    positions_role_by_index: Sequence[Dict[str, Tuple[float, float]]],
) -> None:
    """Write Origin-ready raw source data for the figures made by this script.

    The output is intentionally independent of Matplotlib artists.  It exports
    the same per-image rows, long-format distributions, and XYZ unit-cell values
    that should be imported directly into Origin.
    """
    if not _origin_source_enabled():
        return

    _reset_origin_source_data(out_dir)
    coord = coord_df.copy()
    coord_binned = _origin_current_floor_bins(coord)

    # Master per-image data table: the safest all-purpose Origin source table.
    _save_origin_source_table(
        out_dir,
        "all_per_image_analysis_data",
        "all_motif_current_floor_coordinates_raw",
        coord_binned,
        plot_type="master_table",
        description="Complete per-image analysis table. Use this when a specific figure table is not enough.",
        origin_action="Import as worksheet; choose desired columns as X/Y/Z in Origin.",
    )

    if ci_df is not None and not ci_df.empty:
        _save_origin_source_table(
            out_dir,
            "site_coordinate_bootstrap_CI_per_image",
            "bootstrap_ci_raw",
            ci_df,
            plot_type="uncertainty_table",
            description="Per-image bootstrap confidence intervals for site and motif coordinates.",
        )

    if summary_df is not None and not summary_df.empty:
        _save_origin_source_table(
            out_dir,
            "ensemble_motif_coordinate_summary_bootstrap",
            "ensemble_summary_raw",
            summary_df,
            plot_type="summary_table",
            description="Bootstrap ensemble summary values used in reports.",
        )

    # Motif-coordinate scatter and KDE source points.
    scatter_cols = [
        "name", "short_name", "c_AH_norm", "abs_c_beta_norm", "c_beta_signed_norm",
        "R_orthogonal_norm", "theta_beta_over_AH_deg_norm", "delta_BA_norm", "delta_AH_norm",
        "I_A_norm", "I_B_norm", "I_H_norm", CURRENT_FLOOR_PRIMARY_METRIC,
        "current_floor_bin_for_origin", "fft_orientation_label", "site_assignment_quality",
    ]
    scatter_df = _origin_select_columns(coord_binned, scatter_cols)
    _save_origin_source_table(
        out_dir,
        "normalized_orthogonal_motif_coordinate_scatter_kde",
        "scatter_points_raw",
        scatter_df,
        plot_type="scatter",
        x_columns=["c_AH_norm"],
        y_columns=["abs_c_beta_norm"],
        label_columns=["short_name"],
        description="Raw per-image points for the motif-coordinate scatter/KDE figure. Recompute density in Origin from these X/Y points if needed.",
        origin_action="Set c_AH_norm as X and abs_c_beta_norm as Y; plot as scatter. Optional: use theta/R/current_floor columns for color or labels.",
    )

    # Physical reference vectors used in the scatter figure.
    try:
        x = coord["c_AH_norm"].to_numpy(dtype=float)
        y = coord["abs_c_beta_norm"].to_numpy(dtype=float)
        rr = np.sqrt(x ** 2 + y ** 2)
        rscale = np.nanpercentile(rr[np.isfinite(rr)], 95) if np.isfinite(rr).any() else 1.0
        rscale = rscale if np.isfinite(rscale) and rscale > 0 else 1.0
        proto_rows = []
        for label, proto in physical_soft_membership_prototypes().items():
            proto_rows.append({
                "prototype": label,
                "x_start": 0.0,
                "y_start": 0.0,
                "x_end": float(proto[0] * rscale),
                "y_end": float(proto[1] * rscale),
                "theta_deg": float(np.degrees(np.arctan2(proto[1], proto[0]))),
                "radial_scale_used": float(rscale),
            })
        _save_origin_source_table(
            out_dir,
            "normalized_orthogonal_motif_coordinate_scatter_kde",
            "prototype_reference_vectors",
            pd.DataFrame(proto_rows),
            plot_type="reference_lines",
            x_columns=["x_start", "x_end"],
            y_columns=["y_start", "y_end"],
            label_columns=["prototype"],
            description="Reference prototype vectors overlaid on the motif-coordinate scatter.",
            origin_action="Plot each row as a line segment from x_start/y_start to x_end/y_end if prototype guides are desired.",
        )
    except Exception:
        pass

    rt_cols = ["name", "short_name", "theta_beta_over_AH_deg_norm", "R_orthogonal_norm", "c_AH_norm", "abs_c_beta_norm", CURRENT_FLOOR_PRIMARY_METRIC, "current_floor_bin_for_origin"]
    _save_origin_source_table(
        out_dir,
        "normalized_motif_amplitude_R_vs_angle_theta",
        "R_vs_theta_points_raw",
        _origin_select_columns(coord_binned, rt_cols),
        plot_type="scatter",
        x_columns=["theta_beta_over_AH_deg_norm"],
        y_columns=["R_orthogonal_norm"],
        label_columns=["short_name"],
        description="Raw per-image points for motif amplitude R versus motif angle theta.",
        origin_action="Set theta_beta_over_AH_deg_norm as X and R_orthogonal_norm as Y.",
    )

    if "theta_beta_over_AH_deg_norm" in coord.columns:
        theta_raw = _origin_select_columns(coord, ["name", "short_name", "theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm"])
        if not theta_raw.empty:
            theta_raw = theta_raw.rename(columns={"theta_beta_over_AH_deg_norm": "value"})
            theta_raw["metric"] = "motif_angle_theta_deg"
            theta_raw["theta_regime"] = theta_raw["value"].map(theta_regime_label)
        _save_origin_source_table(
            out_dir,
            "motif_angle_theta_histogram",
            "theta_values_raw_unbinned",
            theta_raw,
            plot_type="histogram_raw_values",
            x_columns=["value"],
            description="Unbinned theta values. Choose histogram binning in Origin.",
            origin_action="Use value as the histogram data column; choose bins in Origin.",
        )

    # Site-conditioned histograms and boxplots, long format.
    site_rows = []
    for site, col, desc in [
        ("A", "I_A_norm", "A intermediate atom-site"),
        ("B", "I_B_norm", "B bright atom-site"),
        ("H", "I_H_norm", "H geometric hollow/low site"),
    ]:
        if col in coord.columns:
            tmp = _origin_select_columns(coord, ["name", "short_name", col]).rename(columns={col: "value"})
            tmp["site"] = site
            tmp["site_description"] = desc
            tmp["metric"] = "site_averaged_normalized_current"
            site_rows.append(tmp)
    if site_rows:
        site_long = pd.concat(site_rows, ignore_index=True)
        _save_origin_source_table(
            out_dir,
            "site_conditioned_intensity_histograms",
            "site_values_long_raw",
            site_long,
            plot_type="histogram_or_boxplot_raw_values",
            x_columns=["value"],
            label_columns=["site"],
            description="Long-format raw A/B/H site intensities for histograms or grouped boxplots.",
            origin_action="For histograms use value grouped by site. For boxplot use site as group and value as Y.",
        )
        _save_origin_source_table(
            out_dir,
            "site_intensity_boxplot",
            "site_values_long_raw",
            site_long,
            plot_type="boxplot_raw_values",
            x_columns=["site"],
            y_columns=["value"],
            label_columns=["site"],
            description="Same raw site values arranged for grouped Origin boxplot.",
            origin_action="Use site as categorical X/group and value as Y.",
        )

    # Current-floor source data.
    if CURRENT_FLOOR_PRIMARY_METRIC in coord_binned.columns:
        floor_cols = [
            "name", "short_name", CURRENT_FLOOR_PRIMARY_METRIC, "raw_current_floor_p05", "raw_current_floor_abs_p05",
            "raw_current_median", "raw_current_abs_mean", "current_floor_bin_for_origin",
            "theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm",
            "R_orthogonal_detrended", "R_orthogonal_raw", "detrended_robust_z_scale", "folded_detrended_robust_z_scale",
        ]
        floor_all = _origin_select_columns(coord_binned, floor_cols)
        _save_origin_source_table(
            out_dir,
            "current_floor_all_relationships",
            "current_floor_motif_points_raw",
            floor_all,
            plot_type="multi_y_scatter_source",
            x_columns=[CURRENT_FLOOR_PRIMARY_METRIC],
            y_columns=["theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm"],
            label_columns=["short_name", "current_floor_bin_for_origin"],
            description="Raw current-floor and motif descriptor values. Use this for any current-floor scatter plot.",
            origin_action=f"Set {CURRENT_FLOOR_PRIMARY_METRIC} as X and choose a motif column as Y.",
        )
        for ycol, figstem in [
            ("theta_beta_over_AH_deg_norm", "current_floor_vs_theta"),
            ("c_AH_norm", "current_floor_vs_c_AH_norm"),
            ("abs_c_beta_norm", "current_floor_vs_abs_c_beta"),
            ("R_orthogonal_norm", "current_floor_vs_R_motif_amplitude"),
        ]:
            if ycol in coord_binned.columns:
                table = _origin_select_columns(coord_binned, ["name", "short_name", CURRENT_FLOOR_PRIMARY_METRIC, ycol, "current_floor_bin_for_origin"])
                _save_origin_source_table(
                    out_dir,
                    figstem,
                    "scatter_points_raw",
                    table,
                    plot_type="scatter",
                    x_columns=[CURRENT_FLOOR_PRIMARY_METRIC],
                    y_columns=[ycol],
                    label_columns=["short_name", "current_floor_bin_for_origin"],
                    description=f"Raw points for {ycol} versus current floor.",
                    origin_action=f"Set {CURRENT_FLOOR_PRIMARY_METRIC} as X and {ycol} as Y.",
                )
        if "theta_beta_over_AH_deg_norm" in coord_binned.columns:
            _save_origin_source_table(
                out_dir,
                "current_floor_bins_theta_boxplot",
                "theta_by_current_floor_bin_raw",
                _origin_select_columns(coord_binned, ["name", "short_name", "current_floor_bin_for_origin", CURRENT_FLOOR_PRIMARY_METRIC, "theta_beta_over_AH_deg_norm"]),
                plot_type="boxplot_raw_values",
                x_columns=["current_floor_bin_for_origin"],
                y_columns=["theta_beta_over_AH_deg_norm"],
                label_columns=["short_name"],
                description="Raw theta values grouped by current-floor tertile.",
                origin_action="Use current_floor_bin_for_origin as categorical group and theta_beta_over_AH_deg_norm as Y.",
            )
        if {"c_AH_norm", "abs_c_beta_norm"}.issubset(coord_binned.columns):
            _save_origin_source_table(
                out_dir,
                "current_floor_bins_motif_coordinate_scatter",
                "motif_coordinate_points_by_floor_bin_raw",
                _origin_select_columns(coord_binned, ["name", "short_name", "current_floor_bin_for_origin", CURRENT_FLOOR_PRIMARY_METRIC, "c_AH_norm", "abs_c_beta_norm", "theta_beta_over_AH_deg_norm", "R_orthogonal_norm"]),
                plot_type="scatter_grouped",
                x_columns=["c_AH_norm"],
                y_columns=["abs_c_beta_norm"],
                label_columns=["current_floor_bin_for_origin", "short_name"],
                description="Raw motif-coordinate scatter points with current-floor bin labels.",
                origin_action="Set c_AH_norm as X, abs_c_beta_norm as Y, and group/color by current_floor_bin_for_origin.",
            )

    # Normalization-floor control points.
    norm_cols = [
        "name", "short_name", CURRENT_FLOOR_PRIMARY_METRIC,
        "theta_beta_over_AH_deg_norm", "R_orthogonal_norm", "R_orthogonal_detrended", "R_orthogonal_raw",
        "detrended_robust_z_scale", "folded_detrended_robust_z_scale", "folded_norm_pre_final_robust_z_scale",
        "raw_current_mad", "raw_current_floor_abs_p05", "raw_current_abs_mean",
    ]
    norm_df = _origin_select_columns(coord, norm_cols)
    if not norm_df.empty:
        _save_origin_source_table(
            out_dir,
            "normalization_floor_control",
            "normalization_floor_control_points_raw",
            norm_df,
            plot_type="multi_y_scatter_source",
            x_columns=[CURRENT_FLOOR_PRIMARY_METRIC],
            y_columns=["R_orthogonal_norm", "R_orthogonal_detrended", "R_orthogonal_raw", "detrended_robust_z_scale"],
            label_columns=["short_name"],
            description="Raw points for normalization-floor artifact controls.",
            origin_action=f"Set {CURRENT_FLOOR_PRIMARY_METRIC} as X and choose one control metric as Y.",
        )

    # PCA, MDS, soft membership, and GMM source data.
    if pca_expl is not None and not pca_expl.empty:
        _save_origin_source_table(
            out_dir,
            "whole_map_pca_explained_variance",
            "pca_explained_variance_raw",
            pca_expl,
            plot_type="line_bar_source",
            x_columns=["PC"],
            y_columns=["explained_variance_ratio", "cumulative"],
            description="Raw PCA explained-variance table.",
            origin_action="Use PC as X. Plot explained_variance_ratio and/or cumulative as Y after multiplying by 100 if percent is desired.",
        )
    if pca_scores is not None and not pca_scores.empty:
        pca_plot = pca_scores.merge(_origin_select_columns(coord, ["name", "short_name", "theta_beta_over_AH_deg_norm", "R_orthogonal_norm"]), on="name", how="left") if "name" in pca_scores.columns else pca_scores.copy()
        _save_origin_source_table(
            out_dir,
            "whole_map_pca_scores_pc1_pc2",
            "pca_scores_raw",
            pca_plot,
            plot_type="scatter",
            x_columns=["PC1"],
            y_columns=["PC2"],
            label_columns=["short_name", "theta_beta_over_AH_deg_norm"],
            description="Raw PCA scores used for PC1/PC2 scatter.",
            origin_action="Set PC1 as X and PC2 as Y. Optional: color by theta_beta_over_AH_deg_norm.",
        )
    if mds is not None:
        mds_df = pd.DataFrame({"name": list(names), "short_name": [short_name(x) for x in names], "MDS1": mds[:, 0], "MDS2": mds[:, 1]})
        if "theta_beta_over_AH_deg_norm" in coord.columns:
            mds_df = mds_df.merge(_origin_select_columns(coord, ["name", "theta_beta_over_AH_deg_norm", "R_orthogonal_norm"]), on="name", how="left")
        _save_origin_source_table(
            out_dir,
            "whole_map_classical_mds_embedding",
            "mds_coordinates_raw",
            mds_df,
            plot_type="scatter",
            x_columns=["MDS1"],
            y_columns=["MDS2"],
            label_columns=["short_name", "theta_beta_over_AH_deg_norm"],
            description="Raw MDS coordinates used for the whole-map MDS embedding.",
            origin_action="Set MDS1 as X and MDS2 as Y. Optional: color by theta_beta_over_AH_deg_norm.",
        )
    if mem_df is not None and not mem_df.empty:
        mem_plot = mem_df.merge(_origin_select_columns(coord, ["name", "short_name", "theta_beta_over_AH_deg_norm"]), on="name", how="left")
        _save_origin_source_table(
            out_dir,
            "soft_membership_probabilities",
            "soft_membership_probabilities_wide_raw",
            mem_plot,
            plot_type="stacked_bar_source",
            x_columns=["short_name"],
            y_columns=[c for c in mem_plot.columns if c.startswith("P_")],
            label_columns=["soft_label_top", "theta_beta_over_AH_deg_norm"],
            description="Wide-format raw soft membership probabilities used for stacked bar plot.",
            origin_action="Sort by theta_beta_over_AH_deg_norm if desired, then plot P_* columns as stacked bars.",
        )
        prob_cols = [c for c in mem_plot.columns if c.startswith("P_")]
        if prob_cols:
            long = mem_plot.melt(id_vars=[c for c in ["name", "short_name", "theta_beta_over_AH_deg_norm", "soft_label_top"] if c in mem_plot.columns], value_vars=prob_cols, var_name="prototype_probability", value_name="probability")
            _save_origin_source_table(
                out_dir,
                "soft_membership_probabilities",
                "soft_membership_probabilities_long_raw",
                long,
                plot_type="long_stacked_bar_source",
                x_columns=["short_name"],
                y_columns=["probability"],
                label_columns=["prototype_probability"],
                description="Long-format soft membership probabilities for grouped/stacked plotting.",
                origin_action="Use short_name as categorical X, probability as Y, and prototype_probability as group/stack.",
            )
    if gmm_df is not None and not gmm_df.empty:
        _save_origin_source_table(
            out_dir,
            "gmm_bic_aic_optional_cluster_check",
            "gmm_bic_aic_raw",
            gmm_df,
            plot_type="line",
            x_columns=["k"],
            y_columns=["BIC", "AIC"],
            description="Raw GMM AIC/BIC values.",
            origin_action="Set k as X and BIC/AIC as Y columns.",
        )

    # Pairwise correlation heatmap source.
    if corr is not None:
        corr_mat = pd.DataFrame(corr, index=names, columns=names)
        corr_mat.insert(0, "name", names)
        _save_origin_source_table(
            out_dir,
            "whole_map_pairwise_correlation_heatmap",
            "pairwise_correlation_matrix_raw",
            corr_mat.reset_index(drop=True),
            plot_type="matrix_heatmap_source",
            description="Pairwise whole-map correlation matrix. Use as matrix heatmap in Origin.",
            origin_action="Import as matrix-like worksheet; use row/column names as labels if needed.",
        )
        rows = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                rows.append({"name_i": ni, "short_name_i": short_name(ni), "name_j": nj, "short_name_j": short_name(nj), "i": i, "j": j, "correlation": float(corr[i, j])})
        _save_origin_source_table(
            out_dir,
            "whole_map_pairwise_correlation_heatmap",
            "pairwise_correlation_long_raw",
            pd.DataFrame(rows),
            plot_type="xyz_heatmap_source",
            x_columns=["i"],
            y_columns=["j"],
            z_columns=["correlation"],
            label_columns=["short_name_i", "short_name_j"],
            description="Long-format pairwise correlation values for Origin heatmap/contour.",
            origin_action="Use i as X, j as Y, and correlation as Z/color.",
        )

    # Lattice/validation dashboard source data.
    if lattice_df is not None and not lattice_df.empty:
        _save_origin_source_table(
            out_dir,
            "unitcell_capture_validation_dashboard",
            "unitcell_capture_validation_raw",
            lattice_df,
            plot_type="validation_scatter_hist_source",
            x_columns=["fold_bin_coverage_fraction", "expected_mag_cycles_per_nm", "lattice_angle_deg"],
            y_columns=["unitcell_split_half_corr", "fft_peak1_mag_cycles_per_nm", "fft_peak2_mag_cycles_per_nm"],
            label_columns=["name"],
            description="Raw lattice and unit-cell capture validation metrics.",
            origin_action="Use selected columns to recreate validation histograms/scatters.",
        )

    if origin_search_df is not None and not origin_search_df.empty:
        _save_origin_source_table(
            out_dir,
            "site_origin_grid_search",
            "site_origin_candidates_raw",
            origin_search_df,
            plot_type="diagnostic_table",
            x_columns=["origin_u"],
            y_columns=["origin_v", "score"],
            label_columns=["name", "basis"],
            description="Raw site-origin candidate table used for per-image A/B/H detection diagnostics.",
        )

    # Unit-cell map source data as true XYZ values, not pcolormesh artist edges.
    try:
        total_rows = int(np.asarray(aligned_norm).shape[0] * np.asarray(aligned_norm).shape[1] * np.asarray(aligned_norm).shape[2])
    except Exception:
        total_rows = 0
    if total_rows and total_rows <= int(globals().get("ORIGIN_SOURCE_UNITCELL_MAX_LONG_ROWS", 1000000)):
        maps_long = _unitcell_maps_long_dataframe(aligned_norm, names, z_col="z_norm")
        _save_origin_source_table(
            out_dir,
            "aligned_unit_cells_with_BAH_overlay",
            "aligned_unit_cell_maps_xyz_raw",
            maps_long,
            plot_type="xyz_colormap_source",
            x_columns=["x_graphite60"],
            y_columns=["y_graphite60"],
            z_columns=["z_norm"],
            label_columns=["short_name"],
            description="Long-format aligned normalized unit-cell map values. These are true map values, not image pixels from Matplotlib.",
            origin_action="Filter one image at a time; use x_graphite60 as X, y_graphite60 as Y, z_norm as Z/color for contour/colormap.",
        )
    else:
        _save_origin_source_table(
            out_dir,
            "aligned_unit_cells_with_BAH_overlay",
            "aligned_unit_cell_maps_skipped_note",
            pd.DataFrame([{"n_rows_requested": total_rows, "max_rows": int(globals().get("ORIGIN_SOURCE_UNITCELL_MAX_LONG_ROWS", 1000000)), "note": "Long-format map export skipped because it would be too large. Increase ORIGIN_SOURCE_UNITCELL_MAX_LONG_ROWS if needed."}]),
            plot_type="note",
            description="Note explaining why long unit-cell map data were not exported.",
        )

    site_rows = []
    for i, nm in enumerate(names):
        if i >= len(positions_role_by_index):
            continue
        pos_i = positions_role_by_index[i]
        for site in ("B", "A", "H"):
            if site not in pos_i:
                continue
            u, v = pos_i[site]
            xg, yg = graphite_uv_to_xy(float(u), float(v))
            site_rows.append({"image_index": i, "name": nm, "short_name": short_name(nm), "site": site, "u": u, "v": v, "x_graphite60": float(xg), "y_graphite60": float(yg)})
    if site_rows:
        _save_origin_source_table(
            out_dir,
            "aligned_unit_cells_with_BAH_overlay",
            "BAH_marker_positions_raw",
            pd.DataFrame(site_rows),
            plot_type="marker_overlay_source",
            x_columns=["x_graphite60"],
            y_columns=["y_graphite60"],
            label_columns=["site", "short_name"],
            description="B/A/H marker coordinates used for unit-cell overlay figures.",
            origin_action="Overlay these X/Y marker positions on the corresponding unit-cell map; group by image/name.",
        )

    # Theta-quantile map source data.
    try:
        qdf, avg_maps = compute_theta_quantile_table(aligned_norm, coord, out_dir=None)
        if qdf is not None and not qdf.empty:
            _save_origin_source_table(
                out_dir,
                "quantile_ordered_unit_cells_by_theta",
                "theta_quantile_metadata_raw",
                qdf,
                plot_type="metadata_table",
                x_columns=["percentile"],
                y_columns=["theta_percentile_deg"],
                description="Theta quantile metadata used to build quantile-ordered unit-cell maps.",
            )
        if avg_maps:
            parts = []
            for unit_map, (_, row) in zip(avg_maps, qdf.iterrows()):
                arr = np.asarray(unit_map, dtype=float)
                Uc, Vc, Xc, Yc = graphite_grid_centers(arr.shape[0])
                tmp = pd.DataFrame({
                    "percentile": int(row["percentile"]),
                    "theta_percentile_deg": float(row["theta_percentile_deg"]),
                    "u_center": Uc.ravel(),
                    "v_center": Vc.ravel(),
                    "x_graphite60": Xc.ravel(),
                    "y_graphite60": Yc.ravel(),
                    "z": arr.ravel(),
                })
                parts.append(tmp)
            qmap_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            _save_origin_source_table(
                out_dir,
                "quantile_ordered_unit_cells_by_theta",
                "theta_quantile_avg_maps_xyz_raw",
                qmap_long,
                plot_type="xyz_colormap_source",
                x_columns=["x_graphite60"],
                y_columns=["y_graphite60"],
                z_columns=["z"],
                label_columns=["percentile", "theta_percentile_deg"],
                description="Long-format theta-quantile averaged unit-cell maps for Origin colormap plotting.",
                origin_action="Filter by percentile; use x_graphite60 as X, y_graphite60 as Y, z as Z/color.",
            )
    except Exception as exc:
        warnings.warn(f"Origin source theta-quantile export skipped: {exc}")

    index_df = _flush_origin_source_index(out_dir)
    _compile_origin_source_workbook(out_dir, index_df)
    _write_origin_source_readme(out_dir, index_df)


def _finite_values(x) -> np.ndarray:
    vals = np.asarray(x, dtype=float).ravel()
    return vals[np.isfinite(vals)]


def _fmt_float(x, nd: int = 4) -> str:
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{nd}g}"


def _offdiag_values(mat: Optional[np.ndarray]) -> np.ndarray:
    if mat is None:
        return np.array([], dtype=float)
    arr = np.asarray(mat, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return np.array([], dtype=float)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return arr[mask]


def _metric_summary(summary_df: pd.DataFrame, metric: str) -> Dict[str, float]:
    if summary_df is None or summary_df.empty or "metric" not in summary_df:
        return {}
    rows = summary_df.loc[summary_df["metric"] == metric]
    if rows.empty:
        return {}
    r = rows.iloc[0]
    return {k: float(r[k]) for k in r.index if k != "metric" and isinstance(r[k], (int, float, np.integer, np.floating))}


def _spearman(x, y) -> Tuple[float, float]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = np.isfinite(xx) & np.isfinite(yy)
    if m.sum() < 4:
        return np.nan, np.nan
    try:
        res = spearmanr(xx[m], yy[m])
        return float(res.correlation), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def make_alignment_quality_table(corr_before: np.ndarray, corr_after: np.ndarray, shifts: Sequence[Tuple[int, int]]) -> pd.DataFrame:
    b = _offdiag_values(corr_before)
    a = _offdiag_values(corr_after)
    shift_mag = np.asarray([math.sqrt(float(s[0]) ** 2 + float(s[1]) ** 2) for s in shifts], dtype=float)
    return pd.DataFrame([{
        "median_pairwise_corr_before_alignment": float(np.nanmedian(b)) if b.size else np.nan,
        "median_pairwise_corr_after_alignment": float(np.nanmedian(a)) if a.size else np.nan,
        "mean_pairwise_corr_before_alignment": float(np.nanmean(b)) if b.size else np.nan,
        "mean_pairwise_corr_after_alignment": float(np.nanmean(a)) if a.size else np.nan,
        "delta_median_corr_after_minus_before": float(np.nanmedian(a) - np.nanmedian(b)) if b.size and a.size else np.nan,
        "shift_magnitude_px_median": float(np.nanmedian(shift_mag)) if shift_mag.size else np.nan,
        "shift_magnitude_px_max": float(np.nanmax(shift_mag)) if shift_mag.size else np.nan,
    }])


def make_unitcell_capture_quality_outputs(lattice_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Save drift-aware unit-cell capture validation summaries and plots."""
    if lattice_df.empty:
        return pd.DataFrame()
    df = lattice_df.copy()
    rows = []
    def add(metric: str, values: np.ndarray, note: str = "") -> None:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append({
            "metric": metric,
            "n_valid": int(vals.size),
            "mean": float(np.nanmean(vals)) if vals.size else np.nan,
            "median": float(np.nanmedian(vals)) if vals.size else np.nan,
            "std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan,
            "p05": float(np.nanpercentile(vals, 5)) if vals.size else np.nan,
            "p25": float(np.nanpercentile(vals, 25)) if vals.size else np.nan,
            "p75": float(np.nanpercentile(vals, 75)) if vals.size else np.nan,
            "p95": float(np.nanpercentile(vals, 95)) if vals.size else np.nan,
            "note": note,
        })
    for col, note in [
        ("fold_bin_coverage_fraction", "Unit-cell bin coverage for full folded map."),
        ("unitcell_split_half_corr", "Correlation between independently folded alternating apparent unit-cell subsets."),
        ("unitcell_split_half_min_coverage", "Minimum coverage between the two split halves."),
        ("unitcell_split_half_corr_to_full_0", "Half-0 folded map correlation to full folded map."),
        ("unitcell_split_half_corr_to_full_1", "Half-1 folded map correlation to full folded map."),
        ("fft_peak1_mag_cycles_per_nm", "Selected first FFT peak magnitude."),
        ("fft_peak2_mag_cycles_per_nm", "Selected second FFT peak magnitude."),
        ("lattice_a1_nm", "Apparent real-space vector length; drift/shear diagnostic, not structural claim."),
        ("lattice_a2_nm", "Apparent real-space vector length; drift/shear diagnostic, not structural claim."),
        ("lattice_angle_deg", "Apparent angle; slow-scan drift/shear diagnostic, not structural pass/fail."),
    ]:
        if col in df.columns:
            add(col, df[col].to_numpy(dtype=float), note)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "unitcell_capture_validation_summary.csv", index=False, encoding="utf-8-sig")

    # Per-image compact table with explicit flags.
    flag_df = pd.DataFrame()
    needed = ["name", "fold_bin_coverage_fraction", "unitcell_split_half_corr", "unitcell_split_half_min_coverage", "fft_peak1_mag_cycles_per_nm", "fft_peak2_mag_cycles_per_nm", "expected_mag_cycles_per_nm", "lattice_a1_nm", "lattice_a2_nm", "lattice_angle_deg"]
    have = [c for c in needed if c in df.columns]
    if have:
        flag_df = df[have].copy()
        if "fold_bin_coverage_fraction" in flag_df:
            flag_df["flag_low_full_fold_coverage"] = flag_df["fold_bin_coverage_fraction"] < MIN_UNITCELL_BIN_COVERAGE_FRACTION
        if "unitcell_split_half_corr" in flag_df:
            flag_df["flag_low_split_half_corr"] = flag_df["unitcell_split_half_corr"] < MIN_SPLIT_HALF_CORR_MEDIAN
        if "unitcell_split_half_min_coverage" in flag_df:
            flag_df["flag_low_split_half_coverage"] = flag_df["unitcell_split_half_min_coverage"] < MIN_SPLIT_HALF_COVERAGE_FRACTION
        if {"fft_peak1_mag_cycles_per_nm", "fft_peak2_mag_cycles_per_nm", "expected_mag_cycles_per_nm"}.issubset(flag_df.columns):
            e = flag_df["expected_mag_cycles_per_nm"].replace(0, np.nan)
            flag_df["fft_peak1_frac_error"] = (flag_df["fft_peak1_mag_cycles_per_nm"] - e).abs() / e
            flag_df["fft_peak2_frac_error"] = (flag_df["fft_peak2_mag_cycles_per_nm"] - e).abs() / e
            flag_df["flag_fft_frequency_outside_broad_graphite_band"] = (flag_df["fft_peak1_frac_error"] > RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION) | (flag_df["fft_peak2_frac_error"] > RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION)
        flag_cols = [c for c in flag_df.columns if c.startswith("flag_")]
        if flag_cols:
            flag_df["n_unitcell_capture_flags"] = flag_df[flag_cols].sum(axis=1)
        flag_df.to_csv(out_dir / "unitcell_capture_validation_per_image.csv", index=False, encoding="utf-8-sig")

    # Save raw distributions behind the histogram panels in the dashboard.
    try:
        dash_rows = []
        id_cols = [c for c in ["name", "short_name"] if c in df.columns]
        for metric, desc in [
            ("unitcell_split_half_corr", "Split-half folded-map correlation values used in dashboard histogram."),
            ("lattice_angle_deg", "Apparent real-space lattice angle values used in dashboard histogram; drift/shear diagnostic."),
        ]:
            if metric in df.columns:
                tmp = df[id_cols + [metric]].copy() if id_cols else df[[metric]].copy()
                tmp = tmp.rename(columns={metric: "value"})
                tmp["metric"] = metric
                tmp["metric_description"] = desc
                dash_rows.append(tmp)
        if dash_rows:
            _save_raw_distribution_table(
                out_dir,
                "unitcell_capture_validation_dashboard_histograms",
                pd.concat(dash_rows, ignore_index=True),
                "Raw values behind histogram panels in the unit-cell capture validation dashboard.",
                value_column="value",
                suggested_bins=30,
            )
    except Exception:
        pass

    # Diagnostic plot.  Subplots are used inside the saved analysis artifact to
    # keep related validation metrics together.
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8))
        ax = axes[0, 0]
        if "unitcell_split_half_corr" in df:
            ax.hist(df["unitcell_split_half_corr"].dropna().to_numpy(dtype=float), bins=30)
            ax.axvline(MIN_SPLIT_HALF_CORR_MEDIAN, linestyle="--", linewidth=1)
            ax.set_xlabel("split-half folded-map correlation")
            ax.set_ylabel("count")
            ax.set_title("Repeatability of folded unit-cell motif")
        ax = axes[0, 1]
        if {"fold_bin_coverage_fraction", "unitcell_split_half_corr"}.issubset(df.columns):
            ax.scatter(df["fold_bin_coverage_fraction"], df["unitcell_split_half_corr"], s=12, alpha=0.65)
            ax.axhline(MIN_SPLIT_HALF_CORR_MEDIAN, linestyle="--", linewidth=1)
            ax.axvline(MIN_UNITCELL_BIN_COVERAGE_FRACTION, linestyle="--", linewidth=1)
            ax.set_xlabel("full fold bin coverage")
            ax.set_ylabel("split-half corr")
            ax.set_title("Coverage vs. split-half reliability")
        ax = axes[1, 0]
        if {"fft_peak1_mag_cycles_per_nm", "fft_peak2_mag_cycles_per_nm", "expected_mag_cycles_per_nm"}.issubset(df.columns):
            e = df["expected_mag_cycles_per_nm"].to_numpy(dtype=float)
            ax.scatter(e, df["fft_peak1_mag_cycles_per_nm"], s=12, alpha=0.65, label="G1")
            ax.scatter(e, df["fft_peak2_mag_cycles_per_nm"], s=12, alpha=0.65, label="G2")
            lo = np.nanmin(e) * (1 - RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION)
            hi = np.nanmax(e) * (1 + RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION)
            ax.plot([lo, hi], [lo, hi], linewidth=1)
            ax.set_xlabel("expected graphite |G| cycles/nm")
            ax.set_ylabel("selected FFT |G| cycles/nm")
            ax.set_title("FFT peak frequency sanity check")
            ax.legend(fontsize=8)
        ax = axes[1, 1]
        if "lattice_angle_deg" in df:
            ax.hist(df["lattice_angle_deg"].dropna().to_numpy(dtype=float), bins=30)
            ax.set_xlabel("apparent real-space angle (deg)")
            ax.set_ylabel("count")
            ax.set_title("Apparent angle: AFM drift/shear diagnostic")
        fig.tight_layout()
        savefig(out_dir / "unitcell_capture_validation_dashboard.png")
    except Exception:
        pass
    return summary


def make_normalization_robustness_table(coord_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("c_AH", "c_AH_norm", "c_AH_rank"),
        ("abs_c_beta", "abs_c_beta_norm", "abs_c_beta_rank"),
        ("theta", "theta_beta_over_AH_deg_norm", "theta_beta_over_AH_deg_rank"),
        ("R", "R_orthogonal_norm", "R_orthogonal_rank"),
    ]
    for label, cn, cr in pairs:
        if cn in coord_df and cr in coord_df:
            rho, pval = _spearman(coord_df[cn], coord_df[cr])
            pear = np.corrcoef(coord_df[cn].to_numpy(dtype=float), coord_df[cr].to_numpy(dtype=float))[0, 1]
            rows.append({
                "metric": label,
                "pearson_corr_robust_z_vs_rank": float(pear) if np.isfinite(pear) else np.nan,
                "spearman_corr_robust_z_vs_rank": rho,
                "spearman_p": pval,
                "passes_threshold": bool(np.isfinite(rho) and abs(rho) >= MIN_NORMALIZATION_ROBUSTNESS_CORR),
            })
    return pd.DataFrame(rows)


def make_artifact_correlation_table(coord_df: pd.DataFrame, lattice_df: pd.DataFrame) -> pd.DataFrame:
    if lattice_df.empty:
        return pd.DataFrame()
    merged = coord_df.merge(lattice_df, on="name", how="left", suffixes=("", "_lattice"))
    motif_metrics = [
        "c_AH_norm",
        "abs_c_beta_norm",
        "R_orthogonal_norm",
        "theta_beta_over_AH_deg_norm",
        "delta_BA_norm",
        "delta_AH_norm",
    ]
    diagnostic_metrics = [
        "lattice_a1_nm",
        "lattice_a2_nm",
        "lattice_angle_deg",
        "fft_peak1_mag_cycles_per_nm",
        "fft_peak2_mag_cycles_per_nm",
        "fold_bin_coverage_fraction",
        "unitcell_split_half_corr",
        "unitcell_split_half_min_coverage",
        "unitcell_shift_v_px",
        "unitcell_shift_u_px",
    ]
    rows = []
    for m in motif_metrics:
        if m not in merged:
            continue
        for d in diagnostic_metrics:
            if d not in merged:
                continue
            rho, pval = _spearman(merged[m], merged[d])
            if np.isfinite(rho):
                rows.append({
                    "motif_metric": m,
                    "diagnostic_metric": d,
                    "spearman_rho": rho,
                    "spearman_p": pval,
                    "abs_rho": abs(rho),
                    "caution_level": "strong" if abs(rho) >= ARTIFACT_SPEARMAN_STRONG_ABS_RHO else ("caution" if abs(rho) >= ARTIFACT_SPEARMAN_CAUTION_ABS_RHO else "ok"),
                })
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False) if rows else pd.DataFrame()


def make_validation_outputs(
    out_dir: Path,
    records: Sequence[ImageRecord],
    lattice_rows: Sequence[Dict[str, object]],
    coord_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    site_choice: Dict[str, object],
    mem_df: pd.DataFrame,
    pca_expl: pd.DataFrame,
    gmm_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    origin_search_df: pd.DataFrame,
    alignment_quality_df: pd.DataFrame,
    normalization_robustness_df: pd.DataFrame,
) -> Dict[str, object]:
    """Create validation checks, key-number tables, and text summaries."""
    n = len(coord_df)
    lattice_df = pd.DataFrame(lattice_rows)
    unitcell_capture_summary_df = make_unitcell_capture_quality_outputs(lattice_df, out_dir) if not lattice_df.empty else pd.DataFrame()
    coord_df_local = coord_df.copy()
    if not shift_df.empty:
        coord_df_local = coord_df_local.merge(shift_df, on="name", how="left", suffixes=("", "_align"))

    def add_check(rows, category, check, value, threshold, status, interpretation):
        rows.append({
            "category": category,
            "check": check,
            "value": value,
            "threshold_or_reference": threshold,
            "status": status,
            "interpretation": interpretation,
        })

    validation_rows: List[Dict[str, object]] = []

    # Image extraction / processing.
    add_check(validation_rows, "input", "processed_image_count", n, ">= 5 recommended for ensemble statistics", "PASS" if n >= 5 else "CAUTION", "Number of folded maps used for ensemble statistics.")

    # Drift-aware lattice checks.
    if not lattice_df.empty:
        a1 = lattice_df.get("lattice_a1_nm", pd.Series(dtype=float)).to_numpy(dtype=float)
        a2 = lattice_df.get("lattice_a2_nm", pd.Series(dtype=float)).to_numpy(dtype=float)
        len_ok = np.isfinite(a1) & np.isfinite(a2) & (a1 >= LATTICE_LENGTH_MIN_NM) & (a1 <= LATTICE_LENGTH_MAX_NM) & (a2 >= LATTICE_LENGTH_MIN_NM) & (a2 <= LATTICE_LENGTH_MAX_NM)
        len_frac = float(np.nanmean(len_ok)) if len_ok.size else np.nan
        add_check(validation_rows, "lattice", "drift_aware_lattice_length_fraction", len_frac, f">= {MIN_LATTICE_LENGTH_PASS_FRACTION:.2f}; angle reported only as drift diagnostic", "PASS" if len_frac >= MIN_LATTICE_LENGTH_PASS_FRACTION else "CAUTION", "Apparent lattice lengths remain in a broad graphite-compatible range after allowing AFM drift/shear.")

        exp_col = lattice_df.get("expected_mag_cycles_per_nm", pd.Series(np.nan, index=lattice_df.index)).to_numpy(dtype=float)
        m1 = lattice_df.get("fft_peak1_mag_cycles_per_nm", pd.Series(np.nan, index=lattice_df.index)).to_numpy(dtype=float)
        m2 = lattice_df.get("fft_peak2_mag_cycles_per_nm", pd.Series(np.nan, index=lattice_df.index)).to_numpy(dtype=float)
        mag_ok = np.isfinite(exp_col) & np.isfinite(m1) & np.isfinite(m2) & (np.abs(m1 - exp_col) / exp_col <= RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION) & (np.abs(m2 - exp_col) / exp_col <= RECIPROCAL_MAG_FRACTIONAL_TOL_FOR_VALIDATION)
        mag_frac = float(np.nanmean(mag_ok)) if mag_ok.size else np.nan
        add_check(validation_rows, "lattice", "fft_peak_magnitude_fraction", mag_frac, f">= {MIN_RECIPROCAL_MAG_PASS_FRACTION:.2f}; expected graphite reciprocal radius with broad drift tolerance", "PASS" if mag_frac >= MIN_RECIPROCAL_MAG_PASS_FRACTION else "CAUTION", "Checks that FFT peaks are near the graphite lattice frequency rather than very-low-frequency drift/background.")

        coverage = lattice_df.get("fold_bin_coverage_fraction", pd.Series(dtype=float)).to_numpy(dtype=float)
        cov_med = float(np.nanmedian(coverage)) if coverage.size else np.nan
        add_check(validation_rows, "folding", "median_unitcell_bin_coverage", cov_med, f">= {MIN_UNITCELL_BIN_COVERAGE_FRACTION:.2f}", "PASS" if np.isfinite(cov_med) and cov_med >= MIN_UNITCELL_BIN_COVERAGE_FRACTION else "CAUTION", "Fraction of 64x64 unit-cell bins receiving at least one pixel during folding.")

        # Split-half reliability: this is the most direct check that repeated
        # apparent unit cells in the same image fold to the same motif.
        split_corr = lattice_df.get("unitcell_split_half_corr", pd.Series(dtype=float)).to_numpy(dtype=float)
        split_cov = lattice_df.get("unitcell_split_half_min_coverage", pd.Series(dtype=float)).to_numpy(dtype=float)
        split_corr_med = float(np.nanmedian(split_corr)) if split_corr.size else np.nan
        split_cov_med = float(np.nanmedian(split_cov)) if split_cov.size else np.nan
        split_valid = np.isfinite(split_corr) & np.isfinite(split_cov) & (split_cov >= MIN_SPLIT_HALF_COVERAGE_FRACTION)
        split_pass_frac = float(np.nanmean(split_valid & (split_corr >= MIN_SPLIT_HALF_CORR_MEDIAN))) if split_corr.size else np.nan
        status_corr = "PASS" if np.isfinite(split_corr_med) and split_corr_med >= MIN_SPLIT_HALF_CORR_MEDIAN else "CAUTION"
        add_check(validation_rows, "folding", "split_half_unitcell_corr_median", split_corr_med, f">= {MIN_SPLIT_HALF_CORR_MEDIAN:.2f} suggested; >={MIN_SPLIT_HALF_CORR_STRONG:.2f} strong", status_corr, "Alternating apparent unit-cell subsets are folded independently; correlation tests whether the same motif repeats within images.")
        add_check(validation_rows, "folding", "split_half_min_coverage_median", split_cov_med, f">= {MIN_SPLIT_HALF_COVERAGE_FRACTION:.2f}", "PASS" if np.isfinite(split_cov_med) and split_cov_med >= MIN_SPLIT_HALF_COVERAGE_FRACTION else "CAUTION", "Each split half should still cover enough unit-cell bins for a meaningful split-half correlation.")
        add_check(validation_rows, "folding", "split_half_reliable_fraction", split_pass_frac, f"corr>={MIN_SPLIT_HALF_CORR_MEDIAN:.2f} and coverage>={MIN_SPLIT_HALF_COVERAGE_FRACTION:.2f}", "PASS" if np.isfinite(split_pass_frac) and split_pass_frac >= 0.50 else "CAUTION", "Fraction of images with usable split-half coverage and repeatable folded motif.")

    # Alignment.
    if not alignment_quality_df.empty:
        aq = alignment_quality_df.iloc[0]
        delta_corr = float(aq.get("delta_median_corr_after_minus_before", np.nan))
        status = "PASS" if np.isfinite(delta_corr) and delta_corr >= -0.05 else "CAUTION"
        add_check(validation_rows, "alignment", "median_pairwise_correlation_change", delta_corr, ">= -0.05; exact increase not required for diverse motifs", status, "Alignment should not substantially degrade whole-unit-cell motif similarity.")

    # B/A/H ordering.
    if {"I_B_norm", "I_A_norm", "I_H_norm"}.issubset(coord_df.columns):
        order_ok = (coord_df["I_B_norm"] > coord_df["I_A_norm"]) & (coord_df["I_A_norm"] > coord_df["I_H_norm"])
        order_frac = float(order_ok.mean())
        add_check(validation_rows, "site_roles", "B_greater_A_greater_H_fraction", order_frac, f">= {MIN_SITE_ORDERING_FRACTION:.2f}", "PASS" if order_frac >= MIN_SITE_ORDERING_FRACTION else "CAUTION", "Checks whether the ensemble-derived B/A/H labels is actually reproduced after folding.")

    # Bootstrap site confidence.
    if not ci_df.empty and {"delta_BA_norm_ci95_low", "delta_AH_norm_ci95_low"}.issubset(ci_df.columns):
        dba_sig = (ci_df["delta_BA_norm_ci95_low"] > 0).mean()
        dah_sig = (ci_df["delta_AH_norm_ci95_low"] > 0).mean()
        add_check(validation_rows, "site_uncertainty", "delta_BA_CI_positive_fraction", float(dba_sig), ">= 0.70 suggested", "PASS" if dba_sig >= 0.70 else "CAUTION", "Fraction of images where bootstrap CI supports B brighter than A.")
        add_check(validation_rows, "site_uncertainty", "delta_AH_CI_positive_fraction", float(dah_sig), ">= 0.70 suggested", "PASS" if dah_sig >= 0.70 else "CAUTION", "Fraction of images where bootstrap CI supports A above H.")

    # Origin-search stability.
    if not origin_search_df.empty and len(origin_search_df) >= 2:
        top = origin_search_df.sort_values("score", ascending=False).head(10).copy()
        score1 = float(top.iloc[0]["score"])
        score2 = float(top.iloc[1]["score"]) if len(top) > 1 else np.nan
        med_top = float(np.nanmedian(top["score"]))
        sep = score1 - score2 if np.isfinite(score2) else np.nan
        add_check(validation_rows, "site_origin", "top_origin_score_separation", sep, "larger is better; inspect top_origin_candidates.csv if small", "PASS" if np.isfinite(sep) and sep > max(0.02, 0.02 * abs(score1)) else "CAUTION", "How clearly the best B/A/H origin wins over the next candidate.")
        top.to_csv(out_dir / "top_site_origin_candidates.csv", index=False, encoding="utf-8-sig")

    # Normalization robustness.
    if not normalization_robustness_df.empty:
        min_rho = float(np.nanmin(np.abs(normalization_robustness_df["spearman_corr_robust_z_vs_rank"])))
        add_check(validation_rows, "normalization", "min_abs_spearman_rank_vs_robust_z", min_rho, f">= {MIN_NORMALIZATION_ROBUSTNESS_CORR:.2f}", "PASS" if min_rho >= MIN_NORMALIZATION_ROBUSTNESS_CORR else "CAUTION", "Descriptor trends should be similar under robust-z and rank normalization.")

    # Continuum/hard-cluster checks.
    if not mem_df.empty:
        amb_frac = float(mem_df["soft_ambiguous"].mean())
        add_check(validation_rows, "continuum", "soft_ambiguous_fraction", amb_frac, "descriptive; nonzero ambiguity supports soft/continuous interpretation", "INFO", "Soft-assignment ambiguity is shown rather than forced into hard labels.")
    if not gmm_df.empty:
        best = gmm_df.loc[gmm_df["BIC"].idxmin()]
        best_k = int(best["k"])
        bic1 = float(gmm_df.loc[gmm_df["k"] == 1, "BIC"].iloc[0]) if (gmm_df["k"] == 1).any() else np.nan
        bic2 = float(gmm_df.loc[gmm_df["k"] == 2, "BIC"].iloc[0]) if (gmm_df["k"] == 2).any() else np.nan
        delta_bic_2_minus_1 = bic2 - bic1 if np.isfinite(bic1) and np.isfinite(bic2) else np.nan
        add_check(validation_rows, "continuum", "GMM_BIC_best_k", best_k, "k=1 favors one connected distribution; k>1 means inspect scatter/KDE", "PASS" if best_k == 1 else "INFO", "Optional cluster check; not used as hard proof.")
        add_check(validation_rows, "continuum", "delta_BIC_k2_minus_k1", delta_bic_2_minus_1, "> 0 favors k=1 over k=2", "PASS" if np.isfinite(delta_bic_2_minus_1) and delta_bic_2_minus_1 > 0 else "INFO", "Positive ΔBIC means a two-cluster GMM is not preferred over one cluster.")

    # Artifact-correlation checks.
    artifact_df = make_artifact_correlation_table(coord_df_local, lattice_df)
    if not artifact_df.empty:
        artifact_df.to_csv(out_dir / "artifact_correlation_check_spearman.csv", index=False, encoding="utf-8-sig")
        max_abs = float(artifact_df["abs_rho"].max())
        worst = artifact_df.iloc[0]
        status = "PASS" if max_abs < ARTIFACT_SPEARMAN_CAUTION_ABS_RHO else ("CAUTION" if max_abs < ARTIFACT_SPEARMAN_STRONG_ABS_RHO else "STRONG_CAUTION")
        add_check(validation_rows, "artifact", "max_abs_spearman_motif_vs_diagnostic", max_abs, f"< {ARTIFACT_SPEARMAN_CAUTION_ABS_RHO:.2f} preferred", status, f"Worst correlation: {worst['motif_metric']} vs {worst['diagnostic_metric']}.")

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(out_dir / "analysis_validation_summary.csv", index=False, encoding="utf-8-sig")

    # Per-image suspicious/diagnostic table.
    suspicious = coord_df.copy()
    suspicious["flag_site_order_not_B_gt_A_gt_H"] = ~((suspicious["I_B_norm"] > suspicious["I_A_norm"]) & (suspicious["I_A_norm"] > suspicious["I_H_norm"]))
    if not ci_df.empty:
        suspicious = suspicious.merge(ci_df, on="name", how="left", suffixes=("", "_ci"))
        if "delta_BA_norm_ci95_low" in suspicious:
            suspicious["flag_delta_BA_CI_not_positive"] = ~(suspicious["delta_BA_norm_ci95_low"] > 0)
        if "delta_AH_norm_ci95_low" in suspicious:
            suspicious["flag_delta_AH_CI_not_positive"] = ~(suspicious["delta_AH_norm_ci95_low"] > 0)
    if not mem_df.empty:
        suspicious = suspicious.merge(mem_df[["name", "soft_label_top", "soft_margin_top_minus_second", "soft_entropy", "soft_ambiguous"]], on="name", how="left")
    flag_cols = [c for c in suspicious.columns if c.startswith("flag_")]
    if flag_cols:
        suspicious["n_flags"] = suspicious[flag_cols].sum(axis=1)
        suspicious.sort_values(["n_flags", "theta_beta_over_AH_deg_norm"], ascending=[False, True]).to_csv(out_dir / "per_image_validation_flags.csv", index=False, encoding="utf-8-sig")

    # Key numbers for manuscript/results.
    def sm(metric, key="mean"):
        d = _metric_summary(summary_df, metric)
        return d.get(key, np.nan)
    def smci(metric):
        d = _metric_summary(summary_df, metric)
        return d.get("mean_ci95_low", np.nan), d.get("mean_ci95_high", np.nan)

    theta_vals = _finite_values(coord_df["theta_beta_over_AH_deg_norm"] if "theta_beta_over_AH_deg_norm" in coord_df else [])
    hybrid_frac = float(np.mean((theta_vals >= THETA_AH_DOMINANT_MAX_DEG) & (theta_vals <= THETA_BETA_DOMINANT_MIN_DEG))) if theta_vals.size else np.nan
    ah_frac = float(np.mean(theta_vals < THETA_AH_DOMINANT_MAX_DEG)) if theta_vals.size else np.nan
    beta_frac = float(np.mean(theta_vals > THETA_BETA_DOMINANT_MIN_DEG)) if theta_vals.size else np.nan
    combined_tail_frac = float(ah_frac + beta_frac) if np.isfinite(ah_frac) and np.isfinite(beta_frac) else np.nan
    theta_quantiles = {int(p): (float(np.nanpercentile(theta_vals, p)) if theta_vals.size else np.nan) for p in THETA_QUANTILE_PERCENTILES}

    # Save theta distribution summary here as well, so it exists even if plotting is disabled.
    theta_summary_row = {
        "N_theta_valid": int(theta_vals.size),
        "theta_mean_deg": float(np.nanmean(theta_vals)) if theta_vals.size else np.nan,
        "theta_median_deg": float(np.nanmedian(theta_vals)) if theta_vals.size else np.nan,
        "theta_AH_like_tail_fraction_lt_20deg": ah_frac,
        "theta_hybrid_fraction_20_to_70deg": hybrid_frac,
        "theta_B_like_tail_fraction_gt_70deg": beta_frac,
        "theta_combined_tail_fraction": combined_tail_frac,
    }
    for p, v in theta_quantiles.items():
        theta_summary_row[f"theta_p{p:02d}_deg"] = v
    pd.DataFrame([theta_summary_row]).to_csv(out_dir / "theta_distribution_summary.csv", index=False, encoding="utf-8-sig")

    pca_pc1 = float(pca_expl["explained_variance_ratio"].iloc[0]) if not pca_expl.empty and len(pca_expl) >= 1 else np.nan
    pca_pc2 = float(pca_expl["explained_variance_ratio"].iloc[1]) if not pca_expl.empty and len(pca_expl) >= 2 else np.nan
    split_corr_vals = _finite_values(lattice_df["unitcell_split_half_corr"] if "unitcell_split_half_corr" in lattice_df else [])
    split_cov_vals = _finite_values(lattice_df["unitcell_split_half_min_coverage"] if "unitcell_split_half_min_coverage" in lattice_df else [])
    split_corr_median = float(np.nanmedian(split_corr_vals)) if split_corr_vals.size else np.nan
    split_corr_p25 = float(np.nanpercentile(split_corr_vals, 25)) if split_corr_vals.size else np.nan
    split_corr_p75 = float(np.nanpercentile(split_corr_vals, 75)) if split_corr_vals.size else np.nan
    split_cov_median = float(np.nanmedian(split_cov_vals)) if split_cov_vals.size else np.nan
    fold_cov_vals = _finite_values(lattice_df["fold_bin_coverage_fraction"] if "fold_bin_coverage_fraction" in lattice_df else [])
    fold_cov_median = float(np.nanmedian(fold_cov_vals)) if fold_cov_vals.size else np.nan
    pca_pc12 = float(pca_expl["explained_variance_ratio"].iloc[:2].sum()) if not pca_expl.empty else np.nan
    pca_pc15 = float(pca_expl["explained_variance_ratio"].iloc[:5].sum()) if not pca_expl.empty else np.nan

    key_rows = []
    def key(metric, value, unit, meaning):
        key_rows.append({"metric": metric, "value": value, "unit": unit, "meaning": meaning})

    key("N_processed_images", n, "images", "Number of same-condition atomic images used after folding.")
    key("site_coordinate_mode", SITE_COORDINATE_MODE, "", "A/B/H site-registration procedure.")
    key("site_registry_scope", site_choice.get("site_registry_scope", "mode_specific"), "", "Whether one registry or multiple acquisition-specific registries were used.")
    key("site_registry_calibration_source", site_choice.get("site_registry_calibration_source", ""), "", "Map statistic used to calibrate the primary A/B/H registry.")
    key("site_positions_fixed_across_acquisitions", bool(site_choice.get("site_positions_fixed_across_acquisitions", False)), "boolean", "True when identical fractional A/B/H positions were applied to every acquisition.")
    key("site_basis", site_choice.get("basis", "NA"), "", "Selected graphite two-atom + single-hollow geometric basis.")
    key("site_geometry_model", SITE_GEOMETRY_MODEL, "", "Graphite-correct site model with one geometric hollow per primitive cell.")
    key("site_mask_distance_metric", SITE_MASK_DISTANCE_METRIC, "", "Metric used for Gaussian site masks in fractional unit-cell coordinates.")
    key("origin_u", site_choice.get("origin", (np.nan, np.nan))[0], "fractional u", "Selected unit-cell origin.")
    key("origin_v", site_choice.get("origin", (np.nan, np.nan))[1], "fractional v", "Selected unit-cell origin.")
    key("role_assignment", f"B={site_choice['role']['B_key']}, A={site_choice['role']['A_key']}, H={site_choice['role']['H_key']}", "", "Ensemble-mean intensity-role mapping.")
    key("fold_bin_coverage_fraction_median", fold_cov_median, "fraction", "Median fraction of unit-cell grid bins receiving pixels during folding.")
    key("unitcell_split_half_corr_median", split_corr_median, "correlation", "Median correlation between independently folded alternating apparent unit-cell subsets; direct unit-cell capture reliability check.")
    key("unitcell_split_half_corr_IQR", f"[{_fmt_float(split_corr_p25)}, {_fmt_float(split_corr_p75)}]", "correlation", "IQR of split-half folded-map correlations.")
    key("unitcell_split_half_min_coverage_median", split_cov_median, "fraction", "Median minimum unit-cell bin coverage across the two split halves.")
    for metric, meaning in [
        ("c_AH_norm", "Mean apparent atom-vs-hollow/contact-registry coordinate."),
        ("abs_c_beta_norm", "Mean sublattice-polarization magnitude."),
        ("R_orthogonal_norm", "Mean motif contrast amplitude in orthogonal coordinate space."),
        ("theta_beta_over_AH_deg_norm", "Mean motif angle; intermediate values indicate A/B/H hybrid motifs."),
        ("delta_BA_norm", "Mean B-bright minus A-intermediate site contrast."),
        ("delta_AH_norm", "Mean A-intermediate minus H-low site contrast."),
    ]:
        lo, hi = smci(metric)
        key(metric + "_mean", sm(metric, "mean"), "normalized units" if "theta" not in metric else "deg", meaning)
        key(metric + "_CI95_mean", f"[{_fmt_float(lo)}, {_fmt_float(hi)}]", "", "Bootstrap 95% CI for ensemble mean.")
    cAH_mean = sm("c_AH_norm", "mean")
    cb_mean = sm("abs_c_beta_norm", "mean")
    key("abs_c_beta_over_c_AH_mean_ratio", cb_mean / cAH_mean if np.isfinite(cAH_mean) and cAH_mean != 0 else np.nan, "ratio", "Relative strength of sublattice polarization vs contact-registry coordinate.")
    key("theta_hybrid_fraction", hybrid_frac, "fraction", f"Fraction with {THETA_AH_DOMINANT_MAX_DEG:g}° <= θ <= {THETA_BETA_DOMINANT_MIN_DEG:g}°.")
    key("theta_AH_dominant_fraction", ah_frac, "fraction", f"Fraction with θ < {THETA_AH_DOMINANT_MAX_DEG:g}°.")
    key("theta_beta_dominant_fraction", beta_frac, "fraction", f"Fraction with θ > {THETA_BETA_DOMINANT_MIN_DEG:g}°.")
    key("theta_combined_tail_fraction", combined_tail_frac, "fraction", f"Fraction with θ < {THETA_AH_DOMINANT_MAX_DEG:g}° or θ > {THETA_BETA_DOMINANT_MIN_DEG:g}°.")
    for p in THETA_QUANTILE_PERCENTILES:
        key(f"theta_p{int(p):02d}_deg", theta_quantiles.get(int(p), np.nan), "deg", f"{int(p)}th percentile of motif angle θ; used for Fig. 3b/3c quantile ordering.")
    if not mem_df.empty:
        key("soft_ambiguous_fraction", float(mem_df["soft_ambiguous"].mean()), "fraction", "Soft assignment uncertainty; supports reporting uncertainty instead of hard labels.")
        key("soft_top_label_counts", str(mem_df["soft_label_top"].value_counts().to_dict()), "counts", "Prototype-nearest counts; descriptive, not hard cluster sizes.")
    if not gmm_df.empty:
        best = gmm_df.loc[gmm_df["BIC"].idxmin()]
        key("GMM_BIC_best_k", int(best["k"]), "components", "Optional check for separated Gaussian clusters; k=1 favors continuous cloud.")
    key("PCA_PC1_explained", 100 * pca_pc1 if np.isfinite(pca_pc1) else np.nan, "%", "Whole-map PCA PC1 explained variance.")
    key("PCA_PC2_explained", 100 * pca_pc2 if np.isfinite(pca_pc2) else np.nan, "%", "Whole-map PCA PC2 explained variance.")
    key("PCA_PC1_PC2_cumulative", 100 * pca_pc12 if np.isfinite(pca_pc12) else np.nan, "%", "Cumulative variance of first two whole-map PCs.")
    key("PCA_PC1_to_PC5_cumulative", 100 * pca_pc15 if np.isfinite(pca_pc15) else np.nan, "%", "Cumulative variance of first five whole-map PCs.")

    key_df = pd.DataFrame(key_rows)
    key_df.to_csv(out_dir / "key_numbers_for_manuscript.csv", index=False, encoding="utf-8-sig")

    # Text summary.
    md_lines = []
    md_lines.append("# Key numbers for manuscript")
    md_lines.append("")
    md_lines.append(f"- Number of same-condition images: **{n}**")
    md_lines.append(f"- Site basis/origin: **{site_choice.get('basis', 'NA')}**, origin = ({site_choice.get('origin', (np.nan, np.nan))[0]:.5f}, {site_choice.get('origin', (np.nan, np.nan))[1]:.5f})")
    md_lines.append(f"- Site geometry model: **{SITE_GEOMETRY_MODEL}**; hollow = single geometric hollow, mask metric = **{SITE_MASK_DISTANCE_METRIC}**")
    md_lines.append(f"- Role assignment: **B={site_choice['role']['B_key']}, A={site_choice['role']['A_key']}, H={site_choice['role']['H_key']}**")
    md_lines.append(f"- Unit-cell capture validation: fold coverage median **{_fmt_float(fold_cov_median)}**, split-half folded-map corr median **{_fmt_float(split_corr_median)}** IQR **[{_fmt_float(split_corr_p25)}, {_fmt_float(split_corr_p75)}]**")
    md_lines.append(f"- c_AH mean: **{_fmt_float(sm('c_AH_norm','mean'))}**; 95% CI **{_fmt_float(smci('c_AH_norm')[0])}–{_fmt_float(smci('c_AH_norm')[1])}**")
    md_lines.append(f"- |c_beta| mean: **{_fmt_float(sm('abs_c_beta_norm','mean'))}**; 95% CI **{_fmt_float(smci('abs_c_beta_norm')[0])}–{_fmt_float(smci('abs_c_beta_norm')[1])}**")
    md_lines.append(f"- |c_beta| / c_AH mean ratio: **{_fmt_float(cb_mean / cAH_mean if np.isfinite(cAH_mean) and cAH_mean != 0 else np.nan)}**")
    md_lines.append(f"- Motif angle θ mean: **{_fmt_float(sm('theta_beta_over_AH_deg_norm','mean'))}°**; 95% CI **{_fmt_float(smci('theta_beta_over_AH_deg_norm')[0])}–{_fmt_float(smci('theta_beta_over_AH_deg_norm')[1])}°**")
    md_lines.append(f"- Hybrid θ fraction ({THETA_AH_DOMINANT_MAX_DEG:g}–{THETA_BETA_DOMINANT_MIN_DEG:g}°): **{100*hybrid_frac:.1f}%**")
    md_lines.append(f"- θ tails: AH-like θ<{THETA_AH_DOMINANT_MAX_DEG:g}° = **{100*ah_frac:.1f}%**, B-like θ>{THETA_BETA_DOMINANT_MIN_DEG:g}° = **{100*beta_frac:.1f}%**, combined = **{100*combined_tail_frac:.1f}%**")
    qtxt = ", ".join([f"p{int(p)}={_fmt_float(theta_quantiles.get(int(p), np.nan))}°" for p in THETA_QUANTILE_PERCENTILES])
    md_lines.append(f"- θ percentiles (5/25/50/75/95): **{qtxt}**")
    if not mem_df.empty:
        md_lines.append(f"- Soft ambiguous points: **{int(mem_df['soft_ambiguous'].sum())}/{len(mem_df)}**")
        md_lines.append(f"- Soft top-label counts: **{mem_df['soft_label_top'].value_counts().to_dict()}**")
    if not gmm_df.empty:
        best = gmm_df.loc[gmm_df["BIC"].idxmin()]
        md_lines.append(f"- Optional GMM BIC best k: **{int(best['k'])}**")
    if not pca_expl.empty:
        md_lines.append(f"- Whole-map PCA PC1/PC2: **{100*pca_pc1:.1f}% / {100*pca_pc2:.1f}%**, PC1+PC2 = **{100*pca_pc12:.1f}%**, PC1–PC5 = **{100*pca_pc15:.1f}%**")
    md_lines.append("")
    md_lines.append("## Descriptive summary")
    md_lines.append("")
    md_lines.append(
        "The same-condition ensemble is described by role-based B/A/H site coordinates after drift-aware lattice folding, using a graphite-correct two-atom plus single-hollow site model. "
        f"The mean orthogonal motif coordinates are c_AH = {_fmt_float(sm('c_AH_norm','mean'))} and |c_beta| = {_fmt_float(sm('abs_c_beta_norm','mean'))}, "
        f"with a mean motif angle θ = {_fmt_float(sm('theta_beta_over_AH_deg_norm','mean'))}°. "
        "Thus the images are not pure endpoint motifs; they mainly occupy an A/B/H-resolved hybrid regime. "
        "Soft assignment and the optional GMM check should be used as uncertainty/continuum diagnostics rather than hard class labels."
    )
    (out_dir / "key_numbers_for_manuscript.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Human-readable validation report.
    rep = []
    rep.append("Analysis validation dashboard")
    rep.append("=============================")
    rep.append("")
    status_counts = validation_df["status"].value_counts().to_dict() if not validation_df.empty else {}
    rep.append(f"Status counts: {status_counts}")
    rep.append("")
    rep.append("Validation checks:")
    for _, r in validation_df.iterrows():
        rep.append(f"  [{r['status']}] {r['category']} / {r['check']}: value={_fmt_float(r['value']) if isinstance(r['value'], (int, float, np.integer, np.floating)) else r['value']} | reference={r['threshold_or_reference']}")
        rep.append(f"      {r['interpretation']}")
    rep.append("")
    rep.append("Most important manuscript numbers are in key_numbers_for_manuscript.csv and key_numbers_for_manuscript.md.")
    rep.append("Use analysis_validation_summary.csv to document sanity checks and per_image_validation_flags.csv to inspect outliers.")
    (out_dir / "analysis_validation_report.txt").write_text("\n".join(rep), encoding="utf-8")

    # Simple dashboard plot.
    try:
        if not validation_df.empty:
            fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.35 * len(validation_df))))
            y = np.arange(len(validation_df))
            status_to_num = {"PASS": 3, "INFO": 2, "CAUTION": 1, "STRONG_CAUTION": 0, "FAIL": 0}
            vals = [status_to_num.get(str(s), 1) for s in validation_df["status"]]
            ax.barh(y, vals)
            ax.set_yticks(y)
            ax.set_yticklabels([f"{r.category}: {r.check}" for r in validation_df.itertuples()], fontsize=7)
            ax.set_xlim(0, 3.2)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["strong caution", "caution", "info", "pass"], fontsize=8)
            ax.set_title("Analysis validation dashboard")
            ax.invert_yaxis()
            savefig(out_dir / "analysis_validation_dashboard.png")
    except Exception as exc:
        warnings.warn(f"Validation dashboard plot skipped: {exc}")

    return {
        "validation_df": validation_df,
        "key_df": key_df,
        "artifact_df": artifact_df,
        "unitcell_capture_summary_df": unitcell_capture_summary_df,
    }



# =============================================================================
# Raw current-floor analysis
# =============================================================================


def compute_current_floor_metrics(raw_img: np.ndarray, prefix: str = "raw_current") -> Dict[str, float]:
    """Compute robust scalar current-level/current-floor metrics from the raw current map.

    These metrics are intentionally computed from the full raw image before motif
    normalization.  The lower-percentile current floor is used as a proxy for the
    scalar junction conductance/baseline level; the motif descriptors are computed
    separately from the background-corrected, normalized unit-cell maps.
    """
    vals = np.asarray(raw_img, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"{prefix}_n_finite": 0}
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    qs = {q: float(np.nanpercentile(vals, q)) for q in percentiles}
    floor_q = float(CURRENT_FLOOR_PERCENTILE)
    floor_val = float(np.nanpercentile(vals, floor_q))
    abs_vals = np.abs(vals)
    positive_abs = abs_vals[abs_vals > 0]
    eps = float(np.nanmin(positive_abs)) * 1e-6 if positive_abs.size else SAFE_LOG10_EPS
    eps = max(eps, SAFE_LOG10_EPS)
    floor_abs = abs(floor_val)
    floor_log = math.log10(max(floor_abs, eps))
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    out = {
        f"{prefix}_n_finite": int(vals.size),
        f"{prefix}_mean": float(np.nanmean(vals)),
        f"{prefix}_median": med,
        f"{prefix}_std": float(np.nanstd(vals)),
        f"{prefix}_mad": mad,
        f"{prefix}_min": float(np.nanmin(vals)),
        f"{prefix}_max": float(np.nanmax(vals)),
        f"{prefix}_floor_percentile": floor_q,
        f"{prefix}_floor_p05": floor_val,
        f"{prefix}_floor_abs_p05": floor_abs,
        f"{prefix}_floor_log10_abs_p05": floor_log,
        f"{prefix}_abs_mean": float(np.nanmean(abs_vals)),
        f"{prefix}_abs_median": float(np.nanmedian(abs_vals)),
        f"{prefix}_dynamic_range_q95_minus_q05": qs[95] - qs[5],
        f"{prefix}_iqr_q75_minus_q25": qs[75] - qs[25],
        f"{prefix}_log10_abs_median": math.log10(max(abs(med), eps)),
    }
    for q, val in qs.items():
        out[f"{prefix}_q{q:02d}"] = val
        out[f"{prefix}_log10_abs_q{q:02d}"] = math.log10(max(abs(val), eps))
    return out


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, nboot: int = CURRENT_FLOOR_CORRELATION_BOOTSTRAP_N, seed: int = CURRENT_FLOOR_BOOTSTRAP_RANDOM_SEED) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 4:
        return np.nan, np.nan, np.nan, np.nan
    rho, pval = _spearman(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(nboot, dtype=float)
    idx = np.arange(x.size)
    for b in range(nboot):
        ii = rng.choice(idx, size=x.size, replace=True)
        rb, _ = _spearman(x[ii], y[ii])
        boots[b] = rb
    return rho, pval, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def _effect_interpretation_abs_rho(abs_rho: float) -> str:
    if not np.isfinite(abs_rho):
        return "insufficient_data"
    if abs_rho < 0.20:
        return "negligible"
    if abs_rho < CURRENT_FLOOR_EFFECT_SIZE_CAUTION_ABS_RHO:
        return "weak"
    if abs_rho < CURRENT_FLOOR_EFFECT_SIZE_STRONG_ABS_RHO:
        return "moderate_caution"
    return "strong_caution"


def compute_current_floor_correlations(coord_df: pd.DataFrame) -> pd.DataFrame:
    current_metrics = [
        "raw_current_floor_log10_abs_p05",
        "raw_current_floor_abs_p05",
        "raw_current_median",
        "raw_current_log10_abs_median",
        "raw_current_abs_mean",
        "raw_site_common_mode_raw",
        "raw_current_dynamic_range_q95_minus_q05",
        # Normalization scales are included as diagnostics because normalized R contains 1/scale factors.
        "detrended_log10_robust_z_scale",
        "folded_detrended_log10_robust_z_scale",
        "folded_norm_pre_final_log10_scale",
    ]
    motif_metrics = [
        "theta_beta_over_AH_deg_norm",
        "c_AH_norm",
        "abs_c_beta_norm",
        "R_orthogonal_norm",
        "delta_BA_norm",
        "delta_AH_norm",
        "c_AH_detrended",
        "abs_c_beta_detrended",
        "R_orthogonal_detrended",
        "delta_BA_detrended",
        "delta_AH_detrended",
        "c_AH_raw",
        "abs_c_beta_raw",
        "R_orthogonal_raw",
    ]
    primary_motifs = {"theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm"}
    rows: List[Dict[str, object]] = []
    for cmet in current_metrics:
        if cmet not in coord_df:
            continue
        for mmet in motif_metrics:
            if mmet not in coord_df:
                continue
            x = coord_df[cmet].to_numpy(dtype=float)
            y = coord_df[mmet].to_numpy(dtype=float)
            rho, pval = _spearman(x, y)
            if (not CURRENT_FLOOR_BOOTSTRAP_PRIMARY_ONLY) or (cmet == CURRENT_FLOOR_PRIMARY_METRIC and mmet in primary_motifs):
                rho2, pval2, lo, hi = bootstrap_spearman_ci(x, y)
                if np.isfinite(rho2):
                    rho, pval = rho2, pval2
            else:
                lo, hi = np.nan, np.nan
            m = np.isfinite(x) & np.isfinite(y)
            rows.append({
                "current_metric": cmet,
                "motif_metric": mmet,
                "spearman_rho": rho,
                "spearman_p": pval,
                "spearman_rho_ci95_low": lo,
                "spearman_rho_ci95_high": hi,
                "bootstrap_ci_computed": bool(np.isfinite(lo) and np.isfinite(hi)),
                "abs_rho": abs(rho) if np.isfinite(rho) else np.nan,
                "effect_size_interpretation": _effect_interpretation_abs_rho(abs(rho) if np.isfinite(rho) else np.nan),
                "n": int(m.sum()),
            })
    return pd.DataFrame(rows).sort_values(["current_metric", "abs_rho"], ascending=[True, False]) if rows else pd.DataFrame()


def compute_current_floor_bin_summary(coord_df: pd.DataFrame, floor_metric: str = CURRENT_FLOOR_PRIMARY_METRIC) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = coord_df.copy()
    if floor_metric not in df:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    valid = np.isfinite(df[floor_metric].to_numpy(dtype=float))
    labels = ["low_floor", "mid_floor", "high_floor"] if CURRENT_FLOOR_BIN_COUNT == 3 else [f"bin_{i+1}" for i in range(CURRENT_FLOOR_BIN_COUNT)]
    try:
        df.loc[valid, "current_floor_bin"] = pd.qcut(df.loc[valid, floor_metric], q=CURRENT_FLOOR_BIN_COUNT, labels=labels, duplicates="drop")
    except Exception:
        df.loc[valid, "current_floor_bin"] = pd.cut(df.loc[valid, floor_metric], bins=CURRENT_FLOOR_BIN_COUNT, labels=labels[:CURRENT_FLOOR_BIN_COUNT])
    metrics = ["theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm", "delta_BA_norm", "delta_AH_norm"]
    rows = []
    for bin_name, g in df.groupby("current_floor_bin", dropna=True, observed=False):
        row = {"current_floor_bin": str(bin_name), "n": int(len(g)), "floor_metric": floor_metric, "floor_median": float(np.nanmedian(g[floor_metric]))}
        for m in metrics:
            if m in g:
                vals = g[m].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                row[f"{m}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
                row[f"{m}_median"] = float(np.nanmedian(vals)) if vals.size else np.nan
                row[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    krows = []
    for m in metrics:
        groups = []
        for _, g in df.groupby("current_floor_bin", dropna=True, observed=False):
            vals = g[m].to_numpy(dtype=float) if m in g else np.array([])
            vals = vals[np.isfinite(vals)]
            if vals.size >= 3:
                groups.append(vals)
        if len(groups) >= 2:
            try:
                stat, pval = kruskal(*groups)
            except Exception:
                stat, pval = np.nan, np.nan
            krows.append({"metric": m, "kruskal_H": float(stat), "kruskal_p": float(pval), "n_groups": len(groups)})
    return df[["name", "short_name", floor_metric, "current_floor_bin"]], summary, pd.DataFrame(krows)


def plot_current_floor_histogram(coord_df: pd.DataFrame, out_dir: Path) -> None:
    if CURRENT_FLOOR_PRIMARY_METRIC not in coord_df:
        return
    id_cols = [c for c in ["name", "short_name"] if c in coord_df.columns]
    extra_cols = [c for c in [CURRENT_FLOOR_PRIMARY_METRIC, "raw_current_floor_p05", "raw_current_floor_abs_p05", "raw_current_median", "raw_current_abs_mean"] if c in coord_df.columns]
    raw_df = coord_df[id_cols + extra_cols].copy() if extra_cols else pd.DataFrame()
    if CURRENT_FLOOR_PRIMARY_METRIC in raw_df.columns:
        raw_df = raw_df.rename(columns={CURRENT_FLOOR_PRIMARY_METRIC: "value"})
        raw_df["metric"] = CURRENT_FLOOR_PRIMARY_METRIC
    x = coord_df[CURRENT_FLOOR_PRIMARY_METRIC].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    suggested_bins = min(30, max(6, int(np.sqrt(x.size))))
    if not raw_df.empty:
        _save_raw_distribution_table(
            out_dir,
            "raw_current_floor_log_histogram",
            raw_df,
            "Raw log10(abs(raw current floor)) values before histogram binning.",
            value_column="value",
            suggested_bins=suggested_bins,
        )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(x, bins=suggested_bins, edgecolor="black", alpha=0.75)
    ax.set_xlabel(r"log$_{10}$ |raw current floor|  (raw map, 5th percentile)")
    ax.set_ylabel("number of images")
    ax.set_title("Raw current-floor variability at fixed nominal condition")
    txt = f"range = {np.nanmax(x)-np.nanmin(x):.2f} decades\nIQR = {np.nanpercentile(x,75)-np.nanpercentile(x,25):.2f} decades"
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    savefig(out_dir / "raw_current_floor_log_histogram.png")


def plot_motif_vs_current_floor(coord_df: pd.DataFrame, corr_df: pd.DataFrame, out_dir: Path) -> None:
    floor_metric = CURRENT_FLOOR_PRIMARY_METRIC
    if floor_metric not in coord_df:
        return
    y_metrics = [
        ("theta_beta_over_AH_deg_norm", r"motif angle $\theta$ (deg)"),
        ("c_AH_norm", r"$c_{AH}$"),
        ("abs_c_beta_norm", r"$|c_\beta|$"),
        ("R_orthogonal_norm", r"motif amplitude $R=\sqrt{c_{AH}^2+c_\beta^2}$"),
    ]
    for ycol, ylabel in y_metrics:
        if ycol not in coord_df:
            continue
        fig, ax = plt.subplots(figsize=(6.3, 4.6))
        x = coord_df[floor_metric].to_numpy(dtype=float)
        y = coord_df[ycol].to_numpy(dtype=float)
        ax.scatter(x, y, s=34, edgecolors="black", linewidths=0.4, alpha=0.82)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            # Robust trend guide: ordinary least-squares line in log-floor coordinate.
            try:
                coef = np.polyfit(x[m], y[m], 1)
                xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 100)
                ax.plot(xs, coef[0] * xs + coef[1], lw=1.0, color="0.2", ls="--")
            except Exception:
                pass
        rho_txt = ""
        if not corr_df.empty:
            rrow = corr_df[(corr_df["current_metric"] == floor_metric) & (corr_df["motif_metric"] == ycol)]
            if not rrow.empty:
                r = rrow.iloc[0]
                rho_txt = f"Spearman ρ = {r['spearman_rho']:.3f}\n95% CI [{r['spearman_rho_ci95_low']:.3f}, {r['spearman_rho_ci95_high']:.3f}]\n{r['effect_size_interpretation']}"
        ax.text(0.02, 0.96, rho_txt, transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.set_xlabel(r"log$_{10}$ |raw current floor|  (raw map, 5th percentile)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs raw current floor")
        safe_name = (
            ycol.replace("theta_beta_over_AH_deg_norm", "theta")
                .replace("abs_c_beta_norm", "abs_c_beta")
                .replace("R_orthogonal_norm", "R_motif_amplitude")
        )
        # Direct Origin-friendly source data for this specific correlation plot.
        direct_cols = {
            "name": coord_df["name"].values if "name" in coord_df else np.arange(len(coord_df)),
            "short_name": coord_df["short_name"].values if "short_name" in coord_df else coord_df["name"].values if "name" in coord_df else np.arange(len(coord_df)),
            floor_metric: x,
            ycol: y,
        }
        pd.DataFrame(direct_cols).to_csv(out_dir / f"plot_data_current_floor_vs_{safe_name}.csv", index=False, encoding="utf-8-sig")
        savefig(out_dir / f"current_floor_vs_{safe_name}.png")


def plot_current_floor_bins(coord_df: pd.DataFrame, bin_assign_df: pd.DataFrame, out_dir: Path) -> None:
    if bin_assign_df.empty:
        return
    df = coord_df.merge(bin_assign_df[["name", "current_floor_bin"]], on="name", how="left", suffixes=("", "_bin"))
    df = df.dropna(subset=["current_floor_bin"])
    if df.empty:
        return
    order = ["low_floor", "mid_floor", "high_floor"]
    order = [o for o in order if o in set(df["current_floor_bin"].astype(str))]
    if not order:
        order = sorted(df["current_floor_bin"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    data = [df.loc[df["current_floor_bin"].astype(str) == o, "theta_beta_over_AH_deg_norm"].dropna().to_numpy(dtype=float) for o in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    ax.set_ylabel(r"motif angle $\theta$ (deg)")
    ax.set_xlabel("raw current-floor tertile")
    ax.set_title("Motif angle distribution by raw current-floor bin")
    savefig(out_dir / "current_floor_bins_theta_boxplot.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    markers = {"low_floor": "o", "mid_floor": "s", "high_floor": "^"}
    for o in order:
        g = df.loc[df["current_floor_bin"].astype(str) == o]
        ax.scatter(g["c_AH_norm"], g["abs_c_beta_norm"], label=o, marker=markers.get(o, "o"), s=40, edgecolors="black", linewidths=0.4, alpha=0.78)
    ax.set_xlabel(r"$c_{AH}$")
    ax.set_ylabel(r"$|c_\beta|$")
    ax.set_title("Motif-coordinate space colored by raw current-floor bin")
    ax.legend(fontsize=8)
    savefig(out_dir / "current_floor_bins_motif_coordinate_scatter.png")


def make_current_floor_outputs(out_dir: Path, coord_df: pd.DataFrame) -> Dict[str, object]:
    """Generate current-floor correlation/decoupling analysis and append manuscript key numbers."""
    if CURRENT_FLOOR_PRIMARY_METRIC not in coord_df:
        warnings.warn("Current-floor metrics were not found in coord_df; skipping motif-vs-current-floor analysis.")
        return {"current_floor_corr_df": pd.DataFrame()}

    corr_df = compute_current_floor_correlations(coord_df)
    corr_df.to_csv(out_dir / "current_floor_correlation_spearman.csv", index=False, encoding="utf-8-sig")

    bin_assign_df, bin_summary_df, bin_kruskal_df = compute_current_floor_bin_summary(coord_df)
    if not bin_assign_df.empty:
        bin_assign_df.to_csv(out_dir / "current_floor_bin_assignment.csv", index=False, encoding="utf-8-sig")
    if not bin_summary_df.empty:
        bin_summary_df.to_csv(out_dir / "current_floor_bin_motif_summary.csv", index=False, encoding="utf-8-sig")
    if not bin_kruskal_df.empty:
        bin_kruskal_df.to_csv(out_dir / "current_floor_bin_kruskal_tests.csv", index=False, encoding="utf-8-sig")

    plot_current_floor_histogram(coord_df, out_dir)
    plot_motif_vs_current_floor(coord_df, corr_df, out_dir)
    plot_current_floor_bins(coord_df, bin_assign_df, out_dir)

    floor_vals = coord_df[CURRENT_FLOOR_PRIMARY_METRIC].to_numpy(dtype=float)
    floor_vals = floor_vals[np.isfinite(floor_vals)]
    decade_range = float(np.nanmax(floor_vals) - np.nanmin(floor_vals)) if floor_vals.size else np.nan
    decade_iqr = float(np.nanpercentile(floor_vals, 75) - np.nanpercentile(floor_vals, 25)) if floor_vals.size else np.nan

    primary_motifs = ["theta_beta_over_AH_deg_norm", "c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm"]
    primary_corr = corr_df[(corr_df["current_metric"] == CURRENT_FLOOR_PRIMARY_METRIC) & (corr_df["motif_metric"].isin(primary_motifs))].copy() if not corr_df.empty else pd.DataFrame()
    max_abs_rho = float(primary_corr["abs_rho"].max()) if not primary_corr.empty else np.nan
    worst_metric = str(primary_corr.sort_values("abs_rho", ascending=False).iloc[0]["motif_metric"]) if not primary_corr.empty else "NA"
    decoupling_status = _effect_interpretation_abs_rho(max_abs_rho)

    # Append key numbers to manuscript file.
    key_path = out_dir / "key_numbers_for_manuscript.csv"
    extra_rows = pd.DataFrame([
        {"metric": "raw_current_floor_log10_abs_range_decades", "value": decade_range, "unit": "decades", "meaning": "Range of log10 absolute raw current floor across images."},
        {"metric": "raw_current_floor_log10_abs_IQR_decades", "value": decade_iqr, "unit": "decades", "meaning": "IQR of log10 absolute raw current floor across images."},
        {"metric": "max_abs_spearman_motif_vs_current_floor", "value": max_abs_rho, "unit": "|rho|", "meaning": f"Largest effect-size correlation between primary motif descriptors and raw current floor; worst metric={worst_metric}."},
        {"metric": "current_floor_decoupling_status", "value": decoupling_status, "unit": "", "meaning": "Effect-size interpretation for motif-vs-current-floor coupling."},
    ])
    if key_path.exists():
        try:
            key_df = pd.read_csv(key_path)
            pd.concat([key_df, extra_rows], ignore_index=True).to_csv(key_path, index=False, encoding="utf-8-sig")
        except Exception:
            extra_rows.to_csv(out_dir / "key_numbers_current_floor_only.csv", index=False, encoding="utf-8-sig")
    else:
        extra_rows.to_csv(out_dir / "key_numbers_current_floor_only.csv", index=False, encoding="utf-8-sig")

    md_path = out_dir / "key_numbers_for_manuscript.md"
    lines = []
    lines.append("")
    lines.append("## Raw current-floor decoupling analysis")
    lines.append("")
    lines.append(f"- Raw current floor metric: **raw map 5th percentile**, analyzed as log10 absolute floor.")
    lines.append(f"- Raw current-floor spread: **{decade_range:.3g} decades** range, **{decade_iqr:.3g} decades** IQR.")
    lines.append(f"- Largest |Spearman rho| between primary motif descriptors and current floor: **{max_abs_rho:.3g}**; worst descriptor = **{worst_metric}**; effect size = **{decoupling_status}**.")
    if not primary_corr.empty:
        for m in primary_motifs:
            rr = primary_corr.loc[primary_corr["motif_metric"] == m]
            if rr.empty:
                continue
            r = rr.iloc[0]
            lines.append(f"  - {m} vs current floor: ρ = **{r['spearman_rho']:.3g}**, 95% CI **{r['spearman_rho_ci95_low']:.3g}–{r['spearman_rho_ci95_high']:.3g}**.")
    lines.append("")
    lines.append("Summary: At fixed nominal condition, the raw current floor varies across images; the table above reports its association with each lattice-registered motif descriptor.")
    if md_path.exists():
        with md_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines))
    else:
        md_path.write_text("\n".join(lines), encoding="utf-8")

    # Standalone report.
    rep = []
    rep.append("Raw current-floor vs motif decoupling analysis")
    rep.append("===============================================")
    rep.append("")
    rep.append("Purpose: test whether the normalized A/B/H motif descriptors are controlled by the scalar raw-current floor from raw map.")
    rep.append(f"Primary current-floor metric: {CURRENT_FLOOR_PRIMARY_METRIC}")
    rep.append(f"Current-floor spread: range={decade_range:.4g} decades, IQR={decade_iqr:.4g} decades")
    rep.append(f"Max |rho| among primary motif descriptors: {max_abs_rho:.4g} ({decoupling_status}); worst={worst_metric}")
    rep.append("")
    rep.append("Primary Spearman correlations:")
    if not primary_corr.empty:
        for _, r in primary_corr.sort_values("abs_rho", ascending=False).iterrows():
            rep.append(f"  {r['motif_metric']}: rho={r['spearman_rho']:.4g}, CI95=[{r['spearman_rho_ci95_low']:.4g}, {r['spearman_rho_ci95_high']:.4g}], p={r['spearman_p']:.4g}, effect={r['effect_size_interpretation']}")
    rep.append("")
    rep.append("Interpretation guide:")
    rep.append("  |rho| < 0.20: negligible effect-size correlation; 0.20-0.35: weak; 0.35-0.50: moderate caution; >=0.50: strong caution.")
    rep.append("  With large N, p-values may be small even for weak effects; use rho and bootstrap CI as the primary evidence.")
    rep.append("  Current-floor bins test whether low/mid/high scalar current levels occupy the same motif-coordinate region.")
    rep.append("")
    rep.append("Output files:")
    rep.append("  current_floor_correlation_spearman.csv")
    rep.append("  current_floor_bin_motif_summary.csv")
    rep.append("  current_floor_bin_kruskal_tests.csv")
    rep.append("  raw_current_floor_log_histogram.png")
    rep.append("  current_floor_vs_theta.png")
    rep.append("  current_floor_vs_c_AH_norm.png")
    rep.append("  current_floor_vs_abs_c_beta.png")
    rep.append("  current_floor_vs_R_motif_amplitude.png")
    rep.append("  current_floor_bins_theta_boxplot.png")
    rep.append("  current_floor_bins_motif_coordinate_scatter.png")
    (out_dir / "current_floor_decoupling_report.txt").write_text("\n".join(rep), encoding="utf-8")

    # Also append a current-floor check to validation summary if it exists.
    val_path = out_dir / "analysis_validation_summary.csv"
    val_row = pd.DataFrame([{
        "category": "current_floor",
        "check": "max_abs_spearman_primary_motif_vs_raw_floor",
        "value": max_abs_rho,
        "threshold_or_reference": f"< {CURRENT_FLOOR_EFFECT_SIZE_CAUTION_ABS_RHO:.2f} preferred for decoupling",
        "status": "PASS" if np.isfinite(max_abs_rho) and max_abs_rho < CURRENT_FLOOR_EFFECT_SIZE_CAUTION_ABS_RHO else ("CAUTION" if np.isfinite(max_abs_rho) and max_abs_rho < CURRENT_FLOOR_EFFECT_SIZE_STRONG_ABS_RHO else "STRONG_CAUTION"),
        "interpretation": f"Tests whether motif descriptors systematically depend on scalar raw-current floor; worst={worst_metric}.",
    }])
    if val_path.exists():
        try:
            old = pd.read_csv(val_path)
            pd.concat([old, val_row], ignore_index=True).to_csv(val_path, index=False, encoding="utf-8-sig")
        except Exception:
            val_row.to_csv(out_dir / "analysis_validation_current_floor_only.csv", index=False, encoding="utf-8-sig")
    else:
        val_row.to_csv(out_dir / "analysis_validation_current_floor_only.csv", index=False, encoding="utf-8-sig")

    # Append a compact current-floor validation note to the text dashboard.
    validation_report_path = out_dir / "analysis_validation_report.txt"
    validation_note = (
        "\n\nCurrent-floor decoupling check\n"
        "------------------------------\n"
        f"Primary floor metric: {CURRENT_FLOOR_PRIMARY_METRIC}\n"
        f"Floor spread: range={decade_range:.4g} decades, IQR={decade_iqr:.4g} decades\n"
        f"Max |Spearman rho| primary motif vs raw floor: {max_abs_rho:.4g} ({decoupling_status}); worst={worst_metric}\n"
        "Interpretation: this check tests whether normalized motif descriptors are controlled by the scalar raw-current floor.\n"
    )
    try:
        if validation_report_path.exists():
            with validation_report_path.open("a", encoding="utf-8") as f:
                f.write(validation_note)
        else:
            validation_report_path.write_text(validation_note, encoding="utf-8")
    except Exception:
        pass


    return {
        "current_floor_corr_df": corr_df,
        "current_floor_bin_summary_df": bin_summary_df,
        "current_floor_bin_kruskal_df": bin_kruskal_df,
        "max_abs_rho_primary": max_abs_rho,
        "worst_primary_metric": worst_metric,
        "decoupling_status": decoupling_status,
        "floor_decade_range": decade_range,
        "floor_decade_iqr": decade_iqr,
    }


def _safe_abs_log10_series(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    out = np.full(len(vals), np.nan, dtype=float)
    arr = np.asarray(vals, dtype=float)
    m = np.isfinite(arr)
    out[m] = np.log10(np.maximum(np.abs(arr[m]), SAFE_LOG10_EPS))
    return pd.Series(out, index=values.index)


def make_normalization_floor_control_outputs(out_dir: Path, coord_df: pd.DataFrame) -> Dict[str, object]:
    """Test whether R-floor decoupling is an artifact of robust-z normalization.

    The primary R_orthogonal_norm is computed from per-image robust-z normalized
    folded maps. Because robust-z introduces an image-specific 1/MAD scale, a
    weak correlation between R_norm and raw current floor can be a normalization
    artifact if the robust scale itself tracks floor. This control reports
    normalization-scale correlations, unnormalized background-corrected map amplitude
    correlations, and raw-current-sheet amplitude correlations.
    """
    if not ENABLE_NORMALIZATION_FLOOR_CONTROLS:
        return {"normalization_floor_control_df": pd.DataFrame()}
    floor_metric = CURRENT_FLOOR_PRIMARY_METRIC
    if floor_metric not in coord_df.columns:
        warnings.warn("Skipping normalization-floor controls because raw current-floor metric is unavailable.")
        return {"normalization_floor_control_df": pd.DataFrame()}

    df = coord_df.copy()
    for col in [
        "R_orthogonal_norm", "R_orthogonal_detrended", "R_orthogonal_raw",
        "abs_c_beta_norm", "abs_c_beta_detrended", "abs_c_beta_raw",
        "c_AH_norm", "c_AH_detrended", "c_AH_raw",
        "detrended_robust_z_scale", "folded_detrended_robust_z_scale", "folded_norm_pre_final_robust_z_scale",
        "raw_current_mad",
    ]:
        if col in df.columns:
            df[f"log10_abs_{col}"] = _safe_abs_log10_series(df[col])

    floor_cols = [floor_metric, "raw_current_floor_abs_p05", "raw_current_abs_mean", "raw_current_median"]
    test_metrics = [
        ("theta_beta_over_AH_deg_norm", "scale_invariant_motif_angle", "Theta is a ratio and should be independent of per-image amplitude normalization."),
        ("R_orthogonal_norm", "normalized_motif_amplitude", "Primary R from robust-z normalized folded maps; may contain 1/MAD normalization factors."),
        ("R_orthogonal_detrended", "unnormalized_detrended_motif_amplitude", "R from background-corrected map before robust-z normalization; key control for normalization artifact."),
        ("R_orthogonal_raw", "raw_current_sheet_motif_amplitude", "R from raw current sheet folded using the same lattice; absolute-current diagnostic."),
        ("detrended_robust_z_scale", "whole_flat_image_normalization_scale", "1.4826*MAD scale used to robust-z normalize the background-corrected image."),
        ("folded_detrended_robust_z_scale", "folded_flat_map_normalization_scale", "1.4826*MAD scale of the unnormalized folded background-corrected map."),
        ("folded_norm_pre_final_robust_z_scale", "final_folded_map_normalization_scale", "Scale used by the second robust-z normalization after folding."),
        ("raw_current_mad", "raw_current_MAD", "MAD of the raw current sheet."),
        ("log10_abs_R_orthogonal_detrended", "log_unnormalized_detrended_R", "Log amplitude control for background-corrected R."),
        ("log10_abs_R_orthogonal_raw", "log_raw_current_R", "Log amplitude control for raw-current R."),
        ("log10_abs_detrended_robust_z_scale", "log_flat_normalization_scale", "Log robust-z scale of background-corrected image."),
        ("log10_abs_folded_detrended_robust_z_scale", "log_folded_flat_normalization_scale", "Log robust-z scale of folded background-corrected image."),
    ]

    rows = []
    for fcol in floor_cols:
        if fcol not in df.columns:
            continue
        for metric, role, meaning in test_metrics:
            if metric not in df.columns:
                continue
            x = pd.to_numeric(df[fcol], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            rho, pval, lo, hi = bootstrap_spearman_ci(
                x, y,
                nboot=NORMALIZATION_FLOOR_BOOTSTRAP_N,
                seed=NORMALIZATION_FLOOR_RANDOM_SEED,
            )
            m = np.isfinite(x) & np.isfinite(y)
            rows.append({
                "floor_metric": fcol,
                "tested_metric": metric,
                "metric_role": role,
                "meaning": meaning,
                "spearman_rho": rho,
                "spearman_p": pval,
                "spearman_rho_ci95_low": lo,
                "spearman_rho_ci95_high": hi,
                "abs_rho": abs(rho) if np.isfinite(rho) else np.nan,
                "effect_size_interpretation": _effect_interpretation_abs_rho(abs(rho) if np.isfinite(rho) else np.nan),
                "n": int(m.sum()),
            })
    ctrl_df = pd.DataFrame(rows)
    if not ctrl_df.empty:
        ctrl_df.to_csv(out_dir / "normalization_floor_control_correlations.csv", index=False, encoding="utf-8-sig")

    # Plot/export the most important relationships.
    plot_specs = [
        ("detrended_robust_z_scale", "normalization_scale_vs_current_floor", "Flat-image robust-z scale", True),
        ("folded_detrended_robust_z_scale", "folded_detrended_scale_vs_current_floor", "Folded background-corrected map robust scale", True),
        ("R_orthogonal_norm", "current_floor_vs_R_normalized", "Normalized motif amplitude R", False),
        ("R_orthogonal_detrended", "current_floor_vs_R_detrended_unnormalized", "Unnormalized background-corrected map motif amplitude R", False),
        ("R_orthogonal_raw", "current_floor_vs_R_raw_unnormalized", "Raw-current-sheet motif amplitude R", False),
        ("theta_beta_over_AH_deg_norm", "theta_vs_current_floor_normalization_control", "Motif angle theta (scale-invariant)", False),
    ]
    for ycol, fname, ylabel, logy in plot_specs:
        if ycol not in df.columns:
            continue
        x = pd.to_numeric(df[floor_metric], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            continue
        rho, pval = _spearman(x[m], y[m])
        fig, ax = plt.subplots(figsize=(6.2, 4.7))
        yy = np.log10(np.maximum(np.abs(y), SAFE_LOG10_EPS)) if logy else y
        ax.scatter(x[m], yy[m], s=42, edgecolors="black", linewidths=0.45, alpha=0.78)
        ax.set_xlabel(r"log$_{10}$ |raw current floor|  (raw map, 5th percentile)")
        ax.set_ylabel(("log10 |" + ylabel + "|") if logy else ylabel)
        ax.set_title(fname.replace("_", " "))
        ax.text(0.02, 0.96, f"Spearman ρ={rho:.3g}, p={pval:.2g}\nN={int(m.sum())}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        savefig(out_dir / f"{fname}.png")
        pd.DataFrame({
            "name": df["name"].values if "name" in df.columns else np.arange(len(df)),
            "short_name": df["short_name"].values if "short_name" in df.columns else df["name"].values if "name" in df.columns else np.arange(len(df)),
            floor_metric: x,
            ycol: y,
            f"plot_y_{'log10_abs_' if logy else ''}{ycol}": yy,
        }).to_csv(out_dir / f"plot_data_{fname}.csv", index=False, encoding="utf-8-sig")

    plot_cols = [
        "name", "short_name", floor_metric,
        "theta_beta_over_AH_deg_norm", "R_orthogonal_norm", "R_orthogonal_detrended", "R_orthogonal_raw",
        "detrended_robust_z_scale", "folded_detrended_robust_z_scale", "folded_norm_pre_final_robust_z_scale",
        "raw_current_mad", "raw_current_floor_abs_p05", "raw_current_abs_mean",
    ]
    plot_cols = [c for c in plot_cols if c in df.columns]
    if plot_cols:
        df[plot_cols].to_csv(out_dir / "normalization_floor_control_plot_data.csv", index=False, encoding="utf-8-sig")

    def _rho_for(metric: str, fmetric: str = floor_metric) -> float:
        if ctrl_df.empty:
            return np.nan
        q = ctrl_df[(ctrl_df["floor_metric"] == fmetric) & (ctrl_df["tested_metric"] == metric)]
        return float(q.iloc[0]["spearman_rho"]) if not q.empty else np.nan

    rho_R_norm = _rho_for("R_orthogonal_norm")
    rho_R_flat = _rho_for("R_orthogonal_detrended")
    rho_R_raw = _rho_for("R_orthogonal_raw")
    rho_scale_flat = _rho_for("detrended_robust_z_scale")
    rho_scale_fold = _rho_for("folded_detrended_robust_z_scale")
    rho_theta = _rho_for("theta_beta_over_AH_deg_norm")

    danger_norm_artifact = (
        np.isfinite(rho_R_norm) and abs(rho_R_norm) < 0.20 and
        ((np.isfinite(rho_R_flat) and abs(rho_R_flat) >= 0.35) or (np.isfinite(rho_R_raw) and abs(rho_R_raw) >= 0.35)) and
        ((np.isfinite(rho_scale_flat) and abs(rho_scale_flat) >= 0.35) or (np.isfinite(rho_scale_fold) and abs(rho_scale_fold) >= 0.35))
    )
    robust_decoupling = (
        np.isfinite(rho_R_norm) and abs(rho_R_norm) < 0.20 and
        (not np.isfinite(rho_R_flat) or abs(rho_R_flat) < 0.20) and
        (not np.isfinite(rho_R_raw) or abs(rho_R_raw) < 0.20)
    )
    if danger_norm_artifact:
        verdict = "NORMALIZATION_ARTIFACT_CAUTION"
        claim = "Do not claim absolute amplitude-floor decoupling from normalized R alone. Use 'normalized motif prominence' and report unnormalized amplitude coupling."
    elif robust_decoupling:
        verdict = "ROBUST_AMPLITUDE_FLOOR_DECOUPLING"
        claim = "Normalized and unnormalized motif amplitudes are weakly coupled to current floor; amplitude-floor decoupling is not an obvious normalization artifact."
    else:
        verdict = "MIXED_OR_MODERATE_COUPLING"
        claim = "Interpret normalized R as normalized motif prominence; qualify absolute-amplitude claims using the unnormalized controls."

    md = []
    md.append("# Normalization-floor control for R vs current floor")
    md.append("")
    md.append("Purpose: test whether the apparent decoupling between normalized motif amplitude R and raw current floor is caused by per-image robust-z normalization.")
    md.append("")
    md.append(f"- Verdict: **{verdict}**")
    md.append(f"- Recommended claim: {claim}")
    md.append("")
    md.append("## Primary correlations vs log10 |raw current floor|")
    md.append("")
    md.append(f"- theta, scale-invariant: ρ = **{rho_theta:.4g}**")
    md.append(f"- R_norm, robust-z normalized amplitude: ρ = **{rho_R_norm:.4g}**")
    md.append(f"- R_detrended, unnormalized background-corrected map amplitude: ρ = **{rho_R_flat:.4g}**")
    md.append(f"- R_raw, raw-current-sheet amplitude: ρ = **{rho_R_raw:.4g}**")
    md.append(f"- flat robust-z scale vs floor: ρ = **{rho_scale_flat:.4g}**")
    md.append(f"- folded detrended robust scale vs floor: ρ = **{rho_scale_fold:.4g}**")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("If R_norm is weakly correlated with floor but R_detrended/R_raw and the robust-z scales are strongly correlated with floor, the R_norm floor-decoupling is likely a normalization-induced cancellation. In that case, claim only that normalized motif prominence is weakly floor-coupled, not that absolute contrast amplitude is decoupled.")
    md.append("")
    md.append("If R_detrended and R_raw are also weakly correlated with floor, the decoupling is not an obvious normalization artifact and the absolute-amplitude claim is stronger.")
    md.append("")
    md.append("## Output files")
    md.append("")
    md.append("- normalization_floor_control_correlations.csv")
    md.append("- normalization_floor_control_plot_data.csv")
    md.append("- normalization_scale_vs_current_floor.png")
    md.append("- current_floor_vs_R_detrended_unnormalized.png")
    md.append("- current_floor_vs_R_raw_unnormalized.png")
    md.append("- current_floor_vs_R_normalized.png")
    md.append("- theta_vs_current_floor_normalization_control.png")
    (out_dir / "normalization_floor_control_report.md").write_text("\n".join(md), encoding="utf-8")

    key_md = out_dir / "key_numbers_for_manuscript.md"
    add = []
    add.append("")
    add.append("## Normalization-floor artifact control")
    add.append("")
    add.append(f"- R_norm vs current floor: ρ = **{rho_R_norm:.3g}**")
    add.append(f"- R_detrended vs current floor: ρ = **{rho_R_flat:.3g}**")
    add.append(f"- R_raw vs current floor: ρ = **{rho_R_raw:.3g}**")
    add.append(f"- flat robust-z scale vs current floor: ρ = **{rho_scale_flat:.3g}**")
    add.append(f"- folded background-corrected map robust scale vs current floor: ρ = **{rho_scale_fold:.3g}**")
    add.append(f"- normalization-control verdict: **{verdict}**")
    if key_md.exists():
        with key_md.open("a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(add))
    else:
        key_md.write_text("\n".join(add), encoding="utf-8")

    return {"normalization_floor_control_df": ctrl_df, "normalization_floor_verdict": verdict}

# =============================================================================
# Main analysis
# =============================================================================


def process_records(records: Sequence[ImageRecord], out_dir: Path) -> Dict[str, object]:
    if len(records) < 2:
        raise RuntimeError("At least two images are needed for ensemble motif statistics.")
    folded_raw: List[np.ndarray] = []
    folded_detrended: List[np.ndarray] = []
    folded_norm: List[np.ndarray] = []
    folded_rank: List[np.ndarray] = []
    lattice_rows: List[Dict[str, object]] = []
    current_floor_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for rec in records:
        try:
            raw_current = np.asarray(rec.data, dtype=float)
            corrected, floor_metrics = prepare_current_map(raw_current, BACKGROUND)
            G, diag, power = find_lattice_vectors_fft(corrected)
            raw_current_clean = fill_nan_nearest(raw_current)
            current_floor_row = {
                "name": rec.name, "file": rec.file, "sheet": rec.sheet,
                "background": BACKGROUND,
                "height_px": int(raw_current.shape[0]),
                "width_px": int(raw_current.shape[1]),
                **floor_metrics,
            }

            norm_img, norm_diag = robust_zscore(corrected)
            current_floor_row["detrended_robust_z_median"] = norm_diag.get("median", np.nan)
            current_floor_row["detrended_robust_z_mad"] = norm_diag.get("mad", np.nan)
            current_floor_row["detrended_robust_z_scale"] = norm_diag.get("scale", np.nan)
            current_floor_row["detrended_log10_robust_z_scale"] = math.log10(max(abs(float(norm_diag.get("scale", np.nan))), SAFE_LOG10_EPS)) if np.isfinite(norm_diag.get("scale", np.nan)) else np.nan
            # FFT phase anchors the unit-cell origin before folding.  This fixes
            # A/B/H fractional coordinates from the FFT pattern rather than from
            # intensity-based site fitting after folding.
            phase_offset, phase_diag = estimate_fft_phase_offsets(corrected, G) if FFT_PHASE_ANCHOR_FOLDING else ((0.0, 0.0), {})

            # Split-half unit-cell reliability: validate that alternating apparent
            # real-space unit cells fold to a reproducible motif before using the
            # ensemble statistics.
            split_metrics = unitcell_split_half_reliability(norm_img, G, phase_offset=phase_offset)
            # Fold corrected map for motif descriptors.
            # detrended_fold preserves the unnormalized background-corrected map contrast amplitude.
            # It is used only for normalization-floor artifact controls.
            detrended_fold, _ = fold_image_to_unit_cell(corrected, G, phase_offset=phase_offset)
            norm_fold, counts = fold_image_to_unit_cell(norm_img, G, phase_offset=phase_offset)
            rank_fold, _ = fold_image_to_unit_cell(rank_normalize(corrected), G, phase_offset=phase_offset)

            raw_fold, _ = fold_image_to_unit_cell(raw_current_clean, G, phase_offset=phase_offset)

            # Record the unnormalized folded background-corrected map scale before any final normalization.
            _detrended_fold_for_scale = np.asarray(detrended_fold, dtype=float)
            _detrended_med = float(np.nanmedian(_detrended_fold_for_scale))
            _detrended_mad = float(np.nanmedian(np.abs(_detrended_fold_for_scale - _detrended_med)))
            _detrended_scale = 1.4826 * _detrended_mad
            current_floor_row["folded_detrended_median"] = _detrended_med
            current_floor_row["folded_detrended_mad"] = _detrended_mad
            current_floor_row["folded_detrended_robust_z_scale"] = _detrended_scale
            current_floor_row["folded_detrended_log10_robust_z_scale"] = math.log10(max(abs(_detrended_scale), SAFE_LOG10_EPS)) if np.isfinite(_detrended_scale) else np.nan

            # Normalize the folded motif maps again to remove residual folding offsets/gain.
            norm_fold, norm_fold_diag = robust_zscore(norm_fold)
            rank_fold, _ = robust_zscore(rank_fold)
            current_floor_row["folded_norm_pre_final_robust_z_scale"] = norm_fold_diag.get("scale", np.nan)
            current_floor_row["folded_norm_pre_final_log10_scale"] = math.log10(max(abs(float(norm_fold_diag.get("scale", np.nan))), SAFE_LOG10_EPS)) if np.isfinite(norm_fold_diag.get("scale", np.nan)) else np.nan
            folded_raw.append(raw_fold)
            folded_detrended.append(detrended_fold)
            folded_norm.append(norm_fold)
            folded_rank.append(rank_fold)
            current_floor_rows.append(current_floor_row)
            lattice_rows.append({
                "name": rec.name,
                "file": rec.file,
                "sheet": rec.sheet,
                "background": BACKGROUND,
                "height_px": int(corrected.shape[0]),
                "width_px": int(corrected.shape[1]),
                "detrended_min": float(np.nanmin(corrected)),
                "detrended_max": float(np.nanmax(corrected)),
                "detrended_median": float(np.nanmedian(corrected)),
                "detrended_std": float(np.nanstd(corrected)),
                "raw_current_min": float(np.nanmin(raw_current_clean)),
                "raw_current_max": float(np.nanmax(raw_current_clean)),
                "raw_current_median": float(np.nanmedian(raw_current_clean)),
                "raw_current_std": float(np.nanstd(raw_current_clean)),
                "robust_scale": norm_diag["scale"],
                "detrended_robust_z_scale": norm_diag["scale"],
                "folded_detrended_robust_z_scale": _detrended_scale,
                "folded_norm_pre_final_robust_z_scale": norm_fold_diag.get("scale", np.nan),
                "fold_bin_coverage_fraction": float(np.mean(counts > 0)),
                "fold_bin_count_min": float(np.nanmin(counts)),
                "fold_bin_count_median": float(np.nanmedian(counts)),
                "fold_bin_count_max": float(np.nanmax(counts)),
                "fold_total_pixels": int(np.nansum(counts)),
                **split_metrics,
                **phase_diag,
                **diag,
            })
        except Exception as exc:
            error_rows.append({"name": rec.name, "file": rec.file, "sheet": rec.sheet, "error": str(exc)})
            warnings.warn(f"Skipping image due to processing error: {rec.name} - {exc}")

    if len(folded_norm) < 2:
        raise RuntimeError("Fewer than two images survived lattice registration/folding.")

    # Keep only successful names/records.
    ok_names = [r["name"] for r in lattice_rows]
    ok_records = [rec for rec in records if rec.name in set(ok_names)]
    names = ok_names
    folded_raw_arr = np.stack(folded_raw, axis=0)
    folded_detrended_arr = np.stack(folded_detrended, axis=0)
    folded_norm_arr = np.stack(folded_norm, axis=0)
    folded_rank_arr = np.stack(folded_rank, axis=0)

    # Align normalized maps and apply same circular shifts to raw/rank maps.
    # In FFT-phase fixed mode, skip circular alignment because it would move the
    # pre-defined FFT A/B/H coordinate frame and reintroduce site-phase bias.
    use_fft_fixed_sites = SITE_COORDINATE_MODE == "fft_phase_fixed"
    do_circular_alignment = bool(ALIGN_UNIT_CELLS and not (use_fft_fixed_sites and DISABLE_CIRCULAR_ALIGNMENT_FOR_FFT_FIXED_SITES))
    if do_circular_alignment:
        aligned_norm, shifts = align_unit_cell_maps(folded_norm_arr)
        aligned_raw = np.stack([roll2(m, sh) for m, sh in zip(folded_raw_arr, shifts)], axis=0)
        aligned_detrended = np.stack([roll2(m, sh) for m, sh in zip(folded_detrended_arr, shifts)], axis=0)
        aligned_rank = np.stack([roll2(m, sh) for m, sh in zip(folded_rank_arr, shifts)], axis=0)
    else:
        aligned_norm, aligned_raw, aligned_detrended, aligned_rank = folded_norm_arr, folded_raw_arr, folded_detrended_arr, folded_rank_arr
        shifts = [(0, 0) for _ in names]

    shift_df = pd.DataFrame({"name": names, "unitcell_shift_v_px": [s[0] for s in shifts], "unitcell_shift_u_px": [s[1] for s in shifts]})

    # Alignment quality diagnostics: compare pairwise whole-map correlations before/after circular phase alignment.
    corr_before_alignment, _ = pairwise_corr_distance(folded_norm_arr)
    corr_after_alignment, _ = pairwise_corr_distance(aligned_norm)
    alignment_quality_df = make_alignment_quality_table(corr_before_alignment, corr_after_alignment, shifts)

    # Site coordinates.  The production mode first forms one mean of the
    # phase-aligned unit cells, calibrates one registry on that mean, and applies
    # the resulting A/B/H positions unchanged to every acquisition.
    maps_for_roles = aligned_norm if ROLE_ASSIGNMENT_REFERENCE == "normalized" else aligned_raw
    lattice_tmp_df = pd.DataFrame(lattice_rows)
    fft_orientation_labels = [
        fft_orientation_label_from_angle(x)
        for x in lattice_tmp_df.get("fft_pair_angle_deg_reciprocal", pd.Series([np.nan] * len(names)))
    ]
    fft_orientation_labels = apply_manual_orientation_overrides(names, fft_orientation_labels)
    site_choice_by_label: Dict[str, Dict[str, object]] = {}
    site_choice_by_image: Dict[str, Dict[str, object]] = {}
    positions_role_by_index: List[Dict[str, Tuple[float, float]]] = []

    if SITE_COORDINATE_MODE == "ensemble_mean_fixed":
        if REFINE_A_SITE_LOCAL_AFTER_PICK:
            raise ValueError(
                "Per-image A-site refinement is incompatible with "
                "SITE_COORDINATE_MODE='ensemble_mean_fixed'. Disable refinement "
                "or select an explicit diagnostic site mode."
            )
        site_choice, origin_search_df, site_registry_ensemble_mean = (
            site_choice_from_phase_aligned_ensemble_mean(maps_for_roles)
        )
        positions_role_by_index = fixed_registry_positions_for_acquisitions(site_choice, len(names))
        positions_role = positions_role_by_index
        np.save(out_dir / "site_registry_phase_aligned_ensemble_mean.npy", site_registry_ensemble_mean)
        pd.DataFrame(site_registry_ensemble_mean).to_csv(
            out_dir / "site_registry_phase_aligned_ensemble_mean.csv",
            index=False,
            header=False,
            encoding="utf-8-sig",
        )

    elif SITE_COORDINATE_MODE == "intensity_per_image_grid_search":
        # Image-by-image intensity-based A/B/H detection.  This deliberately
        # ignores FFT 60/120 grouping and does not impose a global orientation.
        # The expensive origin/basis candidate evaluation is vectorized across
        # all images and uses CuPy GPU if available; otherwise it falls back to
        # NumPy CPU vectorization.
        try:
            site_choice_by_image, origin_search_df = grid_search_site_origin_all_maps_per_image_vectorized(
                maps_for_roles, names, fft_orientation_labels, top_n=INTENSITY_PER_IMAGE_SITE_TOP_N
            )
            for nm in names:
                positions_role_by_index.append(site_choice_by_image[nm]["positions_role"])
        except Exception as exc:
            warnings.warn(f"Vectorized/GPU per-image A/B/H search failed; falling back to slow Python loop: {exc}")
            origin_search_parts = []
            for idx, nm in enumerate(names):
                choice_i, os_i = grid_search_site_origin_single_map(maps_for_roles[idx], image_name=nm)
                choice_i["image_index"] = int(idx)
                choice_i["fft_orientation_label"] = fft_orientation_labels[idx] if idx < len(fft_orientation_labels) else "unknown"
                site_choice_by_image[nm] = choice_i
                positions_role_by_index.append(choice_i["positions_role"])
                origin_search_parts.append(os_i)
            origin_search_df = pd.concat(origin_search_parts, ignore_index=True) if origin_search_parts else pd.DataFrame()
        if not site_choice_by_image:
            raise RuntimeError("Per-image intensity A/B/H site detection failed.")
        site_choice = next(iter(site_choice_by_image.values()))
        positions_role = positions_role_by_index

    elif SITE_COORDINATE_MODE == "manual_ABH_fixed_by_orientation":
        # Use fixed, externally calibrated A/B/H coordinates for each FFT orientation
        # family.  No site origin or B/A/H assignment is fitted from image intensity.
        origin_search_parts = []
        for lab in sorted(set(fft_orientation_labels)):
            idxs = [i for i, x in enumerate(fft_orientation_labels) if x == lab]
            if not idxs:
                continue
            choice_lab = manual_fixed_site_choice_for_orientation(lab, n_images=len(idxs))
            site_choice_by_label[lab] = choice_lab
            pr = choice_lab["positions_role"]
            origin_search_parts.append(pd.DataFrame([{
                "fft_orientation_label": lab,
                "basis": choice_lab["basis"],
                "site_geometry_model": choice_lab["site_geometry_model"],
                "origin_definition": choice_lab["origin_definition"],
                "origin_u": choice_lab["origin"][0],
                "origin_v": choice_lab["origin"][1],
                "B_u": pr["B"][0], "B_v": pr["B"][1],
                "A_u": pr["A"][0], "A_v": pr["A"][1],
                "H_u": pr["H"][0], "H_v": pr["H"][1],
                "score": np.nan,
                "diagnostic_only_not_used_for_site_coordinates": False,
                "manual_fixed_coordinates": True,
            }]))
        if not site_choice_by_label:
            raise RuntimeError("No FFT orientation groups were available for manual fixed A/B/H coordinates.")
        origin_search_df = pd.concat(origin_search_parts, ignore_index=True) if origin_search_parts else pd.DataFrame()
        for lab in fft_orientation_labels:
            positions_role_by_index.append(site_choice_by_label.get(lab, next(iter(site_choice_by_label.values())))["positions_role"])
        site_choice = next(iter(site_choice_by_label.values()))
        positions_role = positions_role_by_index
        if RUN_INTENSITY_SITE_GRID_SEARCH_DIAGNOSTIC:
            try:
                _, diag_os = grid_search_site_origin(maps_for_roles)
                diag_os["diagnostic_only_not_used_for_site_coordinates"] = True
                diag_os["fft_orientation_label"] = "all_images_legacy_diagnostic"
                diag_os["warning"] = "Not used because SITE_COORDINATE_MODE=manual_ABH_fixed_by_orientation"
                origin_search_df = pd.concat([origin_search_df, diag_os], ignore_index=True)
            except Exception:
                pass
    elif SITE_COORDINATE_MODE == "fft_orientation_grouped_global":
        origin_search_parts = []
        for lab in sorted(set(fft_orientation_labels)):
            idxs = [i for i, x in enumerate(fft_orientation_labels) if x == lab]
            if not idxs:
                continue
            basis_name = basis_for_fft_orientation_label(lab)
            choice_lab, os_lab = grid_search_site_origin_for_bases(maps_for_roles[idxs], [basis_name], group_label=lab)
            choice_lab["fft_orientation_label"] = lab
            choice_lab["n_images_in_orientation_group"] = len(idxs)
            site_choice_by_label[lab] = choice_lab
            origin_search_parts.append(os_lab)
        if not site_choice_by_label:
            raise RuntimeError("No FFT orientation groups were available for site calibration.")
        origin_search_df = pd.concat(origin_search_parts, ignore_index=True) if origin_search_parts else pd.DataFrame()
        for lab in fft_orientation_labels:
            positions_role_by_index.append(site_choice_by_label.get(lab, next(iter(site_choice_by_label.values())))["positions_role"])
        site_choice = next(iter(site_choice_by_label.values()))
        positions_role = positions_role_by_index
        if RUN_INTENSITY_SITE_GRID_SEARCH_DIAGNOSTIC:
            try:
                _, diag_os = grid_search_site_origin(maps_for_roles)
                diag_os["diagnostic_only_not_used_for_site_coordinates"] = True
                diag_os["fft_orientation_label"] = "all_images_legacy_diagnostic"
                origin_search_df = pd.concat([origin_search_df, diag_os], ignore_index=True)
            except Exception:
                pass
    elif SITE_COORDINATE_MODE == "fft_phase_fixed":
        site_choice = fixed_fft_site_choice()
        positions_role_by_index = [site_choice["positions_role"] for _ in names]
        positions_role = positions_role_by_index
        if RUN_INTENSITY_SITE_GRID_SEARCH_DIAGNOSTIC:
            try:
                _, origin_search_df = grid_search_site_origin(maps_for_roles)
                origin_search_df["diagnostic_only_not_used_for_site_coordinates"] = True
            except Exception:
                origin_search_df = pd.DataFrame()
        else:
            origin_search_df = pd.DataFrame()
    else:
        site_choice, origin_search_df = grid_search_site_origin(maps_for_roles)
        positions_role_by_index = [site_choice["positions_role"] for _ in names]
        positions_role = positions_role_by_index

    # Optional local A-only refinement.  This keeps the B/H anchors from the
    # hard site picker and only adjusts A within a small local window.  It is
    # useful when B and H are visually robust but A is slightly displaced.
    A_refinement_rows: List[Dict[str, object]] = []
    if REFINE_A_SITE_LOCAL_AFTER_PICK:
        for i, nm in enumerate(names):
            pos0 = positions_role_by_index[i]
            pos_ref, diag = refine_A_site_local(maps_for_roles[i], pos0)
            positions_role_by_index[i] = pos_ref
            lab_i = fft_orientation_labels[i] if i < len(fft_orientation_labels) else "unknown"
            ch_i = site_choice_by_image.get(nm)
            if ch_i is None:
                base_choice = site_choice_by_label.get(lab_i, site_choice) if isinstance(site_choice_by_label, dict) else site_choice
                ch_i = dict(base_choice)
                site_choice_by_image[nm] = ch_i
            ch_i["positions_role_initial_before_A_refinement"] = pos0
            ch_i["positions_role"] = pos_ref
            ch_i.update(diag)
            A_refinement_rows.append({
                "name": nm,
                "short_name": short_name(nm),
                "fft_orientation_label": lab_i,
                **diag,
            })
        pd.DataFrame(A_refinement_rows).to_csv(out_dir / "per_image_A_site_local_refinement.csv", index=False, encoding="utf-8-sig")

    # Extract site intensities and motif coordinates for normalized, raw, and rank maps.
    coord_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    ci_rows: List[Dict[str, object]] = []
    for idx, name in enumerate(names):
        row: Dict[str, object] = {
            "name": name,
            "short_name": short_name(name),
            "file": ok_records[idx].file,
            "sheet": ok_records[idx].sheet,
            "unitcell_shift_v_px": shifts[idx][0],
            "unitcell_shift_u_px": shifts[idx][1],
        }
        row["fft_orientation_label"] = fft_orientation_labels[idx] if idx < len(fft_orientation_labels) else "unknown"
        pos_i = positions_role_by_index[idx]
        ch_i = site_choice_by_image.get(name)
        if ch_i is None:
            ch_i = site_choice_by_label.get(row["fft_orientation_label"], site_choice) if isinstance(site_choice_by_label, dict) else site_choice
        row["site_basis_used"] = ch_i.get("basis", "")
        row["site_coordinate_detection"] = ch_i.get("site_coordinate_mode", SITE_COORDINATE_MODE)
        row["site_registry_scope"] = ch_i.get(
            "site_registry_scope",
            "per_image" if SITE_COORDINATE_MODE == "intensity_per_image_grid_search" else "mode_specific",
        )
        row["site_registry_calibration_source"] = ch_i.get("site_registry_calibration_source", "")
        row["site_positions_fixed_across_acquisitions"] = bool(
            ch_i.get("site_positions_fixed_across_acquisitions", False)
        )
        row["site_detection_score"] = ch_i.get("score", np.nan)
        row["site_pick_score_mode"] = ch_i.get("site_pick_score_mode", INTENSITY_SITE_SCORE_MODE)
        row["site_candidate_theta_deg_at_pick"] = ch_i.get("candidate_theta_deg", np.nan)
        row["site_legacy_ABH_balance_score_at_pick"] = ch_i.get("legacy_ABH_balance_score", np.nan)
        row["site_detection_median_delta_BA"] = ch_i.get("median_delta_BA", np.nan)
        row["site_detection_median_delta_AH"] = ch_i.get("median_delta_AH", np.nan)
        row["B_u_used"] = pos_i["B"][0]
        row["B_v_used"] = pos_i["B"][1]
        row["A_u_used"] = pos_i["A"][0]
        row["A_v_used"] = pos_i["A"][1]
        row["H_u_used"] = pos_i["H"][0]
        row["H_v_used"] = pos_i["H"][1]
        # A-only local refinement diagnostics, if enabled.
        if isinstance(ch_i, dict):
            for ak in [
                "A_refinement_enabled", "A_refinement_accepted", "A_refinement_reason",
                "A_initial_u", "A_initial_v", "A_refined_u", "A_refined_v",
                "A_refinement_shift_hex120_frac", "A_refinement_score_initial",
                "A_refinement_score_best", "A_refinement_score_improvement",
                "A_refinement_I_A_initial", "A_refinement_I_A_best",
                "A_refinement_delta_BA_initial", "A_refinement_delta_AH_initial",
                "A_refinement_delta_BA_best", "A_refinement_delta_AH_best",
                "A_refinement_theta_initial_deg", "A_refinement_theta_best_deg",
                "A_refinement_theta_change_deg", "A_refinement_initial_ordered",
                "A_refinement_best_ordered", "A_refinement_top_score_gap_1_2",
                "A_refinement_n_candidates",
            ]:
                if ak in ch_i:
                    row[ak] = ch_i[ak]
        for suffix, m in [("norm", aligned_norm[idx]), ("detrended", aligned_detrended[idx]), ("raw", aligned_raw[idx]), ("rank", aligned_rank[idx])]:
            hard_sites = extract_role_site_intensities(m, pos_i)
            hard_coords = motif_coordinates_from_sites(hard_sites["I_A"], hard_sites["I_B"], hard_sites["I_H"])

            # Preserve the hard best-candidate result for auditing.
            if suffix == "norm":
                for k, v in hard_sites.items():
                    row[f"{k}_{suffix}_hard_sitepick"] = v
                for k, v in hard_coords.items():
                    row[f"{k}_{suffix}_hard_sitepick"] = v

            soft_available_norm = (
                suffix == "norm"
                and USE_SITE_PICKING_UNCERTAINTY_CORRECTION
                and USE_SOFT_SITE_COORDINATES_AS_PRIMARY
                and isinstance(ch_i, dict)
                and np.isfinite(ch_i.get("soft_I_A_norm", np.nan))
                and np.isfinite(ch_i.get("soft_I_B_norm", np.nan))
                and np.isfinite(ch_i.get("soft_I_H_norm", np.nan))
            )
            candidate_ambiguous = False
            if isinstance(ch_i, dict):
                candidate_ambiguous = bool(ch_i.get("site_soft_ambiguous", False))
                # Be conservative: mark as ambiguous also when the plausible-candidate theta spread is large
                # or the best-vs-second-best score gap is too small.
                theta_iqr_i = ch_i.get("site_soft_theta_iqr", np.nan)
                score_gap_i = ch_i.get("site_soft_score_gap_1_2", np.nan)
                if np.isfinite(theta_iqr_i) and float(theta_iqr_i) >= SITE_AMBIGUITY_THETA_IQR_CAUTION_DEG:
                    candidate_ambiguous = True
                if np.isfinite(score_gap_i) and float(score_gap_i) <= SITE_SCORE_GAP_CAUTION:
                    candidate_ambiguous = True
            if SITE_COORDINATE_MODE == "ensemble_mean_fixed":
                use_soft_norm = False
                primary_label = "phase_aligned_ensemble_mean_fixed_registry"
            elif SITE_PRIMARY_POLICY == "soft_all_candidates":
                use_soft_norm = soft_available_norm
                primary_label = "soft_candidate_weighted_all_images"
            elif SITE_PRIMARY_POLICY == "adaptive_confident_hard_ambiguous_soft":
                use_soft_norm = bool(soft_available_norm and candidate_ambiguous)
                primary_label = "soft_candidate_weighted_ambiguous" if use_soft_norm else "hard_best_candidate_confident"
            else:
                use_soft_norm = False
                primary_label = "hard_best_candidate_all_images"

            if suffix == "norm":
                row["site_primary_policy"] = SITE_PRIMARY_POLICY
                row["site_assignment_ambiguous_for_primary"] = bool(candidate_ambiguous)
                row["site_primary_source"] = primary_label
                if isinstance(ch_i, dict):
                    q05_i = ch_i.get("site_soft_theta_q05", np.nan)
                    q95_i = ch_i.get("site_soft_theta_q95", np.nan)
                    row["site_theta_uncertainty_q05_q95_width_deg"] = float(q95_i - q05_i) if np.isfinite(q05_i) and np.isfinite(q95_i) else np.nan
                    if SITE_COORDINATE_MODE == "ensemble_mean_fixed":
                        row["site_assignment_quality"] = "fixed_ensemble_registry"
                    else:
                        row["site_assignment_quality"] = "ambiguous_report_uncertainty" if candidate_ambiguous else "confident"

            if use_soft_norm:
                sites = {
                    "I_A": ch_i.get("soft_I_A_norm", np.nan),
                    "I_B": ch_i.get("soft_I_B_norm", np.nan),
                    "I_H": ch_i.get("soft_I_H_norm", np.nan),
                }
                coords = motif_coordinates_from_sites(sites["I_A"], sites["I_B"], sites["I_H"])
                row["norm_site_coordinate_primary"] = primary_label
            else:
                sites = hard_sites
                coords = hard_coords
                if suffix == "norm":
                    row["norm_site_coordinate_primary"] = primary_label

            for k, v in sites.items():
                row[f"{k}_{suffix}"] = v
            for k, v in coords.items():
                row[f"{k}_{suffix}"] = v

        # Per-image candidate ambiguity diagnostics from the site-search stage.
        for sk in [
            "site_soft_candidate_count", "site_soft_neff", "site_soft_entropy",
            "site_soft_score_gap_1_2", "site_soft_theta_q05", "site_soft_theta_q25",
            "site_soft_theta_q50", "site_soft_theta_q75", "site_soft_theta_q95",
            "site_soft_theta_iqr", "site_soft_theta_sd", "site_soft_c_AH_sd",
            "site_soft_abs_c_beta_sd", "site_soft_R_sd", "site_soft_delta_BA_sd",
            "site_soft_delta_AH_sd", "site_soft_ambiguous",
            "site_soft_theta_weighted_candidate_mean",
            "site_soft_c_AH_weighted_candidate_mean",
            "site_soft_abs_c_beta_weighted_candidate_mean",
            "site_soft_R_weighted_candidate_mean",
            "site_soft_delta_BA_weighted_candidate_mean",
            "site_soft_delta_AH_weighted_candidate_mean",
        ]:
            if isinstance(ch_i, dict) and sk in ch_i:
                row[sk] = ch_i[sk]

        ci_rows.append(bootstrap_site_coordinate_ci(name, aligned_norm[idx], pos_i, rng))
        coord_rows.append(row)

    coord_df = pd.DataFrame(coord_rows)
    if USE_SITE_PICKING_UNCERTAINTY_CORRECTION and "site_soft_neff" in coord_df.columns:
        site_uncertainty_summary = {
            "n_images": int(len(coord_df)),
            "site_primary_policy": SITE_PRIMARY_POLICY,
            "soft_primary_used_any": bool(USE_SOFT_SITE_COORDINATES_AS_PRIMARY),
            "adaptive_soft_primary_fraction": float(np.nanmean((coord_df.get("norm_site_coordinate_primary", pd.Series([], dtype=object)).astype(str).str.contains("soft_candidate_weighted")).astype(float))) if "norm_site_coordinate_primary" in coord_df.columns else np.nan,
            "median_site_soft_neff": float(np.nanmedian(coord_df["site_soft_neff"])),
            "median_site_soft_theta_iqr_deg": float(np.nanmedian(coord_df["site_soft_theta_iqr"])),
            "ambiguous_fraction": float(np.nanmean(coord_df["site_soft_ambiguous"].astype(float))) if "site_soft_ambiguous" in coord_df.columns else np.nan,
            "median_score_gap_1_2": float(np.nanmedian(coord_df["site_soft_score_gap_1_2"])),
            "median_theta_hard_minus_primary_deg": float(np.nanmedian(coord_df["theta_beta_over_AH_deg_norm_hard_sitepick"] - coord_df["theta_beta_over_AH_deg_norm"])) if "theta_beta_over_AH_deg_norm_hard_sitepick" in coord_df.columns and "theta_beta_over_AH_deg_norm" in coord_df.columns else np.nan,
            "median_theta_uncertainty_q05_q95_width_deg": float(np.nanmedian(coord_df["site_theta_uncertainty_q05_q95_width_deg"])) if "site_theta_uncertainty_q05_q95_width_deg" in coord_df.columns else np.nan,
        }
        pd.DataFrame([site_uncertainty_summary]).to_csv(out_dir / "site_picking_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
        if "A_refinement_enabled" in coord_df.columns:
            a_ref_summary = {
                "n_images": int(len(coord_df)),
                "A_refinement_enabled": bool(REFINE_A_SITE_LOCAL_AFTER_PICK),
                "A_refinement_accepted_fraction": float(np.nanmean(coord_df.get("A_refinement_accepted", pd.Series(False, index=coord_df.index)).astype(float))),
                "A_refinement_median_shift_hex120_frac": float(np.nanmedian(coord_df.get("A_refinement_shift_hex120_frac", pd.Series(np.nan, index=coord_df.index)))),
                "A_refinement_p95_shift_hex120_frac": float(np.nanpercentile(coord_df.get("A_refinement_shift_hex120_frac", pd.Series(np.nan, index=coord_df.index)), 95)),
                "A_refinement_median_theta_change_deg": float(np.nanmedian(coord_df.get("A_refinement_theta_change_deg", pd.Series(np.nan, index=coord_df.index)))),
                "A_refinement_p95_abs_theta_change_deg": float(np.nanpercentile(np.abs(coord_df.get("A_refinement_theta_change_deg", pd.Series(np.nan, index=coord_df.index))), 95)),
                "A_refinement_score_mode": "local_A_peakness_plus_ordering_no_ABH_balance_reward",
            }
            pd.DataFrame([a_ref_summary]).to_csv(out_dir / "A_site_local_refinement_summary.csv", index=False, encoding="utf-8-sig")
        if "site_assignment_quality" in coord_df.columns:
            ambig_cols = [c for c in [
                "name", "short_name", "theta_beta_over_AH_deg_norm", "theta_beta_over_AH_deg_norm_hard_sitepick",
                "site_assignment_quality", "site_primary_source", "site_soft_theta_q05", "site_soft_theta_q50",
                "site_soft_theta_q95", "site_soft_theta_iqr", "site_soft_score_gap_1_2", "site_soft_neff",
                "B_u_used", "B_v_used", "A_u_used", "A_v_used", "H_u_used", "H_v_used", "A_refinement_accepted", "A_refinement_shift_hex120_frac", "A_refinement_theta_change_deg"
            ] if c in coord_df.columns]
            coord_df.loc[coord_df["site_assignment_quality"].astype(str).str.contains("ambiguous"), ambig_cols].to_csv(
                out_dir / "site_picking_ambiguous_images.csv", index=False, encoding="utf-8-sig"
            )

        # Sensitivity table: do not change the primary theta, but report whether
        # the main motif statistics are stable when ambiguous site-picks are excluded.
        def _mean_ci95(vals):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return (np.nan, np.nan, np.nan)
            if vals.size == 1:
                return (float(vals[0]), np.nan, np.nan)
            rng_local = np.random.default_rng(BOOTSTRAP_RANDOM_SEED + 991)
            boots = [float(np.mean(rng_local.choice(vals, size=vals.size, replace=True))) for _ in range(1000)]
            return float(np.mean(vals)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

        sens_rows = []
        masks = {
            "all_images_primary_hard_best": np.ones(len(coord_df), dtype=bool),
        }
        if "site_assignment_quality" in coord_df.columns:
            q = coord_df["site_assignment_quality"].astype(str)
            masks["confident_site_assignment_only"] = ~q.str.contains("ambiguous").to_numpy()
            masks["ambiguous_site_assignment_only"] = q.str.contains("ambiguous").to_numpy()
        for label, mask in masks.items():
            sub = coord_df.loc[mask].copy()
            if sub.empty:
                continue
            tmean, tlo, thi = _mean_ci95(sub.get("theta_beta_over_AH_deg_norm", []))
            cahmean, cahlo, cahhi = _mean_ci95(sub.get("c_AH_norm", []))
            cbmean, cblo, cbhi = _mean_ci95(sub.get("abs_c_beta_norm", []))
            sens_rows.append({
                "subset": label,
                "n": int(len(sub)),
                "theta_mean_deg": tmean,
                "theta_mean_ci95_low": tlo,
                "theta_mean_ci95_high": thi,
                "theta_median_deg": float(np.nanmedian(sub["theta_beta_over_AH_deg_norm"])) if "theta_beta_over_AH_deg_norm" in sub else np.nan,
                "hybrid_fraction_20_70": float(np.nanmean((sub["theta_beta_over_AH_deg_norm"] >= 20) & (sub["theta_beta_over_AH_deg_norm"] <= 70))) if "theta_beta_over_AH_deg_norm" in sub else np.nan,
                "AH_tail_fraction_theta_lt20": float(np.nanmean(sub["theta_beta_over_AH_deg_norm"] < 20)) if "theta_beta_over_AH_deg_norm" in sub else np.nan,
                "B_tail_fraction_theta_gt70": float(np.nanmean(sub["theta_beta_over_AH_deg_norm"] > 70)) if "theta_beta_over_AH_deg_norm" in sub else np.nan,
                "c_AH_mean": cahmean,
                "c_AH_mean_ci95_low": cahlo,
                "c_AH_mean_ci95_high": cahhi,
                "abs_c_beta_mean": cbmean,
                "abs_c_beta_mean_ci95_low": cblo,
                "abs_c_beta_mean_ci95_high": cbhi,
                "median_site_theta_uncertainty_width_deg": float(np.nanmedian(sub["site_theta_uncertainty_q05_q95_width_deg"])) if "site_theta_uncertainty_q05_q95_width_deg" in sub else np.nan,
                "median_score_gap_1_2": float(np.nanmedian(sub["site_soft_score_gap_1_2"])) if "site_soft_score_gap_1_2" in sub else np.nan,
                "median_site_neff": float(np.nanmedian(sub["site_soft_neff"])) if "site_soft_neff" in sub else np.nan,
            })
        if sens_rows:
            pd.DataFrame(sens_rows).to_csv(out_dir / "site_picking_primary_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # ABH-prototype theta-pileup diagnostics.
    # This is specifically intended to detect whether the site-picking rule
    # itself is concentrating the selected theta values near the ABH prototype
    # direction (theta ~= 30 deg).  It does not change primary values; it reports
    # whether the hard-pick distribution looks artificially concentrated and
    # compares it to candidate-level alternatives when available.
    # ------------------------------------------------------------------
    theta_bias_rows: List[Dict[str, object]] = []

    def _theta_bias_summary(label: str, vals) -> None:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        row = {
            "distribution": label,
            "n": int(arr.size),
            "site_pick_score_mode": INTENSITY_SITE_SCORE_MODE,
            "theta_prototype_deg": float(THETA_PROTOTYPE_DEG),
            "theta_mean_deg": float(np.mean(arr)),
            "theta_median_deg": float(np.median(arr)),
            "theta_std_deg": float(np.std(arr)),
            "theta_iqr_deg": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "theta_p05_deg": float(np.percentile(arr, 5)),
            "theta_p95_deg": float(np.percentile(arr, 95)),
        }
        for w in THETA_PROTOTYPE_WINDOWS_DEG:
            row[f"fraction_within_{w:g}deg_of_{THETA_PROTOTYPE_DEG:g}"] = float(np.mean(np.abs(arr - THETA_PROTOTYPE_DEG) <= float(w)))
        # A simple concentration index: fraction in the narrow ±2° window divided
        # by fraction in the broader ±10° window.  Values near 0.2 are expected
        # for a locally smooth distribution; much larger values indicate a spike.
        narrow = float(np.mean(np.abs(arr - THETA_PROTOTYPE_DEG) <= 2.0))
        broad = float(np.mean(np.abs(arr - THETA_PROTOTYPE_DEG) <= 10.0))
        row["theta30_spike_index_frac_pm2_over_pm10"] = narrow / broad if broad > 0 else np.nan
        theta_bias_rows.append(row)

    if "theta_beta_over_AH_deg_norm" in coord_df.columns:
        _theta_bias_summary("primary_selected_site_theta", coord_df["theta_beta_over_AH_deg_norm"])
    if "theta_beta_over_AH_deg_norm_hard_sitepick" in coord_df.columns:
        _theta_bias_summary("hard_sitepick_theta", coord_df["theta_beta_over_AH_deg_norm_hard_sitepick"])
    if "site_soft_theta_q50" in coord_df.columns:
        _theta_bias_summary("site_candidate_median_theta_q50", coord_df["site_soft_theta_q50"])

    candidate_compare_rows: List[Dict[str, object]] = []
    if isinstance(origin_search_df, pd.DataFrame) and not origin_search_df.empty and "candidate_theta_deg" in origin_search_df.columns and "name" in origin_search_df.columns:
        # Selected top-ranked candidate under the current score.
        if "candidate_rank_in_image" in origin_search_df.columns:
            top1 = origin_search_df.loc[origin_search_df["candidate_rank_in_image"] == 1].copy()
        else:
            top1 = origin_search_df.sort_values("score", ascending=False).groupby("name", as_index=False).head(1).copy()
        if not top1.empty:
            _theta_bias_summary("candidate_top1_current_score", top1["candidate_theta_deg"])
        # Legacy ABH-balance top candidate, for diagnosing whether the old score would
        # have preferentially selected theta ~= 30 deg.
        if "legacy_ABH_balance_score" in origin_search_df.columns:
            legacy_top = origin_search_df.sort_values("legacy_ABH_balance_score", ascending=False).groupby("name", as_index=False).head(1).copy()
            if not legacy_top.empty:
                _theta_bias_summary("candidate_top1_legacy_ABH_balance_score", legacy_top["candidate_theta_deg"])
                lt = legacy_top[["name", "candidate_theta_deg", "legacy_ABH_balance_score"]].rename(columns={
                    "candidate_theta_deg": "legacy_top_theta_deg",
                    "legacy_ABH_balance_score": "legacy_top_score",
                })
                ct = top1[["name", "candidate_theta_deg", "score"]].rename(columns={
                    "candidate_theta_deg": "current_top_theta_deg",
                    "score": "current_top_score",
                })
                comp = ct.merge(lt, on="name", how="outer")
                comp["theta_current_minus_legacy_deg"] = comp["current_top_theta_deg"] - comp["legacy_top_theta_deg"]
                candidate_compare_rows.extend(comp.to_dict("records"))
        # Candidate cloud diagnostic.  If all saved plausible candidates are also
        # concentrated at 30 deg, the spike may be data-driven; if only the selected
        # candidate is concentrated, the score/ranking rule is suspect.
        _theta_bias_summary("saved_top_candidate_pool", origin_search_df["candidate_theta_deg"])

    if theta_bias_rows:
        pd.DataFrame(theta_bias_rows).to_csv(out_dir / "theta_ABH_prototype_bias_diagnostics.csv", index=False, encoding="utf-8-sig")
    if candidate_compare_rows:
        pd.DataFrame(candidate_compare_rows).to_csv(out_dir / "site_candidate_theta_current_vs_legacy.csv", index=False, encoding="utf-8-sig")

    # Raw-current common-mode site level from raw map, separated from motif shape.
    if {"I_A_raw", "I_B_raw", "I_H_raw"}.issubset(coord_df.columns):
        coord_df["raw_site_common_mode_raw"] = (coord_df["I_A_raw"] + coord_df["I_B_raw"] + coord_df["I_H_raw"]) / 3.0
        coord_df["raw_site_floor_H_raw"] = coord_df["I_H_raw"]
    current_floor_df = pd.DataFrame(current_floor_rows)
    if not current_floor_df.empty:
        current_floor_df.to_csv(out_dir / "raw_current_floor_metrics.csv", index=False, encoding="utf-8-sig")
        coord_df = coord_df.merge(current_floor_df, on=["name", "file"], how="left", suffixes=("", "_floor"))
    # Rename primary normalized columns for shorter plotting names.
    rename_map = {
        "c_AH_norm": "c_AH_norm",
        "c_beta_signed_norm": "c_beta_signed_norm",
        "abs_c_beta_norm": "abs_c_beta_norm",
        "R_orthogonal_norm": "R_orthogonal_norm",
        "theta_beta_over_AH_deg_norm": "theta_beta_over_AH_deg_norm",
    }
    # Columns already have desired names from suffix logic.

    ci_df = pd.DataFrame(ci_rows)

    # Summary / bootstrap over images.
    summary_cols = [
        "I_A_norm", "I_B_norm", "I_H_norm",
        "c_AH_norm", "c_beta_signed_norm", "abs_c_beta_norm",
        "R_orthogonal_norm", "theta_beta_over_AH_deg_norm",
        "delta_BA_norm", "delta_AH_norm",
        "c_AH_detrended", "c_beta_signed_detrended", "abs_c_beta_detrended", "R_orthogonal_detrended",
        "delta_BA_detrended", "delta_AH_detrended",
        "c_AH_raw", "c_beta_signed_raw", "abs_c_beta_raw", "R_orthogonal_raw",
    ]
    summary_df = bootstrap_group_summary(coord_df, [c for c in summary_cols if c in coord_df.columns])

    # Soft membership in orthogonal motif space.
    mem_df = soft_membership(coord_df) if COMPUTE_SOFT_MEMBERSHIP else pd.DataFrame()

    # Whole-map PCA/MDS.
    pca_expl = pd.DataFrame()
    pca_scores = pd.DataFrame()
    corr = dist = mds = None
    if RUN_WHOLE_MAP_PCA_MDS:
        pca_expl, pca_scores, Vt = pca_from_maps(aligned_norm)
        pca_scores.insert(0, "name", names)
        corr, dist = pairwise_corr_distance(aligned_norm)
        mds = classical_mds(dist, 2)

    # Optional GMM BIC/AIC.
    gmm_df = pd.DataFrame()
    if RUN_GMM_BIC_IF_SKLEARN_AVAILABLE and len(coord_df) >= 6:
        try:
            from sklearn.mixture import GaussianMixture
            X = coord_df[["c_AH_norm", "abs_c_beta_norm"]].to_numpy(dtype=float)
            X = X[np.isfinite(X).all(axis=1)]
            if X.shape[0] >= 6:
                # Standardize to avoid units dominating.
                Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
                rows = []
                for k in range(1, min(6, X.shape[0]) + 1):
                    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=BOOTSTRAP_RANDOM_SEED, n_init=20)
                    gmm.fit(Xs)
                    rows.append({"k": k, "BIC": float(gmm.bic(Xs)), "AIC": float(gmm.aic(Xs))})
                gmm_df = pd.DataFrame(rows)
        except Exception as exc:
            warnings.warn(f"Optional GMM BIC skipped: {exc}")

    # Origin-ready raw source data export.
    # This replaces the old Matplotlib-artist-based origin_plot_data export for
    # practical re-plotting in Origin.  Use <output>/origin_source_data first.
    try:
        export_origin_source_plot_data(
            out_dir,
            coord_df=coord_df,
            summary_df=summary_df,
            ci_df=ci_df,
            lattice_df=pd.DataFrame(lattice_rows),
            origin_search_df=origin_search_df,
            mem_df=mem_df,
            pca_expl=pca_expl,
            pca_scores=pca_scores,
            corr=corr,
            mds=mds,
            gmm_df=gmm_df,
            aligned_norm=aligned_norm,
            names=names,
            positions_role_by_index=positions_role_by_index,
        )
    except Exception as exc:
        warnings.warn(f"Origin source-data export failed: {exc}")

    # Save data tables.
    pd.DataFrame(lattice_rows).to_csv(out_dir / "lattice_registration_summary.csv", index=False, encoding="utf-8-sig")
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out_dir / "processing_errors.csv", index=False, encoding="utf-8-sig")
    shift_df.to_csv(out_dir / "unitcell_alignment_shifts.csv", index=False, encoding="utf-8-sig")
    alignment_quality_df.to_csv(out_dir / "unitcell_alignment_quality_summary.csv", index=False, encoding="utf-8-sig")
    origin_search_df.to_csv(out_dir / "site_origin_grid_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "name": names,
        "short_name": [short_name(x) for x in names],
        "fft_orientation_label": fft_orientation_labels,
        "site_coordinate_mode": [SITE_COORDINATE_MODE for _ in names],
        "site_basis_used": coord_df["site_basis_used"].to_list() if "site_basis_used" in coord_df.columns else ["" for _ in names],
    }).to_csv(out_dir / "per_image_fft_orientation_groups.csv", index=False, encoding="utf-8-sig")
    pd.Series(fft_orientation_labels, name="fft_orientation_label").value_counts().rename_axis("fft_orientation_label").reset_index(name="n_images").to_csv(out_dir / "fft_orientation_group_counts.csv", index=False, encoding="utf-8-sig")
    # Save the site-role assignment.  The production mode writes one row for the
    # single ensemble-mean registry; diagnostic modes may write one row per image
    # or orientation family.
    assignment_rows = []
    site_position_rows = []
    if site_choice_by_image:
        choices_to_save = site_choice_by_image
    else:
        choices_to_save = site_choice_by_label if site_choice_by_label else {"all": site_choice}
    for lab, ch in choices_to_save.items():
        pr = ch["positions_role"]
        assignment_rows.append({
            "site_choice_key": lab,
            "name": ch.get("image_name", ""),
            "fft_orientation_label": ch.get("fft_orientation_label", lab),
            "site_coordinate_mode": ch.get("site_coordinate_mode", SITE_COORDINATE_MODE),
            "site_registry_scope": ch.get("site_registry_scope", "mode_specific"),
            "site_registry_calibration_source": ch.get("site_registry_calibration_source", ""),
            "site_positions_fixed_across_acquisitions": bool(ch.get("site_positions_fixed_across_acquisitions", False)),
            "n_images_in_orientation_group": ch.get("n_images_in_orientation_group", len(names)),
            "basis": ch["basis"],
            "site_geometry_model": SITE_GEOMETRY_MODEL,
            "site_mask_distance_metric": SITE_MASK_DISTANCE_METRIC,
            "origin_definition": ch.get("origin_definition", "single_geometric_hollow"),
            "origin_u": ch["origin"][0],
            "origin_v": ch["origin"][1],
            "B_key": ch["role"]["B_key"],
            "A_key": ch["role"]["A_key"],
            "H_key": ch["role"]["H_key"],
            "B_u": pr["B"][0], "B_v": pr["B"][1],
            "A_u": pr["A"][0], "A_v": pr["A"][1],
            "H_u": pr["H"][0], "H_v": pr["H"][1],
            "geoA_u": pr["geoA"][0], "geoA_v": pr["geoA"][1],
            "geoB_u": pr["geoB"][0], "geoB_v": pr["geoB"][1],
            "geoH_u": pr["geoH"][0], "geoH_v": pr["geoH"][1],
            "score": ch.get("score", np.nan),
            "median_delta_BA": ch.get("median_delta_BA", np.nan),
            "median_delta_AH": ch.get("median_delta_AH", np.nan),
        })
        for site, role_desc in [("B", "bright atom-site"), ("A", "intermediate atom-site"), ("H", "single geometric hollow-site")]:
            site_position_rows.append({
                "fft_orientation_label": lab,
                "site": site,
                "u": pr[site][0],
                "v": pr[site][1],
                "role": role_desc,
                "basis": ch["basis"],
            })
    pd.DataFrame(assignment_rows).to_csv(out_dir / "BAH_site_role_assignment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(site_position_rows).to_csv(out_dir / "graphite_corrected_BAH_site_positions.csv", index=False, encoding="utf-8-sig")

    per_img_site_rows = []
    for i, nm in enumerate(names):
        pos_i = positions_role_by_index[i]
        lab = fft_orientation_labels[i] if i < len(fft_orientation_labels) else "unknown"
        ch_i = site_choice_by_image.get(nm)
        if ch_i is None:
            ch_i = site_choice_by_label.get(lab, site_choice) if isinstance(site_choice_by_label, dict) else site_choice
        basis_i = ch_i.get("basis", "")
        for site in ("B", "A", "H"):
            u, v = pos_i[site]
            x_display, y_display = graphite_uv_to_xy(float(u), float(v))
            per_img_site_rows.append({
                "name": nm,
                "short_name": short_name(nm),
                "fft_orientation_label": lab,
                "basis": basis_i,
                "site": site,
                "u": u,
                "v": v,
                "site_coordinate_mode": ch_i.get("site_coordinate_mode", SITE_COORDINATE_MODE),
                "site_registry_scope": ch_i.get("site_registry_scope", "mode_specific"),
                "site_registry_calibration_source": ch_i.get("site_registry_calibration_source", ""),
                "site_positions_fixed_across_acquisitions": bool(ch_i.get("site_positions_fixed_across_acquisitions", False)),
                "display_angle_deg": float(GRAPHITE_UNIT_CELL_DISPLAY_ANGLE_DEG),
                "x_graphite60": float(x_display),
                "y_graphite60": float(y_display),
            })
    pd.DataFrame(per_img_site_rows).to_csv(out_dir / "per_image_BAH_site_positions.csv", index=False, encoding="utf-8-sig")

    if site_choice_by_image:
        det_rows = []
        for nm, ch in site_choice_by_image.items():
            pr = ch["positions_role"]
            det_rows.append({
                "name": nm,
                "short_name": short_name(nm),
                "basis": ch.get("basis", ""),
                "origin_u": ch.get("origin", (np.nan, np.nan))[0],
                "origin_v": ch.get("origin", (np.nan, np.nan))[1],
                "B_u": pr["B"][0], "B_v": pr["B"][1],
                "A_u": pr["A"][0], "A_v": pr["A"][1],
                "H_u": pr["H"][0], "H_v": pr["H"][1],
                "score": ch.get("score", np.nan),
                "B_key": ch.get("role", {}).get("B_key", ""),
                "A_key": ch.get("role", {}).get("A_key", ""),
                "H_key": ch.get("role", {}).get("H_key", ""),
                "median_delta_BA": ch.get("median_delta_BA", np.nan),
                "median_delta_AH": ch.get("median_delta_AH", np.nan),
            })
        pd.DataFrame(det_rows).to_csv(out_dir / "per_image_intensity_ABH_detection_summary.csv", index=False, encoding="utf-8-sig")

    if SITE_COORDINATE_MODE == "ensemble_mean_fixed":
        fixed_note = []
        fixed_note.append("Phase-aligned ensemble-mean fixed A/B/H registry")
        fixed_note.append("==================================================")
        fixed_note.append("")
        fixed_note.append("The arithmetic mean of all phase-aligned folded unit-cell maps was formed first.")
        fixed_note.append("The graphite triplet geometry, origin, and A/B roles were calibrated once on that mean map.")
        fixed_note.append("Exactly the same fractional A/B/H positions were then used for every acquisition.")
        fixed_note.append("No per-image site-origin search or local A-site refinement contributed to the primary coordinates.")
        fixed_note.append("")
        fixed_note.append(f"Number of maps in ensemble mean: {len(names)}")
        fixed_note.append(f"Basis: {site_choice.get('basis', '')}")
        fixed_note.append(f"Origin: {site_choice.get('origin', (np.nan, np.nan))}")
        fixed_note.append(f"B/A/H positions: {site_choice.get('positions_role', {})}")
        fixed_note.append(f"Site-search backend: {_GPU_SITE_SEARCH_STATUS.get('backend', 'unknown')}")
        fixed_note.append("")
        fixed_note.append("Calibration map: site_registry_phase_aligned_ensemble_mean.npy and .csv")
        fixed_note.append("Selected registry: BAH_site_role_assignment.csv")
        (out_dir / "ensemble_mean_fixed_ABH_registry_report.txt").write_text(
            "\n".join(fixed_note), encoding="utf-8"
        )

    if SITE_COORDINATE_MODE == "intensity_per_image_grid_search":
        intensity_note = []
        intensity_note.append("Per-image intensity-defined A/B/H coordinate mode")
        intensity_note.append("================================================")
        intensity_note.append("")
        intensity_note.append("60°/120° FFT orientation grouping is ignored for site assignment.")
        intensity_note.append("For each folded unit-cell map, both graphite triplet geometries and all origin phases are searched.")
        intensity_note.append("H is the geometric hollow candidate; B and A are the brighter and intermediate atom-site candidates.")
        intensity_note.append("This is useful when marker overlays are visibly wrong, but theta can depend on the site-picking rule; inspect per_image_intensity_ABH_detection_summary.csv and site_origin_grid_search.csv.")
        intensity_note.append("")
        intensity_note.append(f"Saved top candidates per image: {INTENSITY_PER_IMAGE_SITE_TOP_N}")
        intensity_note.append(f"Site-search backend: {_GPU_SITE_SEARCH_STATUS.get('backend', 'unknown')}")
        intensity_note.append(f"Backend message: {_GPU_SITE_SEARCH_STATUS.get('message', '')}")
        (out_dir / "intensity_per_image_ABH_mode_report.txt").write_text("\n".join(intensity_note), encoding="utf-8")

    if SITE_COORDINATE_MODE == "manual_ABH_fixed_by_orientation":
        manual_note = []
        manual_note.append("Manual/fixed A/B/H coordinate mode")
        manual_note.append("==================================")
        manual_note.append("")
        manual_note.append("A/B/H sites were NOT fitted from image intensity. They were fixed by MANUAL_ABH_SITES_BY_ORIENTATION after FFT orientation grouping.")
        manual_note.append("This avoids theta distortion toward B-dominant or AH-dominant modes caused by automatic site-origin fitting.")
        manual_note.append("")
        manual_note.append("Current manual coordinates:")
        for lab, ch in choices_to_save.items():
            pr = ch["positions_role"]
            manual_note.append(f"  {lab}: basis={ch.get('basis','')}; B={pr['B']}; A={pr['A']}; H={pr['H']}")
        manual_note.append("")
        manual_note.append("If overlay markers are shifted, edit MANUAL_ABH_SITES_BY_ORIENTATION at the top of the script and rerun.")
        (out_dir / "manual_ABH_site_coordinate_mode_report.txt").write_text("\n".join(manual_note), encoding="utf-8")

    coord_df.to_csv(out_dir / "orthogonal_motif_coordinates_site_currents.csv", index=False, encoding="utf-8-sig")
    ci_df.to_csv(out_dir / "site_coordinate_bootstrap_CI_per_image.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "ensemble_motif_coordinate_summary_bootstrap.csv", index=False, encoding="utf-8-sig")
    if not mem_df.empty:
        mem_df.to_csv(out_dir / "soft_membership_probabilities.csv", index=False, encoding="utf-8-sig")
        radial_scale = float(mem_df["soft_scale_radial_95pct"].iloc[0]) if "soft_scale_radial_95pct" in mem_df else 1.0
        soft_membership_prototype_table(radial_scale).to_csv(out_dir / "soft_membership_physical_prototypes.csv", index=False, encoding="utf-8-sig")
    if not pca_expl.empty:
        pca_expl.to_csv(out_dir / "whole_map_pca_explained_variance.csv", index=False, encoding="utf-8-sig")
        pca_scores.to_csv(out_dir / "whole_map_pca_scores.csv", index=False, encoding="utf-8-sig")
    if corr is not None:
        pd.DataFrame(corr, index=names, columns=names).to_csv(out_dir / "whole_map_pairwise_correlation.csv", encoding="utf-8-sig")
        pd.DataFrame(dist, index=names, columns=names).to_csv(out_dir / "whole_map_pairwise_distance_1_minus_corr.csv", encoding="utf-8-sig")
        pd.DataFrame({"name": names, "MDS1": mds[:, 0], "MDS2": mds[:, 1]}).to_csv(out_dir / "whole_map_classical_mds_coordinates.csv", index=False, encoding="utf-8-sig")
    if not gmm_df.empty:
        gmm_df.to_csv(out_dir / "gmm_bic_aic_optional_cluster_check.csv", index=False, encoding="utf-8-sig")

    # Save arrays for reuse.
    np.savez_compressed(
        out_dir / "motif_analysis_arrays.npz",
        aligned_norm=aligned_norm,
        aligned_raw=aligned_raw,
        aligned_detrended=aligned_detrended,
        aligned_rank=aligned_rank,
        folded_norm=folded_norm_arr,
        folded_raw=folded_raw_arr,
        names=np.array(names, dtype=object),
    )

    # Plots.
    plot_unit_cell_gallery(aligned_norm, names, positions_role, out_dir)
    plot_motif_scatter_kde(coord_df, out_dir)
    plot_R_theta(coord_df, out_dir)
    plot_theta_histogram(coord_df, out_dir)
    plot_site_histograms(coord_df, out_dir)
    plot_quantile_unit_cells(aligned_norm, coord_df, out_dir)
    if corr is not None:
        plot_pairwise_heatmap(corr, names, out_dir)
    if not pca_expl.empty:
        score_df_plot = pca_scores.drop(columns=["name"], errors="ignore")
        plot_pca(pca_expl, score_df_plot, coord_df, out_dir)
    if mds is not None:
        plot_mds(mds, coord_df, out_dir)
    if not mem_df.empty:
        plot_soft_membership(mem_df, coord_df, out_dir)
    if not gmm_df.empty:
        plot_gmm_bic(gmm_df, out_dir)

    # Rank robustness plot/table.
    normalization_robustness_df = pd.DataFrame()
    if COMPUTE_RANK_ROBUSTNESS:
        normalization_robustness_df = make_normalization_robustness_table(coord_df)
        normalization_robustness_df.to_csv(out_dir / "normalization_robustness_rank_vs_z.csv", index=False, encoding="utf-8-sig")
        if {"c_AH_rank", "abs_c_beta_rank"}.issubset(coord_df.columns):
            fig, ax = plt.subplots(figsize=(6.2, 5.0))
            ax.scatter(coord_df["c_AH_norm"], coord_df["abs_c_beta_norm"], label="robust z", marker="o", s=50, edgecolors="black")
            ax.scatter(coord_df["c_AH_rank"], coord_df["abs_c_beta_rank"], label="rank", marker="x", s=50)
            ax.set_xlabel(r"$c_{AH}$")
            ax.set_ylabel(r"$|c_\beta|$")
            ax.set_title("Normalization robustness: robust z vs rank descriptors")
            ax.legend(fontsize=8)
            savefig(out_dir / "normalization_robustness_scatter_z_vs_rank.png")

    # Validation dashboard + manuscript-ready key numbers.
    validation_outputs = make_validation_outputs(
        out_dir=out_dir,
        records=ok_records,
        lattice_rows=lattice_rows,
        coord_df=coord_df,
        summary_df=summary_df,
        site_choice=site_choice,
        mem_df=mem_df,
        pca_expl=pca_expl,
        gmm_df=gmm_df,
        ci_df=ci_df,
        shift_df=shift_df,
        origin_search_df=origin_search_df,
        alignment_quality_df=alignment_quality_df,
        normalization_robustness_df=normalization_robustness_df,
    )

    # Raw current-floor vs motif-shape decoupling analysis.
    current_floor_outputs = make_current_floor_outputs(out_dir, coord_df)

    # Normalization-floor artifact controls for R.
    normalization_floor_outputs = make_normalization_floor_control_outputs(out_dir, coord_df)

    # Interpretation report.
    write_report(out_dir, coord_df, summary_df, site_choice)

    return {
        "names": names,
        "coord_df": coord_df,
        "summary_df": summary_df,
        "site_choice": site_choice,
        "pca_expl": pca_expl,
        "gmm_df": gmm_df,
        "validation_df": validation_outputs.get("validation_df", pd.DataFrame()),
        "key_df": validation_outputs.get("key_df", pd.DataFrame()),
        "current_floor_outputs": current_floor_outputs,
        "normalization_floor_outputs": normalization_floor_outputs,
        "processed": len(names),
        "skipped": len(error_rows),
    }


def write_report(
    out_dir: Path, coord_df: pd.DataFrame, summary_df: pd.DataFrame,
    site_choice: Dict[str, object],
) -> None:
    lines = [
        f"Processed maps: {len(coord_df)}",
        f"Background correction: {BACKGROUND}",
        f"Site mode: {SITE_COORDINATE_MODE}",
        f"Registry scope: {site_choice.get('site_registry_scope', '')}",
        f"A/B/H coordinates: {site_choice.get('positions_role', {})}",
        "",
    ]
    metrics = ["c_AH_norm", "abs_c_beta_norm", "R_orthogonal_norm", "theta_beta_over_AH_deg_norm"]
    lines.append(summary_df.loc[summary_df["metric"].isin(metrics)].to_string(index=False))
    (out_dir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    input_path = Path(path)
    out_dir = ensure_out_dir(input_path)
    files = list_input_files(input_path)
    if not files:
        raise RuntimeError(f"No .xlsx, .xlsm, or .csv current maps found in {input_path}")
    records, extraction_log = extract_all_images(files)
    extraction_log.to_csv(out_dir / "input_extraction_log.csv", index=False, encoding="utf-8-sig")
    if len(records) != len(files):
        raise RuntimeError("Some raw maps could not be read. See input_extraction_log.csv.")
    if len(records) < 2:
        raise RuntimeError("At least two raw maps are required for ensemble analysis.")
    settings = {
        "input_type": "raw_current_map", "excel_sheet": RAW_SHEET,
        "site_mask_metric": "du*du + dv*dv - du*dv",
        "background": BACKGROUND, "numeric_current_multiplier_to_A": NUMERIC_INPUT_MULTIPLIER,
        "scan_size_nm": [SCAN_SIZE_X_NM, SCAN_SIZE_Y_NM],
        "current_floor_percentile": CURRENT_FLOOR_PERCENTILE,
        "current_floor_source": "finite raw pixels before background correction or filling",
        "unit_cell_grid": UNIT_CELL_GRID, "site_coordinate_mode": SITE_COORDINATE_MODE,
        "a_site_refinement": REFINE_A_SITE_LOCAL_AFTER_PICK,
        "reciprocal_metric": "h*h + h*k + k*k",
    }
    (out_dir / "analysis_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    result = process_records(records, out_dir)

    print(f"Processed {result['processed']} maps; skipped {result['skipped']}. Outputs: {out_dir}")


def cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-dir", required=True, help="Directory of raw current maps (.xlsx, .xlsm, .csv)")
    parser.add_argument("--sheet", default="0", help="Excel sheet name or zero-based index (default: 0)")
    parser.add_argument("--background", choices=["plane", "none"], default="plane")
    parser.add_argument("--current-unit", choices=["A", "nA", "pA", "fA"], default="A", help="Unit of numeric input cells")
    parser.add_argument("--scan-size-nm", nargs=2, type=float, default=[2.5, 2.5], metavar=("X", "Y"))
    parser.add_argument("--output-dir", help="Output folder; defaults to INPUT_DIR/analysis_results")
    parser.add_argument("--non-recursive", action="store_true", help="Do not search input subfolders")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy for the site-origin search when available")
    parser.add_argument(
        "--site-mode",
        choices=[
            "ensemble_mean_fixed",
            "intensity_per_image_grid_search",
            "intensity_global_grid_search",
            "manual_ABH_fixed_by_orientation",
            "fft_orientation_grouped_global",
            "fft_phase_fixed",
        ],
        default=SITE_COORDINATE_MODE,
        help=(
            "A/B/H registration mode. Default: calibrate once on the "
            "phase-aligned ensemble mean and fix the positions for all acquisitions"
        ),
    )
    refinement_group = parser.add_mutually_exclusive_group()
    refinement_group.add_argument(
        "--a-refinement",
        dest="a_refinement",
        action="store_true",
        help="Opt in to per-image local A-site refinement (incompatible with ensemble_mean_fixed)",
    )
    refinement_group.add_argument(
        "--no-a-refinement",
        dest="a_refinement",
        action="store_false",
        help="Disable local A-site refinement (retained for command-line compatibility)",
    )
    parser.set_defaults(a_refinement=REFINE_A_SITE_LOCAL_AFTER_PICK)
    args = parser.parse_args(argv)
    if not all(np.isfinite(v) and v > 0 for v in args.scan_size_nm):
        parser.error("--scan-size-nm values must be finite and positive")
    if args.sheet.lstrip("-").isdigit() and int(args.sheet) < 0:
        parser.error("--sheet index must be nonnegative")
    globals()["RAW_SHEET"] = int(args.sheet) if args.sheet.isdigit() else args.sheet
    globals()["BACKGROUND"] = args.background
    globals()["NUMERIC_INPUT_MULTIPLIER"] = {"A": 1.0, "nA": 1e-9, "pA": 1e-12, "fA": 1e-15}[args.current_unit]
    globals()["SCAN_SIZE_X_NM"], globals()["SCAN_SIZE_Y_NM"] = args.scan_size_nm
    if args.site_mode == "ensemble_mean_fixed" and args.a_refinement:
        parser.error(
            "--a-refinement would make the registry acquisition-specific and "
            "cannot be combined with --site-mode ensemble_mean_fixed"
        )

    globals()["path"] = str(Path(args.input_dir).expanduser().resolve())
    globals()["INPUT_ROOT_FOR_PROVENANCE"] = Path(args.input_dir).expanduser().resolve()
    globals()["OUTPUT_FOLDER_NAME"] = (
        str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else "analysis_results"
    )
    globals()["SEARCH_RECURSIVELY"] = not args.non_recursive
    globals()["USE_GPU_ACCELERATION_FOR_SITE_SEARCH"] = bool(args.gpu)
    globals()["SITE_COORDINATE_MODE"] = args.site_mode
    globals()["REFINE_A_SITE_LOCAL_AFTER_PICK"] = bool(args.a_refinement)
    main()


if __name__ == "__main__":
    cli()
