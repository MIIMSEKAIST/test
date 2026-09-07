"""Select sqrt(3)-resolved maps and compute the selected correlations.

The attenuation descriptors are estimated for every folded unit-cell map by
``03_compute_readout_attenuation.py``.  This script independently evaluates the
sqrt(3) peak signal-to-noise ratio in the same 120-degree reciprocal-index
convention, retains maps with SNR >= 5, and computes Spearman correlations
against the current floor on that resolved subset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_MAP_KEY = "aligned_norm"
DEFAULT_OUTPUT_FOLDER = "snr_resolved_readout"
DEFAULT_SNR_THRESHOLD = 5.0

# Reciprocal-index convention for the displayed 120-degree direct unit cell.
# The corresponding quadratic form is q^2 = h^2 + h*k + k^2.
SQRT3_SHELL_HK = (
    (-2, 1),
    (-1, -1),
    (-1, 2),
    (1, -2),
    (1, 1),
    (2, -1),
)
NOISE_Q2_MIN = 2
NOISE_Q2_MAX = 8
NOISE_EXCLUDED_Q2 = frozenset({0, 1, 3, 4})
NOISE_MAX_ABS_INDEX = 5


def decode_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def q2_120(h: int, k: int) -> int:
    """Squared reciprocal radius for the 120-degree direct-cell convention."""

    return int(h * h + h * k + k * k)


def local_noise_indices() -> tuple[tuple[int, int], ...]:
    """Return the deterministic off-shell modes used as the noise reference."""

    modes: list[tuple[int, int]] = []
    for h in range(-NOISE_MAX_ABS_INDEX, NOISE_MAX_ABS_INDEX + 1):
        for k in range(-NOISE_MAX_ABS_INDEX, NOISE_MAX_ABS_INDEX + 1):
            q2 = q2_120(h, k)
            if not NOISE_Q2_MIN <= q2 <= NOISE_Q2_MAX:
                continue
            if q2 in NOISE_EXCLUDED_Q2:
                continue
            modes.append((h, k))
    return tuple(sorted(set(modes), key=lambda hk: (q2_120(*hk), *hk)))


def direct_dft_coefficients(
    unit_map: np.ndarray,
    modes: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Evaluate normalized DFT coefficients at integer fractional-cell modes.

    Folded-map rows are the fractional ``v`` coordinate and columns are ``u``.
    Coefficients are evaluated directly rather than locating peaks in an
    interpolated FFT image.
    """

    image = np.asarray(unit_map, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Each unit-cell map must be two-dimensional; got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Unit-cell maps must contain only finite values")

    image = image - np.nanmean(image)
    n_v, n_u = image.shape
    u = np.arange(n_u, dtype=float) / n_u
    v = np.arange(n_v, dtype=float) / n_v
    uu, vv = np.meshgrid(u, v)

    coefficients = []
    for h, k in modes:
        phase = np.exp(-2j * np.pi * (h * uu + k * vv))
        coefficients.append(np.sum(image * phase) / image.size)
    return np.asarray(coefficients, dtype=np.complex128)


def compute_snr_table(
    maps: np.ndarray,
    names: Sequence[str],
    threshold: float = DEFAULT_SNR_THRESHOLD,
) -> pd.DataFrame:
    """Compute per-image sqrt(3) amplitude SNR and the inclusive selection."""

    maps = np.asarray(maps, dtype=float)
    if maps.ndim != 3:
        raise ValueError(f"Expected maps with shape (n, rows, columns); got {maps.shape}")
    if len(names) != maps.shape[0]:
        raise ValueError(f"Found {maps.shape[0]} maps but {len(names)} names")
    if not math.isfinite(threshold):
        raise ValueError("SNR threshold must be finite")

    noise_modes = local_noise_indices()
    rows: list[dict[str, object]] = []
    for index, (name, unit_map) in enumerate(zip(names, maps)):
        sqrt3_coefficients = direct_dft_coefficients(unit_map, SQRT3_SHELL_HK)
        noise_coefficients = direct_dft_coefficients(unit_map, noise_modes)
        sqrt3_amplitude = float(np.nanmean(np.abs(sqrt3_coefficients)))
        noise_amplitude = float(np.nanmedian(np.abs(noise_coefficients)))
        snr = sqrt3_amplitude / noise_amplitude if noise_amplitude > 0 else np.nan
        rows.append(
            {
                "array_index": index,
                "name": decode_name(name),
                "sqrt3_amp_mean": sqrt3_amplitude,
                "local_noise_amp_median": noise_amplitude,
                "sqrt3_amp_snr": snr,
                "sqrt3_resolved": bool(np.isfinite(snr) and snr >= threshold),
            }
        )
    return pd.DataFrame(rows)


def load_maps(npz_path: Path, map_key: str) -> tuple[np.ndarray, list[str]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Folded-map archive not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        if map_key not in archive:
            raise KeyError(f"{map_key!r} is not present in {npz_path}")
        if "names" not in archive:
            raise KeyError(f"'names' is not present in {npz_path}")
        maps = np.asarray(archive[map_key], dtype=float)
        names = [decode_name(value) for value in archive["names"].tolist()]
    return maps, names


def require_unique_names(table: pd.DataFrame, label: str) -> None:
    if "name" not in table.columns:
        raise KeyError(f"{label} must contain a 'name' column")
    if table["name"].isna().any() or table["name"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contains missing or empty names")
    duplicate = table.loc[table["name"].astype(str).duplicated(), "name"]
    if not duplicate.empty:
        examples = ", ".join(duplicate.astype(str).head(3))
        raise ValueError(f"{label} contains duplicate names, including: {examples}")


def parse_integer_index(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integer values")
    return numeric.astype(int)


def resolve_column(table: pd.DataFrame, requested: str, alternatives: Sequence[str]) -> str:
    if requested:
        if requested not in table.columns:
            raise KeyError(f"Requested column {requested!r} is not present")
        return requested
    for column in alternatives:
        if column in table.columns:
            return column
    raise KeyError(f"None of the required columns are present: {list(alternatives)}")


def merge_readout_and_snr(
    readout: pd.DataFrame,
    snr: pd.DataFrame,
    *,
    threshold: float = DEFAULT_SNR_THRESHOLD,
    w_column: str = "w_rel_primary",
    eta_column: str = "eta_rel_logamp",
    floor_column: str = "raw_current_floor_log10_abs_p05",
) -> pd.DataFrame:
    """Join by acquisition name and construct an auditable analysis mask."""

    readout = readout.copy()
    snr = snr.copy()
    require_unique_names(readout, "readout table")
    require_unique_names(snr, "SNR table")
    readout["name"] = readout["name"].map(decode_name)
    snr["name"] = snr["name"].map(decode_name)
    # Decode first, then recheck in case different byte/string representations
    # collapse to the same identifier.
    require_unique_names(readout, "readout table")
    require_unique_names(snr, "SNR table")
    for table, label in ((readout, "readout table"), (snr, "SNR table")):
        if "array_index" not in table.columns:
            raise KeyError(f"{label} must contain an 'array_index' column")
        table["array_index"] = parse_integer_index(
            table["array_index"], f"{label} array_index"
        )
        if table["array_index"].duplicated().any():
            raise ValueError(f"{label} contains duplicate array_index values")

    missing_readout = sorted(set(snr["name"]) - set(readout["name"]))
    missing_snr = sorted(set(readout["name"]) - set(snr["name"]))
    if missing_readout or missing_snr:
        raise ValueError(
            "The folded-map and readout name sets differ: "
            f"missing readout={len(missing_readout)}, missing SNR={len(missing_snr)}"
        )

    w_column = resolve_column(
        readout,
        w_column,
        ("w_rel_primary", "template_relative_readout_blur_w_rel", "w"),
    )
    eta_column = resolve_column(
        readout,
        eta_column,
        ("eta_rel_logamp", "template_relative_anisotropy_eta_rel", "eta"),
    )
    floor_column = resolve_column(
        readout,
        floor_column,
        ("raw_current_floor_log10_abs_p05", "log10_I_floor"),
    )

    selected_columns = ["name", "array_index", w_column, eta_column, floor_column]
    selected = readout[selected_columns].rename(
        columns={
            "array_index": "readout_array_index",
            w_column: "w",
            eta_column: "eta",
            floor_column: "log10_I_floor",
        }
    )
    merged = snr.merge(selected, on="name", how="left", validate="one_to_one")
    left = pd.to_numeric(merged["array_index"], errors="coerce")
    right = pd.to_numeric(merged["readout_array_index"], errors="coerce")
    if left.isna().any() or right.isna().any() or not np.array_equal(left, right):
        raise ValueError("SNR and readout array_index values do not match")
    merged = merged.drop(columns="readout_array_index")
    for column in ("sqrt3_amp_snr", "w", "eta", "log10_I_floor"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    snr_values = merged["sqrt3_amp_snr"].to_numpy(dtype=float)
    resolved = pd.Series(
        np.isfinite(snr_values) & (snr_values >= threshold),
        index=merged.index,
    )
    if "sqrt3_resolved" in merged:
        saved_raw = merged["sqrt3_resolved"]
        if pd.api.types.is_bool_dtype(saved_raw):
            saved = saved_raw.astype(bool)
        else:
            saved = saved_raw.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "false": False, "0": False}
            )
            if saved.isna().any():
                raise ValueError("Saved sqrt3_resolved flags are not valid Booleans")
        if not np.array_equal(saved.to_numpy(), resolved.to_numpy()):
            raise ValueError("Saved sqrt3_resolved flags disagree with the numeric SNR threshold")
    merged["sqrt3_resolved"] = resolved
    merged["analysis_keep"] = resolved.to_numpy(dtype=bool)

    reasons = np.full(len(merged), "", dtype=object)
    reasons[~np.isfinite(merged["sqrt3_amp_snr"].to_numpy(float))] = "snr_not_finite"
    finite_snr = np.isfinite(merged["sqrt3_amp_snr"].to_numpy(float))
    reasons[finite_snr & ~resolved.to_numpy(dtype=bool)] = "sqrt3_snr_below_threshold"
    merged["exclusion_reason"] = reasons
    return merged


def spearman_table(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parameter in ("w", "eta"):
        x = selected["log10_I_floor"].to_numpy(dtype=float)
        y = selected[parameter].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 4:
            raise ValueError(f"At least four finite pairs are required for {parameter}")
        if np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
            raise ValueError(f"Spearman correlation for {parameter} requires non-constant inputs")
        result = spearmanr(x[finite], y[finite])
        rows.append(
            {
                "x": "log10_I_floor",
                "y": parameter,
                "n": int(finite.sum()),
                "method": "spearman",
                "spearman_rho": float(result.statistic),
                "spearman_p": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--readout-csv", type=Path, default=None)
    parser.add_argument("--maps-npz", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--map-key", default=DEFAULT_MAP_KEY)
    parser.add_argument("--snr-threshold", type=float, default=DEFAULT_SNR_THRESHOLD)
    parser.add_argument("--w-column", default="w_rel_primary")
    parser.add_argument("--eta-column", default="eta_rel_logamp")
    parser.add_argument(
        "--floor-column",
        default="raw_current_floor_log10_abs_p05",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.expanduser().resolve()
    maps_npz = args.maps_npz or analysis_dir / "motif_analysis_arrays.npz"
    readout_csv = args.readout_csv or (
        analysis_dir / "readout_attenuation_120deg" / "readout_parameters.csv"
    )
    out_dir = args.out_dir or analysis_dir / DEFAULT_OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, names = load_maps(maps_npz, args.map_key)
    snr = compute_snr_table(maps, names, threshold=args.snr_threshold)
    # Keep the selected threshold explicit even if a non-default value is used.
    snr_values = snr["sqrt3_amp_snr"].to_numpy(dtype=float)
    snr["sqrt3_resolved"] = np.isfinite(snr_values) & (
        snr_values >= args.snr_threshold
    )
    readout = pd.read_csv(readout_csv)
    merged = merge_readout_and_snr(
        readout,
        snr,
        threshold=args.snr_threshold,
        w_column=args.w_column,
        eta_column=args.eta_column,
        floor_column=args.floor_column,
    )
    selected = merged.loc[merged["analysis_keep"]].copy()
    correlations = spearman_table(selected)

    merged.to_csv(
        out_dir / "snr_resolved_readout_per_image.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(
        out_dir / "snr_resolved_readout_vs_floor_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = pd.DataFrame(
        [
            {
                "n_input": len(merged),
                "snr_threshold": args.snr_threshold,
                "threshold_is_inclusive": True,
                "n_sqrt3_resolved": int(merged["sqrt3_resolved"].sum()),
                "n_analysis_keep": int(merged["analysis_keep"].sum()),
                "n_excluded": int((~merged["analysis_keep"]).sum()),
                "n_w_floor_pairs": int(correlations.loc[correlations["y"] == "w", "n"].iloc[0]),
                "n_eta_floor_pairs": int(correlations.loc[correlations["y"] == "eta", "n"].iloc[0]),
            }
        ]
    )
    summary.to_csv(
        out_dir / "snr_resolved_filter_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    settings = {
        "maps_archive": maps_npz.name,
        "readout_table": readout_csv.name,
        "map_key": args.map_key,
        "snr_threshold": args.snr_threshold,
        "threshold_rule": "sqrt3_amp_snr >= snr_threshold",
        "sqrt3_amplitude": "mean(abs(C_hk)) over SQRT3_SHELL_HK",
        "noise_amplitude": "median(abs(C_hk)) over local off-shell modes",
        "sqrt3_shell_hk": [list(mode) for mode in SQRT3_SHELL_HK],
        "noise_shell_hk": [list(mode) for mode in local_noise_indices()],
        "reciprocal_metric": "q2 = h^2 + h*k + k^2",
        "correlation": "Spearman, parameter-specific finite pairs in the SNR-resolved subset",
    }
    (out_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input maps: {len(merged)}")
    print(f"sqrt(3) SNR >= {args.snr_threshold:g}: {int(merged['sqrt3_resolved'].sum())}")
    print(f"Rows used for correlations: {len(selected)}")
    print(correlations.to_string(index=False))
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
