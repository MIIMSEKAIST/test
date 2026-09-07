"""Analyze a directory of raw C-AFM current maps."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def run(command: list[object]) -> None:
    subprocess.run([str(item) for item in command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="Directory containing raw maps (.xlsx, .xlsm, .csv)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--sheet", default="0", help="Excel sheet name or zero-based index")
    parser.add_argument("--background", choices=["plane", "none"], default="plane")
    parser.add_argument("--current-unit", choices=["A", "nA", "pA", "fA"], default="A")
    parser.add_argument("--scan-size-nm", nargs=2, type=float, default=[2.5, 2.5], metavar=("X", "Y"))
    parser.add_argument("--skip-primary", action="store_true", help="Reuse an existing primary-analysis directory")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-supplementary", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy during the A/B/H origin search")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        parser.error(f"Raw-data directory does not exist: {raw_dir}")
    if results_dir.is_relative_to(raw_dir):
        parser.error("Choose a results directory outside the raw-data directory.")
    if not all(math.isfinite(v) and v > 0 for v in args.scan_size_nm):
        parser.error("--scan-size-nm values must be finite and positive")
    if args.sheet.lstrip("-").isdigit() and int(args.sheet) < 0:
        parser.error("--sheet index must be nonnegative")
    if (
        results_dir.exists()
        and any(results_dir.iterdir())
        and not args.skip_primary
    ):
        parser.error(
            "Results directory is not empty. Choose a new directory, or use "
            "--skip-primary only when intentionally continuing a compatible run."
        )
    analysis_dir = results_dir / "analysis"
    figure_dir = results_dir / "figure_source_data"
    resolved_dir = analysis_dir / "snr_resolved_readout"
    if args.skip_primary:
        settings_path = analysis_dir / "analysis_settings.json"
        if not settings_path.is_file():
            parser.error("--skip-primary requires results from the raw-map pipeline.")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        expected = {
            "input_type": "raw_current_map",
            "excel_sheet": int(args.sheet) if args.sheet.isdigit() else args.sheet,
            "background": args.background,
            "numeric_current_multiplier_to_A": {"A": 1.0, "nA": 1e-9, "pA": 1e-12, "fA": 1e-15}[args.current_unit],
            "scan_size_nm": args.scan_size_nm,
            "site_coordinate_mode": "ensemble_mean_fixed",
            "site_mask_metric": "du*du + dv*dv - du*dv",
            "a_site_refinement": False,
        }
        if any(settings.get(key) != value for key, value in expected.items()):
            parser.error("Existing analysis settings differ from this command. Use the original options or a new results directory.")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_primary:
        run([
            sys.executable,
            SCRIPTS / "00_make_input_manifest.py",
            raw_dir,
            results_dir / "input_manifest.csv",
            "--environment-json",
            results_dir / "run_environment.json",
        ])
        command: list[object] = [
            sys.executable,
            SCRIPTS / "01_extract_motif_coordinates.py",
            "--input-dir",
            raw_dir,
            "--output-dir",
            analysis_dir,
            "--site-mode",
            "ensemble_mean_fixed",
            "--no-a-refinement",
            "--sheet", args.sheet,
            "--background", args.background,
            "--current-unit", args.current_unit,
            "--scan-size-nm", *args.scan_size_nm,
        ]
        if args.gpu:
            command.append("--gpu")
        run(command)

    if not args.skip_validation:
        run(
            [
                sys.executable,
                SCRIPTS / "02_validate_quantile_reconstruction.py",
                "--analysis-dir",
                analysis_dir,
            ]
        )

    run(
        [
            sys.executable,
            SCRIPTS / "03_compute_readout_attenuation.py",
            "--analysis-dir",
            analysis_dir,
        ]
    )
    run(
        [
            sys.executable,
            SCRIPTS / "05_compute_snr_resolved_statistics.py",
            "--analysis-dir",
            analysis_dir,
        ]
    )
    if not args.skip_supplementary:
        run(
            [
                sys.executable,
                SCRIPTS / "04_compute_supplementary_source_data.py",
                "--analysis-dir",
                analysis_dir,
            ]
        )
    run(
        [
            sys.executable,
            SCRIPTS / "04_export_figure_data.py",
            "--analysis-dir",
            analysis_dir,
            "--readout-csv",
            analysis_dir / "readout_attenuation_120deg" / "readout_parameters.csv",
            "--snr-selection-csv",
            resolved_dir / "snr_resolved_readout_per_image.csv",
            "--out-dir",
            figure_dir,
        ]
    )
if __name__ == "__main__":
    main()
