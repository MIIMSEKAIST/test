# Graphite C-AFM analysis

Analysis code for *Junction-state-dependent redistribution of atomic conductance in graphite*.

## Input

One raw current matrix per file, with at least 16 × 16 pixels and no headers
or coordinate rows/columns:

- `.xlsx` / `.xlsm`: the first sheet is read; other sheets are ignored.
  Use `--sheet Current` to select a named sheet.
- `.csv`: a numeric current matrix without headers or coordinate columns.

Numeric values are in amperes by default. Use `--current-unit pA` for numeric
picoampere values. Explicit units in Excel cells, such as `3 pA` or `3pA`,
override this option. Unrecognized units and mixed text are rejected.
The default field of view is 2.5 × 2.5 nm.

Use a directory containing only maps from the same acquisition conditions.
Files are ordered by relative filename; retain acquisition identifiers in the names.
At least two maps are required. Data and calculated results are not included.

## Run

With Python 3.12, from the repository directory:

```bash
python -m pip install -r requirements.txt
python run_all.py data/raw --results-dir results --scan-size-nm 2.5 2.5
```

The current floor is calculated from the finite raw pixels. A fitted plane
is then subtracted from a separate copy for lattice detection and motif analysis.
Use `--background none` to omit this subtraction. A separate flattened map is
not required. This preprocessing is not assumed equivalent to an external
flattening procedure.

Use a new results directory outside the input directory.
`--skip-validation --skip-supplementary` omits the reconstruction and supplementary checks.

## Results

The main outputs under `results/analysis/` are:

- `orthogonal_motif_coordinates_site_currents.csv`: A/B/H signals, θ, R, and current floor.
- `motif_analysis_arrays.npz`: folded maps before and after phase alignment.
- `readout_attenuation_120deg/readout_parameters.csv`: w and η.
- `snr_resolved_readout/`: SNR selection and selected-population correlations.
- `analysis_settings.json`: input units, background correction, scan size, and registry settings.

Figure tables are written to `results/figure_source_data/`.
The input checksums and software versions are recorded in `results/`.
Definitions are in [METHODS.md](METHODS.md); a notebook entry point is in
[notebooks/analyze_maps.ipynb](notebooks/analyze_maps.ipynb).

## Tests

```bash
python -m unittest discover -s tests -v
```
