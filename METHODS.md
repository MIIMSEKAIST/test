# Methods

## Raw-map processing

`01_extract_motif_coordinates.py` reads one raw current matrix per acquisition.
Numeric inputs are converted to amperes. The current floor is
\(I_\mathrm{floor}=Q_{0.05}(I_\mathrm{raw})\), using finite pixels before filling
or background correction.

The default background is a least-squares plane \(ax+by+c\), fitted to finite
raw pixels and subtracted from a separate copy. Missing pixels are then filled
by nearest-neighbor interpolation. There is no line-by-line flattening.
With `--background none`, the motif branch uses the uncorrected map.

For lattice detection, the map is median/MAD normalized, plane-subtracted,
high-pass filtered, and Hann-windowed. These FFT-detection filters are not
applied to the maps used for site intensities. The graphite lattice constant
is 0.246 nm; the scan dimensions are supplied by `--scan-size-nm X Y`.

Equivalent lattice positions are averaged onto a 64 × 64 grid.
Median/MAD-normalized folded cells are aligned by integer circular shifts.
The same lattice and shifts are used for the uncorrected-current branch.
Output suffixes `_raw`, `_detrended`, and `_norm` denote uncorrected
current, background-corrected current, and normalized motif quantities.

## Site coordinates

Both graphite triplet orientations and a 32 × 32 origin grid are searched
once on the arithmetic mean of the phase-aligned normalized cells.
The resulting A/B/H positions are fixed across all acquisitions.
No per-image A-site refinement is used.

Site signals are periodic Gaussian-weighted means with fractional width 0.055.
For the 120° direct cell, the distance is
\(d^2=\Delta u^2+\Delta v^2-\Delta u\Delta v\), minimized over periodic images.

\[
c_{AH}=(I_A+I_B-2I_H)/\sqrt{6},\qquad c_B=(I_B-I_A)/\sqrt{2},
\]
\[
R=\sqrt{c_{AH}^2+c_B^2},\qquad \theta=\operatorname{atan2}(|c_B|,c_{AH}).
\]

## Readout and SNR

`03_compute_readout_attenuation.py` uses \(q^2=h^2+hk+k^2\).
The first and √3 shells have \(q^2=1\) and \(q^2=3\).
For each acquisition, the reference is the arithmetic mean of the complex
Fourier coefficients of the nearest 51 other acquisitions in θ (or all other
finite-θ acquisitions when fewer are available). The target is excluded.
The optional `--theta-sigma-deg` applies Gaussian weights to this complex mean.

Shell amplitudes \(A_1\) and \(A_{\sqrt3}\) are arithmetic means of the six
coefficient magnitudes in each shell. Reference magnitudes are taken only
after averaging the complex coefficients. The relative attenuation is
\[
w=-\tfrac12\{\log(A_{\sqrt3}/A_1)_i-
[\log(A_{\sqrt3}/A_1)]_{\mathrm{ref}(\theta_i)}\}.
\]
For η, the √3 shell supplies three independent directions. For each direction,
the conjugate-pair amplitude gives
\(\delta_m=\log A_{i,m}-\log A_{\mathrm{ref},m}\). An unweighted fit uses
\[
\delta_m-\overline\delta=-\tfrac32
\{\kappa_c\cos(2\phi_m)+\kappa_s\sin(2\phi_m)\},\qquad
\eta=\tfrac12\sqrt{\kappa_c^2+\kappa_s^2}.
\]
The first and 2× shells are not included in the η fit. Nonpositive amplitudes
give undefined log ratios rather than being replaced by an artificial floor.
The per-direction values are written to `sqrt3_directional_fit.csv`.
The three-direction fit has zero residual degrees of freedom; its residuals
are not used as a goodness-of-fit statistic.

`05_compute_snr_resolved_statistics.py` defines √3 SNR as the mean
amplitude of the six \(q^2=3\) modes divided by the median amplitude of the
12 \(q^2=7\) modes. Descriptors are estimated before applying `SNR >= 5`.
Selected Spearman correlations use finite pairs in this subset, with no
additional outlier or w-range filter.

## Reconstruction and figure tables

`02_validate_quantile_reconstruction.py` excludes each θ-quantile window
when estimating the empirical Fourier kernel (relative ridge coefficient
\(2\times10^{-3}\)). The site registry still comes from the full ensemble,
and target intensities come from held-out maps: this is a kernel validation,
not a fully independent registry validation.

`04_compute_supplementary_source_data.py` writes shell detectability,
synthetic-recovery, fit-robustness, and supplementary tables.
`04_export_figure_data.py` writes Figure 4/5 plotting tables.
The Figure 4a/b ensemble uses all folded maps; readout distributions and
Figure 5 statistics use the SNR-resolved subset.
