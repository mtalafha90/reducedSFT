# Final manuscript audit — 13 November 2026

This audit starts from manuscript commit
`73eb6891a2cdda977d25a8c3e15b1054f929e488` and finishes the final review of
the general concern and all 14 referee points. Exact manuscript changes,
including deletions, are recorded under FA01 in `revision_log.tex` in the
manuscript repository (`mtalafha90/Reduced_SP_manuscript`).
The original `main.tex` is preserved.

## Reproduce the targeted numerical checks

With NumPy, SciPy, Pandas and Matplotlib installed, run from the root of this
repository, passing a checkout of the pinned solver commit
`41640e936c5d777a168ff5c0753491b840263c49` (see `../README.md`):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 revisions/final_audit/verify_claims.py --solver ../reducedSFT-41640e9
```

The script checks the hashes of the five imported solver modules before
calculating the reference kernel weights and their moments, the pair-yield
width at two horizons, the reference remnant coefficient, and the effect of
finite decay with a fixed or regenerated initial profile. It also recomputes
the inflow fits from the archived Point 5 responses. `verification.json`
contains the results and input hashes. These are targeted checks; the full
resolution study, 63-model transport scan and bifurcation scan were not rerun.

The manuscript now distinguishes a fitted inflow coefficient from its
pointwise value, a continuous decay identity from its finite-step numerical
implementation, and remnant survival from local amplitude relaxation.
The new moment definitions specify the temporal weighting of the Gaussian
closure. No source prescription or baseline solver is changed.

## Figure 8

The current `../point9/plot_figures.py` and `figure_layout.json` use
“empirical fit”, an approximate equality, “one-to-one reference”, and the
axis label “Transport scale lambda_tr [deg]”. (Entry PA03 later redrew all
eight figures in the journal's artwork style; units are now in parentheses,
“(deg)”.) Equation (51) in the manuscript gives the degrees conversion
explicitly; it was Equation (37) until entry PA09 numbered every displayed
equation. Only Figure 8 was regenerated in
this audit. Its frozen coordinates, fitted coefficients, dimensions and font
sizes are unchanged. The other seven figure PDFs are byte-identical to the
pre-audit versions. Point 9's original verification reports remain historical
records of that earlier rendering.

## Verification scope and limits

The manuscript compiles to 33 pages, retaining all 46 cited works and all
59 numbered reference values. The scaling fit uses an approximate equality
without changing its equation number. There are no unresolved references,
undefined citations, overfull lines or BibTeX warnings. The PDF and corrected
figure were visually inspected; two underfull vertical-box warnings concern
page spacing, with no clipping or overlap.

The reference grid still does not establish percent-level accuracy for the
absolute dipole or physical-response curves. Convergence of the transport
plane and bifurcation thresholds has not been established. Polar-cap and
reversal-time extensions remain analytical proposals; no observational
calibration or stochastic ensemble is added by this audit.
