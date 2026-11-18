# Point 5: numerical resolution and sensitivity

This directory records the grid and time-step study added to the revised manuscript. It uses reducedSFT commit `41640e936c5d777a168ff5c0753491b840263c49` without changing that repository or replacing the original manuscript results.

## Reproduce

Use Python with NumPy, SciPy and Matplotlib. The recorded run used Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. Matplotlib is imported by the baseline solver; the study makes no figures.

Make a checkout of the pinned reducedSFT commit next to this repository (see
`../README.md`), then run from this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python grid_sensitivity.py --code-repo ../../../reducedSFT-41640e9 --output .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python fine_time_check.py --code-repo ../../../reducedSFT-41640e9 --output .
python summarise.py
```

The driver verifies the solver's SHA-256 before running. Numerical comparisons use the exact source, upwind-advection and implicit-diffusion update, evaluated in batches with a tridiagonal band solve. Independent comparisons use the original solver and its Thomas algorithm. Scalar linear and inflow-efficiency responses follow exactly from linearity; these are also checked against the archived scan.

## Files

| File | Role |
|---|---|
| `grid_sensitivity.py` | Five-grid study, reference-grid time checks, endpoint comparison, independent implementation checks and comparison with the 65 archived controlled responses. |
| `grid_sensitivity_results.csv` | All 1456 rows: seven grid/time settings, two endpoint conventions, eight cases and 13 amplitudes. |
| `verification.json` | Pinned code identity, parameters, environment and numerical verification results. |
| `fine_time_check.py` | Additional half-step check at 1441 cells. |
| `fine_time_results.csv` | 208 rows for 1441 cells and 0.125 day, with both endpoint conventions. |
| `fine_time_verification.json` | Two independent forward comparisons at the finest grid and half step. |
| `summarise.py` | Derives grid, time and endpoint comparisons from the CSV files. |
| `summary.json` | Full-precision derived comparisons and locations of their maxima. |
| `README.md` | Method, file inventory, reproduction instructions and interpretation. |

## Definitions

- `legacy`: the original loop, including a full update for every entry in `make_grid()['t_yr']`.
- `exact_11yr`: the same update stopped at exactly 11 Julian years, shortening the final step if needed. This convention is used for all reported grid and time-step comparisons.
- `D`: the original trapezoidal dipole over the cell centres. It omits the outer half-cell interval at each pole.
- `D_midpoint`: `(3/2)*dmu*sum(mu*B)` on the **same evolved field**, covering all cells.
- `P_hat` and `P_hat_midpoint`: each diagnostic divided by its own same-case, same-grid value at T=1.
- `Q` and `Q_midpoint`: each diagnostic divided by its same-grid linear coefficient times T.
- `max_outgoing_Courant`: maximum of `dt/dmu * (max(v_right,0)-min(v_left,0))` across cells and times, with `v=u*cos(latitude)/R_sun`. The full nominal step is used in this bound, including for the shortened final step.
- `scan_wall_seconds`: time for the batched 79-state calculation, not a portable benchmark or the time for a single original-solver run. Two curves are derived by exact scalar multiplication, giving eight cases in the output.

The eight cases are linear, tilt quenching, latitude efficiency, inflow efficiency, combined efficiencies with explicit tilt separation, a physical poleward belt shift, and physical inflows of 2 and 5 m/s at T=1, scaling as T squared. All use the 13 original amplitudes from 0.2 to 2.5.

## Findings and limits

Spatial comparisons hold the step at 0.25 day and compare with 1441 cells. At 181 cells, the maximum relative change in the five controlled normalised curves is 0.0535% with the original diagnostic and 0.0739% with the full-cell diagnostic. Physical-case changes are 3.5501% and 8.5004%, respectively. All dipoles remain positive; the controlled sampled peak amplitudes are unchanged. These are inter-grid differences, not error bounds relative to a continuum solution.

The raw linear coefficient is 11.1299% lower at 181 than at 1441 cells using the original diagnostic. The full-cell estimates differ by 0.7010%. Therefore, neither the original absolute scale nor every physical response has percent-level convergence at 181 cells. The full-cell comparison exposes quadrature sensitivity; it is not a silent replacement of the original published calculations.

At 181 cells, reducing the step from 1 to 0.25 day changes normalised curves by at most 0.03046% and raw dipoles by at most 0.03904% across both diagnostics. At 1441 cells, reducing 0.25 day to 0.125 day gives at most 0.004801% and 0.006770%. Original-loop durations exceed 11 years by 1.25, 0.75 and 0.5 day for steps of 1, 0.5 and 0.25 day. The exact-endpoint adjustment changes the one-day normalised response by at most 0.03711%.

Only the reference single-cycle scans are tested here. The parameter-plane calculations and multi-cycle bifurcation thresholds have not been independently refined. Work and field storage for the direct finite-volume forward update scale with the number of cells per case; this is not a scaling statement about dense-matrix adjoint routines.
