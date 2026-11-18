# Point 11: conditional comparison of quenching mechanisms

This calculation supports the revised interpretation of Figures 1–3. It uses
`reducedSFT` at commit `41640e936c5d777a168ff5c0753491b840263c49` and the original
181-cell, 1-day reference protocol, including its trapezoidal dipole diagnostic.
It does not modify the solver, figures or archived forward simulations.

From the root of this repository, with a checkout of the pinned solver commit
next to it (see `../README.md`):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python revisions/point11/compare_quenching.py --solver ../reducedSFT-41640e9
```

Dependencies are NumPy, SciPy, pandas and the solver's dependencies (including
Matplotlib). The output is `quenching_comparison.json`, including input hashes.

For LQ, the script averages the instantaneous prescribed efficiency with the
discrete adjoint dipole-yield weights. TQ uses the exact amplitude-dependent
four-ring source, including its flux normalisation. IQ uses the prescribed
scalar factor, not an explicit converging velocity. All three are normalised
by the same unquenched yield `a*T`, and checked against the 39 corresponding
Point 5 baseline dipoles before the crossover calculation.

Brent root finding within the brackets supplied by the archived amplitude scan
gives LQ/TQ equality at `T=1.2554080489` and LQ/IQ equality at
`T=1.8679081959`. At `T=1`, the respective efficiencies are approximately
0.874568, 0.866221 and 0.833333: smaller efficiency means greater suppression.
These values are conditional on the original discretisation and prescriptions.
They are not observationally calibrated solar thresholds.

The local analytical latitude threshold compares the prescribed Gaussian LQ
factor with the leading small-separation rational TQ factor. It is distinct
from the cycle-integrated comparison. For the prescribed 20-degree effectivity
range, it approaches 25 degrees at small amplitude, is 22.0937 degrees at
`T=1`, and 14.7835 degrees at `T=2`. Substitution into the original factors
checks equality numerically. The separately measured 12.1-degree ring-pair
effectivity range is not substituted into this controlled comparison.

The interpretation also refers to Talafha et al. (2022), *A&A* 660, A92,
[doi:10.1051/0004-6361/202142572](https://doi.org/10.1051/0004-6361/202142572).
Their transport dependence is discussed in Sections 2.2–2.4 and Figure 4;
their source laws differ from those used here. The present calculation does
not establish a universal transition latitude or dominant solar-dynamo feedback.
