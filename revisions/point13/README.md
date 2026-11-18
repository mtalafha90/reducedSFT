# Point 13: local propagation of amplitude perturbations

Run from the root of this repository with NumPy, SciPy and Matplotlib
installed, passing a checkout of the pinned solver commit (see
`../README.md`):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revisions/point13/perturbation_response.py --solver ../reducedSFT-41640e9
```

The solver must be at commit `41640e936c5d777a168ff5c0753491b840263c49`.
The JSON records hashes of all three imported source files. No solver file is modified.

The calculation reuses the multi-cycle experiment's kernel-exact latitude-efficiency times scalar inflow-efficiency response, with no tilt quenching. It reconstructs the reference linear source profile and its source-free survival over one cycle. It uses the original 181 cells, one-day update, 4019 source steps and trapezoidal dipole diagnostic. No stochastic or multi-cycle SFT ensemble is run.

For the successful-reversal branch `M(T)=T[D0 Q(T)-r]`, at constant `r`, `D0` and `k`, the local multiplier is `m(T)=D0 Q(T)[1-q(T)]-r`, with `q=-T d(ln Q)/dT`. Products of these multipliers give the linear sensitivity across successive map iterations. At a non-zero fixed point, `m*=1-(1+r)q*`. For `0<|m*|<1`, the e-folding scale of the perturbation envelope is `N_rel=-1/ln|m*|` cycles. This is a local deterministic relaxation scale, not a measured stochastic correlation time or prediction horizon. At zero multiplier the first-order response vanishes; nonlinear terms remain. The formula does not describe perturbations that change the reversal branch, a state-dependent remnant coefficient or arbitrary spatial-field perturbations.

The script evaluates fixed points at D0=2.05, 2.50 and 3.20; locates the zero multiplier; reproduces the existing period-doubling threshold; checks one-step derivatives by centred finite differences; and checks six-cycle propagation by differentiating the composed map, at fixed points and along a varying trajectory. Results are in `perturbation_response.json`.

At the reference r=0.9939622569, increasing the feedback first reduces the multiplier from +1 towards zero, then makes it negative and increases its magnitude towards 1. The relaxation scale is therefore not monotonic in quenching strength. This feature limits direct comparison with the stochastic polar-to-toroidal lagged correlations of Kumar, Karak & Vashishth (2021), ApJ 913, 65, DOI [10.3847/1538-4357/abf0a1](https://doi.org/10.3847/1538-4357/abf0a1).
