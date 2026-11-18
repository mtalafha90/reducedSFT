# Settled-dipole closure of the multi-cycle map (PA01)

Run from the root of this repository with NumPy, SciPy and Matplotlib
installed, passing a checkout of the pinned solver commit (see
`../README.md`):

```bash
git worktree add ../reducedSFT-41640e9 41640e936c5d777a168ff5c0753491b840263c49
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python revisions/settled_dipole/verify_settled_dipole.py --solver ../reducedSFT-41640e9
```

The solver must be at commit `41640e936c5d777a168ff5c0753491b840263c49`, the
commit cited in the manuscript; the script refuses any other commit. It imports only
the three published modules whose SHA-256 hashes it records in
`settled_dipole.json`. No solver file is modified. The run takes about half a
minute on one core.

## What is computed

The configuration is the published multi-cycle experiment: kernel-exact
latitude efficiency times scalar inflow efficiency, no tilt quenching,
τ = ∞, 181 cells, one-day update. The forward fast stepper of
`multicycle_feedback_analysis.py` gives the one-cycle propagator G; the
discrete adjoint of `mathematical_framework_analysis.py` gives the kernel.

1. **Premise.** The second eigenvalue of G has modulus 2.3 × 10⁻⁵. A second
   source-free cycle multiplies the dipole of four very different fields
   (a linear cycle, cycles of amplitude 0.5 and 1.5, uniform polar caps) by
   the same factor λ₁ = 0.9999858 to within 2 × 10⁻⁹. The weight that
   defines the settled dipole, Gᵀw, equals the adjoint kernel at the start
   of a cycle, (I + ΔtA)ᵀK⁰, to round-off.
2. **Coefficients.** First-cycle survival of the reference linear cycle,
   r₁ = 0.993962, is the published r. The settled-dipole map gives
   r\* = r₁/(1 + λ₁ − r₁) = 0.988011, so the complete field has a non-zero
   fixed point for D₀ > 1.988 (scalar map: 1.994).
3. **Stability.** The two-variable map's leading multiplier agrees with that
   of the complete cycle-to-cycle field map, J = −G + k (db/dT) wᵀ, to
   better than 10⁻⁷ at every point checked. Period doubling of the complete
   field occurs at D₀ = 3.3349 (T\* = 1.28923), where the settled dipole of
   the new flux peaks (T = 1.28923); the published scalar-map value 3.35216
   is reproduced and lies 0.52 % higher. At D₀ = 2.05, 2.50 and 3.20 the
   complete-field multipliers are 0.8792, 0.1456 and −0.8447, with
   relaxation scales of 7.77, 0.52 and 5.92 cycles; the scalar-map values
   (0.8898, 0.1174, −0.8234; 8.57, 0.47, 5.15 cycles) reproduce the Point 13
   record. The modulus of the complete field's leading multiplier has a
   minimum of 0.0819, at D₀ = 2.564; it never vanishes.
4. **Nonlinear check.** 120 feedback cycles of the complete multi-cycle SFT
   at D₀ = 3.20, 3.34 and 3.60 are compared with the two-variable map, which
   uses the exact P(T) and P_s(T) of each cycle, and with the published
   scalar map. The two-variable map follows the SFT amplitudes to within
   1.3 × 10⁻⁸; the scalar map differs by 0.022–0.081.

These are deterministic properties of the adopted transport and source. They
do not test resolution convergence, stochastic sources or other transport
parameters.
