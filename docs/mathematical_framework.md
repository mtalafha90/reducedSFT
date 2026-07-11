# The Mathematical Framework of the Reduced Response Theory

**Companion analysis:** `mathematical_framework_analysis.py`; figures and the
results table (`framework_results.csv`) in `figures_framework/`. This
document deepens the framework established in
`theory_review_and_enhancement.md`: it makes the dipole-yield kernel (the
adjoint state of the SFT model) fully explicit at the discrete level, and
uses it to *derive* several quantities that the reduced theory otherwise
prescribes. All numbers quoted below are for the reference parameters
(u₀ = 12 m s⁻¹, η = 500 km² s⁻¹, τ = ∞, b_TQ = 0.15, b_LQ = 2.4°,
b_I = 0.2, λ_R = 20°).

---

## 1. The discrete adjoint and exact duality

The experiment's forward update is, in matrix form,

$$
B^{k+1} = \Pi\, M^{-1}\left[(I + \Delta t\, \mathsf{A})\,B^{k}
        + \Delta t\, S^{k}\right],
\qquad k = 0, \dots, N-1,
$$

where 𝖠 is the explicit operator (upwind advection + decay), M the implicit
diffusion matrix, and Π = I − (1/n)𝟙𝟙ᵀ the monopole projection. Because the
finite-volume diffusion stencil is symmetric, M = Mᵀ, and Π = Πᵀ, the final
dipole D = wᵀB^N (with w the trapezoidal weights of 3/2 ∫Bμ dμ) can be
written **exactly** as a sum over emergence steps:

$$
D = \sum_{k=0}^{N-1} \Delta t\, (K^{k})^{\mathsf T} S^{k}
\;\equiv\; \sum_k \omega_k \quad (\text{for the linear source}),
$$

with the **dipole-yield kernel** obtained by one backward recursion,

$$
K^{N-1} = M^{-1}\Pi\, w, \qquad
K^{k-1} = M^{-1}\Pi\,(I + \Delta t\,\mathsf{A})^{\mathsf T} K^{k}.
$$

This is the discrete counterpart of the adjoint transport equation given in
the review document, but derived directly from the code's own update, so the
duality is exact rather than approximate. Verified numerically:

> **Duality check:** D from the adjoint sum and D from the forward linear
> run agree to a relative difference of **1.9 × 10⁻¹⁵** (round-off), and
> both equal the linear coefficient a = 7.8986 × 10⁻² at T = 1.

The practical significance: **one adjoint solve replaces the entire
amplitude scan** for any source-side nonlinearity. Since a source-side
factor Q(t, T) enters linearly, the SFT's response is available in closed
form as D(T) = T Σₖ ωₖ Q(tₖ, T) without running the SFT again (Section 3).

## 2. The kernel and the intrinsic dynamo effectivity range

Figure F1 shows K(μ, t) for different emergence times: flux emerging early
in the cycle has its yield profile fully relaxed by cycle end, while
late-cycle emergence retains a sharper profile.

The **yield function** f(λ) = K⁰/(Πw) — the final dipole produced by a unit
source at latitude λ, relative to the dipole it deposits instantaneously —
is shown in Figure F2 with its Gaussian fit over the activity latitudes
(5°–45°):

| Quantity | Value |
|---|---|
| λ_R at the 11-yr horizon | **29.16°** |
| λ_R asymptotic (kernel evolved a further 40 yr) | **29.16°** |
| Naive advection–diffusion balance estimate √(η/2u₀R) | 9.9° |
| λ_R prescribed in the closure | 20° |

Two conclusions:

1. **The kernel converges within the cycle window** — the 11-yr and
   asymptotic ranges are identical to five significant figures, so 29.2° is
   the transport model's *intrinsic* dynamo effectivity range, not a
   finite-horizon artefact. The naive boundary-layer estimate underestimates
   it because cross-equatorial flux loss happens throughout the poleward
   transit, not only within the equatorial boundary layer; at the moderate
   magnetic Reynolds number of this model (R_m = u₀R⊙/η ≈ 16.7) the range
   is wide, consistent with the growth of λ_R with diffusivity reported by
   Petrovay, Nagy and Yeates (2020) and its role in Talafha et al. (2022).
2. **The prescribed λ_R = 20° is a free demonstration parameter, not the
   model's own.** The controlled experiment is internally consistent because
   the factor is injected on both sides; but a *physical* latitude-quenching
   run (`latitude_quenching_mode = "shift"`) would be governed by the
   intrinsic 29.2°, i.e. weaker quenching than the prescribed closure.
   Since the relative importance of latitude quenching scales as 1/λ_R²,
   this distinction matters for any observational calibration.

**A sign change no positive closure can capture.** The per-time pair yield
ω(t) (Figure F3) is slightly *negative* for emergence in the first ≈ 2 yr
of the cycle, when the belt sits above ≈ 25°: at such latitudes the ring
pair's final dipole contribution reverses sign. A multiplicative factor
Q_LQ > 0 cannot represent locally negative yields; the aggregate response
remains well described (Section 3) because the envelope suppresses those
early times, but this is a structural limitation of Gaussian-factor
closures worth stating in the manuscript.

## 3. Exact source-side reduction and the closure ladder

For the latitude case the source is the linear source multiplied by the
instantaneous scalar factor Q_LQ(t, T), so by linearity the reduced
prediction is **exact**:

$$
D_{\rm LQ}(T) = T \sum_k \omega_k\, Q_{\rm LQ}(t_k, T).
$$

Truncating the cumulant expansion of this weighted average gives the ladder
of closures, each evaluated against the stored SFT scan (Figure F4):

| Closure level | Time weighting | Effective λ̄₀ | max \|ΔP̂\| over the scan |
|---|---|---:|---:|
| Exact kernel average | ωₖ (kernel × source) | — | 3.9 × 10⁻¹¹ * |
| Yield-weighted Gaussian | Gaussian with t̄ = 6.52 yr, σ = 1.48 yr | 19.9° | **5.5 × 10⁻⁵** |
| Envelope Gaussian (default) | Gaussian with t_max = 4.5 yr, σ_t = 2.2 yr | 22.1° | 6.6 × 10⁻² |
| Static (original draft) | δ-function at cycle end | 15.0° | 1.7 × 10⁻¹ |

\* limited by the 10-digit precision of the stored CSV, not by the identity.

Findings:

* The **entire residual** of the default closure reported in the review
  document (≈ 0.066 at the largest amplitudes) is explained: it is the
  covariance between the instantaneous quenching factor and the
  time-dependence of the dipole yield, i.e. the difference between the
  envelope weighting s₁(t) and the true weighting ω(t).
* The true weighting peaks at t̄ = **6.5 yr**, two years after cycle
  maximum (Figure F3): early-cycle flux emerges at high latitude where its
  yield is small (even negative), so the final dipole is built
  predominantly by mid-to-late-cycle emergence at lower latitudes.
* Replacing the envelope moments by the yield-weighted moments — the same
  closed-form Gaussian closure, with (t_max, σ_t) → (t̄_ω, σ_ω) measured
  once from a single adjoint solve — reduces the closure error by three
  orders of magnitude, to **5.5 × 10⁻⁵**, indistinguishable from the SFT in
  Figure F4. The third and higher cumulants of the weighting are
  negligible.
* The ladder cleanly ranks the physical content of the reference latitude:
  static (15°, cycle-end belt) → envelope (22.1°, belt at cycle maximum) →
  yield-weighted (19.9°, dipole-production-weighted belt).

**Recommended refinement for the manuscript:** keep the closed-form Gaussian
factor of the review document, but state that its two time moments should be
the yield-weighted ones, obtainable from one linear run plus one adjoint
solve (or, in practice, from the linear run's dipole-contribution history).
The envelope moments remain a good parameter-free approximation (error
≈ 7 % at worst, ≲ 2 % in the observed range of solar variability).

## 4. The cycle map beyond linear stability

The review document established the analytic period-doubling criterion
q(T\*) ≥ 2. The framework analysis confirms and extends it dynamically
(Figure F5):

* For the reference quenchings the analytic boundary
  𝓜′(T\*) = −1 falls at **D₀ = 2.918**, and the numerically iterated map
  bifurcates to period 2 exactly there, followed by a period-doubling
  cascade into chaotic modulation as the gain increases. In the chaotic
  regime the amplitude makes repeated excursions towards T ≈ 0 —
  qualitatively grand-minimum-like behaviour arising from the
  super-exponential tail of the Gaussian latitude quenching.
* At fixed D₀ = 3, increasing b_LQ from zero destabilises the fixed point
  near b_LQ ≈ 2° and produces bounded period-2 (then period-4) amplitude
  alternation — persistent even–odd cycle modulation without losing the
  cycle altogether.

Together with the identity 𝓜′(T\*) = Q(1)ℛ̂(T\*)/Q(T\*) from the review
document, this closes the loop: the single-cycle response function measured
in the controlled experiments determines quantitatively where the
multi-cycle system sits relative to the onset of modulation.

## 5. What is now derived versus prescribed

| Quantity | Status |
|---|---|
| Linear coefficient a | **Derived**: a = Σₖωₖ = 7.8986 × 10⁻², exact duality 10⁻¹⁵ |
| Reduction P = aTQ for source-side factors | **Exact identity** given the kernel (not an ansatz) |
| Latitude-quenching closure | **Derived** to 5.5 × 10⁻⁵ (yield-weighted Gaussian); envelope form is its parameter-free approximation |
| Effectivity range λ_R | **Measured**: 29.2° intrinsic to the transport; the closure's 20° is a free demonstration parameter |
| Tilt/inflow algebraic forms | Prescribed (Padé closures; see review document §3.2, §3.4) |
| Stability boundary of the cycle map | **Analytic**, confirmed dynamically to the resolution of the bifurcation scan |

## 6. Additions recommended for the manuscript

1. Present the adjoint/kernel formulation as the *definition* of the reduced
   theory for source-side nonlinearities (Section 1 here), with the duality
   check as validation; the reduced relation P = aTQ(T) is then exact for
   time-independent factors and a controlled second-cumulant approximation
   for time-dependent ones.
2. Quote the intrinsic λ_R = 29.2° of the reference transport alongside the
   prescribed λ_R = 20°, and note the 1/λ_R² scaling of latitude-quenching
   importance.
3. Note the sign reversal of the pair yield above ≈ 25° as a structural
   limit of positive multiplicative closures.
4. Give the closure ladder (table above) as the error budget of the
   latitude-quenching factor.
5. Use the bifurcation diagram to illustrate that the analytic boundary
   marks the onset of bounded even–odd modulation, not cycle collapse.
