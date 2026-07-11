# Physical Validation of the Reduced Response Theory

**Companion analysis:** `physical_validation_analysis.py`; figures and the
results table (`validation_results.csv`) in `figures_validation/`.

The controlled experiments of the main paper validate the reduced theory in
*efficiency* mode, where the nonlinear factors are injected into the source
term. This analysis tests the theory against the **physical**
implementations of the nonlinearities available in the code — the poleward
belt shift ("shift" mode) and explicit converging inflows ("physical"
inflow mode) — using only quantities *derived* from the adjoint-kernel
framework, with no tuned parameters. All results are for the reference
parameters (b_TQ = 0.15, b_LQ = 2.4°, b_I = 0.2, λ_R = 20° prescribed).

## Headline results

| Test | Result |
|---|---|
| Exact kernel prediction of the shift-mode scan | max relative error **2 × 10⁻¹³** |
| Exact kernel prediction of the tilt scan | max relative error **1 × 10⁻¹⁴** |
| Improved (cubic-yield) tilt closure | error 0.030 → **0.0001** |
| Pair-yield effectivity range (derived) | λ_R,pair = **10.5°** (fit over 5°–22°) |
| Pair-yield **sign change** | λ_rev ≈ **25°** |
| Shift-mode response sign reversal | measured at **T_rev = 1.11** |
| Gaussian closures for shift mode | fail beyond weak shifts (see below) |
| Physical inflows, 2 m s⁻¹ | b_eff(T = 1) = 0.40; super-quadratic growth |
| Physical inflows, 5 m s⁻¹ | dipole sign reversal at T = 1.40 |

## 1. The pair-yield function and its sign change

The dipole yield of the actual four-ring source structure, computed from the
asymptotic kernel as a function of the pair-centre latitude (Figure V1), is
approximately Gaussian over the activity belt with a **derived effectivity
range λ_R,pair = 10.5°** — but it **changes sign at λ_rev ≈ 25°**: pairs
emerging poleward of 25° contribute to the final dipole with *reversed*
sign in this transport model. Two consequences follow immediately:

1. The point-source yield (framework document: λ_R ≈ 29°) and the pair
   yield are different objects; the quantity relevant to latitude quenching
   of bipolar regions is the pair yield, and it is both narrower and
   sign-changing.
2. Any *positive* multiplicative closure Q_LQ > 0 — Gaussian or otherwise —
   is structurally limited to the regime where the emergence-weighted belt
   stays equatorward of λ_rev.

## 2. Physical (shift-mode) latitude quenching

The SFT was run with the belt physically displaced poleward by δ = b_LQ T²
(Figure V2, in efficiency units Q(T) = D/(aT)):

* **The kernel predicts the physical runs exactly** (max relative error
  2 × 10⁻¹³ on the raw dipole; the source geometry changes with T, so this
  is a genuine prediction of a nonlinear response from one linear adjoint
  solve, not a factor identity).
* **The response reverses sign at T_rev = 1.11**: for stronger cycles the
  belt spends enough of the cycle poleward of λ_rev that the negative-yield
  contributions win. The a-priori estimate from the closure geometry,
  T ≈ √((λ_rev − λ̄₀)/b_LQ) = 1.46, gives the right scale; it is optimistic
  because the early-cycle belt (27° + δ) enters the negative-yield region
  first.
* **Gaussian closures fail well before the reversal.** Even with the derived
  λ_R,pair = 10.5° and yield-weighted time moments (every ingredient
  measured, nothing tuned), the closure deviates by up to 0.30 in Q over
  the weak-shift domain T ≤ 0.8, and by 1.8 over the full scan; the
  prescribed λ_R = 20° closure is worse (0.48 and 2.2). The reason is
  structural, not a calibration issue: near the sign change the yield falls
  *linearly* through zero, while a Gaussian ratio falls log-linearly — the
  local logarithmic slope of the true yield at the weighted belt latitude
  (~20°) is more than twice the fitted Gaussian's.

**Implication for the manuscript.** The efficiency-mode Q_LQ and a physical
belt shift with the *same nominal* b_LQ are **not equivalent** in this
transport regime. The efficiency factor with λ_R = 20° corresponds to a far
milder feedback: matching the small-T behaviour of the physical shift would
require an efficiency-mode strength roughly (20/10.5)² ≈ 3.6 times larger
(b_LQ ≈ 8.7 at λ_R = 20°) — which sits at the edge of the period-doubling
region of the stability map — and even then only up to moderate amplitudes,
beyond which the response is a *signed* competition that requires the kernel
(or a signed closure) rather than any positive Q(T).

## 3. Tilt quenching: exact prediction and an improved closure

The tilt case modifies the source *geometry* (polarity separation
Δλ(T) = Δλ₀/(1 + b_TQ T²)), so it is not a scalar source factor — yet the
kernel predicts the scan **exactly** (max relative error 1 × 10⁻¹⁴,
Figure V3), because the response to *any* source is available once the
kernel is known.

The cycle-aggregated yield as a function of separation is measured to be

$$
A(s) = c_1 s + c_3 s^3, \qquad c_3/c_1 = +1.7\times10^{-3}\ \mathrm{deg}^{-2},
$$

confirming the odd-yield structure asserted in the review document and
quantifying its cubic term. The resulting **improved closure**

$$
Q_{\rm TQ}(T) = \frac{A(\Delta\lambda_0 f_{\rm TQ}(T))}{A(\Delta\lambda_0)},
\qquad f_{\rm TQ} = \frac{1}{1 + b_{\rm TQ}T^2},
$$

reduces the closure error across the scan from 0.030 (naive Q_TQ = f_TQ) to
**0.0001** — the same three-orders-of-magnitude gain the yield-weighted
moments delivered for latitude quenching. The naive closure remains a good
approximation (≤ 2 % relative over the scan, ≤ 3 % absolute in P̂), and its
error now has a measured, first-principles origin.

## 4. Physical converging inflows

Inflows modify the *transport operator*, so no kernel prediction exists —
which is precisely what makes this the hardest test for a source-side
algebraic factor Q_I = 1/(1 + b_I T²) (Figures V4, V5):

* At the reference inflow amplitude (2 m s⁻¹ at T = 1, scaling as T²), the
  effective strength at the calibration point is b_eff(T = 1) = **0.40**,
  and the leading-order fit over T ≤ 1.2 gives b_I ≈ 0.42. The algebraic
  form is adequate for weak-to-moderate cycles but **b_eff grows with T**
  (full-scan fit 1.5): the physical nonlinearity is super-quadratic, and
  the constant-b Padé form under-predicts the suppression of strong cycles.
* At 5 m s⁻¹ the dipole **reverses sign at T = 1.40** — over-quenching to
  reversal, again outside the domain of any positive closure, consistent
  with the domain-of-validity statement in the review document.
* Cross-calibration: b_eff(1) = 0.40 for a 2 m s⁻¹ inflow implies that the
  paper-calibrated b_I = 0.2 (Talafha, Petrovay and Opitz 2025) corresponds
  to an inflow amplitude of roughly **1 m s⁻¹** at cycle maximum in this
  model's convention — a useful consistency check between the reduced
  parameter and the physical flow speed.

## 5. What this adds to the theory

1. **The kernel framework is now validated against physical, geometry- and
   position-changing nonlinearities** — not just injected factors — at the
   10⁻¹³ level, for the cost of one adjoint solve.
2. **Closures have measured domains of validity.** Tilt: excellent
   everywhere, exact with the cubic yield. Latitude (efficiency mode):
   excellent with yield-weighted moments. Latitude (physical shift):
   requires the kernel beyond weak shifts; sign reversal at T_rev ≈ 1.1 for
   the reference b_LQ. Inflows: algebraic form good to T ≈ 1.2 at
   observationally calibrated strengths; super-quadratic beyond; reversal
   for strong inflows.
3. **The manuscript should distinguish carefully between the two
   implementations of latitude quenching**: they are equivalent only in the
   weak-shift limit and only if the efficiency-range parameter is the pair
   yield's (≈ 10.5°), not the point yield's (≈ 29°) nor an arbitrary
   intermediate value.
