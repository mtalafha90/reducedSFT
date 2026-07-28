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
parameters (b_TQ = 0.15, b_LQ = 2.4°, b_I = 0.2, λ_R = 20° prescribed) and
for flux-balanced polarity rings.

## Headline results

| Test | Result |
|---|---|
| Exact kernel prediction of the shift-mode scan | max relative error **1.5 × 10⁻¹³** |
| Exact kernel prediction of the tilt scan | max relative error **7.6 × 10⁻¹⁵** |
| Improved (cubic-yield) tilt closure | error 1.16 × 10⁻² → **4.2 × 10⁻⁵** |
| Pair-yield effectivity range (derived) | λ_R,pair = **12.1°** (fit over 3°–27°) |
| Pair yield sign | **positive at every latitude**, 3.26 at 2° down to 3.1 × 10⁻⁴ at 60° (final-to-emergence dipole ratio) |
| Gaussian closure with the derived λ_R,pair | max \|ΔQ\| = **1.1 × 10⁻²** over the full scan |
| Gaussian closure with the prescribed λ_R = 20° | max \|ΔQ\| = **0.35** |
| Physical inflows, 2 m s⁻¹ | b_eff(T = 1) = 0.149; mildly super-quadratic |
| Physical inflows, 5 m s⁻¹ | b_eff(T = 1) = 0.411; still positive, no reversal |

## 1. The pair-yield function

The dipole yield of a ring pair is the ratio of the final axial dipole it
produces to the dipole it deposits at the moment of emergence; the source
amplitude cancels between the two, so the ratio is dimensionless. Computed
from the asymptotic kernel as a function of the pair-centre latitude
(Figure V1), it is approximately Gaussian over the activity belt with a
**derived effectivity range λ_R,pair = 12.1°**. With flux-balanced polarity rings — each ring
normalised by its own surface integral, so that the two polarities of a pair
carry equal and opposite flux — the yield is **positive at every latitude
sampled** and falls monotonically towards the pole, from 3.26 at 2° to
3.1 × 10⁻⁴ at 60°. Two points follow:

1. The point-source yield (framework document: λ_R ≈ 29.2°) and the pair
   yield are different objects. The quantity relevant to latitude quenching
   of bipolar regions is the pair yield, and it is considerably narrower,
   because the two polarities of a pair partially cancel.
2. A *positive* multiplicative closure Q_LQ > 0 is therefore structurally
   appropriate: there is no latitude at which additional emergence subtracts
   from the final dipole.

**Why balance matters.** Equal *peak* amplitudes do not give equal
integrated flux, because the surface measure is dμ = cos λ dλ; over the
emergence belt the poleward ring is short by 2–4 %. The assembled source is
antisymmetric and carries no net flux either way — what differs is the
balance *within* each pair. That imbalance is small, but the final dipole is
a small difference between large cancelling contributions, so its effect is
not. `polarity_balance_sensitivity.py`
quantifies this directly (see §5).

## 2. Physical (shift-mode) latitude quenching

The SFT was run with the belt physically displaced poleward by δ = b_LQ T²
(Figure V2, in efficiency units Q(T) = D/(aT)):

* **The kernel predicts the physical runs exactly** (max relative error
  1.5 × 10⁻¹³ on the raw dipole). The source geometry changes with T, so
  this is a genuine prediction of a nonlinear response from one linear
  adjoint solve, not a factor identity.
* **The derived Gaussian closure tracks the physical shift closely.** Using
  the derived λ_R,pair = 12.1° together with the yield-weighted time moments
  — every ingredient measured, nothing tuned — the closure reproduces the
  physical response to within max \|ΔQ\| = 1.1 × 10⁻² across the whole scan
  (8.0 × 10⁻³ over the weak-shift domain T ≤ 0.8), against a suppression
  that reaches Q(1) = 0.68.
* **The prescribed λ_R = 20° closure does not.** With the same time moments
  but the closure's demonstration value of λ_R, the error is 0.35 over the
  full scan and 0.134 for T ≤ 0.8 — more than an order of magnitude worse.
  The effectivity range must be *derived* from the transport, not chosen.

**Implication for the manuscript.** Efficiency-mode Q_LQ and a physical belt
shift with the *same nominal* b_LQ are equivalent only if the efficiency
range is the pair yield's. Retaining λ_R = 20° while reproducing the
physical shift would require an efficiency-mode strength larger by
(20/12.1)² ≈ 2.7, i.e. b_LQ ≈ 6.6. That combination remains stable at
moderate gains, but it reaches period doubling at a lower critical gain than
the reference value does, so the two parametrisations are not
interchangeable in the stability analysis either.

## 3. Tilt quenching: exact prediction and an improved closure

The tilt case modifies the source *geometry* (polarity separation
Δλ(T) = Δλ₀/(1 + b_TQ T²)), so it is not a scalar source factor — yet the
kernel predicts the scan **exactly** (max relative error 7.6 × 10⁻¹⁵,
Figure V3), because the response to *any* source is available once the
kernel is known.

The cycle-aggregated yield as a function of separation is measured to be

$$
A(s) = c_1 s + c_3 s^3, \qquad c_3/c_1 = +6.4\times10^{-4}\ \mathrm{deg}^{-2},
$$

confirming the odd-yield structure asserted in the review document and
quantifying its cubic term. The resulting **improved closure**

$$
Q_{\rm TQ}(T) = \frac{A(\Delta\lambda_0 f_{\rm TQ}(T))}{A(\Delta\lambda_0)},
\qquad f_{\rm TQ} = \frac{1}{1 + b_{\rm TQ}T^2},
$$

reduces the closure error across the scan from 1.16 × 10⁻² (naive
Q_TQ = f_TQ) to **4.2 × 10⁻⁵** — the same kind of three-orders-of-magnitude
gain that the yield-weighted moments deliver for latitude quenching. The
naive closure remains a good approximation (≤ 0.8 % relative over the scan),
and its error now has a measured, first-principles origin.

## 4. Physical converging inflows

Inflows modify the *transport operator*, so no kernel prediction exists —
which is precisely what makes this the hardest test for a source-side
rational factor Q_I = 1/(1 + b_I T²) (Figures V4, V5):

* At the reference inflow amplitude (2 m s⁻¹ at T = 1, scaling as T²), the
  effective strength at the calibration point is b_eff(T = 1) = **0.149**,
  and the leading-order fit over T ≤ 1.2 gives b_I ≈ 0.150. The rational
  form is a good description for weak-to-moderate cycles, but **b_eff grows
  with T** (full-scan fit 0.183): the physical nonlinearity is mildly
  super-quadratic, and the constant-b form slightly under-predicts the
  suppression of strong cycles.
* At 5 m s⁻¹ the effective strength rises to b_eff(1) = **0.411** (full-scan
  fit 0.584), and the departure from a constant b is correspondingly larger.
  The dipole nevertheless stays positive across the whole amplitude scan, so
  the rational factor remains inside its domain of validity; it simply
  loses accuracy.
* Cross-calibration: b_eff(1) = 0.149 at 2 m s⁻¹ and 0.411 at 5 m s⁻¹ imply
  that the paper-calibrated b_I = 0.2 (Talafha, Petrovay and Opitz 2025)
  corresponds to an inflow amplitude of roughly **2.6 m s⁻¹** at cycle
  maximum in this model's convention — a useful consistency check between
  the reduced parameter and the physical flow speed.

## 5. Sensitivity to polarity-flux balance

`polarity_balance_sensitivity.py` recomputes the same diagnostics with the
earlier equal-peak rings, using an identical fit window so that the two
estimators are directly comparable:

| Quantity | Equal peaks | Flux balanced |
|---|---|---|
| Linear yield a | 7.90 × 10⁻² | **1.68 × 10⁻¹** |
| Tilt closure error | 2.01 % | **0.78 %** |
| λ_R,pair (fit 3°–27°) | 9.17° | **12.07°** |
| Pair-yield sign change | at 25° | **none** |
| Shift-mode reversal | T_rev = 1.12 | **none** |
| Sign change across the (η, u₀) plane | 63 of 63 sets, lowest at 16° | **8 of 63, lowest at 32.5°** |
| Negative linear yield a | 21 of 63 sets | **0 of 63** |
| Scaling-law scatter | 59 % | **5.8 %** |

The imbalance is a 2–4 % error in the source, but the final dipole is a
small residual of large cancelling contributions, so it changes the linear
yield by more than a factor of two and manufactures a sign change inside the
emergence belt (which reaches 27°). Under balance the eight remaining sign
changes all lie at 32.5° or higher — above the belt — and the deepest
negative excursion is 0.09 % of the equatorward peak, so no result in this
work depends on them.

## 6. What this adds to the theory

1. **The kernel framework is validated against physical, geometry- and
   position-changing nonlinearities** — not just injected factors — at the
   10⁻¹³ level, for the cost of one adjoint solve.
2. **Closures have measured domains of validity.** Tilt: excellent
   everywhere, essentially exact with the cubic yield. Latitude (efficiency
   mode): excellent with yield-weighted moments. Latitude (physical shift):
   accurate to about 1 % in Q *provided* the derived λ_R,pair is used.
   Inflows: rational form good at observationally calibrated strengths,
   mildly super-quadratic beyond.
3. **The manuscript should distinguish carefully between the two
   effectivity ranges**: the prescribed closure parameter (20°, a
   demonstration value) and the derived pair yield (12.1°). They are not
   interchangeable, and only the latter is a property of the transport.
