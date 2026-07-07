# Review and Enhancement of the Reduced Response Theory

**Subject:** *A Reduced Response Theory for Nonlinear Axial-Dipole Modulation in
Babcock–Leighton Surface Flux Transport Models* (draft of 6 July 2026) and the
accompanying code `nonlinear_response_sft_experiment.py`.

This document records a full check of the theory against the implementation,
lists the corrections that were needed, and presents an enhanced formulation
that is physically and mathematically self-consistent. Equation numbers such
as (3) or (25) refer to the draft manuscript.

---

## 1. Summary of findings

| # | Item | Status |
|---|------|--------|
| 1 | Transformation of the SFT equation to the coordinate μ = sin λ, Eq. (3) | **Verified correct** (Section 2.1) |
| 2 | Conservative finite-volume scheme with implicit diffusion, Eqs. (8)–(10) | **Verified correct**; monopole removal, Eq. (11), is redundant in exact arithmetic (Section 2.2) |
| 3 | Axial-dipole diagnostic, Eq. (37) | **Verified correct**: it is exactly the dipole coefficient g₁₀ (Section 2.3) |
| 4 | Fixed-point algebra of the cycle map, Eqs. (49)–(55) | **Verified correct** (Section 2.4) |
| 5 | The linear coefficient *a* in P(T) = aTQ(T), Eq. (1) | **Undefined in the draft** — now defined through the adjoint (Green's function) formulation (Section 3.1) |
| 6 | Tilt-quenching closure Q_TQ = f_TQ, Eqs. (21)–(24) | **Approximation, not identity** — it is the small-separation linearisation of an odd yield function; error O(Δλ₀²), below one per cent here (Section 3.2) |
| 7 | Latitude-quenching factor referenced to the static base latitude λ₀ = 15°, Eqs. (25)–(26) | **Physically inconsistent** with the drifting activity belt (15°–27°) and with the "shift" mode — corrected to a belt-consistent factor with a closed-form envelope average (Section 3.3) |
| 8 | Inflow-like factor Q_I applied to the source, Eqs. (27)–(28) | Acceptable as a closure, but the underlying assumptions must be stated; justified as a Padé [0/1] resummation (Section 3.4) |
| 9 | Product closure Q = Q_TQ Q_LQ Q_I, Eq. (32) | Requires an explicit independence assumption (Section 3.5) |
| 10 | Domain of validity | The reduced theory has Q(T) > 0, so it can describe an over-quenched *slope* (dP/dT < 0) but **never a sign reversal** of the dipole; must be stated (Section 3.6) |
| 11 | Stability classification of the map | Enhanced with exact analytic results: existence and uniqueness of T\*, the quenching-stiffness criterion, and closed-form stability boundaries (Section 3.7) |
| 12 | Link between the response function ℛ and map stability | New exact identity connecting Sections 1.13 and 1.14 of the draft (Section 3.8) |
| 13 | Code defects | `np.trapz` removed in NumPy 2.0; results CSV written without line breaks; face velocities interpolated rather than evaluated; wrong return annotation — all fixed (Section 4) |

---

## 2. What was verified and found correct

### 2.1 The SFT equation in μ = sin λ

The standard azimuthally averaged SFT equation in latitude λ is

$$
\frac{\partial B}{\partial t}
= -\frac{1}{R_\odot \cos\lambda}\frac{\partial}{\partial\lambda}
  \left[u(\lambda)\,B\cos\lambda\right]
+ \frac{\eta}{R_\odot^2 \cos\lambda}\frac{\partial}{\partial\lambda}
  \left[\cos\lambda\,\frac{\partial B}{\partial\lambda}\right]
- \frac{B}{\tau} + s .
$$

With μ = sin λ and ∂/∂λ = cos λ ∂/∂μ, both transport terms reduce to the flux
forms quoted in Eq. (3):

$$
\frac{\partial B}{\partial t}
= -\frac{1}{R_\odot}\frac{\partial}{\partial\mu}
  \left[u\,B\sqrt{1-\mu^2}\right]
+ \frac{\eta}{R_\odot^2}\frac{\partial}{\partial\mu}
  \left[(1-\mu^2)\frac{\partial B}{\partial\mu}\right]
- \frac{B}{\tau} + s .
$$

Both fluxes vanish identically at μ = ±1 because of the factors
√(1−μ²) and (1−μ²), so the total signed flux ∫B dμ (the monopole) is
conserved exactly by the continuous equation. **The transformation in the
draft is correct.**

### 2.2 The numerical scheme and monopole removal

The finite-volume upwind advection and the implicit diffusion matrix are both
written in flux form with zero boundary fluxes, and the row sums of the
diffusion operator telescope to zero. Consequently the discrete scheme
conserves ∫B dμ *exactly* (to round-off), and since the source has its mean
removed, the monopole-removal step of Eq. (11) only cleans floating-point
drift. This is worth one sentence in the manuscript: the step is a
safeguard, not a physical necessity of the scheme.

### 2.3 The axial-dipole diagnostic

For an axisymmetric radial field, the dipole coefficient is

$$
g_{10} = \frac{3}{4\pi}\int B_r \cos\theta \, d\Omega
       = \frac{3}{2}\int_{-1}^{1} B\,\mu \, d\mu ,
$$

which is precisely Eq. (37). The diagnostic is therefore not merely "a proxy
with convenient normalisation" — it *is* the axial dipole moment, and the
manuscript can state this.

### 2.4 Fixed-point algebra

Eqs. (49)–(55) are algebraically correct:
T\* = D₀T\*Q(T\*) gives D₀Q(T\*) = 1, and with 𝓜(T) = D₀TQ(T),

$$
\mathcal{M}'(T_*) = D_0 Q(T_*) + D_0 T_* Q'(T_*)
                  = 1 + T_*\frac{Q'(T_*)}{Q(T_*)} .
$$

Section 3.7 strengthens this with existence, uniqueness, and closed-form
stability results.

---

## 3. Corrections and enhancements

### 3.1 The linear coefficient *a* is now defined (adjoint formulation)

The draft uses P(T) = aTQ(T) without ever defining *a*. Because the
transport operator 𝓛 is linear for fixed (u, η, τ), the final dipole is a
linear functional of the source. Define the **dipole-yield kernel** K(μ,t) as
the solution of the adjoint problem, integrated backwards from the end of the
cycle t_f:

$$
-\frac{\partial K}{\partial t} = \mathcal{L}^\dagger K
= \frac{u\sqrt{1-\mu^2}}{R_\odot}\frac{\partial K}{\partial\mu}
+ \frac{\eta}{R_\odot^2}\frac{\partial}{\partial\mu}
  \left[(1-\mu^2)\frac{\partial K}{\partial\mu}\right]
- \frac{K}{\tau},
\qquad
K(\mu, t_f) = \tfrac{3}{2}\mu .
$$

(The diffusion and decay terms are self-adjoint; the advection term acquires
the non-conservative adjoint form because the boundary fluxes vanish.) Then,
for zero initial field,

$$
D(t_f) = \int_0^{t_f}\!\!\int_{-1}^{1} K(\mu,t)\, s(\mu,t)\, d\mu\, dt .
$$

For the efficiency-mode factors the source is
s = A_s T Q_X(T) ŝ(μ,t), so the reduced relation follows *exactly*:

$$
P(T) = a\,T\,Q_X(T),
\qquad
a \equiv A_s \int_0^{t_f}\!\!\int_{-1}^{1} K\,\hat{s}\; d\mu\, dt .
$$

Operationally, *a* is the slope of the linear run: in code units
a = D_lin(T)/T = 7.899 × 10⁻² for the reference parameters, constant across
the entire amplitude scan (this is the constant column R_NL of the linear
case in `nonlinear_response_results_v2.csv`).

The kernel viewpoint also explains *why* latitude quenching is Gaussian: the
long-time limit of K as a function of emergence latitude is the
"dipole yield" or dynamo-effectivity function, which is close to a Gaussian
exp(−λ²/2λ_R²) with a width λ_R set by the transport parameters
(equatorial diffusive leakage against poleward advection; see Jiang 2020;
Petrovay, Nagy and Yeates 2020; Talafha et al. 2022). In this controlled
experiment λ_R is *prescribed* (20°) rather than derived; the manuscript
should say so and note that in a physical calibration λ_R is a function of
(u₀, η, τ).

### 3.2 Tilt quenching: state the linearisation

In the code, tilt quenching reduces the polarity separation,
Δλ(T) = Δλ₀ f_TQ(T). By antisymmetry of the ring pair, the yield is an odd
function of the separation,

$$
g(\Delta\lambda) = c_1 \Delta\lambda + c_3 \Delta\lambda^3 + \dots ,
$$

(at emergence, for narrow rings at latitude λ_c, the pair dipole is
∝ cos 2λ_c · sin Δλ), so

$$
Q_{\rm TQ}(T) \equiv \frac{g(\Delta\lambda_0 f_{\rm TQ})}{g(\Delta\lambda_0)}
             = f_{\rm TQ}(T)\,\bigl[1 + \mathcal{O}(\Delta\lambda_0^2)\bigr].
$$

The size of the neglected cubic term is set by (Δλ₀/ℓ)², where ℓ is the
smallest latitudinal scale over which the yield varies (the ring width σ_λ,
the yield curvature, and the equatorial antisymmetry scale all contribute);
with Δλ₀ = 5° this is a few per cent, and the measured deviation is ≤ 2 %
over the whole scan (Section 5). The identification Q_TQ = f_TQ in Eq. (24)
is therefore justified — but it is a controlled approximation, not an
identity, and the tilt case is the one place in the *original* reference run
where the SFT-versus-theory agreement (Fig. 3 of the draft) is a non-trivial
test rather than a factor injected on both sides. The manuscript should make
this distinction explicitly.

### 3.3 Latitude quenching: belt-consistent reference latitude (main correction)

**The problem.** The draft's factor, Eq. (25),

$$
Q_{\rm LQ}(T) = \exp\!\left[-\frac{(\lambda_0 + b_{\rm LQ}T^2)^2 - \lambda_0^2}
{2\lambda_R^2}\right],
\qquad \lambda_0 = 15^\circ,
$$

is referenced to the *static* base latitude, while the model's own activity
belt, Eq. (14), drifts from λ₀(0) = 27° down to 15° across the cycle, and the
"shift" implementation applies the poleward displacement b_LQ T² to the
*drifting* belt. The efficiency factor and the shift mode therefore encode
different physics: the quenching penalty of emerging (λ₀ + δ) instead of λ₀
depends strongly on λ₀ — the exponent 2λ₀δ + δ² grows linearly with λ₀.
Referencing it to the cycle-*end* latitude (15°) rather than the
activity-weighted latitude (≈ 22°) understates the linear part of the penalty
by roughly a third.

**The correction.** Apply the instantaneous factor inside the source,

$$
Q_{\rm LQ}(t, T) = \exp\!\left[-\frac{2\lambda_0(t)\,\delta + \delta^2}
{2\lambda_R^2}\right],
\qquad \delta \equiv b_{\rm LQ}T^2 ,
$$

and derive the reduced (time-independent) factor as the envelope-weighted
average. Because λ₀(t) = λ_b + λ_d(1 − t/P) is linear in t, the exponent is
linear in t, and its average against the Gaussian envelope
s₁(t) = exp[−(t−t_max)²/2σ_t²] is a standard Gaussian integral with a
**closed form**:

$$
\bar{Q}_{\rm LQ}(T)
= \exp\!\left[-\frac{2\bar\lambda_0\,\delta + (1-\epsilon)\,\delta^2}
{2\lambda_R^2}\right],
$$

with

$$
\bar\lambda_0 = \lambda_b + \lambda_d\left(1 - \frac{t_{\max}}{P}\right)
\quad\text{(the belt latitude at cycle maximum)},
\qquad
\epsilon = \left(\frac{\lambda_d\,\sigma_t}{\lambda_R\,P}\right)^{\!2} .
$$

For the reference parameters λ̄₀ = 22.09° and ε = 0.0144. The closed form was
verified numerically against the exact weighted average (agreement to machine
precision on the infinite domain; truncating the envelope to the simulated
window t ∈ [0, 11] yr changes the average by at most ≈ 0.5 % at T = 2.5).

**Why this is also a better test.** In the corrected scheme the SFT run
applies Q_LQ(t, T) *inside* the time integral, while the reduced theory uses
its envelope average. The two are no longer identical by construction: the
residual measures the quality of the closure (envelope truncation and the
neglected covariance between the instantaneous factor and the time-dependent
dipole yield). The consistency validation of the latitude case is thereby
promoted from trivial to meaningful. Measured deviations are reported in
`figures_nonlinear_response_v2/theory_agreement_v2.csv` (see Section 5).

The old behaviour remains available (`lq_reference = "static"`), and the
static formula is recovered exactly from the new one by λ̄₀ → λ₀ and ε → 0.

**Units.** The manuscript must state that b_LQ carries units of degrees (per
unit of the dimensionless T²), since it adds to a latitude — Eq. (57) lists
it as a bare number.

### 3.4 The inflow-like factor: state the closure assumptions

Physical inflows act on the *transport* operator, not on the source, so a
source-side factor Q_I(T) requires two assumptions worth stating:

1. **Commutation to leading order.** For weak inflows the perturbation of the
   final dipole is linear in the inflow amplitude, which scales as T²; the
   leading-order effect is therefore a multiplicative reduction
   1 − b_I T² + 𝒪(T⁴).
2. **Padé resummation.** The algebraic form 1/(1 + b_I T²) is the [0/1] Padé
   approximant of that series. It agrees with the perturbative result at
   small T while remaining positive and monotonically decreasing for all T —
   the minimal well-behaved extension.

The same argument justifies the algebraic form of Q_TQ. The manuscript
currently gives the forms without motivation; two sentences to this effect
close the logical gap. The explicit converging-inflow mode remains in the
code for exploratory runs, but note the caveat in Section 3.6.

### 3.5 The product closure

Eq. (32), Q = Q_TQ Q_LQ Q_I, silently assumes the three mechanisms act
independently. In the controlled setup this holds to leading order because
they act on *disjoint properties of the source*: Q_TQ on the polarity
separation (geometry), Q_LQ and Q_I on the amplitude. Interaction terms are
second order (e.g. the yield of a separation change itself depends weakly on
emergence latitude). The assumption should be stated as such; in a fully
physical model (shift mode plus explicit inflows) it would need re-testing.

### 3.6 Domain of validity: the reduced theory cannot change the dipole's sign

All Q factors are strictly positive, so P(T) = aTQ(T) has a fixed sign. The
theory can represent an over-quenched *slope* — ℛ(T) = dP̂/dT < 0, which is
the definition used in Eqs. (46)–(48) and is fully consistent — but it can
never represent a *reversal* of the final dipole. The code history is
instructive: the physical-inflow mode with u_inflow = 5 m s⁻¹ produced exactly
such reversals, which is why the reference run uses 2 m s⁻¹ in that mode. The
manuscript should state the domain of validity explicitly:

> The reduced form P = aTQ with Q > 0 is applicable while the nonlinear
> feedbacks reduce the magnitude of the dipole without reversing its sign.
> Sign-reversing regimes (observed for strong explicit inflows) are outside
> the reduced theory's domain and would require a signed closure.

### 3.7 Stability of the reduced cycle map: exact results

Write the map as T_{n+1} = 𝓜(T_n) = D₀T_nQ(T_n), where D₀ absorbs both the
dynamo gain and the linear coefficient *a* (state this: if the next cycle's
amplitude is k times the preceding dipole, then D₀ = k·a). Define the
**quenching stiffness**

$$
q(T) \equiv -\frac{d\ln Q}{d\ln T} \;\ge 0 ,
$$

so that the multiplier at the fixed point is simply

$$
\mathcal{M}'(T_*) = 1 - q(T_*) .
$$

The following statements are exact for the closures used here:

**(i) Existence and uniqueness.** Every factor of ln Q(T) is strictly
decreasing in T with Q(0) = 1 and Q(∞) = 0, so ln[D₀Q(T)] is strictly
decreasing. A nonzero fixed point exists **iff D₀ > 1**, and it is then
unique. (Consequently the "no fixed point" class of the stability map cannot
occur anywhere in a scan run at D₀ = 3 — the class is retained only for
fixed points beyond the scanned amplitude range or subcritical gains.)

**(ii) General stability criterion.** Since q > 0 whenever any quenching is
active, 𝓜′(T\*) < 1 always: a *tangent* (saddle-node) instability is
impossible. The only route to instability is **period doubling**,
𝓜′(T\*) ≤ −1, i.e.

$$
\boxed{\;q(T_*) \ge 2\;}
$$

**(iii) A single algebraic quenching never destabilises.** For
Q = 1/(1 + bT²) alone, the fixed point is T\* = √((D₀−1)/b) and

$$
\mathcal{M}'(T_*) = \frac{2 - D_0}{D_0} \in (-1, 1)
\quad\text{for all } D_0 > 1 .
$$

The fixed point is unconditionally stable, for *any* quenching strength b and
any gain. This explains at one stroke why simple tilt-quenched
Babcock–Leighton models are so robustly steady: the map cannot period-double.

**(iv) Stacked algebraic quenchings.** For Q = Π_i 1/(1 + b_iT²),
q(T\*) = Σ_i 2s_i with saturation fractions s_i = b_iT\*²/(1 + b_iT\*²) < 1.
Instability requires Σ s_i > 1, hence **at least two mechanisms**, each
already substantially saturated. For two equal strengths, the condition is
independent of b and reads simply D₀ > 4.

**(v) Gaussian latitude quenching destabilises easily.** With the
belt-consistent factor of Section 3.3 (δ\* = b_LQ T\*²),

$$
q(T_*) = \frac{2\delta_*\left[\bar\lambda_0 + (1-\epsilon)\delta_*\right]}
{\lambda_R^2}
= 2\ln D_0 + \frac{(1-\epsilon)\,\delta_*^2}{\lambda_R^2},
$$

where the second equality uses the fixed-point condition. Hence latitude
quenching alone is **always period-doubling unstable when D₀ ≥ e ≈ 2.72**,
and for D₀ < e it destabilises once δ\*² > 2λ_R²(1 − ln D₀)/(1−ε). The
super-exponential tail of the Gaussian factor is what allows q to grow
without bound — algebraic factors saturate at q → 2 per factor, the Gaussian
factor does not. This is the analytic content behind the structure of the
stability map (Figs. 9–11): the unstable region is entered by increasing
b_LQ far more readily than by increasing b_TQ.

**(vi) Numerical implementation.** T\* is now found by bisection on the
strictly monotonic ln[D₀Q(T)] and 𝓜′(T\*) is evaluated from the analytic
derivative d ln Q/dT, replacing the previous finite-difference estimate at
the nearest grid point. The analytic period-doubling boundary 𝓜′ = −1 is
overlaid on the classification map.

One caveat for the text: |𝓜′| < 1 is *local* stability of the period-1 cycle.
Beyond the boundary the map generically enters a period-doubling cascade
(period-2 amplitude alternation, then chaotic modulation), so the label
"unstable/modulated" is apt, but the boundary itself marks the onset of
period-2 behaviour, not of unbounded growth.

### 3.8 The response function determines map stability (new identity)

Sections 1.13 and 1.14 of the draft present ℛ(T) and the map stability as
separate diagnostics. They are in fact linked exactly. With T_ref = 1,

$$
\hat{\mathcal{R}}(T) = \frac{d\hat P}{dT}
= \frac{Q(T)}{Q(1)}\bigl[1 - q(T)\bigr]
\qquad\Longrightarrow\qquad
\mathcal{M}'(T_*) = \frac{Q(1)}{Q(T_*)}\,\hat{\mathcal{R}}(T_*) .
$$

Consequences worth stating in the manuscript:

* sign(𝓜′(T\*)) = sign(ℛ(T\*)): the fixed point sits in the over-quenched
  regime exactly when the approach to it is oscillatory;
* ℛ(T\*) = 0 (saturated response) corresponds to 𝓜′ = 0, the super-stable
  fixed point;
* the period-doubling threshold 𝓜′ = −1 translates into a measurable
  condition on the single-cycle response curve:
  ℛ̂(T\*) ≤ −Q(T\*)/Q(1).

This gives the single-cycle numerical experiment a direct predictive use for
multi-cycle behaviour, which strengthens the paper's central argument.

---

## 4. Code corrections (implemented in this revision)

1. **NumPy 2.0 compatibility.** `np.trapz` was removed in NumPy 2.0; the code
   now uses `np.trapezoid` with a fallback, restoring the ability to run at
   all on current environments.
2. **Results CSV.** Both the header and the data rows of
   `nonlinear_response_results_v2.csv` were written without line breaks,
   producing a single-line, unusable file. Fixed.
3. **Face velocities.** The advective flux now samples u(λ) analytically at
   the cell faces instead of linearly interpolating cell-centre values, so
   the discrete flux is an exact sample of the flux form in Eq. (3).
4. **Return annotation** of `run_sft_case` corrected (five return values).
5. **Belt-consistent latitude quenching** (Section 3.3) implemented on both
   the source side (instantaneous factor) and the theory side (closed-form
   envelope average), switchable via `lq_reference` ("belt", the default,
   or "static" to reproduce the original v2 runs).
6. **Analytic stability map** (Section 3.7(vi)); Figure 7 now uses the same
   illustrative gain D₀ = 3 as the stability map for internal consistency
   (the draft used 1.8 in one place and 3.0 in the other).
7. **Quantitative agreement report** written to `theory_agreement_v2.csv`
   (Section 5).
8. **Diagnostics:** the setup check now also prints the advective Courant
   number u₀Δt/(R⊙Δμ) ≈ 0.13, documenting that the explicit upwind advection
   step is comfortably stable at Δt = 1 day.

## 5. Numerical validation of the enhanced closure

Maximum deviation between the normalised SFT response P̂(T) and the reduced
prediction TQ(T)/Q(1) over the scan T ∈ [0.2, 2.5]
(from `theory_agreement_v2.csv`, regenerated with this revision):

| Case | max abs. deviation | max rel. deviation | Nature of the test |
|------|-------------------:|-------------------:|--------------------|
| linear | < 10⁻¹⁴ | < 10⁻¹⁴ | exact by linearity |
| tilt | 0.020 | 1.9 % | non-trivial: geometric linearisation (Section 3.2) |
| latitude | 0.051 | 11.6 % (at T = 2.5) | non-trivial in belt mode: envelope-average closure (Section 3.3) |
| inflow | < 10⁻¹⁴ | < 10⁻¹⁴ | trivial: factor injected on both sides |
| combined | 0.013 | 10.0 % (at T = 2.5) | product closure plus the above |

Interpretation:

* The **linear and inflow** cases agree to round-off, as they must: by
  linearity of the transport, a time-independent source factor propagates
  multiplicatively into the final dipole. Their agreement validates the code,
  not the theory — the manuscript should say this plainly.
* The **tilt** deviation is smooth and bounded by ≈ 2 %. Its magnitude is set
  by (Δλ₀/ℓ)², where ℓ is the smallest latitudinal scale of the yield
  function (here the ring width and the yield curvature, tens of degrees),
  consistent with the linearisation argument of Section 3.2.
* The **latitude** deviation is below 2 % for T ≤ 1.2 and below 3.5 % for
  T ≤ 1.6 (the observed range of solar cycle variability), rising to
  5 % absolute (12 % relative) only at T = 2.5, deep in the over-quenched
  tail where the response itself is small. The deviation is systematically
  *positive* (SFT above theory): the true time weighting favours late-cycle
  emergence — where the belt sits at lower latitude and the instantaneous
  penalty is weaker — relative to the plain envelope weighting, a positive
  covariance the second-cumulant closure neglects.
* A systematic refinement exists within the same closed form: replace the
  envelope moments (t_max, σ_t) by the *yield-weighted* mean and standard
  deviation of the emergence-time distribution, both measurable once from
  the linear reference run. The Gaussian-average formula for λ̄₀ and ε is
  unchanged; only the two moments are updated.
* The **combined** case is *more* accurate than the latitude case in absolute
  terms (≤ 0.013) because the tilt and latitude residuals have opposite sign
  and partially cancel; its relative deviation at T = 2.5 reflects the near
  total suppression of the response there.

## 6. Suggested wording changes for the manuscript

1. **After Eq. (1):** define *a* via the adjoint kernel (Section 3.1), and
   state that operationally a = P(T)/T of the linear case.
2. **Replace Eqs. (25)–(26)** by the belt-consistent factor and its
   closed-form envelope average (Section 3.3), quoting λ̄₀ = 22.09° and
   ε = 0.0144 for the reference parameters, and give b_LQ its units
   (degrees).
3. **After Eq. (24) and Eq. (28):** add the Padé/linearisation justification
   for the algebraic forms (Sections 3.2 and 3.4).
4. **After Eq. (32):** state the independence assumption of the product
   closure (Section 3.5).
5. **Section 1.13:** add the identity linking ℛ to 𝓜′ (Section 3.8).
6. **Section 1.14:** add the existence/uniqueness statement, the
   quenching-stiffness criterion q(T\*) ≥ 2, and the analytic results (iii)–(v)
   of Section 3.7; note that |𝓜′| < 1 is local period-1 stability.
7. **Section 2 (or the conclusion):** state the domain of validity
   (no sign reversal; Section 3.6).
8. **Figures 1 and 2 of the draft are identical** in the scanned range
   because no case reverses sign; consider keeping only the signed plot and
   citing |D| in the text.
9. **Section 1.3:** one sentence noting that the conservative scheme makes
   the monopole removal a round-off safeguard only (Section 2.2).

## 7. Context in the literature

The corrected closure aligns the experiment with the established picture:
tilt quenching (Dasi-Espuig et al. 2010; Jiao, Jiang and Wang 2021) and
latitude quenching (Jiang 2020) as amplitude-limiting nonlinearities in
Babcock–Leighton models; the Gaussian dependence of the dipole yield on
emergence latitude and its "dynamo effectivity range" λ_R (Petrovay, Nagy
and Yeates 2020; Talafha et al. 2022, where the relative roles of latitude
and tilt quenching as a function of λ_R are quantified); converging inflows
as a transport feedback (Cameron and Schüssler 2012; Martin-Belda and
Cameron 2017); and amplitude modulation through iterated cycle maps,
including period doubling, in reduced dynamo models (Durney 2000;
Charbonneau 2020, and references therein).
