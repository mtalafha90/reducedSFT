# Inflow–Memory Coupling and the Settled-Dipole Map

**Companion analysis:** `inflow_memory_coupling_analysis.py` (about 30
minutes on four cores; `--quick` runs every stage on small grids in about
three minutes as a smoke test). Figures and tables are in
`figures_inflow_memory/`. Regression tests: `tests/test_inflow_memory.py`.

## The question

The memory-corrected cycle map (`multicycle_validation.md`) uses a fixed
transport operator and one constant remnant coefficient r. With physical
inflows towards the activity belts, the transport itself depends on the
cycle amplitude T. A stronger cycle could then change not only the dipole it
builds, but also how the previous cycle's surviving flux evolves. The
question tested here is:

> Does amplitude-dependent transport of the surviving flux change the
> inferred strength of quenching, or the onset of cycle modulation?

## Summary

1. **No. The coupling is negligible.** The action of a cycle's inflows on
   the inherited field is 0.3–0.4 per cent of their action on the cycle's
   own flux. Removing it moves the onset of modulation by 0.08 per cent
   (2 m s⁻¹ inflows) and 0.13 per cent (5 m s⁻¹). This holds on three
   latitude grids, at half the time step, with finite decay, and for other
   choices of inflow timing and centre.
2. **The scalar map is limited by unsettled flux, not by transport.** At
   the end of a cycle, part of its new flux is still in transit and loses
   some of its dipole while it settles at the poles. The scalar map treats
   the whole end-of-cycle dipole as if it survived equally well. Its
   period-doubling gain is therefore too high: by 0.4–0.5 per cent for
   fixed transport, 1.2–1.6 per cent with 2 m s⁻¹ inflows and 2.2–2.7 per
   cent with 5 m s⁻¹, depending on how r is chosen. These are values on the
   reference 181-point grid; finer grids make the error with inflows larger
   (about 2 per cent at 2 m s⁻¹).
3. **One extra variable closes the map.** Carrying the *settled dipole*
   (the dipole the field will have after one further cycle of transport)
   next to the end-of-cycle dipole reproduces the period-doubling gain of
   the complete field to within 0.004 per cent, and follows multi-cycle
   SFT runs cycle by cycle.
4. **A sharper criterion.** Without decay, the complete field
   period-doubles where the settled dipole of a cycle stops growing with
   its amplitude. The weight that defines the settled dipole is the
   dipole-yield kernel at the start of a cycle, so no new machinery is
   needed.
5. **Scalar remnant subtraction**, ΔD_n = D_n − e^(−P/τ) D_{n−1}, is
   accurate at τ = 8 yr (0.5 per cent rms with 2 m s⁻¹ inflows) and less so
   without decay (1.3 per cent rms; inferred inflow strength 4 per cent too
   low).
6. **Inflow centre matters for the new flux, not for memory.** If latitude
   quenching shifts the belt and the inflows follow it, the inflow
   reduction of a T = 1 cycle rises from 13.4 to 17.1 per cent and the
   onset of modulation comes 2.4 per cent earlier.

The first result answers the question as posed. Results 2–4 were not
anticipated: they came from asking why the scalar map still misses the
complete field by a per cent or two once the transport coupling is ruled
out.

## 1. Set-up and the exact split

For transport prescribed by the cycle amplitude, the SFT is linear in B at
fixed T, so the end-of-cycle field splits exactly:

$$
B_{\rm end} = s\,b(T) + G(T)\,B_{\rm start},\qquad
D_{\rm end} = s\,P(T) + w^{\mathsf T} G(T)\,B_{\rm start}.
$$

Here b(T) is the field that the new source builds from a clean slate (with
the inflows of the same cycle acting on it), P(T) = wᵀb(T) is its dipole, s
is the source polarity, and G(T) is the one-cycle propagator, which now
depends on T. The scalar map replaces the second term by r·D_start with a
constant r.

**Model.** The reference transport (u₀ = 12 m s⁻¹, η = 500 km² s⁻¹,
τ = ∞, N_μ = 181, Δt = 1 day), flux-balanced bipolar rings, and
belt-referenced latitude quenching as a source factor. Three transport
settings are compared:

* **fixed transport**: inflow quenching as the source factor Q_I(T), as in
  the published multi-cycle experiment;
* **inflows 2 m s⁻¹** and **inflows 5 m s⁻¹**: the reference solver's
  explicit converging flow towards each activity belt, a Gaussian-weighted
  profile of 12° width whose speed parameter is 2 or 5 m s⁻¹ at T = 1 and
  scales as T² (the peak inflow speed is 0.61 times the parameter). As in
  the reference solver, the flow keeps this strength throughout the cycle.

**Checks.** A vectorised copy of the reference update reproduces the
reference solver to 1.8 × 10⁻¹⁴ in five cases (linear, inflows at 2 and
5 m s⁻¹, latitude quenching in both modes). The split above holds to
7 × 10⁻¹⁴ along a 60-cycle sequence. The reference solver itself is not
modified.

## 2. Remnant survival: transport effect versus structure effect

The survival of the inherited dipole over one cycle, r = wᵀG(T)B/wᵀB,
depends on two different amplitudes (Figure I1, `remnant_survival.csv`):

* the amplitude **T_n of the cycle whose inflows act** on the inherited
  field (the coupling asked about), and
* the amplitude **T_{n−1} of the cycles that left the field behind**, which
  sets its latitudinal structure.

Changing each from 0.5 to 1.5 while holding the other at 1:

| Transport | Δr from the current inflows (T_n) | Δr from the inherited structure (T_{n−1}) |
|---|---|---|
| Fixed transport | 0.0000 | −0.0007 |
| Inflows 2 m s⁻¹ | −0.0018 | −0.0126 |
| Inflows 5 m s⁻¹ | −0.0049 | −0.0762 |

The structure effect is 7 to 16 times larger than the transport effect.
Inflows reduce memory mainly through *what a cycle leaves behind*: a strong
cycle with strong inflows holds more of its flux near the activity belts at
the end of the cycle, and that flux keeps less of its dipole than flux
already in the polar caps. The direct action of the next cycle's inflows on
the inherited field is weak, because most of that field already lies
poleward of the inflow region.

## 3. Adjoint response budget

The adjoint kernel, computed backwards with the time-dependent,
inflow-modified operator, splits the response of the end-of-cycle dipole
exactly to first order (Figure I2). At T = 1, starting from the steady
alternating (Hale) state:

| Contribution to dD_end/dT | Inflows 2 m s⁻¹ | Inflows 5 m s⁻¹ |
|---|---|---|
| Source change | +0.0920 | +0.0752 |
| Transport of new flux | −0.0352 | −0.0698 |
| Transport of inherited flux | +0.00012 | +0.00028 |

The adjoint terms agree with finite differences of separate runs to
4 × 10⁻⁷ or better. The inherited-flux term is 0.3–0.4 per cent of the
new-flux transport term: inflows act on the dipole almost entirely through
the flux of the cycle that drives them. Its sign is positive because
weaker survival of the opposite-polarity inherited dipole helps the new
cycle to reverse it. The time–latitude densities (Figure I2, right) show
where each term acts. The new-flux term follows the drifting activity
belt. The inherited-flux term acts mainly in the first two years of the
cycle, in bands of alternating sign near 25°, 50° and 70° latitude that
largely cancel.

## 4. Stability of the complete field

### 4.1 The complete-field multiplier

With feedback T_{n+1} = k|D_n| and Hale alternation, the cycle-to-cycle
map acts on the whole field profile, not only on its dipole. At the steady
alternating state B\*, which solves (I + G)B\* = b, the perturbation map
of the complete field is

$$
J = -G(T^{\ast}) - k\,(c_{\rm old} - c_{\rm new})\,w^{\mathsf T},\qquad
c_{\rm old} = \partial_T\!\left[G(T)B^{\ast}\right],\quad
c_{\rm new} = \partial_T b(T).
$$

Its eigenvalue of largest modulus is the leading multiplier of the complete
field evolution; period doubling begins where it reaches −1. Dropping
c_old removes the amplitude dependence of inherited-flux transport, and
only that ("frozen").

### 4.2 The settled-dipole map

One cycle of transport brings any field into the slowest dipole mode of
the propagator: the second eigenvalue of the one-cycle propagator is
2.3 × 10⁻⁵ in modulus for fixed transport, and 1.1 × 10⁻⁵ with 2 m s⁻¹
inflows. A further cycle therefore changes the dipole of any field by the
same factor, λ₁ = 0.99999 for τ = ∞ (0.2527 for τ = 8 yr), whatever the
field was. The first cycle is different. At the
end of its own cycle, the flux of the newest cycle is still spread over low
and middle latitudes, and its dipole changes while it settles. Define

* P(T) = wᵀb(T), the end-of-cycle dipole of the new flux;
* P_s(T) = wᵀG b(T), its **settled dipole**, one cycle of transport later;
* r₁ = P_s/P, the first-cycle survival, and u = 1 − r₁, the unsettled
  fraction.

Writing d_n = |D_n| and e_n for the settled dipole of the whole field at
the end of cycle n, the exact recurrence of Section 1 closes on two
variables:

$$
d_n = P(T_n) - e_{n-1},\qquad
e_n = P_s(T_n) - \lambda_1\,e_{n-1},\qquad
T_{n+1} = k\,d_n .
$$

The first equation says that a cycle must cancel the settled dipole it
inherits. The second says what it hands on. Four consequences follow.

1. **The remnant coefficient is derived, not fitted.** At the fixed point
   D₀Q(T\*) = 1 + r\*, with

   $$ r^{\ast} = \frac{r_1}{1 + \lambda_1 - r_1}. $$

   For a linear cycle r₁ = 0.993962, the clean-slate value quoted for the
   scalar map, and this gives r\* = 0.988, the value measured in the steady
   alternating state. The two differ because the inherited field is an
   alternating sum of old cycles in which the newest, least settled
   contribution weighs most.
2. **The scalar map is the special case r₁ = λ₁.** If new flux survived its
   first cycle as well as settled flux does, e_n = r·d_n and the map
   reduces to T_{n+1} = T_n[D₀Q(T_n) − r].
3. **Stability.** The multipliers are the roots of

   $$ \mu^2 - (kP' - \lambda_1)\,\mu + k\,(P_s' - \lambda_1 P') = 0 . $$

   Period doubling (μ = −1) occurs where
   (1 − λ₁)(1 + kP′) + kP_s′ = 0. Without decay λ₁ → 1, and the condition
   becomes **P_s′(T\*) = 0**: the settled dipole of a cycle stops growing
   with its amplitude. In terms of the stiffness q = −dlnQ/dlnT, the
   threshold is q\* = 1 − T u′/(1 − u), slightly below 1, where the scalar
   map gives q\* = 2/(1 + r\*) = 1 + u, slightly above 1.
4. **No superstable point.** The two roots never vanish together: their
   product is k(P_s′ − λ₁P′) ≈ −k(uP′ + u′P), which is not zero while part
   of each cycle's flux is unsettled. The scalar map has a multiplier of
   exactly zero at q = 1/(1 + r); the complete field does not
   (Section 4.4).

With inflows, G depends on the amplitude of the cycle during which the
flux settles, so P_s and λ₁ acquire a weak dependence on T_{n+1}. The
script keeps both terms; they are the 0.1 per cent effect of Section 4.3.

**The settled dipole needs no new machinery.** For fixed transport,
wᵀG is the adjoint dipole-yield kernel at the start of a cycle (the
identity Gᵀw = (I + ΔtA)ᵀK⁰ holds to 10⁻¹³ and is pinned by a regression
test). The settled dipole is therefore the end-of-cycle field weighted by
that kernel instead of by (3/2)μ. The kernel weight is zero at the
equator and flat poleward of about 30°, so it counts flux by the
hemisphere it will end in, not by where it is now.

**Relation to the algebraic method.** The second equation of the map is
the recurrence of the algebraic method of Petrovay, Nagy and Yeates
(2020), which sums the *ultimate* dipole contributions of the active
regions of a cycle. Those authors note that the asymptotic contribution is
reached only "after a transitional period of 2–3 years", so that the
contribution of regions emerging in the last years of a cycle "is
therefore not accurately represented". The first equation of the map
carries exactly that in-transit part: the dipole at cycle minimum, which
sets the next cycle, is the inherited settled dipole plus the new flux at
its *end-of-cycle* weight.

### 4.3 Period-doubling gains

Gain D₀ = ka at which the leading multiplier reaches −1 (Figure I3,
`stability_scan.csv`). Percentages are relative to the complete field.

| Transport | Complete field | Inherited-flux transport frozen | Settled-dipole map | Scalar map, r\*(T) | Scalar map, r = 0.994 |
|---|---|---|---|---|---|
| Fixed transport | 3.3353 | 3.3353 (0.000 %) | 3.3353 (+0.000 %) | 3.3472 (+0.36 %) | 3.3522 (+0.51 %) |
| Inflows 2 m s⁻¹ | 3.1861 | 3.1836 (−0.078 %) | 3.1861 (+0.002 %) | 3.2240 (+1.19 %) | 3.2354 (+1.55 %) |
| Inflows 5 m s⁻¹ | 3.2150 | 3.2109 (−0.129 %) | 3.2152 (+0.004 %) | 3.2884 (+2.28 %) | 3.3030 (+2.73 %) |

"Scalar map, r\*(T)" uses the steady-state survival at each fixed point
and its dependence on the current inflows; it is the best that a map of
the dipole alone can do. "r = 0.994" is the clean-slate coefficient of the
published map. In Figure I3 the complete-field curve is broken where its
leading multiplier passes from one root of the quadratic in Section 4.2 to
the other.

Four results follow.

1. **The coupling asked about is negligible.** Freezing the amplitude
   dependence of inherited-flux transport moves the onset of modulation by
   0.08 per cent (2 m s⁻¹) and 0.13 per cent (5 m s⁻¹).
2. **The settled-dipole map reproduces the complete field.** Over the whole
   scan its leading multiplier differs from that of the complete field by
   at most 1.2 × 10⁻⁷ (fixed transport), 1.5 × 10⁻⁴ (2 m s⁻¹) and
   2.4 × 10⁻³ (5 m s⁻¹). The largest error is at the largest amplitudes,
   where the 5 m s⁻¹ inflows hold flux back strongly enough that one cycle
   no longer settles it completely: at T = 2 the second eigenvalue of the
   propagator rises to 0.036.
3. **The criterion holds.** The complete field reaches the threshold at
   T\* = 1.2891, 1.3136 and 1.0139 in the three settings. The settled
   dipole of the new flux peaks at T = 1.2891, 1.3126 and 1.0125. The
   small differences are the inflow terms of result 1.
4. **The scalar map errs in a known direction and by a known amount.** At
   the threshold the stiffness of the complete field is q\* = 0.9994,
   0.9881 and 0.9744; the scalar map requires q\* = 1.0063, 1.0115 and
   1.0155. The unsettled fraction there is u = 0.6, 1.2 and 1.6 per cent.
   The scalar map is reliable when u and its growth with amplitude are
   both small. Inflows raise both, because they hold flux near the belts.

**Nonlinear check.** Long multi-cycle runs (400 cycles at 16 gains
bracketing both thresholds; Figure I4) confirm the linear thresholds to
within one step of the gain scan (0.005 for fixed transport, 0.02 with
inflows). The complete SFT first sustains a period-two alternation at
D₀ = 3.339, 3.197 and 3.219 in the three settings; the settled-dipole map
does so at exactly the same gains, and the scalar map (with constant
r = r\*(1)) at 3.353, 3.234 and 3.301. Started from the same state, the
settled-dipole map follows the complete SFT cycle by cycle over all 400
cycles at 16, 12 and 16 of the 16 gains (largest difference in T_n:
1.1 × 10⁻⁴, 1.5 × 10⁻³ and 4.1 × 10⁻³). At the remaining four gains
(2 m s⁻¹) it reaches the same attractor, to 5 × 10⁻⁴, but not in step. The
scalar map follows the SFT cycle by cycle at none of the gains, and its
attractor differs from that of the SFT by up to 0.06, 0.12 and 0.15 in T.

### 4.4 The published multi-cycle configuration

The fixed-transport setting is the configuration of the memory-corrected
map in the manuscript (latitude and inflow quenching as source factors,
tilt omitted, τ = ∞). All values are in `published_configuration.csv`.

| Quantity | Scalar map (r = 0.994) | Complete field |
|---|---|---|
| Existence threshold, D₀ | 1.994 | 1.988 |
| Period-doubling gain, D₀ | 3.352 | 3.335 |
| Multiplier at D₀ = 2.05 / 2.50 / 3.20 | +0.890 / +0.117 / −0.823 | +0.879 / +0.146 / −0.845 |
| Relaxation time at D₀ = 2.05 / 2.50 / 3.20 [cycles] | 8.57 / 0.47 / 5.15 | 7.77 / 0.52 / 5.92 |
| Fastest relaxation | one step (multiplier 0 at D₀ = 2.578) | 0.40 cycles (smallest spectral radius 0.082, at D₀ = 2.564) |

The scalar-map column is recomputed here and reproduces the values quoted
in the manuscript's multi-cycle section. The complete field differs by 0.3
per cent in the existence threshold and 0.5 per cent in the
period-doubling gain, which supports the manuscript's wording that the SFT
attractor begins its transition "near the corrected analytic boundary".
The relaxation times differ by up to 15 per cent (5.92 rather than 5.15
cycles at D₀ = 3.20). The complete field also has no gain at which
perturbations vanish in one step: because part of each cycle's flux is
unsettled, its leading multiplier never falls below 0.082 in modulus.
(The table value, 0.0827, is the smallest over the sampled amplitudes; a
direct minimisation, in the manuscript repository's
`revisions/settled_dipole/` record, gives 0.0819 at D₀ = 2.564.)

## 5. Scalar remnant subtraction along a prescribed sequence

Talafha, Petrovay and Opitz (2025) estimate the net contribution of a
cycle as ΔD_n = D_{n+1} − e^(−11/τ) D_n, with τ = 8 yr or ∞. This form is
tested here on a 60-cycle sequence with prescribed random amplitudes
(log-normal, standard deviation 0.3 in ln T, limited to 0.4–2.0; the
first 10 cycles are discarded), with no quenching other than the explicit
inflows. The exact answer is known: it is the clean-slate dipole P(T_n).
The inflow strength is summarised by b in P = aT/(1 + bT²)
(Figure I5, `sequence_test.csv`, `sequence_cycles.csv`).

| τ | Inflows | rms error | Largest error | b, exact | b, from the subtraction |
|---|---|---|---|---|---|
| ∞ | none | 0.71 % | 1.5 % | 0 | −0.002 |
| ∞ | 2 m s⁻¹ | 1.28 % | 3.4 % | 0.1645 | 0.1577 (−4.1 %) |
| ∞ | 5 m s⁻¹ | 5.30 % | 19.4 % | 0.4635 | 0.4240 (−8.5 %) |
| 8 yr | none | 0.31 % | 0.6 % | 0 | −0.001 |
| 8 yr | 2 m s⁻¹ | 0.49 % | 1.2 % | 0.1451 | 0.1423 (−1.9 %) |
| 8 yr | 5 m s⁻¹ | 1.36 % | 4.4 % | 0.4077 | 0.3967 (−2.7 %) |

The subtraction overestimates a cycle's own contribution, because the
inherited dipole survives less well than e^(−P/τ) assumes. The error
grows with the amplitude of the *previous* cycle (Figure I5), as the
settled-dipole picture predicts: a strong cycle leaves more unsettled
flux. The factor e^(−P/τ) is the right one for settled flux
(λ₁ = 0.2527 at τ = 8 yr); it is the unsettled part of D_n that it
misrepresents.

Consequences for the inflow-induced reduction of the cycle contribution,
averaged over the sequence:

| τ | Inflows | Exact reduction | From the subtraction |
|---|---|---|---|
| ∞ | 2 m s⁻¹ | 19.6 % | 19.1 % |
| ∞ | 5 m s⁻¹ | 38.8 % | 37.0 % |
| 8 yr | 2 m s⁻¹ | 17.9 % | 17.7 % |
| 8 yr | 5 m s⁻¹ | 36.0 % | 35.5 % |

At τ = 8 yr the subtraction understates the reduction by 0.1–0.5
percentage points. This test uses the inflow profile of this repository,
not that of the 2025 paper, so it tests the method and not the paper's
numbers.

A second route to the quenching strength is the equilibrium amplitude,
through D₀Q(T\*) = 1 + r\*. A constant r then misreads the fall of r\*
with amplitude as part of Q. Between T\* = 1 and T\* = 1.6 this biases Q
by 0.03 per cent with fixed transport, 0.7 per cent with 2 m s⁻¹ inflows
and 5.4 per cent with 5 m s⁻¹ (`stability_scan.csv`).

## 6. Controls and refinement

Each row repeats the analysis for the 2 m s⁻¹ inflows with one change
(`controls.csv`). The last three columns give period-doubling gains
relative to the complete field.

| Variant | Inflow reduction of P(1) | Inherited ÷ new transport term | Δr, current inflows | Δr, inherited structure | Complete field, D₀ | Frozen | Settled-dipole | Scalar, r\*(T) |
|---|---|---|---|---|---|---|---|---|
| Reference | 12.9 % | −0.0034 | −0.0018 | −0.0126 | 3.1861 | −0.078 % | +0.002 % | +1.19 % |
| Envelope-timed inflow | 12.9 % | −0.0009 | −0.0005 | −0.0047 | 3.2237 | −0.023 % | 0.000 % | +0.61 % |
| Shifted belt, no inflow | 0 | — | 0.0000 | −0.0316 | 3.1483 | 0.000 % | 0.000 % | +1.55 % |
| Shifted belt, inflow at fixed centre | 13.4 % | −0.0057 | −0.0031 | −0.0842 | 3.1608 | −0.061 % | +0.001 % | +2.20 % |
| Shifted belt, inflow follows belt | 17.1 % | −0.0034 | −0.0026 | −0.0860 | 3.0840 | −0.044 % | +0.001 % | +1.66 % |
| Decay τ = 8 yr | 11.7 % | −0.0016 | −0.0005 | −0.0025 | 2.6539 | −0.057 % | 0.000 % | +0.60 % |
| Grid N_μ = 361 | 13.3 % | −0.0034 | −0.0019 | −0.0171 | 3.1702 | −0.080 % | +0.002 % | +1.56 % |
| Grid N_μ = 721 | 13.5 % | −0.0034 | −0.0019 | −0.0200 | 3.1602 | −0.078 % | +0.002 % | +1.78 % |
| Time step 0.5 day | 12.9 % | −0.0034 | −0.0018 | −0.0126 | 3.1861 | −0.078 % | +0.002 % | +1.19 % |

* **Inflow timing.** The reference solver keeps the inflows at full
  strength throughout the cycle. Inflows that follow the activity envelope
  (2.72 m s⁻¹ at maximum, calibrated to give the same P(1)) are weak early
  in the cycle, when the inherited low-latitude flux is present. The
  coupling falls almost four-fold and the scalar map's error halves, so the
  constant-in-time inflow is the conservative case.
* **Inflow centre.** When latitude quenching moves the belt poleward, the
  reference solver keeps the inflows at the unshifted latitude. Letting
  them follow the belt raises the inflow reduction of P(1) from 13.4 to
  17.1 per cent and moves the onset of modulation 2.4 per cent earlier
  (3.161 to 3.084), but changes the inherited-flux coupling little. The
  belt shift itself also changes the inherited structure (Δr = −0.032
  without any inflow), so the scalar map errs by 1.6 per cent even without
  inflows; inflows nearly triple that structure effect.
* **Decay.** With τ = 8 yr every memory effect is smaller, because the
  remnant survives with r ≈ 0.25 rather than 0.99; the scalar map errs by
  0.6 per cent.
* **Refinement.** Halving the time step changes nothing beyond the fifth
  digit. Refining the latitude grid leaves the transport coupling unchanged
  (a ratio of −0.0034 and a frozen shift of −0.08 per cent at all three
  grids) and keeps the settled-dipole map exact (+0.002 per cent). It is
  not converged in two other respects. The absolute period-doubling gain
  falls from 3.186 to 3.170 and 3.160; successive changes shrink by a
  factor of about 0.63, which points to a limit near 3.14, about 1.4 per
  cent below the N_μ = 181 value. The structure effect grows (Δr from
  −0.0126 to −0.0171 and −0.0200), and with it the scalar map's error
  (1.19, 1.56 and 1.78 per cent, tending to about 2.1 per cent). The
  181-point grid therefore understates the scalar map's error by nearly a
  factor of two; the conclusions do not change.

## 7. What this means for the manuscript

Nothing in the manuscript has been changed. The points below are for a
decision.

1. **No published claim needs correcting.** The manuscript says that the
   SFT attractor begins its transition "near the corrected analytic
   boundary"; the complete field gives 3.335 against 3.352 (0.5 per cent).
   The multipliers and relaxation times in the multi-cycle section are
   reduced-map values, and the manuscript already says so.
2. **Two statements hold for the scalar map only.** The sentence "When
   m\* = 0, the linear perturbation vanishes after one step" does not carry
   over to the complete field, whose leading multiplier never falls below
   0.082. Its relaxation times at D₀ = 2.05, 2.50 and 3.20 are 7.8, 0.52
   and 5.9 cycles, against the quoted 8.6, 0.47 and 5.2. A referee who asks
   how well the reduced map represents the full field would receive these
   numbers.
3. **An optional strengthening.** The manuscript notes that "the full
   remnant contribution depends on the spatial structure of the previous
   magnetic field" before approximating it by rD_{n−1}. The settled-dipole
   map replaces that approximation by an exact two-variable recurrence at
   the cost of one extra quantity, the start-of-cycle kernel weight, which
   the adjoint framework already provides. It derives r rather than
   measuring it (r\* = r₁/(1 + λ₁ − r₁)), reproduces the complete-field
   threshold, and gives the criterion P_s′(T\*) = 0. It would fit in one
   paragraph and one equation; whether to add it at this stage is a
   judgement call.
4. **Inflows.** For the inflow paragraph of the Discussion: the action of
   inflows on surviving flux is 0.3–0.4 per cent of their action on new
   flux, so a scalar memory treatment, as in the manuscript and in
   Talafha, Petrovay and Opitz (2025), is not undermined by
   amplitude-dependent transport. In this model, at τ = 8 yr, the
   subtraction ΔD_n = D_{n+1} − e^(−11/τ)D_n recovers the inflow-induced
   reduction of the cycle contribution to within 0.5 percentage points.
5. **Literature.** Tähtinen, Asikainen and Mursula (2026, A&A 708, L21)
   build compressed SFT-to-dipole propagators for fast dipole
   calculations. The manuscript does not cite it; it belongs next to the
   adjoint positioning. Teweldebirhan, Miesch and Gibson (2024) and
   Alhosani, Talafha and Al-Wardat (2026) are already cited.

## 8. Limits

* One transport model (u₀ = 12 m s⁻¹, η = 500 km² s⁻¹) with τ = ∞ and
  8 yr, axisymmetric ring sources, and no scatter in tilt or flux.
* The inflows are the reference solver's idealised profile, with speed
  scaling as T² and, by default, constant within a cycle. The 2025 paper
  uses a saturating (arctangent) scaling and a time-dependent band
  geometry, so the sequence test checks the subtraction method, not that
  paper's numbers.
* Absolute thresholds are not grid-converged (about 1.4 per cent high at
  N_μ = 181). The transport coupling and the settled-dipole closure are
  converged.
* The closure is exact here because one cycle of transport leaves any
  field in the slowest dipole mode. Transport that settles more slowly
  could need more variables; the second eigenvalue of the one-cycle
  propagator is the test. It is about 10⁻⁵ at the reference settings, but
  0.036 for the 5 m s⁻¹ inflows at T = 2, and it may be larger in
  diffusion-dominated models.
* A limited literature search found no earlier cycle map of this
  two-variable form (the papers checked include Petrovay, Nagy and Yeates
  2020, Wang, Jiang and Wang 2025, and Kumar, Karak and Vashishth 2021). A
  fuller search is needed before claiming priority.
