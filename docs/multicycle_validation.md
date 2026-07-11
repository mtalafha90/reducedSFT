# Multi-Cycle Validation: Memory in the Reduced Cycle Map

**Companion analysis:** `multicycle_feedback_analysis.py`; figures and the
results table (`multicycle_results.csv`) in `figures_multicycle/`.

The reduced cycle map T_{n+1} = D₀T_nQ(T_n) rests on one remaining
structural assumption: **memoryless cycle-to-cycle coupling** — each cycle
builds its dipole from a clean slate. This analysis tests that assumption by
running long sequences of consecutive SFT cycles with genuine dynamo
feedback (T_{n+1} = k|D_n|, source polarity alternating to reverse the
standing dipole), and comparing against the reduced map. The quenching is
the combined efficiency-mode latitude + inflow factor, evaluated
kernel-exactly, so **the only difference between the SFT and the maps is
memory**. The fast matrix stepper used for the long runs reproduces the
reference solver to 5 × 10⁻¹⁴.

## 1. The exact recurrence and the memory-corrected map

By linearity of the transport, the end-of-cycle dipole obeys **exactly**

$$
D_n = s_n\, a\, T_n Q(T_n) \;+\; w^{\mathsf T} G_P\, B_{n-1},
$$

where the first term is the new-flux contribution (adjoint kernel), s_n the
source polarity, G_P the one-period propagator, and B_{n−1} the field at
the end of the previous cycle. The memoryless map amounts to dropping the
second term. Collapsing it instead to a scalar,
wᵀG_P B_{n−1} ≈ r·D_{n−1}, gives the **memory-corrected map**, which on the
successful-reversal branch reads

$$
T_{n+1} = T_n\left[D_0 Q(T_n) - r\right],
$$

with three analytic consequences:

| Property | Memoryless | Memory-corrected |
|---|---|---|
| Fixed point | D₀Q(T\*) = 1 | D₀Q(T\*) = 1 + r |
| Existence | D₀ > 1 | D₀ > 1 + r |
| Period doubling | q(T\*) ≥ 2 | q(T\*) ≥ 2/(1 + r) |

## 2. The memory parameter r is not small

Measured three ways (Figure M1):

| Measurement | Value |
|---|---|
| Source-free dipole survival over one period | **r = 0.986** |
| Survival after several periods | → 1.0000 |
| Dominant eigenvalue of G_P | **1.0000** |

With τ = ∞, the advection–diffusion balance admits a genuine **steady
dipolar profile** (flux parked at the poles by the meridional flow, exactly
balancing diffusion), so remnant dipole *never* decays — the eigenvalue is
exactly 1, and the first-period value 0.986 only reflects the transient
adjustment of the end-of-cycle profile towards that steady mode. The
memoryless assumption is therefore not mildly violated but structurally
wrong in this regime: **almost half of the dynamo gain is spent reversing
the previous cycle's polar field** (existence requires D₀ > 1.99 rather
than D₀ > 1).

This is the map-level manifestation of the known flux-accumulation
behaviour of decay-free SFT models; with a finite decay term
(τ ~ 5–10 yr, as used in observationally calibrated models) r would drop
substantially, and the memoryless map would become correspondingly more
accurate. r is a single, cheaply measurable transport property (one
source-free period), and it belongs in the reduced theory as a second
parameter alongside a.

## 3. The memory-corrected map reproduces the SFT dynamics

With r = 0.986 inserted — no other change — the reduced map captures the
multi-cycle SFT quantitatively (Figures M2, M3):

* **One-step predictability.** Predicting T_{n+1} from the SFT's own
  (T_n, D_{n−1}): mean error **97 %** for the memoryless map versus
  **1.3 %** (stable gain) and **1.5 %** (modulated gain) for the
  memory-corrected map — a seventy-fold improvement from one scalar.
* **Fixed points.** At D₀ = 3.68 the SFT mean amplitude is 1.328 against
  the memory map's 1.331 (memoryless: 1.923).
* **Period-doubling onset.** The SFT attractor bifurcates to period-2
  exactly at the memory-corrected analytic boundary **D₀ = 3.35**
  (memoryless prediction: 2.80) and follows the memory map's attractor
  through the subsequent cascade (Figure M3).
* **Phase-locked modulation.** In the period-2 regime the memory map tracks
  the SFT *cycle by cycle*, including the phase of the even–odd
  alternation (Figure M2, middle panel). In the chaotic regime individual
  trajectories separate (as they must), but the attractor envelope and the
  occurrence of grand-minimum-like excursions (T ≈ 0 episodes lasting a
  few cycles, Figure M2, bottom panel) are alike.
* **Return map.** At modulated gains the SFT's (T_n, T_{n+1}) points fall
  on the memory-corrected curve T[D₀Q − r], visibly displaced from the
  memoryless curve D₀TQ (Figure M4).

## 4. Physical interpretation and implications

1. **Polarity reversal is a tax on the dynamo gain.** Each cycle must first
   cancel the standing polar field before building its own; with slow (or
   absent) polar-field decay this consumes a fixed fraction r of the
   produced dipole. Amplitude modulation, saturation, and the
   period-doubling route all shift accordingly.
2. **The reduced theory needs (a, r), not just a.** Both are single-number
   transport properties, measurable from one forward linear run and one
   source-free period (or one adjoint solve plus one eigenvalue).
3. **Even-odd cycle correlations.** The memory term makes consecutive
   cycles anticorrelated beyond what quenching alone produces — a
   testable, observationally relevant signature (cf. the even–odd
   Gnevyshev–Ohl pattern).
4. **Regime dependence.** r is a strong function of the decay time τ and
   of the transport parameters; for observationally calibrated SFT models
   with finite τ, r should be quoted alongside λ_R when the reduced map is
   used for cycle prediction.

## 5. Manuscript recommendations

1. State the memoryless assumption of Eq. (49) explicitly, and present the
   exact recurrence above as its parent; introduce r and the corrected
   fixed-point and stability conditions (Section 1 table).
2. Quote r for the reference transport (0.986; exactly 1 asymptotically for
   τ = ∞) and note the connection to the flux-accumulation problem of
   decay-free SFT models.
3. Replace the map-based stability discussion's thresholds by the
   memory-corrected ones when comparing with multi-cycle simulations:
   existence D₀ > 1 + r, period doubling q(T\*) ≥ 2/(1 + r).
