# Transport Constants Across the SFT Parameter Space

**Companion analysis:** `parameter_space_analysis.py`; figures and tables
(`transport_constants.csv`, `parameter_space_results.csv`) in
`figures_parameter_space/`.

The adjoint-kernel framework reduces the transport model to a handful of
derived constants: the linear dipole yield a, the pair-yield effectivity
range λ_R, and the cycle memory r. Each costs about one second per parameter
set, so the whole (η, u₀) plane can be mapped — the reduced-theory
counterpart of full SFT parameter-space optimisations (Petrovay and Talafha
2019; Petrovay, Nagy and Yeates 2020; Talafha et al. 2022; Alhosani et
al. 2026). Scan: η ∈ [100, 1000] km² s⁻¹, u₀ ∈ [5, 25] m s⁻¹, with the
reference source geometry and flux-balanced polarity rings; maps in
Figure P1.

## 1. A closed-form scaling law for the effectivity range

The measured λ_R (Gaussian width of the pair dipole yield versus emergence
latitude, fitted over the emergence belt, 3°–27°, from one adjoint solve per
parameter set) collapses across the entire plane onto the **quadrature law**
(Figure P2):

$$
\lambda_R^2 \;=\; C^2\,\frac{\eta}{2u_0 R_\odot} \;+\; w_0^2,
\qquad C = 1.189, \quad w_0 = 3.47^\circ,
$$

with a relative scatter (standard deviation of the residuals) of **5.8 %**
and a maximum absolute residual of 24 %, reached in the extreme
advection-dominated corner. The proportional law alone has 17 % scatter and
coefficient 1.23. The interpretation is clean:

* the first term is the **advection–diffusion balance width** of the
  equatorial boundary layer, with a coefficient close to unity — i.e.
  λ_R,transport ≈ √(η/2u₀R⊙) is not merely an order-of-magnitude estimate
  but the actual transport scaling;
* the second term is the **finite ring width** of the source (σ_λ = 5°)
  entering in quadrature: the measured pair yield is the transport
  effectivity profile convolved with the emergence profile, so narrow-λ_R
  models saturate at the source width.

Over the scanned plane λ_R runs from 6.2° (η = 100, u₀ = 25) to 26.8°
(η = 1000, u₀ = 5). For observationally favoured mid-range parameters the
transport term alone gives ≈ 8–14°.

Figure P3 shows the relative residuals of the quadrature law against the
measured λ_R: they are unstructured over most of the plane and grow only
where λ_R approaches the ring width, which is exactly where the convolution
argument behind the quadrature form is at its weakest.

## 2. The pair yield stays positive

With flux-balanced polarity rings the pair yield is positive throughout the
emergence belt in **every one of the 63 parameter sets**, and the linear
yield a is positive everywhere. Eight of the 63 sets — all at low
diffusivity, η ≤ 200 km² s⁻¹ — show a sign change at high latitude, the
lowest at 32.5°, which is above the poleward edge of the belt (27°); the
deepest negative excursion is 0.09 % of the equatorward peak. No result in
this work depends on that far tail.

This is a direct consequence of imposing polarity-flux balance. With rings
of equal peak amplitude, all 63 sets show a sign change, the lowest at 16° —
inside the belt — and 21 of the 63 give a negative linear yield. See §5 of
`physical_validation.md`.

## 3. The cycle memory across the plane

For τ = ∞ the remnant dipole decays only by **cross-equatorial diffusive
annihilation** of the polar caps, which is exponentially slow in the
magnetic Reynolds number R_m = u₀R⊙/η:

* advection-dominated corner (R_m ≳ 15, includes the reference model):
  r_eig = 1.000000 to the precision computed — memory is total, and the
  memory-corrected map of `multicycle_validation.md` is mandatory;
* diffusion-dominated corner (η = 1000, u₀ = 5, R_m ≈ 3.5):
  r_eig = 0.762 — even there, a 24 % per-cycle loss leaves memory far from
  negligible.

The profile-based first-period value r₁ tracks r_eig closely across the
plane (Figure P1, centre panel).

## 4. The decay time enters analytically

Because the decay term −B/τ is proportional to the identity, it commutes
with the transport operator and factors out of every kernel quantity
**exactly**:

$$
K(\tau) = e^{-(t_f - t)/\tau}\,K(\infty), \qquad
a(\tau) = \sum_k \omega_k\, e^{-(t_f - t_k)/\tau}, \qquad
r(\tau) = e^{-P/\tau}\, r(\infty),
$$

and the *shape* of the yield function in latitude — hence λ_R — is
**invariant under τ**. Verified numerically at τ = 5 yr (Figure P4):

| Quantity | Predicted | Measured | Agreement |
|---|---|---|---|
| a(5 yr) | 5.9247 × 10⁻² | 5.9218 × 10⁻² | 4.9 × 10⁻⁴ |
| r(5 yr) | 0.10999 | 0.10925 | 6.7 × 10⁻³ |
| λ_R(5 yr) | = λ_R(∞) | 12.064° vs 12.066° | 2.4 × 10⁻⁴ |

Consequences worth stating in the manuscript:

1. **No new scan is needed for finite τ**: the whole τ-dependence of the
   reduced theory is carried by two closed-form factors applied to the
   τ = ∞ constants.
2. **Decay controls the memory regime.** r(τ) = e^{−P/τ}r(∞) drops from
   0.994 (τ = ∞) to 0.109 (τ = 5 yr): observationally calibrated decay
   times place the Sun in the *weak-memory* regime where the memoryless
   map is a fair approximation, whereas decay-free SFT models are in the
   *strong-memory* regime where it fails. The choice of τ is therefore not
   a detail: it selects the dynamical class of the cycle map.
3. **Decay does not change which latitudes matter** (λ_R invariant); it only
   rescales how much any emergence contributes according to its remaining
   lifetime, and shifts the yield-weighted time moments of the closure
   towards later emergence.

## 5. Reduced-theory summary of a transport model

Any (u₀, η, τ) SFT model of this family is summarised, for the purposes of
the reduced theory, by three numbers measurable in ~1 s:

| Constant | Meaning | Reference value (u₀ = 12, η = 500, τ = ∞) |
|---|---|---|
| a | linear dipole yield | 1.678 × 10⁻¹ |
| λ_R | effectivity range | 12.07° (11-yr horizon; quadrature law above) |
| r | cycle memory | 0.994 (τ = ∞); e^{−P/τ}r(∞) otherwise |

Together with the quenching parameters (b_TQ, b_LQ, b_I) these close the
reduced model: response P(T) = aTQ(T) and map T_{n+1} = T_n[D₀Q(T_n) − r].
