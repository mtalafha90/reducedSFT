# Transport Constants Across the SFT Parameter Space

**Companion analysis:** `parameter_space_analysis.py`; figures and tables
(`transport_constants.csv`, `parameter_space_results.csv`) in
`figures_parameter_space/`.

The adjoint-kernel framework reduces the transport model to a handful of
derived constants: the linear dipole yield a, the pair-yield effectivity
range λ_R, the sign-change latitude λ_rev (domain boundary of positive
multiplicative closures), and the cycle memory r. Each costs about one
second per parameter set, so the whole (η, u₀) plane can be mapped — the
reduced-theory counterpart of full SFT parameter-space optimisations
(Petrovay and Talafha 2019; Petrovay, Nagy and Yeates 2020; Talafha et
al. 2022; Alhosani et al. 2026). Scan: η ∈ [100, 1000] km² s⁻¹,
u₀ ∈ [5, 25] m s⁻¹, with the reference source geometry; maps in Figure P1.

## 1. A closed-form scaling law for the effectivity range

The measured λ_R (Gaussian width of the pair dipole yield versus emergence
latitude, from one adjoint solve per parameter set) collapses across the
entire plane onto the **quadrature law** (Figure P2):

$$
\lambda_R^2 \;=\; C^2\,\frac{\eta}{2u_0 R_\odot} \;+\; w_0^2,
\qquad C = 1.008, \quad w_0 = 4.35^\circ,
$$

with a relative scatter of only **2.5 %** (the proportional law alone has
16 % scatter and coefficient 1.09). The interpretation is clean:

* the first term is the **advection–diffusion balance width** of the
  equatorial boundary layer, with coefficient **one** — i.e.
  λ_R,transport = √(η/2u₀R⊙) is not merely an order-of-magnitude estimate
  but the actual transport scaling;
* the second term is the **finite ring width** of the source (σ_λ = 5°)
  entering in quadrature: the measured pair yield is the transport
  effectivity profile convolved with the emergence profile, so narrow-λ_R
  models saturate at the source width.

Over the scanned plane λ_R runs from 5.8° (η = 100, u₀ = 25) to 22.7°
(η = 1000, u₀ = 5). For observationally favoured mid-range parameters the
transport term alone gives ≈ 7–12°, consistent with the ≈ 10° values in
the literature.

## 2. The closure domain boundary is a fixed multiple of λ_R

The sign-change latitude of the pair yield tracks the effectivity range
(Figure P3):

$$
\lambda_{\rm rev} \;\approx\; (2.33 \pm 0.25)\,\lambda_R .
$$

So the domain of validity of any positive multiplicative latitude-quenching
closure — established in `physical_validation.md` — can be written directly
in transport parameters: the (shifted, emergence-weighted) belt must stay
equatorward of ≈ 2.3 λ_R. Beyond it, high-latitude emergence contributes
with reversed sign and the kernel (or a signed closure) is required.

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
plane (Figure P1, lower left).

## 4. The decay time enters analytically

Because the decay term −B/τ is proportional to the identity, it commutes
with the transport operator and factors out of every kernel quantity
**exactly**:

$$
K(\tau) = e^{-(t_f - t)/\tau}\,K(\infty), \qquad
a(\tau) = \sum_k \omega_k\, e^{-(t_f - t_k)/\tau}, \qquad
r(\tau) = e^{-P/\tau}\, r(\infty),
$$

and the *shape* of the yield function in latitude — hence λ_R and λ_rev —
is **invariant under τ**. Verified numerically at τ = 5 yr (Figure P4):

| Quantity | Predicted | Measured | Agreement |
|---|---|---|---|
| a(5 yr) | 3.386 × 10⁻² | 3.383 × 10⁻² | 9 × 10⁻⁴ |
| r(5 yr) | 0.1091 | 0.1079 | 1 × 10⁻² |
| λ_R(5 yr) | = λ_R(∞) | 10.773° vs 10.776° | 3 × 10⁻⁴ |
| λ_rev(5 yr) | = λ_rev(∞) | 25.0° vs 25.0° | exact |

Consequences worth stating in the manuscript:

1. **No new scan is needed for finite τ**: the whole τ-dependence of the
   reduced theory is carried by two closed-form factors applied to the
   τ = ∞ constants.
2. **Decay controls the memory regime.** r(τ) = e^{−P/τ}r(∞) drops from
   ≈ 0.99 (τ = ∞) to 0.11 (τ = 5 yr): observationally calibrated decay
   times place the Sun in the *weak-memory* regime where the memoryless
   map is a fair approximation, whereas decay-free SFT models are in the
   *strong-memory* regime where it fails. The choice of τ is therefore not
   a detail: it selects the dynamical class of the cycle map.
3. **Decay does not change which latitudes matter** (λ_R, λ_rev
   invariant); it only rescales how much any emergence contributes by its
   remaining lifetime, and shifts the yield-weighted time moments of the
   closure towards later emergence.

## 5. Reduced-theory summary of a transport model

Any (u₀, η, τ) SFT model of this family is summarised, for the purposes of
the reduced theory, by four numbers measurable in ~1 s:

| Constant | Meaning | Reference value (u₀ = 12, η = 500, τ = ∞) |
|---|---|---|
| a | linear dipole yield | 7.90 × 10⁻² |
| λ_R | effectivity range | 10.8° (11-yr horizon; quadrature law above) |
| λ_rev ≈ 2.33 λ_R | positive-closure domain boundary | 25° |
| r | cycle memory | 0.986 (τ = ∞); e^{−P/τ}r(∞) otherwise |

Together with the quenching parameters (b_TQ, b_LQ, b_I) these close the
reduced model: response P(T) = aTQ(T), map T_{n+1} = T_n[D₀Q(T_n) − r],
validity T within the sign-preserving domain.
