#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity of the reduced-theory results to polarity-flux balance.

The controlled source places the two polarities of each hemispheric pair as
Gaussians of equal PEAK amplitude in latitude. Because the surface measure
is dmu = cos(lambda) dlambda, equal peaks do not give equal integrated
flux: the poleward (following) ring carries less flux than the equatorward
(leading) ring by 2-4 per cent over the emergence belt. The global source
remains antisymmetric and carries no net monopole, but each hemispheric
pair is not individually flux balanced, whereas a real bipolar region is.

This script quantifies what that imbalance changes. The balanced source
normalises each ring by its own surface integral,

    Gtilde_i(mu) = G_i(mu) / int G_i(mu) dmu,

and then forms the difference of the two normalised rings, rescaled by the
mean of the original ring fluxes so that the total unsigned flux, and hence
the overall amplitude convention, is preserved. Only the RELATIVE balance
of the two polarities therefore differs between the two source forms.

Reported for both source forms:
    a               linear dipole yield
    tilt closure    max relative deviation of the normalised tilt response
    lambda_R^pair   kernel-derived pair-yield effectivity range
    lambda_rev      first sign-change latitude of the pair yield, and the
                    depth of the negative excursion relative to the peak
    T_rev           reversal amplitude of the physical latitude shift
    C, w0           scaling-law coefficients across the (eta, u0) plane,
                    with the number of parameter sets showing a sign change
                    or a negative linear yield

Outputs are written to ./figures_balance/.
"""

from __future__ import annotations

import os

import numpy as np

import nonlinear_response_sft_experiment as nre
import mathematical_framework_analysis as mfa
import physical_validation_analysis as pva


OUTDIR = "figures_balance"
SECONDS_PER_YEAR = 365.25 * 86400.0
T_VALUES = pva.T_VALUES


# ============================================================
# Ring-pair construction, unbalanced and balanced
# ============================================================

def ring_pair(lat_deg, lat_c, sep, width, dmu, balanced):
    """Monopole-free four-ring source shape.

    balanced=False reproduces the equal-peak construction used in the
    controlled experiments. balanced=True normalises every ring by its own
    surface integral, then restores the original overall flux scale, so
    that each hemispheric pair carries equal and opposite flux.
    """
    NF = np.exp(-0.5 * ((lat_deg - (lat_c + sep / 2.0)) / width) ** 2)
    NL = np.exp(-0.5 * ((lat_deg - (lat_c - sep / 2.0)) / width) ** 2)
    SF = np.exp(-0.5 * ((lat_deg - (-lat_c - sep / 2.0)) / width) ** 2)
    SL = np.exp(-0.5 * ((lat_deg - (-lat_c + sep / 2.0)) / width) ** 2)

    if balanced:
        phis = [r.sum() * dmu for r in (NF, NL, SF, SL)]
        scale = float(np.mean(phis))
        NF, NL, SF, SL = (scale * r / phi for r, phi in
                          zip((NF, NL, SF, SL), phis))

    s = (NF - NL) - (SF - SL)
    return s - s.mean()


def source_series(p, grid, T, balanced, mode="linear"):
    """Per-step source for one cycle.

    mode: "linear"          -- fixed separation, no quenching
          "tilt"            -- separation reduced by the tilt factor
          "latitude-shift"  -- belt displaced poleward by b_LQ T^2
    """
    t_yr = np.asarray(grid["t_yr"])
    lat_deg = np.asarray(grid["lat_deg"])
    dmu = float(grid["dmu"])
    out = np.empty((len(t_yr), p.nmu))
    for k, t in enumerate(t_yr):
        phase = np.clip(t / p.cycle_years, 0.0, 1.0)
        lat_c = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
        sep = p.polarity_sep_deg
        if mode == "tilt":
            sep = sep / (1.0 + p.b_TQ * T**2)
        elif mode == "latitude-shift":
            lat_c = lat_c + p.b_LQ * T**2
        env = float(nre.cycle_envelope(float(t), p))
        shape = ring_pair(lat_deg, float(lat_c), float(sep),
                          p.source_width_deg, dmu, balanced)
        out[k] = p.source_amp * T * env * shape / SECONDS_PER_YEAR
    return out


def dipole(p, grid, K, T, balanced, mode):
    """Exact final dipole from the adjoint kernel."""
    S = source_series(p, grid, T, balanced, mode)
    return float(float(grid["dt"]) * np.einsum("ki,ki->", K, S))


# ============================================================
# Diagnostics
# ============================================================

def pair_yield(p, grid, K_asym, w, balanced, fit_lat_deg=(3.0, 27.0)):
    """Pair-yield curve versus centre latitude.

    Returns the Gaussian effectivity range fitted over the emergence belt and
    the first sign-change latitude, or NaN if the yield stays positive. Both
    source forms use the same fixed fit window, so the two effectivity ranges
    are directly comparable.
    """
    lat_deg = np.asarray(grid["lat_deg"])
    dmu = float(grid["dmu"])
    w_proj = w - w.mean()
    centres = np.linspace(2.0, 60.0, 117)
    f = np.empty_like(centres)
    for i, lc in enumerate(centres):
        shape = ring_pair(lat_deg, float(lc), p.polarity_sep_deg,
                          p.source_width_deg, dmu, balanced)
        f[i] = (K_asym @ shape) / (w_proj @ shape)
    neg = np.where(f <= 0)[0]
    lam_rev = float(centres[neg[0]]) if len(neg) else np.nan
    sel = (centres >= fit_lat_deg[0]) & (centres <= fit_lat_deg[1]) & (f > 0)
    slope, _ = np.polyfit(centres[sel] ** 2, np.log(f[sel]), 1)
    lamR = float(np.sqrt(-1.0 / (2.0 * slope))) if slope < 0 else np.nan
    # depth of any negative excursion, relative to the equatorward peak
    depth = float(min(f.min(), 0.0) / f.max())
    return lamR, lam_rev, depth


def tilt_closure_error(p, grid, K, balanced):
    """Max relative deviation of the normalised tilt response from T Q_TQ/Q_TQ(1)."""
    D = np.array([dipole(p, grid, K, float(T), balanced, "tilt") for T in T_VALUES])
    i1 = int(np.argmin(np.abs(T_VALUES - 1.0)))
    Phat = D / D[i1]
    q = 1.0 / (1.0 + p.b_TQ * T_VALUES**2)
    Pth = T_VALUES * q / (1.0 * (1.0 / (1.0 + p.b_TQ)))
    return float(np.max(np.abs(Phat - Pth) / np.abs(Pth)))


def reversal_amplitude(p, grid, K, balanced):
    """Amplitude at which the physical latitude-shift response changes sign."""
    f = lambda T: dipole(p, grid, K, float(T), balanced, "latitude-shift")
    lo, hi = 0.5, 3.0
    if f(lo) <= 0:
        return np.nan
    if f(hi) > 0:
        return np.nan
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scaling_coefficients(balanced, eta_vals, u0_vals):
    """Quadrature scaling law and sign-change statistics across the plane."""
    from dataclasses import replace
    p_ref = nre.SFTParams()
    lamR, lamrev, depth, a_lin, est = [], [], [], [], []
    for u0 in u0_vals:
        for eta in eta_vals:
            p = replace(p_ref, u0_m_s=float(u0), eta_km2_s=float(eta))
            grid = nre.make_grid(p)
            A, M = mfa.build_operators(p, grid)
            n, dt = p.nmu, float(grid["dt"])
            Minv = np.linalg.inv(M)
            Pi = np.eye(n) - np.ones((n, n)) / n
            w = mfa.dipole_weight(p, grid)
            RT = Minv @ (Pi @ (np.eye(n) + dt * A).T)
            K0 = Minv @ (Pi @ (w - w.mean()))
            K_series = [K0]
            for _ in range(len(np.asarray(grid["t_yr"])) - 1):
                K0 = RT @ K0
                K_series.append(K0)
            K_fwd = np.array(K_series[::-1])
            lr, lv, dp = pair_yield(p, grid, K0, w, balanced)
            lamR.append(lr); lamrev.append(lv); depth.append(dp)
            a_lin.append(dipole(p, grid, K_fwd, 1.0, balanced, "linear"))
            est.append(np.rad2deg(np.sqrt(eta * 1e10 / (2 * u0 * 100 * p_ref.R_sun))))
    lamR = np.array(lamR); lamrev = np.array(lamrev)
    depth = np.array(depth); a_lin = np.array(a_lin); est = np.array(est)
    good = np.isfinite(lamR)
    A_ = np.column_stack([est[good] ** 2, np.ones(good.sum())])
    (C2, w02), *_ = np.linalg.lstsq(A_, lamR[good] ** 2, rcond=None)
    pred = np.sqrt(C2 * est[good] ** 2 + w02)
    res = (lamR[good] - pred) / pred
    return dict(C=float(np.sqrt(C2)), w0=float(np.sqrt(w02)),
                scatter=float(res.std()), maxres=float(np.abs(res).max()),
                n_signchange=int(np.count_nonzero(np.isfinite(lamrev))),
                lamrev_min=float(np.nanmin(lamrev)) if np.any(np.isfinite(lamrev)) else np.nan,
                depth_worst=float(depth.min()),
                n_negative_a=int(np.count_nonzero(a_lin < 0)))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    p = nre.SFTParams()
    grid = nre.make_grid(p)
    A, M = mfa.build_operators(p, grid)
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    K_asym = mfa.extended_kernel(p, grid, A, M, K[0], extra_years=40.0)

    res = {}
    for bal in (False, True):
        tag = "balanced" if bal else "unbalanced"
        a = dipole(p, grid, K, 1.0, bal, "linear")
        tilt = tilt_closure_error(p, grid, K, bal)
        lamR, lamrev, depth = pair_yield(p, grid, K_asym, w, bal)
        Trev = reversal_amplitude(p, grid, K, bal)
        res[tag] = dict(a=a, tilt=tilt, lamR=lamR, lamrev=lamrev,
                        depth=depth, Trev=Trev)
        print(f"{tag:>11}:  a={a:.6e}  tilt_err={100*tilt:.3f}%  "
              f"lamR={lamR:.3f}  lamrev={lamrev:.2f}  "
              f"depth={100*depth:.3f}%  T_rev={Trev:.4f}")

    print("\nScaling coefficients across the (eta,u0) plane ...")
    eta_vals = np.array([100., 200., 300., 400., 500., 600., 700., 800., 1000.])
    u0_vals = np.array([5., 8., 11., 14., 17., 20., 25.])
    n_sets = eta_vals.size * u0_vals.size
    for bal in (False, True):
        tag = "balanced" if bal else "unbalanced"
        s = scaling_coefficients(bal, eta_vals, u0_vals)
        res[tag].update(s)
        print(f"{tag:>11}:  C={s['C']:.4f}  w0={s['w0']:.3f}  "
              f"scatter={100*s['scatter']:.2f}%  max={100*s['maxres']:.2f}%")
        print(f"{'':>11}   sign changes in {s['n_signchange']}/{n_sets} sets "
              f"(lowest at {s['lamrev_min']:.1f} deg, worst depth "
              f"{100*s['depth_worst']:.3f}% of the peak); "
              f"negative a in {s['n_negative_a']}/{n_sets}")

    u, b = res["unbalanced"], res["balanced"]
    with open(os.path.join(OUTDIR, "balance_sensitivity.csv"), "w",
              encoding="utf-8") as f:
        f.write("quantity,unbalanced,balanced,relative_change\n")
        for k in ("a", "tilt", "lamR", "lamrev", "depth", "Trev", "C", "w0",
                  "scatter", "maxres", "n_signchange", "lamrev_min",
                  "depth_worst", "n_negative_a"):
            rel = (b[k] - u[k]) / u[k] if u[k] not in (0.0,) else np.nan
            f.write(f"{k},{u[k]:.8e},{b[k]:.8e},{rel:.6e}\n")
    print(f"\nWrote {OUTDIR}/balance_sensitivity.csv")
