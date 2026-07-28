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
    lambda_rev      sign-change latitude of the pair yield
    T_rev           reversal amplitude of the physical latitude shift
    C, w0, ratio    scaling-law coefficients across the (eta, u0) plane

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

def pair_yield(p, grid, K_asym, w, balanced, fit_hi_frac=0.85):
    """Pair-yield curve versus centre latitude; returns lambda_R, lambda_rev."""
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
    hi = fit_hi_frac * lam_rev if np.isfinite(lam_rev) else 45.0
    sel = (centres >= 3.0) & (centres <= hi) & (f > 0)
    slope, _ = np.polyfit(centres[sel] ** 2, np.log(f[sel]), 1)
    lamR = float(np.sqrt(-1.0 / (2.0 * slope))) if slope < 0 else np.nan
    return lamR, lam_rev


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
    """Quadrature scaling law and sign-change ratio across the (eta, u0) plane."""
    from dataclasses import replace
    p_ref = nre.SFTParams()
    lamR, lamrev, est = [], [], []
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
            for _ in range(len(np.asarray(grid["t_yr"])) - 1):
                K0 = RT @ K0
            lr, lv = pair_yield(p, grid, K0, w, balanced)
            lamR.append(lr); lamrev.append(lv)
            est.append(np.rad2deg(np.sqrt(eta * 1e10 / (2 * u0 * 100 * p_ref.R_sun))))
    lamR = np.array(lamR); lamrev = np.array(lamrev); est = np.array(est)
    good = np.isfinite(lamR) & np.isfinite(lamrev)
    A_ = np.column_stack([est[good] ** 2, np.ones(good.sum())])
    (C2, w02), *_ = np.linalg.lstsq(A_, lamR[good] ** 2, rcond=None)
    pred = np.sqrt(C2 * est[good] ** 2 + w02)
    res = (lamR[good] - pred) / pred
    ratio = lamrev[good] / lamR[good]
    return (float(np.sqrt(C2)), float(np.sqrt(w02)), float(res.std()),
            float(np.abs(res).max()), float(ratio.mean()), float(ratio.std()))


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

    rows = []
    res = {}
    for bal in (False, True):
        tag = "balanced" if bal else "unbalanced"
        a = dipole(p, grid, K, 1.0, bal, "linear")
        tilt = tilt_closure_error(p, grid, K, bal)
        lamR, lamrev = pair_yield(p, grid, K_asym, w, bal)
        Trev = reversal_amplitude(p, grid, K, bal)
        res[tag] = dict(a=a, tilt=tilt, lamR=lamR, lamrev=lamrev, Trev=Trev)
        print(f"{tag:>11}:  a={a:.6e}  tilt_err={100*tilt:.3f}%  "
              f"lamR={lamR:.3f}  lamrev={lamrev:.2f}  T_rev={Trev:.4f}")

    print("\nScaling coefficients across the (eta,u0) plane ...")
    eta_vals = np.array([100., 200., 300., 400., 500., 600., 700., 800., 1000.])
    u0_vals = np.array([5., 8., 11., 14., 17., 20., 25.])
    for bal in (False, True):
        tag = "balanced" if bal else "unbalanced"
        C, w0, sc, mx, rm, rs = scaling_coefficients(bal, eta_vals, u0_vals)
        res[tag].update(C=C, w0=w0, scatter=sc, maxres=mx, ratio=rm, ratio_sd=rs)
        print(f"{tag:>11}:  C={C:.4f}  w0={w0:.3f}  scatter={100*sc:.2f}%  "
              f"max={100*mx:.2f}%  lamrev/lamR={rm:.3f}+/-{rs:.3f}")

    u, b = res["unbalanced"], res["balanced"]
    with open(os.path.join(OUTDIR, "balance_sensitivity.csv"), "w") as f:
        f.write("quantity,unbalanced,balanced,relative_change\n")
        for k in ("a", "tilt", "lamR", "lamrev", "Trev", "C", "w0",
                  "scatter", "maxres", "ratio", "ratio_sd"):
            rel = (b[k] - u[k]) / u[k] if u[k] not in (0.0,) else np.nan
            f.write(f"{k},{u[k]:.8e},{b[k]:.8e},{rel:.6e}\n")
    print(f"\nWrote {OUTDIR}/balance_sensitivity.csv")
