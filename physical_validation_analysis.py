#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical validation of the reduced response theory
(see docs/physical_validation.md).

The controlled experiments validate the reduced theory in "efficiency" mode,
where the nonlinear factors are injected into the source. This script tests
the theory against the *physical* implementations of the nonlinearities,
using only quantities derived from the adjoint (dipole-yield kernel)
framework -- no tuned parameters.

1.  Pair-yield function and derived effectivity range.
    The dipole yield of the actual four-ring source structure centred at
    latitude lambda_c is computed from the asymptotic kernel. A Gaussian fit
    gives the pair effectivity range lambda_R_pair, the derived counterpart
    of the closure's prescribed lambda_R.

2.  Shift-mode latitude quenching.
    The SFT is run with latitude_quenching_mode = "shift" (the activity belt
    physically displaced poleward by b_LQ T^2). Compared against:
      (a) the exact kernel prediction (same transport, so the kernel gives
          the response to the shifted source exactly);
      (b) the Gaussian closure built from derived quantities only
          (lambda_R_pair and the yield-weighted time moments);
      (c) the closure with the prescribed lambda_R = 20 deg, for contrast.

3.  Exact kernel prediction for tilt quenching.
    The tilt case changes the source geometry (polarity separation), so it
    is not a scalar source factor; the kernel nevertheless predicts it
    exactly. The aggregated yield-versus-separation function A(s) is
    computed, its odd polynomial fit A(s) = c1 s + c3 s^3 quantifies the
    cubic correction, and an improved closure
    Q_TQ = A(s0 f_TQ) / A(s0) is tested against the naive Q_TQ = f_TQ.

4.  Physical converging inflows.
    The SFT is run with inflow_mode = "physical" at 2 and 5 m/s. Inflows act
    on the transport operator, so no kernel prediction exists -- that is the
    point of the test: how well can a source-side algebraic factor
    Q_I = 1/(1 + b_I T^2) absorb a transport-side nonlinearity? The
    effective strength b_eff(T) = (1/Q_I - 1)/T^2 measures the quality of
    the algebraic form (constant b_eff = perfect form).

Outputs are saved to ./figures_validation/.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

import nonlinear_response_sft_experiment as nre
import mathematical_framework_analysis as mfa


OUTDIR = "figures_validation"
SECONDS_PER_YEAR = 365.25 * 86400.0

T_VALUES = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.4,
                     1.6, 1.8, 2.0, 2.2, 2.5], dtype=float)


# ============================================================
# 1. Source shapes and kernel predictions
# ============================================================

def ring_pair_shape(lat_deg: np.ndarray, lat_c: float, sep: float, width: float) -> np.ndarray:
    """Monopole-free four-ring source shape centred at latitude lat_c.

    Identical construction to bipolar_ring_source, with explicit centre and
    separation and unit amplitude.
    """
    NF = np.exp(-0.5 * ((lat_deg - (lat_c + sep / 2.0)) / width) ** 2)
    NL = np.exp(-0.5 * ((lat_deg - (lat_c - sep / 2.0)) / width) ** 2)
    SF = np.exp(-0.5 * ((lat_deg - (-lat_c - sep / 2.0)) / width) ** 2)
    SL = np.exp(-0.5 * ((lat_deg - (-lat_c + sep / 2.0)) / width) ** 2)
    s = (NF - NL) - (SF - SL)
    return s - s.mean()


def exact_kernel_prediction(p: nre.SFTParams, grid, K: np.ndarray, T: float,
                            case: str) -> float:
    """Exact final dipole for a source-side case, from the kernel.

    Reproduces the source construction of run_sft_case for the given case
    ("tilt" or "latitude-shift") and contracts it with the kernel. Valid for
    any source geometry because the transport operator is unchanged.
    """
    t_yr = np.asarray(grid["t_yr"])
    lat_deg = np.asarray(grid["lat_deg"])
    dt = float(grid["dt"])
    width = p.source_width_deg

    D = 0.0
    for k, t in enumerate(t_yr):
        phase = np.clip(t / p.cycle_years, 0.0, 1.0)
        lat0 = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
        sep = p.polarity_sep_deg
        if case == "tilt":
            sep = p.polarity_sep_deg / (1.0 + p.b_TQ * T**2)
        elif case == "latitude-shift":
            lat0 = lat0 + p.b_LQ * T**2
        else:
            raise ValueError(case)

        env = float(nre.cycle_envelope(float(t), p))
        shape = ring_pair_shape(lat_deg, float(lat0), float(sep), width)
        amp = p.source_amp * T * env / SECONDS_PER_YEAR
        D += dt * float(K[k] @ (amp * shape))
    return D


def pair_yield_function(p: nre.SFTParams, grid, K_asym: np.ndarray, w: np.ndarray,
                        fit_lat_deg=(5.0, 22.0)):
    """Final-to-initial dipole yield of the ring-pair structure vs latitude.

    Unlike the point yield, the pair yield changes sign at mid latitudes
    (the sign-change latitude lambda_rev is returned), so the Gaussian fit
    is restricted to the positive activity-belt range fit_lat_deg.
    """
    lat_deg = np.asarray(grid["lat_deg"])
    w_proj = w - w.mean()
    centres = np.linspace(2.0, 60.0, 117)
    f_pair = np.empty_like(centres)
    for i, lc in enumerate(centres):
        shape = ring_pair_shape(lat_deg, float(lc), p.polarity_sep_deg, p.source_width_deg)
        f_pair[i] = (K_asym @ shape) / (w_proj @ shape)

    neg = np.where(f_pair <= 0)[0]
    lam_rev = float(centres[neg[0]]) if len(neg) else np.nan

    sel = (centres >= fit_lat_deg[0]) & (centres <= fit_lat_deg[1]) & (f_pair > 0)
    slope, intercept = np.polyfit(centres[sel] ** 2, np.log(f_pair[sel]), 1)
    lamR_pair = float(np.sqrt(-1.0 / (2.0 * slope)))
    f0_pair = float(np.exp(intercept))
    return centres, f_pair, lamR_pair, f0_pair, lam_rev


def aggregated_separation_yield(p: nre.SFTParams, grid, K: np.ndarray,
                                seps: np.ndarray) -> np.ndarray:
    """Cycle-aggregated dipole yield A(s) as a function of polarity separation.

    A(s) = sum_k dt K^k . [env(t_k) shape(lambda0(t_k), s)] / seconds-per-year,
    so that the exact tilt-case dipole is D(T) = T A(sep(T)).
    """
    t_yr = np.asarray(grid["t_yr"])
    lat_deg = np.asarray(grid["lat_deg"])
    dt = float(grid["dt"])
    A_of_s = np.empty_like(seps)
    for i, s in enumerate(seps):
        total = 0.0
        for k, t in enumerate(t_yr):
            phase = np.clip(t / p.cycle_years, 0.0, 1.0)
            lat0 = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
            env = float(nre.cycle_envelope(float(t), p))
            shape = ring_pair_shape(lat_deg, float(lat0), float(s), p.source_width_deg)
            total += dt * float(K[k] @ (p.source_amp * env * shape / SECONDS_PER_YEAR))
        A_of_s[i] = total
    return A_of_s


# ============================================================
# 2. SFT scans for the physical modes
# ============================================================

def run_scan(p: nre.SFTParams, case: str) -> np.ndarray:
    """Signed final dipole D(T) over the amplitude scan for one case."""
    out = np.empty_like(T_VALUES)
    for i, T in enumerate(T_VALUES):
        _, D, _, _, _ = nre.run_sft_case(float(T), case, p)
        out[i] = D
    return out


def normalise(D: np.ndarray) -> np.ndarray:
    """Normalise a scan by its value at T = 1 (present in T_VALUES)."""
    i1 = int(np.argmin(np.abs(T_VALUES - 1.0)))
    return D / D[i1]


def gaussian_shift_closure(T: np.ndarray, p: nre.SFTParams, lamR: float,
                           t_mean: float, t_std: float) -> np.ndarray:
    """Closed-form Gaussian closure for a poleward belt shift delta = b_LQ T^2.

    Same derivation as the efficiency-mode closure: instantaneous factor
    exp(-[2 lambda0(t) delta + delta^2]/(2 lamR^2)) averaged over a Gaussian
    time weighting with moments (t_mean, t_std).
    """
    T = np.asarray(T, dtype=float)
    lam_bar = p.base_lat_deg + p.lat_drift_deg * (1.0 - t_mean / p.cycle_years)
    eps = (p.lat_drift_deg * t_std / (lamR * p.cycle_years)) ** 2
    delta = p.b_LQ * T**2
    return np.exp(-((2.0 * lam_bar * delta + (1.0 - eps) * delta**2) / (2.0 * lamR**2)))


# ============================================================
# 3. Main analysis
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    p = nre.SFTParams()
    grid = nre.make_grid(p)
    t_yr = np.asarray(grid["t_yr"])

    print("Building operators and kernels ...")
    A, M = mfa.build_operators(p, grid)
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    K_asym = mfa.extended_kernel(p, grid, A, M, K[0], extra_years=40.0)
    S_lin = mfa.linear_source_series(p, grid)
    omega = mfa.yield_weights(grid, K, S_lin)
    a_lin = float(omega.sum())

    # Yield-weighted time moments (derived, from the framework analysis)
    t_mean_w = float(np.sum(omega * t_yr) / omega.sum())
    t_std_w = float(np.sqrt(np.sum(omega * (t_yr - t_mean_w) ** 2) / omega.sum()))

    # ---------------------------------------------------------------
    # Pair-yield function and derived effectivity range
    # ---------------------------------------------------------------
    centres, f_pair, lamR_pair, f0_pair, lam_rev = pair_yield_function(p, grid, K_asym, w)
    print(f"Pair-yield effectivity range (fit over 5-22 deg): "
          f"lambda_R_pair = {lamR_pair:.2f} deg "
          f"(prescribed in the closure: {p.lambda_R_deg:.1f} deg)")
    print(f"Pair yield changes sign at lambda_rev ~ {lam_rev:.1f} deg; positive "
          f"multiplicative closures are meaningful only while the "
          f"emergence-weighted belt stays below this.")
    # A-priori reversal estimate: the yield-weighted belt latitude plus the
    # shift reaches lambda_rev at T ~ sqrt((lambda_rev - lambda_bar)/b_LQ).
    lam_bar_w = p.base_lat_deg + p.lat_drift_deg * (1.0 - t_mean_w / p.cycle_years)
    T_rev_est = float(np.sqrt(max(lam_rev - lam_bar_w, 0.0) / p.b_LQ))
    print(f"A-priori reversal estimate for shift mode: "
          f"T ~ sqrt((lambda_rev - lambda_bar)/b_LQ) = {T_rev_est:.2f}")

    # ---------------------------------------------------------------
    # Task 1a: shift-mode latitude quenching
    # ---------------------------------------------------------------
    print("\nRunning shift-mode latitude-quenching scan ...")
    p_shift = replace(p, latitude_quenching_mode="shift")
    D_shift = run_scan(p_shift, "latitude")
    D_shift_exact = np.array([
        exact_kernel_prediction(p, grid, K, float(T), "latitude-shift") for T in T_VALUES
    ])
    raw_err_shift = float(np.max(np.abs(D_shift_exact - D_shift) / np.abs(D_shift)))
    print(f"Exact kernel vs SFT (shift mode), max relative error: {raw_err_shift:.3e}")

    # Compare in efficiency units Q(T) = D/(aT): the shift response reverses
    # sign, so normalising by the (strongly suppressed) T = 1 value would
    # inflate every metric.
    Q_shift = D_shift / (a_lin * T_VALUES)
    Q_shift_exact = D_shift_exact / (a_lin * T_VALUES)
    q_derived = gaussian_shift_closure(T_VALUES, p, lamR_pair, t_mean_w, t_std_w)
    q_prescribed = gaussian_shift_closure(T_VALUES, p, p.lambda_R_deg, t_mean_w, t_std_w)

    # Measured reversal amplitude (sign change of D_shift)
    neg_idx = np.where(D_shift < 0)[0]
    if len(neg_idx):
        j = neg_idx[0]
        T_rev = float(T_VALUES[j - 1] + (T_VALUES[j] - T_VALUES[j - 1])
                      * D_shift[j - 1] / (D_shift[j - 1] - D_shift[j]))
    else:
        T_rev = np.nan
    print(f"Measured sign reversal of the shift-mode response at T_rev = {T_rev:.2f} "
          f"(a-priori estimate {T_rev_est:.2f})")

    weak = T_VALUES <= 0.8
    dev_exact = float(np.max(np.abs(Q_shift_exact - Q_shift)))
    dev_derived = float(np.max(np.abs(q_derived - Q_shift)))
    dev_derived_weak = float(np.max(np.abs(q_derived[weak] - Q_shift[weak])))
    dev_prescribed = float(np.max(np.abs(q_prescribed - Q_shift)))
    dev_prescribed_weak = float(np.max(np.abs(q_prescribed[weak] - Q_shift[weak])))
    print(f"Shift-mode efficiency Q = D/(aT), max |dQ| over the full scan: "
          f"exact kernel {dev_exact:.2e}, derived Gaussian closure {dev_derived:.3f}, "
          f"prescribed closure {dev_prescribed:.3f}")
    print(f"  over the weak-shift domain T <= 0.8: "
          f"derived (lamR={lamR_pair:.1f}) {dev_derived_weak:.3f}, "
          f"prescribed (lamR={p.lambda_R_deg:.0f}) {dev_prescribed_weak:.3f}")

    # ---------------------------------------------------------------
    # Task 2: exact tilt prediction and improved closure
    # ---------------------------------------------------------------
    print("\nRunning tilt-quenching scan ...")
    D_tilt = run_scan(p, "tilt")
    D_tilt_exact = np.array([
        exact_kernel_prediction(p, grid, K, float(T), "tilt") for T in T_VALUES
    ])
    raw_err_tilt = float(np.max(np.abs(D_tilt_exact - D_tilt) / np.abs(D_tilt)))
    print(f"Exact kernel vs SFT (tilt), max relative error: {raw_err_tilt:.3e}")

    seps = np.linspace(0.5, 6.0, 23)
    A_of_s = aggregated_separation_yield(p, grid, K, seps)
    c1, c3 = np.linalg.lstsq(
        np.column_stack([seps, seps**3]), A_of_s, rcond=None
    )[0]
    print(f"Aggregated pair yield A(s) = c1 s + c3 s^3: c1 = {c1:.4e}, c3 = {c3:.4e}, "
          f"c3/c1 = {c3/c1:.5f} deg^-2")

    def A_fit(s: np.ndarray) -> np.ndarray:
        return c1 * s + c3 * s**3

    Phat_tilt = normalise(D_tilt)
    Phat_tilt_exact = normalise(D_tilt_exact)
    f_TQ = 1.0 / (1.0 + p.b_TQ * T_VALUES**2)
    Phat_naive = normalise(T_VALUES * f_TQ)
    Phat_improved = normalise(T_VALUES * A_fit(p.polarity_sep_deg * f_TQ))

    dev_tilt_exact = float(np.max(np.abs(Phat_tilt_exact - Phat_tilt)))
    dev_tilt_naive = float(np.max(np.abs(Phat_naive - Phat_tilt)))
    dev_tilt_improved = float(np.max(np.abs(Phat_improved - Phat_tilt)))
    print(f"Tilt normalised response, max |dP_hat|: exact kernel {dev_tilt_exact:.2e}, "
          f"naive Q_TQ=f_TQ {dev_tilt_naive:.4f}, improved (cubic) {dev_tilt_improved:.4f}")

    # ---------------------------------------------------------------
    # Task 1b: physical converging inflows
    # ---------------------------------------------------------------
    print("\nRunning physical-inflow scans (2 and 5 m/s) ...")
    p_inf2 = replace(p, inflow_mode="physical", inflow_amp_m_s=2.0)
    p_inf5 = replace(p, inflow_mode="physical", inflow_amp_m_s=5.0)
    D_inf2 = run_scan(p_inf2, "inflow")
    D_inf5 = run_scan(p_inf5, "inflow")

    Q_inf2 = D_inf2 / (a_lin * T_VALUES)
    Q_inf5 = D_inf5 / (a_lin * T_VALUES)
    b_eff2 = (1.0 / Q_inf2 - 1.0) / T_VALUES**2
    with np.errstate(divide="ignore", invalid="ignore"):
        b_eff5 = np.where(Q_inf5 > 0, (1.0 / Q_inf5 - 1.0) / T_VALUES**2, np.nan)

    # Least-squares algebraic fits: leading order (T <= 1.2) and full scan
    def fit_b(Q: np.ndarray, Tmax: float = np.inf) -> float:
        good = (Q > 0) & (T_VALUES <= Tmax)
        return float(np.sum((1.0 / Q[good] - 1.0) * T_VALUES[good] ** 2)
                     / np.sum(T_VALUES[good] ** 4))

    b_lead2, b_full2 = fit_b(Q_inf2, 1.2), fit_b(Q_inf2)
    b_lead5, b_full5 = fit_b(Q_inf5, 1.2), fit_b(Q_inf5)
    print(f"Effective algebraic strength, 2 m/s: b_eff(1) = {b_eff2[5]:.3f}, "
          f"leading-order fit (T<=1.2) {b_lead2:.3f}, full-scan fit {b_full2:.3f}")
    print(f"Effective algebraic strength, 5 m/s: b_eff(1) = {b_eff5[5]:.3f}, "
          f"leading-order fit (T<=1.2) {b_lead5:.3f}, full-scan fit {b_full5:.3f}")
    if np.any(D_inf5 < 0):
        Trev = T_VALUES[np.where(D_inf5 < 0)[0][0]]
        print(f"5 m/s inflow reverses the dipole sign at T = {Trev:.2f} "
              f"(outside the domain of any positive Q closure)")

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    # V1: pair-yield function
    lam_plot = np.linspace(0.0, 60.0, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(centres, f_pair, "o", markersize=3, label="pair yield (kernel)")
    plt.plot(lam_plot, f0_pair * np.exp(-lam_plot**2 / (2.0 * lamR_pair**2)), "--",
             label=fr"Gaussian fit, $\lambda_R = {lamR_pair:.1f}^\circ$")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Pair centre latitude [deg]")
    plt.ylabel("Final-to-initial dipole yield")
    plt.title("Dipole yield of the ring-pair source structure")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figV1_pair_yield.png"), dpi=200)
    plt.close()

    # V2: shift-mode validation, in efficiency units Q = D/(aT)
    plt.figure(figsize=(8, 5))
    plt.plot(T_VALUES, Q_shift, "o", markersize=6, label="SFT, physical shift mode")
    plt.plot(T_VALUES, Q_shift_exact, "-", label="exact kernel prediction")
    plt.plot(T_VALUES, q_derived, "--",
             label=fr"derived Gaussian closure ($\lambda_R = {lamR_pair:.1f}^\circ$)")
    plt.plot(T_VALUES, q_prescribed, ":",
             label=fr"prescribed closure ($\lambda_R = {p.lambda_R_deg:.0f}^\circ$)")
    plt.axvline(T_rev, linestyle="--", linewidth=1.2, color="grey",
                label=fr"measured sign reversal $T_{{\rm rev}} = {T_rev:.2f}$")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Latitude-quenching efficiency $Q(T) = D/(aT)$")
    plt.title("Physical (shift-mode) latitude quenching vs reduced theory")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figV2_shift_validation.png"), dpi=200)
    plt.close()

    # V3: tilt exactness and improved closure
    plt.figure(figsize=(8, 5))
    plt.plot(T_VALUES, Phat_tilt, "o", markersize=6, label="SFT, tilt case")
    plt.plot(T_VALUES, Phat_tilt_exact, "-", label="exact kernel prediction")
    plt.plot(T_VALUES, Phat_naive, ":", label=r"naive closure $Q_{\rm TQ}=f_{\rm TQ}$")
    plt.plot(T_VALUES, Phat_improved, "--", label="improved closure (cubic yield)")
    plt.axhline(1.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Normalized response $\hat{P}(T)$")
    plt.title("Tilt quenching: exact kernel prediction and closures")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figV3_tilt_exact.png"), dpi=200)
    plt.close()

    # V4: physical inflows vs algebraic closure
    plt.figure(figsize=(8, 5))
    plt.plot(T_VALUES, D_inf2 / a_lin, "o-", label=r"physical inflow 2 m s$^{-1}$: $D/a$")
    plt.plot(T_VALUES, D_inf5 / a_lin, "s-", label=r"physical inflow 5 m s$^{-1}$: $D/a$")
    plt.plot(T_VALUES, T_VALUES / (1.0 + b_lead2 * T_VALUES**2), "--",
             label=fr"algebraic, leading-order fit $b_I = {b_lead2:.2f}$")
    plt.plot(T_VALUES, T_VALUES / (1.0 + b_lead5 * T_VALUES**2), ":",
             label=fr"algebraic, leading-order fit $b_I = {b_lead5:.2f}$")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"$T\,Q_I(T)$ (dipole in units of $a$)")
    plt.title("Physical converging inflows vs algebraic closure")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figV4_inflow_validation.png"), dpi=200)
    plt.close()

    # V5: effective algebraic strength
    plt.figure(figsize=(8, 5))
    plt.plot(T_VALUES, b_eff2, "o-", label=r"2 m s$^{-1}$")
    plt.plot(T_VALUES, b_eff5, "s-", label=r"5 m s$^{-1}$")
    plt.axhline(0.2, linestyle="--", linewidth=1.2,
                label=r"calibrated $b_I = 0.2$ (Talafha et al. 2025)")
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"$b_{\rm eff}(T) = (1/Q_I - 1)/T^2$")
    plt.title("Constancy test of the algebraic inflow closure")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figV5_binflow_effective.png"), dpi=200)
    plt.close()

    # ---------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------
    with open(os.path.join(OUTDIR, "validation_results.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        f.write(f"lambda_R_pair_deg,{lamR_pair:.6f}\n")
        f.write(f"lambda_rev_deg,{lam_rev:.6f}\n")
        f.write(f"shift_T_rev_measured,{T_rev:.6f}\n")
        f.write(f"shift_T_rev_estimate,{T_rev_est:.6f}\n")
        f.write(f"shift_exact_kernel_max_rel_err,{raw_err_shift:.6e}\n")
        f.write(f"shift_Q_at_T1,{Q_shift[5]:.6f}\n")
        f.write(f"shift_closure_derived_max_abs_devQ_full,{dev_derived:.6e}\n")
        f.write(f"shift_closure_derived_max_abs_devQ_weak,{dev_derived_weak:.6e}\n")
        f.write(f"shift_closure_prescribed_max_abs_devQ_full,{dev_prescribed:.6e}\n")
        f.write(f"shift_closure_prescribed_max_abs_devQ_weak,{dev_prescribed_weak:.6e}\n")
        f.write(f"tilt_exact_kernel_max_rel_err,{raw_err_tilt:.6e}\n")
        f.write(f"tilt_naive_closure_max_abs_dev,{dev_tilt_naive:.6e}\n")
        f.write(f"tilt_improved_closure_max_abs_dev,{dev_tilt_improved:.6e}\n")
        f.write(f"tilt_c3_over_c1_deg-2,{c3/c1:.6e}\n")
        f.write(f"inflow2_b_eff_at_T1,{b_eff2[5]:.6f}\n")
        f.write(f"inflow2_b_fit_leading,{b_lead2:.6f}\n")
        f.write(f"inflow2_b_fit_full,{b_full2:.6f}\n")
        f.write(f"inflow5_b_eff_at_T1,{b_eff5[5]:.6f}\n")
        f.write(f"inflow5_b_fit_leading,{b_lead5:.6f}\n")
        f.write(f"inflow5_b_fit_full,{b_full5:.6f}\n")

    print(f"\nSaved figures and validation_results.csv to: {OUTDIR}")
