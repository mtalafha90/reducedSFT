#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deeper investigation of the mathematical framework of the reduced response
theory (see docs/mathematical_framework.md).

This script makes the central object of the reduced theory -- the
dipole-yield kernel K (the adjoint, or Green's-function, state of the SFT
model) -- fully explicit at the discrete level, and uses it to derive
quantities that the reduced theory otherwise prescribes:

1.  Discrete adjoint and exact duality.
    The forward update of the experiment is

        B^{k+1} = Pi M^{-1} [ (I + dt A) B^k + dt S^k ],

    where A is the explicit (advection + decay) operator, M the implicit
    diffusion matrix (symmetric), and Pi = I - (1/n) 1 1^T the monopole
    projection. The final axial dipole D = w^T B^N is therefore

        D = sum_k dt (K^k)^T S^k,       omega_k = dt (K^k)^T S^k,

    with the kernel obtained by the backward recursion

        K^{N-1} = M^{-1} Pi w,
        K^{k-1} = M^{-1} Pi (I + dt A)^T K^k .

    The duality identity D_adjoint = D_forward is verified to round-off.

2.  The intrinsic dynamo effectivity range lambda_R of the transport model,
    measured from the asymptotic yield function f(lambda) = K^0 / (Pi w) and
    compared with the advection--diffusion estimate sqrt(eta/(2 u0 R)).

3.  A closure ladder for latitude quenching. Because source-side factors act
    by linearity, the exact reduced prediction is the kernel-weighted average

        D_LQ(T) = T sum_k omega_k Q_LQ(t_k, T) / (unit normalisation),

    with zero closure error. Truncating its cumulant expansion gives, in
    order of decreasing accuracy: the yield-weighted Gaussian closure, the
    envelope-weighted Gaussian closure (the default reduced theory), and the
    static closure. Each rung's error is measured against the stored SFT
    scan.

4.  Bifurcation diagrams of the reduced cycle map T_{n+1} = D0 T_n Q(T_n),
    confirming the analytic period-doubling boundary q(T*) = 2 and showing
    the modulated/chaotic regime beyond it.

Outputs are saved to ./figures_framework/.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import nonlinear_response_sft_experiment as nre


OUTDIR = "figures_framework"


# ============================================================
# 1. Discrete forward operators and adjoint kernel
# ============================================================

def build_operators(p: nre.SFTParams, grid) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the explicit operator A and implicit diffusion matrix M.

    A is built column by column by applying the experiment's own explicit
    right-hand side to unit vectors, so the adjoint is exactly consistent
    with the forward code (upwind advection + decay; no inflow here).
    """
    n = p.nmu
    A = np.zeros((n, n))
    zero_src = np.zeros(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = nre.sft_rhs_mu(
            B=e, t=0.0, T=1.0, p=p, grid=grid, source=zero_src, use_inflow=False
        )

    M = (
        np.diag(np.asarray(grid["diff_diag"]))
        + np.diag(np.asarray(grid["diff_lower"]), -1)
        + np.diag(np.asarray(grid["diff_upper"]), 1)
    )
    return A, M


def dipole_weight(p: nre.SFTParams, grid) -> np.ndarray:
    """Weight vector w with D = w^T B = 3/2 trapz(B mu, mu)."""
    mu = np.asarray(grid["mu"])
    tw = np.full(p.nmu, float(grid["dmu"]))
    tw[0] *= 0.5
    tw[-1] *= 0.5
    return 1.5 * mu * tw


def adjoint_kernel(p: nre.SFTParams, grid, A: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward recursion for the dipole-yield kernel K^k, k = 0..N-1.

    K^k gives the final-dipole contribution of a unit source applied during
    step k: D = sum_k dt (K^k)^T S^k. M is symmetric, so M^{-T} = M^{-1}.
    """
    n = p.nmu
    t_yr = np.asarray(grid["t_yr"])
    N = len(t_yr)
    dt = float(grid["dt"])

    w = dipole_weight(p, grid)
    Minv = np.linalg.inv(M)

    def project(v: np.ndarray) -> np.ndarray:
        return v - v.mean()

    propT = (np.eye(n) + dt * A).T

    K = np.empty((N, n))
    K[N - 1] = Minv @ project(w)
    for k in range(N - 2, -1, -1):
        K[k] = Minv @ project(propT @ K[k + 1])
    return K, w


def linear_source_series(p: nre.SFTParams, grid) -> np.ndarray:
    """S^k of the linear reference case (T = 1, no quenchings)."""
    t_yr = np.asarray(grid["t_yr"])
    lat_deg = np.asarray(grid["lat_deg"])
    S = np.empty((len(t_yr), p.nmu))
    for k, t in enumerate(t_yr):
        S[k] = nre.bipolar_ring_source(
            lat_deg=lat_deg, t=float(t), T=1.0, p=p,
            tilt_quenching=False, latitude_quenching=False, inflow_quenching=False,
        )
    return S


# ============================================================
# 2. Diagnostics built on the kernel
# ============================================================

def duality_check(p: nre.SFTParams, grid, K: np.ndarray, S: np.ndarray) -> tuple[float, float]:
    """Compare the adjoint sum with the forward linear run."""
    dt = float(grid["dt"])
    D_adj = float(dt * np.einsum("ki,ki->", K, S))
    _, D_fwd, _, _, _ = nre.run_sft_case(1.0, "linear", p)
    return D_adj, D_fwd


def intrinsic_effectivity_range(p: nre.SFTParams, grid, Kvec: np.ndarray, w: np.ndarray,
                                fit_lat_deg=(5.0, 45.0)) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Yield function f(lambda) = K / (Pi w) and its Gaussian width.

    f(lambda) is the ratio of the final dipole produced by a unit source at
    latitude lambda to the dipole it deposits instantaneously. A Gaussian fit
    f ~ f0 exp(-lambda^2 / (2 lambda_R^2)) over the activity latitudes gives
    the dynamo effectivity range lambda_R of the transport model for the
    evolution horizon encoded in Kvec.
    """
    lat_deg = np.asarray(grid["lat_deg"])
    w_proj = w - w.mean()
    mask_def = np.abs(lat_deg) > 3.0  # avoid the w ~ 0 equator
    f = np.full_like(w, np.nan)
    f[mask_def] = Kvec[mask_def] / w_proj[mask_def]

    sel = (lat_deg >= fit_lat_deg[0]) & (lat_deg <= fit_lat_deg[1]) & (f > 0)
    lam = lat_deg[sel]
    slope, intercept = np.polyfit(lam**2, np.log(f[sel]), 1)
    lamR_fit = float(np.sqrt(-1.0 / (2.0 * slope)))
    f0_fit = float(np.exp(intercept))
    return lat_deg, f, lamR_fit, f0_fit


def effectivity_estimate_deg(p: nre.SFTParams) -> float:
    """Advection-diffusion balance estimate lambda_R ~ sqrt(eta/(2 u0 R))."""
    eta = p.eta_km2_s * 1.0e10
    u0 = p.u0_m_s * 100.0
    return float(np.rad2deg(np.sqrt(eta / (2.0 * u0 * p.R_sun))))


def extended_kernel(p: nre.SFTParams, grid, A: np.ndarray, M: np.ndarray,
                    K0: np.ndarray, extra_years: float) -> np.ndarray:
    """Continue the backward kernel recursion beyond the simulated window.

    The transport operators are autonomous (no inflow in the linear case), so
    evolving K^0 backwards for extra_years approximates the t -> infinity
    (asymptotic) dipole-yield function, free of the finite-horizon
    broadening of the 11-yr cycle window.
    """
    dt = float(grid["dt"])
    n_extra = int(round(extra_years * 365.25 * 86400.0 / dt))
    Minv = np.linalg.inv(M)
    propT = (np.eye(p.nmu) + dt * A).T
    K = K0.copy()
    for _ in range(n_extra):
        v = propT @ K
        v = v - v.mean()
        K = Minv @ v
    return K


def yield_weights(grid, K: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-step contribution omega_k of the linear source to the dipole."""
    return float(grid["dt"]) * np.einsum("ki,ki->k", K, S)


# ============================================================
# 3. Closure ladder for latitude quenching
# ============================================================

def q_lq_instantaneous(t: np.ndarray, T: float, p: nre.SFTParams) -> np.ndarray:
    """Instantaneous latitude-quenching factor along the drifting belt."""
    phase = np.clip(t / p.cycle_years, 0.0, 1.0)
    lam0 = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
    delta = p.b_LQ * T**2
    return np.exp(-((2.0 * lam0 * delta + delta**2) / (2.0 * p.lambda_R_deg**2)))


def gaussian_closure(T: np.ndarray, p: nre.SFTParams, t_mean: float, t_std: float) -> np.ndarray:
    """Second-cumulant Gaussian closure with prescribed time moments.

        Q_LQ(T) = exp(-[2 lam_bar delta + (1 - eps) delta^2] / (2 lamR^2)),
        lam_bar = lambda_b + lambda_d (1 - t_mean / P),
        eps     = (lambda_d t_std / (lamR P))^2 .
    """
    T = np.asarray(T, dtype=float)
    lam_bar = p.base_lat_deg + p.lat_drift_deg * (1.0 - t_mean / p.cycle_years)
    eps = (p.lat_drift_deg * t_std / (p.lambda_R_deg * p.cycle_years)) ** 2
    delta = p.b_LQ * T**2
    return np.exp(-((2.0 * lam_bar * delta + (1.0 - eps) * delta**2) / (2.0 * p.lambda_R_deg**2)))


def closure_ladder(p: nre.SFTParams, grid, omega: np.ndarray,
                   T_values: np.ndarray) -> dict[str, np.ndarray]:
    """Normalised latitude-case predictions P_hat(T) at four closure levels."""
    t_yr = np.asarray(grid["t_yr"])

    # (i) exact kernel-weighted average (zero closure error by linearity)
    def exact_TQ(T: float) -> float:
        return T * float(np.sum(omega * q_lq_instantaneous(t_yr, T, p)))

    exact = np.array([exact_TQ(T) for T in T_values])
    exact /= exact_TQ(1.0)

    # (ii) yield-weighted Gaussian closure
    wsum = omega.sum()
    t_mean_w = float(np.sum(omega * t_yr) / wsum)
    t_std_w = float(np.sqrt(np.sum(omega * (t_yr - t_mean_w) ** 2) / wsum))
    qw = gaussian_closure(T_values, p, t_mean_w, t_std_w)
    yieldw = T_values * qw / (1.0 * gaussian_closure(np.array([1.0]), p, t_mean_w, t_std_w))[0]

    # (iii) envelope-weighted Gaussian closure (the reduced theory's default)
    qe = gaussian_closure(T_values, p, p.source_peak_time_yr, p.source_time_width_yr)
    envelope = T_values * qe / (1.0 * gaussian_closure(np.array([1.0]), p, p.source_peak_time_yr, p.source_time_width_yr))[0]

    # (iv) static closure (original draft)
    qs = gaussian_closure(T_values, p, p.cycle_years, 0.0)  # t = P -> lam_bar = base latitude
    static = T_values * qs / (1.0 * gaussian_closure(np.array([1.0]), p, p.cycle_years, 0.0))[0]

    return {
        "exact": exact,
        "yield_weighted": yieldw,
        "envelope": envelope,
        "static": static,
        "t_mean_w": np.array([t_mean_w]),
        "t_std_w": np.array([t_std_w]),
    }


def load_sft_latitude_scan(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the stored normalised latitude-case SFT scan."""
    T, Phat = [], []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["case"] == "latitude":
                T.append(float(row["T"]))
                Phat.append(float(row["P_hat"]))
    return np.array(T), np.array(Phat)


# ============================================================
# 4. Bifurcation diagrams of the reduced cycle map
# ============================================================

def iterate_map(D0s: np.ndarray, Q_of_T, n_transient: int = 400, n_keep: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Iterate T_{n+1} = D0 T_n Q(T_n) for an array of gains D0."""
    T = np.ones_like(D0s)
    for _ in range(n_transient):
        T = D0s * T * Q_of_T(T)
    xs, ys = [], []
    for _ in range(n_keep):
        T = D0s * T * Q_of_T(T)
        xs.append(D0s)
        ys.append(T)
    return np.concatenate(xs), np.concatenate(ys)


def analytic_period_doubling_gain(p: nre.SFTParams) -> float:
    """Gain D0 at which the reference-Q fixed point reaches M'(T*) = -1.

    Solved by bisection on D0: for each D0 find T* (unique root of
    ln D0 + ln Q = 0) and evaluate q* = -T* dlnQ/dT analytically.
    """
    lam0 = nre.lq_lambda_eff(p)
    eps = nre.lq_eps(p)

    def qstar(D0: float) -> float:
        lo, hi = 1e-6, 50.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            h = np.log(D0) + nre.reduced_lnQ(mid, p.b_TQ, p.b_LQ, p.b_I, lam0, p.lambda_R_deg, eps)
            if h > 0:
                lo = mid
            else:
                hi = mid
        Tstar = 0.5 * (lo + hi)
        return -Tstar * nre.reduced_dlnQ_dT(Tstar, p.b_TQ, p.b_LQ, p.b_I, lam0, p.lambda_R_deg, eps)

    lo, hi = 1.01, 1e4
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if qstar(mid) < 2.0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ============================================================
# 5. Figures
# ============================================================

def make_figures(p: nre.SFTParams, grid, K, w, S, omega, ladder, T_sft, Phat_sft,
                 lat_deg, f_yield, lamR_fit, f0_fit,
                 f_asym, lamR_asym, f0_asym, D0_pd) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    t_yr = np.asarray(grid["t_yr"])

    # ---- F1: kernel evolution -------------------------------------------
    plt.figure(figsize=(8, 5))
    for t_snap in [10.5, 9.0, 7.0, 4.5, 2.0, 0.0]:
        k = int(np.argmin(np.abs(t_yr - t_snap)))
        plt.plot(lat_deg, K[k] / float(grid["dmu"]), label=fr"$t_{{\rm emerge}}={t_snap:.1f}$ yr")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Latitude [deg]")
    plt.ylabel(r"Kernel $K(\mu, t)/\Delta\mu$")
    plt.title("Dipole-yield kernel at different emergence times")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figF1_kernel_evolution.png"), dpi=200)
    plt.close()

    # ---- F2: yield functions and Gaussian fits ---------------------------
    lam_plot = np.linspace(0.0, 60.0, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(lat_deg, f_yield, "o", markersize=3,
             label=r"11-yr horizon: $f(\lambda) = K^0/(\Pi w)$")
    plt.plot(lam_plot, f0_fit * np.exp(-lam_plot**2 / (2.0 * lamR_fit**2)), "--",
             label=fr"Gaussian fit, $\lambda_R = {lamR_fit:.1f}^\circ$")
    plt.plot(lat_deg, f_asym, "s", markersize=3,
             label=r"asymptotic ($+40$ yr) yield")
    plt.plot(lam_plot, f0_asym * np.exp(-lam_plot**2 / (2.0 * lamR_asym**2)), ":",
             label=fr"Gaussian fit, $\lambda_R = {lamR_asym:.1f}^\circ$")
    plt.xlim(0.0, 60.0)
    plt.xlabel("Emergence latitude [deg]")
    plt.ylabel("Final-to-initial dipole yield")
    plt.title("Dynamo effectivity range of the transport model")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figF2_intrinsic_effectivity_range.png"), dpi=200)
    plt.close()

    # ---- F3: emergence-time weights -------------------------------------
    env = np.exp(-0.5 * ((t_yr - p.source_peak_time_yr) / p.source_time_width_yr) ** 2)
    plt.figure(figsize=(8, 5))
    plt.plot(t_yr, omega / np.max(omega), label=r"yield weight $\omega(t)$ (kernel $\times$ source)")
    plt.plot(t_yr, env, "--", label=r"envelope $s_1(t)$ alone")
    plt.xlabel("Emergence time [yr]")
    plt.ylabel("Normalised weight")
    plt.title("Effective emergence-time weighting of the final dipole")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figF3_yield_weights.png"), dpi=200)
    plt.close()

    # ---- F4: closure ladder ----------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(T_sft, Phat_sft, "o", markersize=6, label="SFT latitude case")
    plt.plot(T_sft, ladder["exact"], "-", label="exact kernel average")
    plt.plot(T_sft, ladder["yield_weighted"], "--", label="yield-weighted Gaussian closure")
    plt.plot(T_sft, ladder["envelope"], "-.", label="envelope Gaussian closure (default)")
    plt.plot(T_sft, ladder["static"], ":", label="static closure (draft)")
    plt.axhline(1.0, linewidth=1)
    plt.xlabel(r"Normalized cycle amplitude $T$")
    plt.ylabel(r"Normalized response $\hat{P}(T)$")
    plt.title("Latitude quenching: closure ladder")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figF4_closure_ladder.png"), dpi=200)
    plt.close()

    # ---- F5: bifurcation diagrams ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    D0s = np.linspace(1.02, 30.0, 1600)
    x, y = iterate_map(D0s, lambda T: np.asarray(nre.Q_theory(T, p, "combined")))
    axes[0].plot(x, y, ",", alpha=0.5)
    axes[0].axvline(D0_pd, linestyle="--", linewidth=1.2,
                    label=fr"analytic $\mathcal{{M}}'=-1$ at $D_0={D0_pd:.2f}$")
    axes[0].set_xlabel(r"Linear gain $D_0$")
    axes[0].set_ylabel(r"Attractor of $T_{n+1}=D_0T_nQ(T_n)$")
    axes[0].set_title("Reference quenchings, varying gain")
    axes[0].legend(fontsize=8)

    bLQs = np.linspace(0.0, 20.0, 1600)
    pp = nre.SFTParams()

    def Q_bLQ(T: np.ndarray) -> np.ndarray:
        lam0 = nre.lq_lambda_eff(pp)
        eps = nre.lq_eps(pp)
        delta = bLQs * T**2
        QTQ = 1.0 / (1.0 + pp.b_TQ * T**2)
        QI = 1.0 / (1.0 + pp.b_I * T**2)
        QLQ = np.exp(-((2.0 * lam0 * delta + (1.0 - eps) * delta**2) / (2.0 * pp.lambda_R_deg**2)))
        return QTQ * QI * QLQ

    T = np.ones_like(bLQs)
    D0_fixed = 3.0
    for _ in range(400):
        T = D0_fixed * T * Q_bLQ(T)
    for _ in range(120):
        T = D0_fixed * T * Q_bLQ(T)
        axes[1].plot(bLQs, T, ",", alpha=0.5, color="C0")
    axes[1].set_xlabel(r"Latitude-quenching strength $b_{\rm LQ}$ [deg]")
    axes[1].set_title(fr"$D_0 = {D0_fixed}$, varying $b_{{\rm LQ}}$")

    fig.suptitle("Period doubling and modulation in the reduced cycle map")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figF5_bifurcation.png"), dpi=200)
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    p = nre.SFTParams()
    grid = nre.make_grid(p)
    t_yr = np.asarray(grid["t_yr"])

    print("Building operators and adjoint kernel ...")
    A, M = build_operators(p, grid)
    K, w = adjoint_kernel(p, grid, A, M)
    S = linear_source_series(p, grid)
    omega = yield_weights(grid, K, S)

    # --- duality ----------------------------------------------------------
    D_adj, D_fwd = duality_check(p, grid, K, S)
    rel = abs(D_adj - D_fwd) / abs(D_fwd)
    print(f"Duality check: D_adjoint = {D_adj:.12e}, D_forward = {D_fwd:.12e}, "
          f"relative difference = {rel:.3e}")

    # --- intrinsic effectivity range ---------------------------------------
    lat_deg, f_yield, lamR_fit, f0_fit = intrinsic_effectivity_range(p, grid, K[0], w)
    K_asym = extended_kernel(p, grid, A, M, K[0], extra_years=40.0)
    _, f_asym, lamR_asym, f0_asym = intrinsic_effectivity_range(p, grid, K_asym, w)
    lamR_est = effectivity_estimate_deg(p)
    print(f"Effectivity range at the 11-yr horizon: lambda_R = {lamR_fit:.2f} deg")
    print(f"Asymptotic (+40 yr) effectivity range:  lambda_R = {lamR_asym:.2f} deg")
    print(f"Advection-diffusion balance estimate sqrt(eta/(2 u0 R)) = {lamR_est:.2f} deg; "
          f"prescribed in the closure: {p.lambda_R_deg:.1f} deg")

    # --- closure ladder -----------------------------------------------------
    csv_path = os.path.join("figures_nonlinear_response_v2", "nonlinear_response_results_v2.csv")
    T_sft, Phat_sft = load_sft_latitude_scan(csv_path)
    ladder = closure_ladder(p, grid, omega, T_sft)
    t_mean_w = ladder["t_mean_w"][0]
    t_std_w = ladder["t_std_w"][0]
    print(f"Yield-weighted emergence-time moments: mean = {t_mean_w:.3f} yr "
          f"(envelope: {p.source_peak_time_yr} yr), std = {t_std_w:.3f} yr "
          f"(envelope: {p.source_time_width_yr} yr)")

    print("\nClosure ladder, max |P_hat - P_hat_SFT| over the scan:")
    ladder_rows = []
    for name in ["exact", "yield_weighted", "envelope", "static"]:
        dev = float(np.max(np.abs(ladder[name] - Phat_sft)))
        ladder_rows.append((name, dev))
        print(f"  {name:15s} {dev:.6f}")

    # --- bifurcation --------------------------------------------------------
    D0_pd = analytic_period_doubling_gain(p)
    print(f"\nAnalytic period-doubling gain for reference quenchings: D0 = {D0_pd:.3f}")

    make_figures(p, grid, K, w, S, omega, ladder, T_sft, Phat_sft,
                 lat_deg, f_yield, lamR_fit, f0_fit,
                 f_asym, lamR_asym, f0_asym, D0_pd)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "framework_results.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        f.write(f"duality_relative_difference,{rel:.6e}\n")
        f.write(f"a_linear_coefficient,{D_fwd:.10e}\n")
        f.write(f"lambda_R_horizon11yr_deg,{lamR_fit:.6f}\n")
        f.write(f"lambda_R_asymptotic_deg,{lamR_asym:.6f}\n")
        f.write(f"lambda_R_estimate_deg,{lamR_est:.6f}\n")
        f.write(f"lambda_R_prescribed_deg,{p.lambda_R_deg:.6f}\n")
        f.write(f"t_mean_yield_yr,{t_mean_w:.6f}\n")
        f.write(f"t_std_yield_yr,{t_std_w:.6f}\n")
        f.write(f"D0_period_doubling_reference,{D0_pd:.6f}\n")
        for name, dev in ladder_rows:
            f.write(f"closure_max_abs_dev_{name},{dev:.6e}\n")

    print(f"\nSaved figures and framework_results.csv to: {OUTDIR}")
