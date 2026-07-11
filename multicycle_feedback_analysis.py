#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-cycle SFT simulations with dynamo feedback
(see docs/multicycle_validation.md).

The reduced cycle map T_{n+1} = D0 T_n Q(T_n) assumes memoryless
cycle-to-cycle coupling: each cycle builds its dipole from a clean slate.
In the actual SFT with tau = infinity, the polar field left by cycle n-1
survives into cycle n, so the end-of-cycle dipole obeys the EXACT recurrence
(by linearity of the transport)

    D_n = s_n a T_n Q(T_n)  +  w^T G_P B_{n-1},

where s_n is the source polarity of cycle n (Hale alternation), the first
term is the new-flux contribution (exact, from the adjoint kernel), G_P is
the one-period propagator, and B_{n-1} the field profile at the end of the
previous cycle. Collapsing the functional memory term to a scalar,

    w^T G_P B_{n-1} ~ r D_{n-1},

yields the one-step MEMORY-CORRECTED map: with T_n = k |D_{n-1}| and
successful polarity reversal,

    T_{n+1} = T_n [ D0 Q(T_n) - r ],        D0 = k a ,

whose fixed point satisfies D0 Q(T*) = 1 + r (existence: D0 > 1 + r) and
whose multiplier is M'(T*) = 1 - (1 + r) q(T*): period doubling at

    q(T*) = 2 / (1 + r)

instead of the memoryless q(T*) = 2.

This script measures r (three ways), runs long multi-cycle SFT sequences
with the feedback T_{n+1} = k |D_n|, s_{n+1} = -sign(D_n), and compares the
resulting dynamics (fixed points, period-doubling onset, bifurcation
structure, one-step predictability) against the memoryless and
memory-corrected maps. The quenching used is the combined efficiency-mode
latitude + inflow factor (tilt is omitted so the source SHAPE is
amplitude-independent and the per-cycle source series can be precomputed);
Q(T) is evaluated kernel-exactly, so the ONLY difference between the SFT
and the maps is cycle-to-cycle memory.

Outputs are saved to ./figures_multicycle/.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

import nonlinear_response_sft_experiment as nre
import mathematical_framework_analysis as mfa


OUTDIR = "figures_multicycle"
SECONDS_PER_YEAR = 365.25 * 86400.0


# ============================================================
# 1. Fast matrix propagator (exactly the solver's update)
# ============================================================

def build_fast_stepper(p: nre.SFTParams, grid, A: np.ndarray, M: np.ndarray):
    """One-step matrix P and pre-projected source basis.

    The solver's update is B^{k+1} = Pi Minv[(I + dt A) B^k + dt S^k], so
    with Pstep = Pi Minv (I + dt A) and src_k = Pi Minv shape_k the update
    is B <- Pstep B + dt amp_k src_k, exactly.
    """
    n = p.nmu
    dt = float(grid["dt"])
    Minv = np.linalg.inv(M)
    Pi = np.eye(n) - np.ones((n, n)) / n
    Pstep = Pi @ (Minv @ (np.eye(n) + dt * A))
    return Pstep, Minv, Pi


def precompute_cycle_source(p: nre.SFTParams, grid, Minv: np.ndarray, Pi: np.ndarray):
    """Pre-projected spatial source basis and per-phase scalars for one cycle.

    The shape (four rings at the drifting belt latitude, fixed separation)
    and the envelope depend only on cycle phase; amplitude factors are
    applied per cycle. Returns PMshape[k] = Pi Minv shape_k, the envelope
    series, and the belt-latitude series.
    """
    t_yr = np.asarray(grid["t_yr"])
    lat_deg = np.asarray(grid["lat_deg"])
    N_P = len(t_yr)
    n = p.nmu

    PMshape = np.empty((N_P, n))
    env = np.empty(N_P)
    lam0 = np.empty(N_P)
    width = p.source_width_deg
    sep = p.polarity_sep_deg
    for k, t in enumerate(t_yr):
        phase = np.clip(t / p.cycle_years, 0.0, 1.0)
        lam0[k] = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
        env[k] = float(nre.cycle_envelope(float(t), p))
        NF = np.exp(-0.5 * ((lat_deg - (lam0[k] + sep / 2.0)) / width) ** 2)
        NL = np.exp(-0.5 * ((lat_deg - (lam0[k] - sep / 2.0)) / width) ** 2)
        SF = np.exp(-0.5 * ((lat_deg - (-lam0[k] - sep / 2.0)) / width) ** 2)
        SL = np.exp(-0.5 * ((lat_deg - (-lam0[k] + sep / 2.0)) / width) ** 2)
        shape = (NF - NL) - (SF - SL)
        shape = shape - shape.mean()
        PMshape[k] = Pi @ (Minv @ shape)
    return PMshape, env, lam0


def cycle_amp_series(T: float, sign: float, p: nre.SFTParams,
                     env: np.ndarray, lam0: np.ndarray) -> np.ndarray:
    """Per-phase scalar amplitude of the source for one cycle.

    Combined efficiency-mode quenching without tilt: the instantaneous
    latitude factor (belt mode) times the inflow factor.
    """
    delta = p.b_LQ * T**2
    qlq = np.exp(-((2.0 * lam0 * delta + delta**2) / (2.0 * p.lambda_R_deg**2)))
    qi = 1.0 / (1.0 + p.b_I * T**2)
    return sign * p.source_amp * T * env * qlq * qi / SECONDS_PER_YEAR


def evolve_cycle(B: np.ndarray, amp: np.ndarray, Pstep: np.ndarray,
                 PMshape: np.ndarray, dt: float) -> np.ndarray:
    """Advance the field through one full cycle."""
    for k in range(len(amp)):
        B = Pstep @ B + (dt * amp[k]) * PMshape[k]
    return B


# ============================================================
# 2. Kernel-exact quenching function and map thresholds
# ============================================================

def make_Qx(p: nre.SFTParams, grid, omega: np.ndarray):
    """Kernel-exact Q(T) for the latitude+inflow quenching used here.

    Q(T) = [sum_k omega_k Q_LQ(t_k, T)] / a * Q_I(T): the exact new-flux
    dipole per unit a T. Also returns the analytic d lnQ/dT.
    """
    t_yr = np.asarray(grid["t_yr"])
    phase = np.clip(t_yr / p.cycle_years, 0.0, 1.0)
    lam0 = p.base_lat_deg + p.lat_drift_deg * (1.0 - phase)
    a = float(omega.sum())
    lamR2 = p.lambda_R_deg**2

    def Qx(T: float) -> float:
        delta = p.b_LQ * T**2
        qlq = np.exp(-((2.0 * lam0 * delta + delta**2) / (2.0 * lamR2)))
        return float(np.sum(omega * qlq) / a / (1.0 + p.b_I * T**2))

    def dlnQx_dT(T: float) -> float:
        delta = p.b_LQ * T**2
        qlq = np.exp(-((2.0 * lam0 * delta + delta**2) / (2.0 * lamR2)))
        dexp = -2.0 * p.b_LQ * T * (lam0 + delta) / lamR2
        term_lq = float(np.sum(omega * qlq * dexp) / np.sum(omega * qlq))
        term_i = -2.0 * p.b_I * T / (1.0 + p.b_I * T**2)
        return term_lq + term_i

    return Qx, dlnQx_dT, a


def fixed_point_and_stiffness(D0: float, r: float, Qx, dlnQx_dT) -> tuple[float, float]:
    """T* with D0 Qx(T*) = 1 + r, and q* = -T* dlnQx/dT(T*)."""
    target = np.log((1.0 + r) / D0)
    lo, hi = 1e-6, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        with np.errstate(divide="ignore"):
            above = np.log(Qx(mid)) > target
        if above:
            lo = mid
        else:
            hi = mid
    Tstar = 0.5 * (lo + hi)
    return Tstar, -Tstar * dlnQx_dT(Tstar)


def period_doubling_gain(r: float, Qx, dlnQx_dT) -> float:
    """Gain D0 at which q(T*) = 2/(1 + r) (memoryless case: r = 0)."""
    lo, hi = (1.0 + r) * 1.0001, 1e4
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        _, q = fixed_point_and_stiffness(mid, r, Qx, dlnQx_dT)
        if q < 2.0 / (1.0 + r):
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ============================================================
# 3. Multi-cycle SFT driver and reduced maps
# ============================================================

def run_multicycle_sft(k_gain: float, n_cycles: int, p, grid, Pstep, PMshape,
                       env, lam0, w: np.ndarray, T0: float = 1.0):
    """Consecutive SFT cycles with feedback T_{n+1} = k |D_n|, Hale alternation."""
    dt = float(grid["dt"])
    B = np.zeros(p.nmu)
    T_seq, D_seq = [], []
    T, sign = T0, 1.0
    for _ in range(n_cycles):
        amp = cycle_amp_series(T, sign, p, env, lam0)
        B = evolve_cycle(B, amp, Pstep, PMshape, dt)
        D = float(w @ B)
        T_seq.append(T)
        D_seq.append(D)
        T = k_gain * abs(D)
        sign = -np.sign(D) if D != 0.0 else -sign
    return np.array(T_seq), np.array(D_seq)


def run_map_memory(k_gain: float, n_cycles: int, a: float, r: float, Qx,
                   T0: float = 1.0):
    """Signed one-step-memory map: D_n = s_n a T_n Qx(T_n) + r D_{n-1}."""
    T_seq = []
    T, sign, D_prev = T0, 1.0, 0.0
    for _ in range(n_cycles):
        D = sign * a * T * Qx(T) + r * D_prev
        T_seq.append(T)
        T = k_gain * abs(D)
        sign = -np.sign(D) if D != 0.0 else -sign
        D_prev = D
    return np.array(T_seq)


def run_map_memoryless(D0: float, n_cycles: int, Qx, T0: float = 1.0):
    T_seq = []
    T = T0
    for _ in range(n_cycles):
        T_seq.append(T)
        T = D0 * T * Qx(T)
    return np.array(T_seq)


# ============================================================
# 4. Main analysis
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    p = nre.SFTParams()
    grid = nre.make_grid(p)
    dt = float(grid["dt"])
    t_yr = np.asarray(grid["t_yr"])
    N_P = len(t_yr)

    print("Building operators, kernel, and fast stepper ...")
    A, M = mfa.build_operators(p, grid)
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    S_lin = mfa.linear_source_series(p, grid)
    omega = mfa.yield_weights(grid, K, S_lin)
    Qx, dlnQx_dT, a = make_Qx(p, grid, omega)

    Pstep, Minv, Pi = build_fast_stepper(p, grid, A, M)
    PMshape, env, lam0 = precompute_cycle_source(p, grid, Minv, Pi)

    # --- stepper validation against the reference solver -----------------
    amp_lin = p.source_amp * 1.0 * env / SECONDS_PER_YEAR
    B_lin = evolve_cycle(np.zeros(p.nmu), amp_lin, Pstep, PMshape, dt)
    D_fast = float(w @ B_lin)
    _, D_ref, _, _, _ = nre.run_sft_case(1.0, "linear", p)
    print(f"Fast stepper vs reference solver (linear cycle): "
          f"relative difference {abs(D_fast - D_ref)/abs(D_ref):.3e}")

    # --- memory parameter r ------------------------------------------------
    # (i) source-free survival of the end-of-cycle profile over successive
    # periods; (ii) dominant eigenvalue of the one-period propagator.
    B = B_lin.copy()
    D_prev = float(w @ B)
    r_seq = []
    for _ in range(4):
        B = evolve_cycle(B, np.zeros(N_P), Pstep, PMshape, dt)
        D_now = float(w @ B)
        r_seq.append(D_now / D_prev)
        D_prev = D_now
    r1, r_inf = r_seq[0], r_seq[-1]

    G_P = np.linalg.matrix_power(Pstep, N_P)
    eigvals = np.linalg.eigvals(G_P)
    r_eig = float(np.max(np.abs(eigvals)))
    print(f"Memory parameter: r(first period) = {r1:.4f}, "
          f"r(asymptotic) = {r_inf:.4f}, dominant eigenvalue of G_P = {r_eig:.4f}")
    r = r1

    # --- analytic thresholds ------------------------------------------------
    D0_exist = 1.0 + r
    D0_pd_nomem = period_doubling_gain(0.0, Qx, dlnQx_dT)
    D0_pd_mem = period_doubling_gain(r, Qx, dlnQx_dT)
    print(f"Existence threshold with memory: D0 > 1 + r = {D0_exist:.3f}")
    print(f"Period-doubling gain: memoryless map {D0_pd_nomem:.3f}, "
          f"memory-corrected map {D0_pd_mem:.3f}")

    # --- time series at three gains ----------------------------------------
    n_show = 60
    D0_show = [0.6 * D0_pd_mem, 1.1 * D0_pd_mem, 1.6 * D0_pd_mem]
    series = []
    for D0 in D0_show:
        k_gain = D0 / a
        T_sft, D_sft = run_multicycle_sft(k_gain, n_show, p, grid, Pstep,
                                          PMshape, env, lam0, w)
        T_mapB = run_map_memory(k_gain, n_show, a, r, Qx)
        T_mapA = run_map_memoryless(D0, n_show, Qx)
        series.append((D0, T_sft, T_mapB, T_mapA))
        print(f"D0 = {D0:.2f}: SFT mean T (last 20) = {np.mean(T_sft[-20:]):.3f}, "
              f"memory map {np.mean(T_mapB[-20:]):.3f}, "
              f"memoryless map {np.mean(T_mapA[-20:]):.3f}")

    # --- one-step predictability along the SFT sequence ---------------------
    # At a stable gain and at a modulated gain, predict T_{n+1} from the
    # SFT's own (T_n, D_{n-1}) with each map.
    def one_step_errors(D0: float, n_cyc: int = 50) -> tuple[float, float]:
        k_gain = D0 / a
        T_sft, D_sft = run_multicycle_sft(k_gain, n_cyc, p, grid, Pstep,
                                          PMshape, env, lam0, w)
        errA, errB = [], []
        for n in range(1, n_cyc - 1):
            predA = D0 * T_sft[n] * Qx(T_sft[n])
            s_n = -np.sign(D_sft[n - 1])
            D_predB = s_n * a * T_sft[n] * Qx(T_sft[n]) + r * D_sft[n - 1]
            predB = k_gain * abs(D_predB)
            errA.append(abs(predA - T_sft[n + 1]))
            errB.append(abs(predB - T_sft[n + 1]))
        scale = float(np.mean(T_sft[1:]))
        return float(np.mean(errA) / scale), float(np.mean(errB) / scale)

    errA_stable, errB_stable = one_step_errors(0.6 * D0_pd_mem)
    errA_mod, errB_mod = one_step_errors(1.6 * D0_pd_mem)
    print(f"One-step prediction error (stable gain): memoryless {errA_stable:.3f}, "
          f"memory-corrected {errB_stable:.3f}")
    print(f"One-step prediction error (modulated gain): memoryless {errA_mod:.3f}, "
          f"memory-corrected {errB_mod:.3f}")

    # --- bifurcation scan ----------------------------------------------------
    print("\nRunning multi-cycle bifurcation scan ...")
    D0_scan = np.linspace(1.05 * D0_exist, 2.2 * D0_pd_mem, 34)
    n_cyc, n_keep = 46, 16
    sft_x, sft_y = [], []
    mapB_x, mapB_y = [], []
    for D0 in D0_scan:
        k_gain = D0 / a
        T_sft, _ = run_multicycle_sft(k_gain, n_cyc, p, grid, Pstep,
                                      PMshape, env, lam0, w)
        sft_x.extend([D0] * n_keep)
        sft_y.extend(T_sft[-n_keep:])
        T_mapB = run_map_memory(k_gain, 400, a, r, Qx)
        mapB_x.extend([D0] * 60)
        mapB_y.extend(T_mapB[-60:])

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    # M1: source-free dipole survival
    plt.figure(figsize=(8, 5))
    periods = np.arange(len(r_seq) + 1)
    D_decay = [1.0]
    for rr in r_seq:
        D_decay.append(D_decay[-1] * rr)
    plt.semilogy(periods, D_decay, "o-", label="source-free dipole survival")
    plt.semilogy(periods, r_eig**periods, "--",
                 label=fr"dominant mode, $r_{{\rm eig}} = {r_eig:.3f}$")
    plt.xlabel("Elapsed periods (11 yr each)")
    plt.ylabel(r"$D(nP)/D(0)$")
    plt.title("Cycle-to-cycle memory: dipole survival without sources")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figM1_memory_parameter.png"), dpi=200)
    plt.close()

    # M2: time series at three gains
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, (D0, T_sft, T_mapB, T_mapA) in zip(axes, series):
        ax.plot(T_sft, "o-", markersize=3, label="multi-cycle SFT")
        ax.plot(T_mapB, "s--", markersize=3, alpha=0.8, label="memory-corrected map")
        ax.plot(T_mapA, "^:", markersize=3, alpha=0.8, label="memoryless map")
        ax.set_ylabel(r"$T_n$")
        ax.set_title(fr"$D_0 = {D0:.2f}$")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Cycle number $n$")
    fig.suptitle("Amplitude sequences: SFT with feedback vs reduced maps")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figM2_time_series.png"), dpi=200)
    plt.close(fig)

    # M3: bifurcation comparison
    plt.figure(figsize=(9, 6))
    plt.plot(mapB_x, mapB_y, ",", color="0.6", label="memory-corrected map attractor")
    plt.plot(sft_x, sft_y, "o", markersize=3, label="multi-cycle SFT attractor")
    plt.axvline(D0_pd_mem, linestyle="--", linewidth=1.2, color="C3",
                label=fr"analytic PD (memory), $D_0 = {D0_pd_mem:.2f}$")
    plt.axvline(D0_pd_nomem, linestyle=":", linewidth=1.2, color="C2",
                label=fr"analytic PD (memoryless), $D_0 = {D0_pd_nomem:.2f}$")
    plt.xlabel(r"Linear gain $D_0 = k\,a$")
    plt.ylabel(r"Attractor of $T_n$")
    plt.title("Multi-cycle SFT vs reduced maps: bifurcation structure")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figM3_bifurcation_comparison.png"), dpi=200)
    plt.close()

    # M4: return map at a modulated gain
    D0_ret = 1.6 * D0_pd_mem
    k_gain = D0_ret / a
    T_sft, _ = run_multicycle_sft(k_gain, 80, p, grid, Pstep, PMshape, env, lam0, w)
    Tg = np.linspace(0.01, max(T_sft) * 1.15, 300)
    mapA_curve = np.array([D0_ret * t * Qx(t) for t in Tg])
    mapB_curve = np.array([t * max(D0_ret * Qx(t) - r, 0.0) for t in Tg])
    plt.figure(figsize=(8, 5))
    plt.plot(T_sft[10:-1], T_sft[11:], "o", markersize=4, label="multi-cycle SFT")
    plt.plot(Tg, mapA_curve, ":", label="memoryless map")
    plt.plot(Tg, mapB_curve, "--", label="memory-corrected map (reversal branch)")
    plt.plot(Tg, Tg, "-", linewidth=0.8, color="0.7")
    plt.xlabel(r"$T_n$")
    plt.ylabel(r"$T_{n+1}$")
    plt.title(fr"Return map at $D_0 = {D0_ret:.2f}$")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figM4_return_map.png"), dpi=200)
    plt.close()

    # ---------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------
    with open(os.path.join(OUTDIR, "multicycle_results.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        f.write(f"stepper_validation_rel_diff,{abs(D_fast - D_ref)/abs(D_ref):.6e}\n")
        f.write(f"r_first_period,{r1:.6f}\n")
        f.write(f"r_asymptotic,{r_inf:.6f}\n")
        f.write(f"r_eigenvalue,{r_eig:.6f}\n")
        f.write(f"D0_existence_with_memory,{D0_exist:.6f}\n")
        f.write(f"D0_period_doubling_memoryless,{D0_pd_nomem:.6f}\n")
        f.write(f"D0_period_doubling_memory,{D0_pd_mem:.6f}\n")
        f.write(f"one_step_err_memoryless_stable,{errA_stable:.6f}\n")
        f.write(f"one_step_err_memory_stable,{errB_stable:.6f}\n")
        f.write(f"one_step_err_memoryless_modulated,{errA_mod:.6f}\n")
        f.write(f"one_step_err_memory_modulated,{errB_mod:.6f}\n")

    print(f"\nSaved figures and multicycle_results.csv to: {OUTDIR}")
