#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transport constants of the reduced theory across the SFT parameter space
(see docs/parameter_space.md).

The adjoint-kernel framework reduces the transport model to a handful of
derived constants. This script maps them over the (eta, u0) plane:

    a         linear dipole yield (dipole per unit cycle amplitude),
    lambda_R  pair-yield effectivity range (Gaussian width of the dipole
              yield of the ring-pair source versus emergence latitude),
    lambda_rev  sign-change latitude of the pair yield (domain boundary of
              positive multiplicative closures),
    r         cycle-to-cycle memory (survival of the dipole over one
              period), both profile-based and as the dominant eigenvalue
              of the one-period propagator.

Each parameter set costs about one second: one backward kernel recursion to
the 11-yr horizon (the operationally relevant one: the dipole is measured
at cycle end), one forward linear cycle, one source-free period, and one
matrix power for the propagator spectrum.

Scaling law. The advection-diffusion balance suggests
lambda_R ~ sqrt(eta/(2 u0 R)); the map tests this collapse across the
plane, and the ratio lambda_rev / lambda_R is tested for constancy.

The decay time tau enters ANALYTICALLY: the decay term -B/tau is
proportional to the identity and commutes with the transport, so

    K(tau) = e^{-(t_f - t)/tau} K(infinity)   (kernel, exactly),
    lambda_R(tau) = lambda_R(infinity)        (yield shape unchanged),
    a(tau) = sum_k omega_k e^{-(t_f - t_k)/tau},
    r(tau) = e^{-P/tau} r(infinity).

These identities are verified numerically at tau = 5 yr.

Outputs are saved to ./figures_parameter_space/.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

import nonlinear_response_sft_experiment as nre
import mathematical_framework_analysis as mfa
import multicycle_feedback_analysis as mca
import physical_validation_analysis as pva


OUTDIR = "figures_parameter_space"
SECONDS_PER_YEAR = 365.25 * 86400.0


def transport_constants(p: nre.SFTParams) -> dict:
    """Derived transport constants for one parameter set."""
    grid = nre.make_grid(p)
    t_yr = np.asarray(grid["t_yr"])
    N_P = len(t_yr)
    dt = float(grid["dt"])
    n = p.nmu

    A, M = mfa.build_operators(p, grid)
    Pstep, Minv, Pi = mca.build_fast_stepper(p, grid, A, M)
    w = mfa.dipole_weight(p, grid)

    # Kernel at the 11-yr horizon (vector recursion; no storage needed)
    RT = Minv @ (Pi @ (np.eye(n) + dt * A).T)
    K0 = Minv @ (Pi @ (w - w.mean()))
    for _ in range(N_P - 1):
        K0 = RT @ K0

    # Pair yield versus emergence latitude
    lat_deg = np.asarray(grid["lat_deg"])
    w_proj = w - w.mean()
    centres = np.linspace(2.0, 60.0, 117)
    f_pair = np.empty_like(centres)
    for i, lc in enumerate(centres):
        shape = pva.ring_pair_shape(lat_deg, float(lc), p.polarity_sep_deg, p.source_width_deg)
        f_pair[i] = (K0 @ shape) / (w_proj @ shape)

    neg = np.where(f_pair <= 0)[0]
    lam_rev = float(centres[neg[0]]) if len(neg) else np.nan

    hi_fit = 0.85 * lam_rev if np.isfinite(lam_rev) else 45.0
    sel = (centres >= 3.0) & (centres <= hi_fit) & (f_pair > 0)
    if np.count_nonzero(sel) >= 4:
        slope, _ = np.polyfit(centres[sel] ** 2, np.log(f_pair[sel]), 1)
        lamR = float(np.sqrt(-1.0 / (2.0 * slope))) if slope < 0 else np.nan
    else:
        lamR = np.nan

    # Forward linear cycle -> a; one source-free period -> r1
    PMshape, env, lam0 = mca.precompute_cycle_source(p, grid, Minv, Pi)
    amp_lin = p.source_amp * env / SECONDS_PER_YEAR
    B_end = mca.evolve_cycle(np.zeros(n), amp_lin, Pstep, PMshape, dt)
    a = float(w @ B_end)
    B_free = mca.evolve_cycle(B_end.copy(), np.zeros(N_P), Pstep, PMshape, dt)
    r1 = float((w @ B_free) / a)

    # Dominant mode of the one-period propagator
    G_P = np.linalg.matrix_power(Pstep, N_P)
    r_eig = float(np.max(np.abs(np.linalg.eigvals(G_P))))

    return {"a": a, "lamR": lamR, "lam_rev": lam_rev, "r1": r1, "r_eig": r_eig}


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    p_ref = nre.SFTParams()
    R_sun = p_ref.R_sun

    eta_vals = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0])
    u0_vals = np.array([5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 25.0])

    print(f"Scanning {len(eta_vals)} x {len(u0_vals)} parameter sets ...")
    shape = (len(u0_vals), len(eta_vals))
    a_map = np.full(shape, np.nan)
    lamR_map = np.full(shape, np.nan)
    lamrev_map = np.full(shape, np.nan)
    r1_map = np.full(shape, np.nan)
    reig_map = np.full(shape, np.nan)

    rows = []
    for i, u0 in enumerate(u0_vals):
        for j, eta in enumerate(eta_vals):
            p = replace(p_ref, u0_m_s=float(u0), eta_km2_s=float(eta))
            c = transport_constants(p)
            a_map[i, j] = c["a"]
            lamR_map[i, j] = c["lamR"]
            lamrev_map[i, j] = c["lam_rev"]
            r1_map[i, j] = c["r1"]
            reig_map[i, j] = c["r_eig"]
            rows.append((u0, eta, c["a"], c["lamR"], c["lam_rev"], c["r1"], c["r_eig"]))
        print(f"  u0 = {u0:5.1f} m/s done "
              f"(lamR = {lamR_map[i,0]:.1f} ... {lamR_map[i,-1]:.1f} deg)")

    # --- scaling-law collapse ------------------------------------------------
    ETA, U0 = np.meshgrid(eta_vals, u0_vals)
    lamR_est = np.rad2deg(np.sqrt(ETA * 1.0e10 / (2.0 * U0 * 100.0 * R_sun)))
    good = np.isfinite(lamR_map)
    coef = float(np.sum(lamR_map[good] * lamR_est[good]) / np.sum(lamR_est[good] ** 2))
    scatter = float(np.std(lamR_map[good] / lamR_est[good]))

    # Quadrature law: the measured pair yield is the transport effectivity
    # range convolved with the finite ring width, so
    # lambda_R^2 = C^2 eta/(2 u0 R) + w0^2.
    Adm = np.column_stack([lamR_est[good] ** 2, np.ones(np.count_nonzero(good))])
    (C2, w02), *_ = np.linalg.lstsq(Adm, lamR_map[good] ** 2, rcond=None)
    C_quad, w0_quad = float(np.sqrt(C2)), float(np.sqrt(w02))
    pred_quad = np.sqrt(C2 * lamR_est**2 + w02)
    scatter_quad = float(np.std(lamR_map[good] / pred_quad[good]))

    ratio_revR = lamrev_map / lamR_map
    ratio_mean = float(np.nanmean(ratio_revR))
    ratio_std = float(np.nanstd(ratio_revR))
    print(f"\nScaling law (proportional): lambda_R = C sqrt(eta/(2 u0 R)) with "
          f"C = {coef:.3f} +/- {scatter:.3f}")
    print(f"Scaling law (quadrature): lambda_R^2 = ({C_quad:.3f})^2 eta/(2 u0 R) "
          f"+ ({w0_quad:.2f} deg)^2, relative scatter {scatter_quad:.3f}")
    print(f"Sign-change ratio: lambda_rev / lambda_R = {ratio_mean:.2f} +/- {ratio_std:.2f}")
    print(f"r_eig across the plane: min {np.nanmin(reig_map):.6f}, "
          f"max {np.nanmax(reig_map):.6f} (steady mode, tau = infinity)")

    # --- analytic tau laws, verified at tau = 5 yr ---------------------------
    print("\nVerifying analytic tau laws at tau = 5 yr (reference u0, eta) ...")
    grid_ref = nre.make_grid(p_ref)
    t_yr = np.asarray(grid_ref["t_yr"])
    A_ref, M_ref = mfa.build_operators(p_ref, grid_ref)
    K_ref, w_ref = mfa.adjoint_kernel(p_ref, grid_ref, A_ref, M_ref)
    S_ref = mfa.linear_source_series(p_ref, grid_ref)
    omega_ref = mfa.yield_weights(grid_ref, K_ref, S_ref)
    c_inf = transport_constants(p_ref)

    tau = 5.0
    p_tau = replace(p_ref, tau_yr=tau)
    c_tau = transport_constants(p_tau)

    t_f = t_yr[-1]
    decay_w = np.exp(-(t_f - t_yr) / tau)
    # discrete decay is (1 - dt/tau) per step; correct the continuous factor
    dt_yr = float(grid_ref["dt"]) / SECONDS_PER_YEAR
    n_steps = (t_f - t_yr) / dt_yr
    decay_w_disc = (1.0 - dt_yr / tau) ** n_steps
    a_pred = float(np.sum(omega_ref * decay_w_disc))
    P_yr = t_yr[-1] + dt_yr
    r_pred = float((1.0 - dt_yr / tau) ** (P_yr / dt_yr)) * c_inf["r1"]

    print(f"  a:       measured {c_tau['a']:.6e}, predicted {a_pred:.6e} "
          f"(rel diff {abs(c_tau['a']-a_pred)/a_pred:.2e})")
    print(f"  r1:      measured {c_tau['r1']:.6f}, predicted {r_pred:.6f} "
          f"(rel diff {abs(c_tau['r1']-r_pred)/r_pred:.2e})")
    print(f"  lamR:    tau=inf {c_inf['lamR']:.3f} deg, tau=5yr {c_tau['lamR']:.3f} deg "
          f"(invariance to {abs(c_tau['lamR']-c_inf['lamR'])/c_inf['lamR']:.2e})")
    print(f"  lam_rev: tau=inf {c_inf['lam_rev']:.2f} deg, tau=5yr {c_tau['lam_rev']:.2f} deg")

    # r(tau) curve for the doc
    taus = np.array([2.5, 5.0, 10.0, 20.0, 40.0])
    r_curve = [float((1.0 - dt_yr / t) ** (P_yr / dt_yr)) * c_inf["r1"] for t in taus]

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    def contour_panel(ax, Z, title, cbar_label):
        cs = ax.contourf(ETA, U0, Z, levels=14)
        ax.contour(ETA, U0, Z, levels=cs.levels, colors="k", linewidths=0.4, alpha=0.5)
        ax.set_xlabel(r"Diffusivity $\eta$ [km$^2$ s$^{-1}$]")
        ax.set_ylabel(r"Flow amplitude $u_0$ [m s$^{-1}$]")
        ax.set_title(title)
        plt.colorbar(cs, ax=ax, label=cbar_label)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    contour_panel(axes[0, 0], lamR_map, "Pair-yield effectivity range", r"$\lambda_R$ [deg]")
    contour_panel(axes[0, 1], lamrev_map, "Sign-change latitude", r"$\lambda_{\rm rev}$ [deg]")
    contour_panel(axes[1, 0], r1_map, "Cycle memory (one period)", r"$r$")
    contour_panel(axes[1, 1], a_map, "Linear dipole yield", r"$a$")
    fig.suptitle(r"Transport constants of the reduced theory ($\tau = \infty$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figP1_transport_constant_maps.png"), dpi=200)
    plt.close(fig)

    # P2: scaling collapse
    plt.figure(figsize=(8, 5))
    plt.plot(lamR_est[good], lamR_map[good], "o", markersize=4,
             label="measured (adjoint kernel)")
    xs = np.linspace(0.0, np.nanmax(lamR_est) * 1.05, 200)
    plt.plot(xs, np.sqrt(C2 * xs**2 + w02), "--",
             label=(fr"quadrature law: $\lambda_R^2 = ({C_quad:.2f})^2\,\eta/(2u_0R)"
                    fr" + ({w0_quad:.1f}^\circ)^2$"))
    plt.plot(xs, xs, ":", linewidth=1, color="0.6", label="one-to-one (pure transport)")
    plt.xlabel(r"$\sqrt{\eta/(2u_0R_\odot)}$ [deg]")
    plt.ylabel(r"Measured $\lambda_R$ [deg]")
    plt.title("Scaling collapse of the effectivity range")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figP2_lambdaR_scaling.png"), dpi=200)
    plt.close()

    # P3: lambda_rev vs lambda_R
    plt.figure(figsize=(8, 5))
    plt.plot(lamR_map[good], lamrev_map[good], "o", markersize=4)
    xs = np.linspace(0.0, np.nanmax(lamR_map) * 1.05, 50)
    plt.plot(xs, ratio_mean * xs, "--",
             label=fr"$\lambda_{{\rm rev}} = ({ratio_mean:.2f} \pm {ratio_std:.2f})\,\lambda_R$")
    plt.xlabel(r"$\lambda_R$ [deg]")
    plt.ylabel(r"$\lambda_{\rm rev}$ [deg]")
    plt.title("Closure domain boundary versus effectivity range")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figP3_lambdarev_vs_lambdaR.png"), dpi=200)
    plt.close()

    # P4: analytic tau laws
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(taus, r_curve, "o--", label=r"$r(\tau) = e^{-P/\tau} r(\infty)$ (analytic)")
    axes[0].plot([tau], [c_tau["r1"]], "s", markersize=9, label=r"measured at $\tau = 5$ yr")
    axes[0].axhline(c_inf["r1"], linestyle=":", linewidth=1, label=r"$r(\infty)$")
    axes[0].set_xlabel(r"Decay time $\tau$ [yr]")
    axes[0].set_ylabel(r"Cycle memory $r$")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Memory versus decay time")
    axes[1].plot([0, 1], [c_inf["lamR"], c_tau["lamR"]], "o-")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([r"$\tau = \infty$", r"$\tau = 5$ yr"])
    axes[1].set_ylabel(r"$\lambda_R$ [deg]")
    axes[1].set_ylim(0, c_inf["lamR"] * 1.4)
    axes[1].set_title("Effectivity range is invariant under decay")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "figP4_tau_laws.png"), dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Tables
    # ---------------------------------------------------------------
    with open(os.path.join(OUTDIR, "transport_constants.csv"), "w", encoding="utf-8") as f:
        f.write("u0_m_s,eta_km2_s,a,lambda_R_deg,lambda_rev_deg,r1,r_eig\n")
        for u0, eta, a, lamR, lam_rev, r1, r_eig in rows:
            f.write(f"{u0:.1f},{eta:.1f},{a:.6e},{lamR:.4f},{lam_rev:.4f},"
                    f"{r1:.6f},{r_eig:.6f}\n")

    with open(os.path.join(OUTDIR, "parameter_space_results.csv"), "w", encoding="utf-8") as f:
        f.write("quantity,value\n")
        f.write(f"scaling_coefficient_C,{coef:.6f}\n")
        f.write(f"scaling_scatter,{scatter:.6f}\n")
        f.write(f"quadrature_C,{C_quad:.6f}\n")
        f.write(f"quadrature_w0_deg,{w0_quad:.6f}\n")
        f.write(f"quadrature_scatter,{scatter_quad:.6f}\n")
        f.write(f"lambda_rev_over_lambda_R_mean,{ratio_mean:.6f}\n")
        f.write(f"lambda_rev_over_lambda_R_std,{ratio_std:.6f}\n")
        f.write(f"r_eig_min,{np.nanmin(reig_map):.6f}\n")
        f.write(f"r_eig_max,{np.nanmax(reig_map):.6f}\n")
        f.write(f"tau5_a_measured,{c_tau['a']:.6e}\n")
        f.write(f"tau5_a_predicted,{a_pred:.6e}\n")
        f.write(f"tau5_r1_measured,{c_tau['r1']:.6f}\n")
        f.write(f"tau5_r1_predicted,{r_pred:.6f}\n")
        f.write(f"tau5_lamR_deg,{c_tau['lamR']:.4f}\n")
        f.write(f"tauinf_lamR_deg,{c_inf['lamR']:.4f}\n")

    print(f"\nSaved figures and tables to: {OUTDIR}")
