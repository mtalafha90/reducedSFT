#!/usr/bin/env python3
"""Settled-dipole closure of the multi-cycle LQ+IQ experiment.

Verifies the numbers added to Section sec:multicycle after the final audit.
Only the published reducedSFT modules are used (commit 41640e9): the
forward fast stepper of the multi-cycle experiment, the reference solver
grid and the discrete adjoint kernel. The configuration is the published
multi-cycle experiment: kernel-exact latitude efficiency times scalar
inflow efficiency, no tilt quenching, tau = infinity, 181 cells, one-day
update.

For a field B left at the end of a cycle, its settled dipole is the dipole
it will have after one further source-free cycle, w^T G B. If one cycle of
transport brings any field into the slowest dipole mode, the remnant term
of the exact recurrence D_n = s_n P(T_n) + D_rem,n obeys

    D_rem,n+1 = s_n P_s(T_n) + lambda_1 D_rem,n,

where P(T) = w^T b(T) and P_s(T) = w^T G b(T) are the end-of-cycle and
settled dipoles of a cycle's new flux b(T), and lambda_1 is the one-cycle
survival of an already settled field. The script checks that premise,
evaluates the two-variable map and compares it with the complete-field
cycle map and with long multi-cycle SFT runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.optimize import brentq, minimize_scalar

SOLVER_COMMIT = "41640e936c5d777a168ff5c0753491b840263c49"
MODULES = ("nonlinear_response_sft_experiment.py",
           "mathematical_framework_analysis.py",
           "multicycle_feedback_analysis.py")
GAINS = (2.05, 2.50, 3.20)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solver", type=Path, required=True,
                        help="reducedSFT checkout at commit 41640e9")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("settled_dipole.json"))
    args = parser.parse_args()
    solver = args.solver.resolve()
    commit = subprocess.run(["git", "-C", str(solver), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    assert commit == SOLVER_COMMIT, f"use reducedSFT commit {SOLVER_COMMIT}, not {commit}"
    sys.path.insert(0, str(solver))
    import nonlinear_response_sft_experiment as nre
    import mathematical_framework_analysis as mfa
    import multicycle_feedback_analysis as mc

    p = nre.SFTParams()
    grid = nre.make_grid(p)
    dt, n = float(grid["dt"]), p.nmu
    N = len(grid["t_yr"])
    A, M = mfa.build_operators(p, grid)
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    omega = mfa.yield_weights(grid, K, mfa.linear_source_series(p, grid))
    Qx, dlnQx, a = mc.make_Qx(p, grid, omega)
    Pstep, Minv, Pi = mc.build_fast_stepper(p, grid, A, M)
    PMshape, env, lam0 = mc.precompute_cycle_source(p, grid, Minv, Pi)
    G = np.linalg.matrix_power(Pstep, N)

    def new_flux(T):
        """End-of-cycle field of the LQ+IQ source of amplitude T, positive sign."""
        return mc.evolve_cycle(np.zeros(n), mc.cycle_amp_series(T, 1.0, p, env, lam0),
                               Pstep, PMshape, dt)

    # --- premise: one cycle of transport settles any field ---------------------
    eig = np.sort(np.abs(np.linalg.eigvals(G)))[::-1]
    w_s = G.T @ w                                   # settled-dipole weight
    k_start = (np.eye(n) + dt * A).T @ K[0]         # adjoint kernel, start of cycle
    kernel_identity = float(np.max(np.abs(w_s - k_start)) / np.max(np.abs(w_s)))
    b_lin = mc.evolve_cycle(np.zeros(n), p.source_amp * env / mc.SECONDS_PER_YEAR,
                            Pstep, PMshape, dt)
    r1_linear = float(w_s @ b_lin / (w @ b_lin))
    assert abs(r1_linear - 0.993962) < 5e-7          # published one-cycle coefficient
    test_fields = {"linear cycle": b_lin, "T = 0.5": new_flux(0.5),
                   "T = 1.5": new_flux(1.5),
                   "uniform polar caps": (np.abs(np.asarray(grid["lat_deg"])) > 60.0)
                   * np.sign(np.asarray(grid["mu"]))}
    second_cycle = {}
    for label, B in test_fields.items():
        GB = G @ B
        second_cycle[label] = {"first_cycle_survival": float(w @ GB / (w @ B)),
                               "second_cycle_survival": float(w @ (G @ GB) / (w @ GB))}
    lam1 = second_cycle["linear cycle"]["second_cycle_survival"]
    spread = max(abs(v["second_cycle_survival"] - lam1) for v in second_cycle.values())
    assert spread < 1e-8 and eig[1] < 1e-4
    r_star_linear = r1_linear / (1.0 + lam1 - r1_linear)

    # --- building blocks along the fixed-point family -------------------------
    h = 1e-4

    def blocks(T):
        b, bp, bm = new_flux(T), new_flux(T + h), new_flux(T - h)
        P, Ps = float(w @ b), float(w_s @ b)
        dP = float(w @ (bp - bm)) / (2 * h)
        dPs = float(w_s @ (bp - bm)) / (2 * h)
        k = T / (P - Ps / (1.0 + lam1))
        J2 = np.array([[k * dP, -k], [dPs, -lam1]])
        mu = np.linalg.eigvals(J2)
        return dict(T=T, b=b, db=(bp - bm) / (2 * h), P=P, Ps=Ps, dP=dP, dPs=dPs,
                    k=k, D0=a * k, mu=mu, lead=mu[np.argmax(np.abs(mu))],
                    rho=float(np.max(np.abs(mu))))

    def complete_field(T):
        """Fixed point and leading multiplier of the complete-field cycle map."""
        bl = blocks(T)
        Bstar = np.linalg.solve(np.eye(n) + G, bl["b"])
        k_full = T / float(w @ Bstar)
        J = -G + k_full * np.outer(bl["db"], w)      # fixed transport: dG/dT = 0
        ev = np.linalg.eigvals(J)
        return a * k_full, ev[np.argmax(np.abs(ev))], bl

    # duality: forward P(T) against the kernel-exact a T Q(T)
    duality = max(abs(blocks(T)["P"] / (a * T * Qx(T)) - 1.0) for T in (0.5, 1.0, 2.0))

    def T_at_gain(D0):
        return brentq(lambda T: blocks(T)["D0"] - D0, 0.05, 2.4, xtol=1e-12)

    # --- complete-field thresholds and multipliers -----------------------------
    T_pd = brentq(lambda T: blocks(T)["lead"].real + 1.0, 1.1, 1.5, xtol=1e-10)
    D0_pd, lead_pd, bl_pd = complete_field(T_pd)
    T_peak = brentq(lambda T: blocks(T)["dPs"], 1.0, 1.6, xtol=1e-10)
    D0_exist = 1.0 + r_star_linear

    rows, closure = [], []
    for D0 in GAINS:
        T = T_at_gain(D0)
        D0_full, lead_full, bl = complete_field(T)
        closure.append(abs(lead_full - bl["lead"]))
        rows.append({"D0": D0, "T_star": T, "D0_complete_field_check": D0_full,
                     "multiplier_two_variable": float(bl["lead"].real),
                     "multiplier_complete_field": float(lead_full.real),
                     "relaxation_cycles": float(-1.0 / np.log(bl["rho"]))})
    # smallest spectral radius: the complete field never relaxes in one step
    res = minimize_scalar(lambda T: blocks(T)["rho"], bounds=(0.7, 1.1),
                          method="bounded", options={"xatol": 1e-6})
    bl_min = blocks(res.x)
    closure.append(abs(complete_field(T_pd)[1] - bl_pd["lead"]))

    # --- published scalar map, reproduced -------------------------------------
    r = r1_linear
    T_s = brentq(lambda T: -T * dlnQx(T) - 2.0 / (1.0 + r), 0.5, 2.0, xtol=1e-12)
    D0_pd_scalar = (1.0 + r) / Qx(T_s)
    assert abs(D0_pd_scalar - 3.352157) < 2e-6
    scalar_rows = []
    for D0 in GAINS:
        T = brentq(lambda T: D0 * Qx(T) - (1.0 + r), 1e-6, 2.5, xtol=1e-14)
        m = 1.0 - (1.0 + r) * (-T * dlnQx(T))
        scalar_rows.append({"D0": D0, "T_star": T, "multiplier": m,
                            "relaxation_cycles": float(-1.0 / np.log(abs(m)))})
    T_zero = brentq(lambda T: -T * dlnQx(T) - 1.0 / (1.0 + r), 0.3, 1.5, xtol=1e-12)

    # --- nonlinear check: complete multi-cycle SFT against both maps ----------
    def two_variable_run(k, n_cycles):
        T, s, E, out = 1.0, 1.0, 0.0, []
        for _ in range(n_cycles):
            out.append(T)
            b = new_flux(T)
            D = s * float(w @ b) + E
            E = s * float(w_s @ b) + lam1 * E
            T, s = k * abs(D), -np.sign(D)
        return np.array(out)

    nonlinear = []
    for D0 in (3.20, 3.34, 3.60):
        k = D0 / a
        T_sft, _ = mc.run_multicycle_sft(k, 120, p, grid, Pstep, PMshape, env, lam0, w)
        T_two = two_variable_run(k, 120)
        T_map = mc.run_map_memory(k, 120, a, r, Qx)
        nonlinear.append({"D0": D0, "cycles": 120,
                          "max_abs_diff_two_variable": float(np.max(np.abs(T_two - T_sft))),
                          "max_abs_diff_scalar_map": float(np.max(np.abs(T_map - T_sft))),
                          "sft_last_four": [float(x) for x in T_sft[-4:]],
                          "two_variable_last_four": [float(x) for x in T_two[-4:]],
                          "scalar_map_last_four": [float(x) for x in T_map[-4:]]})

    out = {
        "solver_commit": commit,
        "source_sha256": {m: sha256(solver / m) for m in MODULES},
        "configuration": "Published multi-cycle experiment: kernel-exact latitude "
                         "efficiency times scalar inflow efficiency, no tilt quenching, "
                         "tau = infinity, 181 cells, one-day update.",
        "a": a,
        "forward_vs_kernel_P_max_rel_diff": duality,
        "propagator_eigenvalues_abs_leading_three": [float(x) for x in eig[:3]],
        "settled_weight_vs_start_of_cycle_kernel_max_rel_diff": kernel_identity,
        "test_fields": second_cycle,
        "lambda_1": lam1,
        "r1_linear": r1_linear,
        "r_star_linear": r_star_linear,
        "existence_gain_complete_field": D0_exist,
        "existence_gain_scalar_map": 1.0 + r,
        "period_doubling": {"T_star": T_pd, "D0_complete_field": D0_pd,
                            "complete_field_multiplier_there": float(lead_pd.real),
                            "D0_scalar_map": D0_pd_scalar,
                            "relative_difference": D0_pd_scalar / D0_pd - 1.0,
                            "T_peak_of_settled_dipole": T_peak},
        "fixed_points_complete_field": rows,
        "fixed_points_scalar_map": scalar_rows,
        "smallest_spectral_radius": {"value": bl_min["rho"], "T_star": float(res.x),
                                     "D0": bl_min["D0"],
                                     "relaxation_cycles": float(-1.0 / np.log(bl_min["rho"]))},
        "scalar_map_zero_multiplier_D0": (1.0 + r) / Qx(T_zero),
        "two_variable_vs_complete_field_multiplier_max_abs_diff": float(max(closure)),
        "nonlinear_check": nonlinear,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("r1_linear", "lambda_1", "r_star_linear",
                                           "existence_gain_complete_field",
                                           "period_doubling",
                                           "two_variable_vs_complete_field_multiplier_max_abs_diff")},
                     indent=2))
    for row in rows:
        print(row)
    print(out["smallest_spectral_radius"])
    for row in nonlinear:
        print({k: row[k] for k in ("D0", "max_abs_diff_two_variable", "max_abs_diff_scalar_map")})


if __name__ == "__main__":
    main()
