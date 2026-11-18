#!/usr/bin/env python3
"""Recover the eight manuscript plots without changing the scientific model.

Requires reducedSFT at 41640e936c5d777a168ff5c0753491b840263c49.
Run with OPENBLAS_NUM_THREADS=1; see README.md for the complete command.

The single-cycle scan table is read from the solver checkout
(figures_nonlinear_response_v2/nonlinear_response_results_v2.csv). It is
byte-identical (SHA-256 2fa2dade...) to the copy in the manuscript
repository's figures/ folder, which this script read before revisions/ moved
to the code repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, type=Path)
    args = parser.parse_args()
    solver = args.solver.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]          # repository root that holds revisions/
    sys.path.insert(0, str(solver))
    import nonlinear_response_sft_experiment as nre
    import mathematical_framework_analysis as mfa
    import physical_validation_analysis as pva
    import multicycle_feedback_analysis as mca

    started = time.perf_counter()
    checks = {"solver_commit": "41640e936c5d777a168ff5c0753491b840263c49"}
    source_paths = [solver / f for f in (
        "nonlinear_response_sft_experiment.py", "mathematical_framework_analysis.py",
        "physical_validation_analysis.py", "multicycle_feedback_analysis.py",
        "figures_parameter_space/transport_constants.csv",
        "figures_parameter_space/parameter_space_results.csv",
        "figures_multicycle/multicycle_results.csv",
        "figures_validation/validation_results.csv",
        "figures_framework/framework_results.csv")]
    scan_path = solver / "figures_nonlinear_response_v2/nonlinear_response_results_v2.csv"
    grid_path = repo / "revisions/point5/grid_sensitivity_results.csv"
    checks["input_sha256"] = {
        ("reducedSFT/" + str(f.relative_to(solver))): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source_paths + [scan_path]}
    checks["input_sha256"][str(grid_path.relative_to(repo))] = (
        hashlib.sha256(grid_path.read_bytes()).hexdigest())

    p = nre.SFTParams()
    scan = pd.read_csv(scan_path)
    cases = ["linear", "tilt", "latitude", "inflow", "combined"]
    T = scan.loc[scan.case == "linear", "T"].to_numpy()
    data = {"T": T, "cases": np.array(cases), "Tfine": np.linspace(T.min(), T.max(), 500)}
    assert len(T) == 13
    for case in cases:
        d = scan.loc[scan.case == case]
        np.testing.assert_array_equal(d["T"].to_numpy(), T)
        for col in ("P_signed_main", "P_hat", "R_hat"):
            data[f"{case}_{col}"] = d[col].to_numpy()
        data[f"{case}_theory"] = nre.theory_normalized(data["Tfine"], p, case)

    baseline = pd.read_csv(grid_path)
    baseline = baseline[(baseline.nmu == 181) & (baseline.dt_days == 1.)
                        & (baseline.protocol == "legacy")]

    def original_D(case):
        rows = baseline[baseline.case == case].sort_values("T")
        np.testing.assert_array_equal(rows["T"].to_numpy(), T)
        return rows.D.to_numpy()

    # Point 5's batched baseline was checked against the original forward solver.
    checks["archived_scan_vs_point5_max_abs_D"] = max(
        float(np.max(np.abs(original_D(c) - data[f"{c}_P_signed_main"])))
        for c in cases)
    assert checks["archived_scan_vs_point5_max_abs_D"] < 5e-10

    print("Recovering adjoint curves ...", flush=True)
    grid = nre.make_grid(p)
    A, M = mfa.build_operators(p, grid)
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    S = mfa.linear_source_series(p, grid)
    omega = mfa.yield_weights(grid, K, S)
    ladder = mfa.closure_ladder(p, grid, omega, T)
    for name, values in ladder.items():
        data["ladder_" + name] = values
    a = float(omega.sum())
    checks["a"] = a
    checks["adjoint_vs_baseline_linear_relative_error"] = abs(a / original_D("linear")[5] - 1.)
    assert checks["adjoint_vs_baseline_linear_relative_error"] < 1e-11
    checks["closure_max_abs_errors"] = {
        name: float(np.max(np.abs(ladder[name] - data["latitude_P_hat"])))
        for name in ("exact", "yield_weighted", "envelope", "static")}

    K_asym = mfa.extended_kernel(p, grid, A, M, K[0], extra_years=40.)
    _, _, lamR, _ = pva.pair_yield_function(p, grid, K_asym, w)
    data["shift_Q"] = original_D("shift") / (a * T)
    data["shift_exact_Q"] = np.array([
        pva.exact_kernel_prediction(p, grid, K, float(t), "latitude-shift")
        for t in T]) / (a * T)
    for name, width in (("derived", lamR), ("prescribed", p.lambda_R_deg)):
        data[f"shift_{name}_Q"] = pva.gaussian_shift_closure(
            T, p, width, float(ladder["t_mean_w"][0]), float(ladder["t_std_w"][0]))
    checks["lambda_R_pair_deg"] = lamR
    checks["shift_exact_kernel_max_rel_err"] = float(np.max(np.abs(
        data["shift_exact_Q"] / data["shift_Q"] - 1.)))
    assert checks["shift_exact_kernel_max_rel_err"] < 1e-11

    validation = pd.read_csv(solver / "figures_validation/validation_results.csv").set_index("quantity").value
    assert abs(lamR - validation["lambda_R_pair_deg"]) < 5e-7
    for speed in (2, 5):
        Q = original_D(f"physical_{speed}") / (a * T)
        good = (Q > 0) & (T <= 1.2)
        b = float(np.sum((1. / Q[good] - 1.) * T[good]**2) / np.sum(T[good]**4))
        data[f"inflow{speed}_D_over_a"] = T * Q
        data[f"inflow{speed}_rational"] = T / (1. + b * T**2)
        checks[f"inflow{speed}_b_fit_leading"] = b
        assert abs(b - validation[f"inflow{speed}_b_fit_leading"]) < 5e-7

    transport = pd.read_csv(solver / "figures_parameter_space/transport_constants.csv")
    x = np.rad2deg(np.sqrt(transport.eta_km2_s.to_numpy() * 1e10 /
                          (2. * transport.u0_m_s.to_numpy() * 100. * p.R_sun)))
    y = transport.lambda_R_deg.to_numpy()
    C2, w02 = np.linalg.lstsq(np.column_stack((x*x, np.ones(len(x)))), y*y, rcond=None)[0]
    checks["transport_refit_from_rounded_C"] = float(np.sqrt(C2))
    checks["transport_refit_from_rounded_w0_deg"] = float(np.sqrt(w02))
    summary = pd.read_csv(solver / "figures_parameter_space/parameter_space_results.csv").set_index("quantity").value
    # Widths were archived to four decimal places. Propagate that rounding
    # through the linear least-squares fit of squared widths, then compare
    # with the separately archived (six-decimal) original coefficients.
    rounding = np.abs(np.linalg.pinv(np.column_stack((x*x, np.ones(len(x)))))) @ (2*y*5e-5 + (5e-5)**2)
    C, w0 = float(summary["quadrature_C"]), float(summary["quadrature_w0_deg"])
    assert abs(C2 - C*C) <= rounding[0] + 2*C*5e-7 + (5e-7)**2
    assert abs(w02 - w0*w0) <= rounding[1] + 2*w0*5e-7 + (5e-7)**2
    checks["transport_C"] = C
    checks["transport_w0_deg"] = w0
    checks["transport_squared_coefficient_rounding_bounds"] = rounding.tolist()
    data["transport_x"] = x
    data["transport_y"] = y
    data["transport_curve_x"] = np.linspace(0., x.max() * 1.05, 200)
    data["transport_curve_y"] = np.sqrt(C*C * data["transport_curve_x"]**2 + w0*w0)
    checks["transport_points"] = len(x)
    assert len(x) == 63

    # Exactly the original Figure M3 algorithm, gain grid and retained cycles.
    print("Recovering the original 34-gain bifurcation scan ...", flush=True)
    Qx, dlnQ, a_map = mca.make_Qx(p, grid, omega)
    Pstep, Minv, Pi = mca.build_fast_stepper(p, grid, A, M)
    PMshape, env, lam0 = mca.precompute_cycle_source(p, grid, Minv, Pi)
    dt = float(grid["dt"])
    Blin = mca.evolve_cycle(np.zeros(p.nmu), p.source_amp * env / mca.SECONDS_PER_YEAR,
                            Pstep, PMshape, dt)
    Bfree = mca.evolve_cycle(Blin.copy(), np.zeros(len(env)), Pstep, PMshape, dt)
    r = float((w @ Bfree) / (w @ Blin))
    d0mem = mca.period_doubling_gain(r, Qx, dlnQ)
    d0nomem = mca.period_doubling_gain(0., Qx, dlnQ)
    checks.update(r_first_period=r, D0_period_doubling_memory=d0mem,
                  D0_period_doubling_memoryless=d0nomem,
                  fast_stepper_vs_baseline_relative_error=float(abs((w @ Blin) / original_D("linear")[5] - 1.)))
    assert checks["fast_stepper_vs_baseline_relative_error"] < 1e-11
    multicycle = pd.read_csv(solver / "figures_multicycle/multicycle_results.csv").set_index("quantity").value
    for key in ("r_first_period", "D0_period_doubling_memory", "D0_period_doubling_memoryless"):
        assert abs(checks[key] - multicycle[key]) < 5e-7, key
    gains = np.linspace(1.05 * (1. + r), 2.2 * d0mem, 34)
    sft, maps = [], []
    for i, gain in enumerate(gains):
        Ts, _ = mca.run_multicycle_sft(gain / a_map, 46, p, grid, Pstep, PMshape, env, lam0, w)
        Tm = mca.run_map_memory(gain / a_map, 400, a_map, r, Qx)
        sft.append(Ts[-16:])
        maps.append(Tm[-60:])
        print(f"Gain {i+1}/34 complete; elapsed {time.perf_counter()-started:.1f} s", flush=True)
    data.update(bifurcation_gains=gains, bifurcation_sft=np.array(sft), bifurcation_map=np.array(maps))
    for name, values in data.items():
        if np.issubdtype(values.dtype, np.number):
            assert np.all(np.isfinite(values)), name
    np.savez_compressed(here / "figure_data.npz", **data)
    checks["runtime_seconds"] = time.perf_counter() - started
    checks["figure_data_sha256"] = hashlib.sha256((here / "figure_data.npz").read_bytes()).hexdigest()
    checks["numpy_version"] = np.__version__
    (here / "numerical_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print("Saved figure_data.npz and numerical_checks.json", flush=True)


if __name__ == "__main__":
    main()
