#!/usr/bin/env python3
"""Reproduce the Point 5 resolution study without changing the published solver.

Requires numpy, scipy and the matplotlib dependency of reducedSFT. Run:
  python grid_sensitivity.py --code-repo /path/to/reducedSFT --output results

The finite-volume update, source, projection, diagnostic and original time
sampling are reproduced in batches, using LAPACK for the tridiagonal solve.
Source/RHS/solve equivalence, independent full forward runs, and the archived
65-point controlled scan are checked before interpreting the results.
An additional endpoint diagnostic stops at exactly 11 years, so temporal
refinement can be assessed without the original time-loop overshoot.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.linalg import solve_banded

CODE_COMMIT = "41640e936c5d777a168ff5c0753491b840263c49"
SOLVER_SHA256 = "2962d1281854b237e06bb38e9f07b69f4b7cdf683d0a46a65eb7a001908281b0"
T = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.5])
CASES = ("linear", "tilt", "latitude", "inflow", "combined", "shift", "physical_2", "physical_5")
CONTROLLED = CASES[:5]
SECONDS_PER_YEAR = 365.25 * 86400.
SLICES = {name: slice(1 + 13*i, 1 + 13*(i+1)) for i, name in
          enumerate(("tilt", "latitude", "combined", "shift", "physical_2", "physical_5"))}


def import_solver(root):
    path = root / "nonlinear_response_sft_experiment.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOLVER_SHA256:
        raise RuntimeError("Solver differs from the pinned baseline; do not mix versions.")
    sys.path.insert(0, str(root))
    return importlib.import_module("nonlinear_response_sft_experiment")


def source_and_velocity(nre, p, grid, t):
    """Batch the same 79 independent forward states; no fitted approximation."""
    lat = np.asarray(grid["lat_deg"])[:, None]
    centre = nre.activity_belt_latitude(t, 1., p, False)
    centres = np.r_[centre, np.full(13, centre), centre + p.b_LQ*T*T]
    seps = np.r_[p.polarity_sep_deg, p.polarity_sep_deg/(1+p.b_TQ*T*T),
                 np.full(13, p.polarity_sep_deg)]
    nf = np.exp(-.5*((lat-(centres+seps/2))/p.source_width_deg)**2)
    nl = np.exp(-.5*((lat-(centres-seps/2))/p.source_width_deg)**2)
    sf = np.exp(-.5*((lat-(-centres-seps/2))/p.source_width_deg)**2)
    sl = np.exp(-.5*((lat-(-centres+seps/2))/p.source_width_deg)**2)
    rings = [nf, nl, sf, sl]
    phis = np.array([r.mean(axis=0) for r in rings])
    scale = phis.mean(axis=0)
    nf, nl, sf, sl = [r*(scale/phi) for r, phi in zip(rings, phis)]
    shapes = (nf-nl)-(sf-sl)
    qlq = np.exp(-(2*centre*p.b_LQ*T*T + (p.b_LQ*T*T)**2)/(2*p.lambda_R_deg**2))
    qi = 1/(1+p.b_I*T*T)
    src = np.empty((p.nmu, 79))
    src[:, 0] = shapes[:, 0]
    src[:, SLICES["tilt"]] = shapes[:, 1:14]*T
    src[:, SLICES["latitude"]] = shapes[:, :1]*(T*qlq)
    src[:, SLICES["combined"]] = shapes[:, 1:14]*(T*qlq*qi)
    src[:, SLICES["shift"]] = shapes[:, 14:]*T
    src[:, SLICES["physical_2"]] = shapes[:, :1]*T
    src[:, SLICES["physical_5"]] = shapes[:, :1]*T
    src *= p.source_amp*nre.cycle_envelope(t, p)/SECONDS_PER_YEAR
    src -= src.mean(axis=0)
    u = np.broadcast_to(nre.meridional_flow(grid["lat_faces"], grid["u0"])[:, None],
                        (p.nmu+1, 79)).copy()
    profile = nre.activity_inflow(grid["lat_faces_deg"], t, 1., replace(p, inflow_amp_m_s=1.))
    u[:, SLICES["physical_2"]] += profile[:, None]*(2*T*T)
    u[:, SLICES["physical_5"]] += profile[:, None]*(5*T*T)
    return src, u*np.asarray(grid["cos_faces"])[:, None]/p.R_sun


def band_matrix(grid, factor=1.):
    n = len(grid["mu"])
    ab = np.zeros((3, n))
    ab[0, 1:] = factor*np.asarray(grid["diff_upper"])
    ab[1] = 1 + factor*(np.asarray(grid["diff_diag"])-1)
    ab[2, :-1] = factor*np.asarray(grid["diff_lower"])
    return ab


def rhs(B, src, v, dmu):
    flux = np.zeros((len(B)+1, B.shape[1]))
    flux[1:-1] = v[1:-1]*np.where(v[1:-1] >= 0, B[:-1], B[1:])
    return src-np.diff(flux, axis=0)/dmu


def advance(B, explicit, dt, ab):
    new = solve_banded((1, 1), ab, B+dt*explicit, check_finite=False)
    return new-new.mean(axis=0)


def column_for(case, i):
    if case in ("linear", "inflow"):
        return 0
    return SLICES[case].start+i


def baseline_params(p, case):
    if case == "shift":
        return replace(p, latitude_quenching_mode="shift"), "latitude"
    if case.startswith("physical_"):
        return replace(p, inflow_mode="physical", inflow_amp_m_s=float(case.split("_")[1])), "inflow"
    return p, case


def verify_steps(nre):
    """Independent source, explicit update and Thomas-solve comparisons."""
    checks = []
    rng = np.random.default_rng(2145)
    for nmu in (91, 721):
        p = nre.SFTParams(nmu=nmu, dt_days=.25)
        g = nre.make_grid(p)
        B = rng.normal(size=(nmu, 79))
        ab = band_matrix(g)
        errors = {"source": 0., "rhs": 0., "step": 0.}
        for t in (0., 4.5, 11.):
            src, v = source_and_velocity(nre, p, g, t)
            explicit = rhs(B, src, v, g["dmu"])
            fast = advance(B, explicit, g["dt"], ab)
            for case in CASES:
                if case in ("linear", "inflow"):
                    continue  # Linearity is checked against the complete archived scan.
                pp, cc = baseline_params(p, case)
                for i in (0, 5, 12):
                    j = column_for(case, i)
                    sr = nre.bipolar_ring_source(g["lat_deg"], t, float(T[i]), pp,
                                                cc in ("tilt", "combined"),
                                                cc in ("latitude", "combined"),
                                                cc in ("inflow", "combined"))
                    rr = nre.sft_rhs_mu(B[:, j], t, float(T[i]), pp, g, sr,
                                        pp.inflow_mode == "physical")
                    step = nre.implicit_diffusion_step(B[:, j]+g["dt"]*rr, g)
                    step -= step.mean()
                    for key, actual, expected in (("source", src[:, j], sr),
                                                  ("rhs", explicit[:, j], rr),
                                                  ("step", fast[:, j], step)):
                        err = float(np.max(abs(actual-expected))/np.max(abs(expected)))
                        errors[key] = max(errors[key], err)
                        assert err < 2e-12, (nmu, case, t, T[i], key, err)
        checks.append({"nmu": nmu, **errors})
    return checks


def diagnostics(p, g, B, protocol, max_cfl, wall_seconds):
    tw = np.full(p.nmu, g["dmu"]); tw[[0, -1]] *= .5
    raw = (1.5*np.asarray(g["mu"])*tw) @ B
    midpoint = (1.5*np.asarray(g["mu"])*g["dmu"]) @ B
    records = []
    for case in CASES:
        if case == "linear":
            ds, dm = raw[0]*T, midpoint[0]*T
            cfl = np.full(13, max_cfl[0])
        elif case == "inflow":
            ds, dm = raw[0]*T/(1+p.b_I*T*T), midpoint[0]*T/(1+p.b_I*T*T)
            cfl = np.full(13, max_cfl[0])
        else:
            sl = SLICES[case]
            ds, dm, cfl = raw[sl], midpoint[sl], max_cfl[sl]
        phat, phatm = ds/ds[5], dm/dm[5]
        for i, amp in enumerate(T):
            records.append(dict(nmu=p.nmu, dt_days=p.dt_days, protocol=protocol,
                                case=case, T=float(amp), D=float(ds[i]), D_midpoint=float(dm[i]),
                                P_hat=float(phat[i]), P_hat_midpoint=float(phatm[i]),
                                Q=float(ds[i]/(raw[0]*amp)), Q_midpoint=float(dm[i]/(midpoint[0]*amp)),
                                max_outgoing_Courant=float(cfl[i]), scan_wall_seconds=wall_seconds))
    return records


def scan(nre, p):
    """Run the legacy time loop; also capture its exact-horizon counterpart."""
    assert np.isinf(p.tau_yr) and p.balanced_polarities and p.lq_reference == "belt"
    g = nre.make_grid(p); dt = float(g["dt"]); ab = band_matrix(g)
    B = np.zeros((p.nmu, 79)); fixed = None; max_cfl = np.zeros(79)
    start = time.perf_counter()
    horizon = p.cycle_years*SECONDS_PER_YEAR
    for k, t in enumerate(g["t_yr"]):
        src, v = source_and_velocity(nre, p, g, float(t))
        outgoing = np.maximum(v[1:], 0)-np.minimum(v[:-1], 0)
        max_cfl = np.maximum(max_cfl, dt/g["dmu"]*outgoing.max(axis=0))
        assert np.max(max_cfl) <= 1, (p.nmu, p.dt_days, max_cfl.max())
        explicit = rhs(B, src, v, g["dmu"])
        remaining = horizon-k*dt
        if fixed is None and 0 < remaining <= dt:
            fixed = advance(B, explicit, remaining, band_matrix(g, remaining/dt))
        B = advance(B, explicit, dt, ab)
        if not np.isfinite(B).all():
            raise FloatingPointError((p.nmu, p.dt_days, k))
    assert fixed is not None
    elapsed = time.perf_counter()-start
    records = diagnostics(p, g, B, "legacy", max_cfl, elapsed)
    records += diagnostics(p, g, fixed, "exact_11yr", max_cfl, elapsed)
    return records, B, g


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def verify_forward(nre, p, fast, grid, cases):
    checks = []
    for case, amp in cases:
        pp, cc = baseline_params(p, case)
        start = time.perf_counter()
        ref = nre.run_sft_case(amp, cc, pp)
        elapsed = time.perf_counter()-start
        i = int(np.where(T == amp)[0][0]); j = column_for(case, i)
        actual = fast[:, j].copy()
        if case == "linear":
            actual *= amp
        elif case == "inflow":
            actual *= amp/(1+p.b_I*amp*amp)
        err = float(np.max(abs(actual-ref[-1]))/np.max(abs(ref[-1])))
        assert err < 5e-12, (case, p.nmu, p.dt_days, amp, err)
        checks.append(dict(nmu=p.nmu, dt_days=p.dt_days, case=case, T=amp,
                           field_max_relative_difference=err, original_solver_seconds=elapsed))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    nre = import_solver(args.code_repo.resolve())
    verification = {"step_comparisons": verify_steps(nre), "forward_comparisons": []}
    results = []
    # Spatial comparisons use the same 0.25-day step, including the strongest
    # physical inflow at the finest grid. Temporal checks use 181 cells.
    setups = [(181, 1.), (181, .5), (91, .25), (181, .25), (361, .25), (721, .25), (1441, .25)]
    for nmu, dt_days in setups:
        p = nre.SFTParams(nmu=nmu, dt_days=dt_days)
        rows, B, grid = scan(nre, p)
        results += rows
        checks = [("linear", 1.)]
        if nmu == 181 and dt_days == 1.:
            checks += [(c, 2.5) for c in CASES if c != "linear"]
        if nmu == 721:
            checks += [("shift", 2.5), ("physical_5", 2.5)]
        verification["forward_comparisons"] += verify_forward(nre, p, B, grid, checks)
        write_csv(args.output/"grid_sensitivity_results.csv", results)
        print(f"N={nmu}, dt={dt_days}: a={rows[5]['D']:.10f}, "
              f"a_mid={rows[5]['D_midpoint']:.10f}, "
              f"max_C={max(r['max_outgoing_Courant'] for r in rows):.6f}, "
              f"scan_seconds={rows[0]['scan_wall_seconds']:.2f}", flush=True)
    archived = list(csv.DictReader((args.code_repo/"figures_nonlinear_response_v2/nonlinear_response_results_v2.csv").open()))
    refrows = {(r["case"], r["T"]): r for r in results if r["nmu"] == 181 and r["dt_days"] == 1. and r["protocol"] == "legacy"}
    archive_error = 0.
    for old in archived:
        new = refrows[(old["case"], float(old["T"]))]
        archive_error = max(archive_error, abs(new["D"]-float(old["D_signed"]))/abs(new["D"]))
        assert abs(new["P_hat"]-float(old["P_hat"])) < 1e-9
    assert archive_error < 1e-9
    verification["archived_D_max_relative_difference"] = archive_error
    metadata = dict(code_commit=CODE_COMMIT, solver_sha256=SOLVER_SHA256,
                    study_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    setups=setups, parameters=asdict(nre.SFTParams()), verification=verification)
    # Infinity is represented explicitly as text, keeping valid standard JSON.
    metadata["parameters"]["tau_yr"] = "infinity"
    (args.output/"verification.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print("All independent comparisons and all 65 archived controlled responses passed.", flush=True)


if __name__ == "__main__":
    main()
