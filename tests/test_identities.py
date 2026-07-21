# -*- coding: utf-8 -*-
"""
Regression tests pinning the exact identities of the reduced response
theory. Every quantitative claim in the manuscript rests on one of these:

- symmetry and conservativity of the discrete transport operators;
- the adjoint duality (dipole from the backward kernel recursion equals
  the forward run) and the value of the linear yield a;
- equivalence of the fast matrix stepper with the reference solver;
- the closed-form envelope average behind the belt-consistent
  latitude-quenching factor, and its static limit;
- the analytic derivative dlnQ/dT and the map-multiplier algebra;
- the cycle-memory parameter r;
- end-to-end agreement of the latitude case with the reduced prediction.

Total runtime is of the order of ten seconds.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nonlinear_response_sft_experiment as nre  # noqa: E402
import mathematical_framework_analysis as mfa  # noqa: E402
import multicycle_feedback_analysis as mca  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    p = nre.SFTParams()
    grid = nre.make_grid(p)
    A, M = mfa.build_operators(p, grid)
    return p, grid, A, M


# ------------------------------------------------------------------
# Discrete operators
# ------------------------------------------------------------------

def test_diffusion_matrix_symmetric(setup):
    p, grid, A, M = setup
    assert np.max(np.abs(M - M.T)) == 0.0


def test_advection_conserves_flux(setup):
    # The explicit operator is in flux form with zero boundary fluxes,
    # so each column sums to zero: total signed flux is conserved.
    p, grid, A, M = setup
    col_sums = np.abs(A.sum(axis=0))
    assert np.max(col_sums) < 1e-16 * np.max(np.abs(A))


def test_implicit_diffusion_conserves_flux(setup):
    # Row sums of the off-diagonal part cancel the diagonal excess:
    # (I - dt L) preserves the mean of B exactly.
    p, grid, A, M = setup
    rng = np.random.default_rng(1)
    B = rng.standard_normal(p.nmu)
    B_new = nre.implicit_diffusion_step(B, grid)
    assert abs(B_new.mean() - B.mean()) < 1e-14 * np.max(np.abs(B))


# ------------------------------------------------------------------
# Adjoint duality and the linear yield
# ------------------------------------------------------------------

def test_duality_and_linear_yield(setup):
    p, grid, A, M = setup
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    S = mfa.linear_source_series(p, grid)
    D_adj = float(float(grid["dt"]) * np.einsum("ki,ki->", K, S))
    _, D_fwd, _, _, _ = nre.run_sft_case(1.0, "linear", p)
    assert abs(D_adj - D_fwd) / abs(D_fwd) < 1e-12
    # regression pin of the linear yield for the reference parameters
    assert D_fwd == pytest.approx(7.8986343643e-02, rel=1e-8)


def test_fast_stepper_matches_solver(setup):
    p, grid, A, M = setup
    Pstep, Minv, Pi = mca.build_fast_stepper(p, grid, A, M)
    PMshape, env, lam0 = mca.precompute_cycle_source(p, grid, Minv, Pi)
    amp = p.source_amp * env / mca.SECONDS_PER_YEAR
    B = mca.evolve_cycle(np.zeros(p.nmu), amp, Pstep, PMshape, float(grid["dt"]))
    D_fast = float(mfa.dipole_weight(p, grid) @ B)
    _, D_ref, _, _, _ = nre.run_sft_case(1.0, "linear", p)
    assert abs(D_fast - D_ref) / abs(D_ref) < 1e-12


# ------------------------------------------------------------------
# Latitude-quenching closure
# ------------------------------------------------------------------

def test_qlq_closed_form_matches_gaussian_average():
    # The belt-mode factor is the exact Gaussian-envelope average of the
    # instantaneous factor with the unclipped linear belt drift.
    p = nre.SFTParams()
    t = np.linspace(-60.0, 70.0, 200001)
    wgt = np.exp(-0.5 * ((t - p.source_peak_time_yr) / p.source_time_width_yr) ** 2)
    lam0 = p.base_lat_deg + p.lat_drift_deg * (1.0 - t / p.cycle_years)
    for T in [0.5, 1.0, 2.0]:
        d = p.b_LQ * T**2
        q = np.exp(-(2 * lam0 * d + d**2) / (2 * p.lambda_R_deg**2))
        exact = np.trapezoid(wgt * q, t) / np.trapezoid(wgt, t)
        assert float(nre.Q_LQ(T, p)) == pytest.approx(exact, rel=1e-9)


def test_qlq_static_limit_is_draft_formula():
    from dataclasses import replace
    p = replace(nre.SFTParams(), lq_reference="static")
    for T in [0.5, 1.0, 2.0]:
        lam0, lamR = p.base_lat_deg, p.lambda_R_deg
        lamT = lam0 + p.b_LQ * T**2
        draft = np.exp(-((lamT**2 - lam0**2) / (2.0 * lamR**2)))
        assert float(nre.Q_LQ(T, p)) == pytest.approx(draft, rel=1e-14)


# ------------------------------------------------------------------
# Map algebra
# ------------------------------------------------------------------

def test_dlnQ_analytic_vs_numerical():
    p = nre.SFTParams()
    lam0 = nre.lq_lambda_eff(p)
    eps = nre.lq_eps(p)
    args = (p.b_TQ, p.b_LQ, p.b_I, lam0, p.lambda_R_deg, eps)
    for T in [0.3, 1.0, 1.7]:
        h = 1e-6
        num = (nre.reduced_lnQ(T + h, *args) - nre.reduced_lnQ(T - h, *args)) / (2 * h)
        assert nre.reduced_dlnQ_dT(T, *args) == pytest.approx(num, rel=1e-6)


def test_single_algebraic_quenching_multiplier():
    # For Q = 1/(1 + b T^2) alone: T* = sqrt((D0-1)/b) and
    # M'(T*) = (2 - D0)/D0, unconditionally stable for D0 > 1.
    b, D0 = 0.5, 3.0
    Tstar = np.sqrt((D0 - 1.0) / b)
    Mprime = 1.0 + Tstar * nre.reduced_dlnQ_dT(Tstar, b, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert Mprime == pytest.approx((2.0 - D0) / D0, rel=1e-12)
    assert -1.0 < Mprime < 1.0


# ------------------------------------------------------------------
# Cycle memory and end-to-end agreement
# ------------------------------------------------------------------

def test_memory_parameter_regression(setup):
    p, grid, A, M = setup
    Pstep, Minv, Pi = mca.build_fast_stepper(p, grid, A, M)
    PMshape, env, lam0 = mca.precompute_cycle_source(p, grid, Minv, Pi)
    dt = float(grid["dt"])
    w = mfa.dipole_weight(p, grid)
    amp = p.source_amp * env / mca.SECONDS_PER_YEAR
    B = mca.evolve_cycle(np.zeros(p.nmu), amp, Pstep, PMshape, dt)
    D1 = float(w @ B)
    B = mca.evolve_cycle(B, np.zeros(len(env)), Pstep, PMshape, dt)
    r1 = float(w @ B) / D1
    assert r1 == pytest.approx(0.9860, abs=2e-3)


def test_latitude_case_matches_reduced_theory(setup):
    # End-to-end: belt-mode latitude case at a strong amplitude agrees
    # with the envelope closure within its documented error budget.
    p, grid, A, M = setup
    _, D1, _, _, _ = nre.run_sft_case(1.0, "latitude", p)
    _, D16, _, _, _ = nre.run_sft_case(1.6, "latitude", p)
    phat_sft = (D16 / D1)
    phat_th = float(nre.theory_normalized(np.array([1.6]), p, "latitude")[0])
    assert abs(phat_sft - phat_th) < 0.05
