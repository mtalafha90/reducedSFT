# -*- coding: utf-8 -*-
"""
Regression tests for the inflow-memory coupling analysis
(inflow_memory_coupling_analysis.py, docs/inflow_memory_coupling.md):

- the vectorised solver reproduces the reference solver with physical inflows;
- the new-flux / inherited-flux split is exact (linearity at fixed inflows);
- the adjoint budget with the inflow-modified operator equals finite
  differences of separate runs;
- the Hale-alternating periodic state maps onto its negative;
- the two-variable settled-dipole map has the fixed point and the leading
  multiplier of the complete field, and follows a multi-cycle SFT run
  cycle by cycle.

Total runtime is of the order of one minute.
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nonlinear_response_sft_experiment as nre  # noqa: E402
import inflow_memory_coupling_analysis as imc  # noqa: E402


@pytest.fixture(scope="module")
def ref_model():
    p = nre.SFTParams()
    return imc.Model(p, imc.Coupling(inflow_amp_m_s=2.0, lat_mode="efficiency"))


def test_vectorised_solver_matches_reference_with_inflows():
    p = replace(nre.SFTParams(), inflow_mode="physical", inflow_amp_m_s=2.0)
    mdl = imc.Model(p, imc.Coupling(inflow_amp_m_s=2.0, lat_mode="none"))
    D_vec = float(mdl.dipole(mdl.new_flux(1.5))[0])
    D_ref = nre.run_sft_case(1.5, "inflow", p)[1]
    assert abs(D_vec - D_ref) / abs(D_ref) < 1e-12


def test_new_and_inherited_flux_split_is_exact(ref_model):
    mdl = ref_model
    B_start = -mdl.new_flux(0.8)[:, 0]
    T = 1.3
    X = mdl.cycle(np.column_stack([B_start, B_start, np.zeros(mdl.n)]), T,
                  np.array([1.0, 0.0, 1.0]), T_flow=np.array([T, T, T]))
    D_full, R, P = mdl.dipole(X)
    assert abs(D_full - (P + R)) < 1e-12 * abs(P)


def test_adjoint_budget_matches_finite_differences(ref_model):
    mdl = ref_model
    B_start = -mdl.new_flux(1.0)[:, 0]
    src, tnew, told, _, _ = imc.adjoint_budget(mdl, 1.0, B_start, 1.0)
    fd = imc.finite_difference_budget(mdl, 1.0, B_start, 1.0)
    for adj, ref in zip((src, tnew, told), fd):
        assert abs(adj - ref) < 1e-4 * abs(ref)
    # The inherited-flux transport term is small but non-zero.
    assert 0.0 < abs(told) < 0.01 * abs(tnew)


def test_periodic_state_is_hale_antiperiodic(ref_model):
    mdl = ref_model
    T = 1.0
    Bs = mdl.periodic_state(T)
    B_next = mdl.cycle(Bs, T, -1.0)[:, 0]
    assert np.max(np.abs(B_next + Bs)) < 1e-10 * np.max(np.abs(Bs))


def test_settled_dipole_map_has_complete_field_multiplier(ref_model):
    mdl = ref_model
    st = imc.full_state_stability(mdl, 1.3, imc.linear_yield(mdl.p))
    # Same fixed point and the same leading multiplier as the complete field.
    assert abs(st["D0_two_var"] - st["D0"]) < 1e-4 * st["D0"]
    assert abs(st["two_var"] - st["full"]) < 1e-3
    # The scalar map misses by far more than the closure error.
    assert abs(st["scalar"] - st["full"]) > 0.03
    # Steady-state remnant coefficient from first-cycle and settled survival.
    r_pred = st["r1"] / (1.0 + st["lam1"] - st["r1"])
    assert abs(r_pred - st["r_star"]) < 1e-4
    # New flux survives its first further cycle less well than settled flux.
    assert st["r1"] < st["lam1"] < 1.0


def test_settled_dipole_map_follows_multicycle_sft(ref_model):
    mdl = ref_model
    a = imc.linear_yield(mdl.p)
    k = 3.1 / a
    n_cycles = 10
    T_sft = imc.run_feedback(mdl, np.array([k]), n_cycles)[:, 0]
    T_tab = np.linspace(0.6, 2.6, 81)
    T_flow_tab = np.linspace(0.6, 2.6, 5)
    P_tab = mdl.dipole(mdl.new_flux(T_tab))
    Ps_tab, lam_tab = imc.settled_tables(mdl, T_tab, T_flow_tab)
    T_two = imc.run_settled_map(k, n_cycles, T_tab, P_tab, T_flow_tab, Ps_tab, lam_tab)
    assert T_sft.min() > T_tab[0] and T_sft.max() < T_tab[-1]
    assert np.max(np.abs(T_two - T_sft)) < 2e-3
