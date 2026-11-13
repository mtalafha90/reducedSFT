# -*- coding: utf-8 -*-
"""
Regression tests for the inflow-memory coupling analysis
(inflow_memory_coupling_analysis.py, docs/inflow_memory_coupling.md):

- the vectorised solver reproduces the reference solver with physical inflows;
- the new-flux / inherited-flux split is exact (linearity at fixed inflows);
- the adjoint budget with the inflow-modified operator equals finite
  differences of separate runs;
- the Hale-alternating periodic state maps onto its negative.

Total runtime is of the order of twenty seconds.
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
