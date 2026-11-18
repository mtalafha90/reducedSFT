#!/usr/bin/env python3
"""Evaluate local perturbation propagation for the existing LQ+IQ cycle map."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.optimize import brentq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, required=True)
    args = parser.parse_args()
    solver = args.solver.resolve()
    sys.path.insert(0, str(solver))
    import nonlinear_response_sft_experiment as nre
    import mathematical_framework_analysis as mfa
    import multicycle_feedback_analysis as mc

    p = nre.SFTParams()
    grid = nre.make_grid(p)
    A, M = mfa.build_operators(p, grid)
    K, w = mfa.adjoint_kernel(p, grid, A, M)
    omega = mfa.yield_weights(grid, K, mfa.linear_source_series(p, grid))
    Q, dlnQ, a = mc.make_Qx(p, grid, omega)
    _, dipole, _, _, B = nre.run_sft_case(1.0, 'linear', p)
    Pstep, _, _ = mc.build_fast_stepper(p, grid, A, M)
    G = np.linalg.matrix_power(Pstep, len(grid['t_yr']))
    r = float((w @ (G @ B)) / (w @ B))
    assert abs(r - 0.993962) < 5e-7
    assert abs(a / dipole - 1) < 1e-11

    def q(t):
        return -t * dlnQ(t)

    def slope(t, gain):
        return gain * Q(t) * (1 - q(t)) - r

    def mapping(t, gain):
        return t * (gain * Q(t) - r)

    rows = []
    derivative_errors = []
    fixed_point_errors = []
    propagation_errors = []
    for gain in [2.05, 2.5, 3.2]:
        tstar = brentq(lambda t: gain * Q(t) - (1 + r), 1e-6, 2.5,
                       xtol=1e-14)
        stiffness = q(tstar)
        multiplier = 1 - (1 + r) * stiffness
        fixed_point_errors.append(abs(mapping(tstar, gain) - tstar))
        h = 1e-5
        numerical = (mapping(tstar + h, gain) - mapping(tstar - h, gain)) / (2*h)
        derivative_errors.append(abs(numerical - multiplier))
        assert abs(slope(tstar, gain) - multiplier) < 1e-12
        assert 0 < abs(multiplier) < 1
        # Finite differences of the six-fold map, independently of the
        # product-rule formula, verify the multi-cycle sensitivity.
        plus, minus = tstar + 1e-5, tstar - 1e-5
        for _ in range(6):
            assert gain * Q(plus) > r and gain * Q(minus) > r
            plus, minus = mapping(plus, gain), mapping(minus, gain)
        numerical_six = (plus - minus) / 2e-5
        propagation_errors.append(abs(numerical_six - multiplier**6))
        rows.append({'D0': gain, 'T_star': tstar, 'Q_star': Q(tstar),
                     'q_star': stiffness, 'multiplier': multiplier,
                     'relaxation_cycles': -1/np.log(abs(multiplier)),
                     'six_cycle_sensitivity': multiplier**6})

    # Away from a fixed point, compare the derivative of a composed map
    # with the product of the local multipliers on the reference orbit.
    gain, initial, count = 2.5, 0.8, 6
    t, product = initial, 1.0
    for _ in range(count):
        assert gain * Q(t) > r
        product *= slope(t, gain)
        t = mapping(t, gain)
    plus, minus, h = initial + 1e-5, initial - 1e-5, 1e-5
    for _ in range(count):
        assert gain * Q(plus) > r and gain * Q(minus) > r
        plus, minus = mapping(plus, gain), mapping(minus, gain)
    product_error = abs((plus - minus)/(2*h) - product)

    t_zero = brentq(lambda t: q(t) - 1/(1+r), 1e-6, 2.5)
    gain_zero = (1+r)/Q(t_zero)
    pd_gain = mc.period_doubling_gain(r, Q, dlnQ)
    assert abs(pd_gain - 3.352157) < 5e-7
    assert max(derivative_errors) < 1e-8
    assert max(propagation_errors + [product_error]) < 1e-7
    assert max(fixed_point_errors) < 1e-12
    sources = ['nonlinear_response_sft_experiment.py',
               'mathematical_framework_analysis.py',
               'multicycle_feedback_analysis.py']
    result = {
        'solver_commit': '41640e936c5d777a168ff5c0753491b840263c49',
        'source_sha256': {f: hashlib.sha256((solver/f).read_bytes()).hexdigest()
                          for f in sources},
        'definition': 'Kernel-exact latitude-efficiency times scalar inflow efficiency; no tilt quenching, as in the original multi-cycle experiment.',
        'discretisation': {'nmu': p.nmu, 'dt_days': p.dt_days,
                           'source_steps': len(grid['t_yr'])},
        'a': a, 'r': r, 'fixed_points': rows,
        'zero_linear_multiplier': {'T_star': t_zero, 'D0': gain_zero,
                                   'q_star': q(t_zero)},
        'period_doubling_D0': pd_gain,
        'checks': {'max_fixed_point_residual': max(fixed_point_errors),
                   'max_multiplier_finite_difference_error': max(derivative_errors),
                   'max_fixed_point_six_cycle_error': max(propagation_errors),
                   'nonstationary_six_cycle_product_error': product_error},
        'scope': 'Local deterministic sensitivity on the successful-reversal branch with constant r and k. Relaxation time is not a measured stochastic correlation time or forecast horizon. No stochastic dynamo or new multi-cycle SFT ensemble is run.'}
    out = Path(__file__).with_name('perturbation_response.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
