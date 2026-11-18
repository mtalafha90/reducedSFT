#!/usr/bin/env python3
"""Reproduce the conditional quenching comparisons added for referee Point 11."""
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, required=True)
    args = parser.parse_args()
    solver = args.solver.resolve()
    here = Path(__file__).resolve().parent
    manuscript = here.parents[1]
    sys.path.insert(0, str(solver))
    import nonlinear_response_sft_experiment as nre
    import mathematical_framework_analysis as mfa
    import physical_validation_analysis as pva

    p = nre.SFTParams()
    grid = nre.make_grid(p)
    A, M = mfa.build_operators(p, grid)
    K, _ = mfa.adjoint_kernel(p, grid, A, M)
    omega = mfa.yield_weights(grid, K, mfa.linear_source_series(p, grid))
    a = float(omega.sum())
    assert np.all(omega > 0)

    def q_lq(t):
        return float(omega @ mfa.q_lq_instantaneous(grid['t_yr'], t, p) / a)

    @lru_cache(maxsize=None)
    def q_tq(t):
        return pva.exact_kernel_prediction(p, grid, K, t, 'tilt') / (a * t)

    def q_iq(t):
        return float(nre.Q_I(t, p))

    def critical_latitude(t):
        return p.lambda_R_deg**2 * np.log1p(p.b_TQ*t*t) / (p.b_LQ*t*t) - p.b_LQ*t*t/2

    baseline_path = manuscript / 'revisions/point5/grid_sensitivity_results.csv'
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[(baseline.nmu == p.nmu) & (baseline.dt_days == p.dt_days)
                        & (baseline.protocol == 'legacy')]
    amplitude = np.sort(baseline.loc[baseline.case == 'linear', 'T'].unique())
    predictions = {case: np.array([func(float(t)) for t in amplitude])
                   for case, func in [('latitude', q_lq), ('tilt', q_tq), ('inflow', q_iq)]}
    errors = {}
    for case, values in predictions.items():
        rows = baseline[baseline.case == case].sort_values('T')
        np.testing.assert_array_equal(rows['T'].to_numpy(), amplitude)
        errors[case] = float(np.max(np.abs(a * amplitude * values / rows.D.to_numpy() - 1)))
        assert errors[case] < 1e-11

    roots = {}
    for name, other, bracket in [('LQ_equals_exact_TQ', q_tq, (1.2, 1.4)),
                                 ('LQ_equals_IQ', q_iq, (1.8, 2.0))]:
        def difference(t):
            return q_lq(t) - other(t)
        root = float(brentq(difference, *bracket, xtol=1e-12))
        roots[name] = {'T': root, 'Q': q_lq(root), 'residual_Q': difference(root),
                       'bracket_T': list(bracket),
                       'difference_below': difference(root - .01),
                       'difference_above': difference(root + .01)}
        assert abs(difference(root)) < 1e-11
        assert difference(root - .01) > 0 > difference(root + .01)

    thresholds = {}
    for t in (.2, 1., 2., 2.5):
        critical = critical_latitude(t)
        delta = p.b_LQ * t*t
        factor = np.exp(-(2*critical*delta + delta*delta) / (2*p.lambda_R_deg**2))
        np.testing.assert_allclose(factor, nre.Q_TQ(t, p), rtol=1e-14)
        thresholds[str(t)] = float(critical)
    mean_latitude = float(omega @ (p.base_lat_deg + p.lat_drift_deg *
        (1 - np.clip(grid['t_yr'] / p.cycle_years, 0, 1))) / a)
    inputs = [solver / f for f in ['nonlinear_response_sft_experiment.py',
              'mathematical_framework_analysis.py', 'physical_validation_analysis.py']]
    inputs.append(baseline_path)
    result = {
        'solver_commit': '41640e936c5d777a168ff5c0753491b840263c49',
        'input_sha256': {(str(f.relative_to(solver)) if f.is_relative_to(solver)
                          else str(f.relative_to(manuscript))):
                         hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        'discretisation': {'nmu': p.nmu, 'dt_days': p.dt_days,
                           'source_steps': len(grid['t_yr']),
                           'protocol': 'original legacy loop and trapezoidal dipole'},
        'parameters': {'b_LQ_deg': p.b_LQ, 'b_TQ': p.b_TQ, 'b_IQ': p.b_I,
                       'lambda_R_cl_deg': p.lambda_R_deg},
        'a': a, 'baseline_max_relative_D_error': errors,
        'amplitudes': amplitude.tolist(),
        'efficiencies': {k: v.tolist() for k, v in predictions.items()},
        'cycle_integrated_crossovers': roots,
        'local_latitude_threshold_deg': thresholds,
        'local_weak_amplitude_threshold_deg': p.b_TQ*p.lambda_R_deg**2/p.b_LQ,
        'kernel_weighted_mean_unshifted_latitude_deg': mean_latitude,
        'LQ_weak_amplitude_quadratic_coefficient': mean_latitude*p.b_LQ/p.lambda_R_deg**2,
        'scope': 'Conditional reference-model comparison; no global solar hierarchy or calibrated transport transition.'}
    (here/'quenching_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'errors': errors, 'crossovers': roots, 'local_thresholds': thresholds}, indent=2))


if __name__ == '__main__':
    main()
