"""Reproduce the targeted scientific checks used in the final manuscript audit.

Run against reducedSFT commit 41640e936c5d777a168ff5c0753491b840263c49.
This is not a new grid-convergence study or a transport-parameter scan.
"""
import argparse
from pathlib import Path
import sys, json, csv, hashlib
import numpy as np
from dataclasses import replace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--solver', type=Path, required=True,
                    help='Path to a checkout of the pinned reducedSFT commit')
parser.add_argument('--output', type=Path,
                    default=Path(__file__).with_name('verification.json'))
args = parser.parse_args()
solver = args.solver.resolve()
repo = Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_HASHES = {'nonlinear_response_sft_experiment.py': '2962d1281854b237e06bb38e9f07b69f4b7cdf683d0a46a65eb7a001908281b0', 'mathematical_framework_analysis.py': '08368af1c11a37d5b33d1bf7550b431f65a9ff2a0578ef2c908bdfa7e0e386c1', 'multicycle_feedback_analysis.py': 'fe113277dacf8da0b102b026fadec1d017f4734badceb13d38110a69e638c0e6', 'physical_validation_analysis.py': '4583faea2045454695ed1ae7991fb85f4f8116d17f7c9d8e33dc45101d3395a3', 'parameter_space_analysis.py': '46ea3d8ac41813152223df563000c2b40de67e027c8b935c7d58a1e16c415c13'}
for name, expected in EXPECTED_SOURCE_HASHES.items():
    if hashlib.sha256((solver / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Unexpected solver source: {name}; use the pinned commit')
sys.path.insert(0, str(solver))
import nonlinear_response_sft_experiment as nre
import mathematical_framework_analysis as mfa
import physical_validation_analysis as pva
import multicycle_feedback_analysis as mca
import parameter_space_analysis as psa

p=nre.SFTParams(); grid=nre.make_grid(p)
A,M=mfa.build_operators(p,grid)
K,w=mfa.adjoint_kernel(p,grid,A,M)
S=mfa.linear_source_series(p,grid)
omega=mfa.yield_weights(grid,K,S)
t=np.asarray(grid['t_yr']); dt=float(grid['dt']); N=len(t)
a=float(omega.sum())
mean=float(omega@t/a)
std=float(np.sqrt(omega@(t-mean)**2/a))
Kext=mfa.extended_kernel(p,grid,A,M,K[0],40.)
centres,fp,lr_ext,f0=pva.pair_yield_function(p,grid,Kext,w)
_,fp11,lr11,_=pva.pair_yield_function(p,grid,K[0],w)

Pt,Minv,Pi=mca.build_fast_stepper(p,grid,A,M)
PMshape,env,lam0=mca.precompute_cycle_source(p,grid,Minv,Pi)
B=mca.evolve_cycle(np.zeros(p.nmu),p.source_amp*env/mca.SECONDS_PER_YEAR,Pt,PMshape,dt)
G=np.linalg.matrix_power(Pt,N)
r=float(w@G@B/(w@B))
tau=5.; pt=replace(p,tau_yr=tau)
At,Mt=mfa.build_operators(pt,nre.make_grid(pt))
Ptt,_,_=mca.build_fast_stepper(pt,nre.make_grid(pt),At,Mt)
Gt=np.linalg.matrix_power(Ptt,N)
decay=float(np.exp(-N*dt/(mca.SECONDS_PER_YEAR*tau)))
fixed_r=float(w@Gt@B/(w@B))
ct=psa.transport_constants(pt)
ci=psa.transport_constants(p)
qcomb={str(x):float(nre.Q_theory(x,p,'combined')) for x in [.2,2.5]}
csvfile=repo/'revisions/point5/grid_sensitivity_results.csv'
rows=list(csv.DictReader(csvfile.open()))
legacy=[x for x in rows if x['protocol']=='legacy' and int(x['nmu'])==181 and float(x['dt_days'])==1.]
inflows={}
for case in ('physical_2','physical_5'):
 rr=sorted([x for x in legacy if x['case']==case],key=lambda x:float(x['T']))
 Ts=np.array([float(x['T']) for x in rr]); Q=np.array([float(x['Q']) for x in rr])
 if not len(Ts):raise ValueError(sorted(set(x['case'] for x in rows)))
 beff=(1/Q-1)/Ts**2
 fit=lambda sel:float(((1/Q[sel]-1)*Ts[sel]**2).sum()/(Ts[sel]**4).sum())
 inflows[case]={'b_at_1':float(beff[np.where(Ts==1)[0][0]]),'b_at_2_5':float(beff[-1]),'fit_weak':fit(Ts<=1.2),'fit_full':fit(Ts>0),'monotone_b':bool(np.all(np.diff(beff)>0))}

report={'solver_commit':'41640e936c5d777a168ff5c0753491b840263c49',
 'grid_results_sha256':hashlib.sha256(csvfile.read_bytes()).hexdigest(),'a':a,'omega_min':float(omega.min()),'normalised_weight_sum':float((omega/a).sum()),
 't_mean_yr':mean,'t_std_yr':std,'weighted_latitude_deg':float(p.base_lat_deg+p.lat_drift_deg*(1-mean/p.cycle_years)),
 'lambda_pair_11yr_deg':lr11,'lambda_pair_extended_deg':lr_ext,'extended_vs_11yr_relative':abs(lr_ext/lr11-1),
 'pair_yield_dynamic_range':float(fp[0]/fp[-1]),'pair_yield_min':float(fp.min()),'r':r,
 'same_profile_r_tau5':fixed_r,'continuous_prediction_same_profile':decay*r,
 'same_profile_relative_difference':abs(fixed_r/(decay*r)-1),'regenerated_profile_r_tau5':ct['r1'],
 'regenerated_profile_relative_difference':abs(ct['r1']/(decay*r)-1),
 'discrete_decay_factorisation_relative_matrix_residual':float(np.linalg.norm(Ptt-(1-dt/(mca.SECONDS_PER_YEAR*tau))*Pt)/np.linalg.norm(Ptt)),
 'tau5_a':ct['a'],'tau5_lambda_pair':ct['lamR'],'tauinf_lambda_pair':ci['lamR'],
 'combined_endpoint_efficiencies':qcomb,'inflow_coefficients':inflows,
 'source_hashes':{f:hashlib.sha256((solver/f).read_bytes()).hexdigest() for f in ['nonlinear_response_sft_experiment.py','mathematical_framework_analysis.py','multicycle_feedback_analysis.py','physical_validation_analysis.py','parameter_space_analysis.py']}}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
