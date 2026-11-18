# Checks for the Solar Physics revision

These folders hold the additional calculations and records made while
revising the manuscript *When Stronger Cycles Build Weaker Polar Fields: An
Adjoint Response Theory for Surface Flux Transport* in response to the
referee. They moved here from the manuscript repository
([`mtalafha90/Reduced_SP_manuscript`](https://github.com/mtalafha90/Reduced_SP_manuscript))
on 18 November 2026. That repository's Git history shows how they were made,
and its revision log, `revision_log.tex`, records each manuscript change.

| Folder | Referee point | Contents |
|---|---|---|
| `point5/` | 5 | Grid and time-step sensitivity study (Section 2.2 and Table 2) |
| `point9/` | 9 | Frozen figure data and the script that draws the eight manuscript figures |
| `point11/` | 11 | Conditional comparison of the quenching mechanisms |
| `point13/` | 13 | Local propagation of amplitude perturbations in the cycle map |
| `settled_dipole/` | 13 (entry PA01) | Settled-dipole closure of the multi-cycle map |
| `final_audit/` | all | Targeted checks used in the final audit |

## Running the checks

The scripts use the published solver at commit
`41640e936c5d777a168ff5c0753491b840263c49`, the commit cited in the
manuscript, not the current branch. Make a second checkout of that commit
next to this repository, then run each script from the root of this
repository. For example:

```bash
git worktree add ../reducedSFT-41640e9 41640e936c5d777a168ff5c0753491b840263c49
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 revisions/settled_dipole/verify_settled_dipole.py --solver ../reducedSFT-41640e9
```

Each folder's README gives its own command, inputs and outputs. To redraw the
manuscript figures, pass a checkout of the manuscript repository:

```bash
python3 revisions/point9/plot_figures.py --manuscript /path/to/Reduced_SP_manuscript
```

After the move, the settled-dipole, Point 11, Point 13, final-audit and
figure-data scripts were rerun from this location against the pinned commit,
with NumPy 2.4.6 (the records were made with NumPy 2.3.5):

- the settled-dipole and final-audit results match their records exactly;
- the Point 11, Point 13 and figure-data results match to floating-point
  round-off (relative differences of order 10⁻¹⁵), and the eight figures
  redrawn from the regenerated data are byte-identical to the manuscript's
  PDFs;
- `point11/quenching_comparison.json` lists SHA-256 `d5e97a73…` for
  `point5/grid_sensitivity_results.csv`, but the only committed version of
  that file has `f357e98d…`, the value recorded by `point9/numerical_checks.json`
  and by the rerun. The rerun with the committed file gives the same Point 11
  results, so the stored hash refers to an uncommitted working copy.

The stored records were not replaced. The Point 5 grid study was not rerun;
its scripts take the solver path as an argument and did not need changing.
