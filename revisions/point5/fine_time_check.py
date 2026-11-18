#!/usr/bin/env python3
"""Halve the time step at the finest grid used in the Point 5 study."""
import argparse
import hashlib
import json
from pathlib import Path

import grid_sensitivity as study

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--code-repo", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
nre = study.import_solver(args.code_repo.resolve())
p = nre.SFTParams(nmu=1441, dt_days=.125)
rows, field, grid = study.scan(nre, p)
checks = study.verify_forward(nre, p, field, grid,
                             [("linear", 1.), ("physical_5", 2.5)])
study.write_csv(args.output/"fine_time_results.csv", rows)
metadata = {"code_commit": study.CODE_COMMIT, "solver_sha256": study.SOLVER_SHA256,
            "study_sha256": hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
            "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "forward_comparisons": checks}
(args.output/"fine_time_verification.json").write_text(json.dumps(metadata, indent=2)+"\n")
print("Finest-grid half-time-step run and independent forward checks passed.", flush=True)
