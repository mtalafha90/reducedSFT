#!/usr/bin/env python3
"""Derive the reported comparisons from grid_sensitivity_results.csv."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "grid_sensitivity_results.csv").open()))
for r in rows:
    for key in r:
        if key not in ("case", "protocol"):
            r[key] = float(r[key])
data = {(r["nmu"], r["dt_days"], r["protocol"], r["case"], r["T"]): r for r in rows}
assert len(data) == len(rows) == 1456
cases = ("linear", "tilt", "latitude", "inflow", "combined", "shift", "physical_2", "physical_5")
amplitudes = sorted({r["T"] for r in rows})


def compare(nmu, dt, ref_nmu, ref_dt, group, field, protocol="exact_11yr", ref_protocol=None):
    differences = []
    for case in group:
        for t in amplitudes:
            x = data[nmu, dt, protocol, case, t][field]
            y = data[ref_nmu, ref_dt, ref_protocol or protocol, case, t][field]
            differences.append((abs(x/y-1)*100, abs(x-y), case, t))
    rel = max(differences, key=lambda a: a[0]); ab = max(differences, key=lambda a: a[1])
    return {"max_relative_percent": rel[0], "relative_at": [rel[2], rel[3]],
            "max_absolute": ab[1], "absolute_at": [ab[2], ab[3]]}


summary = {"reference_grid": 1441, "common_dt_days": .25,
           "protocol": "exact_11yr", "spatial": [], "temporal": [], "endpoint": []}
for nmu in (91, 181, 361, 721, 1441):
    a = data[nmu, .25, "exact_11yr", "linear", 1.]
    entry = {"nmu": nmu, "a_trapezoid": a["D"], "a_full_cell": a["D_midpoint"],
             "max_outgoing_Courant": max(r["max_outgoing_Courant"] for r in rows if r["nmu"] == nmu and r["dt_days"] == .25),
             "controlled_P_hat": compare(nmu, .25, 1441, .25, cases[:5], "P_hat"),
             "physical_P_hat": compare(nmu, .25, 1441, .25, cases[5:], "P_hat"),
             "controlled_P_hat_full_cell": compare(nmu, .25, 1441, .25, cases[:5], "P_hat_midpoint"),
             "physical_P_hat_full_cell": compare(nmu, .25, 1441, .25, cases[5:], "P_hat_midpoint"),
             "by_case": {c: {field: compare(nmu, .25, 1441, .25, [c], field)
                             for field in ("D", "D_midpoint", "P_hat", "P_hat_midpoint", "Q", "Q_midpoint")}
                         for c in cases}}
    summary["spatial"].append(entry)
for dt in (1., .5):
    summary["temporal"].append({"nmu": 181, "dt_days": dt,
                                 **{field: compare(181, dt, 181, .25, cases, field)
                                    for field in ("D", "D_midpoint", "P_hat", "P_hat_midpoint")}})
for dt in (1., .5, .25):
    summary["endpoint"].append({"nmu": 181, "dt_days": dt,
                                 **{field: compare(181, dt, 181, dt, cases, field,
                                                    protocol="legacy", ref_protocol="exact_11yr")
                                    for field in ("D", "P_hat")}})
summary["all_positive_dipoles"] = all(r["D"] > 0 and r["D_midpoint"] > 0 for r in rows)
summary["sampled_controlled_peak_amplitudes"] = {
    str(nmu): {case: max(amplitudes, key=lambda t: data[nmu, .25, "exact_11yr", case, t]["D"])
               for case in cases[:5]} for nmu in (91, 181, 361, 721, 1441)}
fine_rows = list(csv.DictReader((ROOT / "fine_time_results.csv").open()))
assert len(fine_rows) == 208
fine = {(r["case"], float(r["T"])): r for r in fine_rows if r["protocol"] == "exact_11yr"}
summary["finest_grid_time_check"] = {"nmu": 1441, "dt_days": .25, "reference_dt_days": .125}
for field in ("D", "D_midpoint", "P_hat", "P_hat_midpoint"):
    values = []
    for case in cases:
        for t in amplitudes:
            x = data[1441, .25, "exact_11yr", case, t][field]
            y = float(fine[case, t][field])
            values.append((abs(x/y-1)*100, case, t))
    v = max(values)
    summary["finest_grid_time_check"][field] = {"max_relative_percent": v[0], "at": [v[1], v[2]]}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
for entry in summary["spatial"]:
    print(entry["nmu"], f"a={entry['a_trapezoid']:.5f}", f"a_FV={entry['a_full_cell']:.5f}",
          f"controlled={entry['controlled_P_hat']['max_relative_percent']:.5f}%",
          f"physical={entry['physical_P_hat']['max_relative_percent']:.5f}%")
print("Temporal:", summary["temporal"])
print("Endpoint:", summary["endpoint"])
print("Positive:", summary["all_positive_dipoles"])
print("Sampled peaks:", summary["sampled_controlled_peak_amplitudes"])
