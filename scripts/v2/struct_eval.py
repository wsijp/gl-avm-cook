"""Evaluation of S and P against the Assessor, and the selection between them by a rule fixed before fitting
(P: the published model; S: the simpler one-block-surface model). Calls ratio_v2.py unchanged; adds nothing to its
statistics.

Per year (2023 south: keys = each township's prep eval.parquet; 2022 north: keys = p8_eval/keys/prep):
models S, P (unrounded posterior-median group values from struct_run.py's runs) and the rebuilt Assessor values;
pairs S-assessor, P-assessor, S-P; time-adjusted prices; 2,000 resamples (seed 1) clustered by valuation group within
townships. Also the rounded and unadjusted samples.
Selection: one ratio_v2.py run on both years' keys together, models S and P, pair S-P; the rule: choose S unless the
90% upper bound of pooled MdAPE(S) - MdAPE(P) exceeds +0.5 points.

Usage (repo root, after struct_run.py --models S P has finished): .venv/bin/python -I public/scripts/v2/struct_eval.py
       [--runs runs_seed2.json --tag _seed2 --models X]: the second-seed check of the chosen model X.
Writes public/runs/v2/struct_eval/ (values_<model>_<year><tag>.parquet, ratio_<year>_<sample><tag>.json,
ratio_pooled<tag>.json, selection.json); logs struct_ratio_*.log.
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "public/runs/v2/struct_eval"
YEARS = {"2023": dict(assessor="ccao/runs/2023_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                      key=lambda T, prep: f"{prep}/t{T}/eval.parquet"),
         "2022": dict(assessor="ccao/runs/2022_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                      key=lambda T, prep: f"public/runs/v2/p8_eval/keys/prep/t{T}/eval.parquet")}
SAMPLES = {"primary": [], "rounded": ["--rounded"], "unadjusted": ["--sample", "unadjusted"]}
THRESHOLD = 0.5


def ratio(args, out, logname):
    cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py", *args, "--out", str(out.relative_to(REPO))]
    with open(REPO / f"notes/v2/logs/{logname}.log", "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    print(out.name, "exit", rc, flush=True)
    assert rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs.json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", nargs="+", default=["S", "P"])
    a = ap.parse_args()
    runs = json.loads((OUT / a.runs).read_text())
    bad = [k for k, s in runs.items() if s.get("status") != "done" and s["model"] in a.models]
    assert not bad, f"not finished: {bad}"
    keys = {}
    for year, cfg in YEARS.items():
        keys[year] = []
        vpaths = {}
        for m in a.models:
            rs = sorted((s for s in runs.values() if s["model"] == m and s["year"] == year), key=lambda s: s["town"])
            assert len(rs) == (17 if year == "2023" else 13), (m, year, len(rs))
            vals = pd.concat([pd.read_parquet(REPO / s["run"] / "values.parquet",
                                              columns=["meta_pin", "value", "value_rounded"]) for s in rs])
            assert vals["meta_pin"].is_unique
            vpaths[m] = OUT / f"values_{m}_{year}{a.tag}.parquet"
            vals.to_parquet(vpaths[m], index=False)
            keys[year] = [cfg["key"](s["town"], s["prep"]) for s in rs]
        models = sum([["--model", f"{m}={vpaths[m].relative_to(REPO)}"] for m in a.models], [])
        pairs = sum([["--pair", f"{m}-assessor"] for m in a.models], []) + (["--pair", "S-P"] if len(a.models) == 2 else [])
        for sample, extra in SAMPLES.items():
            if a.tag and sample != "primary":
                continue
            ratio([*sum([["--eval", k] for k in keys[year]], []), *models, "--model", f"assessor={cfg['assessor']}",
                   *pairs, *extra], OUT / f"ratio_{year}_{sample}{a.tag}.json", f"struct_ratio_{year}_{sample}{a.tag}")
    # pooled over both years: the selection (S and P), or the second-seed pooled figure (one model)
    for m in a.models:
        pd.concat([pd.read_parquet(OUT / f"values_{m}_{y}{a.tag}.parquet") for y in YEARS]).to_parquet(
            OUT / f"values_{m}_pooled{a.tag}.parquet", index=False)
    evals = sum([["--eval", k] for y in YEARS for k in keys[y]], [])
    models = sum([["--model", f"{m}={(OUT / f'values_{m}_pooled{a.tag}.parquet').relative_to(REPO)}"] for m in a.models], [])
    ratio([*evals, *models, *(["--pair", "S-P"] if len(a.models) == 2 else [])], OUT / f"ratio_pooled{a.tag}.json",
          f"struct_ratio_pooled{a.tag}")
    if len(a.models) == 2 and not a.tag:
        d = json.loads((OUT / "ratio_pooled.json").read_text())["overall"]
        s = d["pairs"]["S-P"]["mdape"]
        sel = dict(rule="choose S unless the 90% upper bound of pooled MdAPE(S) - MdAPE(P) exceeds +0.5 points",
                   n=d["n"], n_clusters=d["n_clusters"], mdape_S=d["models"]["S"]["mdape"]["point"],
                   mdape_P=d["models"]["P"]["mdape"]["point"], delta=s, choice="S" if s["hi90"] <= THRESHOLD else "P")
        (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
        print(json.dumps(sel, indent=1), flush=True)


if __name__ == "__main__":
    main()
