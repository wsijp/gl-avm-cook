"""The chosen model's headline study, in the layout the level scripts (d38_*.py) and equity scripts read.

For the chosen model X (struct_eval/selection.json, or --model) and a runs file of struct_run.py:
  chosen<tag>/runs_<year>.json        township -> prep, run, name, seed
  chosen<tag>/values_<year>.parquet    X's values (meta_pin, value, value_rounded)
  chosen<tag>/ratio_<year>_<sample>.json  ratio_v2.py with models gl = X and the rebuilt Assessor values,
                                       pair gl-assessor, on the fixed evaluation keys; samples primary,
                                       rounded, q1_unadjusted, unadjusted (primary only with --tag)
  chosen<tag>/ratio_pooled_primary.json  both years' keys together, gl = X, assessor = both rebuilds (distinct PINs)
  chosen<tag>/centred.json             level-free (median-centred) MdAPE, as in p7b_centred.py, for every primary/rounded study
Usage (repo root): .venv/bin/python -I public/scripts/v2/struct_chosen.py [--model S|P] [--runs runs_seed2.json --tag _seed2]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p7b_centred import centred, mdape, rebuild, table  # noqa: E402
from struct_eval import YEARS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "public/runs/v2/struct_eval"
SAMPLES = {"primary": [], "rounded": ["--rounded"], "q1_unadjusted": ["--sample", "q1_unadjusted"],
           "unadjusted": ["--sample", "unadjusted"]}


def ratio(args, out):
    cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py", *args, "--out", str(out.relative_to(REPO))]
    with open(REPO / f"notes/v2/logs/struct_chosen_{out.parent.name}_{out.stem}.log", "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    print(out.name, "exit", rc, flush=True)
    assert rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--runs", default="runs.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    X = a.model or json.loads((EV / "selection.json").read_text())["choice"]
    out = EV / f"chosen{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    runs = json.loads((EV / a.runs).read_text())
    allkeys, vparts, aparts = [], [], []
    for year, cfg in YEARS.items():
        rs = {s["town"]: s for s in runs.values() if s["model"] == X and s["year"] == year}
        assert len(rs) == (17 if year == "2023" else 13) and all(s["status"] == "done" for s in rs.values())
        rj = {T: dict(prep=s["prep"], run=s["run"], name=Path(s["run"]).name, seed=s["seed"], status="done")
              for T, s in sorted(rs.items())}                       # township order = the studies' key order
        (out / f"runs_{year}.json").write_text(json.dumps(rj, indent=1))
        vals = pd.concat([pd.read_parquet(REPO / s["run"] / "values.parquet", columns=["meta_pin", "value", "value_rounded"])
                          for s in rs.values()])
        assert vals["meta_pin"].is_unique
        vals.to_parquet(out / f"values_{year}.parquet", index=False)
        vparts.append(vals)
        av = pd.read_parquet(REPO / cfg["assessor"], columns=["meta_pin", "pred_pin_final_fmv_round"]).dropna()
        aparts.append(av[av["meta_pin"].isin(vals["meta_pin"])])   # this year's triad only (the 2022 rebuild is county-wide)
        keys = [cfg["key"](T, s["prep"]) for T, s in sorted(rs.items())]
        allkeys += keys
        for sample, extra in SAMPLES.items():
            if a.tag and sample != "primary":
                continue
            ratio([*sum([["--eval", k] for k in keys], []), "--model", f"gl={(out / f'values_{year}.parquet').relative_to(REPO)}",
                   "--model", f"assessor={cfg['assessor']}", "--pair", "gl-assessor", *extra],
                  out / f"ratio_{year}_{sample}.json")
    pd.concat(vparts).to_parquet(out / "values_pooled.parquet", index=False)
    ass = pd.concat(aparts)
    assert ass["meta_pin"].is_unique
    ass.to_parquet(out / "assessor_pooled.parquet", index=False)
    ratio([*sum([["--eval", k] for k in allkeys], []), "--model", f"gl={(out / 'values_pooled.parquet').relative_to(REPO)}",
           "--model", f"assessor={(out / 'assessor_pooled.parquet').relative_to(REPO)}", "--pair", "gl-assessor"],
          out / "ratio_pooled_primary.json")
    # level-free MdAPE (median-centred), as in p7b_centred.py, with its resample check
    res = {}
    for path in sorted(out.glob("ratio_*_primary.json")) + sorted(out.glob("ratio_*_rounded.json")):
        cfg = json.loads(path.read_text())
        ev, P, V, boots = rebuild(cfg)
        allm = np.ones(len(ev), bool)
        _, bs = table(allm, P, V, boots, mdape)
        saved = np.load(path.with_suffix(".boot.npz"))
        for n in V:
            assert np.array_equal(np.array(bs[n]), saved[f"{n}:mdape"]), f"{path.name}: resamples differ ({n})"
        r = dict(source=str(path.relative_to(REPO)), resamples_checked=True)
        r["overall"], _ = table(allm, P, V, boots, centred)
        r["by_town"] = {t: table((ev["town"] == t).to_numpy(), P, V, boots, centred)[0] for t in sorted(ev["town"].unique())}
        key = path.stem.replace("ratio_", "")
        res[key] = r
        o = r["overall"]
        print(f"{key}: level-free MdAPE gl {o['models']['gl']['point']:.2f} assessor {o['models']['assessor']['point']:.2f} "
              f"diff {o['gl-assessor']['point']:+.2f} [{o['gl-assessor']['lo90']:+.2f}, {o['gl-assessor']['hi90']:+.2f}]", flush=True)
    (out / "centred.json").write_text(json.dumps(res, indent=1))
    (out / "model.json").write_text(json.dumps(dict(model=X, runs=a.runs, tag=a.tag), indent=1))


if __name__ == "__main__":
    main()
