"""The level correction made in advance, added after the fact (not fixed before computing): factors computed WITHOUT
time adjustment, so that nothing estimated after 1 January enters (the time index uses sales of the evaluation
year). Factor = median of value / sale price over the same October-December sales (q4_sales_<year>.parquet from
d38_level.py); every value divided by it; ratio_v2.py, unchanged, on the headline evaluation keys, same resamples.

Usage (repo root, after d38_level.py): .venv/bin/python -I public/scripts/v2/d38_unadjusted.py [--structure]
--structure: the published model P, after d38_level.py --structure (writes struct_eval/chosen/d38/).
Writes public/runs/v2/d38_eval/values_<year>_{gl,ass}_raw.parquet, ratio_<year>_raw.json, unadjusted.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d38_level  # noqa: E402
from d38_level import REPO, YEARS  # noqa: E402


def main():
    if "--structure" in sys.argv:
        d38_level.use_structure()
    OUT = d38_level.OUT
    res = {}
    for year, cfg in YEARS.items():
        q = pd.read_parquet(OUT / f"q4_sales_{year}.parquet")
        f = {"gl": float(np.median(q["v_gl"] / q["price"])), "assessor": float(np.median(q["v_assessor"] / q["price"]))}
        gl = pd.read_parquet(REPO / cfg["gl"], columns=["meta_pin", "value"])
        a = pd.read_parquet(REPO / cfg["assessor"], columns=["meta_pin", "pred_pin_final_fmv_round"]).drop_duplicates("meta_pin")
        a = a.dropna()
        pd.DataFrame({"meta_pin": gl["meta_pin"], "value": gl["value"] / f["gl"]}).to_parquet(
            OUT / f"values_{year}_gl_raw.parquet", index=False)
        pd.DataFrame({"meta_pin": a["meta_pin"], "value": a["pred_pin_final_fmv_round"] / f["assessor"]}).to_parquet(
            OUT / f"values_{year}_ass_raw.parquet", index=False)
        runs = json.loads((REPO / cfg["runs"]).read_text())
        evals = sum([["--eval", cfg["eval_path"](T, r)] for T, r in runs.items()], [])
        out = OUT / f"ratio_{year}_raw.json"
        cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py", *evals,
               "--model", f"glraw={(OUT / f'values_{year}_gl_raw.parquet').relative_to(REPO)}",
               "--model", f"assraw={(OUT / f'values_{year}_ass_raw.parquet').relative_to(REPO)}",
               "--pair", "glraw-assraw", "--out", str(out.relative_to(REPO))]
        assert subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL).returncode == 0
        o = json.loads(out.read_text())["overall"]
        res[year] = dict(factors=f, mdape={m: o["models"][m]["mdape"]["point"] for m in ["glraw", "assraw"]},
                         median_ratio={m: o["models"][m]["median_ratio"]["point"] for m in ["glraw", "assraw"]},
                         diff_mdape=o["pairs"]["glraw-assraw"]["mdape"])
        d = res[year]["diff_mdape"]
        print(f"{year}: factors {f}; dMdAPE {d['point']:+.2f} [{d['lo90']:+.2f}, {d['hi90']:+.2f}]; "
              f"median ratios {res[year]['median_ratio']}", flush=True)
    (OUT / "unadjusted.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
