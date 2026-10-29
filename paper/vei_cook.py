"""VEI (IAAO draft 2026, Appendix E) and PRD for the hedonic model (GL) and the Assessor's model on the paper's tests.

Reads stored outputs only:
- 1 January tests (2023 south and west suburbs, 2022 north suburbs): the evaluation sales listed in
  public/runs/v2/struct_eval/chosen/ratio_<y>_primary.json ("evals"), time-adjusted price `adj_price`;
  hedonic model values `values_<y>.parquet` (column value, unrounded); Assessor `d72/values_<y>_assfin.parquet`
  (column value: pred_pin_final_fmv, before rounding).
- County sale-prediction test: public/runs/v2/county2025_ixa_eval/test_sales.parquet (pred_gl, pred_assessor,
  meta_sale_price; area = City or suburbs).
Gives, per model, the VEI with its significance (Appendix E steps 6-7; null when not significant), whether the
first and last groups' intervals overlap, the group medians and the PRD. Table 2 (table2.py) gives the VEI
differences with intervals. Writes paper/vei_results.json and prints a table.

Usage, from the working folder that holds this repository as public/:
    .venv/bin/python -I public/paper/vei_cook.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vei import vei  # noqa: E402

REPO = HERE.parents[1]   # the working folder (this file is public/paper/vei_cook.py)
CH = REPO / "public/runs/v2/struct_eval/chosen"


def prd(v, p):
    r = v / p
    return r.mean() / (v.sum() / p.sum())


def one(label, gl, ass, price):
    out = {}
    for name, v in (("GL", gl), ("Assessor", ass)):
        res = vei(v, price)
        out[name] = dict(vei=round(res["vei"], 2),
                         significance=None if res["significance"] is None else round(res["significance"], 2),
                         first_last_overlap=res["overlap"], n=int(len(v)), prd=round(float(prd(v, price)), 3),
                         group_medians=[round(g["median"], 4) for g in res["groups"]])
    print(f"{label:28s} n={len(price):6d}  VEI GL {out['GL']['vei']:7.2f} (sig {out['GL']['significance']})"
          f"  Assessor {out['Assessor']['vei']:7.2f} (sig {out['Assessor']['significance']})"
          f"  | PRD GL {out['GL']['prd']} Assessor {out['Assessor']['prd']}")
    return out


def main():
    results = {}
    for y in ("2023", "2022"):
        head = json.loads((CH / f"ratio_{y}_primary.json").read_text())
        ev = pd.concat([pd.read_parquet(REPO / p) for p in head["evals"]], ignore_index=True)
        gl = pd.read_parquet(CH / f"values_{y}.parquet", columns=["meta_pin", "value"]).set_index("meta_pin")["value"]
        ass = pd.read_parquet(CH / f"d72/values_{y}_assfin.parquet").set_index("meta_pin")["value"]
        v_gl, v_as = ev["meta_pin"].map(gl).to_numpy(float), ev["meta_pin"].map(ass).to_numpy(float)
        assert not np.isnan(v_gl).any() and not np.isnan(v_as).any()
        results[f"1 January {y}"] = one(f"1 January {y}", v_gl, v_as, ev["adj_price"].to_numpy(float))
    t = pd.read_parquet(REPO / "public/runs/v2/county2025_ixa_eval/test_sales.parquet")
    for label, m in (("County test, all", slice(None)), ("County test, City", t["area"].str.contains("ity")),
                     ("County test, suburbs", ~t["area"].str.contains("ity"))):
        s = t[m]
        results[label] = one(label, s["pred_gl"].to_numpy(float), s["pred_assessor"].to_numpy(float),
                             s["meta_sale_price"].to_numpy(float))
    (HERE / "vei_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
