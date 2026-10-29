"""The paper's headline table (Table 2, two panels), computed from the stored predictions.

Panel A: 1 January assessment tests (2023 south and west suburbs; 2022 north suburbs), the hedonic model (gl)
against the Assessor's model before rounding (pred_pin_final_fmv), sale prices time-adjusted to 1 January.
Panel B: county sale-prediction test (all; Chicago; suburbs), the hedonic model against the Assessor's unrounded
predictions.
Measures per model: MdAPE (%), median ratio, COD, VEI. Differences, hedonic minus Assessor, with paired 90%
bootstrap intervals (2,000 draws, seed 1; within each township, whole clusters are drawn with replacement: the
valuation group in the 1 January tests, the property in the county test; the same draw for both models): MdAPE,
VEI, and MdAPE after level removal (each model's values divided by its own median ratio on the same sales; an
after-the-fact check, the same definition in both panels). A second comparison against prices as paid (no time
adjustment) is stored under "unadjusted" for the 1 January tests.
Check: point MdAPE, median ratio and COD must equal the repository's stored values (asserted).
Reads stored outputs only; writes paper/table2.json and paper/table2.md.

Usage, from the working folder that holds this repository as public/:
    nice -n 10 .venv/bin/python -I public/paper/table2.py [B]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vei import vei  # noqa: E402

REPO = HERE.parents[1]   # the working folder (this file is public/paper/table2.py)
CH = REPO / "public/runs/v2/struct_eval/chosen"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def measures(v, p):
    r = v / p
    m = np.median(r)
    return dict(mdape=100 * np.median(np.abs(r - 1)), median_ratio=m, cod=100 * np.mean(np.abs(r - m)) / m,
                vei=vei(v, p)["vei"], lf_mdape=100 * np.median(np.abs(r / m - 1)))


def draws(df, strata, cluster, rng):
    st = [[ix.to_numpy() for _, ix in g.groupby(cluster).groups.items()] for _, g in df.groupby(strata)]
    for _ in range(B):
        yield np.concatenate([np.concatenate([m[i] for i in rng.integers(0, len(m), len(m))]) for m in st])


def run(df, gl, ass, price, strata, cluster):
    df = df.reset_index(drop=True)
    g, a, p = (df[c].to_numpy(float) for c in (gl, ass, price))
    mg, ma = measures(g, p), measures(a, p)
    diffs = {k: [] for k in ("mdape", "vei", "lf_mdape")}
    for i in draws(df, strata, cluster, np.random.default_rng(1)):
        bg, ba = measures(g[i], p[i]), measures(a[i], p[i])
        for k in diffs:
            diffs[k].append(bg[k] - ba[k])
    out = dict(n=len(df), gl={k: float(v) for k, v in mg.items()}, assessor={k: float(v) for k, v in ma.items()})
    for k, d in diffs.items():
        lo, hi = np.percentile(d, [5, 95])
        out[f"d_{k}"] = dict(point=float(mg[k] - ma[k]), lo=float(lo), hi=float(hi))
    return out


def main():
    res = {}
    for y in ("2023", "2022"):
        head = json.loads((CH / f"ratio_{y}_primary.json").read_text())
        ev = pd.concat([pd.read_parquet(REPO / p).assign(town=Path(p).parent.name.lstrip("t"))
                        for p in head["evals"]], ignore_index=True)
        gl = pd.read_parquet(CH / f"values_{y}.parquet", columns=["meta_pin", "value"]).set_index("meta_pin")["value"]
        ass = pd.read_parquet(CH / f"d72/values_{y}_assfin.parquet").set_index("meta_pin")["value"]
        ev["gl"], ev["ass"] = ev["meta_pin"].map(gl), ev["meta_pin"].map(ass)
        r = run(ev, "gl", "ass", "adj_price", "town", "group_id")
        # Check of the time adjustment: the same comparison against prices as paid (no adjustment to 1 January).
        r["unadjusted"] = run(ev, "gl", "ass", "price", "town", "group_id")
        stored = json.loads((CH / f"d72/ratio_{y}_primary.json").read_text())["overall"]["models"]
        for k in ("mdape", "median_ratio", "cod"):
            assert abs(r["gl"][k] - stored["gl"][k]["point"]) < 1e-9, (y, "gl", k)
            assert abs(r["assessor"][k] - stored["assfin"][k]["point"]) < 1e-9, (y, "assessor", k)
        res[f"A. 1 January {y}"] = r
        print("done", y, flush=True)
    t = pd.read_parquet(REPO / "public/runs/v2/county2025_ixa_eval/test_sales.parquet")
    stored = pd.read_csv(REPO / "public/runs/v2/county2025_ixa_eval/summary.csv").set_index("unit")
    for label, unit, m in (("B. County, all", "county", t.index == t.index), ("B. County, City", "City", t["area"] == "City"),
                           ("B. County, suburbs", "suburbs", t["area"] == "suburbs")):
        r = run(t[m], "pred_gl", "pred_assessor", "meta_sale_price", "town", "meta_pin")
        for k, col in (("mdape", "mdape"), ("median_ratio", "median_ratio"), ("cod", "cod")):
            assert abs(r["gl"][k] - stored.loc[unit, f"{col}_gl"]) < 1e-9, (unit, "gl", k)
            assert abs(r["assessor"][k] - stored.loc[unit, f"{col}_assessor"]) < 1e-9, (unit, "assessor", k)
        res[label] = r
        print("done", label, flush=True)
    (HERE / "table2.json").write_text(json.dumps(res, indent=1))
    ci = lambda d, f="{:+.2f}": f"{f.format(d['point'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"
    lines = ["| Test | Sales | MdAPE % GL / Assessor | Median ratio GL / Assessor | COD GL / Assessor | "
             "VEI GL / Assessor | MdAPE difference | MdAPE difference after retrospective level removal | VEI difference |",
             "|---|---:|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        g, a = r["gl"], r["assessor"]
        lines.append(f"| {k} | {r['n']:,} | {g['mdape']:.2f} / {a['mdape']:.2f} | {g['median_ratio']:.3f} / "
                     f"{a['median_ratio']:.3f} | {g['cod']:.1f} / {a['cod']:.1f} | {g['vei']:.1f} / {a['vei']:.1f} | "
                     f"{ci(r['d_mdape'])} | {ci(r['d_lf_mdape'])} | {ci(r['d_vei'], '{:+.1f}')} |")
    (HERE / "table2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
