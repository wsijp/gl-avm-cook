"""Equity: PRD and the other ratio statistics for the Assessor's PUBLISHED values beside the GL and the rebuilt
Assessor model; median ratio by sale-price decile; multi-family; scale check.

Per assessment year (2023 south: the 2023 GL values, the 2023 rebuild, published 2023 values from
p4_fetch_published_2023.py; 2022 north: the 2022 GL values, the 2022 rebuild, published 2022 values from
ay2022/p4_fetch_published_2022.py):
  published value = 10 x mailed_tot (the mailed total assessed value, residential level of assessment 10%), as in
  p4_external_check.py (both years); every evaluation key has one (asserted).
  Versions corrected in advance: the GL and rebuilt values divided by d38_level.py's factors
  (d38_eval/values_*_adv); the published values divided by their own median ratio on the same October-December sales
  (q4_sales_<year>.parquet; the few of those sales without a published value are left out and counted).
ratio_v2.py, unchanged, runs once per year on the headline evaluation keys with all six models and the pairs below,
same 2,000 resamples (seed 1). Checks: the GL and rebuilt figures equal the stored headline study exactly; every corrected
model's PRD, PRB and COD equal its uncorrected model's (scale invariance).
Deciles: of the time-adjusted sale price over all evaluation sales of the year (the same cut points for every model);
median ratio per decile and model, with 90% intervals from ratio_v2's resamples, rebuilt here by its own procedure
(same seed, same order; checked by reproducing its stored overall MdAPE draws).

Usage (repo root): .venv/bin/python -I public/scripts/v2/equity_eval.py [--structure]
--structure: the published model P in place of the earlier GL model (struct_chosen.py and
d38_level.py --structure first); writes public/runs/v2/struct_eval/chosen/equity/.
Writes public/runs/v2/equity_eval/ (values_<year>_published[_adv].parquet, ratio_<year>.json, deciles_<year>.csv,
summary.json).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "public/runs/v2/equity_eval"
D38 = REPO / "public/runs/v2/d38_eval"
YEARS = {
    "2023": dict(published="public/data/v2/p4_published_2023_south.parquet",
                 gl="public/runs/v2/p7_eval/values_all.parquet",
                 assessor="ccao/runs/2023_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                 runs="public/runs/v2/p7_eval/runs.json", eval_path=lambda T, r: f"{r['prep']}/t{T}/eval.parquet",
                 headline="public/runs/v2/p7_eval/ratio_all_primary.json"),
    "2022": dict(published="public/data/v2/ay2022/p8_published_2022_north.parquet",
                 gl="public/runs/v2/p8_eval/values_all.parquet",
                 assessor="ccao/runs/2022_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                 runs="public/runs/v2/p8_eval/runs.json",
                 eval_path=lambda T, r: f"public/runs/v2/p8_eval/keys/prep/t{T}/eval.parquet",
                 headline="public/runs/v2/p8_eval/ratio_all_primary.json"),
}
MODELS = ["gl", "assessor", "published", "gladv", "assadv", "pubadv"]
PAIRS = ["gl-assessor", "gl-published", "assessor-published", "gladv-assadv", "gladv-pubadv"]
STATS = ["median_ratio", "cod", "prd", "prb", "mdape"]


def use_structure():
    """--structure: the published model P (struct_chosen.py) in place of the earlier GL model; same keys and rules."""
    global OUT, D38
    OUT, D38 = REPO / "public/runs/v2/struct_eval/chosen/equity", REPO / "public/runs/v2/struct_eval/chosen/d38"
    for y in YEARS:
        YEARS[y].update(gl=f"public/runs/v2/struct_eval/chosen/values_{y}.parquet", runs=f"public/runs/v2/struct_eval/chosen/runs_{y}.json",
                        headline=f"public/runs/v2/struct_eval/chosen/ratio_{y}_primary.json")


def boots_like_ratio_v2(ev, n=2000, seed=1):
    """ratio_v2.py's resamples: valuation groups within townships, one index set per resample."""
    rng = np.random.default_rng(seed)
    strata = []
    for _, g in ev.groupby("town"):
        strata.append([ix.to_numpy() for _, ix in g.groupby("group_id").groups.items()])
    out = []
    for _ in range(n):
        idx = []
        for members in strata:
            pick = rng.integers(0, len(members), len(members))
            idx.append(np.concatenate([members[i] for i in pick]))
        out.append(np.concatenate(idx))
    return out


def main():
    if "--structure" in sys.argv:
        use_structure()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for year, cfg in YEARS.items():
        print(f"\n=== {year}", flush=True)
        pub = pd.read_parquet(REPO / cfg["published"], columns=["pin", "mailed_tot", "mailed_hie"])
        assert not pub["pin"].duplicated().any()
        pub = pd.DataFrame({"meta_pin": pub["pin"].astype(str),
                            "value": 10 * pd.to_numeric(pub["mailed_tot"], errors="coerce"),
                            "mailed_hie": pd.to_numeric(pub["mailed_hie"], errors="coerce")})
        pub = pub[pub["value"] > 0]
        pub[["meta_pin", "value"]].to_parquet(OUT / f"values_{year}_published.parquet", index=False)
        q = pd.read_parquet(D38 / f"q4_sales_{year}.parquet")
        qv = q["meta_pin"].map(pub.set_index("meta_pin")["value"])
        n_q4_nopub = int(qv.isna().sum())          # October-December sales without a published value: left out
        fpub = float(np.median((qv / q["adj_price"]).dropna()))
        pd.DataFrame({"meta_pin": pub["meta_pin"], "value": pub["value"] / fpub}).to_parquet(
            OUT / f"values_{year}_published_adv.parquet", index=False)
        factors = json.loads((D38 / "factors.json").read_text())[year]
        print(f"  factors: GL {factors['gl']['point']:.4f}, rebuilt {factors['assessor']['point']:.4f}, "
              f"published {fpub:.4f} (n {len(q):,}; {n_q4_nopub} without a published value left out)", flush=True)

        runs = json.loads((REPO / cfg["runs"]).read_text())
        eval_paths = [cfg["eval_path"](T, r) for T, r in runs.items()]
        paths = dict(gl=cfg["gl"], assessor=cfg["assessor"],
                     published=(OUT / f"values_{year}_published.parquet").relative_to(REPO),
                     gladv=(D38 / f"values_{year}_gl_adv.parquet").relative_to(REPO),
                     assadv=(D38 / f"values_{year}_ass_adv.parquet").relative_to(REPO),
                     pubadv=(OUT / f"values_{year}_published_adv.parquet").relative_to(REPO))
        out = OUT / f"ratio_{year}.json"
        cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py",
               *sum([["--eval", p] for p in eval_paths], []),
               *sum([["--model", f"{m}={paths[m]}"] for m in MODELS], []),
               *sum([["--pair", p] for p in PAIRS], []), "--out", str(out.relative_to(REPO))]
        rc = subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdin=subprocess.DEVNULL).returncode
        assert rc == 0
        res = json.loads(out.read_text())
        head = json.loads((REPO / cfg["headline"]).read_text())
        for m in ["gl", "assessor"]:
            for s in STATS:
                assert res["overall"]["models"][m][s] == head["overall"]["models"][m][s], (year, m, s)
        assert res["overall"]["pairs"]["gl-assessor"] == head["overall"]["pairs"]["gl-assessor"], year
        # Scale invariance: dividing every value by one number leaves PRD, PRB and COD unchanged.
        scale = {}
        for m, madv in [("gl", "gladv"), ("assessor", "assadv"), ("published", "pubadv")]:
            for s in ["prd", "prb", "cod"]:
                for where, block in [("overall", res["overall"]),
                                     ("multi-family", res["subgroups"]["class_group=multi-family"])]:
                    d = abs(block["models"][m][s]["point"] - block["models"][madv][s]["point"])
                    scale[f"{where} {m} {s}"] = d
                    assert d < 1e-9, (year, where, m, s, d)
        print(f"  scale check: max |PRD, PRB, COD corrected - as is| = {max(scale.values()):.1e}", flush=True)

        # Deciles of the time-adjusted price; median ratio per decile and model, intervals on ratio_v2's resamples.
        ev = pd.concat([pd.read_parquet(REPO / p).assign(town=Path(p).parent.name.lstrip("t")) for p in eval_paths],
                       ignore_index=True)
        vals = {}
        for m in ["gl", "assessor", "published"]:
            if m == "assessor":
                v = pd.read_parquet(REPO / paths[m], columns=["meta_pin", "pred_pin_final_fmv_round"]).drop_duplicates(
                    "meta_pin").set_index("meta_pin")["pred_pin_final_fmv_round"]
            else:
                v = pd.read_parquet(REPO / paths[m]).set_index("meta_pin")["value"]
            vals[m] = ev["meta_pin"].map(v).astype(float).to_numpy()
            assert not np.isnan(vals[m]).any(), (year, m)
        P = ev["adj_price"].to_numpy()
        boots = boots_like_ratio_v2(ev)
        stored = np.load(out.with_suffix(".boot.npz"))["gl:mdape"]
        mine = np.array([100 * np.median(np.abs(vals["gl"][b] - P[b]) / P[b]) for b in boots])
        assert np.allclose(mine, stored, rtol=0, atol=1e-12), year
        cuts = np.quantile(P, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(cuts, P, side="right") - 1, 0, 9)
        rows = []
        for k in range(10):
            sel = dec == k
            row = dict(decile=k + 1, n=int(sel.sum()), price_lo=float(cuts[k]), price_hi=float(cuts[k + 1]),
                       price_median=float(np.median(P[sel])))
            for m in vals:
                r = vals[m] / P
                bs = [np.median(r[b][sel[b]]) for b in boots]
                row[f"{m}_median_ratio"] = float(np.median(r[sel]))
                row[f"{m}_lo90"] = float(np.quantile(bs, 0.05))
                row[f"{m}_hi90"] = float(np.quantile(bs, 0.95))
            rows.append(row)
        dt = pd.DataFrame(rows)
        dt.to_csv(OUT / f"deciles_{year}.csv", index=False)
        print(dt[["decile", "n", "price_median", "gl_median_ratio", "assessor_median_ratio",
                  "published_median_ratio"]].round(3).to_string(index=False), flush=True)

        hie = ev["meta_pin"].map(pub.set_index("meta_pin")["mailed_hie"]).fillna(0)
        same = (vals["published"] == vals["assessor"])
        summary[year] = dict(
            n=res["overall"]["n"], factors=dict(gl=factors["gl"]["point"], assessor=factors["assessor"]["point"],
                                                published=fpub, n_q4=int(len(q)), n_q4_without_published=n_q4_nopub),
            published_match=dict(eval_keys_with_published=int(len(ev)),
                                 rebuilt_equal_published=int(same.sum()), share_equal=float(same.mean()),
                                 eval_keys_with_mailed_hie=int((hie > 0).sum())),
            scale_check_max_abs_diff=max(scale.values()),
            overall=res["overall"], multi_family=res["subgroups"]["class_group=multi-family"],
            by_class={k: v for k, v in res["subgroups"].items() if k.startswith("class_group")})
        o = res["overall"]["models"]
        for m in MODELS:
            print(f"  {m:9s} " + " ".join(f"{s} {o[m][s]['point']:.4f} [{o[m][s]['lo90']:.4f}, {o[m][s]['hi90']:.4f}]"
                                          for s in STATS), flush=True)
        for p in PAIRS:
            d = res["overall"]["pairs"][p]
            print(f"  {p:20s} " + " ".join(f"{s} {d[s]['point']:+.4f} [{d[s]['lo90']:+.4f}, {d[s]['hi90']:+.4f}]"
                                           for s in STATS), flush=True)
        print(f"  evaluation keys where rebuilt = published exactly: {same.mean():.1%}; with a mailed HIE value: "
              f"{int((hie > 0).sum())}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
