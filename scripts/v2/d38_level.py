"""The level correction each model could have made in advance.

For each assessment year (2023 south triad: the 2023 GL values and the 2023 rebuild; 2022 north triad: the 2022 GL
values and the 2022 rebuild) and each model: factor = median of value / time-adjusted price over the October-December
sales of the prior year in that year's training input, selected by the rules below (fixed before computing); every
value of the model is divided by its factor. ratio_v2.py, unchanged, then runs on the same evaluation keys as the
headline studies, the models as is (gl, assessor) and corrected in advance (gladv, assadv), pairs gl-assessor and
gladv-assadv, same 2,000 resamples. The as-is pair must reproduce the headline JSON of the year exactly. The
level-free (after-the-fact) version is read from the stored median-centred results.

Usage (repo root): .venv/bin/python -I public/scripts/v2/d38_level.py [--structure]
--structure: the published model P (struct_chosen.py's outputs) in place of the earlier GL model;
writes public/runs/v2/struct_eval/chosen/d38/.
Writes public/runs/v2/d38_eval/ (q4_sales_<year>.parquet, values_<year>_{gl,ass}_adv.parquet, factors.json,
ratio_<year>.json, summary.json).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "public/runs/v2/d38_eval"
SIZE_KEYS = ["char_bldg_sf", "char_land_sf", "char_rooms", "char_beds", "char_fbath", "char_hbath", "char_gar1_size",
             "char_bsmt", "char_attic_fnsh"]
YEARS = {
    "2023": dict(train="ccao/inputs/2023/training_data.parquet", roll="ccao/inputs/2023/assessment_data.parquet",
                 q=("2022-10-01", "2022-12-31"), folds="public/data/v2/p3_time_index_folds.csv",
                 gl="public/runs/v2/p7_eval/values_all.parquet",
                 assessor="ccao/runs/2023_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                 runs="public/runs/v2/p7_eval/runs.json", eval_path=lambda T, r: f"{r['prep']}/t{T}/eval.parquet",
                 headline="public/runs/v2/p7_eval/ratio_all_primary.json", outlier_flag=True,
                 exclude=["30172130150000"],
                 centred=("public/runs/v2/p7b_eval/centred.json", "seed1_all_primary")),
    "2022": dict(train="ccao/inputs/2022/training_data.parquet", roll="ccao/inputs/2022/assessment_data.parquet",
                 q=("2021-10-01", "2021-12-31"), folds="public/data/v2/ay2022/p8_time_index_folds.csv",
                 gl="public/runs/v2/p8_eval/values_all.parquet",
                 assessor="ccao/runs/2022_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                 runs="public/runs/v2/p8_eval/runs.json",
                 eval_path=lambda T, r: f"public/runs/v2/p8_eval/keys/prep/t{T}/eval.parquet",
                 headline="public/runs/v2/p8_eval/ratio_all_primary.json", outlier_flag=False, exclude=[],
                 centred=("public/runs/v2/p8_eval/centred.json", "all_primary")),
}


def use_structure():
    """--structure: the published model P (struct_chosen.py) in place of the earlier GL model; same keys and rules."""
    global OUT
    OUT = REPO / "public/runs/v2/struct_eval/chosen/d38"
    for y in YEARS:
        YEARS[y].update(gl=f"public/runs/v2/struct_eval/chosen/values_{y}.parquet", runs=f"public/runs/v2/struct_eval/chosen/runs_{y}.json",
                        headline=f"public/runs/v2/struct_eval/chosen/ratio_{y}_primary.json", centred=("public/runs/v2/struct_eval/chosen/centred.json", f"{y}_primary"))


def fold_of(pin):
    return int(hashlib.md5(pin.encode()).hexdigest(), 16) % 5


def q4_sales(cfg):
    """October-December prior-year sales by the selection rules; returns the table and a cascade of counts."""
    cols = ["meta_pin", "meta_card_num", "meta_sale_date", "meta_sale_document_num", "meta_sale_price", "meta_class",
            "ind_pin_is_multicard", "hie_num_active", "meta_tieback_proration_rate"] + SIZE_KEYS
    if cfg["outlier_flag"]:
        cols.append("sv_is_outlier")
    t = pd.read_parquet(REPO / cfg["train"], columns=cols)
    # Chosen rule: a document on more than one PIN anywhere in the training input is multi-PIN.
    multi = t.groupby("meta_sale_document_num")["meta_pin"].nunique()
    multi_docs = set(multi[multi > 1].index)
    t["date"] = pd.to_datetime(t["meta_sale_date"])
    lo, hi = pd.Timestamp(cfg["q"][0]), pd.Timestamp(cfg["q"][1])
    t = t[(t["date"] >= lo) & (t["date"] <= hi)]
    casc = [("training-input rows in the quarter", len(t))]
    keep = t["ind_pin_is_multicard"].eq(False).fillna(False).astype(bool)
    if cfg["outlier_flag"]:
        keep &= t["sv_is_outlier"].eq(False).fillna(False).astype(bool)
    t = t[keep].sort_values(["meta_pin", "meta_sale_document_num", "meta_card_num"])
    t = t.drop_duplicates(["meta_pin", "meta_sale_document_num"])
    casc.append(("the Assessor's training filter of the year (sale documents)", len(t)))
    t = t[~t["meta_sale_document_num"].isin(multi_docs)]
    casc.append(("single-PIN document in the training input", len(t)))
    gl = pd.read_parquet(REPO / cfg["gl"], columns=["meta_pin", "value"]).set_index("meta_pin")["value"]
    a = pd.read_parquet(REPO / cfg["assessor"], columns=["meta_pin", "pred_pin_final_fmv_round"]).drop_duplicates("meta_pin")
    a = a.set_index("meta_pin")["pred_pin_final_fmv_round"].dropna()
    t = t[t["meta_pin"].isin(gl.index) & t["meta_pin"].isin(a.index) & ~t["meta_pin"].isin(cfg["exclude"])]
    casc.append(("PIN valued by both models in the triad", len(t)))
    roll = pd.read_parquet(REPO / cfg["roll"], columns=["meta_pin", "meta_card_num", "meta_class"] + SIZE_KEYS)
    roll = roll.sort_values(["meta_pin", "meta_card_num"]).drop_duplicates("meta_pin").set_index("meta_pin")
    t = t.join(roll[["meta_class"] + SIZE_KEYS].add_prefix("roll_"), on="meta_pin")
    t = t[t["meta_class"] == t["roll_meta_class"]]
    casc.append(("class unchanged between sale record and roll", len(t)))
    same = pd.Series(True, index=t.index)
    for c in SIZE_KEYS:
        same &= (t[c] == t["roll_" + c]) | (t[c].isna() & t["roll_" + c].isna())
    t = t[same & ~(t["hie_num_active"].fillna(0) > 0)]
    casc.append(("property state unchanged (nine size keys) and no active HIE at sale", len(t)))
    t = t.sort_values(["meta_pin", "date", "meta_sale_document_num"]).drop_duplicates("meta_pin")
    casc.append(("earliest such sale per PIN (sales used)", len(t)))
    folds = pd.read_csv(REPO / cfg["folds"], index_col=0)
    t["month"] = t["date"].dt.strftime("%Y-%m")
    t["fold"] = t["meta_pin"].map(fold_of)
    t["log_adjustment"] = [-folds.loc[m, str(k)] for m, k in zip(t["month"], t["fold"])]
    t["price"] = t["meta_sale_price"].astype(float)
    t["adj_price"] = t["price"] * np.exp(t["log_adjustment"])
    t["v_gl"] = t["meta_pin"].map(gl).astype(float)
    t["v_assessor"] = t["meta_pin"].map(a).astype(float)
    t["prorated_at_sale"] = t["meta_tieback_proration_rate"] < 1
    keepcols = ["meta_pin", "meta_sale_document_num", "date", "month", "fold", "meta_class", "price", "log_adjustment",
                "adj_price", "v_gl", "v_assessor", "prorated_at_sale"]
    return t[keepcols].reset_index(drop=True), casc, gl, a


def factor(v, p, rng, boot=2000):
    r = v / p
    pt = float(np.median(r))
    bs = [np.median(r[rng.integers(0, len(r), len(r))]) for _ in range(boot)]
    return dict(point=pt, lo90=float(np.quantile(bs, 0.05)), hi90=float(np.quantile(bs, 0.95)), n=int(len(r)))


def outlier_sensitivity_2022(q):
    """2022 only: also drop October-December 2021 sales flagged sv_is_outlier in the 2025 input (later vintage)."""
    e = pd.read_parquet(REPO / "ccao/inputs/2025/training_data.parquet",
                        columns=["meta_pin", "meta_sale_document_num", "sv_is_outlier"])
    e = e.drop_duplicates(["meta_pin", "meta_sale_document_num"])
    m = q.merge(e, on=["meta_pin", "meta_sale_document_num"], how="left")
    flag = m["sv_is_outlier"]
    out = dict(n_all=int(len(m)), n_flag_true=int(flag.eq(True).sum()), n_flag_missing=int(flag.isna().sum()))
    k = m[flag.eq(False).fillna(False).astype(bool)]
    out["n_kept"] = int(len(k))
    out["gl"] = float(np.median(k["v_gl"] / k["adj_price"]))
    out["assessor"] = float(np.median(k["v_assessor"] / k["adj_price"]))
    return out


def main():
    if "--structure" in sys.argv:
        use_structure()
    OUT.mkdir(parents=True, exist_ok=True)
    factors, summary = {}, {}
    for year, cfg in YEARS.items():
        print(f"\n=== {year}", flush=True)
        q, casc, gl, a = q4_sales(cfg)
        for name, n in casc:
            print(f"  {name}: {n:,}", flush=True)
        q.to_parquet(OUT / f"q4_sales_{year}.parquet", index=False)
        rng = np.random.default_rng(1)
        f = dict(cascade=casc, n_prorated_at_sale=int(q["prorated_at_sale"].sum()),
                 months=q["month"].value_counts().sort_index().to_dict(),
                 gl=factor(q["v_gl"].to_numpy(), q["adj_price"].to_numpy(), rng),
                 assessor=factor(q["v_assessor"].to_numpy(), q["adj_price"].to_numpy(), rng),
                 unadjusted=dict(gl=float(np.median(q["v_gl"] / q["price"])),
                                 assessor=float(np.median(q["v_assessor"] / q["price"]))))
        if year == "2022":
            f["sensitivity_2025_outlier_flag"] = outlier_sensitivity_2022(q)
        factors[year] = f
        print(f"  factor GL {f['gl']['point']:.4f} [{f['gl']['lo90']:.4f}, {f['gl']['hi90']:.4f}]; "
              f"Assessor {f['assessor']['point']:.4f} [{f['assessor']['lo90']:.4f}, {f['assessor']['hi90']:.4f}]; "
              f"n {f['gl']['n']:,}; unadjusted {f['unadjusted']}", flush=True)
        if "sensitivity_2025_outlier_flag" in f:
            print("  2022 sensitivity (2025 outlier flag):", f["sensitivity_2025_outlier_flag"], flush=True)
        pd.DataFrame({"meta_pin": gl.index, "value": gl.to_numpy() / f["gl"]["point"]}).to_parquet(
            OUT / f"values_{year}_gl_adv.parquet", index=False)
        pd.DataFrame({"meta_pin": a.index, "value": a.to_numpy() / f["assessor"]["point"]}).to_parquet(
            OUT / f"values_{year}_ass_adv.parquet", index=False)
        runs = json.loads((REPO / cfg["runs"]).read_text())
        evals = sum([["--eval", cfg["eval_path"](T, r)] for T, r in runs.items()], [])
        out = OUT / f"ratio_{year}.json"
        cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py", *evals,
               "--model", f"gl={cfg['gl']}", "--model", f"assessor={cfg['assessor']}",
               "--model", f"gladv={(OUT / f'values_{year}_gl_adv.parquet').relative_to(REPO)}",
               "--model", f"assadv={(OUT / f'values_{year}_ass_adv.parquet').relative_to(REPO)}",
               "--pair", "gl-assessor", "--pair", "gladv-assadv", "--out", str(out.relative_to(REPO))]
        rc = subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdin=subprocess.DEVNULL).returncode
        assert rc == 0
        res = json.loads(out.read_text())
        head = json.loads((REPO / cfg["headline"]).read_text())
        # Check: the as-is models and pair reproduce the stored headline study exactly (same keys, same resamples).
        for m in ["gl", "assessor"]:
            for s in ["mdape", "cod", "median_ratio"]:
                assert res["overall"]["models"][m][s] == head["overall"]["models"][m][s], (year, m, s)
        assert res["overall"]["pairs"]["gl-assessor"] == head["overall"]["pairs"]["gl-assessor"], year
        for m, madv in [("gl", "gladv"), ("assessor", "assadv")]:
            assert abs(res["overall"]["models"][m]["cod"]["point"] - res["overall"]["models"][madv]["cod"]["point"]) < 1e-9
        cpath, ckey = cfg["centred"]
        c = json.loads((REPO / cpath).read_text())[ckey]["overall"]
        o = res["overall"]
        summary[year] = dict(
            n=o["n"], factors={"gl": f["gl"]["point"], "assessor": f["assessor"]["point"]},
            as_is={m: {s: o["models"][m][s] for s in ["mdape", "cod", "median_ratio"]} for m in ["gl", "assessor"]},
            in_advance={m: {s: o["models"][madv][s] for s in ["mdape", "cod", "median_ratio"]}
                        for m, madv in [("gl", "gladv"), ("assessor", "assadv")]},
            level_free={m: {"mdape": c["models"][m]} for m in ["gl", "assessor"]},
            diff=dict(as_is={s: o["pairs"]["gl-assessor"][s] for s in ["mdape", "cod", "median_ratio"]},
                      in_advance={s: o["pairs"]["gladv-assadv"][s] for s in ["mdape", "cod", "median_ratio"]},
                      level_free={"mdape": c["gl-assessor"]}),
            subgroups={k: {"in_advance": v["pairs"]["gladv-assadv"]["mdape"], "as_is": v["pairs"]["gl-assessor"]["mdape"],
                           "n": v["n"]} for k, v in res["subgroups"].items()},
            by_town={t: {"in_advance": v["pairs"]["gladv-assadv"]["mdape"], "as_is": v["pairs"]["gl-assessor"]["mdape"],
                         "n": v["n"]} for t, v in res["by_town"].items()})
        d = summary[year]["diff"]
        print(f"  dMdAPE as is {d['as_is']['mdape']['point']:+.2f}; in advance {d['in_advance']['mdape']['point']:+.2f} "
              f"[{d['in_advance']['mdape']['lo90']:+.2f}, {d['in_advance']['mdape']['hi90']:+.2f}]; "
              f"level-free {d['level_free']['mdape']['point']:+.2f}", flush=True)
    (OUT / "factors.json").write_text(json.dumps(factors, indent=1, default=str))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
