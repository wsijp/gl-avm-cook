"""Ratio study for any set of models on identical evaluation keys.

Statistics per model, ratio = value / price (price time-adjusted to 1 January 2023 unless a sensitivity says
otherwise): n, median ratio, COD (100 x mean |ratio - median| / median), PRD (mean ratio / (sum value / sum
price)), PRB (the published construction, as in the Assessor's assessr::prb: OLS slope of (ratio - median) /
median on log2(((value / median) + price) / 2)), MdAPE (100 x median |value - price| / price). Intervals: 90%
percentile bootstrap (default 2,000 resamples) clustered by valuation group (a singleton group is its PIN),
stratified by township, the same resamples for every model, so differences are paired. Subgroups: class group
(single-family 202-209/234/278, multi-family 211-212, town/rowhome 210/295) and prior-sale history (a training sale
of the PIN or not); each with n, MdAPE and median ratio per model and paired differences. Own-sale gap:
median ratio of PINs with a training sale minus that of PINs without, per model.

Models: --model NAME=PATH, PATH a values.parquet from value_v2.py (column `value`, or `value_rounded` with
--rounded) or, with NAME=assessor, the Assessor's benchmark parquet (column pred_pin_final_fmv_round; context
only). Every model must have a value for every evaluation key; missing values stop the run.
Pairs: --pair A-B gives differences A minus B.
Samples: primary (adjusted price; default), q1_unadjusted (Q1 2023 sales, sale price), unadjusted (all
primary sales, sale price). The stricter-validation sensitivity needs time adjustments for fall-back sales that
the time-index step did not compute (only primary sales have one); it is not implemented here.

Usage (repo root): .venv/bin/python -I public/scripts/v2/ratio_v2.py --eval public/data/v2/prep/t28/eval.parquet
    --model block=public/runs/v2/X/values.parquet --model house=public/runs/v2/Y/values.parquet --pair block-house
    [--model assessor=ccao/runs/2023_rebuild/output/assessment_pin/model_assessment_pin.parquet]
    [--sample primary] [--rounded] [--boot 2000] [--seed 1] --out ratio.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_GROUP = {**{str(c): "single-family" for c in [202, 203, 204, 205, 206, 207, 208, 209, 234, 278]},
               "211": "multi-family", "212": "multi-family", "210": "town/rowhome", "295": "town/rowhome"}
STATS = ["median_ratio", "cod", "prd", "prb", "mdape"]


def stats(v, p):
    r = v / p
    m = np.median(r)
    x = np.log2(((v / m) + p) / 2)
    y = (r - m) / m
    xc = x - x.mean()
    prb = float((xc * (y - y.mean())).sum() / (xc ** 2).sum()) if (xc ** 2).sum() > 0 else np.nan
    return dict(median_ratio=m, cod=100 * np.mean(np.abs(r - m)) / m, prd=r.mean() / (v.sum() / p.sum()),
                prb=prb, mdape=100 * np.median(np.abs(v - p) / p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--sample", choices=["primary", "q1_unadjusted", "unadjusted"], default="primary")
    ap.add_argument("--rounded", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ev = pd.concat([pd.read_parquet(p).assign(town=Path(p).parent.name.lstrip("t")) for p in a.eval], ignore_index=True)
    if a.sample == "q1_unadjusted":
        ev = ev[ev["q1"]].copy()
    ev["p"] = ev["adj_price"] if a.sample == "primary" else ev["price"]
    ev["class_group"] = ev["roll_class"].astype(str).map(CLASS_GROUP)
    ev["history"] = np.where(ev["has_training_sale"], "training sale", "no training sale")
    assert ev["class_group"].notna().all()
    names = []
    for m in a.model:
        name, path = m.split("=", 1)
        if name == "assessor":
            v = pd.read_parquet(path, columns=["meta_pin", "pred_pin_final_fmv_round"]).drop_duplicates("meta_pin")
            v = v.set_index("meta_pin")["pred_pin_final_fmv_round"]
        else:
            v = pd.read_parquet(path).set_index("meta_pin")["value_rounded" if a.rounded else "value"]
        ev[f"v:{name}"] = ev["meta_pin"].map(v).astype(float)
        miss = int(ev[f"v:{name}"].isna().sum())
        if miss:
            raise SystemExit(f"model {name}: {miss} evaluation keys without a value")
        names.append(name)
    pairs = [tuple(p.split("-", 1)) for p in a.pair]

    # Clusters (valuation groups) within strata (townships); one bootstrap index set per resample, shared by models.
    ev = ev.reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    strata = []
    for _, g in ev.groupby("town"):
        members = [ix.to_numpy() for _, ix in g.groupby("group_id").groups.items()]
        strata.append(members)
    boots = []
    for _ in range(a.boot):
        idx = []
        for members in strata:
            pick = rng.integers(0, len(members), len(members))
            idx.append(np.concatenate([members[i] for i in pick]))
        boots.append(np.concatenate(idx))

    def table(mask):
        """Point estimates and paired bootstrap intervals for the rows in `mask` (boolean array)."""
        sub = np.flatnonzero(mask)
        P = ev["p"].to_numpy()
        V = {n: ev[f"v:{n}"].to_numpy() for n in names}
        point = {n: stats(V[n][sub], P[sub]) for n in names}
        bs = {n: {s: [] for s in STATS} for n in names}
        for b in boots:
            bi = b[mask[b]]
            if len(bi) < 2:
                continue
            for n in names:
                st = stats(V[n][bi], P[bi])
                for s in STATS:
                    bs[n][s].append(st[s])
        out = dict(n=int(len(sub)), n_clusters=int(ev.loc[sub, "group_id"].nunique()), models={}, pairs={})
        for n in names:
            out["models"][n] = {s: dict(point=float(point[n][s]), lo90=float(np.quantile(bs[n][s], 0.05)),
                                        hi90=float(np.quantile(bs[n][s], 0.95))) for s in STATS}
        for x, y in pairs:
            out["pairs"][f"{x}-{y}"] = {}
            for s in STATS:
                d = np.array(bs[x][s]) - np.array(bs[y][s])
                out["pairs"][f"{x}-{y}"][s] = dict(point=float(point[x][s] - point[y][s]),
                                                   lo90=float(np.quantile(d, 0.05)), hi90=float(np.quantile(d, 0.95)))
        return out, bs

    res = dict(sample=a.sample, rounded=a.rounded, boot=a.boot, seed=a.seed, evals=a.eval, models=a.model,
               cluster="valuation group (singleton = PIN), stratified by township")
    res["overall"], bs_all = table(np.ones(len(ev), bool))
    res["by_town"] = {t: table((ev["town"] == t).to_numpy())[0] for t in sorted(ev["town"].unique())} if ev["town"].nunique() > 1 else {}
    res["subgroups"] = {}
    for col in ["class_group", "history"]:
        for lev in sorted(ev[col].unique()):
            res["subgroups"][f"{col}={lev}"] = table((ev[col] == lev).to_numpy())[0]
    # Own-sale gap per model and its paired difference in absolute value.
    has = ev["has_training_sale"].to_numpy()
    P = ev["p"].to_numpy()
    gap = {}
    for n in names:
        r = ev[f"v:{n}"].to_numpy() / P
        g = [np.median(r[b][has[b]]) - np.median(r[b][~has[b]]) for b in boots]
        gap[n] = dict(point=float(np.median(r[has]) - np.median(r[~has])), lo90=float(np.quantile(g, 0.05)),
                      hi90=float(np.quantile(g, 0.95)), draws=g)
    res["own_sale_gap"] = {n: {k: v for k, v in gap[n].items() if k != "draws"} for n in names}
    for x, y in pairs:
        d = np.abs(np.array(gap[x]["draws"])) - np.abs(np.array(gap[y]["draws"]))
        res["own_sale_gap"][f"abs {x} - abs {y}"] = dict(point=abs(gap[x]["point"]) - abs(gap[y]["point"]),
                                                         lo90=float(np.quantile(d, 0.05)), hi90=float(np.quantile(d, 0.95)))
    Path(a.out).write_text(json.dumps(res, indent=1))
    np.savez_compressed(Path(a.out).with_suffix(".boot.npz"),
                        **{f"{n}:{s}": np.array(bs_all[n][s]) for n in names for s in STATS})
    o = res["overall"]
    for n in names:
        print(n, {s: round(o["models"][n][s]["point"], 4) for s in STATS})
    for p, d in o["pairs"].items():
        print(p, {s: (round(d[s]["point"], 3), round(d[s]["lo90"], 3), round(d[s]["hi90"], 3)) for s in STATS})
    print("own-sale gap", json.dumps(res["own_sale_gap"]))


if __name__ == "__main__":
    main()
