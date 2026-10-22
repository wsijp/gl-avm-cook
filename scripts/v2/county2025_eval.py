"""Sale-prediction test: GL against the Assessor's 2025 model on their held-out test sales.

Joins every township's predictions.parquet (county2025_predict.py) with the Assessor's test_card
(`pred_card_initial_fmv`) on (meta_pin, meta_sale_document_num, meta_card_num); all 38,177 test sales, nothing
excluded. Ratio r = prediction / sale price. Per model: MdAPE (median |r - 1|, %), median ratio, COD
(100 x mean |r - median r| / median r), PRD (mean r / (sum prediction / sum price)); level-free MdAPE (each model's
predictions divided by its own median ratio in the unit, recomputed in every resample; an after-the-fact level
diagnostic). Units (by the benchmark sale row's township code): each township, City (70-77), suburbs (10-39), county;
diagnostics: forecast horizon (2023-10..12, 2024-01..06, 2024-07..12), whether the PIN sold in the training period,
county without the 38 prorated test sales. Differences GL - Assessor with 90% percentile intervals from 2,000 paired
bootstrap resamples of test PINs (every sale of a drawn PIN, with multiplicity) within each township (seed 20261020);
pooled statistics from the concatenated sale rows; every unit uses the same resamples restricted to it. Intervals
are conditional on the fitted models and pointwise. Convergence flag (fixed before fitting): max rank-normalised R-hat > 1.05,
min bulk ESS < 100, any E-BFMI < 0.2, or > 1% of the township's test sales with latent split R-hat > 1.01; flagged
townships stay in every headline figure, and the pooled rows are repeated without them.

Usage (repo root): .venv/bin/python -I public/scripts/v2/county2025_eval.py
Writes public/runs/v2/county2025_eval/: test_sales.parquet, summary.csv, summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "public/runs/v2"
OUT = RUNS / "county2025_eval"
TC = REPO / "ccao/runs/2025_full/output/test_card/model_test_card.parquet"
KEYS = ["meta_pin", "meta_sale_document_num", "meta_card_num"]
CITY = {"70", "71", "72", "73", "74", "75", "76", "77"}
NAMES = {"10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet", "15": "Cicero",
         "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont", "20": "Leyden", "21": "Lyons",
         "22": "Maine", "23": "New Trier", "24": "Niles", "25": "Northfield", "26": "Norwood Park", "27": "Oak Park",
         "28": "Orland", "29": "Palatine", "30": "Palos", "31": "Proviso", "32": "Rich", "33": "River Forest",
         "34": "Riverside", "35": "Schaumburg", "36": "Stickney", "37": "Thornton", "38": "Wheeling", "39": "Worth",
         "70": "Hyde Park", "71": "Jefferson", "72": "Lake", "73": "Lake View", "74": "North Chicago",
         "75": "Rogers Park", "76": "South Chicago", "77": "West Chicago"}
B, SEED, Q = 2000, 20261020, (0.05, 0.95)


def wquantile(v, W):
    """median of v for each row of integer weights W (resamples x n), as np.median of the resampled rows: the middle
    row, or the mean of the two middle rows when the resample has an even number of rows (corrected after a
    review of the results; the first version took the lower middle row)"""
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(W[:, o], axis=1)
    t = cw[:, -1:]
    i1 = (cw >= np.ceil(t / 2)).argmax(axis=1)
    i2 = (cw >= np.floor(t / 2) + 1).argmax(axis=1)
    return (v[o][i1] + v[o][i2]) / 2


def stats(pred, price, W=None):
    """MdAPE (%), median ratio, COD, PRD, level-free MdAPE (%): point (W None) or per resample"""
    r = pred / price
    if W is None:
        med = np.median(r)
        return dict(mdape=100 * np.median(np.abs(r - 1)), median_ratio=med,
                    cod=100 * np.mean(np.abs(r - med)) / med, prd=np.mean(r) / (pred.sum() / price.sum()),
                    mdape_level_free=100 * np.median(np.abs(r / med - 1)))
    med = wquantile(r, W)
    sw = W.sum(axis=1)
    return dict(mdape=100 * wquantile(np.abs(r - 1), W), median_ratio=med,
                cod=100 * (W * np.abs(r[None, :] - med[:, None])).sum(axis=1) / sw / med,
                prd=(W @ r / sw) / ((W @ pred) / (W @ price)),
                mdape_level_free=100 * np.array([wquantile(np.abs(r / m - 1), W[i:i + 1])[0] for i, m in enumerate(med)]))


def main():
    global OUT
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--out":            # optional; default the published test's
        OUT = REPO / sys.argv[2]
    runs = json.loads((OUT / "runs.json").read_text())
    gl = pd.concat([pd.read_parquet(REPO / s["run"] / "predictions.parquet") for s in runs.values()
                    if s.get("status") == "done"], ignore_index=True)
    tc = pd.read_parquet(TC, columns=KEYS + ["pred_card_initial_fmv", "meta_sale_price", "meta_township_code"])
    d = tc.merge(gl.drop(columns=["meta_sale_price", "meta_township_code"]), on=KEYS, how="left", validate="one_to_one")
    missing = d[d["pred"].isna()]
    print(f"test sales {len(d):,}; with a GL prediction {d['pred'].notna().sum():,}; missing {len(missing):,} "
          f"(townships {sorted(missing['meta_township_code'].unique())})")
    d = d.rename(columns={"pred": "pred_gl", "pred_card_initial_fmv": "pred_assessor", "town": "fit_region"})
    d["town"] = d["meta_township_code"]
    d["area"] = np.where(d["town"].isin(CITY), "City", "suburbs")
    d["horizon"] = pd.cut(pd.to_datetime(d["date"]), pd.to_datetime(["2023-10-01", "2024-01-01", "2024-07-01", "2025-01-01"]),
                          right=False, labels=["2023-10..12", "2024-01..06", "2024-07..12"]).astype(str)
    d.to_parquet(OUT / "test_sales.parquet", index=False)
    both = d[d["pred_gl"].notna()].reset_index(drop=True)
    price = both["meta_sale_price"].to_numpy(float)

    # paired bootstrap weights: test PINs resampled with replacement within each township, all their sales kept
    rng = np.random.default_rng(SEED)
    W = np.zeros((B, len(both)), np.float32)
    for T, g in both.groupby("town"):
        pins, inv = np.unique(g["meta_pin"].to_numpy(), return_inverse=True)
        cnt = rng.multinomial(len(pins), np.full(len(pins), 1 / len(pins)), size=B)   # (B, PINs)
        W[:, g.index.to_numpy()] = cnt[:, inv]

    conv = pd.DataFrame(runs).T
    ps = both.groupby("fit_region")["latent_rhat"].apply(lambda x: float((x > 1.01).mean()))
    conv["share_pred_rhat_gt_1_01"] = ps.reindex(conv.index)
    conv["flagged"] = ((conv["max_rhat"] > 1.05) | (conv["min_ess_bulk"] < 100) | (conv["ebfmi_min"] < 0.2)
                       | (conv["share_pred_rhat_gt_1_01"] > 0.01))
    flagged = sorted(conv.index[conv["flagged"]].tolist())
    ok = ~both["fit_region"].isin(flagged).to_numpy()
    hist = both["has_training_sale"].to_numpy(bool)
    units = [("county", np.ones(len(both), bool)), ("City", both["area"].eq("City").to_numpy()),
             ("suburbs", both["area"].eq("suburbs").to_numpy())]
    units += [(f"{T} {NAMES.get(T, '')}", both["town"].eq(T).to_numpy()) for T in sorted(both["town"].unique())]
    units += [(f"horizon {h}", both["horizon"].eq(h).to_numpy()) for h in ["2023-10..12", "2024-01..06", "2024-07..12"]]
    units += [("PIN sold in training period", hist), ("PIN not sold in training period", ~hist),
              ("county without prorated test sales", both["meta_tieback_proration_rate"].ge(1).to_numpy())]
    if flagged:
        units += [(f"{u} without flagged townships", m & ok) for u, m in units[:3]]
    # added after a review of the results (not fixed before fitting): pooled rows without the townships whose fit had
    # divergent transitions
    div = sorted(conv.index[conv["divergent"] > 0].tolist())
    nodiv = ~both["fit_region"].isin(div).to_numpy()
    units += [(f"{u} without townships with divergences", m & nodiv) for u, m in units[:3]]
    rows = []
    for name, m in units:
        if not m.any():
            continue
        pg, pa, pr, Wm = both["pred_gl"].to_numpy(float)[m], both["pred_assessor"].to_numpy(float)[m], price[m], W[:, m]
        sg, sa = stats(pg, pr), stats(pa, pr)
        bg, ba = stats(pg, pr, Wm), stats(pa, pr, Wm)
        row = dict(unit=name, n=int(m.sum()), pins=int(both.loc[m, "meta_pin"].nunique()))
        for k in sg:
            diff = bg[k] - ba[k]
            row.update({f"{k}_gl": sg[k], f"{k}_assessor": sa[k], f"d_{k}": sg[k] - sa[k],
                        f"d_{k}_lo": float(np.quantile(diff, Q[0])), f"d_{k}_hi": float(np.quantile(diff, Q[1]))})
        rows.append(row)
        print(f"{name:20s} n {row['n']:6,}  MdAPE GL {sg['mdape']:5.2f}  Assessor {sa['mdape']:5.2f}  "
              f"diff {row['d_mdape']:+.2f} [{row['d_mdape_lo']:+.2f}, {row['d_mdape_hi']:+.2f}]  "
              f"median ratio {sg['median_ratio']:.3f} / {sa['median_ratio']:.3f}  COD {sg['cod']:.1f} / {sa['cod']:.1f}  "
              f"PRD {sg['prd']:.3f} / {sa['prd']:.3f}", flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    conv.to_csv(OUT / "convergence.csv")
    # the Assessor on all test sales whatever the GL's coverage
    pa_all, pr_all = d["pred_assessor"].to_numpy(float), d["meta_sale_price"].to_numpy(float)
    assessor_all = {u: {k: float(v) for k, v in stats(pa_all[m], pr_all[m]).items()} | dict(n=int(m.sum()))
                    for u, m in [("county", np.ones(len(d), bool)), ("City", d["area"].eq("City").to_numpy()),
                                 ("suburbs", d["area"].eq("suburbs").to_numpy())]}
    js = dict(n_test=len(d), n_with_gl=len(both), missing_townships=sorted(missing["meta_township_code"].unique().tolist()),
              assessor_all_test_sales=assessor_all, townships_with_divergences=div,
              bootstrap=dict(resamples=B, seed=SEED, cluster="meta_pin", strata="township (sale row's code)",
                             interval="90% percentile, paired, conditional on the fitted models, pointwise"),
              convergence=dict(flagged=flagged, max_rhat_max=float(conv["max_rhat"].max()),
                               min_ess_bulk_min=float(conv["min_ess_bulk"].min()), ebfmi_min=float(conv["ebfmi_min"].min()),
                               pred_rhat_max=float(conv["pred_rhat_max"].max()),
                               share_pred_rhat_gt_1_01_max=float(conv["share_pred_rhat_gt_1_01"].max()),
                               townships_max_rhat_gt_1_01=sorted(conv.index[conv["max_rhat"] > 1.01].tolist()),
                               divergent_total=int(conv["divergent"].sum()),
                               max_treedepth_share_max=float(conv["max_treedepth_share"].max()),
                               sum_to_zero_all_pass=bool(conv["sum_to_zero_pass"].all()),
                               runtime_h_total=float(conv["runtime_s"].sum() / 3600)),
              summary=rows)
    (OUT / "summary.json").write_text(json.dumps(js, indent=1, default=float))
    print(json.dumps({k: v for k, v in js.items() if k != "summary"}, indent=1, default=float))


if __name__ == "__main__":
    main()
