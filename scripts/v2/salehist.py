"""A property's sale history known at sale time, and its diagnosis on training data.

history(d, source): for every row of d (needs meta_pin, date), from the sales in `source` (the 2025 input's rows, any
flag; a sale = a distinct (meta_pin, meta_sale_document_num), counted once per date) strictly before the row's date:
  n_prev        number of earlier sale dates of the PIN in the data
  months_since  months since the latest earlier sale (NaN if none)
  first_new     the PIN's first sale in the data was in or before its year built (first-sale row's char_yrblt) and
                this row is a later sale (n_prev >= 1)
The data start on 1 January 2016: "no earlier sale" means none since 2016 (left-censored), and gaps longer than the
time since 2016 cannot be seen. Dates only; no earlier price is used.

Diagnosis (main): training sales (before 1 May 2023) of the six development townships (77, 73, 76, 74, 31, 11),
from the model P preparation (prep_loop_p; P = the published model:
its coded characteristics, neighbourhood). Per township and segment (houses = single-family + town/rowhouses; flats =
211-212), OLS of log price on neighbourhood x year fixed effects (absorbed by demeaning), the prep's characteristics
without the sale count, and one history coding at a time; standard errors clustered by PIN. Codings compared by an
inner time split of the training data (fit before 1 May 2022, predict 1 May 2022 - 30 Apr 2023; MdAPE): no history,
the 4-year count (as now), sold before, sold before + gap bands, sold before + log months. Plus the new-build second
sale check. Development sales (1 May - 19 October 2023) are not used for any choice; one descriptive table of P's
development ratios by gap band (P, seed 1) is printed at the end, labelled as such.

Usage (repo root): .venv/bin/python -I public/scripts/v2/salehist.py
Writes notes/v2/salehist_diag/diag.md, coef.csv, codings.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
INPUT = REPO / "ccao/inputs/2025/training_data.parquet"
OUT = REPO / "notes/v2/salehist_diag"
TOWNS = ["77", "73", "76", "74", "31", "11"]
FLAT = ("211", "212")
BANDS = [0, 12, 24, 48, 72, 1e9]
BAND_NAMES = ["gap<1y", "gap1-2y", "gap2-4y", "gap4-6y", "gap6y+"]


def source(before=None):
    s = pd.read_parquet(INPUT, columns=["meta_pin", "meta_sale_document_num", "meta_sale_date", "char_yrblt"])
    s["date"] = pd.to_datetime(s["meta_sale_date"])
    if before is not None:
        s = s[s["date"] < before]
    return s.drop_duplicates(["meta_pin", "meta_sale_document_num"])


def history(d, src):
    u = src.sort_values(["meta_pin", "date"]).drop_duplicates(["meta_pin", "date"])[["meta_pin", "date", "char_yrblt"]]
    u = u.reset_index(drop=True)
    g = u.groupby("meta_pin", sort=False)
    u["n_prev"] = g.cumcount()
    u["prev_date"] = g["date"].shift()
    first = g.first()
    u["first_new"] = (u["n_prev"] >= 1) & (u["meta_pin"].map(first["date"]).dt.year
                                            <= u["meta_pin"].map(first["char_yrblt"]).astype(float))
    m = d[["meta_pin", "date"]].merge(u[["meta_pin", "date", "n_prev", "prev_date", "first_new"]],
                                      on=["meta_pin", "date"], how="left", validate="many_to_one")
    assert m["n_prev"].notna().all(), "every row's own sale must be in the source"
    out = pd.DataFrame(index=d.index)
    out["n_prev"] = m["n_prev"].to_numpy().astype(int)
    out["months_since"] = ((m["date"] - m["prev_date"]).dt.days / 30.4375).to_numpy()
    out["first_new"] = m["first_new"].to_numpy().astype(bool)
    return out


def codings(h):
    """history regressors for each candidate coding (unstandardised)."""
    sold = (h["n_prev"] > 0).astype(float)
    band = pd.cut(h["months_since"], BANDS, right=False, labels=BAND_NAMES)
    B = pd.get_dummies(band).astype(float).reindex(columns=BAND_NAMES, fill_value=0.0)
    lm = np.where(sold > 0, np.log(h["months_since"].clip(lower=1)), 0.0)
    return {"none": pd.DataFrame(index=h.index),
            "count4y": h[["count4y"]].astype(float),
            "sold": pd.DataFrame({"sold": sold}),
            "sold+bands": B.drop(columns=["gap6y+"]).assign(sold=sold)[["sold"] + BAND_NAMES[:-1]],
            "sold+logm": pd.DataFrame({"sold": sold, "sold_logm": lm}),
            "bands+new": B.drop(columns=["gap6y+"]).assign(sold=sold, first_new=h["first_new"].astype(float))}


def demean(X, gid):
    return X - X.groupby(gid).transform("mean")


def main():
    import statsmodels.api as sm
    OUT.mkdir(parents=True, exist_ok=True)
    src = source(before="2023-10-20")
    rows_c, rows_k, L = [], [], []
    allh = []
    for T in TOWNS:
        t = pd.read_parquet(REPO / f"public/data/v2/prep_loop_p/t{T}/train.parquet")
        t["is_flat"] = t["meta_class"].astype(str).isin(FLAT)
        h = history(t, src)
        h["count4y"] = t["meta_sale_count_past_n_years"].astype(float)
        h["town"], h["is_flat"], h["date"] = T, t["is_flat"], t["date"]
        allh.append(h)
        xc = [c for c in t.columns if c.startswith("x:") and "sale_count" not in c]
        for seg, sm_ in (("houses", ~t["is_flat"]), ("flats", t["is_flat"])):
            tt, hh = t[sm_].copy(), h[sm_].copy()
            if len(tt) < 300:
                continue
            gid = tt["nbhd_id"].astype(str) + "|" + tt["date"].dt.year.astype(str)
            C = codings(hh)
            for name, H in C.items():
                X = pd.concat([tt[xc], H], axis=1)
                X = X.loc[:, X.std() > 0]
                y = tt["log_price"]
                # full training: coefficients
                Xd, yd = demean(X, gid), y - y.groupby(gid).transform("mean")
                r = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(tt["meta_pin"])[0]})
                for c in H.columns:
                    if c in r.params:
                        rows_k.append(dict(town=T, segment=seg, coding=name, term=c, coef=r.params[c], se=r.bse[c],
                                           n_term=int((H[c] != 0).sum()), n=len(tt)))
                # inner time split: fit before 2022-05, predict 2022-05..2023-04 (FE: nbhd x year from the fit part;
                # validation year 2022/2023 -> its own nbhd's latest fitted year effect would be needed; use nbhd FE +
                # month FE instead for the split)
                fit_m = tt["date"] < "2022-05-01"
                gid2 = tt["nbhd_id"].astype(str)
                Xf = X[fit_m]
                mon = tt["date"].dt.to_period("M").astype(str)
                Mf = pd.get_dummies(mon[fit_m], drop_first=True).astype(float)
                # neighbourhood FE by demeaning (fit part), month effects as dummies (fit part); validation sales take
                # the last fitted month's effect (held flat, as the GL's prediction rule)
                Z = pd.concat([Xf, Mf], axis=1)
                Z = Z.loc[:, Z.std() > 0]
                gm = Z.groupby(gid2[fit_m]).transform("mean")
                ym = y[fit_m].groupby(gid2[fit_m]).transform("mean")
                b = np.linalg.lstsq((Z - gm).to_numpy(), (y[fit_m] - ym).to_numpy(), rcond=None)[0]
                bb = pd.Series(b, index=Z.columns)
                alpha = (y[fit_m] - Z @ bb).groupby(gid2[fit_m]).mean()
                v = ~fit_m & gid2.isin(alpha.index)
                Xv = X[v].reindex(columns=Z.columns, fill_value=0.0)
                last = Mf.columns[-1]
                for c in Mf.columns:
                    if c in Xv.columns:
                        Xv[c] = 1.0 if c == last else 0.0
                pred = Xv @ bb + gid2[v].map(alpha)
                ape = np.abs(np.exp(pred - y[v]) - 1)
                rr = np.exp(pred - y[v])
                hv = hh[v]
                rows_c.append(dict(town=T, segment=seg, coding=name, n_val=int(v.sum()), mdape=100 * np.median(ape),
                                   mdape_resold=100 * np.median(ape[hv["n_prev"] > 0]),
                                   mdape_other=100 * np.median(ape[hv["n_prev"] == 0]),
                                   ratio_resold=float(np.median(rr[hv["n_prev"] > 0])),
                                   ratio_other=float(np.median(rr[hv["n_prev"] == 0])),
                                   n_val_resold=int((hv["n_prev"] > 0).sum())))
        print(f"t{T} done", flush=True)
    K, C = pd.DataFrame(rows_k), pd.DataFrame(rows_c)
    K.to_csv(OUT / "coef.csv", index=False)
    C.to_csv(OUT / "codings.csv", index=False)
    H = pd.concat(allh)

    # ------------------------------------------------------------ report
    L += ["# Sale history on training data (D52) — diagnosis", "",
          "_salehist.py, Claude Code (sale-history thread), 2026-10-20. Training sales (before 1 May 2023) of the loop "
          "townships (testbed 77, 73, 76, 74; guards 31, 11), P preparation. History from the 2025 input's sales (any "
          "flag) strictly before each sale's date; data start 2016-01 (left-censored). Houses = single-family + "
          "town/rowhouses; flats = 211-212; condominium units are not in the sample._", ""]
    L += ["## 1. How common", "", "| town | segment | sales | sold before | gap<1y | 1-2y | 2-4y | 4-6y | 6y+ | "
          "4-year count >= 1 | later sale after new-build first sale |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (T, f), g in H.groupby(["town", "is_flat"], sort=False):
        band = pd.cut(g["months_since"], BANDS, right=False, labels=BAND_NAMES).value_counts()
        L.append(f"| {T} | {'flats' if f else 'houses'} | {len(g):,} | {(g['n_prev'] > 0).sum():,} | "
                 + " | ".join(f"{band[b]:,}" for b in BAND_NAMES) + f" | {(g['count4y'] >= 1).sum():,} | {g['first_new'].sum():,} |")
    # agreement of the 4-year count with history
    w4 = (H["n_prev"] > 0) & (H["months_since"] < 48)
    L += ["", f"The Assessor's 4-year count against our history (training rows): count >= 1 on {int((H['count4y'] >= 1).sum()):,} "
          f"rows; an earlier sale within 48 months in our data on {int(w4.sum()):,}; both {int(((H['count4y'] >= 1) & w4).sum()):,}; "
          f"count >= 1 but no earlier sale within 48 months {int(((H['count4y'] >= 1) & ~w4).sum()):,}; "
          f"earlier sale within 48 months but count 0 {int(((H['count4y'] == 0) & w4).sum()):,}.", ""]
    # left-censoring: share with an earlier sale by sale year
    L += ["Left-censoring: share of training sales with an earlier sale in the data, by sale year: "
          + ", ".join(f"{y} {100 * g['n_prev'].gt(0).mean():.0f}%" for y, g in H.groupby(H['date'].dt.year)), ""]
    L += ["## 2. Premiums (OLS on training sales; log points, clustered SE)", "",
          "Coding `bands+new`: sold before (reference gap: 6 years or more) + gap bands + later sale after a new-build "
          "first sale. Coding `sold+logm`: sold before + log months since the last sale (times sold before). "
          "`count4y`: the Assessor's count (as now).", "",
          "| town | segment | coding | term | coef | se | n with term |", "|---|---|---|---|---:|---:|---:|"]
    for _, r in K[K["coding"].isin(["count4y", "sold", "sold+logm", "bands+new"])].iterrows():
        L.append(f"| {r.town} | {r.segment} | {r.coding} | {r.term} | {r.coef:+.3f} | {r.se:.3f} | {r.n_term:,} |")
    L += ["", "## 3. Codings compared on an inner time split of the training data", "",
          "Fit on training sales before 1 May 2022 (neighbourhood + month effects, characteristics, history coding), "
          "predict 1 May 2022 - 30 Apr 2023 with the month effect held at April 2022. MdAPE (%) of the validation "
          "sales; resold = with an earlier sale in the data.", "",
          "| town | segment | n | resold | " + " | ".join(C["coding"].unique()) + " |", "|---|---|---:|---:|" + "---:|" * C["coding"].nunique()]
    for (T, seg), g in C.groupby(["town", "segment"], sort=False):
        g = g.set_index("coding")
        L.append(f"| {T} | {seg} | {g['n_val'].iloc[0]:,} | {g['n_val_resold'].iloc[0]:,} | "
                 + " | ".join(f"{g.loc[c, 'mdape']:.2f}" for c in C["coding"].unique()) + " |")
    L += ["", "Resold validation sales: MdAPE / median ratio by coding:", ""]
    for (T, seg), g in C.groupby(["town", "segment"], sort=False):
        g = g.set_index("coding")
        L.append(f"- {T} {seg}: " + "; ".join(f"{c} {g.loc[c, 'mdape_resold']:.1f} / {g.loc[c, 'ratio_resold']:.3f}"
                                             for c in C["coding"].unique()))
    # pooled (sales-weighted) summary of the coding comparison
    L += ["", "Pooled over towns and segments (validation-sales-weighted mean of MdAPE; resold; other):", ""]
    for c, g in C.groupby("coding", sort=False):
        w = g["n_val"]
        L.append(f"- {c}: all {np.average(g['mdape'], weights=w):.2f}; resold {np.average(g['mdape_resold'], weights=g['n_val_resold']):.2f} "
                 f"(ratio {np.average(g['ratio_resold'], weights=g['n_val_resold']):.3f}); other "
                 f"{np.average(g['mdape_other'], weights=w - g['n_val_resold']):.2f} (ratio {np.average(g['ratio_other'], weights=w - g['n_val_resold']):.3f})")

    # ------------------------------------------------------------ descriptive: P's development ratios by gap band
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import loop_eval as le
        s = le.sales()
        p, _ = le.load_arm("P_s1")
        a, _ = le.load_arm("Assessor")
        s = s.merge(p, on=le.KEYS).merge(a.rename(columns={"pred": "pred_a"}), on=le.KEYS)
        d = pd.read_parquet(INPUT, columns=le.KEYS + ["meta_sale_date"])
        s = s.merge(d.drop_duplicates(le.KEYS), on=le.KEYS)
        s["date"] = pd.to_datetime(s["meta_sale_date"])
        hs = history(s, src)
        s = pd.concat([s, hs], axis=1)
        s["band"] = np.where(s["n_prev"] == 0, "no earlier sale",
                             pd.cut(s["months_since"], BANDS, right=False, labels=BAND_NAMES).astype(str))
        s["new"] = np.where(s["first_new"], "after new-build first sale", "")
        s["rg"], s["ra"] = s["pred"] / s["meta_sale_price"], s["pred_a"] / s["meta_sale_price"]
        L += ["", "## 4. Descriptive only (development sales; not used for any choice): P seed 1 and the Assessor's "
              "development refit, median ratio and MdAPE by gap band", "",
              "| region | band | n | GL ratio | Assessor ratio | GL MdAPE | Assessor MdAPE |", "|---|---|---:|---:|---:|---:|---:|"]
        s["region"] = np.where(s["town"].isin(le.TESTBED), "testbed", "guards")
        for (rg, b), g in s.groupby(["region", "band"]):
            L.append(f"| {rg} | {b} | {len(g):,} | {g['rg'].median():.3f} | {g['ra'].median():.3f} | "
                     f"{100 * (g['rg'] - 1).abs().median():.1f} | {100 * (g['ra'] - 1).abs().median():.1f} |")
        for rg, g in s[s["first_new"]].groupby("region"):
            L.append(f"| {rg} | after new-build first sale | {len(g):,} | {g['rg'].median():.3f} | {g['ra'].median():.3f} | "
                     f"{100 * (g['rg'] - 1).abs().median():.1f} | {100 * (g['ra'] - 1).abs().median():.1f} |")
    except Exception as e:  # noqa: BLE001
        L += ["", f"(descriptive development table not produced: {e!r})"]
    (OUT / "diag.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
