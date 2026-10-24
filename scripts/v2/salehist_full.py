"""Complete sale history from the Assessor's Parcel Sales open data (wvhk-k5uv, 1999 onwards), and the left-censoring
check of the sale-history model's history on training data (H2: separate house and flat models with each home's sale
history: an earlier sale since 2016, and the gap since it in bands).

Why: H2's history (salehist.py) comes from the 2025 input, whose sales start in January 2016, so a 2016 training sale
is coded "not sold before" even when it sold in 2013. The Assessor's 4-year count has no such censoring: their ingest
pulls sales from min_sale_year - n_years_prior = 2012 (ccao/model-res-avm/pipeline/00-ingest.R lines 108-110), counts
(330-355), then filters training sales to 2016-2024 (383-386).

source(before): the earlier sales a history may use, filtered as close as possible to the Assessor's sale query
(00-ingest.R lines 87-114) and count (330-355):
  - residential classes of their model input (the classes in the 2025 input: 202-212, 218, 219, 234, 278, 295) in the
    record's year (their INNER JOIN on vw_card_res_input by pin and year);
  - deed_type Warranty / Trustee / Other (= their codes 01 / 02 / 05; mapping checked on the 65,003 input sales of
    the six development townships, all matched by PIN and document number);
  - not is_multisale, not sale_filter_same_sale_within_365, not sale_filter_less_than_10k, not sale_filter_deed_type;
  - not an outlier (their count uses earlier sales with !sv_is_outlier): the flag exists only for sales in the 2025
    input (2016 onwards; matched by PIN and document number); earlier sales cannot be screened (deviation, stated);
  - one record per PIN x document; a sale that is in the 2025 input takes the input's date (0.17% of matched sales
    have a different date in the open data), so a row's own sale is never "earlier" than itself;
  - dates strictly before `before` (default: the start of the Assessor's 2025 test sales, 2023-10-20, so no test sale
    is read).
  Dates only: no earlier price is used.

history(d, src): for each row of d (meta_pin, date), from src strictly before the row's date:
  n_prev       number of earlier sale dates (since 1999)
  n_prev_4y    number within 4 years (the Assessor's window: more than 0 days and at most 4 years before)
  months_since months since the latest earlier sale (NaN if none)

check: the censoring check, training sales (before 1 May 2023) of the H2 preparations, houses and flats separately
-> notes/v2/morning_diag/censoring.md.

Usage (repo root): .venv/bin/python -I public/scripts/v2/salehist_full.py check
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PARCEL = REPO / "public/data/v2/parcel_sales/sales_t77-73-76-74-31-11.parquet"
# Lake 72 and Hyde Park 70, a second download read together with the first (townships disjoint, so the six
# townships' histories are unchanged: checked)
PARCEL_FILES = [PARCEL, REPO / "public/data/v2/parcel_sales/sales_t72-70.parquet"]
INPUT = REPO / "ccao/inputs/2025/training_data.parquet"
OUT = REPO / "notes/v2/morning_diag"
RES_CLASSES = ("202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "218", "219", "234", "278",
               "295")
DEEDS = ("Warranty", "Trustee", "Other")                 # = the Assessor's '01', '02', '05'
HELD_OUT_START = "2023-10-20"
TOWNS = ["77", "73", "76", "74", "31", "11"]
TOWNS_DOWNLOADED = TOWNS + ["72", "70"]                  # townships with a Parcel Sales download
TESTBED = ["77", "73", "76", "74"]
MONTH = 30.4375


def source(before=HELD_OUT_START, stats=None):
    p = pd.concat([pd.read_parquet(f, columns=["pin", "year", "class", "sale_date", "doc_no", "deed_type", "is_multisale",
                                               "sale_filter_same_sale_within_365", "sale_filter_less_than_10k",
                                               "sale_filter_deed_type"]) for f in PARCEL_FILES], ignore_index=True)
    n0 = len(p)
    p["doc"] = p["doc_no"].str.strip()
    keep = (p["class"].isin(RES_CLASSES) & p["deed_type"].isin(DEEDS) & p["is_multisale"].eq("false")
            & p["sale_filter_same_sale_within_365"].eq("false") & p["sale_filter_less_than_10k"].eq("false")
            & p["sale_filter_deed_type"].eq("false"))
    p = p[keep].drop_duplicates(["pin", "doc"]).copy()
    p["date"] = pd.to_datetime(p["sale_date"]).dt.normalize()
    i = pd.read_parquet(INPUT, columns=["meta_pin", "meta_sale_document_num", "meta_sale_date", "sv_is_outlier"])
    i = i.drop_duplicates(["meta_pin", "meta_sale_document_num"])
    i["idate"] = pd.to_datetime(i["meta_sale_date"])
    p = p.merge(i.rename(columns={"meta_pin": "pin", "meta_sale_document_num": "doc"})[["pin", "doc", "idate", "sv_is_outlier"]],
                on=["pin", "doc"], how="left", validate="one_to_one")
    in_input = p["idate"].notna()
    redated = int((in_input & (p["idate"] != p["date"])).sum())
    p.loc[in_input, "date"] = p.loc[in_input, "idate"]
    outlier = p["sv_is_outlier"].eq(True)
    p = p[~outlier & (p["date"] < before)]
    if stats is not None:
        stats.update(rows_downloaded=n0, kept_by_filter=int(keep.sum()), in_2025_input=int(in_input.sum()),
                     redated_to_input=redated, outliers_dropped=int(outlier.sum()), before=str(before),
                     kept=len(p), first_date=str(p["date"].min().date()))
    return p[["pin", "doc", "date"]].reset_index(drop=True)


def history(d, src):
    """d: rows with meta_pin, date. Earlier sale dates strictly before each row's date (one per PIN x date)."""
    u = src.drop_duplicates(["pin", "date"]).sort_values(["pin", "date"])
    dates = {k: g["date"].to_numpy() for k, g in u.groupby("pin", sort=False)}
    out = np.zeros((len(d), 3))
    out[:, 2] = np.nan
    four = np.timedelta64(int(round(4 * 365.25)), "D")
    for j, (pin, day) in enumerate(zip(d["meta_pin"].to_numpy(), pd.to_datetime(d["date"]).to_numpy())):
        x = dates.get(pin)
        if x is None:
            continue
        k = np.searchsorted(x, day, side="left")           # x[:k] < day
        if k == 0:
            continue
        out[j, 0] = k
        out[j, 1] = k - np.searchsorted(x, day - four, side="left")
        out[j, 2] = (day - x[k - 1]) / np.timedelta64(1, "D") / MONTH
    return pd.DataFrame({"n_prev": out[:, 0].astype(int), "n_prev_4y": out[:, 1].astype(int), "months_since": out[:, 2]},
                        index=d.index)


# ------------------------------------------------------------------------------------------------- censoring check
def partial_residual(run, t):
    """H2 fit `run` on its training rows t: log price minus (centred x) beta, mu[month], zeta[nbhd] at posterior means;
    = intercept + u + psi + eps (the intercept and the u / psi draws are not stored). Centred per fit afterwards."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    z = np.load(REPO / "public/runs/v2" / run / "draws.npz")
    xcols = [str(c) for c in z["xcols"]]
    X = t[xcols].to_numpy(float)
    b, mu, zeta = z["beta"].mean(0), z["mu"].mean(0), z["zeta"].mean(0)
    units = pd.read_parquet(REPO / "public/data/v2" / PREPS[run.split("_")[1]] / f"t{run.split('_t')[-1]}" / "units.parquet")
    kid = {k: i for i, k in enumerate(sorted(units["nbhd_id"].unique()))}
    r = (t["log_price"].to_numpy() - (X - X.mean(0)) @ b - mu[t["month"].to_numpy() - 1]
         - zeta[t["nbhd_id"].map(kid).to_numpy()])
    return r - np.median(r)


PREPS = {"H2h": "prep_loop_h2h", "H2f": "prep_loop_h2f"}


def group_of(n16, n99, m99):
    """current coding (since 2016) against complete history (since 1999)."""
    return np.where(n16 > 0, "resold since 2016",
                    np.where(n99 == 0, "never sold since 1999",
                             np.where(m99 < 48, "censored: sold < 4 y before, pre-2016", "censored: sold >= 4 y before, pre-2016")))


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    st = {}
    src = source(stats=st)
    L = ["# Left-censoring of the sale history (MORNING, D56-D57): training data", "",
         "_salehist_full.py check, Claude Code (morning thread), 2026-10-21. Training sales (before 1 May 2023) of the "
         "H2 preparations (houses `prep_loop_h2h`, flats `prep_loop_h2f`), testbed 77, 73, 76, 74 and guards 31, 11. "
         "Current coding = H2's history from the 2025 input (sales since January 2016). Complete history = the "
         "Parcel Sales open data (wvhk-k5uv) from 1999, filtered as the Assessor's sale query (module docstring). "
         "Houses = single-family + town/rowhouses; flats = two- to six-flat buildings (211-212); condominium units are "
         "not in the sample._", "",
         f"Source: {st['rows_downloaded']:,} records downloaded; {st['kept_by_filter']:,} pass the class / deed / "
         f"flag filter (one per PIN x document: {st['kept']:,} kept before {st['before']} after dropping "
         f"{st['outliers_dropped']:,} flagged outliers); {st['in_2025_input']:,} are 2025-input sales "
         f"({st['redated_to_input']:,} re-dated to the input's date); first date {st['first_date']}.", ""]
    rows, devrows = [], []
    for seg, arm in (("houses", "H2h"), ("flats", "H2f")):
        for T in TOWNS:
            o = REPO / "public/data/v2" / PREPS[arm] / f"t{T}"
            t = pd.read_parquet(o / "train.parquet")
            h = history(t, src)
            x = pd.DataFrame(dict(town=T, seg=seg, year=t["date"].dt.year, n16=t["hist_n_prev"].to_numpy(),
                                  m16=t["hist_months"].to_numpy(), n99=h["n_prev"].to_numpy(), n4=h["n_prev_4y"].to_numpy(),
                                  m99=h["months_since"].to_numpy(), count4y=t["meta_sale_count_past_n_years"].astype(float).to_numpy(),
                                  res=partial_residual(f"loop_{arm}_s1_t{T}", t)))
            rows.append(x)
            r = pd.read_parquet(o / "roll.parquet")
            hr = history(r, src)
            p = pd.read_parquet(REPO / f"public/runs/v2/loop_{arm}_s1_t{T}/predictions.parquet")
            r = r.merge(p[["meta_pin", "meta_sale_document_num", "meta_card_num", "pred"]],
                        on=["meta_pin", "meta_sale_document_num", "meta_card_num"], how="left", validate="one_to_one")
            devrows.append(pd.DataFrame(dict(town=T, seg=seg, n16=r["hist_n_prev"].to_numpy(), n99=hr["n_prev"].to_numpy(),
                                             m99=hr["months_since"].to_numpy(), ratio=(r["pred"] / r["meta_sale_price"]).to_numpy())))
    D = pd.concat(rows, ignore_index=True)
    D["group"] = group_of(D["n16"], D["n99"], D["m99"])
    # 1. agreement of our 4-year count with the Assessor's
    agree = (np.minimum(D["n4"], 9) == np.minimum(D["count4y"], 9))
    L += ["## 1. Does the filtered open data reproduce the Assessor's 4-year count?", "",
          f"On {len(D):,} training sales: our count within 4 years equals `meta_sale_count_past_n_years` on "
          f"{100 * agree.mean():.2f}%; count > 0 agrees on {100 * ((D['n4'] > 0) == (D['count4y'] > 0)).mean():.2f}% "
          f"(theirs > 0, ours 0: {int(((D['count4y'] > 0) & (D['n4'] == 0)).sum()):,}; ours > 0, theirs 0: "
          f"{int(((D['count4y'] == 0) & (D['n4'] > 0)).sum()):,}). By year (count > 0 agreement): "
          + ", ".join(f"{y} {100 * ((g['n4'] > 0) == (g['count4y'] > 0)).mean():.1f}%" for y, g in D.groupby("year")), ""]
    # 2. shares by year
    L += ["## 2. Share of training sales with an earlier sale, by year (%)", "",
          "| segment | year | sales | current coding: sold since 2016 | complete: within 4 y | complete: since 1999 | "
          "current 'not sold' that sold before 2016 (share of the reference group) |", "|---|---|---:|---:|---:|---:|---:|"]
    for (seg, y), g in D.groupby(["seg", "year"], sort=False):
        ref = g["n16"] == 0
        L.append(f"| {seg} | {y} | {len(g):,} | {100 * (g['n16'] > 0).mean():.1f} | {100 * (g['n4'] > 0).mean():.1f} | "
                 f"{100 * (g['n99'] > 0).mean():.1f} | {100 * (g.loc[ref, 'n99'] > 0).mean():.1f} |")
    # 3. residual level by year and group
    L += ["", "## 3. H2's partial residual by year and history group (median, log points)", "",
          "Partial residual = log price minus H2's fitted characteristics and history terms, township path and "
          "neighbourhood level (posterior means, seed 1); it still contains the block / building effect and the "
          "neighbourhood x month term; centred on each fit's median. A censored resale (sold before 2016, coded "
          "'not sold') whose residual sits above the never-sold group's means the reference level mixes two groups. "
          "Cells: median (n).", ""]
    for seg in ("houses", "flats"):
        g0 = D[D["seg"] == seg]
        groups = ["never sold since 1999", "censored: sold < 4 y before, pre-2016", "censored: sold >= 4 y before, pre-2016",
                  "resold since 2016"]
        L += [f"**{seg}**", "", "| year | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
        for y, g in g0.groupby("year"):
            L.append(f"| {y} | " + " | ".join(
                (f"{g.loc[g['group'] == k, 'res'].median():+.3f} ({(g['group'] == k).sum():,})" if (g["group"] == k).sum() >= 20
                 else f"– ({(g['group'] == k).sum():,})") for k in groups) + " |")
        L.append(f"| all | " + " | ".join(f"{g0.loc[g0['group'] == k, 'res'].median():+.3f} ({(g0['group'] == k).sum():,})"
                                          for k in groups) + " |")
        L.append("")
    # 4. development sales (descriptive): H2's 'other' group split by complete history
    V = pd.concat(devrows, ignore_index=True)
    V["group"] = group_of(V["n16"], V["n99"], V["m99"])
    V["region"] = np.where(V["town"].isin(TESTBED), "testbed", "guards")
    L += ["## 4. Development sales (descriptive): H2 seed 1 by the same groups", "",
          "Median ratio / MdAPE (%) (n). The 'other' group of `salehist_diag/dev_groups.md` is the first three columns "
          "together.", "", "| region | segment | never sold since 1999 | censored < 4 y | censored >= 4 y | resold since 2016 |",
          "|---|---|---|---|---|---|"]
    for (rg, seg), g in V.groupby(["region", "seg"], sort=False):
        cells = []
        for k in ["never sold since 1999", "censored: sold < 4 y before, pre-2016", "censored: sold >= 4 y before, pre-2016",
                  "resold since 2016"]:
            q = g.loc[g["group"] == k, "ratio"]
            cells.append(f"{q.median():.3f} / {100 * (q - 1).abs().median():.1f} ({len(q):,})" if len(q) >= 10 else f"– ({len(q)})")
        L.append(f"| {rg} | {seg} | " + " | ".join(cells) + " |")
    D.to_parquet(OUT / "censoring_train.parquet", index=False)
    V.to_parquet(OUT / "censoring_dev.parquet", index=False)
    (OUT / "censoring.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("DONE")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    else:
        raise SystemExit(__doc__)
