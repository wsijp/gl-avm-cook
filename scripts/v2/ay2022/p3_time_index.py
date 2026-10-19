# 2022 version of scripts/v2/p3_time_index.py: the same code with the 2022 assessment year's dates, file names and
# north-suburb labels, and the differences listed in REPRODUCE.md ("What to expect").
"""Time adjustment of 2022 evaluation sales to the 1 January 2022 lien date.

One common, sales-based, cross-fitted index for the north triad (the benchmark area), using no model
value or ratio. Specification:
  log(price) = month effect (24 months, 2021-01 .. 2022-12) + neighbourhood effect + class effect
               + b1 log(building sq ft) + b2 log(land sq ft) + b3 age + b4 age^2 + error,
OLS on single-card, non-prorated, non-outlier sales of model classes in the north triad, 2021-2022, from the
2025 input (one validation vintage; sale-year characteristics). Lien-date level L = mean of the December 2021
and January 2022 month effects. A sale in month m is adjusted by exp(L - d_m): adjusted price = price x that.
Cross-fitting: PINs are split into 5 folds by a hash of the PIN; the adjustment for a sale in fold k uses the
month effects estimated without fold k, so no evaluation sale's own price enters its own adjustment.
Uncertainty: standard errors clustered by PIN (full-sample fit).

Usage (repo root): .venv/bin/python -I public/scripts/v2/ay2022/p3_time_index.py ccao/inputs public/data/v2/ay2022 <figure dir>
Writes p8_time_index.csv (full-sample index, CI, fold range),
p8_time_index_folds.csv, p8_eval_time_adjustment.parquet (one row per primary evaluation sale, all triads:
fold and log adjustment; outside the north the north index is listed but its use there is not proposed)
and the figure p8_time_index.png.
"""
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

inp, out, figdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
figdir.mkdir(parents=True, exist_ok=True)
K = 5
MODEL_CLASSES = ["202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "234", "278", "295"]


def fold_of(pin):
    return int(hashlib.md5(pin.encode()).hexdigest(), 16) % K


e = pd.read_parquet(inp / "2025/training_data.parquet", columns=[
    "meta_pin", "meta_card_num", "meta_sale_date", "meta_sale_document_num", "meta_sale_price", "meta_class",
    "meta_nbhd_code", "meta_triad_code", "ind_pin_is_multicard", "meta_pin_num_cards", "ind_pin_is_prorated",
    "meta_tieback_proration_rate", "sv_is_outlier", "char_bldg_sf", "char_land_sf", "char_yrblt"])
e["date"] = pd.to_datetime(e["meta_sale_date"])
e = e[e["date"].dt.year.isin([2021, 2022])]
e = e.sort_values(["meta_pin", "meta_sale_document_num", "meta_card_num"]).drop_duplicates(
    ["meta_pin", "meta_sale_document_num"])
n0 = len(e)
e = e[(e["meta_triad_code"] == "2") & ~e["sv_is_outlier"].astype(bool) & ~e["ind_pin_is_multicard"].astype(bool)
      & (e["meta_pin_num_cards"] == 1) & ~e["ind_pin_is_prorated"].astype("boolean").fillna(True).astype(bool)
      & (e["meta_tieback_proration_rate"] >= 1) & e["meta_class"].isin(MODEL_CLASSES)]
e = e[(e["char_bldg_sf"] > 0) & (e["char_land_sf"] > 0) & e["char_yrblt"].notna()]
e["month"] = e["date"].dt.strftime("%Y-%m")
e["age"] = (e["date"].dt.year - e["char_yrblt"]).clip(lower=0) / 10
e["lp"] = np.log(e["meta_sale_price"])
e["lb"], e["ll"] = np.log(e["char_bldg_sf"]), np.log(e["char_land_sf"])
e["fold"] = e["meta_pin"].map(fold_of)
print(f"2021-2022 sale documents {n0}; index sample (north, filters above) {len(e)}; PINs {e['meta_pin'].nunique()};"
      f" 2022 sales {int((e['date'].dt.year == 2022).sum())}")
print("sales per month:", e["month"].value_counts().sort_index().to_dict())
F = "lp ~ 0 + C(month) + C(meta_nbhd_code) + C(meta_class) + lb + ll + age + I(age**2)"
MONTHS = sorted(e["month"].unique())
assert len(MONTHS) == 24


def month_effects(df, cluster=False):
    fit = smf.ols(F, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["meta_pin"])[0]}) \
        if cluster else smf.ols(F, data=df).fit()
    names = [f"C(month)[{m}]" for m in MONTHS]
    return fit, fit.params[names].to_numpy(), names


fit, d, names = month_effects(e, cluster=True)
print(f"full fit: n {int(fit.nobs)}, R2 {fit.rsquared:.3f}; b(log bldg sf) {fit.params['lb']:.3f},"
      f" b(log land sf) {fit.params['ll']:.3f}")
V = fit.cov_params().loc[names, names].to_numpy()
iD, iJ = MONTHS.index("2021-12"), MONTHS.index("2022-01")
L = (d[iD] + d[iJ]) / 2
Cm = np.eye(24)
Cm[:, iD] -= 0.5
Cm[:, iJ] -= 0.5  # row m of Cm @ d = d_m - L
idx = Cm @ d
se = np.sqrt(np.diag(Cm @ V @ Cm.T))

fold_idx = {}
for k in range(K):
    _, dk, _ = month_effects(e[e["fold"] != k])
    fold_idx[k] = Cm @ dk
fi = pd.DataFrame(fold_idx, index=MONTHS)
tab = pd.DataFrame({"log_index": idx, "se": se, "lo95": idx - 1.96 * se, "hi95": idx + 1.96 * se,
                    "index": np.exp(idx), "n_sales": e["month"].value_counts().reindex(MONTHS).to_numpy(),
                    "fold_min": fi.min(axis=1), "fold_max": fi.max(axis=1)}, index=MONTHS)
tab["adjustment_factor"] = np.exp(-tab["log_index"])
pd.set_option("display.width", 200)
print(tab.round(4).to_string())
m23 = [m for m in MONTHS if m.startswith("2022")]
print("2022 months: largest |fold - full| in log index:", f"{(fi.loc[m23].sub(tab.loc[m23, 'log_index'], axis=0)).abs().max().max():.4f}")
print("2022 log index range:", f"{tab.loc[m23, 'log_index'].min():.4f} .. {tab.loc[m23, 'log_index'].max():.4f}")
tab.to_csv(out / "p8_time_index.csv", index_label="month")
fi.to_csv(out / "p8_time_index_folds.csv", index_label="month")

# Per evaluation sale: fold and log adjustment (L - d_m) from the fold-excluded fit.
ev = pd.read_parquet(out / "p8_eval_sales.parquet")
ev = ev[ev["eligible_primary"]].copy()
ev["month"] = ev["date"].dt.strftime("%Y-%m")
ev["fold"] = ev["meta_pin"].map(fold_of)
ev["log_adjustment"] = [-fi.loc[m, f] for m, f in zip(ev["month"], ev["fold"])]
ev[["meta_pin", "meta_sale_document_num", "town", "triad", "month", "fold", "log_adjustment"]].to_parquet(
    out / "p8_eval_time_adjustment.parquet", index=False)
print("evaluation sales with an adjustment:", len(ev), "; north:", int((ev["triad"] == "2").sum()))

# Figure: the 2021-2022 index (log scale shown as % from the lien-date level) with its 95% interval.
x = pd.to_datetime([m + "-15" for m in MONTHS])
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
pct = lambda v: 100 * (np.exp(v) - 1)
ax.fill_between(x, pct(tab["lo95"]), pct(tab["hi95"]), color="#2a78d6", alpha=0.18, linewidth=0)
ax.plot(x, pct(tab["log_index"]), color="#2a78d6", linewidth=2, marker="o", markersize=4)
ax.axhline(0, color="#52514e", linewidth=0.8)
ax.axvline(pd.Timestamp("2022-01-01"), color="#52514e", linewidth=0.8, linestyle=(0, (3, 3)))
ax.text(pd.Timestamp("2022-01-08"), ax.get_ylim()[1] * 0.92, "lien date 1 Jan 2022", color="#52514e", fontsize=8)
ax.set_ylabel("price level vs 1 Jan 2022 (%)", color="#0b0b0b", fontsize=9)
ax.set_title("North triad: monthly hedonic time index, 2021-2022 (95% interval, clustered by PIN)",
             color="#0b0b0b", fontsize=10, loc="left")
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
for s_ in ["left", "bottom"]:
    ax.spines[s_].set_color("#c9c8c2")
ax.tick_params(colors="#52514e", labelsize=8)
fig.tight_layout()
fig.savefig(figdir / "p8_time_index.png", facecolor=fig.get_facecolor())
print("figure:", figdir / "p8_time_index.png")
