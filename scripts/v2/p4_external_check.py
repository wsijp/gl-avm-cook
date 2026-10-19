"""Loose external check of the rebuilt 2023 run against the Assessor's published 2023 mailed values.

Applies the pass/fail bands fixed before the check was run (listed at the end of this script). Ratio = rebuilt pred_pin_final_fmv_round / (10 x mailed_tot), south-triad
PINs in classes 202-212, 234, 278, 295. No sale price is read.

Usage (repo root): .venv/bin/python -I public/scripts/v2/p4_external_check.py ccao public/data/v2
Writes the table public/data/v2/p4_external_check_township.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

cc, data = Path(sys.argv[1]), Path(sys.argv[2])
CLASSES = ["202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "234", "278", "295"]
ap = pd.read_parquet(cc / "runs/2023_rebuild/output/assessment_pin/model_assessment_pin.parquet",
                     columns=["meta_pin", "pred_pin_final_fmv_round"])
roll = pd.read_parquet(cc / "inputs/2023/assessment_data.parquet",
                       columns=["meta_pin", "meta_class", "meta_triad_code", "meta_township_code"]).drop_duplicates("meta_pin")
sup = pd.read_parquet(data / "blocks_2023.parquet", columns=["pin", "on_roll", "supported"])
sup = sup[sup["on_roll"]].set_index("pin")["supported"]
pub = pd.read_parquet(data / "p4_published_2023_south.parquet")
pub["ref"] = 10 * pd.to_numeric(pub["mailed_tot"])
pub["ref_cert"] = 10 * pd.to_numeric(pub["certified_tot"])
pub["ref_board"] = 10 * pd.to_numeric(pub["board_tot"])
assert pub["pin"].is_unique

d = roll[(roll["meta_triad_code"] == "3") & roll["meta_class"].isin(CLASSES)].merge(ap, on="meta_pin", how="left")
d["supported"] = d["meta_pin"].map(sup).fillna(False).astype(bool)
d = d.merge(pub[["pin", "ref", "ref_cert", "ref_board", "class"]], left_on="meta_pin", right_on="pin", how="left")
print("south roll PINs in model classes:", len(d), "; supported:", int(d["supported"].sum()))
print("rebuilt value missing:", int(d["pred_pin_final_fmv_round"].isna().sum()),
      "; not in published 2023:", int(d["pin"].isna().sum()),
      "; published mailed value missing or zero:", int((d["pin"].notna() & ~(d["ref"] > 0)).sum()))
print("published class differs from roll class:", int((d["pin"].notna() & (d["class"] != d["meta_class"])).sum()))
d["ok"] = d["pred_pin_final_fmv_round"].notna() & (d["ref"] > 0)
d["ratio"] = d["pred_pin_final_fmv_round"] / d["ref"]


def stats(x):
    r = x["ratio"]
    rho = spearmanr(x["pred_pin_final_fmv_round"], x["ref"]).statistic if len(x) > 2 else np.nan
    return pd.Series({"n": len(x), "median_ratio": r.median(), "spearman": rho,
                      "within_5pct": (r.sub(1).abs() <= .05).mean(), "within_10pct": (r.sub(1).abs() <= .10).mean(),
                      "within_20pct": (r.sub(1).abs() <= .20).mean(), "off_by_factor_2": int(((r > 2) | (r < .5)).sum()),
                      "median_ratio_certified": (x["pred_pin_final_fmv_round"] / x["ref_cert"]).median(),
                      "median_ratio_board": (x["pred_pin_final_fmv_round"] / x["ref_board"]).median()})


m = d[d["ok"]]
ms = m[m["supported"]]
cov = d.loc[d["supported"], "ok"].mean()
pooled, pooled_s = stats(m), stats(ms)
tw = m.groupby("meta_township_code").apply(stats)
tws = ms.groupby("meta_township_code").apply(stats)
pd.set_option("display.width", 220)
print("\nPooled (all matched):\n", pooled.round(4).to_string())
print("\nPooled (supported):\n", pooled_s.round(4).to_string())
print("\nPer township (all matched):\n", tw.round(4).to_string())
print("\nPer township (supported):\n", tws.round(4).to_string())
tw.join(tws.add_suffix("_supported")).to_csv(data / "p4_external_check_township.csv")

print("\n=== Pre-registered bands (P4 report §3)")
bands = {
    "1 coverage of supported south PINs >= 0.95": (cov, cov >= .95),
    "2 pooled median ratio in [0.95, 1.05]": (pooled["median_ratio"], .95 <= pooled["median_ratio"] <= 1.05),
    "3 every township median ratio in [0.90, 1.10]": ((tw["median_ratio"].min(), tw["median_ratio"].max()),
                                                       bool(tw["median_ratio"].between(.90, 1.10).all())),
    "4a pooled Spearman >= 0.90": (pooled["spearman"], pooled["spearman"] >= .90),
    "4b every township Spearman >= 0.80": (tw["spearman"].min(), bool((tw["spearman"] >= .80).all())),
}
for k, (v, p) in bands.items():
    print(f"{k}: value {v} -> {'PASS' if p else 'FAIL'}")
print("ALL BANDS:", "PASS" if all(p for _, p in bands.values()) else "FAIL")
