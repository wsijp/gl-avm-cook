"""Evaluation eligibility, property-state comparability, outcome-source reconciliation and
historical sales outside the valued population (counts only).

Inputs: 2023 sales in the 2025 model input (outcomes only), the 2023 roll and training data, the
PIN -> unit table of blocks.py (supported flag), and the open-data Parcel Sales of 2023 (p3_fetch_opendata_sales.py).
Prices are compared between sources for identity only (equal or not); no price is summarised here and no
model value exists yet.

Usage (repo root): .venv/bin/python -I public/scripts/v2/p3_eligibility.py ccao/inputs public/data/v2
Writes p3_eval_cascade_township.csv, p3_eval_sales.parquet (one row
per 2023 sale document with every rule's flag; the primary set is `eligible_primary`),
p3_reconciliation.csv, p3_historical_township.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

inp, out = Path(sys.argv[1]), Path(sys.argv[2])
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
MODEL_CLASSES = ["202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "234", "278", "295"]
# Physical characteristics compared between the roll (2022 record) and the sale-year record.
PHYS = ["char_bldg_sf", "char_land_sf", "char_rooms", "char_beds", "char_fbath", "char_hbath", "char_gar1_size",
        "char_bsmt", "char_attic_fnsh", "char_yrblt"]
NAMES = {"10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet", "15": "Cicero",
         "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont", "20": "Leyden", "21": "Lyons",
         "22": "Maine", "23": "New Trier", "24": "Niles", "25": "Northfield", "26": "Norwood Park",
         "27": "Oak Park", "28": "Orland", "29": "Palatine", "30": "Palos", "31": "Proviso", "32": "Rich",
         "33": "River Forest", "34": "Riverside", "35": "Schaumburg", "36": "Stickney", "37": "Thornton",
         "38": "Wheeling", "39": "Worth", "70": "Hyde Park", "71": "Jefferson", "72": "Lake", "73": "Lake View",
         "74": "North Chicago", "75": "Rogers Park", "76": "South Chicago", "77": "West Chicago"}
ROLE = {"28": "engineering", "32": "pilot (fallback engineering)", "27": "pilot", "30": "pilot", "15": "pilot",
        "21": "confirmation", "12": "confirmation", "34": "confirmation", "39": "confirmation (proposed)",
        "31": "confirmation (proposed)", "19": "smoke", "33": "smoke"}


def h(t):
    print(f"\n=== {t}", flush=True)


# ------------------------------------------------------------------ roll (one row per PIN, card 1)
a = pd.read_parquet(inp / "2023/assessment_data.parquet", columns=[
    "meta_pin", "meta_card_num", "meta_class", "meta_township_code", "meta_triad_code", "meta_nbhd_code",
    "hie_num_expired"] + PHYS)
a = a.sort_values(["meta_pin", "meta_card_num"]).drop_duplicates("meta_pin").set_index("meta_pin")
b = pd.read_parquet(out / "blocks_2023.parquet", columns=["pin", "on_roll", "supported", "category", "unit_id"])
b = b[b["on_roll"]].set_index("pin")
assert b.index.is_unique and len(b) == len(a)
a = a.join(b[["supported", "category", "unit_id"]])
print("roll PINs", len(a), "; supported", int(a["supported"].sum()))

# ------------------------------------------------------------------ 2023 sales in the 2025 input
e = pd.read_parquet(inp / "2025/training_data.parquet", columns=[
    "meta_pin", "meta_card_num", "meta_sale_date", "meta_sale_document_num", "meta_sale_price",
    "meta_sale_deed_type", "meta_class", "meta_township_code", "meta_triad_code", "meta_nbhd_code",
    "ind_pin_is_multicard", "ind_pin_is_prorated", "meta_tieback_proration_rate", "hie_num_active",
    "char_recent_renovation", "sv_is_outlier", "sv_outlier_reason1", "sv_outlier_reason2", "sv_outlier_reason3",
    "sv_run_id"] + PHYS)
e["date"] = pd.to_datetime(e["meta_sale_date"])
e = e[e["date"].dt.year == 2023].sort_values(["meta_pin", "meta_sale_document_num", "meta_card_num"])
cards = e.groupby(["meta_pin", "meta_sale_document_num"]).size().rename("n_card_rows")
s = e.drop_duplicates(["meta_pin", "meta_sale_document_num"]).set_index(["meta_pin", "meta_sale_document_num"])
s = s.join(cards).reset_index()
assert not s["meta_sale_document_num"].duplicated().any()  # no multi-PIN document in the input
reasons = s[["sv_outlier_reason1", "sv_outlier_reason2", "sv_outlier_reason3"]]
s["ptax203"] = reasons.eq("PTAX-203 Exclusion").any(axis=1)
s["family"] = reasons.eq("Family Sale").any(axis=1)
print("2023 sale documents", len(s), "on", s["meta_pin"].nunique(), "PINs; card rows", len(e))

# ------------------------------------------------------------------ reconciliation with open data
h("Outcome-source reconciliation (document level) against open-data Parcel Sales (wvhk-k5uv)")
o = pd.read_parquet(out / "p3_opendata_sales_2023.parquet")
o["num_parcels_sale"] = pd.to_numeric(o["num_parcels_sale"])
o["date"] = pd.to_datetime(o["sale_date"]).dt.normalize()
o["price"] = pd.to_numeric(o["sale_price"])
print("open-data rows dated 2023:", len(o), "; documents", o["doc_no"].nunique())
o_res = o[o["class"].isin(MODEL_CLASSES + ["201", "218", "219"])]
# Aligned filters: the Assessor's upstream ingest drops multi-PIN sales and the three sale filters.
o_al = o_res[~o_res["is_multisale"] & (o_res["num_parcels_sale"] <= 1) & ~o_res["sale_filter_same_sale_within_365"]
             & ~o_res["sale_filter_less_than_10k"] & ~o_res["sale_filter_deed_type"]]
print("residential-class rows", len(o_res), "; after aligned filters (single-PIN, three sale filters)", len(o_al))
key_o = set(zip(o_al["pin"], o_al["doc_no"]))
key_s = set(zip(s["meta_pin"], s["meta_sale_document_num"]))
key_o_all = set(zip(o["pin"], o["doc_no"]))
docs_multi = set(o.loc[o["is_multisale"] | (o["num_parcels_sale"] > 1), "doc_no"])
both = key_o & key_s
inp_only = key_s - key_o
src_only = key_o - key_s
s["in_source_aligned"] = [k in key_o for k in zip(s["meta_pin"], s["meta_sale_document_num"])]
s["in_source_any"] = [k in key_o_all for k in zip(s["meta_pin"], s["meta_sale_document_num"])]
s["source_multisale"] = s["meta_sale_document_num"].isin(docs_multi)
rec = pd.DataFrame({"count": {
    "matched (PIN, document)": len(both), "input only": len(inp_only), "source only (aligned filters)": len(src_only),
    "input only, but in source under another filter or class": int((~s["in_source_aligned"] & s["in_source_any"]).sum()),
    "input only, absent from source entirely": int((~s["in_source_any"]).sum()),
    "input documents that are multi-PIN in source": int(s["source_multisale"].sum())}})
print(rec)
m = s[s["in_source_aligned"]].merge(o_al[["pin", "doc_no", "date", "price", "class", "deed_type"]],
                                    left_on=["meta_pin", "meta_sale_document_num"], right_on=["pin", "doc_no"])
print("matched: price identical", f"{(m['price'] == m['meta_sale_price']).mean():.4%}")
m = m.rename(columns={"date_x": "date_input", "date_y": "date_source"})
print("matched: date identical", f"{(m['date_source'] == m['date_input'].dt.normalize()).mean():.4%}",
      "; class identical", f"{(m['class'] == m['meta_class']).mean():.4%}")
print("deed type input x source (matched):")
print(pd.crosstab(m["meta_sale_deed_type"], m["deed_type"]))
# Source-only documents: where are they in the input? (other year, other PIN, absent)
so = o_al.set_index(["pin", "doc_no"]).loc[list(src_only)].reset_index()
e_all = pd.read_parquet(inp / "2025/training_data.parquet", columns=["meta_pin", "meta_sale_document_num", "meta_sale_date"])
docs_any = set(e_all["meta_sale_document_num"])
so["doc_in_input_any_year"] = so["doc_no"].isin(docs_any)
so["pin_on_roll"] = so["pin"].isin(a.index)
so["pin_sold_2023_in_input"] = so["pin"].isin(set(s["meta_pin"]))
print("source-only documents:", len(so), "; document elsewhere in input:", int(so["doc_in_input_any_year"].sum()),
      "; PIN on 2023 roll:", int(so["pin_on_roll"].sum()), "; PIN has another 2023 sale in input:",
      int(so["pin_sold_2023_in_input"].sum()))
print("source-only by class:", so["class"].value_counts().head(10).to_dict())
print("source-only by deed type:", so["deed_type"].value_counts().to_dict())
inp_only_df = s[~s["in_source_aligned"]]
print("input-only by outlier flag:", inp_only_df["sv_is_outlier"].value_counts().to_dict(),
      "; by class:", inp_only_df["meta_class"].value_counts().head(8).to_dict())
rec.to_csv(out / "p3_reconciliation.csv")

# ------------------------------------------------------------------ roll status and property state
s = s.join(a[["meta_class", "meta_nbhd_code", "meta_township_code", "meta_triad_code", "supported", "category",
              "unit_id", "hie_num_expired"] + PHYS].add_prefix("roll_"), on="meta_pin")
s["on_roll"] = s["roll_supported"].notna()
s["town"] = s["roll_meta_township_code"].fillna(s["meta_township_code"])
s["triad"] = s["roll_meta_triad_code"].fillna(s["meta_triad_code"])
s["class_changed"] = s["on_roll"] & (s["meta_class"] != s["roll_meta_class"])
s["nbhd_changed"] = s["on_roll"] & (s["meta_nbhd_code"] != s["roll_meta_nbhd_code"])

h("Property state: sale-year (2023) record vs roll (2022 record) for on-roll, supported PINs")
on = s[s["on_roll"] & s["roll_supported"].astype(bool) & ~s["sv_is_outlier"]]
diffs = {}
for c in PHYS:
    x, y = on[c], on["roll_" + c]
    d = ~((x == y) | (x.isna() & y.isna()))
    s.loc[on.index, "diff_" + c] = d
    diffs[c] = int(d.sum())
print("non-outlier sales on supported roll PINs:", len(on))
print("records differ by characteristic:", diffs)
dc = [f"diff_{c}" for c in PHYS]
s[dc] = s[dc].astype("boolean").fillna(False).astype(bool)
size_keys = ["char_bldg_sf", "char_land_sf", "char_rooms", "char_beds", "char_fbath", "char_hbath", "char_gar1_size",
             "char_bsmt", "char_attic_fnsh"]
s["phys_changed"] = s[[f"diff_{c}" for c in size_keys]].any(axis=1)
s["yrblt_changed_only"] = s["diff_char_yrblt"] & ~s["phys_changed"]
s["hie_active_at_sale"] = s["hie_num_active"].fillna(0) > 0
s["roll_hie_expired"] = s["roll_hie_num_expired"].fillna(0) > 0
on = s.loc[on.index]
print("any of", size_keys, "differs:", int(on["phys_changed"].sum()), f"({on['phys_changed'].mean():.2%})")
print("year built differs only:", int(on["yrblt_changed_only"].sum()))
print("active HIE in the sale-year record:", int(on["hie_active_at_sale"].sum()),
      "; of which physical record differs:", int((on["hie_active_at_sale"] & on["phys_changed"]).sum()))
print("roll PIN with hie_num_expired > 0 (improvement now on the roll):", int(on["roll_hie_expired"].sum()),
      "; of which physical record differs:", int((on["roll_hie_expired"] & on["phys_changed"]).sum()))
print("recent-renovation indicator (2025 feature) set:", int(on["char_recent_renovation"].fillna(False).astype(bool).sum()))
print("building area change, sale-year minus roll (sq ft), where it differs:")
dd = (on["char_bldg_sf"] - on["roll_char_bldg_sf"])[on["diff_char_bldg_sf"]]
print(dd.describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_dict())
print("share of those with |change| > 10%:", f"{(dd.abs() / on.loc[dd.index, 'roll_char_bldg_sf'] > .1).mean():.2%}")

# ------------------------------------------------------------------ eligibility cascade
h("Eligibility cascade (sale documents; each step applied to those passing the previous)")
steps = [
    ("all 2023 single-PIN sale documents", pd.Series(True, index=s.index)),
    ("validation: not sv_is_outlier (2025-input vintage)", ~s["sv_is_outlier"].astype(bool)),
    ("not a multi-PIN document in open data", ~s["source_multisale"]),
    ("PIN on the 2023 roll", s["on_roll"]),
    ("PIN in the supported population (D25/D29)", s["roll_supported"].fillna(False).astype(bool)),
    ("class unchanged between roll and sale record", ~s["class_changed"]),
    ("physical record unchanged (size keys) and no active HIE at sale", ~s["phys_changed"] & ~s["hie_active_at_sale"]),
]
keep = pd.Series(True, index=s.index)
casc = []
for name, f in steps:
    keep = keep & f
    casc.append((name, keep.copy()))
    print(f"{name}: {int(keep.sum())} documents, {s.loc[keep, 'meta_pin'].nunique()} PINs")
s["eligible_doc"] = keep
strict_doc = keep & ~s["ptax203"] & ~s["family"]
strict_first = s[strict_doc].sort_values(["meta_pin", "date", "meta_sale_document_num"]).drop_duplicates("meta_pin")
s["eligible_primary_strict"] = s.index.isin(strict_first.index)
first = s[keep].sort_values(["meta_pin", "date", "meta_sale_document_num"]).drop_duplicates("meta_pin")
s["eligible_primary"] = s.index.isin(first.index)
print("one sale per PIN (earliest eligible in 2023):", int(s["eligible_primary"].sum()))
print("repeat eligible 2023 sales dropped by the one-per-PIN rule:", int(keep.sum() - s["eligible_primary"].sum()))
s["q1"] = s["date"].dt.month <= 3
print("primary in Q1 2023:", int((s["eligible_primary"] & s["q1"]).sum()))
print("neighbourhood changed (kept; valued in the roll neighbourhood):",
      int((s["eligible_primary"] & s["nbhd_changed"]).sum()))
print("stricter validation sensitivity: primary sales with a PTAX-203 exclusion or family-sale reason:",
      int((s["eligible_primary"] & (s["ptax203"] | s["family"])).sum()),
      "; stricter rule applied before the earliest-sale selection:", int(s["eligible_primary_strict"].sum()),
      "PINs (south", int((s["eligible_primary_strict"] & (s["triad"] == "3")).sum()), ")")
print("primary sales whose roll PIN is a unit (210/295):", int((s["eligible_primary"] & s["meta_class"].isin(["210", "295"])).sum()))

h("Dispositions of non-outlier 2023 sales that fail a rule (first failing rule)")
ok = ~s["sv_is_outlier"].astype(bool)
fail = pd.Series("", index=s.index)
for name, f in steps[2:]:
    fail = fail.where(fail != "", np.where(ok & ~f, name, ""))
print(fail[ok & (fail != "")].value_counts())
print("off-roll non-outlier sales by class:", s.loc[ok & ~s["on_roll"], "meta_class"].value_counts().to_dict())
print("on-roll excluded (category) among non-outlier sales:",
      s.loc[ok & s["on_roll"] & ~s["roll_supported"].fillna(False).astype(bool), "roll_category"].value_counts().to_dict())
print("class changes (supported, non-outlier) roll -> sale:",
      s.loc[ok & s["roll_supported"].fillna(False).astype(bool) & s["class_changed"]].groupby(
          ["roll_meta_class", "meta_class"]).size().sort_values(ascending=False).head(12).to_dict())

h("Per township (documents after each step; last columns: primary PINs, Q1 primary, units among primary)")
tab = pd.DataFrame({name: s[f].groupby(s["town"]).size() for name, f in casc}).fillna(0).astype(int)
tab.columns = ["all", "valid", "not_multi", "on_roll", "supported", "class_same", "state_same"]
tab["primary"] = s[s["eligible_primary"]].groupby("town").size()
tab["primary_q1"] = s[s["eligible_primary"] & s["q1"]].groupby("town").size()
tab["primary_units"] = s[s["eligible_primary"] & s["meta_class"].isin(["210", "295"])].groupby("town").size()
p2 = pd.read_csv(out / "p2_eval_sale_counts_township.csv", dtype={"meta_pin": str})  # first column is the township
tab["p2_count"] = p2.set_index("meta_pin")["pins_2023_supported"]
tab = tab.fillna(0).astype(int)
tab.insert(0, "name", [NAMES.get(t, t) for t in tab.index])
tab.insert(1, "triad", s.groupby("town")["triad"].first().reindex(tab.index).map({"1": "City", "2": "North", "3": "South"}))
tab.insert(2, "role", [ROLE.get(t, "") for t in tab.index])
tab = tab.sort_values(["triad", "primary"], ascending=[False, False])
print(tab.to_string())
print("totals:", tab.drop(columns=["name", "triad", "role"]).sum().to_dict())
print("south triad totals:", tab[tab["triad"] == "South"].drop(columns=["name", "triad", "role"]).sum().to_dict())
tab.to_csv(out / "p3_eval_cascade_township.csv")

keepcols = ["meta_pin", "meta_sale_document_num", "date", "meta_sale_price", "meta_sale_deed_type", "meta_class",
            "meta_nbhd_code", "town", "triad", "roll_unit_id", "roll_meta_nbhd_code", "roll_meta_class",
            "sv_is_outlier", "ptax203", "family", "source_multisale", "in_source_aligned", "on_roll", "roll_supported",
            "class_changed", "nbhd_changed", "phys_changed", "yrblt_changed_only", "hie_active_at_sale",
            "roll_hie_expired", "eligible_doc", "eligible_primary", "eligible_primary_strict", "q1"]
s[keepcols].to_parquet(out / "p3_eval_sales.parquet", index=False)

# ------------------------------------------------------------------ historical sales outside the valued population
h("Historical (training) sales 2014-2022 by roll status (the Assessor's training filter: not multicard, not outlier)")
t = pd.read_parquet(inp / "2023/training_data.parquet", columns=[
    "meta_pin", "meta_card_num", "meta_sale_document_num", "meta_class", "meta_nbhd_code", "meta_township_code",
    "meta_triad_code", "ind_pin_is_multicard", "sv_is_outlier", "meta_tieback_proration_rate", "ind_pin_is_prorated"])
# As R's filter(!ind_pin_is_multicard, !sv_is_outlier): a missing flag drops the row.
t = t[t["ind_pin_is_multicard"].eq(False).fillna(False).astype(bool) & t["sv_is_outlier"].eq(False).fillna(False).astype(bool)]
print("training rows after the filter:", len(t), "; PINs", t["meta_pin"].nunique())
bt = pd.read_parquet(out / "blocks_2023.parquet", columns=["pin", "on_roll", "unit_id"]).set_index("pin")
t = t.join(a[["meta_class", "meta_nbhd_code", "category"]].add_prefix("roll_"), on="meta_pin")
t = t.join(bt["unit_id"], on="meta_pin")
t["prorated_at_sale"] = (t["meta_tieback_proration_rate"] < 1) | t["ind_pin_is_prorated"].astype("boolean").fillna(False)
t["status"] = np.select(
    [t["roll_category"].isna(), t["roll_category"] != "supported",
     t["meta_class"] != t["roll_meta_class"], t["meta_nbhd_code"] != t["roll_meta_nbhd_code"]],
    ["off-roll PIN", "roll PIN excluded: " + t["roll_category"].fillna(""), "supported, class changed",
     "supported, neighbourhood changed"], "supported, unchanged")
st = t["status"].value_counts()
print(pd.DataFrame({"sales": st, "share": (st / len(t)).round(4)}))
print("placed in a block unit:", int(t["unit_id"].notna().sum()), "; not placed:", int(t["unit_id"].isna().sum()),
      t.loc[t["unit_id"].isna(), "status"].value_counts().to_dict())
t.loc[t["unit_id"].isna(), ["meta_pin", "meta_sale_document_num", "meta_card_num", "meta_township_code", "status"]].to_csv(
    out / "p3_historical_unplaced.csv", index=False)
print("prorated at sale on PINs supported on the roll:", int((t["prorated_at_sale"] & t["status"].str.startswith("supported")).sum()))
print("prorated at sale (single-card):", int(t["prorated_at_sale"].sum()))
print("classes outside the model classes:", t.loc[~t["meta_class"].isin(MODEL_CLASSES), "meta_class"].value_counts().to_dict())
ht = t.groupby(["meta_township_code", "status"]).size().unstack(fill_value=0)
ht.insert(0, "name", [NAMES.get(x, x) for x in ht.index])
ht.to_csv(out / "p3_historical_township.csv")
print(ht.loc[[k for k in ROLE if k in ht.index]].to_string())
