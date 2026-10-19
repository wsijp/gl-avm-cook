"""Roll inventory, training-sale counts, unit-group audit and graph cost per township.

Uses only the 2023 roll (assessment_data), 2023 complex ids and 2023 training data (sales 2014–2022).
No 2023 outcome prices. Training prices are used only for per-township counts (share under $75k,
median price) for township selection (training-period data only).

Graph cost follows an earlier graph-Laplacian construction: one block per neighbourhood, kNN-6
within the block, fragments bridged, dense eigendecomposition per block; the alpha basis is padded to
(nodes x (largest block - 1)). Nodes here: houses plus one node per town/rowhome complex group, over
the roll (no sale needed).

Usage (repo root): .venv/bin/python -I public/scripts/v2/p2_population.py ccao/inputs public/data/v2
Prints a summary and writes CSV tables to the output directory.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csgraph
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

inp, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
FT = 0.3048
K_NN = 6
NT = 108  # months 2014-01 .. 2022-12 (time graph of a 2014–2022 fit)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def h(t):
    print(f"\n=== {t}", flush=True)


# ------------------------------------------------------------------ roll, one row per PIN
CH = ["char_bldg_sf", "char_yrblt", "char_rooms", "char_beds", "char_fbath", "char_hbath", "char_land_sf",
      "char_gar1_size", "char_bsmt", "char_attic_fnsh"]
a = pd.read_parquet(inp / "2023/assessment_data.parquet", columns=[
    "meta_pin", "meta_card_num", "meta_class", "meta_township_code", "meta_township_name", "meta_triad_code",
    "meta_nbhd_code", "meta_pin_num_cards", "ind_pin_is_multicard", "ind_pin_is_prorated",
    "meta_tieback_proration_rate", "meta_tieback_key_pin", "loc_x_3435", "loc_y_3435"] + CH)
a = a.sort_values(["meta_pin", "meta_card_num"])
pin = a.groupby("meta_pin").agg(
    cls=("meta_class", "first"), town=("meta_township_code", "first"), town_name=("meta_township_name", "first"),
    triad=("meta_triad_code", "first"), nbhd=("meta_nbhd_code", "first"), cards=("meta_card_num", "size"), num_cards=("meta_pin_num_cards", "first"),
    multicard=("ind_pin_is_multicard", "first"), prorated_flag=("ind_pin_is_prorated", "first"),
    rate=("meta_tieback_proration_rate", "first"), x=("loc_x_3435", "first"), y=("loc_y_3435", "first"),
    **{c: (c, "first") for c in CH})
# Predicate: prorated = flag set OR rate < 1; multicard = flag set,
# flag missing, more than one card row, or card count missing. Exceptions are counted below.
pin["flag_missing"] = pin["multicard"].isna() | pin["num_cards"].isna()
pin["multicard"] = pin["multicard"].astype("boolean").fillna(True).astype(bool) | (pin["cards"] != 1) | pin["flag_missing"]
pin["prorated"] = (pin["rate"] < 1) | pin["prorated_flag"].astype("boolean").fillna(False).astype(bool)
print("exceptions: flagged prorated with rate 1:", int((pin["prorated_flag"].astype("boolean").fillna(False) & (pin["rate"] >= 1)).sum()),
      "; multicard flag or card count missing:", int(pin["flag_missing"].sum()))
pin["no_xy"] = pin["x"].isna() | pin["y"].isna()
pin["unit"] = pin["cls"].isin(["210", "295"])
pin["odd_class"] = ~pin["cls"].isin(["202", "203", "204", "205", "206", "207", "208", "209", "210", "211",
                                     "212", "234", "278", "295"])
pin["supported"] = ~pin["multicard"] & ~pin["prorated"] & ~pin["no_xy"] & ~pin["odd_class"]
pin["x_m"], pin["y_m"] = pin["x"] * FT, pin["y"] * FT

h("Roll inventory (PINs)")
print("PINs", len(pin), "; supported (single-card, not prorated, coordinates, model class):", pin["supported"].sum(),
      f"({pin['supported'].mean():.2%})")
cat = np.select([pin["odd_class"], pin["no_xy"], pin["multicard"] & pin["prorated"], pin["multicard"], pin["prorated"]],
                ["class outside model classes", "no coordinates", "multicard and prorated", "multicard", "prorated"],
                "supported")
pin["category"] = cat
print(pd.Series(cat).value_counts().to_string())
print("\nprorated PINs by class:"); print(pin[pin["prorated"]]["cls"].value_counts().head(15).to_string())
print("\nprorated share within class:"); print(pin.groupby("cls")["prorated"].mean().round(4).to_string())
print("\nmulticard share within class:"); print(pin.groupby("cls")["multicard"].mean().round(4).to_string())
print("\nproration: PINs per tieback key (prorated PINs):")
tk = a.drop_duplicates("meta_pin").set_index("meta_pin")["meta_tieback_key_pin"]
pin["tieback"] = tk
print(pin[pin["prorated"]].groupby("tieback").size().describe().to_string())
inv = pin.pivot_table(index=["triad", "town", "town_name"], columns="category", values="cls", aggfunc="size",
                      fill_value=0)
inv["total"] = inv.sum(axis=1)
inv["supported_share"] = (inv["supported"] / inv["total"]).round(4)
inv.to_csv(out / "p2_roll_inventory_township.csv")
cl = pin.pivot_table(index="cls", columns="category", values="town", aggfunc="size", fill_value=0)
cl.to_csv(out / "p2_roll_inventory_class.csv")
print("\nby class:"); print(cl.to_string())

# ------------------------------------------------------------------ training sales per township
h("Training sales (2023 input, 01-train.R filter: not multicard, not outlier)")
t = pd.read_parquet(inp / "2023/training_data.parquet", columns=[
    "meta_pin", "meta_sale_date", "meta_sale_price", "meta_class", "meta_township_code", "ind_pin_is_multicard",
    "sv_is_outlier", "meta_tieback_proration_rate"])
t = t[~t["ind_pin_is_multicard"].astype(bool) & ~t["sv_is_outlier"].astype(bool)].copy()
t["yr"] = pd.to_datetime(t["meta_sale_date"]).dt.year
g = t.groupby("meta_township_code")
ts = pd.DataFrame({
    "train_sales": g.size(), "train_pins": g["meta_pin"].nunique(),
    "sales_2020_22": t[t["yr"] >= 2020].groupby("meta_township_code").size(),
    "share_lt75k": g["meta_sale_price"].apply(lambda s: (s < 75000).mean()).round(4),
    "share_lt75k_2020_22": t[t["yr"] >= 2020].groupby("meta_township_code")["meta_sale_price"]
    .apply(lambda s: (s < 75000).mean()).round(4),
    "median_price_2022": t[t["yr"] == 2022].groupby("meta_township_code")["meta_sale_price"].median(),
    "unit_sales": t[t["meta_class"].isin(["210", "295"])].groupby("meta_township_code").size(),
    "prorated_sales": t[t["meta_tieback_proration_rate"] < 1].groupby("meta_township_code").size(),
}).fillna(0)
print(ts.to_string())
sold = set(t["meta_pin"])
pin["sold_train"] = pin.index.isin(sold)
print("roll PINs with a training sale:", pin["sold_train"].sum(), f"({pin['sold_train'].mean():.2%});",
      "training PINs not on the 2023 roll:", len(sold - set(pin.index)))

# ------------------------------------------------------------------ unit groups
h("Unit groups (complex_id_data, classes 210/295)")
c = pd.read_parquet(inp / "2023/complex_id_data.parquet").set_index("meta_pin")
u = pin[pin["unit"]].join(c[["meta_complex_id"]], how="left")
print("unit PINs:", len(u), "; without complex id:", u["meta_complex_id"].isna().sum(),
      "; multicard:", u["multicard"].sum(), "; prorated:", u["prorated"].sum(), "; no coordinates:", u["no_xy"].sum())
rows = []
for gid, d in u.groupby("meta_complex_id"):
    xy = d[["x_m", "y_m"]].dropna().to_numpy()
    rows.append(dict(
        group=gid, n=len(d), town=d["town"].iloc[0], n_town=d["town"].nunique(), n_nbhd=d["nbhd"].nunique(),
        n_class=d["cls"].nunique(), cls=d["cls"].iloc[0], n_rate=d["rate"].nunique(),
        n_prorated=int(d["prorated"].sum()), n_multicard=int(d["multicard"].sum()),
        maxdist_m=float(pdist(xy).max()) if len(xy) > 1 else 0.0,
        n_sold=int(d["sold_train"].sum()), train_sales=int(t["meta_pin"].isin(d.index).sum()),
        **{f"rng_{k[5:]}": float(d[k].max() - d[k].min()) if d[k].notna().any() else np.nan
           for k in ["char_bldg_sf", "char_yrblt", "char_rooms", "char_beds", "char_fbath", "char_hbath",
                     "char_land_sf"]},
        n_bsmt=d["char_bsmt"].nunique(), n_gar=d["char_gar1_size"].nunique()))
G = pd.DataFrame(rows).set_index("group")
G.to_csv(out / "p2_unit_groups.csv")
multi = G[G["n"] > 1]
print("groups:", len(G), "; singletons:", (G["n"] == 1).sum(), "; groups with >=2 PINs:", len(multi),
      "holding", int(multi["n"].sum()), "PINs")
print("size distribution (groups >=2):"); print(multi["n"].describe(percentiles=[.5, .75, .9, .99]).round(1).to_string())
print("PINs by group size band:")
band = pd.cut(G["n"], [0, 1, 2, 5, 10, 25, 50, 100, 1000])
print(G.groupby(band, observed=True)["n"].agg(["size", "sum"]).to_string())
print("groups spanning >1 township:", (G["n_town"] > 1).sum(), "; >1 class:", (G["n_class"] > 1).sum(),
      "; >1 proration rate:", (G["n_rate"] > 1).sum())
print("groups spanning >1 neighbourhood:", (multi["n_nbhd"] > 1).sum(), "holding", int(multi.loc[multi["n_nbhd"] > 1, "n"].sum()), "PINs")
print("max pairwise distance (m), groups >=2:"); print(multi["maxdist_m"].describe(percentiles=[.5, .9, .99]).round(0).to_string())
print("groups with max distance > 500 m:", (multi["maxdist_m"] > 500).sum(), "; > 1 km:", (multi["maxdist_m"] > 1000).sum())
for k in ["bldg_sf", "yrblt", "rooms", "beds", "fbath", "hbath", "land_sf"]:
    s = multi[f"rng_{k}"]
    print(f"within-group range {k}: median {s.median():.0f}, 90% {s.quantile(.9):.0f}, max {s.max():.0f}, share zero {(s == 0).mean():.2%}")
print("groups (>=2) with any training sale:", (multi["n_sold"] > 0).mean().round(3),
      "; unit PINs with own training sale:", u["sold_train"].mean().round(3))
print("\nwidest groups (by max distance):")
print(multi.sort_values("maxdist_m", ascending=False).head(12)[
    ["n", "town", "cls", "n_nbhd", "maxdist_m", "rng_bldg_sf", "rng_yrblt", "rng_rooms", "rng_land_sf", "n_sold"]].to_string())
print("\nlargest groups:")
print(multi.sort_values("n", ascending=False).head(8)[
    ["n", "town", "cls", "n_nbhd", "maxdist_m", "rng_bldg_sf", "rng_yrblt", "rng_rooms", "rng_land_sf", "n_sold"]].to_string())

h("Unit groups restricted to the supported population (members that are supported)")
us = u[u["supported"]]
rows = []
for gid, d in us.groupby("meta_complex_id"):
    if len(d) < 2:
        continue
    xy = d[["x_m", "y_m"]].to_numpy()
    rows.append(dict(group=gid, n=len(d), n_nbhd=d["nbhd"].nunique(), maxdist_m=float(pdist(xy).max()),
                     **{f"rng_{k[5:]}": float(d[k].max() - d[k].min()) for k in
                        ["char_bldg_sf", "char_yrblt", "char_rooms", "char_beds", "char_fbath", "char_hbath",
                         "char_land_sf"]},
                     n_beds=d["char_beds"].nunique(), n_gar=d["char_gar1_size"].nunique(),
                     n_bsmt=d["char_bsmt"].nunique()))
GS = pd.DataFrame(rows).set_index("group")
GS.to_csv(out / "p2_unit_groups_supported.csv")
print("supported unit PINs:", len(us), "; groups with >=2 supported members:", len(GS), "holding", int(GS["n"].sum()))
print("exact-key disagreements (beds / garage / basement):", int((GS["n_beds"] > 1).sum()), int((GS["n_gar"] > 1).sum()),
      int((GS["n_bsmt"] > 1).sum()), "; full-roll groups with such disagreements:",
      int(((G["n"] > 1) & ((G["rng_beds"] > 0) | (G["n_gar"] > 1) | (G["n_bsmt"] > 1))).sum()))
for k in ["bldg_sf", "yrblt", "rooms", "beds", "fbath", "hbath", "land_sf"]:
    s_ = GS[f"rng_{k}"]
    print(f"supported within-group range {k}: median {s_.median():.0f}, 90% {s_.quantile(.9):.0f}, max {s_.max():.0f}, share zero {(s_ == 0).mean():.2%}")
print("crossing neighbourhoods:", int((GS["n_nbhd"] > 1).sum()), "; max distance 99%:", round(GS["maxdist_m"].quantile(.99)),
      "; > 500 m:", int((GS["maxdist_m"] > 500).sum()), "; > 1 km:", int((GS["maxdist_m"] > 1000).sum()))

h("2023 evaluation-sale COUNTS per township (2025 input; non-outlier, one per PIN-document; no prices used)")
e = pd.read_parquet(inp / "2025/training_data.parquet", columns=["meta_pin", "meta_sale_date", "meta_sale_document_num",
                                                                  "sv_is_outlier"])
e = e[(pd.to_datetime(e["meta_sale_date"]).dt.year == 2023) & ~e["sv_is_outlier"].astype(bool)]
e = e.drop_duplicates(["meta_pin", "meta_sale_document_num"])
e1 = e.sort_values("meta_sale_date").drop_duplicates("meta_pin")  # earliest per PIN (the primary rule)
ev = pd.DataFrame({
    "sales_2023": e[e["meta_pin"].isin(pin.index)].groupby(e["meta_pin"].map(pin["town"])).size(),
    "pins_2023_supported": e1[e1["meta_pin"].isin(pin.index[pin["supported"]])].groupby(
        e1["meta_pin"].map(pin["town"])).size(),
    "pins_2023_supported_q1": e1[e1["meta_pin"].isin(pin.index[pin["supported"]]) &
                                 (pd.to_datetime(e1["meta_sale_date"]).dt.month <= 3)].groupby(
        e1["meta_pin"].map(pin["town"])).size(),
}).fillna(0).astype(int)
ev.to_csv(out / "p2_eval_sale_counts_township.csv")
print(ev.to_string())

# ------------------------------------------------------------------ nodes and graph cost
h("Graph cost per township (v2 nodes: houses + one node per unit group; whole roll vs supported only)")
pin["group"] = c["meta_complex_id"].reindex(pin.index)


def nodes_of(df):
    """Houses are nodes; units collapse to their complex group (location = mean, neighbourhood = mode)."""
    df = df[~df["no_xy"]]
    houses = df[~df["unit"]][["town", "nbhd", "x_m", "y_m"]].copy()
    units = df[df["unit"]]
    gn = units.groupby("group").agg(town=("town", "first"), nbhd=("nbhd", lambda s: s.mode().iloc[0]),
                                    x_m=("x_m", "mean"), y_m=("y_m", "mean"), n=("town", "size"))
    return pd.concat([houses.assign(n=1), gn])


def cost(nodes):
    sz = nodes.groupby("nbhd").size()
    mx = int(sz.max())
    return dict(nodes=len(nodes), nbhds=len(sz), largest_block=mx, latent=int((sz - 1).clip(lower=0).sum()),
                padded_basis=len(nodes) * (mx - 1), basis_mb=len(nodes) * (mx - 1) * 8 / 1e6,
                work=float((sz * (sz - 1)).sum()), eig_work=float((sz ** 3).sum()),
                psi_latent=(len(sz) - 1) * (NT - 1))


res = []
for town, d in pin.groupby("town"):
    r_all = cost(nodes_of(d))
    r_sup = cost(nodes_of(d[d["supported"]]))
    sold_nodes = d[d["sold_train"] & ~d["no_xy"]][["nbhd"]]  # earlier construction: parcels with a training sale
    r_v1 = cost(sold_nodes)
    res.append(dict(town=town, town_name=d["town_name"].iloc[0], triad=d["triad"].iloc[0],
                    roll_pins=len(d), **{f"all_{k}": v for k, v in r_all.items()},
                    **{f"sup_{k}": v for k, v in r_sup.items()},
                    v1style_nodes=r_v1["nodes"], v1style_largest=r_v1["largest_block"], v1style_work=r_v1["work"]))
R = pd.DataFrame(res).set_index("town")
R["work_ratio_vs_v1style"] = (R["all_work"] / R["v1style_work"]).round(1)
R.to_csv(out / "p2_graph_cost_township.csv")
print(R[["town_name", "triad", "roll_pins", "all_nodes", "all_nbhds", "all_largest_block", "all_basis_mb",
         "sup_nodes", "sup_largest_block", "sup_basis_mb", "v1style_nodes", "v1style_largest",
         "work_ratio_vs_v1style"]].round(1).to_string())

# ------------------------------------------------------------------ time dense eigendecompositions
h("Dense eigendecomposition timing (kNN-6 Laplacian of a real neighbourhood block)")


def laplacian(xy):
    n = len(xy)
    _, idx = cKDTree(xy).query(xy, k=K_NN + 1)
    i = np.repeat(np.arange(n), K_NN + 1)
    j = idx.ravel()
    keep = i != j
    W = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n)).tocsr()
    W = ((W + W.T) > 0).astype(float)
    ncomp, lab = csgraph.connected_components(W, directed=False)
    L = csgraph.laplacian(W).toarray()
    return L, ncomp


allnodes = pd.concat([nodes_of(d) for _, d in pin.groupby("town")])
bs = allnodes.groupby("nbhd").size().sort_values(ascending=False)
print("largest blocks county-wide (nodes):", bs.head(8).to_dict())
tri3 = set(pin.loc[pin["triad"].astype(str) == "3", "nbhd"])
bs3 = bs[bs.index.isin(tri3)]
print("largest blocks in triad 3 (south):", bs3.head(8).to_dict())
timings = []
for nb in [bs.index[0], bs3.index[0], bs3.index[min(5, len(bs3) - 1)], bs3.index[min(20, len(bs3) - 1)]]:
    xy = allnodes.loc[allnodes["nbhd"] == nb, ["x_m", "y_m"]].to_numpy()
    L, ncomp = laplacian(xy)
    t0 = time.time()
    lam, U = np.linalg.eigh(L)
    dt = time.time() - t0
    timings.append(dict(nbhd=nb, n=len(xy), fragments=ncomp, seconds=round(dt, 1),
                        matrix_gb=round(len(xy) ** 2 * 8 / 1e9, 2)))
    print(timings[-1], flush=True)
    del L, lam, U
pd.DataFrame(timings).to_csv(out / "p2_eigh_timing.csv", index=False)
print("numpy config:"); np.show_config()
