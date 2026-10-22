"""Sale-prediction test: prepare all 38 townships for the GL against the Assessor's 2025 held-out test sales.

Split (the Assessor's pipeline/01-train.R): the 2025 input's rows with the multicard flag and sv_is_outlier both
false (a missing flag drops the row), stable sort by sale date, the first floor(0.9 n) rows train, the rest test.
Checked: 381,763 sales, 38,177 test, first test sale 2023-10-20, test keys equal their test_card.

Placement: the census-block units of the 1 January 2023 tests (blocks_2023_units.parquet, unchanged). A PIN placed by
blocks.py's PIN map (blocks_2023.parquet) takes that unit. Any other PIN is placed by its latest 2025-input coordinates: point in a TIGER
2020 block (lowest GEOID on a shared boundary, as blocks.py); one unit in the block -> that unit; a block split
between neighbourhoods -> the unit of the nearest 2023 roll PIN in the block (blocks.py's Voronoi rule); a block
without units (dropped as water-only) -> the unit of the nearest roll PIN placed by blocks.py. Coordinates only, no prices.
A sale's fit region is its unit's `town` (blocks.py); its neighbourhood is its unit's label.

Regressors: prep_v2.py's coding rules on the 26 char_* of the 2025 params.yaml (17 -> 16 categorical with
char_class; 10 numeric), plus meta_sale_count_past_n_years as supplied, standardised; no proration rate (the
2025 model has none). Every transformation fitted on the region's training sales only; test sales use their own
sale-row characteristics (age = sale year - year built) and their own count.

Writes, for each township T, into <out>/t<T>/, in fit_struct.py's layout (roll.parquet holds the test sales, the rows
to predict): units.parquet, edges.parquet, nbhd_edges.parquet, train.parquet, roll.parquet, coding.json,
prep_manifest.json; and <out>/split_summary.json, <out>/placement.csv (every PIN placed by coordinates).

Usage (repo root): .venv/bin/python -I public/scripts/v2/county2025_prep.py [--towns 72 71 ...]
                   [--out public/data/v2/prep_county2025]
"""
import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
INPUT = "ccao/inputs/2025/training_data.parquet"
TC = "ccao/runs/2025_full/output/test_card/model_test_card.parquet"
DATA = "public/data/v2"
TIGER = "notes/v2/explore_blocks/tl_2020_17_tabblock20.zip"
CAT = ["char_air", "char_apts", "char_attic_fnsh", "char_attic_type", "char_bsmt", "char_bsmt_fin", "char_class",
       "char_ext_wall", "char_gar1_att", "char_gar1_cnst", "char_gar1_size", "char_heat", "char_porch",
       "char_roof_cnst", "char_tp_dsgn", "char_type_resd"]
NUM = ["char_yrblt", "char_beds", "char_bldg_sf", "char_fbath", "char_frpl", "char_hbath", "char_land_sf", "char_ncu",
       "char_rooms", "char_recent_renovation"]
COUNT = "meta_sale_count_past_n_years"
KEYS = ["meta_pin", "meta_sale_document_num", "meta_card_num"]
SPLIT_PROP = 0.9
RARE = 0.01
FIRST_YEAR = 2016
NT = 94                                            # 2016-01 .. 2023-10
CITY = ["70", "71", "72", "73", "74", "75", "76", "77"]


def sha(path):
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def split():
    """The Assessor's time split (01-train.R) on the 2025 input."""
    cols = KEYS + ["meta_sale_date", "meta_sale_price", "meta_township_code", "meta_nbhd_code", "meta_class",
                   "ind_pin_is_multicard", "sv_is_outlier", "loc_x_3435", "loc_y_3435",
                   "meta_tieback_proration_rate", COUNT] + CAT + NUM
    d = pd.read_parquet(REPO / INPUT, columns=list(dict.fromkeys(cols)))
    n0 = len(d)
    d = d[d["ind_pin_is_multicard"].eq(False).fillna(False).astype(bool)
          & d["sv_is_outlier"].eq(False).fillna(False).astype(bool)].copy()
    d["date"] = pd.to_datetime(d["meta_sale_date"])
    d = d.sort_values("date", kind="stable").reset_index(drop=True)
    n_train = int(len(d) * SPLIT_PROP)
    d["test"] = d.index >= n_train
    s = dict(rows=n0, filtered=len(d), train=n_train, test=len(d) - n_train,
             last_train_date=str(d.loc[n_train - 1, "date"].date()), first_test_date=str(d.loc[n_train, "date"].date()),
             last_test_date=str(d["date"].max().date()))
    assert (s["filtered"], s["test"], s["first_test_date"]) == (381763, 38177, "2023-10-20"), s
    tc = pd.read_parquet(REPO / TC, columns=KEYS)
    m = d[d["test"]][KEYS].merge(tc, on=KEYS, how="outer", indicator=True)
    assert (m["_merge"] == "both").all() and len(m) == 38177, "test keys must equal the Assessor's test_card"
    return d, s


def place(d):
    """unit_idx for every sale: blocks.py's PIN map, else coordinates (see the module docstring)."""
    import geopandas as gpd
    import shapely
    B = pd.read_parquet(REPO / DATA / "blocks_2023.parquet", columns=["pin", "on_roll", "unit_idx"]).set_index("pin")
    U = pd.read_parquet(REPO / DATA / "blocks_2023_units.parquet", columns=["unit_idx", "block_geoid", "town"])
    d["unit_idx"] = d["meta_pin"].map(B["unit_idx"])
    d.loc[d["unit_idx"] < 0, "unit_idx"] = np.nan
    todo = d[d["unit_idx"].isna()].sort_values("date").drop_duplicates("meta_pin", keep="last")
    todo = todo[["meta_pin", "loc_x_3435", "loc_y_3435", "meta_nbhd_code", "meta_township_code"]].reset_index(drop=True)
    assert todo[["loc_x_3435", "loc_y_3435"]].notna().all().all(), "a PIN to place by coordinates has none"
    blk = gpd.read_file(f"zip://{REPO / TIGER}", where="COUNTYFP20 = '031'", engine="pyogrio",
                        columns=["COUNTYFP20", "GEOID20"]).sort_values("GEOID20").reset_index(drop=True).to_crs(3435)
    pts = shapely.points(todo["loc_x_3435"].to_numpy(float), todo["loc_y_3435"].to_numpy(float))
    ip, ib = blk.sindex.query(pts, predicate="intersects")
    first = pd.Series(ib).groupby(ip).min()                       # blocks sorted by GEOID: lowest GEOID
    todo["block_geoid"] = None
    todo.loc[first.index, "block_geoid"] = blk["GEOID20"].to_numpy()[first.to_numpy()]
    # 2023 roll PINs placed by blocks.py, with their roll coordinates (the labelled points of blocks.py)
    a = pd.read_parquet(REPO / "ccao/inputs/2023/assessment_data.parquet",
                        columns=["meta_pin", "meta_card_num", "loc_x_3435", "loc_y_3435"])
    a = a.sort_values(["meta_pin", "meta_card_num"]).drop_duplicates("meta_pin").set_index("meta_pin")
    rp = B[B["on_roll"] & (B["unit_idx"] >= 0)].join(a, how="inner").dropna(subset=["loc_x_3435"])
    rp["block_geoid"] = rp["unit_idx"].map(U.set_index("unit_idx")["block_geoid"])
    units_of_block = U.groupby("block_geoid")["unit_idx"].apply(list)
    tree_all = cKDTree(rp[["loc_x_3435", "loc_y_3435"]].to_numpy(float))
    rule, uidx = [], []
    for g, x, y in zip(todo["block_geoid"], todo["loc_x_3435"], todo["loc_y_3435"]):
        us = units_of_block.get(g) if g is not None else None
        if us is not None and len(us) == 1:
            rule.append("block with one unit"); uidx.append(us[0])
        elif us is not None:
            sub = rp[rp["block_geoid"] == g]
            j = cKDTree(sub[["loc_x_3435", "loc_y_3435"]].to_numpy(float)).query([x, y])[1]
            rule.append("split block: nearest roll PIN"); uidx.append(int(sub["unit_idx"].iloc[j]))
        else:
            j = tree_all.query([x, y])[1]
            rule.append("no unit in block: nearest roll PIN" if g is not None else "outside every block: nearest roll PIN")
            uidx.append(int(rp["unit_idx"].iloc[j]))
    todo["rule"], todo["unit_idx"] = rule, uidx
    todo["unit_town"] = todo["unit_idx"].map(U.set_index("unit_idx")["town"])
    d["placed_by"] = np.where(d["unit_idx"].notna(), "P2b PIN map", "coordinates")
    d["unit_idx"] = d["unit_idx"].fillna(d["meta_pin"].map(todo.set_index("meta_pin")["unit_idx"])).astype(int)
    d["town"] = d["unit_idx"].map(U.set_index("unit_idx")["town"])
    return d, todo


def numeric_frame(d, year):
    x = pd.DataFrame(index=d.index)
    x["age"] = year - d["char_yrblt"].astype(float)
    for c in ["char_beds", "char_fbath", "char_frpl", "char_hbath", "char_ncu", "char_rooms"]:
        x[c] = d[c].astype(float)
    x["log_bldg_sf"] = np.log(d["char_bldg_sf"].astype(float).where(lambda s: s > 0))
    x["log_land_sf"] = np.log(d["char_land_sf"].astype(float).where(lambda s: s > 0))
    x[COUNT] = d[COUNT].astype(float)
    return x


def fit_coding(tr):
    """Coding table from the region's training sales only (as prep_v2.fit_coding, 2025 columns)."""
    x = numeric_frame(tr, tr["date"].dt.year)
    med = x.median()
    xf = x.fillna(med)
    num = {c: dict(median=float(med[c]), mean=float(xf[c].mean()), sd=float(xf[c].std()),
                   n_missing_train=int(x[c].isna().sum())) for c in x.columns}
    reno = tr["char_recent_renovation"].astype("boolean")
    cat = {}
    for c in CAT:
        s = tr[c].astype("string").fillna("missing")
        share = s.value_counts(normalize=True)
        rare = sorted(l for l in share[share < RARE].index if l != "missing")
        s = s.where(~s.isin(rare), "other")
        counts = s.value_counts()
        cat[c] = dict(levels=sorted(counts.index.tolist()), reference=counts.index[0], merged_into_other=rare,
                      train_counts={k: int(v) for k, v in counts.items()})
    return dict(numeric=num, renovation=dict(missing_as=False, n_missing_train=int(reno.isna().sum())),
                categorical=cat, rare_threshold=RARE)


def apply_coding(d, coding, counts):
    x = numeric_frame(d, d["date"].dt.year)
    X = pd.DataFrame(index=d.index)
    for c, p in coding["numeric"].items():
        counts[f"filled_{c}"] = int(x[c].isna().sum())
        X[c] = (x[c].fillna(p["median"]) - p["mean"]) / p["sd"] if p["sd"] > 0 else 0.0
    reno = d["char_recent_renovation"].astype("boolean")
    counts["filled_char_recent_renovation"] = int(reno.isna().sum())
    X["char_recent_renovation"] = reno.fillna(False).astype(float)
    for c, p in coding["categorical"].items():
        s = d[c].astype("string").fillna("missing")
        s = s.where(~s.isin(p["merged_into_other"]), "other")
        unseen = ~s.isin(p["levels"])
        counts[f"unseen_{c}"] = int(unseen.sum())
        s = s.where(~unseen, "other" if "other" in p["levels"] else p["reference"])
        for lev in p["levels"]:
            if lev != p["reference"]:
                X[f"{c}={lev}"] = (s == lev).astype(float)
    return X


def prep_town(T, d, out, inputs):
    o = out / f"t{T}"
    o.mkdir(parents=True, exist_ok=True)
    counts = {}
    # ---------------------------------------------------------------- units and graph (as prep_v2.py)
    U = pd.read_parquet(REPO / DATA / "blocks_2023_units.parquet")
    units = U[U["town"] == T].sort_values("unit_idx").reset_index(drop=True)
    units["nbhd_id"] = units["nbhd_assigned"].astype(str)
    units["node"] = np.arange(len(units))
    node_of = dict(zip(units["unit_idx"], units["node"]))
    E = pd.read_parquet(REPO / DATA / "blocks_2023_edges.parquet")
    E = E[E["unit_a"].isin(node_of) & E["unit_b"].isin(node_of)].copy()
    E["a"], E["b"] = E["unit_a"].map(node_of), E["unit_b"].map(node_of)
    E[["a", "b"]] = np.sort(E[["a", "b"]].to_numpy(), axis=1)
    E = E.drop_duplicates(["a", "b"]).sort_values(["a", "b"]).reset_index(drop=True)
    n = len(units)
    ncomp, comp = connected_components(coo_matrix((np.ones(len(E)), (E["a"], E["b"])), shape=(n, n)), directed=False)
    units["comp"] = comp
    nb_of_node = units["nbhd_id"].to_numpy()
    ne = pd.DataFrame({"x": nb_of_node[E["a"]], "y": nb_of_node[E["b"]]})
    ne = ne[ne["x"] != ne["y"]]
    ne = pd.DataFrame(np.sort(ne.to_numpy(), axis=1), columns=["nbhd_a", "nbhd_b"]).drop_duplicates()
    G = nx.Graph()
    G.add_nodes_from(sorted(units["nbhd_id"].unique()))
    G.add_edges_from(ne.itertuples(index=False))
    counts.update(units=n, edges=len(E), components=int(ncomp), nbhds=int(units["nbhd_id"].nunique()),
                  nbhd_edges=len(ne), nbhd_components=nx.number_connected_components(G))

    # ---------------------------------------------------------------- sales of the region
    s = d[d["town"] == T].copy()
    s["node"] = s["unit_idx"].map(node_of).astype(int)
    s["nbhd_id"] = nb_of_node[s["node"]]
    s["month"] = (s["date"].dt.year - FIRST_YEAR) * 12 + s["date"].dt.month
    s["log_price"] = np.log(s["meta_sale_price"].astype(float))
    t, e = s[~s["test"]].copy(), s[s["test"]].copy()
    assert t["month"].between(1, NT).all() and (e["month"] >= NT).all()
    t = t.sort_values(["date", "meta_pin", "meta_sale_document_num"], kind="stable").reset_index(drop=True)
    e = e.sort_values(["date", "meta_pin", "meta_sale_document_num"], kind="stable").reset_index(drop=True)
    counts.update(train_sales=len(t), train_pins=int(t["meta_pin"].nunique()), test_sales=len(e),
                  test_pins=int(e["meta_pin"].nunique()),
                  train_other_town_code=int((t["meta_township_code"] != T).sum()),
                  test_other_town_code=int((e["meta_township_code"] != T).sum()),
                  test_nbhd_differs_from_unit=int((e["meta_nbhd_code"] != e["nbhd_id"]).sum()),
                  train_placed_by_coordinates=int((t["placed_by"] == "coordinates").sum()),
                  test_placed_by_coordinates=int((e["placed_by"] == "coordinates").sum()),
                  test_pin_has_training_sale=int(e["meta_pin"].isin(t["meta_pin"]).sum()),
                  test_nbhd_without_training_sale=int((~e["nbhd_id"].isin(t["nbhd_id"])).sum()),
                  test_unit_without_training_sale=int((~e["node"].isin(t["node"])).sum()),
                  units_with_training_sale=int(t["node"].nunique()))
    # ---------------------------------------------------------------- regressors (prep_v2.py's coding on the 2025 columns + count)
    coding = fit_coding(t)
    ct, ce = {}, {}
    Xt, Xe = apply_coding(t, coding, ct), apply_coding(e, coding, ce)
    keep = [c for c in Xt.columns if Xt[c].std() > 0]
    dropped = [c for c in Xt.columns if c not in keep]
    collinear = []
    while True:                                   # drop collinear columns, as prep_v2.py
        Z = Xt[keep].to_numpy() - Xt[keep].to_numpy().mean(axis=0)
        rel = np.abs(np.diag(np.linalg.qr(Z, mode="r"))) / np.linalg.norm(Z, axis=0)
        bad = np.flatnonzero(rel < 1e-7)
        if not len(bad):
            break
        collinear.append(keep.pop(int(bad[0])))
    coding.update(columns=keep, dropped_no_variation=dropped, dropped_collinear=collinear,
                  fill_and_unseen_counts=dict(train={k: v for k, v in ct.items() if v},
                                              test={k: v for k, v in ce.items() if v}))
    (o / "coding.json").write_text(json.dumps(coding, indent=1))
    base = ["meta_pin", "meta_sale_document_num", "meta_card_num", "date", "month", "meta_sale_price", "log_price",
            "meta_class", "meta_township_code", "meta_nbhd_code", "meta_tieback_proration_rate", COUNT, "placed_by",
            "unit_idx", "node", "nbhd_id"]
    t = pd.concat([t[base], Xt[keep].add_prefix("x:")], axis=1)
    e = pd.concat([e[base], Xe[keep].add_prefix("x:")], axis=1)
    e["has_training_sale"] = e["meta_pin"].isin(t["meta_pin"])
    units[["node", "unit_idx", "unit_id", "nbhd_id", "comp", "pass_through", "kind", "n_supported",
           "n_train_sales"]].to_parquet(o / "units.parquet", index=False)
    E[["a", "b", "length_m", "same_nbhd"]].to_parquet(o / "edges.parquet", index=False)
    ne.to_parquet(o / "nbhd_edges.parquet", index=False)
    t.to_parquet(o / "train.parquet", index=False)
    e.to_parquet(o / "roll.parquet", index=False)
    outs = sorted(p for p in o.iterdir() if p.name != "prep_manifest.json")
    man = dict(town=T, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs=inputs,
               outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outs},
               coding_sha256=hashlib.sha256((o / "coding.json").read_bytes()).hexdigest(), n_months=NT, counts=counts)
    (o / "prep_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"t{T}: units {n:,} ({ncomp} components), nbhds {counts['nbhds']}; train {len(t):,}, test {len(e):,} "
          f"(other town code {counts['test_other_town_code']}, placed by coordinates "
          f"{counts['test_placed_by_coordinates']}, unit without training sale "
          f"{counts['test_unit_without_training_sale']}); regressors {len(keep)} (dropped {dropped + collinear})",
          flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", nargs="*")
    ap.add_argument("--out", default="public/data/v2/prep_county2025")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    d, s = split()
    print(f"split: {s}", flush=True)
    d, todo = place(d)
    todo.to_csv(out / "placement.csv", index=False)
    s["placed_by_coordinates"] = dict(pins=len(todo), rules=todo["rule"].value_counts().to_dict(),
                                      sales=int((d["placed_by"] == "coordinates").sum()),
                                      test_sales=int(((d["placed_by"] == "coordinates") & d["test"]).sum()),
                                      unit_town_differs_from_township_code=int(
                                          (todo["unit_town"] != todo["meta_township_code"]).sum()))
    s["no_region"] = int(d["town"].isin(["", None]).sum() + d["town"].isna().sum())
    assert s["no_region"] == 0, "a sale falls in a unit without a township"
    s["by_region"] = {T: dict(train=int((~g["test"]).sum()), test=int(g["test"].sum()))
                      for T, g in d.groupby("town")}
    s["region_differs_from_township_code"] = dict(train=int(((d["town"] != d["meta_township_code"]) & ~d["test"]).sum()),
                                                  test=int(((d["town"] != d["meta_township_code"]) & d["test"]).sum()))
    print(f"placement: {s['placed_by_coordinates']}; region differs from the sale's township code: "
          f"{s['region_differs_from_township_code']}", flush=True)
    inputs = {p: sha(p) for p in [INPUT, TC, f"{DATA}/blocks_2023.parquet", f"{DATA}/blocks_2023_units.parquet",
                                  f"{DATA}/blocks_2023_edges.parquet", "ccao/inputs/2023/assessment_data.parquet", TIGER]}
    s["inputs"] = inputs
    towns = a.towns or sorted(s["by_region"])
    s["towns"] = {T: prep_town(T, d, out, inputs) for T in towns}
    if not a.towns:
        # key check: every sale in exactly one region, keys unique, totals as the split
        for fname, n in [("train.parquet", 343586), ("roll.parquet", 38177)]:
            k = pd.concat([pd.read_parquet(out / f"t{T}" / fname, columns=KEYS) for T in towns], ignore_index=True)
            assert len(k) == n and not k.duplicated().any(), f"{fname}: {len(k)} rows, keys must be unique and number {n}"
        nc = d[(d["placed_by"] == "coordinates") & d["test"]]
        U = pd.read_parquet(REPO / DATA / "blocks_2023_units.parquet", columns=["unit_idx", "nbhd_assigned"])
        s["placed_by_coordinates"]["test_nbhd_differs_from_unit"] = int(
            (nc["meta_nbhd_code"] != nc["unit_idx"].map(U.set_index("unit_idx")["nbhd_assigned"])).sum())
        print(f"key contract: 343,586 training and 38,177 test rows, each once; test rows placed by coordinates whose "
              f"neighbourhood code differs from their unit's: {s['placed_by_coordinates']['test_nbhd_differs_from_unit']}")
    (out / "split_summary.json").write_text(json.dumps(s, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
