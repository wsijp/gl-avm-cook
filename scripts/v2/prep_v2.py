"""Prepare one fit region (township) for the graph-Laplacian models.

Writes, for township T, into <out>/t<T>/:
  units.parquet        the region's block units (units whose blocks.py label `town` is T; the 18 unlabelled
                       island units have no town and so never enter), with `comp` (connected component of
                       the region's rook graph; no bridges, no units from other townships)
  edges.parquet        rook edges with both ends in the region (binary in use)
  nbhd_edges.parquet   neighbourhood adjacency: two neighbourhoods are adjacent when a rook edge joins units
                       of the two (implementation of the paper's neighbourhood graph on block geography)
  train.parquet        training sales: the Assessor's 2023 training set under their filter (multicard and
                       sv_is_outlier both false; a missing flag drops the row), placed by the PIN-level
                       mapping (blocks_2023.parquet), in the region by their unit's town; unplaceable sales
                       are dropped (county-wide 145, p3_historical_unplaced.csv) and counted
  roll.parquet         valued roll PINs (`supported` in blocks_2023.parquet) in the region, card-1
                       characteristics of the 2023 assessment_data, proration rate set to 1, valuation
                       group (the Assessor's complex group; singletons for PINs outside a group)
  eval.parquet         primary evaluation sales (`eligible_primary` in p3_eval_sales.parquet) of the
                       region with the cross-fitted time adjustment of p3_time_index.py (p3_eval_time_adjustment.parquet,
                       adjusted price = price x exp(log_adjustment)), the strict-validation and Q1 flags, and
                       whether the PIN has a training sale (prior-sale history)
  houses.parquet,      house graph for the engineering comparator only: one node per valued PIN or
  house_edges.parquet  training PIN of the region, kNN-6 within the neighbourhood, fragments joined by MST
                       bridges between nearest houses (as in an earlier graph-Laplacian construction)
  coding.json          the regressor coding table (fitted on the region's training sales only); its sha256
                       is in prep_manifest.json
  prep_manifest.json   input and output sha256, counts, unseen-level and fill counts

Regressor coding: the 27 2023 char_* plus meta_tieback_proration_rate. log building and land
area; age = sale year - year built (training) or 2023 - year built (roll); numeric columns standardised with the
training mean and sd; char_recent_renovation as 0/1; categoricals as indicators, the most common training level
the reference, levels under 1% of training sales merged into "other", missing its own level (never merged). Unseen roll levels go
to "other" if it exists, else to the reference; missing numerics get the training median (raw scale). Columns
that do not vary in the training sales, or are exact linear combinations of earlier columns there, are dropped
and listed (neither is identified).

Usage (repo root): .venv/bin/python -I public/scripts/v2/prep_v2.py --town 28 [--inputs ccao/inputs]
                   [--data public/data/v2] [--out public/data/v2/prep]
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

CAT = ["char_air", "char_apts", "char_attic_fnsh", "char_attic_type", "char_bsmt", "char_bsmt_fin", "char_ext_wall",
       "char_gar1_area", "char_gar1_att", "char_gar1_cnst", "char_gar1_size", "char_heat", "char_porch",
       "char_roof_cnst", "char_tp_dsgn", "char_tp_plan", "char_type_resd"]
NUM = ["char_yrblt", "char_beds", "char_bldg_sf", "char_fbath", "char_frpl", "char_hbath", "char_land_sf", "char_ncu",
       "char_rooms", "char_recent_renovation"]
RATE = "meta_tieback_proration_rate"
RARE = 0.01
K_NN = 6
FIRST_YEAR, LAST_YEAR = 2014, 2022   # training months 2014-01 .. 2022-12
ROLL_YEAR = 2023                      # age at 1 January 2023


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_frame(d, year):
    """Raw numeric regressors before filling and scaling: age replaces year built; logs of the two areas."""
    x = pd.DataFrame(index=d.index)
    x["age"] = year - d["char_yrblt"].astype(float)
    for c in ["char_beds", "char_fbath", "char_frpl", "char_hbath", "char_ncu", "char_rooms"]:
        x[c] = d[c].astype(float)
    x["log_bldg_sf"] = np.log(d["char_bldg_sf"].astype(float).where(lambda s: s > 0))
    x["log_land_sf"] = np.log(d["char_land_sf"].astype(float).where(lambda s: s > 0))
    x[RATE] = d[RATE].astype(float)
    return x


def fit_coding(tr):
    """Coding table from the region's training sales only."""
    x = numeric_frame(tr, tr["sale_year"])
    med = x.median()
    xf = x.fillna(med)
    num = {c: dict(median=float(med[c]), mean=float(xf[c].mean()), sd=float(xf[c].std()),
                   n_missing_train=int(x[c].isna().sum())) for c in x.columns}
    reno = tr["char_recent_renovation"].astype("boolean")
    cat = {}
    for c in CAT:
        s = tr[c].astype("string").fillna("missing")
        share = s.value_counts(normalize=True)
        # "missing" keeps its own level however rare; only observed levels are merged.
        rare = sorted(l for l in share[share < RARE].index if l != "missing")
        s = s.where(~s.isin(rare), "other")
        counts = s.value_counts()
        ref = counts.index[0]
        cat[c] = dict(levels=sorted(counts.index.tolist()), reference=ref, merged_into_other=rare,
                      train_counts={k: int(v) for k, v in counts.items()})
    return dict(numeric=num, renovation=dict(missing_as=False, n_missing_train=int(reno.isna().sum())),
                categorical=cat, rare_threshold=RARE)


def apply_coding(d, year, coding, counts):
    """Design matrix for sales (year = sale year) or roll PINs (year = 2023) under a fitted coding table."""
    x = numeric_frame(d, year)
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


def knn_edges(xy, k):
    k = min(k, len(xy) - 1)
    if k < 1:
        return set()
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    return {(min(i, j), max(i, j)) for i, row in enumerate(idx) for j in row if j != i}


def house_graph(h):
    """kNN-6 within each neighbourhood; fragments joined by MST bridges between nearest houses."""
    edges, bridges = [], []
    for nb, g in h.groupby("nbhd", sort=True):
        ids = g["house_idx"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        G = nx.Graph()
        G.add_nodes_from(range(len(ids)))
        G.add_edges_from(knn_edges(xy, K_NN))
        frags = [sorted(c) for c in nx.connected_components(G)]
        if len(frags) > 1:
            trees = [cKDTree(xy[f]) for f in frags]
            F = nx.Graph()
            for a in range(len(frags)):
                for b in range(a + 1, len(frags)):
                    dist, j = trees[b].query(xy[frags[a]])
                    i = int(np.argmin(dist))
                    F.add_edge(a, b, weight=float(dist[i]), pair=(frags[a][i], frags[b][int(j[i])]))
            for _, _, e in nx.minimum_spanning_tree(F).edges(data=True):
                G.add_edge(*e["pair"])
                bridges.append(dict(nbhd=nb, length_ft=round(e["weight"])))
        edges += [(ids[i], ids[j]) for i, j in G.edges()]
    e = pd.DataFrame(edges, columns=["a", "b"])
    e[["a", "b"]] = np.sort(e[["a", "b"]].to_numpy(), axis=1)
    return e.drop_duplicates().sort_values(["a", "b"]).reset_index(drop=True), pd.DataFrame(bridges)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", required=True)
    ap.add_argument("--inputs", default="ccao/inputs")
    ap.add_argument("--data", default="public/data/v2")
    ap.add_argument("--out", default="public/data/v2/prep")
    a = ap.parse_args()
    T, inp, data = a.town, Path(a.inputs), Path(a.data)
    out = Path(a.out) / f"t{T}"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    print(f"=== prep_v2 township {T}", flush=True)

    # ---------------------------------------------------------------- units and graph
    U = pd.read_parquet(data / "blocks_2023_units.parquet")
    units = U[U["town"] == T].sort_values("unit_idx").reset_index(drop=True)
    units["nbhd_id"] = units["nbhd_assigned"].astype(str)
    units["node"] = np.arange(len(units))
    node_of = dict(zip(units["unit_idx"], units["node"]))
    E = pd.read_parquet(data / "blocks_2023_edges.parquet")
    E = E[E["unit_a"].isin(node_of) & E["unit_b"].isin(node_of)].copy()
    E["a"], E["b"] = E["unit_a"].map(node_of), E["unit_b"].map(node_of)
    E[["a", "b"]] = np.sort(E[["a", "b"]].to_numpy(), axis=1)
    E = E.drop_duplicates(["a", "b"]).sort_values(["a", "b"]).reset_index(drop=True)
    n = len(units)
    A = coo_matrix((np.ones(len(E)), (E["a"], E["b"])), shape=(n, n))
    ncomp, comp = connected_components(A + A.T, directed=False)
    units["comp"] = comp
    counts.update(units=n, edges=len(E), components=int(ncomp),
                  component_sizes=sorted(np.bincount(comp).tolist(), reverse=True),
                  pass_through_units=int(units["pass_through"].sum()))
    nb_of_node = units["nbhd_id"].to_numpy()
    ne = pd.DataFrame({"x": nb_of_node[E["a"]], "y": nb_of_node[E["b"]]})
    ne = ne[ne["x"] != ne["y"]]
    ne = pd.DataFrame(np.sort(ne.to_numpy(), axis=1), columns=["nbhd_a", "nbhd_b"]).drop_duplicates()
    nbhds = sorted(units["nbhd_id"].unique())
    G = nx.Graph()
    G.add_nodes_from(nbhds)
    G.add_edges_from(ne.itertuples(index=False))
    counts.update(nbhds=len(nbhds), nbhd_edges=len(ne), nbhd_components=nx.number_connected_components(G))
    print(f"units {n:,} ({counts['pass_through_units']:,} pass-through), rook edges {len(E):,}, components "
          f"{ncomp} {counts['component_sizes'][:5]}; neighbourhoods {len(nbhds)}, adjacencies {len(ne)}, "
          f"neighbourhood-graph components {counts['nbhd_components']}")

    # ---------------------------------------------------------------- PIN -> unit (PIN-level mapping of blocks.py)
    B = pd.read_parquet(data / "blocks_2023.parquet", columns=["pin", "on_roll", "supported", "unit_idx"])
    B = B.set_index("pin")

    # ---------------------------------------------------------------- training sales
    tcols = ["meta_pin", "meta_card_num", "meta_sale_date", "meta_sale_document_num", "meta_sale_price", "meta_class",
             "meta_township_code", "ind_pin_is_multicard", "sv_is_outlier", "loc_x_3435", "loc_y_3435", RATE] + CAT + NUM
    t = pd.read_parquet(inp / "2023/training_data.parquet", columns=tcols)
    t = t[t["ind_pin_is_multicard"].eq(False).fillna(False).astype(bool)
          & t["sv_is_outlier"].eq(False).fillna(False).astype(bool)].copy()
    counts["county_training_filtered"] = len(t)
    t["unit_idx"] = t["meta_pin"].map(B["unit_idx"])
    t.loc[t["unit_idx"] < 0, "unit_idx"] = np.nan   # blocks.py codes an unplaced PIN as unit_idx -1
    counts["county_training_unplaced"] = int(t["unit_idx"].isna().sum())
    assert counts["county_training_unplaced"] == 145, "MANIFEST §6: 145 unplaceable training sales"
    t = t[t["unit_idx"].isin(node_of)].copy()
    t["node"] = t["unit_idx"].map(node_of).astype(int)
    t["date"] = pd.to_datetime(t["meta_sale_date"])
    t["sale_year"] = t["date"].dt.year
    t["month"] = (t["date"].dt.year - FIRST_YEAR) * 12 + t["date"].dt.month   # 1 .. 108
    assert t["month"].between(1, (LAST_YEAR - FIRST_YEAR + 1) * 12).all()
    t["log_price"] = np.log(t["meta_sale_price"].astype(float))
    t["nbhd_id"] = units["nbhd_id"].to_numpy()[t["node"]]
    t = t.sort_values(["date", "meta_pin", "meta_sale_document_num"], kind="stable").reset_index(drop=True)
    counts.update(train_sales=len(t), train_pins=int(t["meta_pin"].nunique()),
                  train_sales_other_town_code=int((t["meta_township_code"] != T).sum()),
                  train_prorated=int((t[RATE] < 1).sum()))
    print(f"training sales: county {counts['county_training_filtered']:,} after the filter, "
          f"{counts['county_training_unplaced']} unplaceable; region {len(t):,} on {counts['train_pins']:,} PINs "
          f"({counts['train_sales_other_town_code']} with another sale-year township code; "
          f"{counts['train_prorated']} prorated at sale)")

    # ---------------------------------------------------------------- valued roll PINs
    rcols = ["meta_pin", "meta_card_num", "meta_class", "meta_township_code", "meta_nbhd_code", "loc_x_3435",
             "loc_y_3435", RATE] + CAT + NUM
    r = pd.read_parquet(inp / "2023/assessment_data.parquet", columns=rcols)
    r = r.sort_values(["meta_pin", "meta_card_num"]).drop_duplicates("meta_pin")
    r = r[r["meta_pin"].map(B["supported"]).eq(True)].copy()
    r["unit_idx"] = r["meta_pin"].map(B["unit_idx"])
    r = r[r["unit_idx"].isin(node_of)].copy()
    r["node"] = r["unit_idx"].map(node_of).astype(int)
    r["nbhd_id"] = units["nbhd_id"].to_numpy()[r["node"]]
    r[RATE] = 1.0  # valued PINs are not prorated; predicted at rate 1
    r = r.sort_values("meta_pin").reset_index(drop=True)
    counts.update(valued_pins=len(r), valued_other_town_code=int((r["meta_township_code"] != T).sum()),
                  valued_nbhd_differs_from_unit=int((r["meta_nbhd_code"] != r["nbhd_id"]).sum()))
    cx = pd.read_parquet(inp / "2023/complex_id_data.parquet", columns=["meta_pin", "meta_complex_id"])
    cx = cx.drop_duplicates("meta_pin").set_index("meta_pin")["meta_complex_id"].astype(str)
    r["complex_id"] = r["meta_pin"].map(cx)
    r["group_id"] = np.where(r["complex_id"].notna(), "c" + r["complex_id"].astype(str), "p" + r["meta_pin"])
    gsize = r.groupby("group_id")["meta_pin"].transform("size")
    r["group_valued_size"] = gsize
    allm = cx.reset_index()
    allm = allm[allm["meta_complex_id"].isin(r["complex_id"].dropna())]
    outside = allm[~allm["meta_pin"].isin(r["meta_pin"])]
    outside.to_csv(out / "group_members_outside.csv", index=False)
    counts.update(groups_ge2_valued=int(r.loc[gsize >= 2, "group_id"].nunique()),
                  pins_in_groups_ge2=int((gsize >= 2).sum()),
                  group_members_outside_valued=len(outside))
    print(f"valued PINs {len(r):,} ({counts['valued_other_town_code']} with another roll township code; "
          f"{counts['valued_nbhd_differs_from_unit']} whose roll neighbourhood differs from the unit label); "
          f"{counts['groups_ge2_valued']} groups with >= 2 valued members ({counts['pins_in_groups_ge2']:,} PINs); "
          f"{len(outside)} group members outside the valued population (listed)")

    # ---------------------------------------------------------------- regressors
    coding = fit_coding(t)
    ct, cr = {}, {}
    Xt = apply_coding(t, t["sale_year"], coding, ct)
    Xr = apply_coding(r, ROLL_YEAR, coding, cr)
    keep = [c for c in Xt.columns if Xt[c].std() > 0]
    dropped = [c for c in Xt.columns if c not in keep]
    # Exactly collinear columns are not identified either (and make the QR's R singular): drop, in column order,
    # every column whose residual after projection on the earlier kept columns (training sales, centred) is
    # below 1e-7 of its own norm.
    collinear = []
    while True:
        Z = Xt[keep].to_numpy() - Xt[keep].to_numpy().mean(axis=0)
        rel = np.abs(np.diag(np.linalg.qr(Z, mode="r"))) / np.linalg.norm(Z, axis=0)
        bad = np.flatnonzero(rel < 1e-7)
        if not len(bad):
            break
        collinear.append(keep.pop(int(bad[0])))
    coding["columns"] = keep
    coding["dropped_no_variation"] = dropped
    coding["dropped_collinear"] = collinear
    coding["fill_and_unseen_counts"] = dict(train={k: v for k, v in ct.items() if v},
                                            roll={k: v for k, v in cr.items() if v})
    (out / "coding.json").write_text(json.dumps(coding, indent=1))
    print(f"regressors: {len(keep)} columns ({len(dropped)} dropped for no variation: {dropped}; "
          f"{len(collinear)} dropped as exactly collinear: {collinear}); "
          f"roll fills/unseen: {coding['fill_and_unseen_counts']['roll']}")
    t = pd.concat([t.drop(columns=CAT + NUM), Xt[keep].add_prefix("x:")], axis=1)
    r = pd.concat([r.drop(columns=CAT + NUM), Xr[keep].add_prefix("x:")], axis=1)

    # ---------------------------------------------------------------- evaluation sales
    e = pd.read_parquet(data / "p3_eval_sales.parquet")
    e = e[e["eligible_primary"] & (e["town"] == T)].copy()
    ta = pd.read_parquet(data / "p3_eval_time_adjustment.parquet")
    e = e.merge(ta[["meta_pin", "meta_sale_document_num", "month", "fold", "log_adjustment"]],
                on=["meta_pin", "meta_sale_document_num"], how="left", validate="one_to_one")
    assert e["log_adjustment"].notna().all()
    e["price"] = e["meta_sale_price"].astype(float)
    e["adj_price"] = e["price"] * np.exp(e["log_adjustment"])
    e["has_training_sale"] = e["meta_pin"].isin(t["meta_pin"])
    missing = set(e["meta_pin"]) - set(r["meta_pin"])
    assert not missing, f"{len(missing)} evaluation PINs are not valued PINs of the region"
    e = e.merge(r[["meta_pin", "group_id", "group_valued_size", "meta_class"]].rename(columns={"meta_class": "roll_class"}),
                on="meta_pin", how="left")
    e = e[["meta_pin", "meta_sale_document_num", "date", "month", "fold", "price", "log_adjustment", "adj_price",
           "roll_class", "group_id", "group_valued_size", "has_training_sale", "eligible_primary_strict", "q1"]]
    e = e.sort_values("meta_pin").reset_index(drop=True)
    counts.update(eval_sales=len(e), eval_with_training_sale=int(e["has_training_sale"].sum()),
                  eval_strict=int(e["eligible_primary_strict"].sum()), eval_q1=int(e["q1"].sum()))
    print(f"evaluation sales (primary) {len(e)}; with a training sale of the PIN {counts['eval_with_training_sale']}; "
          f"strict {counts['eval_strict']}; Q1 {counts['eval_q1']}")

    # ---------------------------------------------------------------- house graph (engineering comparator)
    hr = r[["meta_pin", "loc_x_3435", "loc_y_3435", "nbhd_id"]].assign(valued=True)
    lt = t.sort_values("date").drop_duplicates("meta_pin", keep="last")
    lt = lt[~lt["meta_pin"].isin(hr["meta_pin"])][["meta_pin", "loc_x_3435", "loc_y_3435", "nbhd_id"]]
    # Training PINs that are on the roll but not valued take their roll coordinates where they exist.
    rall = pd.read_parquet(inp / "2023/assessment_data.parquet", columns=["meta_pin", "meta_card_num", "loc_x_3435", "loc_y_3435"])
    rall = rall.sort_values(["meta_pin", "meta_card_num"]).drop_duplicates("meta_pin").set_index("meta_pin")
    for c in ["loc_x_3435", "loc_y_3435"]:
        lt[c] = lt["meta_pin"].map(rall[c]).fillna(lt[c])
    houses = pd.concat([hr, lt.assign(valued=False)], ignore_index=True)
    assert houses[["loc_x_3435", "loc_y_3435"]].notna().all().all()
    houses = houses.rename(columns={"loc_x_3435": "x", "loc_y_3435": "y"})
    houses = houses.sort_values(["nbhd_id", "meta_pin"]).reset_index(drop=True)
    houses["house_idx"] = np.arange(len(houses))
    he, hb = house_graph(houses.rename(columns={"nbhd_id": "nbhd"}))
    counts.update(houses=len(houses), houses_not_valued=int((~houses["valued"]).sum()), house_edges=len(he),
                  house_bridges=len(hb), house_nbhds=int(houses["nbhd_id"].nunique()))
    print(f"house graph: {len(houses):,} houses ({counts['houses_not_valued']:,} training-only), {len(he):,} edges, "
          f"{len(hb)} fragment bridges in {houses['nbhd_id'].nunique()} neighbourhoods")

    # ---------------------------------------------------------------- write
    units[["node", "unit_idx", "unit_id", "nbhd_id", "comp", "pass_through", "kind", "n_supported", "n_train_sales"]].to_parquet(out / "units.parquet", index=False)
    E[["a", "b", "length_m", "same_nbhd"]].to_parquet(out / "edges.parquet", index=False)
    ne.to_parquet(out / "nbhd_edges.parquet", index=False)
    t.drop(columns=["meta_sale_date", "ind_pin_is_multicard", "sv_is_outlier"]).to_parquet(out / "train.parquet", index=False)
    r.to_parquet(out / "roll.parquet", index=False)
    e.to_parquet(out / "eval.parquet", index=False)
    houses.to_parquet(out / "houses.parquet", index=False)
    he.to_parquet(out / "house_edges.parquet", index=False)
    hb.to_csv(out / "house_bridges.csv", index=False)
    ins = [inp / "2023/training_data.parquet", inp / "2023/assessment_data.parquet", inp / "2023/complex_id_data.parquet",
           data / "blocks_2023.parquet", data / "blocks_2023_units.parquet", data / "blocks_2023_edges.parquet",
           data / "p3_eval_sales.parquet", data / "p3_eval_time_adjustment.parquet"]
    outs = sorted(p for p in out.iterdir() if p.name != "prep_manifest.json")
    man = dict(town=T, script_sha256=sha(__file__), inputs={str(p): sha(p) for p in ins},
               outputs={p.name: sha(p) for p in outs}, coding_sha256=sha(out / "coding.json"), counts=counts)
    (out / "prep_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"coding sha256 {man['coding_sha256']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
