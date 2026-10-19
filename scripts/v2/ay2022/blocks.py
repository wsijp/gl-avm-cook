# 2022 version of scripts/v2/blocks.py: the same code with the 2022 assessment year's dates, file names and
# north-suburb labels, and the differences listed in REPRODUCE.md ("What to expect").
"""Census-block spatial units and their rook graph for the 2022 roll.

Standalone. Inputs: the 2020 TIGER/Line tabulation blocks for Illinois (tl_2020_17_tabblock20.zip;
Cook = COUNTYFP20 031) and the Assessor's 2022 inputs (assessment_data = the roll, training_data =
sales 2013-2021). No sale price is read.

1. Point-in-polygon: every roll PIN (card-1 coordinates, EPSG:3435) and every training PIN (coordinates
   of its latest training row) to a 2020 block. Points on a shared boundary go to the lowest GEOID.
   Unmatched points are reported, never assigned.
2. Units: a block whose roll PINs fall in more than one Assessor neighbourhood is split into
   block x neighbourhood units; other blocks are one unit. Splits are defined by roll PINs only. A unit's
   portion of the block is the set of points nearer to its roll PINs than to the block's other roll PINs
   (Voronoi within the block). Training PINs on the roll take their roll unit; off-roll training PINs take
   the unit of the block (or Voronoi portion) they fall in.
3. Graph: rook contiguity (shared boundary of positive length) between units. Block-to-block edges come
   from the TIGER geometry; where a block is split, the shared boundary is cut into <= 2 m pieces and each
   piece is allocated to the Voronoi portion it lies in; split portions of one block are adjacent where
   their Voronoi portions share a boundary. Blocks with no eligible parcel stay as pass-through units;
   water-only blocks (ALAND20 = 0) holding no PIN are dropped.
4. Units without roll PINs have no neighbourhood. For per-township and per-neighbourhood counts they are
   given one by a provisional inventory rule: repeatedly, the neighbourhood with the longest shared
   boundary among already-labelled neighbours (ties: lowest code). This is not a modelling decision.

"Supported" (eligible) follows p2_population.py: single-card, not prorated, coordinates, model class.
Training sales are counted after the Assessor's 2022 training filter (not multicard).

Usage (repo root):
  .venv/bin/python -I public/scripts/v2/ay2022/blocks.py <path to tl_2020_17_tabblock20.zip> \
      ccao/inputs public/data/v2/ay2022
Writes blocks_2022.parquet (PIN -> unit), blocks_2022_units.parquet, blocks_2022_edges.parquet,
blocks_2022_township.csv, blocks_2022_timing.csv and blocks_2022_manifest.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix, csgraph, identity
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

tiger_zip, inp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
FT = 0.3048
SEG_FT = 2 / FT  # boundary pieces of at most 2 m for allocation to split portions
MIN_INTERNAL_M = 0.01  # internal Voronoi boundaries shorter than 1 cm are float artefacts
MODEL_CLASSES = ["202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "234", "278", "295"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
T0 = time.time()


def h(t):
    print(f"\n=== {t}  [{time.time() - T0:.0f} s]", flush=True)


def sha256(p):
    m = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


# ------------------------------------------------------------------ blocks
h("TIGER 2020 tabulation blocks, Cook County")
blk = gpd.read_file(f"zip://{tiger_zip}", where="COUNTYFP20 = '031'", engine="pyogrio",
                    columns=["COUNTYFP20", "GEOID20", "MTFCC20", "ALAND20", "AWATER20", "HOUSING20", "POP20"])
blk = blk.sort_values("GEOID20").reset_index(drop=True)
assert len(blk) > 80000 and blk.GEOID20.is_unique and (blk.COUNTYFP20 == "031").all()
print("blocks", len(blk), "; CRS", blk.crs, "; ALAND20 = 0:", int((blk.ALAND20 == 0).sum()),
      "; with housing:", int((blk.HOUSING20 > 0).sum()))
geo_ll = blk.geometry.values  # EPSG:4269, exact shared vertices
blk_ft = blk.to_crs(3435)
geo_ft = blk_ft.geometry.values

# ------------------------------------------------------------------ roll (one row per PIN) and training PINs
h("Roll and training PINs")
a = pd.read_parquet(inp / "2022/assessment_data.parquet", columns=[
    "meta_pin", "meta_card_num", "meta_class", "meta_township_code", "meta_township_name", "meta_triad_code",
    "meta_nbhd_code", "meta_pin_num_cards", "ind_pin_is_multicard", "ind_pin_is_prorated",
    "meta_tieback_proration_rate", "loc_x_3435", "loc_y_3435", "loc_longitude", "loc_latitude"])
a = a.sort_values(["meta_pin", "meta_card_num"])
roll = a.groupby("meta_pin").agg(
    cls=("meta_class", "first"), town=("meta_township_code", "first"), town_name=("meta_township_name", "first"),
    triad=("meta_triad_code", "first"), nbhd=("meta_nbhd_code", "first"), cards=("meta_card_num", "size"),
    num_cards=("meta_pin_num_cards", "first"), multicard=("ind_pin_is_multicard", "first"),
    prorated_flag=("ind_pin_is_prorated", "first"), rate=("meta_tieback_proration_rate", "first"),
    x=("loc_x_3435", "first"), y=("loc_y_3435", "first"), lon=("loc_longitude", "first"), lat=("loc_latitude", "first"))
# eligibility predicate as in p2_population.py
flag_missing = roll["multicard"].isna() | roll["num_cards"].isna()
multicard = roll["multicard"].astype("boolean").fillna(True).astype(bool) | (roll["cards"] != 1) | flag_missing
prorated = (roll["rate"] < 1) | roll["prorated_flag"].astype("boolean").fillna(False).astype(bool)
no_xy = roll["x"].isna() | roll["y"].isna()
odd = ~roll["cls"].isin(MODEL_CLASSES)
roll["supported"] = ~multicard & ~prorated & ~no_xy & ~odd
roll["category"] = np.select([odd, no_xy, multicard & prorated, multicard, prorated],
                             ["class outside model classes", "no coordinates", "multicard and prorated", "multicard",
                              "prorated"], "supported")
print("roll PINs", len(roll), "; supported", int(roll.supported.sum()), "; without coordinates", int(no_xy.sum()))
print(roll.category.value_counts().to_string())
town_name = roll.groupby("town")["town_name"].first()
nbhd_town = roll.groupby("nbhd")["town"].agg(lambda s: s.mode().iloc[0])

# check that the State Plane coordinates agree with the longitude/latitude the Assessor also supplies
s = roll.dropna(subset=["x", "lon"]).sample(20000, random_state=1)
chk = gpd.GeoSeries(gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(3435)
dxy = np.hypot(chk.x.values - s.x.values, chk.y.values - s.y.values) * FT
print(f"State Plane vs lon/lat (20,000 PINs): median {np.median(dxy):.2f} m, 99th pct {np.quantile(dxy, .99):.2f} m, "
      f"max {dxy.max():.1f} m")

t = pd.read_parquet(inp / "2022/training_data.parquet", columns=[
    "meta_pin", "meta_sale_document_num", "meta_sale_date", "meta_nbhd_code", "meta_township_code",
    "ind_pin_is_multicard", "loc_x_3435", "loc_y_3435"])
t["retained"] = (t.ind_pin_is_multicard == False)  # noqa: E712 (the 2022 training filter of the Assessor's 01-train.R)
t = t.sort_values(["meta_pin", "meta_sale_date"])
xy_var = t.dropna(subset=["loc_x_3435"]).groupby("meta_pin")[["loc_x_3435", "loc_y_3435"]].agg(lambda v: v.max() - v.min())
moved = (np.hypot(xy_var.loc_x_3435, xy_var.loc_y_3435) * FT > 1).sum()
tr = t.groupby("meta_pin").agg(tx=("loc_x_3435", "last"), ty=("loc_y_3435", "last"), tnbhd=("meta_nbhd_code", "last"),
                               ttown=("meta_township_code", "last"), rows=("meta_pin", "size"))
tr["train_sales"] = t[t.retained].groupby("meta_pin")["meta_sale_document_num"].nunique()
tr["train_sales"] = tr["train_sales"].fillna(0).astype(int)
print("training rows", len(t), "; training PINs", len(tr), "; retained sales", int(tr.train_sales.sum()),
      "on", int((tr.train_sales > 0).sum()), "PINs; PINs not on the roll", int((~tr.index.isin(roll.index)).sum()),
      f"; PINs whose training coordinates vary by > 1 m across rows {moved}")

# ------------------------------------------------------------------ point in polygon
h("Point-in-polygon")


def to_block(x, y):
    """Block index for each point (-1 = unmatched), lowest GEOID where a point lies on a shared boundary."""
    ok = ~(np.isnan(x) | np.isnan(y))
    pts = shapely.points(x[ok], y[ok])
    ip, ib = blk_ft.sindex.query(pts, predicate="intersects")
    res = np.full(len(x), -1)
    m = pd.Series(ib).groupby(ip).min()  # blocks are sorted by GEOID, so min index = lowest GEOID
    multi = int((pd.Series(ib).groupby(ip).size() > 1).sum())
    sub = np.full(ok.sum(), -1)
    sub[m.index.values] = m.values
    res[ok] = sub
    return res, multi


roll["bidx"], multi_r = to_block(roll.x.to_numpy(float), roll.y.to_numpy(float))
tr["bidx_t"], multi_t = to_block(tr.tx.to_numpy(float), tr.ty.to_numpy(float))
for lab, d, col, hasxy, multi in [("roll", roll, "bidx", ~no_xy, multi_r), ("training", tr, "bidx_t", tr.tx.notna(), multi_t)]:
    un = hasxy & (d[col] < 0)
    print(f"{lab}: {len(d):,} PINs; matched {(d[col] >= 0).sum():,}; no coordinates {(~hasxy).sum():,}; "
          f"coordinates outside every block {un.sum():,}; on a shared boundary (lowest GEOID taken) {multi:,}")
    if un.any():
        xx = d.loc[un, "x" if lab == "roll" else "tx"].to_numpy(float)
        yy = d.loc[un, "y" if lab == "roll" else "ty"].to_numpy(float)
        _, dist = blk_ft.sindex.nearest(shapely.points(xx, yy), return_distance=True)
        print(f"  distance of unmatched points to the nearest Cook block (m): {np.round(np.sort(dist * FT), 1).tolist()[:40]}")
        if lab == "roll":
            print("  unmatched roll PINs by township and category:")
            print(d[un].groupby(["town", "category"]).size().to_string())
both = tr.index.intersection(roll.index)
rb, tb = roll.loc[both, "bidx"], tr.loc[both, "bidx_t"]
print(f"training PINs on the roll: {len(both):,}; roll and training coordinates give different blocks for "
      f"{((rb != tb) & (rb >= 0) & (tb >= 0)).sum():,} (the roll's is used)")
nb_diff = (roll.loc[both, "nbhd"] != tr.loc[both, "tnbhd"]).sum()
print(f"training PINs on the roll whose latest training neighbourhood differs from the roll's: {nb_diff:,} (the roll's is used)")

# ------------------------------------------------------------------ units
h("Units: split blocks that straddle neighbourhoods")
rp = roll[roll.bidx >= 0]
bn = rp.groupby("bidx")["nbhd"].agg(lambda s: tuple(sorted(set(s))))
split_blocks = bn[bn.map(len) > 1]
sup_bn = rp[rp.supported].groupby("bidx")["nbhd"].nunique()
print(f"blocks with roll PINs {len(bn):,}; straddling (roll PINs in > 1 neighbourhood) {len(split_blocks):,}, "
      f"giving {int(split_blocks.map(len).sum()):,} units; straddling on supported PINs alone {(sup_bn > 1).sum():,}")
print("neighbourhoods per straddling block:", split_blocks.map(len).value_counts().sort_index().to_dict())
straddle_town = rp[rp.bidx.isin(split_blocks.index)].groupby("bidx")["town"].nunique()
print(f"straddling blocks that also cross a township line: {(straddle_town > 1).sum():,}")

# labelled points of split blocks (distinct coordinates; a coordinate in two neighbourhoods takes the lowest code)
sp = rp[rp.bidx.isin(split_blocks.index)][["bidx", "x", "y", "nbhd"]].copy()
lab = sp.groupby(["bidx", "x", "y"])["nbhd"].agg(["min", "nunique"]).reset_index()
conflict_xy = lab[lab["nunique"] > 1]
conflict_pins = sp.merge(conflict_xy, on=["bidx", "x", "y"]).query("nbhd != min")
print(f"coordinates shared by PINs of different neighbourhoods inside straddling blocks: {len(conflict_xy):,} "
      f"({len(conflict_pins):,} PINs whose geometric portion is another neighbourhood's; their unit stays their own neighbourhood's)")
lab = lab.rename(columns={"min": "nbhd"})[["bidx", "x", "y", "nbhd"]]
trees = {b: (cKDTree(g[["x", "y"]].to_numpy(float)), g["nbhd"].to_numpy()) for b, g in lab.groupby("bidx")}

blk["has_pin"] = False
blk.loc[np.unique(np.r_[roll.bidx[roll.bidx >= 0], tr.bidx_t[tr.bidx_t >= 0]]), "has_pin"] = True
water = (blk.ALAND20 == 0) & ~blk.has_pin
print(f"water-only blocks (ALAND20 = 0) dropped: {water.sum():,}; ALAND20 = 0 blocks kept because a PIN falls in them: "
      f"{((blk.ALAND20 == 0) & blk.has_pin).sum():,}")
keep = ~water.to_numpy()

rows = []
for b in np.flatnonzero(keep):
    nb = bn.get(b)
    if nb is None:
        rows.append((b, ""))
    else:
        rows.extend((b, n) for n in nb)
units = pd.DataFrame(rows, columns=["bidx", "nbhd"])
units["unit_id"] = blk.GEOID20.values[units.bidx] + ":" + units.nbhd
units["split"] = units.bidx.isin(split_blocks.index)
uidx = {(b, n): i for i, (b, n) in enumerate(zip(units.bidx, units.nbhd))}
block_unit = {b: i for i, (b, n, s) in enumerate(zip(units.bidx, units.nbhd, units.split)) if not s}
print("units", len(units))

# PIN -> unit
roll["unit"] = [uidx.get((b, n), -1) if b >= 0 else -1 for b, n in zip(roll.bidx, roll.nbhd)]
tr["unit"] = -1
on_roll = tr.index.isin(roll.index)
tr.loc[on_roll, "unit"] = roll.loc[tr.index[on_roll], "unit"].to_numpy()
off = tr[~on_roll & (tr.bidx_t >= 0)]
u_off = []
for p, b, x, y in zip(off.index, off.bidx_t, off.tx, off.ty):
    if b in trees:
        tree, nbs = trees[b]
        u_off.append(uidx[(b, nbs[tree.query([x, y])[1]])])
    else:
        u_off.append(block_unit[b])
tr.loc[off.index, "unit"] = u_off
off_nb = units.nbhd.values[np.array(u_off)] if len(u_off) else np.array([])
print(f"off-roll training PINs mapped {len(off):,}: into units with roll PINs {(off_nb != '').sum():,} "
      f"(training neighbourhood differs from the unit's for {((off_nb != '') & (off_nb != off.tnbhd.to_numpy())).sum():,}); "
      f"into blocks with no roll PIN {(off_nb == '').sum():,}")

# ------------------------------------------------------------------ rook contiguity of blocks
h("Rook contiguity")
kb = np.flatnonzero(keep)
tree_ll = shapely.STRtree(geo_ll[kb])
qi, qj = tree_ll.query(geo_ll[kb], predicate="intersects")
m = qi < qj
bi, bj = kb[qi[m]], kb[qj[m]]
shared = shapely.intersection(shapely.boundary(geo_ll[bi]), shapely.boundary(geo_ll[bj]))
shared_ft = gpd.GeoSeries(shared, crs=blk.crs).to_crs(3435).values
length_m = shapely.length(shared_ft) * FT
rook = length_m > 0
print(f"intersecting block pairs {m.sum():,}: rook (shared boundary > 0) {rook.sum():,}; corner-only (queen, dropped) "
      f"{(~rook).sum():,}; rook edges shorter than 1 m {(length_m[rook] < 1).sum():,}")
bi, bj, shared_ft, length_m = bi[rook], bj[rook], shared_ft[rook], length_m[rook]

edges = []  # (unit a, unit b, length m, kind)
plain = ~(np.isin(bi, split_blocks.index) | np.isin(bj, split_blocks.index))
for a_, b_, l_ in zip(bi[plain], bj[plain], length_m[plain]):
    edges.append((block_unit[a_], block_unit[b_], l_, "block"))
# shared boundaries touching a split block: cut into <= 2 m pieces, allocate each to its Voronoi portion
cut = np.flatnonzero(~plain)
parts, pidx = shapely.get_parts(shapely.segmentize(shared_ft[cut], SEG_FT), return_index=True)
parts_l = parts[shapely.get_type_id(parts) == 1]
pidx_l = pidx[shapely.get_type_id(parts) == 1]
crd, ci = shapely.get_coordinates(parts_l, return_index=True)
same = ci[1:] == ci[:-1]
p0, p1 = crd[:-1][same], crd[1:][same]
mid = (p0 + p1) / 2
seg_len = np.hypot(*(p1 - p0).T) * FT
seg_edge = cut[pidx_l[ci[:-1][same]]]
side_u = {}
for side, barr in (("a", bi), ("b", bj)):
    bb = barr[seg_edge]
    u = np.full(len(bb), -1)
    for b in np.unique(bb):
        sel = bb == b
        if b in trees:
            tree_, nbs = trees[b]
            u[sel] = [uidx[(b, n)] for n in nbs[tree_.query(mid[sel])[1]]]
        else:
            u[sel] = block_unit[b]
    side_u[side] = u
alloc = pd.DataFrame({"a": side_u["a"], "b": side_u["b"], "l": seg_len}).groupby(["a", "b"])["l"].sum()
edges.extend((a_, b_, l_, "block (split side)") for (a_, b_), l_ in alloc.items())
print(f"block edges touching a straddling block {len(cut):,}, cut into {len(seg_len):,} pieces; "
      f"allocated length {seg_len.sum():,.0f} m of {length_m[cut].sum():,.0f} m")

# portions of one straddling block: Voronoi of its labelled points, clipped to the block, dissolved by neighbourhood
n_multi_part, n_internal_short = 0, 0
for b, g in lab.groupby("bidx"):
    poly = geo_ft[b]
    pts = shapely.points(g[["x", "y"]].to_numpy(float))
    cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(pts), extend_to=poly, ordered=True))
    cells = shapely.intersection(cells, poly)
    nbs = g["nbhd"].to_numpy()
    pieces = {n: shapely.union_all(cells[nbs == n]) for n in np.unique(nbs)}
    n_multi_part += sum(shapely.get_num_geometries(p) > 1 for p in pieces.values())
    keys = sorted(pieces)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            l_ = shapely.length(shapely.intersection(shapely.boundary(pieces[keys[i]]),
                                                     shapely.boundary(pieces[keys[j]]))) * FT
            if l_ > MIN_INTERNAL_M:
                edges.append((uidx[(b, keys[i])], uidx[(b, keys[j])], l_, "within straddling block"))
            elif l_ > 0:
                n_internal_short += 1
print(f"straddling-block portions in more than one piece: {n_multi_part:,}; internal boundaries under 1 cm dropped: "
      f"{n_internal_short:,}")
E = pd.DataFrame(edges, columns=["u", "v", "length_m", "kind"])
E[["u", "v"]] = np.sort(E[["u", "v"]].to_numpy(), axis=1)
E = E[E.u != E.v].groupby(["u", "v"], as_index=False).agg(length_m=("length_m", "sum"), kind=("kind", "first"))
print("unit edges", len(E), E.kind.value_counts().to_dict())

# ------------------------------------------------------------------ provisional neighbourhood of units without roll PINs
h("Neighbourhood for units without roll PINs (inventory rule)")
nb = units.nbhd.to_numpy().copy()
both_dir = pd.concat([E[["u", "v", "length_m"]], E.rename(columns={"u": "v", "v": "u"})[["u", "v", "length_m"]]])
rounds = 0
while True:
    cand = both_dir[(nb[both_dir.u] == "") & (nb[both_dir.v] != "")].copy()
    if cand.empty:
        break
    cand["nb"] = nb[cand.v]
    best = cand.groupby(["u", "nb"])["length_m"].sum().reset_index().sort_values(["u", "length_m", "nb"],
                                                                                   ascending=[True, False, True])
    best = best.drop_duplicates("u")
    nb[best.u.to_numpy()] = best.nb.to_numpy()
    rounds += 1
units["nbhd_assigned"] = nb
units["nbhd_rule"] = np.where(units.nbhd != "", "roll PINs", np.where(nb != "", "boundary rule", "none"))
print(f"rounds {rounds}; units labelled by the rule {(units.nbhd_rule == 'boundary rule').sum():,}; "
      f"left without a neighbourhood {(units.nbhd_rule == 'none').sum():,}")
units["town"] = units.nbhd_assigned.map(nbhd_town).fillna("")

# ------------------------------------------------------------------ unit attributes
for c in ["ALAND20", "AWATER20", "HOUSING20", "POP20"]:
    units[c.lower()] = blk[c].values[units.bidx]
units["block_geoid"] = blk.GEOID20.values[units.bidx]
rr = roll[roll.unit >= 0]
units["n_roll"] = rr.groupby("unit").size().reindex(units.index, fill_value=0).to_numpy()
units["n_supported"] = rr[rr.supported].groupby("unit").size().reindex(units.index, fill_value=0).to_numpy()
units["n_unit_class"] = rr[rr.supported & rr.cls.isin(["210", "295"])].groupby("unit").size().reindex(units.index, fill_value=0).to_numpy()
tt = tr[tr.unit >= 0]
units["n_train_pins"] = tt.groupby("unit").size().reindex(units.index, fill_value=0).to_numpy()
units["n_train_sales"] = tt.groupby("unit")["train_sales"].sum().reindex(units.index, fill_value=0).to_numpy()
units["pass_through"] = units.n_supported == 0
units["kind"] = np.select([units.n_supported > 0, units.n_roll > 0, units.n_train_pins > 0],
                          ["eligible parcels", "excluded roll parcels only", "off-roll training PINs only"], "no PIN")
print(units.kind.value_counts().to_string())

# ------------------------------------------------------------------ per-township graphs
h("Per township")
E["same_nbhd"] = nb[E.u] == nb[E.v]
E["town_u"], E["town_v"] = units.town.values[E.u], units.town.values[E.v]
print(f"edges crossing neighbourhoods {(~E.same_nbhd).sum():,}; crossing townships {(E.town_u != E.town_v).sum():,}")


def comps(n_nodes, eu, ev):
    if n_nodes == 0:
        return 0, 0, 0, np.array([], int)
    A = coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(n_nodes, n_nodes))
    k, lab_ = csgraph.connected_components(A, directed=False)
    sizes = np.bincount(lab_)
    deg = np.bincount(np.r_[eu, ev], minlength=n_nodes)
    return k, int((deg == 0).sum()), int(sizes.max()), lab_


units["comp_town"] = -1
units["comp_nbhd"] = -1
rows = []
for town in sorted(t_ for t_ in units.town.unique() if t_):
    U = units.index[units.town == town].to_numpy()
    loc = pd.Series(np.arange(len(U)), index=U)
    et = E[(E.town_u == town) & (E.town_v == town)]
    eu, ev = loc[et.u].to_numpy(), loc[et.v].to_numpy()
    k_all, sing_all, big_all, lab_all = comps(len(U), eu, ev)
    wn = et.same_nbhd.to_numpy()
    k_nb, sing_nb, big_nb, lab_nb = comps(len(U), eu[wn], ev[wn])
    units.loc[U, "comp_town"] = lab_all
    units.loc[U, "comp_nbhd"] = lab_nb
    ut = units.loc[U]
    sup = ut[ut.n_supported > 0]
    comp_sup_all = sup.groupby(lab_all[loc[sup.index]]).size()
    nb_sizes = ut.groupby("nbhd_assigned").size()
    rp_t = roll[roll.town == town]
    rows.append({
        "town": town, "town_name": town_name.get(town, ""), "triad": rp_t.triad.iloc[0] if len(rp_t) else "",
        "roll_pins": len(rp_t), "supported_pins": int(rp_t.supported.sum()),
        "roll_unmatched": int(((rp_t.bidx < 0) & rp_t.x.notna()).sum()), "roll_no_xy": int(rp_t.x.isna().sum()),
        "units": len(U), "units_eligible": int((ut.n_supported > 0).sum()),
        "pass_through": int(ut.pass_through.sum()),
        "pt_excluded_only": int((ut.kind == "excluded roll parcels only").sum()),
        "pt_offroll_training_only": int((ut.kind == "off-roll training PINs only").sum()),
        "pt_no_pin": int((ut.kind == "no PIN").sum()),
        "straddling_blocks": int(ut[ut.split].bidx.nunique()), "split_units": int(ut.split.sum()),
        "edges": len(et), "edges_within_nbhd": int(wn.sum()), "edges_cross_nbhd": int((~wn).sum()),
        "components": k_all, "singletons": sing_all, "largest_component": big_all,
        "components_with_eligible": int(len(comp_sup_all)),
        "components_nbhd_graph": k_nb, "singletons_nbhd_graph": sing_nb,
        "eligible_units_no_train_sale": int((sup.n_train_sales == 0).sum()),
        "median_train_sales_eligible_unit": float(sup.n_train_sales.median()) if len(sup) else np.nan,
        "median_supported_pins_eligible_unit": float(sup.n_supported.median()) if len(sup) else np.nan,
        "nbhds": int((nb_sizes.index != "").sum()), "largest_nbhd_block": int(nb_sizes.max()),
        "largest_nbhd": nb_sizes.idxmax(), "train_sales": int(ut.n_train_sales.sum()),
    })
T = pd.DataFrame(rows)
T.to_csv(out / "blocks_2022_township.csv", index=False)
print(T.drop(columns=["town_name"]).to_string(index=False))
nn = units[units.town == ""]
print(f"units outside every township (no neighbourhood reachable): {len(nn):,}; of which with a PIN: {(nn.kind != 'no PIN').sum():,}")

# ------------------------------------------------------------------ timing: dense vs sparse build of the largest blocks
h("Timing: dense and sparse build")


def laplacian(U):
    loc = pd.Series(np.arange(len(U)), index=U)
    e = E[E.u.isin(U) & E.v.isin(U)]
    i, j = loc[e.u].to_numpy(), loc[e.v].to_numpy()
    A = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(len(U), len(U))).tocsr()
    return A, len(e)


def time_build(label, U, dense=True):
    t0 = time.perf_counter()
    A, ne = laplacian(U)
    deg = np.asarray(A.sum(1)).ravel()
    L = (coo_matrix((deg, (np.arange(len(U)), np.arange(len(U)))), shape=A.shape) - A).tocsc()
    k, labs = csgraph.connected_components(A, directed=False)
    lu = splu(L + 1e-6 * identity(len(U), format="csc"))
    t_sparse = time.perf_counter() - t0
    rec = {"graph": label, "nodes": len(U), "edges": ne, "components": k, "sparse_s": round(t_sparse, 3),
           "sparse_lu_nnz": int(lu.L.nnz + lu.U.nnz)}
    if dense:
        t0 = time.perf_counter()
        Ld = L.toarray()
        w, V = np.linalg.eigh(Ld)
        keep_ = w > 1e-8 * max(w.max(), 1)
        ginv_diag = (V[:, keep_] ** 2 / w[keep_]).sum(1)  # diagonal of the generalised inverse (scaling factor)
        scale = float(np.exp(np.mean(np.log(ginv_diag[ginv_diag > 0])))) if (ginv_diag > 0).any() else np.nan
        rec.update({"dense_s": round(time.perf_counter() - t0, 3), "dense_mb": round(Ld.nbytes / 1e6, 1),
                    "zero_eigenvalues": int((~keep_).sum()), "scaling_geomean_ginv": round(scale, 4)})
    print(rec, flush=True)
    return rec


big = units[units.nbhd_assigned != ""].groupby("nbhd_assigned").size().sort_values(ascending=False)
south = big[big.index.map(lambda n: roll.loc[roll.nbhd == n, "triad"].iloc[0] if (roll.nbhd == n).any() else "") == "2"]
timing = [time_build(f"largest neighbourhood {big.index[0]}", units.index[units.nbhd_assigned == big.index[0]].to_numpy()),
          time_build(f"largest north neighbourhood {south.index[0]}", units.index[units.nbhd_assigned == south.index[0]].to_numpy())]
tbig = T.sort_values("units", ascending=False).iloc[0]
tsouth = T[T.triad == "2"].sort_values("units", ascending=False).iloc[0]
for lab_, tw in [("largest township", tbig), ("largest north township", tsouth)]:
    U = units.index[units.town == tw.town].to_numpy()
    timing.append(time_build(f"{lab_} {tw.town} {tw.town_name} (all edges)", U, dense=len(U) <= 25000))
pd.DataFrame(timing).to_csv(out / "blocks_2022_timing.csv", index=False)

# ------------------------------------------------------------------ write
h("Write")
pins = pd.DataFrame(index=roll.index.union(tr.index))
pins.index.name = "pin"
pins["on_roll"] = pins.index.isin(roll.index)
pins["in_training"] = pins.index.isin(tr.index)
pins["supported"] = roll.supported.reindex(pins.index).fillna(False).astype(bool)
pins["category"] = roll.category.reindex(pins.index).fillna("off roll")
pins["nbhd"] = roll.nbhd.reindex(pins.index).fillna(tr.tnbhd.reindex(pins.index))
pins["town"] = roll.town.reindex(pins.index).fillna(tr.ttown.reindex(pins.index))
pins["train_sales"] = tr.train_sales.reindex(pins.index).fillna(0).astype(int)
br = roll.bidx.reindex(pins.index).fillna(-1).astype(int)
bt = tr.bidx_t.reindex(pins.index).fillna(-1).astype(int)
bsel = np.where(pins.on_roll, br, bt)
pins["block_geoid"] = np.where(bsel >= 0, blk.GEOID20.values[np.maximum(bsel, 0)], None)
pins["block_geoid_training_xy"] = np.where(bt >= 0, blk.GEOID20.values[np.maximum(bt, 0)], None)
uu = np.where(pins.on_roll, roll.unit.reindex(pins.index).fillna(-1), tr.unit.reindex(pins.index).fillna(-1)).astype(int)
pins["unit_idx"] = uu
pins["unit_id"] = np.where(uu >= 0, units.unit_id.values[np.maximum(uu, 0)], None)
pins["match"] = np.select([pins.unit_idx >= 0, pins.block_geoid.isna() & pins.on_roll & roll.x.reindex(pins.index).isna()],
                          ["matched", "no coordinates"], "unmatched")
pins.loc[~pins.on_roll & tr.tx.reindex(pins.index).isna(), "match"] = "no coordinates"
print(pins.groupby(["on_roll", "match"]).size().to_string())
pins.reset_index().to_parquet(out / "blocks_2022.parquet", index=False)
units.index.name = "unit_idx"
units.drop(columns=["bidx"]).reset_index().to_parquet(out / "blocks_2022_units.parquet", index=False)
E.rename(columns={"u": "unit_a", "v": "unit_b"}).drop(columns=["town_u", "town_v"]).to_parquet(
    out / "blocks_2022_edges.parquet", index=False)

files = {f: out / f for f in ["blocks_2022.parquet", "blocks_2022_units.parquet", "blocks_2022_edges.parquet",
                              "blocks_2022_township.csv", "blocks_2022_timing.csv"]}
manifest = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": "public/scripts/v2/blocks.py",
    "script_sha256": sha256(__file__),
    "inputs": {
        "tiger": {"path": str(tiger_zip), "sha256": sha256(tiger_zip), "bytes": tiger_zip.stat().st_size,
                  "url": "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_17_tabblock20.zip",
                  "vintage": "TIGER/Line 2020, 2020 tabulation blocks (TABBLOCK20), Illinois; Cook = COUNTYFP20 031"},
        "assessment_data": {"path": str(inp / "2022/assessment_data.parquet"), "sha256": sha256(inp / "2022/assessment_data.parquet")},
        "training_data": {"path": str(inp / "2022/training_data.parquet"), "sha256": sha256(inp / "2022/training_data.parquet")},
    },
    "rules": {"point_in_polygon": "EPSG:3435 Assessor coordinates; boundary points to lowest GEOID; unmatched reported",
              "split": "block x neighbourhood where roll PINs fall in > 1 neighbourhood; portions by Voronoi of roll PINs",
              "contiguity": "rook, shared boundary > 0; split sides allocated in <= 2 m pieces; internal portions > 1 cm",
              "water": "ALAND20 = 0 blocks without a PIN dropped",
              "nbhd_for_empty_units": "provisional inventory rule: longest shared boundary with labelled neighbours"},
    "counts": {"blocks": len(blk), "water_dropped": int(water.sum()), "units": len(units),
               "units_eligible": int((units.n_supported > 0).sum()), "pass_through": int(units.pass_through.sum()),
               "straddling_blocks": len(split_blocks), "edges": len(E), "roll_pins": len(roll),
               "roll_pins_matched": int((roll.unit >= 0).sum()), "training_pins": len(tr),
               "training_pins_matched": int((tr.unit >= 0).sum())},
    "outputs": {f: {"sha256": sha256(p), "bytes": p.stat().st_size} for f, p in files.items()},
}
(out / "blocks_2022_manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(manifest["counts"], indent=1))
print(f"done in {time.time() - T0:.0f} s")
