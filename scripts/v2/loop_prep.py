"""Model development: prepare townships on the development split.

The 2025 county preparation (county2025_prep.py: placement, the characteristic coding of prep_v2.py on the 2025 char_*
columns + sale count, units, graphs) unchanged, on a different split: the Assessor's filter (multicard and outlier rows
dropped), sales before 20 October 2023 only (the held-out 2025 test sales are not read past this filter), training =
sales before 1 May 2023 (88 months, 2016-01 .. 2023-04), development evaluation = 1 May - 19 October 2023. Coding fitted
on the fit's training sales only. roll.parquet holds the development evaluation sales to predict.

Options:
  --fcode           F-code (an earlier flat coding, not used in the final model): explicit flat indicator seg_flat;
                    char_apts out of the categorical set, replaced for flats by log_nflats = log(n/2) (UNKNOWN / NONE /
                    missing -> the training median n of the township's flats with a known count) and nflats_unknown;
                    char_class rare levels merged within segment only (other_house, other_flat). Houses = single-family
                    + town/rowhouses (no town/row indicator).
  --segment S       keep only the sales of segment S (house | flat); coding fitted on that segment's training rows.
  --flat-nodes flatblock  (with --segment flat) census blocks containing flat buildings: see flat_block_graph().
  --flat-nodes building   (with --segment flat) the location field's nodes are individual flat buildings (PINs with a
                    sale in training or development), k = 5 nearest-neighbour graph within each neighbourhood.

  --scode           S-code (the coding fitted separately per segment; replaced F-code in every later model variant): see
                    use_scode().
  --salehist        sale history in place of the 4-year sale count: see use_salehist().
  --seghist         (with --salehist, for a fit where houses and flats share the published model's coding) one copy of
                    the history columns per segment (house rows / flat rows, 0 elsewhere).

  --context         six proximity / census columns as numeric main effects (an earlier variant, not used in the final
                    model): see use_context().
  --histsize        (with --salehist and --segment) history x centred log building area (per flat for flats; an earlier
                    variant, not used in the final model): see use_histsize(), add_size_centred().

  --fullhist 4y|long  (with --salehist and --segment) complete sale history from the Parcel Sales open data since 1999
                    (salehist_full.py) instead of the 2025 input's (since 2016): see use_fullhist().
  --gapcode bands|curve|exp|expc  (with --fullhist long) the gap bands of the separate house and flat models, or
                    hist_sold only + the random-walk gap curve of gl_loop.stan over hist_gbin (loop_fit.py --gcurve);
                    exp = no history column, the premium is gl_loop.stan's a * exp(-g / tau) (loop_fit.py --gexp);
                    expc = the same + hist_sold.

  --county          the Assessor's held-out design instead of the development split: county2025_prep.split() (their
                    01-train.R split, 94 months 2016-01 .. 2023-10; roll = the 38,177 held-out test sales), sale
                    history from every input sale strictly before each sale's date (no cutoff). Used only for the one
                    county test of the final model. With --fullhist: complete history from the Parcel Sales of every
                    township (PARCEL_D77 added), strictly before each sale's date, no cutoff.

Every train/roll row carries is_flat (meta_class 211-212).

Usage (repo root): .venv/bin/python -I public/scripts/v2/loop_prep.py --towns 77 73 76 74 31 11 --out public/data/v2/prep_loop_p
                   [--fcode | --scode] [--segment house|flat] [--flat-nodes building]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import county2025_prep as cp  # noqa: E402
import loop_fold as lf  # noqa: E402  (GL_FOLD selects the development fold; default = fold 1, 2023 cutoff)

COUNTY = "--county" in sys.argv                      # the Assessor's held-out design (94 months); else the development split
cp.NT = 94 if COUNTY else lf.NT
REPO = cp.REPO
DEV_START, HELD_OUT_START = lf.DEV_START, lf.HELD_OUT_START
FLAT = ("211", "212")
KNN = 5
NFLATS = {"TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6}
# the Parcel Sales download of the 30 townships outside the eight used in model development, for --county --fullhist
TOWNS_D77 = [str(x) for x in range(10, 40) if x not in (11, 31)] + ["71", "75"]
PARCEL_D77 = f"public/data/v2/parcel_sales/sales_t{'-'.join(TOWNS_D77)}.parquet"


def split():
    cols = cp.KEYS + ["meta_sale_date", "meta_sale_price", "meta_township_code", "meta_nbhd_code", "meta_class",
                      "ind_pin_is_multicard", "sv_is_outlier", "loc_x_3435", "loc_y_3435",
                      "meta_tieback_proration_rate", cp.COUNT, "char_apts"] + cp.CAT + cp.NUM
    d = pd.read_parquet(REPO / cp.INPUT, columns=list(dict.fromkeys(cols)))
    d = d[d["ind_pin_is_multicard"].eq(False).fillna(False).astype(bool)
          & d["sv_is_outlier"].eq(False).fillna(False).astype(bool)].copy()
    d["date"] = pd.to_datetime(d["meta_sale_date"])
    d = d[d["date"] < HELD_OUT_START].sort_values("date", kind="stable").reset_index(drop=True)
    d["test"] = d["date"] >= DEV_START
    s = dict(filtered_before_held_out=len(d), train=int((~d["test"]).sum()), dev=int(d["test"].sum()),
             last_train_date=str(d.loc[~d["test"], "date"].max().date()), first_dev_date=str(d.loc[d["test"], "date"].min().date()),
             last_dev_date=str(d["date"].max().date()))
    assert (s["filtered_before_held_out"], s["train"], s["dev"]) == lf.COUNTS, s
    return d, s


def county_split():
    """county2025_prep.split() (the Assessor's held-out design), reading char_apts even when S-/F-code has taken it
    out of the categorical set."""
    cat = list(cp.CAT)
    if "char_apts" not in cp.CAT:
        cp.CAT.append("char_apts")
    try:
        return cp.split()
    finally:
        cp.CAT[:] = cat


def is_flat(d):
    return d["meta_class"].astype(str).isin(FLAT)


def use_fcode():
    """F-code (explicit flat indicator and number of flats) wrapped around county2025_prep's characteristic coding."""
    fit0, apply0 = cp.fit_coding, cp.apply_coding
    if "char_apts" in cp.CAT:
        cp.CAT.remove("char_apts")                     # out of the categorical set (replaced by log_nflats)

    def seg(c):
        return np.where(pd.Series(c).astype(str).isin(FLAT), "flat", "house")

    def fit_coding(tr):
        coding = fit0(tr)
        s = tr["char_class"].astype("string").fillna("missing")
        sg = pd.Series(seg(tr["meta_class"]), index=tr.index)
        share = s.value_counts(normalize=True)
        rare = set(l for l in share[share < cp.RARE].index if l != "missing")
        merged = s.where(~s.isin(rare), "other_" + sg)
        counts = merged.value_counts()
        coding["categorical"]["char_class"] = dict(levels=sorted(counts.index.tolist()), reference=counts.index[0],
                                                   merged_into_other=[], rare_within_segment=sorted(rare),
                                                   train_counts={k: int(v) for k, v in counts.items()})
        f = is_flat(tr)
        n = tr.loc[f, "char_apts"].astype("string").map(NFLATS)
        coding["fcode"] = dict(nflats_median_known=float(n.median()) if n.notna().any() else 2.0,
                               train_flats=int(f.sum()), train_flats_unknown=int(n.isna().sum()),
                               note="log_nflats = log(n/2) for flats (0 for houses); unknown n -> the median")
        return coding

    def apply_coding(d, coding, counts):
        p = coding["categorical"]["char_class"]
        d = d.copy()
        s = d["char_class"].astype("string").fillna("missing")
        sg = pd.Series(seg(d["meta_class"]), index=d.index)
        s = s.where(~s.isin(p["rare_within_segment"]), "other_" + sg)
        unseen = ~s.isin(p["levels"])
        other = "other_" + sg
        s = s.where(~unseen, other.where(other.isin(p["levels"]), p["reference"]))
        counts["unseen_char_class_segment"] = int(unseen.sum())
        # the characteristic coding step below sees only known levels and the reference; empty merge list
        d["char_class"] = s
        X = apply0(d, coding, counts)
        f = is_flat(d)
        n = d["char_apts"].astype("string").map(NFLATS)
        unk = f & n.isna()
        counts["nflats_unknown"] = int(unk.sum())
        X["seg_flat"] = f.astype(float)
        X["log_nflats"] = np.where(f, np.log(n.fillna(coding["fcode"]["nflats_median_known"]) / 2), 0.0)
        X["nflats_unknown"] = unk.astype(float)
        return X

    cp.fit_coding, cp.apply_coding = fit_coding, apply_coding


PER_FLAT = ["char_beds", "char_rooms", "char_fbath", "char_hbath", "char_bldg_sf"]


def use_scode():
    """S-code: the characteristic coding fitted separately on each segment's training rows; each
    segment's columns (prefix h: / f:) are its own rows' values and 0 on the other segment's rows, so every
    characteristic coefficient is segment-specific; plus seg_flat. For flats, bedrooms, rooms, full and half baths and
    building area are per flat (divided by n, the number of flats; unknown n -> the training median of known counts),
    with log_nflats = log(n/2) and nflats_unknown. char_apts out of the categorical set; char_class rare levels
    (< 1% of the segment's training sales) merged within the segment."""
    fit0, apply0 = cp.fit_coding, cp.apply_coding
    if "char_apts" in cp.CAT:
        cp.CAT.remove("char_apts")

    def nflats(d, med):
        n = d["char_apts"].astype("string").map(NFLATS)
        return n.fillna(med).astype(float), n.isna()

    def per_flat(d, med):
        d = d.copy()
        n, _ = nflats(d, med)
        for c in PER_FLAT:
            d[c] = d[c].astype(float) / n
        return d

    def fit_coding(tr):
        f = is_flat(tr)
        n = tr.loc[f, "char_apts"].astype("string").map(NFLATS)
        med = float(n.median()) if n.notna().any() else 2.0
        segs = {}
        if (~f).any():
            segs["house"] = fit0(tr[~f])
        if f.any():
            segs["flat"] = fit0(per_flat(tr[f], med))
        return dict(scode=True, segments=segs, nflats_median_known=med, train_flats=int(f.sum()),
                    train_flats_unknown=int(n.isna().sum()), per_flat=PER_FLAT,
                    note="h:/f: columns are segment-specific (0 on the other segment's rows); flats' " + ", ".join(PER_FLAT)
                         + " divided by n; log_nflats = log(n/2), unknown n -> the median")

    def apply_coding(d, coding, counts):
        f = is_flat(d).to_numpy()
        med = coding["nflats_median_known"]
        parts = []
        for seg, pre, rows in (("house", "h:", ~f), ("flat", "f:", f)):
            if seg not in coding["segments"] or not rows.any():
                continue
            dd = d[rows] if seg == "house" else per_flat(d[rows], med)
            c = {}
            X = apply0(dd, coding["segments"][seg], c)
            counts.update({f"{pre}{k}": v for k, v in c.items()})
            if seg == "flat":
                n, unk = nflats(d[rows], med)
                X["log_nflats"] = np.log(n / 2)
                X["nflats_unknown"] = unk.astype(float)
                counts["nflats_unknown"] = int(unk.sum())
            parts.append(X.add_prefix(pre))
        # every segment's columns, also when this frame holds only one segment's rows
        cols = []
        for seg, pre in (("house", "h:"), ("flat", "f:")):
            if seg in coding["segments"]:
                p = coding["segments"][seg]
                names = list(p["numeric"]) + ["char_recent_renovation"] + [
                    f"{c}={l}" for c, q in p["categorical"].items() for l in q["levels"] if l != q["reference"]]
                if seg == "flat":
                    names += ["log_nflats", "nflats_unknown"]
                cols += [pre + x for x in names]
        X = pd.concat(parts).reindex(index=d.index, columns=cols).fillna(0.0) if parts else pd.DataFrame(0.0, index=d.index, columns=cols)
        X.insert(0, "seg_flat", f.astype(float))   # first, so the collinearity rule never drops the flat level
        return X

    cp.fit_coding, cp.apply_coding = fit_coding, apply_coding


def add_history(d, before, full=None):
    """The sale history of every row from the 2025 input's sales strictly before its date (salehist.py).
    With full: complete history from the Parcel Sales open data since 1999 (salehist_full.py,
    filtered as the Assessor's sale query); hist_n_prev then counts earlier sales since 1999, hist_n_prev_4y those within
    4 years, hist_gbin the gap bin of the latest earlier sale (GBIN_EDGES; 0 = no earlier sale)."""
    d = d.copy()
    if full:
        import salehist_full
        st = {}
        h = salehist_full.history(d, salehist_full.source(before=before or "2099-01-01", stats=st))
        d["hist_n_prev"], d["hist_months"], d["hist_first_new"] = h["n_prev"], h["months_since"], False
        d["hist_n_prev_4y"] = h["n_prev_4y"]
        d["hist_gbin"] = np.where(h["n_prev"] > 0, np.searchsorted(GBIN_EDGES, h["months_since"], side="right"), 0)
        return d, st
    import salehist
    h = salehist.history(d, salehist.source(before=before))
    d["hist_n_prev"], d["hist_months"], d["hist_first_new"] = h["n_prev"], h["months_since"], h["first_new"]
    return d, {}


GBIN_EDGES = [0, 6, 12, 18, 24, 36, 48, 60, 72, 96, 120, 144, 180, 240]   # months; 14 bins, the last open (for --gapcode curve)


def use_fullhist(mode, gapcode):
    """History columns from complete history (add_history(full=...)).
    mode 4y:        hist_gap_lt1y, hist_gap_1_2y, hist_gap_2_4y (latest earlier sale within 12 / 12-24 / 24-48 months);
                    reference = no earlier sale within 4 years (the Assessor's window); no hist_sold (= their sum).
    mode long, gapcode bands: the separate house and flat models' columns (hist_sold = an earlier sale since 1999;
                    bands < 1, 1-2, 2-4 y).
    mode long, gapcode curve: hist_sold only; the gap premium is the whitened random walk over hist_gbin in
                    gl_loop.stan (loop_fit.py --gcurve), outside the QR.
    mode long, gapcode exp: no history column (the premium a * exp(-g / tau) is in gl_loop.stan, outside the
                    QR, from hist_n_prev and hist_months); expc: hist_sold as well (a constant for "sold since 1999").
    Replaces the count as use_salehist does; houses and flats are separate fits (no per-segment copies)."""
    nf0 = cp.numeric_frame

    def numeric_frame(d, year):
        x = nf0(d, year).drop(columns=[cp.COUNT])
        m = d["hist_months"]
        bands = {"hist_gap_lt1y": m < 12, "hist_gap_1_2y": (m >= 12) & (m < 24), "hist_gap_2_4y": (m >= 24) & (m < 48)}
        if mode == "4y":                                 # the Assessor's inclusive endpoint (0, 4 y] (found in review)
            H = dict(bands, hist_gap_2_4y=(m >= 24) & (d["hist_n_prev_4y"] > 0))
        elif gapcode == "bands":
            H = {"hist_sold": d["hist_n_prev"] > 0, **bands}
        elif gapcode == "exp":
            H = {}
        else:
            H = {"hist_sold": d["hist_n_prev"] > 0}
        return pd.concat([x, pd.DataFrame(H, index=d.index).astype(float)], axis=1)

    cp.numeric_frame = numeric_frame


def use_salehist(seghist):
    """The count (meta_sale_count_past_n_years) is replaced in the coding's numeric frame by the sale history
    known at sale time: hist_sold (an earlier sale of the PIN since 2016-01) and gap bands hist_gap_lt1y,
    hist_gap_1_2y, hist_gap_2_4y (months since the latest earlier sale; 4 years or more is hist_sold's reference).
    Standardised by the existing rule like every numeric column (irrelevant through the QR). With seghist, one copy per
    segment (h_ house rows, f_ flat rows). The raw count stays in train/roll (record only)."""
    nf0 = cp.numeric_frame

    def numeric_frame(d, year):
        x = nf0(d, year).drop(columns=[cp.COUNT])
        m = d["hist_months"]
        H = pd.DataFrame({"hist_sold": (d["hist_n_prev"] > 0), "hist_gap_lt1y": m < 12,
                          "hist_gap_1_2y": (m >= 12) & (m < 24), "hist_gap_2_4y": (m >= 24) & (m < 48)},
                         index=d.index).astype(float)
        if seghist:
            f = is_flat(d).to_numpy()[:, None]
            H = pd.concat([(H * ~f).add_prefix("h_"), (H * f).add_prefix("f_")], axis=1)
        return pd.concat([x, H], axis=1)

    cp.numeric_frame = numeric_frame


CONTEXT = ["prox_nearest_cta_stop_dist_ft", "prox_nearest_major_road_dist_ft", "prox_nearest_park_dist_ft",
           "prox_num_foreclosure_per_1000_pin_past_5_years", "acs5_median_income_household_past_year",
           "acs5_percent_income_below_poverty_level"]


def use_context():
    """Six proximity / census columns from the 2025 input (an earlier variant, not used in the final model), fixed
    before fitting:
    log1p of the distances to the nearest CTA stop, major road and park; foreclosures per 1,000 PINs in the past 5
    years; log tract median household income; tract share below the poverty level. Negative distances / counts / shares
    and non-positive income are missing. Through the existing numeric coding: the fit's training median fill, mean and
    SD (each segment's own in S-code), the QR and the same N(0, 1) prior on the standardised coefficients."""
    for c in CONTEXT:
        if c not in cp.NUM:
            cp.NUM.append(c)
    nf0 = cp.numeric_frame

    def numeric_frame(d, year):
        x = nf0(d, year)
        v = {c: pd.to_numeric(d[c], errors="coerce").astype(float) for c in CONTEXT}
        for c in CONTEXT[:3]:
            x["ctx_log1p_" + c.replace("prox_nearest_", "").replace("_dist_ft", "")] = np.log1p(v[c].where(v[c] >= 0))
        x["ctx_foreclosure_per_1000"] = v[CONTEXT[3]].where(v[CONTEXT[3]] >= 0)
        x["ctx_log_tract_income"] = np.log(v[CONTEXT[4]].where(v[CONTEXT[4]] > 0))
        x["ctx_tract_poverty"] = v[CONTEXT[5]].where(v[CONTEXT[5]] >= 0)
        return x

    cp.numeric_frame = numeric_frame


def add_size_centred(d, segment):
    """For --histsize: a = log building area (houses) or log building area per flat (flats; n = char_apts, unknown n -> the
    township's flat-training median of known counts, as S-code); non-positive -> missing; filled with the township's
    segment-training median of a, minus the township's segment-training mean (after filling). d holds one segment."""
    d = d.copy()
    sf = pd.to_numeric(d["char_bldg_sf"], errors="coerce").astype(float)
    d["hx_a"] = np.nan
    stats = {}
    for T, g in d.groupby("town"):
        tr = ~g["test"]
        x = sf[g.index]
        if segment == "flat":
            n = g["char_apts"].astype("string").map(NFLATS)
            med_n = float(n[tr].median()) if n[tr].notna().any() else 2.0
            x = x / n.fillna(med_n).astype(float)
        a = np.log(x.where(x > 0))
        med = float(a[tr].median())
        af = a.fillna(med)
        mean = float(af[tr].mean())
        d.loc[g.index, "hx_a"] = af - mean
        stats[str(T)] = dict(median_log_sf=med, mean_log_sf=mean, filled=int(a.isna().sum()),
                             **(dict(median_nflats_known=med_n) if segment == "flat" else {}))
    return d, stats


def use_histsize():
    """The four history indicators x hx_a (add_size_centred), as numeric columns after the history columns;
    standardised by the existing rule (training statistics), the QR, N(0, 1) priors. Main effects kept."""
    nf0 = cp.numeric_frame

    def numeric_frame(d, year):
        x = nf0(d, year)
        for c in ("hist_sold", "hist_gap_lt1y", "hist_gap_1_2y", "hist_gap_2_4y"):
            x[c + "_x_logsf"] = x[c] * d["hx_a"].astype(float)
        return x

    cp.numeric_frame = numeric_frame


def building_graph(T, o, d_all, inputs):
    """Replace the prep's block units and edges by flat-building nodes and a k-NN graph within neighbourhood.
    d_all: the township's flat sales (training and development) with node (block unit) and nbhd_id."""
    t = pd.read_parquet(o / "train.parquet")
    r = pd.read_parquet(o / "roll.parquet")
    units_b = pd.read_parquet(o / "units.parquet")
    ne_b = pd.read_parquet(o / "nbhd_edges.parquet")
    pins = pd.concat([t[["meta_pin", "node", "nbhd_id", "date"]], r[["meta_pin", "node", "nbhd_id", "date"]]])
    pins = pins.sort_values("date").drop_duplicates("meta_pin", keep="last")[["meta_pin", "node", "nbhd_id"]]
    # coordinates: latest in the 2025 input (any row of the PIN), else the 2023 roll's, else the block's representative point
    xy = d_all.sort_values("date").dropna(subset=["loc_x_3435", "loc_y_3435"]).drop_duplicates("meta_pin", keep="last")
    pins = pins.merge(xy[["meta_pin", "loc_x_3435", "loc_y_3435"]], on="meta_pin", how="left")
    src = pd.Series("2025 input", index=pins.index)
    miss = pins["loc_x_3435"].isna()
    if miss.any():
        a = pd.read_parquet(REPO / "ccao/inputs/2023/assessment_data.parquet",
                            columns=["meta_pin", "loc_x_3435", "loc_y_3435"]).dropna().drop_duplicates("meta_pin")
        a = a.set_index("meta_pin")
        pins.loc[miss, "loc_x_3435"] = pins.loc[miss, "meta_pin"].map(a["loc_x_3435"])
        pins.loc[miss, "loc_y_3435"] = pins.loc[miss, "meta_pin"].map(a["loc_y_3435"])
        src[miss] = "2023 roll"
    miss = pins["loc_x_3435"].isna()
    if miss.any():
        import geopandas as gpd
        U = pd.read_parquet(REPO / cp.DATA / "blocks_2023_units.parquet", columns=["unit_idx", "block_geoid"])
        gid = pins.loc[miss, "node"].map(units_b.set_index("node")["unit_idx"]).map(U.set_index("unit_idx")["block_geoid"])
        blk = gpd.read_file(f"zip://{REPO / cp.TIGER}", where="COUNTYFP20 = '031'", engine="pyogrio",
                            columns=["COUNTYFP20", "GEOID20"])
        blk = blk[blk["GEOID20"].isin(set(gid))].to_crs(3435).set_index("GEOID20")
        rp = blk.geometry.representative_point()
        pins.loc[miss, "loc_x_3435"] = gid.map(rp.x).to_numpy()
        pins.loc[miss, "loc_y_3435"] = gid.map(rp.y).to_numpy()
        src[miss] = "block representative point"
    assert pins[["loc_x_3435", "loc_y_3435"]].notna().all().all()
    pins["xy_source"] = src
    pins = pins.sort_values(["nbhd_id", "meta_pin"]).reset_index(drop=True)
    pins["block_node"] = pins["node"]
    pins["node"] = np.arange(len(pins))
    # k-NN within neighbourhood, symmetrised, binary
    ea, eb, el = [], [], []
    for k, g in pins.groupby("nbhd_id", sort=True):
        if len(g) < 2:
            continue
        P = g[["loc_x_3435", "loc_y_3435"]].to_numpy(float)
        kk = min(KNN, len(g) - 1)
        dist, idx = cKDTree(P).query(P, k=kk + 1)
        nodes = g["node"].to_numpy()
        for i in range(len(g)):
            for j, dd in zip(idx[i, 1:], dist[i, 1:]):
                a_, b_ = sorted((nodes[i], nodes[j]))
                ea.append(a_); eb.append(b_); el.append(dd * cp_ft())
    E = pd.DataFrame(dict(a=ea, b=eb, length_m=el)).drop_duplicates(["a", "b"]).sort_values(["a", "b"]).reset_index(drop=True)
    E["same_nbhd"] = True
    present = set(pins["nbhd_id"])
    ne = ne_b[ne_b["nbhd_a"].isin(present) & ne_b["nbhd_b"].isin(present)].reset_index(drop=True)
    units = pd.DataFrame(dict(node=pins["node"], unit_idx=-1, unit_id=pins["meta_pin"], nbhd_id=pins["nbhd_id"],
                              comp=0, pass_through=False, kind="flat building", n_supported=1,
                              n_train_sales=pins["meta_pin"].map(t["meta_pin"].value_counts()).fillna(0).astype(int),
                              x=pins["loc_x_3435"].to_numpy(float), y=pins["loc_y_3435"].to_numpy(float),
                              block_node=pins["block_node"], xy_source=pins["xy_source"].to_numpy()))
    node_of = dict(zip(pins["meta_pin"], pins["node"]))
    t["block_node"], r["block_node"] = t["node"], r["node"]
    t["node"], r["node"] = t["meta_pin"].map(node_of).astype(int), r["meta_pin"].map(node_of).astype(int)
    assert (t["nbhd_id"].to_numpy() == units["nbhd_id"].to_numpy()[t["node"]]).all()
    assert (r["nbhd_id"].to_numpy() == units["nbhd_id"].to_numpy()[r["node"]]).all()
    units.to_parquet(o / "units.parquet", index=False)
    E.to_parquet(o / "edges.parquet", index=False)
    ne.to_parquet(o / "nbhd_edges.parquet", index=False)
    t.to_parquet(o / "train.parquet", index=False)
    r.to_parquet(o / "roll.parquet", index=False)
    deg = pd.concat([E["a"], E["b"]]).value_counts()
    return dict(node_type="building", knn=KNN, buildings=len(units), building_edges=len(E),
                nbhds_with_flats=int(units["nbhd_id"].nunique()), nbhd_edges_kept=len(ne),
                one_building_nbhds=int((units["nbhd_id"].value_counts() == 1).sum()),
                mean_degree=float(deg.reindex(units["node"], fill_value=0).mean()),
                xy_source=units["xy_source"].value_counts().to_dict(),
                buildings_with_training_sale=int((units["n_train_sales"] > 0).sum()))


def flat_block_graph(o):
    """Flats on census blocks containing flat buildings. The prep's block units are restricted to the blocks with a
    flat sale in training or development (the same rule as building nodes, aggregated to blocks); rook edges among them
    kept; pieces within a neighbourhood are joined by the published model's MST bridges on block polygons at fit time
    (fit_struct.build_structure); neighbourhood edges restricted to neighbourhoods present."""
    t = pd.read_parquet(o / "train.parquet")
    r = pd.read_parquet(o / "roll.parquet")
    units = pd.read_parquet(o / "units.parquet")
    E = pd.read_parquet(o / "edges.parquet")
    ne = pd.read_parquet(o / "nbhd_edges.parquet")
    keep = np.unique(np.r_[t["node"].to_numpy(), r["node"].to_numpy()])
    new_of = np.full(len(units), -1)
    new_of[keep] = np.arange(len(keep))
    u = units.iloc[keep].reset_index(drop=True)
    u["block_node"] = u["node"]
    u["node"] = np.arange(len(u))
    E = E[(new_of[E["a"]] >= 0) & (new_of[E["b"]] >= 0)].copy()
    E["a"], E["b"] = new_of[E["a"]], new_of[E["b"]]
    present = set(u["nbhd_id"])
    ne = ne[ne["nbhd_a"].isin(present) & ne["nbhd_b"].isin(present)].reset_index(drop=True)
    t["block_node"], r["block_node"] = t["node"], r["node"]
    t["node"], r["node"] = new_of[t["node"]], new_of[r["node"]]
    assert (t["node"] >= 0).all() and (r["node"] >= 0).all()
    assert (t["nbhd_id"].to_numpy() == u["nbhd_id"].to_numpy()[t["node"]]).all()
    for x, f in ((u, "units"), (E.reset_index(drop=True), "edges"), (ne, "nbhd_edges"), (t, "train"), (r, "roll")):
        x.to_parquet(o / f"{f}.parquet", index=False)
    return dict(node_type="flat blocks", blocks_township=len(units), blocks_with_flat_sale=len(u), block_edges=len(E),
                nbhds_with_flats=int(u["nbhd_id"].nunique()), nbhd_edges_kept=len(ne),
                blocks_with_training_sale=int(np.isin(np.arange(len(u)), t["node"].unique()).sum()))


def cp_ft():
    return 1200 / 3937                                   # EPSG:3435 is in US survey feet; metres per foot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fcode", action="store_true")
    ap.add_argument("--scode", action="store_true")
    ap.add_argument("--segment", choices=["house", "flat"])
    ap.add_argument("--flat-nodes", choices=["block", "building", "flatblock"], default="block")
    ap.add_argument("--salehist", action="store_true")
    ap.add_argument("--seghist", action="store_true")
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--histsize", action="store_true")
    ap.add_argument("--county", action="store_true")
    ap.add_argument("--fullhist", choices=["4y", "long"], help="A7: complete history from the Parcel Sales open data")
    ap.add_argument("--gapcode", choices=["bands", "curve", "exp", "expc"], default="bands", help="A7, A10; with --fullhist long")
    a = ap.parse_args()
    assert not a.histsize or (a.salehist and not a.seghist and a.segment in ("house", "flat"))
    assert a.salehist or not a.seghist
    assert not a.fullhist or (a.salehist and not a.seghist and not a.histsize and a.segment)
    assert a.gapcode == "bands" or a.fullhist == "long"
    if a.fullhist:
        import salehist_full
        if a.county:                 # every township's Parcel Sales (a third download, townships disjoint)
            salehist_full.PARCEL_FILES = salehist_full.PARCEL_FILES + [REPO / PARCEL_D77]
            salehist_full.TOWNS_DOWNLOADED = salehist_full.TOWNS_DOWNLOADED + TOWNS_D77
        assert set(a.towns) <= set(salehist_full.TOWNS_DOWNLOADED), "no Parcel Sales download for a requested township"
    assert a.flat_nodes == "block" or a.segment == "flat"
    assert not (a.fcode and a.scode)
    if a.fcode:
        use_fcode()
    if a.scode:
        use_scode()
    if a.fullhist:
        use_fullhist(a.fullhist, a.gapcode)
    elif a.salehist:
        use_salehist(a.seghist)
    if a.context:
        use_context()
    if a.histsize:
        use_histsize()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    d, s = county_split() if a.county else split()
    print(f"split: {s}", flush=True)
    if a.salehist:
        d, hst = add_history(d, before=None if a.county else HELD_OUT_START, full=a.fullhist)
        s["salehist"] = dict(source=("Parcel Sales open data (wvhk-k5uv), salehist_full.source: " + json.dumps(hst)
                                     if a.fullhist else "2025 input, any flag, distinct PIN x document, one per date, "
                                    + ("all dates (county design)" if a.county else "before " + HELD_OUT_START)),
                             sold_before=int((d["hist_n_prev"] > 0).sum()),
                             sold_before_train=int(((d["hist_n_prev"] > 0) & ~d["test"]).sum()),
                             sold_before_dev=int(((d["hist_n_prev"] > 0) & d["test"]).sum()),
                             later_sale_after_new_build_first=int(d["hist_first_new"].sum()))
    d, todo = cp.place(d)
    if a.segment:
        d = d[is_flat(d) == (a.segment == "flat")].reset_index(drop=True)
    if a.histsize:
        d, s["histsize"] = add_size_centred(d, a.segment)
        assert d["hx_a"].notna().all()
    todo.to_csv(out / "placement.csv", index=False)
    inputs = {p: cp.sha(p) for p in [cp.INPUT, f"{cp.DATA}/blocks_2023.parquet", f"{cp.DATA}/blocks_2023_units.parquet",
                                     f"{cp.DATA}/blocks_2023_edges.parquet", "ccao/inputs/2023/assessment_data.parquet",
                                     cp.TIGER]}
    s.update(inputs=inputs, options=dict(fcode=a.fcode, scode=a.scode, segment=a.segment, flat_nodes=a.flat_nodes,
                                         salehist=a.salehist, seghist=a.seghist, county=a.county,
                                         context=a.context, histsize=a.histsize, fullhist=a.fullhist,
                                         gapcode=a.gapcode,
                                         n_gbin=len(GBIN_EDGES) if a.gapcode == "curve" else 0),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), towns={})
    for T in a.towns:
        c = cp.prep_town(T, d, out, inputs)
        o = out / f"t{T}"
        for f in ("train.parquet", "roll.parquet"):
            x = pd.read_parquet(o / f)
            x["is_flat"] = is_flat(x)
            if a.salehist:
                hc = ["hist_n_prev", "hist_months", "hist_first_new"] + (["hist_n_prev_4y", "hist_gbin"] if a.fullhist else [])
                x = x.merge(d[cp.KEYS + hc], on=cp.KEYS, how="left", validate="one_to_one")
                assert x["hist_n_prev"].notna().all()
            x.to_parquet(o / f, index=False)
        if a.flat_nodes == "building":
            c["building_graph"] = building_graph(T, o, d[d["town"] == T], inputs)
            print(f"t{T}: {c['building_graph']}", flush=True)
        elif a.flat_nodes == "flatblock":
            c["flat_block_graph"] = flat_block_graph(o)
            print(f"t{T}: {c['flat_block_graph']}", flush=True)
        man = json.loads((o / "prep_manifest.json").read_text())
        man.update(loop_options=s["options"], loop_script_sha256=s["script_sha256"], counts=c,
                   node_type={"building": "building", "flatblock": "flat blocks"}.get(a.flat_nodes, "block"),
                   outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.iterdir())
                            if p.name != "prep_manifest.json"})
        (o / "prep_manifest.json").write_text(json.dumps(man, indent=1))
        s["towns"][T] = c
    s["by_region"] = {T: dict(train=c["train_sales"], test=c["test_sales"]) for T, c in s["towns"].items()}  # county2025_run.py
    (out / "split_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
