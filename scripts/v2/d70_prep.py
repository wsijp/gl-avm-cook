"""House preparations for model variants rh1h (resale terms scaled by house age + repeated clean sales; not used in the
final model) and ixah (explicit age interactions by house class and by building size: the final house model), each
derived from the house preparation with the flip-purchase flags (prep_loop_k1h; Lake / Hyde Park prep_loop_k1h_lhp) by
adding columns; everything else (keys, every parent column, units, graphs, the since-2016 history, flip-purchase flags)
is the parent's, checked. Fixed before fitting.

Raw age = the preparation's own house age at sale (x:h:age back-transformed with its training mean and SD).
rh1h — a = (min(age, 120) - 60) / 60 (bounded, continuous; fixed before fitting):
  rh_sold_age  = hist_sold * a          (hist_sold = an earlier qualifying sale since 2016, hist_n_prev > 0)
  rh_taper_age = w(g) * a               (w = the taper of the neighbourhood-varying resale premium: 1 to 2 years,
                                         cosine to 0 at 4 years; g = hist_months / 12)
  rh_clean2_4y = 1 if meta_sale_count_past_n_years >= 2 (the Assessor's count of non-outlier sales of the PIN in the
                 4 years strictly before the sale; checked against hist_n_prev: count <= earlier sales since 2016 in
                 every township's 2020+ training rows)
ixah — ac = age - the township's house-training mean age; s = log building area (back-transformed) - its training mean:
  ix_age_x_<class> = ac * I(class indicator), for every class indicator column the parent keeps in that township;
  ix_age_x_logsf  = ac * s.
Training and development rows alike (history strictly before each row's sale). New columns are standardised with their
training mean and SD and get the N(0, 1) prior of every coefficient (inside the QR), as every sale-history column.
A new column with no training variation, or collinear with the parent's columns (QR relative diagonal <= 1e-7), is
dropped and listed in coding.json (the existing rule).

Usage (repo root): .venv/bin/python -I public/scripts/v2/d70_prep.py [--arms rh1h ixah]
Writes public/data/v2/prep_loop_{rh1h,ixah}{,_lhp}/t<T>/.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "public/scripts/v2"))
from loop_fit import np_taper  # noqa: E402  (the resale-premium taper, the same function the fit uses)

import loop_fold as lf  # noqa: E402  (the development fold's preparation folder)
DATA = REPO / lf.DATA
SRC = "prep_loop_k1h"
TOWNS = {"": ["77", "73", "76", "74", "31", "11"], "_lhp": ["72", "70"]}


def raw_numeric(df, coding, name):
    s = coding["segments"]["house"]["numeric"][name]
    return df[f"x:h:{name}"].to_numpy(float) * s["sd"] + s["mean"]


def rh1_cols(df, coding):
    age = raw_numeric(df, coding, "age")
    a = (np.minimum(age, 120) - 60) / 60
    sold = (df["hist_n_prev"] > 0).to_numpy()
    g = np.where(sold, df["hist_months"].to_numpy(float) / 12, 0.0)
    w = np.where(sold, np_taper(g), 0.0)
    cnt, late = df["meta_sale_count_past_n_years"].to_numpy(float), (pd.to_datetime(df["date"]) >= "2020-01-01").to_numpy()
    assert (cnt[late] <= df["hist_n_prev"].to_numpy(float)[late]).all(), "the count must be strictly before the sale"
    return {"h:rh_sold_age": sold * a, "h:rh_taper_age": w * a,
            "h:rh_clean2_4y": (df["meta_sale_count_past_n_years"].to_numpy(float) >= 2).astype(float)}


def ixa_cols(df, coding, age_mean, sf_mean):
    ac = raw_numeric(df, coding, "age") - age_mean
    s = raw_numeric(df, coding, "log_bldg_sf") - sf_mean
    out = {}
    for c in coding["columns"]:
        if c.startswith("h:char_class="):
            ind = (df["x:" + c].to_numpy(float) > 0).astype(float)    # a standardised 0/1 column is > 0 exactly where 1
            out["h:ix_age_x_" + c.split("=")[1]] = ac * ind
    out["h:ix_age_x_logsf"] = ac * s
    return out


def build(arm, sfx, T):
    i, o = DATA / (SRC + sfx) / f"t{T}", DATA / f"prep_loop_{arm}{sfx}" / f"t{T}"
    if o.exists():
        shutil.rmtree(o)
    shutil.copytree(i, o)
    coding = json.loads((i / "coding.json").read_text())
    t, r = pd.read_parquet(i / "train.parquet"), pd.read_parquet(i / "roll.parquet")
    if arm == "rh1h":
        raw_t, raw_r = rh1_cols(t, coding), rh1_cols(r, coding)
    else:
        am, sm = raw_numeric(t, coding, "age").mean(), raw_numeric(t, coding, "log_bldg_sf").mean()
        raw_t, raw_r = ixa_cols(t, coding, am, sm), ixa_cols(r, coding, am, sm)
    spec, dropped = {}, {}
    cols = list(coding["columns"])
    for c in raw_t:
        m_, s_ = float(raw_t[c].mean()), float(raw_t[c].std(ddof=1))
        if not s_ > 0:
            dropped[c] = "no training variation"
            continue
        t["x:" + c], r["x:" + c] = (raw_t[c] - m_) / s_, (raw_r[c] - m_) / s_
        Z = t[["x:" + k for k in cols + [c]]].to_numpy()
        Z = Z - Z.mean(axis=0)
        rel = np.abs(np.diag(np.linalg.qr(Z, mode="r"))) / np.linalg.norm(Z, axis=0)
        if rel[-1] <= 1e-7:
            dropped[c] = f"collinear (rel {rel[-1]:.1e})"
            t.drop(columns="x:" + c, inplace=True)
            r.drop(columns="x:" + c, inplace=True)
            continue
        assert (rel > 1e-7).all()
        cols.append(c)
        spec[c] = dict(mean=m_, sd=s_, share_nonzero_train=float((raw_t[c] != 0).mean()),
                       share_nonzero_dev=float((raw_r[c] != 0).mean()))
    coding["columns"] = cols
    coding["d70"] = dict(arm=arm, columns=spec, dropped=dropped, note=__doc__.split("\n\n")[0].replace("\n", " "))
    (o / "coding.json").write_text(json.dumps(coding, indent=1))
    t.to_parquet(o / "train.parquet", index=False)
    r.to_parquet(o / "roll.parquet", index=False)
    man = json.loads((i / "prep_manifest.json").read_text())
    man["loop_options"] = dict(man["loop_options"], d70=arm)
    man["parent"] = SRC + sfx
    man["coding_sha256"] = hashlib.sha256((o / "coding.json").read_bytes()).hexdigest()
    man["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.iterdir())
                      if p.name != "prep_manifest.json"}
    (o / "prep_manifest.json").write_text(json.dumps(man, indent=1))
    for f in ("train.parquet", "roll.parquet"):
        a_, b_ = pd.read_parquet(i / f), pd.read_parquet(o / f)
        assert a_.equals(b_[a_.columns]), (T, f)
        assert set(b_.columns) - set(a_.columns) == {"x:" + c for c in spec}, (T, f)
    print(f"prep_loop_{arm}{sfx} t{T}: added {len(spec)} {list(spec)}; dropped {dropped}; "
          f"nonzero train {dict((k, round(v['share_nonzero_train'], 3)) for k, v in spec.items())}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["rh1h", "ixah"])
    a = ap.parse_args()
    for arm in a.arms:
        for sfx, towns in TOWNS.items():
            for T in towns:
                build(arm, sfx, T)
            ss = json.loads((DATA / f"{SRC}{sfx}" / "split_summary.json").read_text())
            ss["options"] = dict(ss["options"], d70=arm)
            ss["parent"] = SRC + sfx
            (DATA / f"prep_loop_{arm}{sfx}" / "split_summary.json").write_text(json.dumps(ss, indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
