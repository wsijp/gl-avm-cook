"""House preparations for model variants k1h (flip-purchase indicators), k2h (a long-tenure level) and k12h (both),
each derived from the house preparation of the separate house and flat models with sale history since 2016
(prep_loop_h2h; Lake / Hyde Park prep_loop_h2h_lhp) by adding columns; everything else (keys, every parent column,
units, graphs, the since-2016 history and gap bands) is the parent's, checked.

k1h — flip purchase (exploratory): for a TRAINING sale, the NEXT distinct sale date of the same PIN among the same
preparation's retained house training sales (strictly later date, so before 1 May 2023; the next sale's price is never
used) is < 12 months or 12-< 24 months later (months = days / 30.4375, the history convention of salehist.py); two
mutually exclusive columns. Development rows: raw 0 before standardisation (their next sale is unknown; nothing is read
from development or held-out sales). A sale of the last two training years cannot be flagged at the full window
(right-censoring; stated).
k2h — long tenure: the complete history (salehist_full via the complete-history house preparation prep_loop_ehh:
hist_n_prev, hist_months = months since the latest earlier qualifying sale since 1999) has a latest earlier sale dated
before 1 January 2016, AND the parent's since-2016 history (the 2025 input) has none (hist_n_prev = 0); training and
development rows alike (history strictly before the row's date, as for complete history). Rows where the two sources
disagree (complete history has a 2016+ earlier sale that the parent does not, or the reverse) keep the long-tenure
column at 0, i.e. stay as the parent codes them; counted in coding.json.
New columns are standardised with their training mean and SD (the existing rule) and get the N(0, 1) prior of every
coefficient (inside the QR).

k1f (run later) — the same flip-purchase columns for flats, on the flat model's preparation (prep_loop_epf: complete
history since 1999, the fading resale premium curve; building nodes), prefix f:.

Usage (repo root): .venv/bin/python -I public/scripts/v2/d63_prep.py [--arms k1h k2h k12h k1f]
Writes public/data/v2/prep_loop_{k1h,k2h,k12h,k1f}{,_lhp}/t<T>/.
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
import loop_fold as lf  # noqa: E402  (the development fold's preparation folder)
DATA = REPO / lf.DATA
KEYS = ["meta_pin", "meta_sale_document_num", "meta_card_num"]
TOWNS = {"": ["77", "73", "76", "74", "31", "11"], "_lhp": ["72", "70"]}
BAND_DEF = {"lt12": (0, 12), "12_24": (12, 24)}                 # months
ARMS = dict(k1h=("prep_loop_h2h", "h", True, False), k2h=("prep_loop_h2h", "h", False, True),
            k12h=("prep_loop_h2h", "h", True, True), k1f=("prep_loop_epf", "f", True, False))   # src, prefix, flip, pre2016


def flip_cols(t, bands, pre="h"):
    d = pd.to_datetime(t["date"])
    o = pd.DataFrame({"pin": t["meta_pin"].to_numpy(), "d": d.to_numpy(), "i": np.arange(len(t))})
    o = o.sort_values(["pin", "d", "i"])
    # the next strictly later sale date of the same PIN
    nd = np.full(len(t), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for _, g in o.groupby("pin", sort=False):
        ds, ix = g["d"].to_numpy(), g["i"].to_numpy()
        for k in range(len(ds)):
            later = ds[ds > ds[k]]
            if len(later):
                nd[ix[k]] = later.min()
    gap = (pd.to_datetime(nd) - d.to_numpy()).days.to_numpy() / 30.4375
    out = {}
    for b in bands:
        lo, hi = BAND_DEF[b]
        out[f"{pre}:flipbuy_{b}"] = ((gap >= lo) & (gap < hi)).astype(float)
    return out, gap


def build(arm, sfx, T, bands):
    src, pre, flip, pre2016 = ARMS[arm]
    i, o = DATA / (src + sfx) / f"t{T}", DATA / f"prep_loop_{arm}{sfx}" / f"t{T}"
    if o.exists():
        shutil.rmtree(o)
    shutil.copytree(i, o)
    coding = json.loads((i / "coding.json").read_text())
    t, r = pd.read_parquet(i / "train.parquet"), pd.read_parquet(i / "roll.parquet")
    raw_t, raw_r, info = {}, {}, {}
    if flip:
        ft, gap = flip_cols(t, bands, pre)
        for c, v in ft.items():
            raw_t[c], raw_r[c] = v, np.zeros(len(r))
        info["flip"] = dict(bands=bands, share=dict((c, float(v.mean())) for c, v in ft.items()),
                            n_training_with_later_training_sale=int(np.isfinite(gap).sum()))
    if pre2016:
        for df, raw, f in ((t, raw_t, "train.parquet"), (r, raw_r, "roll.parquet")):
            e = pd.read_parquet(DATA / f"prep_loop_ehh{sfx}" / f"t{T}" / f, columns=KEYS + ["hist_n_prev", "hist_months"])
            m = df[KEYS].merge(e, on=KEYS, how="left", validate="one_to_one")
            assert m["hist_n_prev"].notna().all() and (m[KEYS].to_numpy() == df[KEYS].to_numpy()).all()
            n99, n16 = m["hist_n_prev"].to_numpy(), df["hist_n_prev"].to_numpy()
            prev = pd.to_datetime(df["date"]) - pd.to_timedelta(np.round(m["hist_months"].fillna(0).to_numpy() * 30.4375), unit="D")
            c_pre = (n99 > 0) & (prev < "2016-01-01").to_numpy()        # complete history: latest earlier sale before 2016
            c_16 = (n99 > 0) & ~c_pre                                    # complete history: an earlier sale since 2016
            info.setdefault("audit", {})[f] = dict(
                n=len(df), h2_none_complete_none=int(((n16 == 0) & (n99 == 0)).sum()),
                h2_none_complete_pre2016=int(((n16 == 0) & c_pre).sum()), h2_none_complete_2016plus=int(((n16 == 0) & c_16).sum()),
                h2_sold_complete_2016plus=int(((n16 > 0) & c_16).sum()), h2_sold_complete_pre2016=int(((n16 > 0) & c_pre).sum()),
                h2_sold_complete_none=int(((n16 > 0) & (n99 == 0)).sum()))
            raw["h:hist_pre2016"] = ((n16 == 0) & c_pre).astype(float)
        info["pre2016"] = dict(share_train=float(raw_t["h:hist_pre2016"].mean()), share_dev=float(raw_r["h:hist_pre2016"].mean()))
    spec = {}
    for c in raw_t:
        m_, s_ = float(raw_t[c].mean()), float(raw_t[c].std(ddof=1))
        assert s_ > 0, (T, c)
        t["x:" + c], r["x:" + c] = (raw_t[c] - m_) / s_, (raw_r[c] - m_) / s_
        spec[c] = dict(mean=m_, sd=s_)
    cols = coding["columns"] + list(raw_t)
    Z = t[["x:" + c for c in cols]].to_numpy()
    Z = Z - Z.mean(axis=0)
    rel = np.abs(np.diag(np.linalg.qr(Z, mode="r"))) / np.linalg.norm(Z, axis=0)
    assert (rel > 1e-7).all(), f"t{T}: collinear column {np.array(cols)[rel <= 1e-7]}"
    coding["columns"] = cols
    coding["d63"] = dict(arm=arm, columns=spec, info=info, note=__doc__.split("\n\n")[1].replace("\n", " "))
    (o / "coding.json").write_text(json.dumps(coding, indent=1))
    t.to_parquet(o / "train.parquet", index=False)
    r.to_parquet(o / "roll.parquet", index=False)
    man = json.loads((i / "prep_manifest.json").read_text())
    man["loop_options"] = dict(man["loop_options"], d63=arm)
    man["parent"] = src + sfx
    man["coding_sha256"] = hashlib.sha256((o / "coding.json").read_bytes()).hexdigest()
    man["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.iterdir())
                      if p.name != "prep_manifest.json"}
    (o / "prep_manifest.json").write_text(json.dumps(man, indent=1))
    for f in ("train.parquet", "roll.parquet"):
        a, b = pd.read_parquet(i / f), pd.read_parquet(o / f)
        assert a.equals(b[a.columns]), (T, f)
        assert set(b.columns) - set(a.columns) == {"x:" + c for c in raw_t}, (T, f)
    print(f"prep_loop_{arm}{sfx} t{T}: {json.dumps(info)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", nargs="+", default=["lt12", "12_24"])
    ap.add_argument("--arms", nargs="+", default=["k1h", "k2h", "k12h"])
    a = ap.parse_args()
    for arm in a.arms:
        for sfx, towns in TOWNS.items():
            for T in towns:
                build(arm, sfx, T, a.bands)
            src = ARMS[arm][0]
            ss = json.loads((DATA / f"{src}{sfx}" / "split_summary.json").read_text())
            ss["options"] = dict(ss["options"], d63=arm)
            ss["parent"] = src + sfx
            (DATA / f"prep_loop_{arm}{sfx}" / "split_summary.json").write_text(json.dumps(ss, indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
