"""The README's map for the final sale-prediction model: the value of one standard house in every census block across
Cook County in October 2023 (the last training month), from the final model's 38 house fits county2025_IXah_t<T>
(gl_loop.stan with the forecasting time model and the neighbourhood-varying resale terms).

Value of a block unit = posterior median, over the 4,000 draws, of
  intercept + mu[October 2023] + zeta[k] + psi[k, October 2023] + u[s] + (x_std - training mean)' beta,
the same quantity as the published model's map (fig_location_map.py), for a house that has not sold before (so the
resale premium, the flip discount and their neighbourhood deviations are zero). The block effects u and the
neighbourhood paths at the last month are not kept in draws.npz: they are rebuilt here from the saved chains
(chains/*.csv: z, sigma_u, psi_raw, sigma_psi, intercept) with the fit's own whiteners (loop_fit.make_data). Check: the
same rebuild reproduces every test sale's value held at October 2023 (draws.npz latent_apr) to float32 precision.

x_std: the median of each numeric house characteristic and the most common level of each categorical one over the house
training sales of all 38 townships (decoded from each township's coded regressors), coded with each township's own
coding; no earlier sale; no flip purchase; the age interactions computed from the standard house's age, class and
building area as in d70_prep.py. The map itself is drawn by fig_location_map.py's code (colour scale +-1 log point).

Writes the block-unit values to public/runs/v2/county2025_ixa_eval/map_units/t<T>.parquet (reused if present) and the
map to public/figures/location_map.png. Usage (from the working folder):
.venv/bin/python -I public/scripts/v2/final_map.py [--out <png>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "public/scripts/v2"))
sys.argv.append("--county")                      # loop_fit.py reads the county design (94 months) from the command line
import loop_fit as lfit  # noqa: E402
import fig_location_map as flm  # noqa: E402
sys.argv.remove("--county")

RUNS = REPO / "public/runs/v2"
PREP = "public/data/v2/county_d77/prep_loop_ixah"
CACHE = RUNS / "county2025_ixa_eval/map_units"
TOWNS = [str(x) for x in list(range(10, 40)) + list(range(70, 78))]
SEG = "h"


def decode(t, coding):
    """house characteristics back from one township's coded regressors (raw numerics, categorical levels)"""
    seg = coding["segments"]["house"]
    x = pd.DataFrame(index=t.index)
    for c, p in seg["numeric"].items():
        if c.startswith("hist_"):
            continue
        if f"x:{SEG}:{c}" in t:
            x[c] = t[f"x:{SEG}:{c}"] * p["sd"] + p["mean"]
    for c, p in seg["categorical"].items():
        s = pd.Series(p["reference"], index=t.index)
        for lev in p["levels"]:
            if f"x:{SEG}:{c}={lev}" in t:
                s[t[f"x:{SEG}:{c}={lev}"] == 1] = lev
        x[c] = s
    return x


def standard_home(runs):
    xs, cats = [], set()
    for r in runs:
        prep = REPO / r["prep"] / f"t{r['town']}"
        coding = json.loads((prep / "coding.json").read_text())
        xs.append(decode(pd.read_parquet(prep / "train.parquet"), coding))
        cats |= set(coding["segments"]["house"]["categorical"])
    x = pd.concat(xs)
    num = [c for c in x.columns if c not in cats]
    return ({c: float(x[c].median()) for c in num},
            {c: x.loc[x[c] != "other", c].value_counts().index[0] for c in sorted(cats)})


def encode(home, coding, xcols):
    """the standard house in one township's coded regressors"""
    num, cat = home
    seg = coding["segments"]["house"]
    x = {c: 0.0 for c in xcols}
    raw = {}
    for c, p in seg["numeric"].items():
        v = 0.0 if c.startswith("hist_") else num.get(c, p["mean"])   # no earlier sale; a column dropped everywhere: unused
        raw[c] = v
        x[f"x:{SEG}:{c}"] = (v - p["mean"]) / p["sd"] if p["sd"] > 0 else 0.0
    if f"x:{SEG}:char_recent_renovation" in x:
        x[f"x:{SEG}:char_recent_renovation"] = num.get("char_recent_renovation", 0.0)
    levels = {}
    for c, p in seg["categorical"].items():
        lev = "other" if cat[c] in p["merged_into_other"] else cat[c]
        if lev not in p["levels"]:
            lev = "other" if "other" in p["levels"] else p["reference"]
        levels[c] = lev
        for lv in p["levels"]:
            if lv != p["reference"] and f"x:{SEG}:{c}={lv}" in x:
                x[f"x:{SEG}:{c}={lv}"] = float(lev == lv)
    for c, p in coding["d63"]["columns"].items():             # flip-purchase flags: 0
        x[f"x:{c}"] = (0.0 - p["mean"]) / p["sd"]
    ac = raw["age"] - seg["numeric"]["age"]["mean"]           # age interactions (d70_prep.py)
    s = raw["log_bldg_sf"] - seg["numeric"]["log_bldg_sf"]["mean"]
    for c, p in coding["d70"]["columns"].items():
        k = c.split("ix_age_x_")[1]
        v = ac * s if k == "logsf" else ac * float(levels["char_class"] == k)
        x[f"x:{c}"] = (v - p["mean"]) / p["sd"]
    return np.array([x[c] for c in xcols])


def chains(run):
    files = sorted((run / "chains").glob("gl_loop-*_[0-9].csv"))
    assert len(files) == 4, files
    head = pd.read_csv(files[0], comment="#", nrows=0).columns
    keep = [c for c in head if c.split(".")[0] in ("intercept", "sigma_u", "sigma_psi") or c.startswith(("z.", "psi_raw."))]
    v = pd.concat([pd.read_csv(f, comment="#", usecols=keep) for f in files], ignore_index=True)
    return v.rename(columns={"intercept.1": "intercept", "sigma_u.1": "sigma_u", "sigma_psi.1": "sigma_psi"})   # length-1 vectors


def town_units(r, home):
    T = r["town"]
    f = CACHE / f"t{T}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    prep = REPO / r["prep"] / f"t{T}"
    run = RUNS / f"county2025_IXah_t{T}"
    M = lfit.make_data(prep, False, forecast=True, np_=True)
    st, data = M["st"], M["data"]
    d = np.load(run / "draws.npz")
    assert list(d["xcols"]) == list(M["xcols"])
    v = chains(run)
    nd = len(v)
    z = v[[c for c in v.columns if c.startswith("z.")]].to_numpy()          # z.1 .. z.n, in order
    Kl, Tl = st["A_comp"].shape[1], st["A_t"].shape[1]
    psi_raw = v[[f"psi_raw.{i + 1}.{j + 1}" for j in range(Tl) for i in range(Kl)]].to_numpy().reshape(nd, Tl, Kl)
    psi_last = np.einsum("kl,dml,m->dk", st["A_comp"], psi_raw, st["A_t"][-1]) * v["sigma_psi"].to_numpy()[:, None]
    u = np.asarray((st["A_u"] @ (z * v["sigma_u"].to_numpy()[:, None]).T).T)
    base = v["intercept"].to_numpy() + d["mu"][:, -1]
    # check: the rebuild reproduces the saved value of every test sale held at October 2023
    rr = M["r"]
    node_r = rr["node"].to_numpy().astype(int)
    kr = rr["nbhd_id"].map(st["kid"]).to_numpy()
    Xr = rr[M["xcols"]].to_numpy() - M["xmean"]
    nvc = d["nvc"].astype(float)
    lat_nvc = np.einsum("dij,ij->di", nvc[:, kr, :], lfit.np_loadings(rr, flags=False))
    lat = base[:, None] + d["beta"] @ Xr.T + d["zeta"][:, kr] + psi_last[:, kr] + u[:, node_r] + lat_nvc
    err = float(np.abs(lat - d["latent_apr"]).max())
    assert err < 1e-3, (T, err)
    units = M["units"].copy()
    assert (units["node"].to_numpy() == np.arange(len(units))).all() and u.shape[1] == len(units)
    x_std = encode(home, M["coding"], M["xcols"]) - M["xmean"]
    k = units["nbhd_id"].map(st["kid"]).to_numpy()
    val = (base + d["beta"] @ x_std)[:, None] + d["zeta"][:, k] + psi_last[:, k] + u
    units["value"] = np.median(val, axis=0)
    units["check_max_abs_error"] = err
    CACHE.mkdir(parents=True, exist_ok=True)
    units.to_parquet(f, index=False)
    print(f"t{T}: {len(units):,} units, {nd} draws; rebuild check max |error| {err:.1e}", flush=True)
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public/figures/location_map.png")
    a = ap.parse_args()
    runs = [dict(town=T, prep=PREP, year="2023", run=f"public/runs/v2/county2025_IXah_t{T}") for T in TOWNS]
    flm.published_runs = lambda which: runs
    flm.standard_home = standard_home
    flm.unit_values = town_units
    sys.argv = [sys.argv[0], "--set", "county2025", "--out", a.out]
    flm.main()


if __name__ == "__main__":
    main()
