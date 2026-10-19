# 2022 version of scripts/v2/value_v2.py: the same code with the 2022 assessment year's dates, file names and
# north-suburb labels, and the differences listed in REPRODUCE.md ("What to expect").
"""Value every valued PIN of a fitted region at 1 January 2022.

Per posterior draw d and valued PIN i (the draws.npz and design.npz written by fit_v2.py or fit_struct.py):
  latent_di = intercept_d + x_i' beta_d + mu_d[Dec 2021] + zeta_d[k_i] + psi_d[k_i, Dec 2021] + u_d[s_i]
The valuation uses the last training month (December 2021) carried flat to 1 January 2022.
  raw_di   = exp(latent_di)                                  raw value
  group_di = mean of raw_dj over the valued members j of i's valuation group (singletons: raw_di)
Point values = posterior medians: `raw`, `value` (group-equalised; the value used in the ratio study),
`value_rounded` (the Assessor's rule: floor to 1 / 500 / 5,000 / 10,000 with breaks 1k / 10k / 100k,
params.yaml:370-372), and the 5% and 95% quantiles of the group value.

Usage (repo root): .venv/bin/python -I public/scripts/v2/ay2022/value_v2.py public/runs/v2/<run> --prep public/data/v2/prep_p8
Writes <run>/values.parquet and adds a `valuation` block to <run>/report.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

BREAKS, STEPS = [1000, 10000, 100000], [1, 500, 5000, 10000]


def assessor_round(v):
    v = np.asarray(v, float)
    step = np.select([v < BREAKS[0], v < BREAKS[1], v < BREAKS[2]], STEPS[:3], STEPS[3])
    return np.floor(v / step) * step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--prep", default="public/data/v2/prep")
    a = ap.parse_args()
    run = Path(a.run)
    rep = json.loads((run / "report.json").read_text())
    D = np.load(run / "draws.npz")
    G = np.load(run / "design.npz")
    roll = pd.read_parquet(Path(a.prep) / f"t{rep['town']}" / "roll.parquet", columns=["meta_pin", "group_id", "meta_class"])
    pins = G["pins"].astype(str)
    assert (roll["meta_pin"].to_numpy() == pins).all()

    base = D["intercept"][:, None] + D["mu"][:, -1][:, None]                       # (draws, 1)
    lat = base + D["beta"] @ G["X_c"].T + D["zeta"][:, G["nbhd"]] + D["psi_last"][:, G["nbhd"]] + D["u"][:, G["node"]]
    rhat = None
    if rep["method"] == "nuts":
        # Split R-hat (Gelman et al. 2013, classic form) of each PIN's latent log value: the components (zeta, u)
        # trade off along a ridge, so the sum that values the PIN is the quantity whose convergence matters.
        c, m = rep["settings"]["chains"], rep["settings"]["samples"] // rep["settings"].get("thin", 1)
        x = lat.reshape(c, m, -1)
        h = m // 2
        x = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)                     # (2c, h, PINs)
        W = x.var(axis=1, ddof=1).mean(axis=0)
        Bn = x.mean(axis=1).var(axis=0, ddof=1)
        rhat = np.sqrt(((h - 1) / h * W + Bn) / W)
    raw = np.exp(lat)                                                               # (draws, PINs)
    gid, ginv = np.unique(roll["group_id"].to_numpy(), return_inverse=True)
    M = csr_matrix((np.ones(len(pins)), (np.arange(len(pins)), ginv)))             # PIN -> group indicator
    grp = (np.asarray(raw @ M) / np.bincount(ginv)[None, :])[:, ginv]
    out = pd.DataFrame(dict(meta_pin=pins, group_id=roll["group_id"], meta_class=roll["meta_class"],
                            raw=np.median(raw, axis=0), value=np.median(grp, axis=0),
                            value_q05=np.quantile(grp, 0.05, axis=0), value_q95=np.quantile(grp, 0.95, axis=0)))
    out["value_rounded"] = assessor_round(out["value"])
    out.to_parquet(run / "values.parquet", index=False)
    multi = out[out.groupby("group_id")["meta_pin"].transform("size") > 1]
    spread = multi.groupby("group_id")["raw"].agg(lambda s: s.max() / s.min() - 1)
    rep["valuation"] = dict(n_pins=len(out), n_draws=int(raw.shape[0]), n_groups_ge2=int(multi["group_id"].nunique()),
                            max_group_values_distinct=int(multi.groupby("group_id")["value"].nunique().max()) if len(multi) else 0,
                            raw_spread_within_groups_median=float(spread.median()) if len(spread) else None,
                            value_median=float(out["value"].median()), carried_month="2021-12 (flat to 2022-01-01)")
    if rhat is not None:
        rep["valuation"]["latent_split_rhat"] = dict(max=float(rhat.max()), p99=float(np.quantile(rhat, 0.99)),
                                                     median=float(np.median(rhat)), share_above_1_01=float((rhat > 1.01).mean()))
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["valuation"], indent=1))


if __name__ == "__main__":
    main()
