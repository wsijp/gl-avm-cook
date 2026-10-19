"""Convergence of every effect of a set of struct runs.

Per township-year (struct_run.py's runs file, one model), from each fit's report.json (fit_struct.py's rank-normalised
split R-hat, bulk and folded, and bulk/tail ESS of every component, mcmc_diag.py) plus, computed here from the draws,
the same diagnostics of every valued home's log group value at 1 January (value_v2.py's construction: exp of the
latent value, averaged over the valued members of its valuation group; in chunks of homes). Thresholds, fixed before
fitting: R-hat <= 1.01, bulk and tail ESS >= 400. Also the sum-to-zero result, divergences and E-BFMI.

Usage (repo root): .venv/bin/python -I public/scripts/v2/struct_convergence.py --model S [--runs runs.json]
Writes public/runs/v2/struct_eval/convergence_<model><tag>.csv and .json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcmc_diag as md  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "public/runs/v2/struct_eval"
RHAT_MAX, ESS_MIN = 1.01, 400
COMPONENTS = ["intercept", "beta", "mu", "zeta", "psi_all", "u_valued_units", "location_valued_units",
              "level_plus_location", "scalars", "z", "lp__"]


def home_values(run, prep, chains, m):
    """rank-normalised R-hat and bulk and tail ESS of every valued home's log group value (chunks of 4,000 homes)"""
    D, G = np.load(run / "draws.npz"), np.load(run / "design.npz")
    roll = pd.read_parquet(REPO / prep / f"t{json.loads((run / 'report.json').read_text())['town']}" / "roll.parquet",
                           columns=["meta_pin", "group_id"])
    assert (roll["meta_pin"].to_numpy() == G["pins"].astype(str)).all()
    _, ginv = np.unique(roll["group_id"].to_numpy(), return_inverse=True)
    n = len(ginv)
    M = csr_matrix((np.ones(n), (np.arange(n), ginv)))
    base = (D["intercept"] + D["mu"][:, -1])[:, None]
    lat = base + D["beta"] @ G["X_c"].T + D["zeta"][:, G["nbhd"]] + D["psi_last"][:, G["nbhd"]] + D["u"][:, G["node"]]
    grp = np.log(np.asarray(np.exp(lat) @ M) / np.bincount(ginv)[None, :])        # (draws, groups)
    rh, eb, et = [], [], []
    for c0 in range(0, grp.shape[1], 4000):
        x = grp[:, c0:c0 + 4000].reshape(chains, m, -1)
        rh.append(md.rhat(x))
        eb.append(md.ess_bulk(x))
        et.append(md.ess_tail(x))
    return np.concatenate(rh), np.concatenate(eb), np.concatenate(et)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", default="runs.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    runs = json.loads((EV / a.runs).read_text())
    rows, res = [], {}
    for k, s in sorted(runs.items(), key=lambda x: (x[1]["year"], x[1]["town"])):
        if s["model"] != a.model or s.get("status") != "done":
            continue
        run = REPO / s["run"]
        rep = json.loads((run / "report.json").read_text())
        d = rep["diagnostics"]
        bc = d["by_component"]
        row = dict(year=s["year"], town=s["town"], run=run.name, runtime_min=round(rep["timing"]["runtime_s"] / 60, 1),
                   sum_to_zero=rep["sum_to_zero"]["all_pass"],
                   sum_to_zero_worst=max(c["max_rel"] for c in rep["sum_to_zero"]["checks"].values()),
                   divergent=d["divergent"], ebfmi_min=min(d["ebfmi"]), treedepth_max_share=d["max_treedepth_share"],
                   max_rhat=d["max_rhat"], min_ess_bulk=d["min_ess_bulk"], min_ess_tail=d["min_ess_tail"])
        for c in COMPONENTS:
            if c in bc:
                row[f"{c}_rhat"] = bc[c]["rhat_max"]
                row[f"{c}_ess_bulk"] = bc[c]["ess_bulk_min"]
                row[f"{c}_ess_tail"] = bc[c]["ess_tail_min"]
                row[f"{c}_share_rhat_gt"] = bc[c]["share_rhat_gt_1_01"]
        rh, eb, et = home_values(run, s["prep"], rep["settings"]["chains"], rep["settings"]["samples"])
        row.update(values_rhat=float(rh.max()), values_share_rhat_gt=float((rh > RHAT_MAX).mean()),
                   values_ess_bulk=float(eb.min()), values_ess_tail=float(et.min()), n_groups=int(len(rh)))
        row["all_within_target"] = bool(row["max_rhat"] <= RHAT_MAX and row["min_ess_bulk"] >= ESS_MIN and
                                        row["min_ess_tail"] >= ESS_MIN and row["values_rhat"] <= RHAT_MAX and
                                        row["values_ess_bulk"] >= ESS_MIN and row["values_ess_tail"] >= ESS_MIN and
                                        row["divergent"] == 0 and row["ebfmi_min"] >= 0.3)
        rows.append(row)
        print(f"{s['year']} {s['town']:>2} R-hat {row['max_rhat']:.3f} ESS {row['min_ess_bulk']:.0f}/{row['min_ess_tail']:.0f} "
              f"values {row['values_rhat']:.3f}/{row['values_ess_bulk']:.0f} div {row['divergent']} "
              f"E-BFMI {row['ebfmi_min']:.2f} sum0 {row['sum_to_zero_worst']:.0e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(EV / f"convergence_{a.model}{a.tag}.csv", index=False)
    summ = dict(n=len(df), all_within_target=int(df["all_within_target"].sum()), sum_to_zero_all=bool(df["sum_to_zero"].all()),
                worst={c: float(df[c].max()) for c in df if c.endswith("_rhat")},
                lowest_ess={c: float(df[c].min()) for c in df if c.endswith("_ess_bulk") or c.endswith("_ess_tail")},
                divergent_total=int(df["divergent"].sum()), ebfmi_min=float(df["ebfmi_min"].min()))
    (EV / f"convergence_{a.model}{a.tag}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
