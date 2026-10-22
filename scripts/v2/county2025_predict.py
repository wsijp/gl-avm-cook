"""Sale-prediction test: predict a fitted township's test sales.

Per posterior draw d and test sale i (fit_struct.py's draws.npz and design.npz; the design rows are the prep's
roll.parquet = the township's test sales):
  latent_di = intercept_d + x_i' beta_d + mu_d[last] + zeta_d[k_i] + psi_d[k_i, last] + u_d[s_i]
with "last" the last training month (October 2023) for every test sale (held flat after it).
Prediction = posterior median of exp(latent); also the 5% and 95% quantiles and the split
R-hat of each sale's latent log value (as value_v2.py). No noise draw, no group averaging, no rounding.

Usage (repo root): .venv/bin/python -I public/scripts/v2/county2025_predict.py public/runs/v2/county2025_P_t72 \
                   [--prep public/data/v2/prep_county2025]
Writes <run>/predictions.parquet and adds a `prediction` block to <run>/report.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NT = 94


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--prep", default="public/data/v2/prep_county2025")
    a = ap.parse_args()
    run = Path(a.run)
    rep = json.loads((run / "report.json").read_text())
    D = np.load(run / "draws.npz")
    G = np.load(run / "design.npz")
    e = pd.read_parquet(Path(a.prep) / f"t{rep['town']}" / "roll.parquet",
                        columns=["meta_pin", "meta_sale_document_num", "meta_card_num", "date", "meta_sale_price",
                                 "meta_township_code", "nbhd_id", "has_training_sale", "meta_tieback_proration_rate"])
    assert (e["meta_pin"].to_numpy() == G["pins"].astype(str)).all()
    assert D["mu"].shape[1] == NT, f"expected {NT} months, draws have {D['mu'].shape[1]}"
    base = D["intercept"][:, None] + D["mu"][:, -1][:, None]
    lat = base + D["beta"] @ G["X_c"].T + D["zeta"][:, G["nbhd"]] + D["psi_last"][:, G["nbhd"]] + D["u"][:, G["node"]]
    c, m = rep["settings"]["chains"], rep["settings"]["samples"]
    x = lat.reshape(c, m, -1)
    h = m // 2
    x = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
    W = x.var(axis=1, ddof=1).mean(axis=0)
    Bn = x.mean(axis=1).var(axis=0, ddof=1)
    rhat = np.sqrt(((h - 1) / h * W + Bn) / W)
    pred = np.exp(lat)
    e["pred"] = np.median(pred, axis=0)
    e["pred_q05"], e["pred_q95"] = np.quantile(pred, [0.05, 0.95], axis=0)
    e["latent_rhat"] = rhat
    e["town"] = rep["town"]
    e.to_parquet(run / "predictions.parquet", index=False)
    r = e["pred"] / e["meta_sale_price"]
    rep["prediction"] = dict(n_test=len(e), n_draws=int(pred.shape[0]), carried_month="2023-10 (flat for every test sale)",
                             mdape=float((r - 1).abs().median()), median_ratio=float(r.median()),
                             latent_split_rhat=dict(max=float(rhat.max()), p99=float(np.quantile(rhat, 0.99)),
                                                    median=float(np.median(rhat)),
                                                    share_above_1_01=float((rhat > 1.01).mean())))
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["prediction"], indent=1))


if __name__ == "__main__":
    main()
