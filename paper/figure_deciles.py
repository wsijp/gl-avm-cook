"""The paper's Figure 2: median ratio (value / sale price) by price group, both models, for the three comparisons.

Groups are the VEI's own: sales ranked by the average of sale price and the model's value (value rescaled by the
overall median ratio), split into ten equal groups (IAAO 2026 draft, Appendix E; vei.py). Each model is grouped by its
own proxy, as in its VEI. Reads stored outputs only; writes paper/ratio_by_decile.pdf and paper/figure_deciles.json.

Usage, from the working folder that holds this repository as public/:
    .venv/bin/python -I public/paper/figure_deciles.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vei import vei  # noqa: E402

REPO = HERE.parents[1]   # the working folder (this file is public/paper/figure_deciles.py)
CH = REPO / "public/runs/v2/struct_eval/chosen"
FIG = HERE / "ratio_by_decile"


def jan(year):
    head = json.loads((CH / f"ratio_{year}_primary.json").read_text())
    ev = pd.concat([pd.read_parquet(REPO / p) for p in head["evals"]], ignore_index=True)
    gl = pd.read_parquet(CH / f"values_{year}.parquet", columns=["meta_pin", "value"]).set_index("meta_pin")["value"]
    ass = pd.read_parquet(CH / f"d72/values_{year}_assfin.parquet").set_index("meta_pin")["value"]
    return ev["meta_pin"].map(gl).to_numpy(float), ev["meta_pin"].map(ass).to_numpy(float), ev["adj_price"].to_numpy(float)


def main():
    t = pd.read_parquet(REPO / "public/runs/v2/county2025_ixa_eval/test_sales.parquet")
    tests = {"1 January 2023": jan("2023"), "1 January 2022": jan("2022"),
             "County sale prediction": (t["pred_gl"].to_numpy(float), t["pred_assessor"].to_numpy(float),
                                        t["meta_sale_price"].to_numpy(float))}
    out = {}
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (name, (g, a, p)) in zip(axes, tests.items()):
        rg, ra = vei(g, p), vei(a, p)
        mg = [x["median"] for x in rg["groups"]]; ma = [x["median"] for x in ra["groups"]]
        out[name] = dict(hedonic=mg, assessor=ma, vei_hedonic=rg["vei"], vei_assessor=ra["vei"])
        x = np.arange(1, 11)
        ax.axhline(1.0, color="0.6", lw=0.8)
        ax.plot(x, mg, "o-", color="black", ms=3.5, lw=1.2, label="Hedonic model")
        ax.plot(x, ma, "s--", color="0.45", ms=3.5, lw=1.2, label="Assessor's model")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([1, 5, 10]); ax.set_xticklabels(["1\nlowest", "5", "10\nhighest"])
        ax.set_xlabel("price group (tenths)")
        ax.grid(axis="y", color="0.9", lw=0.6)
    axes[0].set_ylabel("median ratio, value / price")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG}.pdf")
    (HERE / "figure_deciles.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, "VEI", round(v["vei_hedonic"], 1), round(v["vei_assessor"], 1),
              "| first/last hedonic", round(v["hedonic"][0], 3), round(v["hedonic"][-1], 3),
              "| assessor", round(v["assessor"][0], 3), round(v["assessor"][-1], 3))
    print("written", FIG)


if __name__ == "__main__":
    main()
