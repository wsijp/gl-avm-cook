"""Equity: median ratio by sale-price decile, GL, rebuilt Assessor and published Assessor, both years.

Reads public/runs/v2/equity_eval/deciles_<year>.csv (equity_eval.py); writes equity_deciles.png to a working figures
folder outside the release.
Usage (repo root): .venv/bin/python -I public/scripts/v2/equity_figure.py [--structure]
--structure: the published model P; reads public/runs/v2/struct_eval/chosen/equity/, writes
structure_equity_deciles.png to the same folder.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SERIES = [("gl", "GL (our model)", "#2a78d6", "o"), ("assessor", "Assessor, our rebuild", "#eb6834", "s"),
          ("published", "Assessor, published", "#1baf7a", "^")]
TITLES = {"2023": "2023, south and west suburbs (11,356 sales)", "2022": "2022, north suburbs (10,825 sales)"}


def main():
    src, out = "public/runs/v2/equity_eval", REPO / "notes/v2/figures/equity_deciles.png"
    if "--structure" in sys.argv:
        src, out = "public/runs/v2/struct_eval/chosen/equity", REPO / "notes/v2/figures/structure_equity_deciles.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, year in zip(axes, ["2023", "2022"]):
        d = pd.read_csv(REPO / src / f"deciles_{year}.csv")
        x = d["decile"].to_numpy()
        ax.axhspan(0.90, 1.10, color="#e9e8e4", zorder=0, label="IAAO level range 0.90–1.10")
        ax.axhline(1.0, color="#6b6a64", lw=1, zorder=1)
        for i, (k, lab, col, mk) in enumerate(SERIES):
            off = (i - 1) * 0.12
            ax.errorbar(x + off, d[f"{k}_median_ratio"],
                        yerr=[d[f"{k}_median_ratio"] - d[f"{k}_lo90"], d[f"{k}_hi90"] - d[f"{k}_median_ratio"]],
                        color=col, marker=mk, ms=6, lw=2, elinewidth=1, capsize=0, label=lab, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}\n${p / 1000:,.0f}k" for k, p in zip(x, d["price_median"])], fontsize=8)
        ax.set_xlabel("sale-price decile (median time-adjusted price)", fontsize=9)
        ax.set_title(TITLES[year], fontsize=10)
        ax.grid(axis="y", color="#e9e8e4", lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("median ratio (value ÷ time-adjusted price)", fontsize=9)
    axes[0].set_ylim(0.80, 1.60)
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Cheaper homes are valued relatively higher by all three: median ratio by sale-price decile "
                 "(90% intervals)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
