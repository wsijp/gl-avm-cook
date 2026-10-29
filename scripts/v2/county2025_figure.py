"""Sale-prediction test: per-township MdAPE difference, GL - Assessor, with 90% intervals, sorted.

Reads public/runs/v2/county2025_eval/summary.csv (county2025_eval.py). Pooled rows (county, City, suburbs) on top,
then the 38 townships sorted by the difference. City townships blue circles, suburbs orange squares (colour and shape,
so identity is not colour alone); the values are in the results table.

Usage (repo root): .venv/bin/python -I public/scripts/v2/county2025_figure.py
Writes notes/v2/figures/county2025_mdape_difference.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CITY_C, SUB_C = "#2a78d6", "#eb6834"


def main():
    s = pd.read_csv(REPO / "public/runs/v2/county2025_eval/summary.csv", dtype={"unit": str})
    s["code"] = s["unit"].str.split(" ").str[0]
    town = s[s["code"].str.fullmatch(r"\d\d")].copy()
    town["city"] = town["code"].astype(int) >= 70
    town = town.sort_values("d_mdape", ascending=False)
    pooled = s.set_index("unit").loc[["suburbs", "City", "county"]].reset_index()
    rows = [(f"{r.unit[3:]} ({r.n:,})", r, CITY_C if r.city else SUB_C, "o" if r.city else "s") for r in town.itertuples()]
    rows += [(None, None, None, None)]
    rows += [(f"{'County' if r.unit == 'county' else r.unit}, pooled ({r.n:,})", r,
              CITY_C if r.unit == "City" else SUB_C if r.unit == "suburbs" else INK, "D") for r in pooled.itertuples()]
    fig, ax = plt.subplots(figsize=(7.2, 10.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, (lab, r, c, mk) in enumerate(rows):
        labels.append(lab or "")
        if r is None:
            continue
        ax.plot([r.d_mdape_lo, r.d_mdape_hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r.d_mdape], [y], s=46 if mk == "D" else 34, marker=mk, color=c, edgecolor=SURFACE, linewidth=2,
                   zorder=3)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8, color=INK2)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("MdAPE difference, hedonic model minus Assessor's model, percentage points\nbelow 0: hedonic model closer to the price",
                  fontsize=8, color=INK2)
    ax.set_title("Hedonic model (published) minus Assessor's model\ntest sales Oct 2023 – Dec 2024, with 90% intervals",
                 fontsize=9, color=INK, loc="left")
    h = [plt.Line2D([], [], color=CITY_C, marker="o", lw=2, label="City of Chicago township"),
         plt.Line2D([], [], color=SUB_C, marker="s", lw=2, label="suburban township"),
         plt.Line2D([], [], color=INK, marker="D", lw=2, label="county, pooled")]
    ax.legend(handles=h, loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    out = REPO / "notes/v2/figures/county2025_mdape_difference.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
