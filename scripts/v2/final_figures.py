"""The README's two figures for the final sale-prediction model.

Price index (public/figures/final_price_index.png): each township's monthly house price path, fitted
over the training months (January 2016 to October 2023) and forecast by the model for the test months (November 2023 to
December 2024).

The path is the township term of gl_loop.stan's forecasting time model (mu_fit over the training months, mu_fc beyond:
the damped trend plus the calendar-month effect), from the 38 house fits county2025_IXah_t<T> (draws.npz: mu, mu_fc).
Shown as the posterior median of exp(path - path in January 2016) x 100, so every township starts at 100; neighbourhood
paths and the other terms are not included. Left panel the eight City of Chicago townships, right panel the 30 suburban
townships; the shaded band is the 90% interval of the forecast for the median township path of each panel's townships
(the pointwise median over townships, draw by draw).

By township (public/figures/final_by_township.png): the difference in MdAPE, final GL minus the Assessor's model, on
the 38,177 test sales, with its 90% interval, for each township and pooled (public/runs/v2/county2025_ixa_eval/
summary.csv, written by d77_county.py eval); the pooled rows mark the usefulness bar's limit fixed before the test
(upper 90% bound below +0.3 county-wide and in the suburbs, below +0.75 in the City). The same figure as d77_county.py's,
with plain labels.

Usage (from the working folder): .venv/bin/python -I public/scripts/v2/final_figures.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "public/runs/v2"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CITY_C, SUB_C = "#2a78d6", "#eb6834"
TOWNS = [str(x) for x in list(range(10, 40)) + list(range(70, 78))]
NAMES = {"70": "Hyde Park", "71": "Jefferson", "72": "Lake", "73": "Lake View", "74": "North Chicago",
         "75": "Rogers Park", "76": "South Chicago", "77": "West Chicago"}
MONTHS = pd.period_range("2016-01", "2024-12", freq="M")       # 94 training months, then 14 forecast months
NT = 94
BAR = {"county": 0.3, "City": 0.75, "suburbs": 0.3}
TOWN_NAMES = {**NAMES, "10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet",
              "15": "Cicero", "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont", "20": "Leyden",
              "21": "Lyons", "22": "Maine", "23": "New Trier", "24": "Niles", "25": "Northfield", "26": "Norwood Park",
              "27": "Oak Park", "28": "Orland", "29": "Palatine", "30": "Palos", "31": "Proviso", "32": "Rich",
              "33": "River Forest", "34": "Riverside", "35": "Schaumburg", "36": "Stickney", "37": "Thornton",
              "38": "Wheeling", "39": "Worth"}


def paths():
    out = {}
    for T in TOWNS:
        d = np.load(RUNS / f"county2025_IXah_t{T}" / "draws.npz")
        p = np.concatenate([d["mu"], d["mu_fc"]], axis=1)
        assert p.shape[1] == len(MONTHS)
        out[T] = 100 * np.exp(p - p[:, :1])                     # draws x months, January 2016 = 100
    return out


def by_township(out):
    s = pd.read_csv(RUNS / "county2025_ixa_eval/summary.csv", dtype={"unit": str})
    s["code"] = s["unit"].str.split(" ").str[0]
    town = s[s["code"].str.fullmatch(r"\d\d")].copy()
    town["city"] = town["code"].astype(int) >= 70
    town = town.sort_values("d_mdape", ascending=False)
    pooled = s.set_index("unit").loc[["suburbs", "City", "county"]].reset_index()
    rows = [(f"{TOWN_NAMES[r.code]} ({r.n:,})", r, CITY_C if r.city else SUB_C, "o" if r.city else "s")
            for r in town.itertuples()]
    rows += [(None, None, None, None)]
    rows += [(f"{dict(county='County', City='City of Chicago', suburbs='Suburbs')[r.unit]}, pooled ({r.n:,})", r,
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
        if mk == "D":                                           # the usefulness bar's limit for this pooled row
            ax.plot([BAR[r.unit]] * 2, [y - 0.35, y + 0.35], color=INK2, lw=1.2, zorder=2)
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
    ax.set_xlabel("MdAPE difference, GL minus Assessor (percentage points; below 0 = GL closer to the price)",
                  fontsize=8, color=INK2)
    ax.set_title("GL minus Assessor's model, test sales Oct 2023 – Dec 2024, with 90% intervals",
                 fontsize=8.5, color=INK, loc="left")
    h = [plt.Line2D([], [], color=CITY_C, marker="o", lw=2, label="City of Chicago township"),
         plt.Line2D([], [], color=SUB_C, marker="s", lw=2, label="suburban township"),
         plt.Line2D([], [], color=INK, marker="D", lw=2, label="county, pooled"),
         plt.Line2D([], [], color=INK2, marker="|", lw=0, markersize=10, label="usefulness bar's limit (pooled rows)")]
    ax.legend(handles=h, loc="center right", fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "public/figures/final_price_index.png")
    ap.add_argument("--out-towns", type=Path, default=REPO / "public/figures/final_by_township.png")
    a = ap.parse_args()
    by_township(a.out_towns)
    P = paths()
    x = MONTHS.to_timestamp(how="start")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, facecolor=SURFACE)
    for ax, city in zip(axes, (True, False)):
        ax.set_facecolor(SURFACE)
        towns = [T for T in TOWNS if (int(T) >= 70) == city]
        c = CITY_C if city else SUB_C
        for T in towns:
            m = np.median(P[T], axis=0)
            ax.plot(x[:NT], m[:NT], color=c, lw=1.0 if city else 0.7, alpha=0.9 if city else 0.55, zorder=2)
            ax.plot(x[NT - 1:], m[NT - 1:], color=c, lw=1.0 if city else 0.7, alpha=0.9 if city else 0.55,
                    ls=(0, (2, 1.5)), zorder=2)
        if city:                                                # names at the right end, spread so they do not overlap
            ends = sorted((float(np.median(P[T][:, -1])), T) for T in towns)
            ys = []
            for y, T in ends:
                ys.append(max(y, ys[-1] + 7.5) if ys else y)
            for (y0, T), y in zip(ends, ys):
                ax.annotate(NAMES[T], (x[-1], y0), xytext=(x[-1] + pd.Timedelta(days=60), y), fontsize=6.5,
                            color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
        mid = np.median(np.stack([P[T] for T in towns]), axis=0)  # draws x months: the median township, draw by draw
        lo, med, hi = np.quantile(mid, [0.05, 0.5, 0.95], axis=0)
        ax.fill_between(x[NT - 1:], lo[NT - 1:], hi[NT - 1:], color=c, alpha=0.15, lw=0, zorder=1)
        ax.plot(x[:NT], med[:NT], color=INK, lw=1.8, zorder=3)
        ax.plot(x[NT - 1:], med[NT - 1:], color=INK, lw=1.8, ls=(0, (2, 1.5)), zorder=3)
        ax.axvline(x[NT - 1], color=INK2, lw=0.8, zorder=1)
        ax.text(x[NT - 1], 0.02, "last training month\n(October 2023) ", transform=ax.get_xaxis_transform(),
                fontsize=7, color=INK2, va="bottom", ha="right")
        ax.set_title(f"{'City of Chicago: 8 townships' if city else 'Suburbs: 30 townships'}", fontsize=9,
                     color=INK, loc="left")
        ax.set_xlim(x[0] - pd.Timedelta(days=90), x[-1] + pd.Timedelta(days=640))   # room for the City's names
        ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2016, 2026, 2)], [str(y) for y in range(2016, 2026, 2)])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    axes[0].set_ylabel("house price index, January 2016 = 100", fontsize=8, color=INK2)
    h = [plt.Line2D([], [], color=MUTED, lw=1, label="one township, fitted"),
         plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (2, 1.5)), label="one township, forecast"),
         plt.Line2D([], [], color=INK, lw=1.8, label="median township"),
         plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.3, lw=0, label="90% interval of its forecast")]
    axes[1].legend(handles=h, loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    for city in (True, False):
        towns = [T for T in TOWNS if (int(T) >= 70) == city]
        mid = np.median(np.median(np.stack([P[T] for T in towns]), axis=0), axis=0)
        print(f"{'City' if city else 'suburbs'}: median township Oct 2023 {mid[NT - 1]:.1f}, Dec 2024 {mid[-1]:.1f} "
              f"(forecast change {100 * (mid[-1] / mid[NT - 1] - 1):+.1f}%)")
    print(a.out)


if __name__ == "__main__":
    main()
