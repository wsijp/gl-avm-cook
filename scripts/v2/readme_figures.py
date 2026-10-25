"""The figures of the public README, in plain labels (public/figures/).

  accuracy_level.png   each model's MdAPE as is, corrected in advance and level-free, and the GL - Assessor difference
                       (from public/runs/v2/d38_eval/summary.json; the figure of d38_report.py)
  by_township.png      paired MdAPE difference GL - Assessor by township, both years (from the stored ratio studies
                       public/runs/v2/p7_eval/ratio_all_primary.json and p8_eval/ratio_all_primary.json)

Usage (repo root): .venv/bin/python -I public/scripts/v2/readme_figures.py [--structure]
--structure: from the published model P (public/runs/v2/struct_eval/chosen/); the README uses these. The three
1 January figures compare GL with the Assessor's model output before rounding (`pred_pin_final_fmv`, as the README
table), from d72_unrounded.py's ratio studies (chosen/d72/ratio_<year>_primary.json, models gl, assfin, gladv,
assfinadv):
  accuracy_level.png   as is (gl, assfin), corrected in advance with time-adjusted October-December prices (gladv,
                       assfinadv) and level-free (median-centred, as p7b_centred.py, computed here on the same
                       resamples, rebuilt by p7b_centred.rebuild and checked against the study's saved MdAPE draws)
  by_township.png      the stored pair gl-assfin per township
  price_deciles.png    median ratio by decile of the time-adjusted price, as equity_eval.py (same cut points, same
                       resamples): GL, the Assessor's model before rounding, and the Assessor's published values,
                       which exist only rounded (shown for reference); GL and published reproduce the stored
                       chosen/equity/deciles_<year>.csv (to 1e-12, the CSV round trip)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "public/scripts/v2"))
from p7b_centred import centred, rebuild, table  # noqa: E402
FIG = REPO / "public/figures"
BLUE, ORANGE, GREEN, INK, INK2, MUTED, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#b9b8b2", "#fcfcfb"
NAMES = {"10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet", "15": "Cicero",
         "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont", "20": "Leyden", "21": "Lyons",
         "22": "Maine", "23": "New Trier", "24": "Niles", "25": "Northfield", "26": "Norwood Park",
         "27": "Oak Park", "28": "Orland", "29": "Palatine", "30": "Palos", "31": "Proviso", "32": "Rich",
         "33": "River Forest", "34": "Riverside", "35": "Schaumburg", "36": "Stickney", "37": "Thornton",
         "38": "Wheeling", "39": "Worth"}
STRUCT = "--structure" in sys.argv
DAGGER = {"12", "37"}  # our rebuild of the Assessor's 2023 values differs most from the published values here
D72 = REPO / "public/runs/v2/struct_eval/chosen/d72"
EQ = REPO / "public/runs/v2/struct_eval/chosen/equity"
YEARS = [("2023", "2023, south and west suburbs"), ("2022", "2022, north suburbs")]
ASS = "Assessor (our rebuild, before rounding)"


def study(y):
    """d72's unrounded ratio study for year y, its evaluation table, values and resamples (checked against its draws)."""
    cfg = json.loads((D72 / f"ratio_{y}_primary.json").read_text())
    ev, P, V, boots = rebuild(cfg)
    saved = np.load(D72 / f"ratio_{y}_primary.boot.npz")
    for n in ["gl", "assfin"]:
        mine = np.array([100 * np.median(np.abs(V[n][b] - P[b]) / P[b]) for b in boots])
        assert np.array_equal(mine, saved[f"{n}:mdape"]), (y, n)
    return cfg, ev, P, V, boots


def by_township():
    studies = [("2023, south and west suburbs", "public/runs/v2/p7_eval/ratio_all_primary.json"),
               ("2022, north suburbs", "public/runs/v2/p8_eval/ratio_all_primary.json")]
    pair = "gl-assessor"
    if STRUCT:
        studies = [(t, f"public/runs/v2/struct_eval/chosen/d72/ratio_{y}_primary.json") for y, t in YEARS]
        pair = "gl-assfin"
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw=dict(height_ratios=[1]))
    for a, (title, path) in zip(axes, studies):
        r = json.loads((REPO / path).read_text())
        rows = [(t, v["n"], v["pairs"][pair]["mdape"]) for t, v in r["by_town"].items()]
        rows.sort(key=lambda x: -x[2]["point"])
        o = r["overall"]
        rows.append(("all", o["n"], o["pairs"][pair]["mdape"]))
        for i, (t, n, d) in enumerate(rows):
            y = i + (1 if t == "all" else 0)
            col = INK if t == "all" else BLUE
            a.plot([d["lo90"], d["hi90"]], [y, y], color=col, lw=2, solid_capstyle="round")
            a.plot(d["point"], y, "o", color=col, ms=7 if t == "all" else 6, mec=SURFACE, mew=1.5)
        labels = [f"{NAMES.get(t, 'all townships')}{' †' if t in DAGGER else ''} ({n:,})" for t, n, _ in rows]
        a.set_yticks([i + (1 if t == "all" else 0) for i, (t, _, _) in enumerate(rows)], labels)
        a.axvline(0, color=MUTED, lw=1)
        a.set_xlim(-5, 10.5 if STRUCT else 9)
        a.set_title(title, fontsize=10, loc="left")
        a.set_xlabel("GL − Assessor (our rebuild), MdAPE points (90% interval)\nbelow zero: GL more accurate")
        a.grid(axis="x", color="#e7e6e1", lw=0.8)
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(axis="y", length=0)
    fig.text(0.01, 0.005, ("Both models' values before rounding. " if STRUCT else "") + "Number of sales in brackets. † Our rebuild of the Assessor's model differs most from its "
             "published values in Bloom and Thornton.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "by_township.png", dpi=150)


def accuracy_level(data):
    """MdAPE as is, corrected in advance and level-free, GL against the Assessor's model before rounding."""
    versions = [("as is", "gl", "assfin"), ("corrected\nin advance", "gladv", "assfinadv"),
                ("level-free\n(after the fact)", None, None)]
    res = {}
    for y, _ in YEARS:
        cfg, ev, P, V, boots = data[y]
        o = json.loads((D72 / f"ratio_{y}_primary.json").read_text())["overall"]
        lf, _ = table(np.ones(len(ev), bool), P, {"gl": V["gl"], "assessor": V["assfin"]}, boots, centred)
        res[y] = [(o["models"][g]["mdape"]["point"], o["models"][a]["mdape"]["point"], o["pairs"][f"{g}-{a}"]["mdape"])
                  for _, g, a in versions[:2]] + [(lf["models"]["gl"]["point"], lf["models"]["assessor"]["point"],
                                                   lf["gl-assessor"])]
        print(f"  {y} level-free MdAPE: GL {res[y][2][0]:.2f}, rebuild {res[y][2][1]:.2f}, "
              f"difference {lf['gl-assessor']['point']:+.2f} [{lf['gl-assessor']['lo90']:+.2f}, "
              f"{lf['gl-assessor']['hi90']:+.2f}]", flush=True)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for k, (y, title) in enumerate(YEARS):
        a = ax[k]
        for j, col, lab in [(0, BLUE, "GL"), (1, ORANGE, ASS)]:
            a.plot(range(3), [r[j] for r in res[y]], color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=lab)
        a.set_xticks(range(3), [v[0] for v in versions])
        a.set_title(title, fontsize=9.5, loc="left")
        a.set_ylabel("MdAPE, %" if k == 0 else "")
        a.grid(axis="y", color="#e7e6e1", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(-0.3, 2.3)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.04, 1.0))
    a = ax[2]
    for k, (y, title) in enumerate(YEARS):
        for j in range(3):
            d = res[y][j][2]
            a.errorbar(j + (k - 0.5) * 0.25, d["point"], yerr=[[d["point"] - d["lo90"]], [d["hi90"] - d["point"]]],
                       fmt="o" if k == 0 else "s", color=INK2, ms=7, mfc=SURFACE if k else INK2, lw=1.5, capsize=0,
                       label=title if j == 0 else None)
    a.axhline(0, color=MUTED, lw=1)
    a.set_xticks(range(3), [v[0] for v in versions])
    a.set_ylabel("GL − Assessor, MdAPE points\n(90% interval)")
    a.set_title("Difference (below zero: GL more accurate)", fontsize=9.5, loc="left")
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e7e6e1", lw=0.8)
    a.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "accuracy_level.png", dpi=150)


def price_deciles(data):
    """Median ratio by sale-price decile: GL, the Assessor's model before rounding, published values (rounded)."""
    series = [("gl", "GL", BLUE, "o"), ("assfin", ASS, ORANGE, "s"),
              ("published", "Assessor, published (rounded)", GREEN, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    top = 1.60
    for ax, (y, title) in zip(axes, YEARS):
        cfg, ev, P, V, boots = data[y]
        pub = pd.read_parquet(EQ / f"values_{y}_published.parquet").set_index("meta_pin")["value"]
        vals = {"gl": V["gl"], "assfin": V["assfin"], "published": ev["meta_pin"].map(pub).astype(float).to_numpy()}
        assert not np.isnan(vals["published"]).any(), y
        cuts = np.quantile(P, np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(cuts, P, side="right") - 1, 0, 9)
        rows = []
        for k in range(10):
            sel = dec == k
            row = dict(decile=k + 1, price_median=float(np.median(P[sel])))
            for m, v in vals.items():
                r = v / P
                bs = [np.median(r[b][sel[b]]) for b in boots]
                row[m] = (float(np.median(r[sel])), float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)))
            rows.append(row)
        old = pd.read_csv(EQ / f"deciles_{y}.csv")   # equity_eval.py: same keys, cut points and resamples
        for m in ["gl", "published"]:
            got = np.array([r[m] for r in rows])
            ref = old[[f"{m}_median_ratio", f"{m}_lo90", f"{m}_hi90"]].to_numpy()
            assert np.allclose(got, ref, rtol=0, atol=1e-12), (y, m)
        x = np.array([r["decile"] for r in rows])
        ax.axhspan(0.90, 1.10, color="#e9e8e4", zorder=0, label="IAAO level range 0.90–1.10")
        ax.axhline(1.0, color="#6b6a64", lw=1, zorder=1)
        for i, (m, lab, col, mk) in enumerate(series):
            mid, lo, hi = (np.array([r[m][j] for r in rows]) for j in range(3))
            top = max(top, hi.max() + 0.02)
            ax.errorbar(x + (i - 1) * 0.12, mid, yerr=[mid - lo, hi - mid], color=col, marker=mk, ms=6, lw=2,
                        elinewidth=1, capsize=0, label=lab, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['decile']}\n${r['price_median'] / 1000:,.0f}k" for r in rows], fontsize=8)
        ax.set_xlabel("sale-price decile (median time-adjusted price)", fontsize=9)
        ax.set_title(f"{title} ({len(P):,} sales)", fontsize=10)
        ax.grid(axis="y", color="#e9e8e4", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("median ratio (value ÷ time-adjusted price)", fontsize=9)
    axes[0].set_ylim(0.80, top)
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Cheaper homes are valued relatively higher by all three: median ratio by sale-price decile "
                 "(90% intervals)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "price_deciles.png", dpi=150)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    by_township()
    if STRUCT:   # the 1 January figures before rounding (by_township above, these two from d72's studies)
        data = {y: study(y) for y, _ in YEARS}
        accuracy_level(data)
        price_deciles(data)
        print("wrote public/figures/by_township.png, accuracy_level.png, price_deciles.png")
        return
    subprocess.run([str(REPO / ".venv/bin/python"), "-I", str(REPO / "public/scripts/v2/d38_report.py")], check=True)
    shutil.copyfile(REPO / "notes/v2/figures/d38_level.png", FIG / "accuracy_level.png")
    print("wrote public/figures/by_township.png, accuracy_level.png")


if __name__ == "__main__":
    main()
