"""Figure of the level correction made in advance, from public/runs/v2/d38_eval/summary.json (d38_level.py).

Left: each model's MdAPE in the three versions (as is, corrected in advance, level-free), per year. Right: the paired
difference GL - Assessor with its 90% cluster-bootstrap interval.

Usage (repo root): .venv/bin/python -I public/scripts/v2/d38_report.py [--structure]
Writes d38_level.png to a working figures folder outside the release; with --structure (the published model P) reads
public/runs/v2/struct_eval/chosen/d38/summary.json and writes structure_d38_level.png there. readme_figures.py copies
it to public/figures/accuracy_level.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
BLUE, ORANGE, INK, INK2, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#b9b8b2", "#fcfcfb"
VERSIONS = [("as_is", "as is"), ("in_advance", "corrected\nin advance"), ("level_free", "level-free\n(after the fact)")]
YEARS = [("2023", "2023, south and west suburbs"), ("2022", "2022, north suburbs")]


def main():
    src, name = "public/runs/v2/d38_eval", "d38_level.png"
    if "--structure" in sys.argv:
        src, name = "public/runs/v2/struct_eval/chosen/d38", "structure_d38_level.png"
    s = json.loads((REPO / src / "summary.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    for k, (y, title) in enumerate(YEARS):
        a = ax[k]
        for m, col, lab in [("gl", BLUE, "GL"), ("assessor", ORANGE, "Assessor (our rebuild)")]:
            vals = [s[y][v][m]["mdape"]["point"] for v, _ in VERSIONS]
            a.plot(range(3), vals, color=col, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=lab)
        a.set_xticks(range(3), [lab for _, lab in VERSIONS])
        a.set_title(title, fontsize=9.5, loc="left")
        a.set_ylabel("MdAPE, %" if k == 0 else "")
        a.grid(axis="y", color="#e7e6e1", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
        a.set_xlim(-0.3, 2.3)
        if k == 0:
            a.legend(frameon=False, loc="upper right")
    a = ax[2]
    for k, (y, title) in enumerate(YEARS):
        for j, (v, lab) in enumerate(VERSIONS):
            d = s[y]["diff"][v]["mdape"]
            x = j + (k - 0.5) * 0.25
            mk = "o" if k == 0 else "s"
            a.errorbar(x, d["point"], yerr=[[d["point"] - d["lo90"]], [d["hi90"] - d["point"]]], fmt=mk, color=INK2,
                       ms=7, mfc=SURFACE if k else INK2, lw=1.5, capsize=0, label=title if j == 0 else None)
    a.axhline(0, color=MUTED, lw=1)
    a.set_xticks(range(3), [lab for _, lab in VERSIONS])
    a.set_ylabel("GL − Assessor, MdAPE points\n(90% interval)")
    a.set_title("Difference (below zero: GL more accurate)", fontsize=9.5, loc="left")
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color="#e7e6e1", lw=0.8)
    a.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = REPO / "notes/v2/figures" / name
    fig.savefig(out, dpi=150)
    print("wrote", out.relative_to(REPO))


if __name__ == "__main__":
    main()
