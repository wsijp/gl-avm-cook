"""The figures of the public README, in plain labels (public/figures/).

  accuracy_level.png   each model's MdAPE as is, corrected in advance and level-free, and the GL - Assessor difference
                       (from public/runs/v2/d38_eval/summary.json; the figure of d38_report.py)
  by_township.png      paired MdAPE difference GL - Assessor by township, both years (from the stored ratio studies
                       public/runs/v2/p7_eval/ratio_all_primary.json and p8_eval/ratio_all_primary.json)

Usage (repo root): .venv/bin/python -I public/scripts/v2/readme_figures.py [--structure]
--structure: from the published model P (public/runs/v2/struct_eval/chosen/: ratio_<year>_primary.json and
d38/summary.json); the README uses these.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "public/figures"
BLUE, INK, INK2, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#b9b8b2", "#fcfcfb"
NAMES = {"10": "Barrington", "11": "Berwyn", "12": "Bloom", "13": "Bremen", "14": "Calumet", "15": "Cicero",
         "16": "Elk Grove", "17": "Evanston", "18": "Hanover", "19": "Lemont", "20": "Leyden", "21": "Lyons",
         "22": "Maine", "23": "New Trier", "24": "Niles", "25": "Northfield", "26": "Norwood Park",
         "27": "Oak Park", "28": "Orland", "29": "Palatine", "30": "Palos", "31": "Proviso", "32": "Rich",
         "33": "River Forest", "34": "Riverside", "35": "Schaumburg", "36": "Stickney", "37": "Thornton",
         "38": "Wheeling", "39": "Worth"}
STRUCT = "--structure" in sys.argv
DAGGER = {"12", "37"}  # our rebuild of the Assessor's 2023 values differs most from the published values here


def by_township():
    studies = [("2023, south and west suburbs", "public/runs/v2/p7_eval/ratio_all_primary.json"),
               ("2022, north suburbs", "public/runs/v2/p8_eval/ratio_all_primary.json")]
    if STRUCT:
        studies = [(t, f"public/runs/v2/struct_eval/chosen/ratio_{y}_primary.json")
                   for (t, _), y in zip(studies, ["2023", "2022"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw=dict(height_ratios=[1]))
    for a, (title, path) in zip(axes, studies):
        r = json.loads((REPO / path).read_text())
        rows = [(t, v["n"], v["pairs"]["gl-assessor"]["mdape"]) for t, v in r["by_town"].items()]
        rows.sort(key=lambda x: -x[2]["point"])
        o = r["overall"]
        rows.append(("all", o["n"], o["pairs"]["gl-assessor"]["mdape"]))
        for i, (t, n, d) in enumerate(rows):
            y = i + (1 if t == "all" else 0)
            col = INK if t == "all" else BLUE
            a.plot([d["lo90"], d["hi90"]], [y, y], color=col, lw=2, solid_capstyle="round")
            a.plot(d["point"], y, "o", color=col, ms=7 if t == "all" else 6, mec=SURFACE, mew=1.5)
        labels = [f"{NAMES.get(t, 'all townships')}{' †' if t in DAGGER else ''} ({n:,})" for t, n, _ in rows]
        a.set_yticks([i + (1 if t == "all" else 0) for i, (t, _, _) in enumerate(rows)], labels)
        a.axvline(0, color=MUTED, lw=1)
        a.set_xlim(-5, 9)
        a.set_title(title, fontsize=10, loc="left")
        a.set_xlabel("GL − Assessor (our rebuild), MdAPE points (90% interval)\nbelow zero: GL more accurate")
        a.grid(axis="x", color="#e7e6e1", lw=0.8)
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(axis="y", length=0)
    fig.text(0.01, 0.005, "Number of sales in brackets. † Our rebuild of the Assessor's model differs most from its "
             "published values in Bloom and Thornton.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "by_township.png", dpi=150)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    by_township()
    subprocess.run([str(REPO / ".venv/bin/python"), "-I", str(REPO / "public/scripts/v2/d38_report.py")]
                   + (["--structure"] if STRUCT else []), check=True)
    shutil.copyfile(REPO / "notes/v2/figures" / ("structure_d38_level.png" if STRUCT else "d38_level.png"),
                    FIG / "accuracy_level.png")
    print("wrote public/figures/by_township.png, accuracy_level.png")


if __name__ == "__main__":
    main()
