"""The README numbers that no other script writes (public README "Results", "Against IAAO guidance", "Limits").

1. How closely our rebuild of the Assessor's model reproduces the published values, per year: share of homes whose
   rebuilt value equals 10 x the published mailed assessed value exactly, and share within $1 of the assessed value
   (|rebuilt / 10 - mailed_tot| <= 1), overall and for Bloom (12) and Thornton (37). Population as p4_external_check.py
   (and ay2022/p4_external_check.py): roll PINs of the year's triad in classes 202-212, 234, 278, 295 with a rebuilt
   value and a published mailed value above zero.
2. The 2023 MdAPE difference GL - Assessor without Bloom and Thornton: ratio_v2.py, unchanged, on the chosen model's
   2023 keys minus those two townships (same settings as the headline); also PRD and PRB there for the GL, the rebuild
   and the published values (equity_eval.py's <chosen>/equity/values_2023_published.parquet, model "published").
3. IAAO measures on single-family sales (class group single-family), per year and model, read from the chosen model's
   headline studies (ratio_<year>_primary.json, subgroup class_group=single-family).
Writes <chosen>/readme_numbers.json and <chosen>/ratio_2023_excl_12_37.json; prints a summary.

Usage (repo root, after struct_chosen.py): .venv/bin/python -I public/scripts/v2/readme_numbers.py
       [--chosen public/runs/v2/struct_eval/chosen]
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CLASSES = ["202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "234", "278", "295"]
YEARS = {"2023": dict(triad="3", rebuild="ccao/runs/2023_rebuild", published="public/data/v2/p4_published_2023_south.parquet"),
         "2022": dict(triad="2", rebuild="ccao/runs/2022_rebuild", published="public/data/v2/ay2022/p8_published_2022_north.parquet")}


def reproduction(year, cfg):
    ap = pd.read_parquet(REPO / cfg["rebuild"] / "output/assessment_pin/model_assessment_pin.parquet",
                         columns=["meta_pin", "pred_pin_final_fmv_round"])
    roll = pd.read_parquet(REPO / f"ccao/inputs/{year}/assessment_data.parquet",
                           columns=["meta_pin", "meta_class", "meta_triad_code", "meta_township_code"]).drop_duplicates("meta_pin")
    pub = pd.read_parquet(REPO / cfg["published"], columns=["pin", "mailed_tot"])
    pub["mailed"] = pd.to_numeric(pub["mailed_tot"])
    d = roll[(roll["meta_triad_code"] == cfg["triad"]) & roll["meta_class"].isin(CLASSES)]
    d = d.merge(ap, on="meta_pin").merge(pub[["pin", "mailed"]], left_on="meta_pin", right_on="pin")
    d = d[d["pred_pin_final_fmv_round"].notna() & (d["mailed"] > 0)]
    v, m = d["pred_pin_final_fmv_round"].astype(float), d["mailed"].astype(float)
    d["exact"], d["within_1"] = v == 10 * m, (v / 10 - m).abs() <= 1
    out = dict(n=len(d), exact=float(d["exact"].mean()), within_1=float(d["within_1"].mean()))
    if year == "2023":
        out["within_1_by_town"] = {t: float(d.loc[d["meta_township_code"] == t, "within_1"].mean()) for t in ["12", "37"]}
        b = d[d["meta_township_code"] == "12"]
        out["bloom_median_rebuilt_over_published"] = float((b["pred_pin_final_fmv_round"] / (10 * b["mailed"])).median())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chosen", default="public/runs/v2/struct_eval/chosen")
    a = ap.parse_args()
    ch = REPO / a.chosen
    res = dict(reproduction={y: reproduction(y, c) for y, c in YEARS.items()})

    head = json.loads((ch / "ratio_2023_primary.json").read_text())
    keys = [k for k in head["evals"] if Path(k).parent.name not in ("t12", "t37")]
    assert len(keys) == len(head["evals"]) - 2
    out = ch / "ratio_2023_excl_12_37.json"
    cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py", *sum([["--eval", k] for k in keys], []),
           *sum([["--model", m] for m in head["models"]], []),
           "--model", f"published={(ch / 'equity/values_2023_published.parquet').relative_to(REPO)}",
           "--pair", "gl-assessor", "--pair", "gl-published", "--out", str(out.relative_to(REPO))]
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    ex = json.loads(out.read_text())["overall"]
    res["mdape_2023_excl_bloom_thornton"] = ex["pairs"]["gl-assessor"]["mdape"]
    res["excl_bloom_thornton_2023"] = dict(n=ex["n"], **{m: {s: ex["models"][m][s] for s in ["prd", "prb", "mdape"]}
                                                         for m in ex["models"]})

    res["single_family"] = {}
    for y in YEARS:
        sg = json.loads((ch / f"ratio_{y}_primary.json").read_text())["subgroups"]["class_group=single-family"]
        res["single_family"][y] = dict(n=sg["n"], **{m: {s: sg["models"][m][s]["point"] for s in ["cod", "prd", "prb"]}
                                                     for m in sg["models"]})
    (ch / "readme_numbers.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
