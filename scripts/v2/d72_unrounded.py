"""The 1 January tests compared like with like: the README's headline (both models before rounding).

An earlier comparison set GL's unrounded values against the Assessor's final values, which its pipeline rounds down
(02-assess.R §3.3; params.yaml round_type floor). Here, on the evaluation keys of the published model P
(struct_eval/chosen, as evaluated), with ratio_v2.py unchanged (time-adjusted prices, 2,000 resamples, seed 1, clustered by
valuation group within townships; the same resamples for every model, so differences are paired):

(a) unrounded: GL `value` against the Assessor's `pred_pin_final_fmv` (assfin): the model's card predictions summed
    to the PIN (`pred_pin_initial_fmv`, assinit) and then averaged within townhome complexes, multi-card capped and
    averaged across prorated buildings (02-assess.R §3.1-3.2), before rounding. On the evaluation sales only the
    complex averaging changes a value, and GL averages over the same groups, so assfin is the matched counterpart;
    assinit is reported alongside.
(b) rounded alike: GL `value_rounded` (the Assessor's rule, value_v2.py) against `pred_pin_final_fmv_round`.
(c) corrected in advance, as d38_level.py / d38_unadjusted.py but on the unrounded pair (a): each model divided by
    its median value / price over the October-December sales of the prior year (struct_eval/chosen/d38/q4_sales_*);
    `raw` = sale prices as they are (README main text), `adv` = prices time-adjusted (README figure version).
Sensitivities quoted in the README, on pair (a): sale prices not time-adjusted; 2023 without Bloom and Thornton.

Checks: gl and gl-assessor (rounded final values) reproduce the stored headline JSON exactly; the GL in-advance
factors reproduce the stored ones.

Usage (repo root): nice -n 15 .venv/bin/python -I public/scripts/v2/d72_unrounded.py
Writes public/runs/v2/struct_eval/chosen/d72/ (values_*, ratio_<year>_{primary,rounded,unadjusted}.json,
ratio_2023_excl_12_37.json, summary.json).
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CH = REPO / "public/runs/v2/struct_eval/chosen"
OUT = CH / "d72"
ASSESSOR = "ccao/runs/{y}_rebuild/output/assessment_pin/model_assessment_pin.parquet"
STATS = ["mdape", "median_ratio", "cod", "prd"]


def rel(p):
    return str(Path(p).relative_to(REPO))


def ratio(evals, models, pairs, out, extra=()):
    cmd = [str(REPO / ".venv/bin/python"), "-I", "public/scripts/v2/ratio_v2.py",
           *sum([["--eval", e] for e in evals], []), *sum([["--model", f"{n}={p}"] for n, p in models], []),
           *sum([["--pair", p] for p in pairs], []), *extra, "--out", rel(out)]
    rc = subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL).returncode
    assert rc == 0, out
    return json.loads(out.read_text())


def pick(block, names):
    return {n: {s: block[n][s] for s in STATS} for n in names}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for y in ["2023", "2022"]:
        print(f"\n=== {y}", flush=True)
        head = json.loads((CH / f"ratio_{y}_primary.json").read_text())
        head_r = json.loads((CH / f"ratio_{y}_rounded.json").read_text())
        evals = head["evals"]
        assert head_r["evals"] == evals
        gl = pd.read_parquet(CH / f"values_{y}.parquet", columns=["meta_pin", "value", "value_rounded"])
        a = pd.read_parquet(REPO / ASSESSOR.format(y=y), columns=["meta_pin", "pred_pin_initial_fmv", "pred_pin_final_fmv",
                                                                  "pred_pin_final_fmv_round"]).drop_duplicates("meta_pin").dropna()
        paths = {"assfin": OUT / f"values_{y}_assfin.parquet", "assinit": OUT / f"values_{y}_assinit.parquet"}
        pd.DataFrame({"meta_pin": a["meta_pin"], "value": a["pred_pin_final_fmv"],
                      "value_rounded": a["pred_pin_final_fmv_round"]}).to_parquet(paths["assfin"], index=False)
        pd.DataFrame({"meta_pin": a["meta_pin"], "value": a["pred_pin_initial_fmv"]}).to_parquet(paths["assinit"], index=False)

        # (c) factors on the prior October-December sales, as d38_level.py / d38_unadjusted.py
        q = pd.read_parquet(CH / f"d38/q4_sales_{y}.parquet")
        q["v_assfin"] = q["meta_pin"].map(a.set_index("meta_pin")["pred_pin_final_fmv"]).astype(float)
        assert q["v_assfin"].notna().all()
        f = {"raw": {"gl": float(np.median(q["v_gl"] / q["price"])), "assfin": float(np.median(q["v_assfin"] / q["price"]))},
             "adv": {"gl": float(np.median(q["v_gl"] / q["adj_price"])), "assfin": float(np.median(q["v_assfin"] / q["adj_price"]))},
             "n": int(len(q))}
        old_raw = json.loads((CH / "d38/unadjusted.json").read_text())[y]["factors"]
        old_adv = json.loads((CH / "d38/summary.json").read_text())[y]["factors"]
        assert abs(f["raw"]["gl"] - old_raw["gl"]) < 1e-12 and abs(f["adv"]["gl"] - old_adv["gl"]) < 1e-12
        f["old_rounded_assessor"] = {"raw": old_raw["assessor"], "adv": old_adv["assessor"]}
        print(f"  factors (n {f['n']:,}): {f['raw']} no time adj.; {f['adv']} time-adj.", flush=True)
        for k in ["raw", "adv"]:
            for m, v in [("gl", gl.set_index("meta_pin")["value"]), ("assfin", a.set_index("meta_pin")["pred_pin_final_fmv"])]:
                paths[f"{m}{k}"] = OUT / f"values_{y}_{m}_{k}.parquet"
                pd.DataFrame({"meta_pin": v.index, "value": v.to_numpy() / f[k][m]}).to_parquet(paths[f"{m}{k}"], index=False)

        models = [("gl", rel(CH / f"values_{y}.parquet")), ("assessor", ASSESSOR.format(y=y))] + \
                 [(n, rel(p)) for n, p in paths.items()]
        pairs = ["gl-assessor", "gl-assfin", "gl-assinit", "assfin-assinit", "glraw-assfinraw", "gladv-assfinadv"]
        res = ratio(evals, models, pairs, OUT / f"ratio_{y}_primary.json")
        o = res["overall"]
        for m in ["gl", "assessor"]:
            for s in ["mdape", "cod", "median_ratio", "prd"]:
                assert o["models"][m][s] == head["overall"]["models"][m][s], (y, m, s)
        assert o["pairs"]["gl-assessor"] == head["overall"]["pairs"]["gl-assessor"], y
        print("  check: gl and gl-assessor reproduce the stored headline exactly", flush=True)

        res_r = ratio(evals, [("gl", rel(CH / f"values_{y}.parquet")), ("assfin", rel(paths["assfin"]))], ["gl-assfin"],
                      OUT / f"ratio_{y}_rounded.json", ["--rounded"])
        # README sensitivities, unrounded pair: prices not time-adjusted; 2023 without Bloom (12) and Thornton (37)
        sens = {}
        res_u = ratio(evals, [("gl", rel(CH / f"values_{y}.parquet")), ("assfin", rel(paths["assfin"]))], ["gl-assfin"],
                      OUT / f"ratio_{y}_unadjusted.json", ["--sample", "unadjusted"])
        sens["unadjusted"] = res_u["overall"]["pairs"]["gl-assfin"]["mdape"]
        if y == "2023":
            ex = json.loads((CH / "ratio_2023_excl_12_37.json").read_text())["evals"]
            res_x = ratio(ex, [("gl", rel(CH / f"values_{y}.parquet")), ("assfin", rel(paths["assfin"]))], ["gl-assfin"],
                          OUT / "ratio_2023_excl_12_37.json")
            sens["excl_12_37"] = dict(n=res_x["overall"]["n"], diff=res_x["overall"]["pairs"]["gl-assfin"]["mdape"],
                                      mdape={m: res_x["overall"]["models"][m]["mdape"]["point"] for m in ["gl", "assfin"]})
        print(f"  sensitivities (MdAPE gl-assfin): {sens}", flush=True)
        orr = res_r["overall"]
        assert orr["models"]["gl"]["mdape"] == head_r["overall"]["models"]["gl"]["mdape"], y
        assert abs(orr["models"]["assfin"]["mdape"]["point"] - head_r["overall"]["models"]["assessor"]["mdape"]["point"]) < 1e-12

        summary[y] = dict(
            n=o["n"], n_clusters=o["n_clusters"], factors=f, sensitivities=sens,
            unrounded=dict(models=pick(o["models"], ["gl", "assfin", "assinit"]),
                           diff={"gl-assfin": {s: o["pairs"]["gl-assfin"][s] for s in STATS},
                                 "gl-assinit": {s: o["pairs"]["gl-assinit"][s] for s in STATS},
                                 "assfin-assinit": {s: o["pairs"]["assfin-assinit"][s] for s in STATS}}),
            rounded=dict(models=pick(orr["models"], ["gl", "assfin"]), diff={s: orr["pairs"]["gl-assfin"][s] for s in STATS}),
            previous_headline=dict(models=pick(o["models"], ["gl", "assessor"]), diff={s: o["pairs"]["gl-assessor"][s] for s in STATS}),
            in_advance_raw=dict(models=pick(o["models"], ["glraw", "assfinraw"]), diff={s: o["pairs"]["glraw-assfinraw"][s] for s in STATS}),
            in_advance_adv=dict(models=pick(o["models"], ["gladv", "assfinadv"]), diff={s: o["pairs"]["gladv-assfinadv"][s] for s in STATS}),
            subgroups={k: {"unrounded": v["pairs"]["gl-assfin"]["mdape"], "in_advance_raw": v["pairs"]["glraw-assfinraw"]["mdape"],
                           "n": v["n"]} for k, v in res["subgroups"].items()},
            by_town={t: {"unrounded": v["pairs"]["gl-assfin"]["mdape"], "n": v["n"]} for t, v in res["by_town"].items()})
        s = summary[y]
        for lab, blk, a_, b_ in [("unrounded", s["unrounded"], "gl", "assfin"), ("rounded", s["rounded"], "gl", "assfin"),
                                 ("in advance, raw", s["in_advance_raw"], "glraw", "assfinraw"),
                                 ("in advance, time-adj.", s["in_advance_adv"], "gladv", "assfinadv")]:
            d = blk["diff"]["gl-assfin"] if lab == "unrounded" else blk["diff"]
            mm = blk["models"]
            print(f"  {lab}: " + "; ".join(f"{s_} {mm[a_][s_]['point']:.4f} vs {mm[b_][s_]['point']:.4f} "
                                           f"(d {d[s_]['point']:+.3f} [{d[s_]['lo90']:+.3f}, {d[s_]['hi90']:+.3f}])"
                                           for s_ in STATS), flush=True)
        ai = s["unrounded"]["models"]["assinit"]
        print(f"  assinit: " + "; ".join(f"{s_} {ai[s_]['point']:.4f}" for s_ in STATS) +
              f"; gl-assinit MdAPE {s['unrounded']['diff']['gl-assinit']['mdape']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
