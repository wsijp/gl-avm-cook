"""County test of the sale-history model: the adopted sale-history model variant (H2: separate house and flat models
with each home's sale history) on all 38 townships of the Assessor's 2025 split, once, against the Assessor's 2025 test
predictions and model P's county test (P = the published model).

  run  --arm H2   H2's fits: house-only fits in every township (loop_fit.py --county, prep_county2025_h2h), flat-only
                  fits on flat-building nodes where flats are eligible (>= 300 distinct flat training PINs and >= 24
                  distinct months; prep_county2025_h2f), and for the other townships' flats an H1 fit (P fitted to
                  houses and flats together with sale-history columns; fit_struct.py via county2025_fit.py +
                  county2025_predict.py, prep_county2025_h1). 4 chains x 1,000 + 1,000, seed 1, 94 months, held flat
                  at October 2023. At most 3 fits at a time, each under nice 15, largest first, a launch only with
                  >= 25% memory free. Restart-safe (a fit with predictions is skipped).
       (H1's fits: county2025_run.py --prep public/data/v2/prep_county2025_h1 --prefix county2025_H1
                  --out public/runs/v2/county2025_h1_eval)
  eval --arm H1|H2   for H2 first composes each township's predictions (houses from H2h, flats from H2f or the H1
                  fallback) and writes runs.json in county2025_eval.py's format (convergence: the worse of the
                  township's fits); then county2025_eval.py --out <dir> (the county evaluation unchanged), then the
                  sale-history tables: resold vs other test sales (an earlier sale in the data before the sale's date)
                  and by gap band, the GL model variant vs the Assessor and vs model P, paired, with the same 2,000
                  PIN-within-township resamples.

  --house / --flat: later model variants (not used in the final model): house fits from prep_county2025_<house> (H2h,
                  N1h, N2h) and eligible flat fits from prep_county2025_<flat> (H2f, N3f, N4f); fits named
                  county2025_<kind>_t<T> (existing ones are reused: restart-safe); output county2025_<house>+<flat>_eval
                  (lower case). The cap of 3 fits is machine-wide under a lock shared with the model-development driver.

Usage (repo root): nohup caffeinate -is .venv/bin/python -I public/scripts/v2/salehist_county.py run --arm H2 \
                   > notes/v2/logs/salehist_county_h2.log 2>&1 < /dev/null &
                   .venv/bin/python -I public/scripts/v2/salehist_county.py eval --arm H1
Writes public/runs/v2/county2025_<h1|h2>_eval/ (runs.json, summary.*, salehist.json/.md).
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv/bin/python")
SCR = REPO / "public/scripts/v2"
RUNS = REPO / "public/runs/v2"
LOGS = REPO / "notes/v2/logs"
KEYS = ["meta_pin", "meta_sale_document_num", "meta_card_num"]
T38 = [str(x) for x in list(range(10, 40)) + list(range(70, 78))]
PREP = {"H1": "public/data/v2/prep_county2025_h1", "H2h": "public/data/v2/prep_county2025_h2h",
        "H2f": "public/data/v2/prep_county2025_h2f"}
PREP.update({k: f"public/data/v2/prep_county2025_{k.lower()}" for k in ("N1h", "N2h", "N3f", "N4f")})   # later variants
MAX_PARALLEL, MIN_FREE = 3, 25
BANDS = [0, 12, 24, 48, 1e9]
BAND_NAMES = ["gap<1y", "gap1-2y", "gap2-4y", "gap4y+"]


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def free_percent():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    line = [x for x in out.splitlines() if "free percentage" in x]
    return int(line[-1].split(":")[-1].strip().rstrip("%")) if line else 100


def flat_eligible():
    out = []
    for T in T38:
        t = pd.read_parquet(REPO / PREP["H1"] / f"t{T}" / "train.parquet", columns=["meta_pin", "month", "is_flat"])
        f = t[t["is_flat"]]
        if f["meta_pin"].nunique() >= 300 and f["month"].nunique() >= 24:
            out.append(T)
    return out


def jobs(house="H2h", flat="H2f"):
    el = flat_eligible()
    size = {T: len(pd.read_parquet(REPO / PREP["H1"] / f"t{T}" / "train.parquet", columns=["meta_pin"])) for T in T38}
    J = [dict(kind=house, town=T, name=f"county2025_{house}_t{T}") for T in T38]
    J += [dict(kind=flat, town=T, name=f"county2025_{flat}_t{T}") for T in el]
    J += [dict(kind="H1", town=T, name=f"county2025_H1_t{T}") for T in T38 if T not in el]
    for j in J:
        j["size"] = size[j["town"]]
    return sorted(J, key=lambda j: -j["size"]), el


lock = threading.Lock()


def running_fits():
    out = subprocess.run(["pgrep", "-f", "public/scripts/v2/(loop_fit|county2025_fit).py"], capture_output=True, text=True).stdout
    return len(out.split())


def run_one(j):
    rd = RUNS / j["name"]
    if (rd / "predictions.parquet").exists():
        log(f"skip {j['name']} (predictions exist)")
        return j["name"], 0
    import fcntl
    with lock, open(LOGS / ".loop_run.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)                 # machine-wide cap shared with the model-development driver
        while free_percent() < MIN_FREE or running_fits() >= MAX_PARALLEL:
            log(f"waiting before {j['name']} ({free_percent()}% free, {running_fits()} fits running)")
            time.sleep(60)
        log(f"launch {j['name']}")
        time.sleep(5)
    t0 = time.time()
    lf = open(LOGS / f"{j['name']}.log", "a")
    if j["kind"] == "H1":
        cmds = [[PY, "-I", str(SCR / "county2025_fit.py"), "--town", j["town"], "--model", "P", "--prep", PREP["H1"],
                 "--seed", "1", "--name", j["name"]],
                [PY, "-I", str(SCR / "county2025_predict.py"), str(rd), "--prep", PREP["H1"]]]
        if (rd / "report.json").exists():
            cmds = cmds[1:]
    else:
        cmds = [[PY, "-I", str(SCR / "loop_fit.py"), "--county", "--town", j["town"], "--prep", PREP[j["kind"]],
                 "--name", j["name"], "--seed", "1", "--warmup", "1000", "--samples", "1000"]]
    rc = 0
    for c in cmds:
        rc = subprocess.run(["nice", "-n", "15"] + c, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
        if rc:
            break
    lf.close()
    log(f"{'done' if rc == 0 else 'FAILED (exit %d)' % rc} {j['name']} in {(time.time() - t0) / 60:.1f} min")
    return j["name"], rc


def run(arm, house="H2h", flat="H2f"):
    assert arm == "H2", "H1's county fits run through county2025_run.py (see the docstring)"
    J, el = jobs(house, flat)
    log(f"=== salehist_county run {house}+{flat}: {len(J)} fits; flats eligible {el}; order {[j['name'] for j in J]}")
    with ThreadPoolExecutor(MAX_PARALLEL) as ex:
        res = list(ex.map(run_one, J))
    bad = [n for n, rc in res if rc]
    if bad:
        log(f"STOPPED at: {len(bad)} failed fits: {bad}")
        sys.exit(1)
    log("DONE")


def loop_fit_conv(rep, pred):
    d = rep["diagnostics"]
    bc = {k: v for k, v in d["by_component"].items() if k != "scalars_each"}
    return dict(max_rhat=max(v["rhat_max"] for v in bc.values()), min_ess_bulk=min(v["ess_bulk_min"] for v in bc.values()),
                min_ess_tail=min(v["ess_tail_min"] for v in bc.values()), divergent=d["divergent"],
                ebfmi_min=min(d["ebfmi"]), max_treedepth_share=d["max_treedepth_share"],
                sum_to_zero_pass=rep["sum_to_zero"]["all_pass"], pred_rhat_max=float(pred["latent_rhat"].max()),
                runtime_s=rep["timing"]["runtime_s"], n_train=rep["n_train"])


def struct_conv(rep):
    d = rep["diagnostics"]
    return dict(max_rhat=d["max_rhat"], min_ess_bulk=d["min_ess_bulk"], min_ess_tail=d["min_ess_tail"],
                divergent=d["divergent"], ebfmi_min=min(d["ebfmi"]), max_treedepth_share=d["max_treedepth_share"],
                sum_to_zero_pass=rep["sum_to_zero"]["all_pass"], pred_rhat_max=rep["prediction"]["latent_split_rhat"]["max"],
                runtime_s=rep["timing"]["runtime_s"], n_train=rep["n_train"])


def compose_h2(out, house="H2h", flat="H2f"):
    """houses from <house> (H2h), flats from <flat> (H2f; eligible townships) or the H1 fallback; runs.json in county2025_eval.py's
    format."""
    el = flat_eligible()
    runs = {}
    for T in T38:
        roll = pd.read_parquet(REPO / PREP["H1"] / f"t{T}" / "roll.parquet",
                               columns=KEYS + ["meta_class", "is_flat", "meta_tieback_proration_rate", "date",
                                               "meta_sale_price", "meta_township_code", "nbhd_id", "has_training_sale"])
        parts, convs, src = [], [], {}
        h = pd.read_parquet(RUNS / f"county2025_{house}_t{T}" / "predictions.parquet")
        convs.append(loop_fit_conv(json.loads((RUNS / f"county2025_{house}_t{T}" / "report.json").read_text()), h))
        parts.append(h[KEYS + ["pred", "pred_q05", "pred_q95", "latent_rhat"]])
        src["houses"] = f"county2025_{house}_t{T}"
        if roll["is_flat"].any():
            if T in el:
                f = pd.read_parquet(RUNS / f"county2025_{flat}_t{T}" / "predictions.parquet")
                convs.append(loop_fit_conv(json.loads((RUNS / f"county2025_{flat}_t{T}" / "report.json").read_text()), f))
                src["flats"] = f"county2025_{flat}_t{T}"
            else:
                f = pd.read_parquet(RUNS / f"county2025_H1_t{T}" / "predictions.parquet")
                f = f[f[KEYS].merge(roll[KEYS + ["is_flat"]], on=KEYS, how="left")["is_flat"].to_numpy(bool)]
                convs.append(struct_conv(json.loads((RUNS / f"county2025_H1_t{T}" / "report.json").read_text())))
                src["flats"] = f"county2025_H1_t{T} (fallback: flats not eligible for their own model)"
            parts.append(f[KEYS + ["pred", "pred_q05", "pred_q95", "latent_rhat"]])
        p = pd.concat(parts, ignore_index=True)
        assert not p.duplicated(KEYS).any()
        m = roll.merge(p, on=KEYS, how="left", validate="one_to_one")
        assert m["pred"].notna().all() and len(m) == len(roll), f"t{T}: a test sale without a prediction"
        m["town"] = T
        d = out / "composite" / f"t{T}"
        d.mkdir(parents=True, exist_ok=True)
        m.to_parquet(d / "predictions.parquet", index=False)
        c = dict(max_rhat=max(x["max_rhat"] for x in convs), min_ess_bulk=min(x["min_ess_bulk"] for x in convs),
                 min_ess_tail=min(x["min_ess_tail"] for x in convs), divergent=sum(x["divergent"] for x in convs),
                 ebfmi_min=min(x["ebfmi_min"] for x in convs), max_treedepth_share=max(x["max_treedepth_share"] for x in convs),
                 sum_to_zero_pass=all(x["sum_to_zero_pass"] for x in convs), pred_rhat_max=max(x["pred_rhat_max"] for x in convs),
                 runtime_s=sum(x["runtime_s"] for x in convs), n_train=sum(x["n_train"] for x in convs))
        runs[T] = dict(town=T, status="done", run=str(d.relative_to(REPO)), sources=src, **c)
    (out / "runs.json").write_text(json.dumps(runs, indent=1))


def salehist_tables(arm, out):
    """resold vs other test sales and gap bands: the model variant vs the Assessor and vs model P, paired bootstrap."""
    sys.path.insert(0, str(SCR))
    import county2025_eval as ce
    import salehist
    ts = pd.read_parquet(out / "test_sales.parquet")
    p0 = pd.read_parquet(RUNS / "county2025_eval/test_sales.parquet", columns=KEYS + ["pred_gl"])
    ts = ts.merge(p0.rename(columns={"pred_gl": "pred_p"}), on=KEYS, how="left", validate="one_to_one")
    assert ts["pred_p"].notna().all()
    ts["date"] = pd.to_datetime(ts["date"])
    h = salehist.history(ts, salehist.source())
    ts["n_prev"], ts["months"] = h["n_prev"].to_numpy(), h["months_since"].to_numpy()
    cls = pd.read_parquet(REPO / "ccao/inputs/2025/training_data.parquet", columns=KEYS + ["meta_class"])
    ts = ts.drop(columns=[c for c in ts.columns if c.startswith("meta_class")]).merge(
        cls.drop_duplicates(KEYS), on=KEYS, how="left", validate="one_to_one")
    ts["flat"] = ts["meta_class"].astype(str).isin(["211", "212"])
    band = pd.cut(ts["months"], BANDS, right=False, labels=BAND_NAMES).astype(str)
    rng = np.random.default_rng(ce.SEED)
    W = np.zeros((ce.B, len(ts)), np.float32)
    for T, g in ts.groupby("town"):
        pins, inv = np.unique(g["meta_pin"].to_numpy(), return_inverse=True)
        cnt = rng.multinomial(len(pins), np.full(len(pins), 1 / len(pins)), size=ce.B)
        W[:, g.index.to_numpy()] = cnt[:, inv]
    price = ts["meta_sale_price"].to_numpy(float)
    res = []
    resold = ts["n_prev"].to_numpy() > 0
    for area, am in [("county", np.ones(len(ts), bool)), ("City", ts["area"].eq("City").to_numpy()),
                     ("suburbs", ts["area"].eq("suburbs").to_numpy())]:
        groups = [("all", am), ("resold", am & resold), ("other", am & ~resold), ("houses", am & ~ts["flat"].to_numpy()),
                  ("flats", am & ts["flat"].to_numpy())]
        if area == "county":
            groups += [(b, am & (band == b).to_numpy()) for b in BAND_NAMES]
        for g, m in groups:
            if m.sum() < 30:
                continue
            row = dict(area=area, group=g, n=int(m.sum()))
            for k, col in (("arm", "pred_gl"), ("assessor", "pred_assessor"), ("p", "pred_p")):
                r = ts[col].to_numpy(float)[m] / price[m]
                row[f"mdape_{k}"] = 100 * np.median(np.abs(r - 1))
                row[f"ratio_{k}"] = float(np.median(r))
            for k, col in (("assessor", "pred_assessor"), ("p", "pred_p")):
                a = ce.wquantile(np.abs(ts["pred_gl"].to_numpy(float)[m] / price[m] - 1), W[:, m])
                b = ce.wquantile(np.abs(ts[col].to_numpy(float)[m] / price[m] - 1), W[:, m])
                lo, hi = np.quantile(100 * (a - b), [0.05, 0.95])
                row[f"d_vs_{k}"], row[f"d_vs_{k}_lo"], row[f"d_vs_{k}_hi"] = row["mdape_arm"] - row[f"mdape_{k}"], lo, hi
            res.append(row)
    R = pd.DataFrame(res)
    R.to_csv(out / "salehist.csv", index=False)
    L = [f"# County test of {arm}: sale-history groups", "", f"_salehist_county.py eval; {len(ts):,} held-out test sales; "
         "resold = an earlier sale of the PIN in the data (since 2016) before the sale's date; differences in MdAPE points "
         "(arm minus comparison; negative = arm better), 90% paired PIN-within-township bootstrap (2,000)._", "",
         f"| area | group | n | {arm} | P | Assessor | {arm} − P | {arm} − Assessor | median ratio {arm} / P / Assessor |",
         "|---|---|---:|---:|---:|---:|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.area} | {r.group} | {r.n:,} | {r.mdape_arm:.2f} | {r.mdape_p:.2f} | {r.mdape_assessor:.2f} | "
                 f"{r.d_vs_p:+.2f} [{r.d_vs_p_lo:+.2f}, {r.d_vs_p_hi:+.2f}] | {r.d_vs_assessor:+.2f} [{r.d_vs_assessor_lo:+.2f}, "
                 f"{r.d_vs_assessor_hi:+.2f}] | {r.ratio_arm:.3f} / {r.ratio_p:.3f} / {r.ratio_assessor:.3f} |")
    (out / "salehist.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def evaluate(arm, house="H2h", flat="H2f"):
    label = arm if (house, flat) == ("H2h", "H2f") else f"{house}+{flat}"
    out = RUNS / f"county2025_{label.lower()}_eval"
    if arm == "H2":
        compose_h2(out, house, flat)
    arm = label
    r = subprocess.run([PY, "-I", str(SCR / "county2025_eval.py"), "--out", str(out.relative_to(REPO))], cwd=REPO)
    assert r.returncode == 0
    salehist_tables(arm, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "eval"])
    ap.add_argument("--arm", choices=["H1", "H2"], required=True)
    ap.add_argument("--house", default="H2h", choices=["H2h", "N1h", "N2h"])
    ap.add_argument("--flat", default="H2f", choices=["H2f", "N3f", "N4f"])
    a = ap.parse_args()
    run(a.arm, a.house, a.flat) if a.step == "run" else evaluate(a.arm, a.house, a.flat)


if __name__ == "__main__":
    main()
