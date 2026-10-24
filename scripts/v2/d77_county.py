"""The final sale-prediction test of the final model (IXa: separate house and flat models, both with each home's sale
history and the forecasting time model; for houses also flip-purchase flags, a resale premium and flip discount that
vary by neighbourhood, and age interactions by house class and by building size) on the Assessor's 2025 split, run once, against the Assessor's 2025 test
predictions (and against model P's and the sale-history model H2's county tests).

  prep   houses: the H2 county preparation (prep_county2025_h2h: houses with segment-specific coding, history from the
         2025 input strictly before each sale, no cutoff) + the flip-purchase flags (d63_prep.build) + the age
         interactions (d70_prep.build), written under public/data/v2/county_d77/ (prep_loop_k1h, prep_loop_ixah: the
         names used in model development, in a separate folder, so no development preparation is touched); flats:
         loop_prep.py --county with the options of the flat model with complete sale history since 1999 (flats with
         segment-specific coding, building nodes, complete Parcel Sales history, fading resale premium) for the 18
         flat-eligible townships (salehist_county.flat_eligible: >= 300 flat training PINs and >= 24 months),
         public/data/v2/county_d77/prep_county_fcf.
  run    56 fits, loop_fit.py --county, 4 chains x 1,000 + 1,000, seed 1: county2025_IXah_t<T> (38 house fits;
         --forecast --np) and county2025_FCf_t<T> (18 flat fits; --forecast --gexp). At most 3 fits machine-wide (a
         lock shared with the model-development driver), nice 15, >= 25% memory free, longest expected first (the H2
         county fits' runtimes). Restart-safe (a fit with predictions is skipped).
  eval   compose each township (houses IXah; flats FCf or, in the 20 other townships, the H2 county test's fallback
         fits county2025_H1_t<T> (P with sale-history columns, fitted to houses and flats together), reused) with
         salehist_county.compose_h2; county2025_eval.py unchanged; the sale-history tables (vs P and the Assessor);
         then d77.json/d77.md: IXa against H2, houses and flats, the warning group found in development (City houses
         over 100 years old with no sale since 1999, complete history), the usefulness bar fixed before the test, and
         the figure.

Usage (repo root): .venv/bin/python -I public/scripts/v2/d77_county.py prep
                   nohup caffeinate -is .venv/bin/python -I public/scripts/v2/d77_county.py run \
                       > notes/v2/logs/d77_county_run.log 2>&1 < /dev/null &
                   .venv/bin/python -I public/scripts/v2/d77_county.py eval
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
SCR = REPO / "public/scripts/v2"
sys.path.insert(0, str(SCR))
import salehist_county as sc  # noqa: E402

PY, RUNS, LOGS, KEYS, T38 = sc.PY, sc.RUNS, sc.LOGS, sc.KEYS, sc.T38
D77 = "public/data/v2/county_d77"
PREP = {"IXah": f"{D77}/prep_loop_ixah", "FCf": f"{D77}/prep_county_fcf"}
FLAGS = {"IXah": ["--forecast", "--np"], "FCf": ["--forecast", "--gexp"]}
OUT = RUNS / "county2025_ixa_eval"
BAR = {"county": 0.3, "City": 0.75, "suburbs": 0.3}                # usefulness bar (fixed before the test): upper 90% bound of IXa - Assessor below these


def prep():
    prep_houses()
    prep_flats()


def prep_houses():
    import d63_prep
    import d70_prep
    data = REPO / D77
    data.mkdir(parents=True, exist_ok=True)
    d63_prep.DATA = d70_prep.DATA = data
    d63_prep.ARMS["k1h"] = ("../prep_county2025_h2h", "h", True, False)
    for T in T38:
        d63_prep.build("k1h", "", T, ["lt12", "12_24"])
    for T in T38:
        d70_prep.build("ixah", "", T)
    ss = json.loads((REPO / "public/data/v2/prep_county2025_h2h/split_summary.json").read_text())
    for arm, parent in (("k1h", "prep_county2025_h2h"), ("ixah", "county_d77/prep_loop_k1h")):
        ss["options"] = dict(ss["options"], d63="k1", **({"d70": "ixah"} if arm == "ixah" else {}))
        ss["parent"] = parent
        (data / f"prep_loop_{arm}" / "split_summary.json").write_text(json.dumps(ss, indent=1, default=str))
    print("houses DONE", flush=True)


def prep_flats():
    el = sc.flat_eligible()
    assert el == ["11", "15", "17", "20", "21", "24", "27", "31", "37", "39", "70", "71", "72", "73", "74", "75", "76", "77"], el
    cmd = [PY, "-I", str(SCR / "loop_prep.py"), "--county", "--towns", *el, "--out", PREP["FCf"], "--scode", "--segment",
           "flat", "--flat-nodes", "building", "--salehist", "--fullhist", "long", "--gapcode", "exp"]
    print(" ".join(cmd), flush=True)
    assert subprocess.run(cmd, cwd=REPO).returncode == 0
    print("DONE", flush=True)


def jobs():
    el = sc.flat_eligible()
    J = [dict(kind="IXah", town=T, name=f"county2025_IXah_t{T}", ref=f"county2025_H2h_t{T}") for T in T38]
    J += [dict(kind="FCf", town=T, name=f"county2025_FCf_t{T}", ref=f"county2025_H2f_t{T}") for T in el]
    for j in J:
        j["expected"] = json.loads((RUNS / j["ref"] / "report.json").read_text())["timing"]["runtime_s"]
    return sorted(J, key=lambda j: -j["expected"])


lock = threading.Lock()


def run_one(j, seed=1, warmup=1000, samples=1000, runs="public/runs/v2"):
    rd = REPO / runs / j["name"]
    if (rd / "predictions.parquet").exists():
        sc.log(f"skip {j['name']} (predictions exist)")
        return j["name"], 0
    import fcntl
    with lock, open(LOGS / ".loop_run.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)                 # machine-wide cap shared with the development driver and salehist_county.py
        while sc.free_percent() < sc.MIN_FREE or sc.running_fits() >= sc.MAX_PARALLEL:
            sc.log(f"waiting before {j['name']} ({sc.free_percent()}% free, {sc.running_fits()} fits running)")
            time.sleep(60)
        sc.log(f"launch {j['name']}")
        cmd = ["nice", "-n", "15", PY, "-I", str(SCR / "loop_fit.py"), "--county", "--town", j["town"],
               "--prep", PREP[j["kind"]], "--name", j["name"], "--seed", str(seed), "--warmup", str(warmup),
               "--samples", str(samples), "--runs", runs] + FLAGS[j["kind"]] + j.get("extra", [])
        t0 = time.time()
        lf = open(LOGS / f"{j['name']}.log", "a")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        time.sleep(5)                                  # visible to running_fits() before the lock is released
    rc = proc.wait()
    lf.close()
    sc.log(f"{'done' if rc == 0 else 'FAILED (exit %d)' % rc} {j['name']} in {(time.time() - t0) / 60:.1f} min")
    return j["name"], rc


def run():
    J = jobs()
    sc.log(f"=== d77_county run: {len(J)} fits, order {[j['name'] for j in J]}")
    with ThreadPoolExecutor(sc.MAX_PARALLEL) as ex:
        res = list(ex.map(run_one, J))
    bad = [n for n, rc in res if rc]
    if bad:
        sc.log(f"STOPPED at: {len(bad)} failed fits: {bad}")
        sys.exit(1)
    sc.log("DONE")


def smoke():
    """Smoke test before the run: River Forest houses and Oak Park flats, 1 chain x (30 + 30), into a scratch folder; checks the
    mechanics (preparation, county wrapper, horizon-0 rule, key contract); accuracy not computed."""
    runs = "public/runs/v2/_d77_smoke"
    # flats 200 + 200: at 30 + 30 one chain stuck at the depth limit left a scalar constant, which loop_fit.py's
    # diagnostics drop (a smoke-only artefact, 2026-10-24)
    for j, it in ((dict(kind="IXah", town="33", name="smoke_IXah_t33"), 30), (dict(kind="FCf", town="27", name="smoke_FCf_t27"), 200)):
        j["extra"] = ["--chains", "1"]
        name, rc = run_one(j, warmup=it, samples=it, runs=runs)
        assert rc == 0, name
        rd = REPO / runs / name
        p = pd.read_parquet(rd / "predictions.parquet")
        roll = pd.read_parquet(REPO / PREP[j["kind"]] / f"t{j['town']}" / "roll.parquet", columns=KEYS + ["month"])
        assert len(p) == len(roll) and (p[KEYS].to_numpy() == roll[KEYS].to_numpy()).all(), "key contract"
        assert p["pred"].gt(0).all() and np.isfinite(p["pred"]).all()
        rep = json.loads((rd / "report.json").read_text())
        h0 = (roll["month"] == 94).to_numpy()
        assert rep["forecast"]["n_horizon0"] == int(h0.sum())
        # horizon 0 takes the value at the last training month: pred equals pred_apr exactly there
        assert np.allclose(p.loc[h0, "pred"], p.loc[h0, "pred_apr"]), "horizon-0 rule"
        print(f"smoke {name}: {len(p)} test sales, {int(h0.sum())} at horizon 0, forecast {rep['forecast']['checks']}, "
              f"H {rep['forecast']['H']}, n_train {rep['n_train']}, regressors {rep['n_regressors']}", flush=True)
    print("SMOKE OK", flush=True)


# ---------------------------------------------------------------------------------------------------- evaluation
def weights(ts):
    import county2025_eval as ce
    rng = np.random.default_rng(ce.SEED)
    W = np.zeros((ce.B, len(ts)), np.float32)
    for T, g in ts.groupby("town"):
        pins, inv = np.unique(g["meta_pin"].to_numpy(), return_inverse=True)
        cnt = rng.multinomial(len(pins), np.full(len(pins), 1 / len(pins)), size=ce.B)
        W[:, g.index.to_numpy()] = cnt[:, inv]
    return W


def extra_tables():
    """IXa against H2 (the sale-history model's county test) and the Assessor by area x houses / flats, and the warning
    group found in development (City houses over 100 years old with no sale since 1999);
    the same resamples as county2025_eval.py (same seed, same construction, same row order)."""
    import county2025_eval as ce
    import salehist_full
    import loop_prep
    ts = pd.read_parquet(OUT / "test_sales.parquet")
    ts = ts[ts["pred_gl"].notna()].reset_index(drop=True)
    h2 = pd.read_parquet(RUNS / "county2025_h2_eval/test_sales.parquet", columns=KEYS + ["pred_gl"])
    p0 = pd.read_parquet(RUNS / "county2025_eval/test_sales.parquet", columns=KEYS + ["pred_gl"])
    ts = ts.merge(h2.rename(columns={"pred_gl": "pred_h2"}), on=KEYS, how="left", validate="one_to_one")
    ts = ts.merge(p0.rename(columns={"pred_gl": "pred_p"}), on=KEYS, how="left", validate="one_to_one")
    assert ts["pred_h2"].notna().all() and ts["pred_p"].notna().all()
    inp = pd.read_parquet(REPO / "ccao/inputs/2025/training_data.parquet",
                          columns=KEYS + ["meta_class", "char_yrblt"]).drop_duplicates(KEYS)
    ts = ts.drop(columns=[c for c in ts.columns if c.startswith("meta_class")]).merge(inp, on=KEYS, how="left",
                                                                                     validate="one_to_one")
    ts["flat"] = ts["meta_class"].astype(str).isin(["211", "212"])
    salehist_full.PARCEL_FILES = salehist_full.PARCEL_FILES + [REPO / loop_prep.PARCEL_D77]
    ts["date"] = pd.to_datetime(ts["date"])
    h = salehist_full.history(ts, salehist_full.source(before="2099-01-01"))
    ts["never_sold_1999"] = h["n_prev"].to_numpy() == 0
    ts["age"] = ts["date"].dt.year - pd.to_numeric(ts["char_yrblt"], errors="coerce")
    W = weights(ts)
    price = ts["meta_sale_price"].to_numpy(float)
    city = ts["area"].eq("City").to_numpy()
    fl = ts["flat"].to_numpy()
    warn = city & ~fl & ts["never_sold_1999"].to_numpy() & (ts["age"] > 100).to_numpy()
    groups = []
    for area, am in [("county", np.ones(len(ts), bool)), ("City", city), ("suburbs", ~city)]:
        groups += [(area, "all", am), (area, "houses", am & ~fl), (area, "flats", am & fl)]
    groups += [("City", "houses over 100 years old, not sold since 1999", warn)]
    rows = []
    for area, g, m in groups:
        row = dict(area=area, group=g, n=int(m.sum()))
        ape = {k: np.abs(ts[c].to_numpy(float)[m] / price[m] - 1) for k, c in
               (("ixa", "pred_gl"), ("assessor", "pred_assessor"), ("h2", "pred_h2"), ("p", "pred_p"))}
        boot = {k: ce.wquantile(v, W[:, m]) for k, v in ape.items()}
        for k, v in ape.items():
            row[f"mdape_{k}"] = 100 * float(np.median(v))
            row[f"ratio_{k}"] = float(np.median(ts[{"ixa": "pred_gl", "assessor": "pred_assessor", "h2": "pred_h2",
                                                     "p": "pred_p"}[k]].to_numpy(float)[m] / price[m]))
        for k in ("assessor", "h2", "p"):
            lo, hi = np.quantile(100 * (boot["ixa"] - boot[k]), [0.05, 0.95])
            row[f"d_vs_{k}"], row[f"d_vs_{k}_lo"], row[f"d_vs_{k}_hi"] = row["mdape_ixa"] - row[f"mdape_{k}"], lo, hi
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "d77_groups.csv", index=False)
    return R


def bar():
    s = pd.read_csv(OUT / "summary.csv", dtype={"unit": str}).set_index("unit")
    res = {}
    for u, lim in BAR.items():
        r = s.loc[u]
        res[u] = dict(d_mdape=r["d_mdape"], lo=r["d_mdape_lo"], hi=r["d_mdape_hi"], limit=lim, passes=bool(r["d_mdape_hi"] < lim))
    r = s.loc["county"]
    res["PRD"] = dict(prd_ixa=r["prd_gl"], prd_assessor=r["prd_assessor"], d=r["d_prd"], lo=r["d_prd_lo"], hi=r["d_prd_hi"],
                      passes=bool(r["prd_gl"] <= r["prd_assessor"]))
    res["1 January (D72)"] = dict(note="already true (-0.40 2023 south, -0.33 2022 north); not retested", passes=True)
    return res


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import county2025_figure as cf
    s = pd.read_csv(OUT / "summary.csv", dtype={"unit": str})
    s["code"] = s["unit"].str.split(" ").str[0]
    town = s[s["code"].str.fullmatch(r"\d\d")].copy()
    town["city"] = town["code"].astype(int) >= 70
    town = town.sort_values("d_mdape", ascending=False)
    pooled = s.set_index("unit").loc[["suburbs", "City", "county"]].reset_index()
    rows = [(f"{r.unit[3:]} ({r.n:,})", r, cf.CITY_C if r.city else cf.SUB_C, "o" if r.city else "s") for r in town.itertuples()]
    rows += [(None, None, None, None)]
    rows += [(f"{'County' if r.unit == 'county' else r.unit}, pooled ({r.n:,})", r,
              cf.CITY_C if r.unit == "City" else cf.SUB_C if r.unit == "suburbs" else cf.INK, "D") for r in pooled.itertuples()]
    fig, ax = plt.subplots(figsize=(7.2, 10.5), facecolor=cf.SURFACE)
    ax.set_facecolor(cf.SURFACE)
    labels = []
    for y, (lab, r, c, mk) in enumerate(rows):
        labels.append(lab or "")
        if r is None:
            continue
        ax.plot([r.d_mdape_lo, r.d_mdape_hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r.d_mdape], [y], s=46 if mk == "D" else 34, marker=mk, color=c, edgecolor=cf.SURFACE, linewidth=2, zorder=3)
        if mk == "D" and r.unit in BAR:                # the usefulness bar's limit for this pooled row
            ax.plot([BAR[r.unit]] * 2, [y - 0.35, y + 0.35], color=cf.INK2, lw=1.2, zorder=2)
    ax.axvline(0, color=cf.INK2, lw=1, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8, color=cf.INK2)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="x", color=cf.GRID, lw=0.6)
    ax.tick_params(axis="x", colors=cf.MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(cf.GRID)
    ax.set_xlabel("MdAPE difference, final GL (IXa) minus Assessor (percentage points; below 0 = GL closer to the price)",
                  fontsize=8, color=cf.INK2)
    ax.set_title("Final GL (IXa) minus Assessor, 2025 test sales (Oct 2023 – Dec 2024), 90% intervals",
                 fontsize=8.5, color=cf.INK, loc="left")
    h = [plt.Line2D([], [], color=cf.CITY_C, marker="o", lw=2, label="City of Chicago township"),
         plt.Line2D([], [], color=cf.SUB_C, marker="s", lw=2, label="suburban township"),
         plt.Line2D([], [], color=cf.INK, marker="D", lw=2, label="county, pooled"),
         plt.Line2D([], [], color=cf.INK2, marker="|", lw=0, markersize=10, label="usefulness bar's limit (pooled rows)")]
    ax.legend(handles=h, loc="center right", fontsize=7.5, frameon=False, labelcolor=cf.INK2)
    fig.tight_layout()
    out = REPO / "notes/v2/figures/final_county_mdape_difference.png"
    fig.savefig(out, dpi=160, facecolor=cf.SURFACE)
    print(out)


def evaluate():
    sc.compose_h2(OUT, house="IXah", flat="FCf")
    r = subprocess.run([PY, "-I", str(SCR / "county2025_eval.py"), "--out", str(OUT.relative_to(REPO))], cwd=REPO)
    assert r.returncode == 0
    sc.salehist_tables("IXa", OUT)
    R = extra_tables()
    b = bar()
    (OUT / "d77.json").write_text(json.dumps(dict(bar=b, groups=R.to_dict("records")), indent=1, default=float))
    L = ["# D77 final county test: extra tables", "", "_d77_county.py eval; IXa = the final model; differences in MdAPE "
         "points (IXa minus comparison; negative = IXa better), 90% paired PIN-within-township bootstrap (2,000, seed "
         "20261020, as county2025_eval.py)._", "", "| area | group | n | IXa | Assessor | H2 | P | IXa − Assessor | IXa − H2 | "
         "IXa − P | median ratio IXa / Assessor |", "|---|---|---:|---:|---:|---:|---:|---|---|---|---|"]
    for _, x in R.iterrows():
        L.append(f"| {x.area} | {x.group} | {x.n:,} | {x.mdape_ixa:.2f} | {x.mdape_assessor:.2f} | {x.mdape_h2:.2f} | "
                 f"{x.mdape_p:.2f} | {x.d_vs_assessor:+.2f} [{x.d_vs_assessor_lo:+.2f}, {x.d_vs_assessor_hi:+.2f}] | "
                 f"{x.d_vs_h2:+.2f} [{x.d_vs_h2_lo:+.2f}, {x.d_vs_h2_hi:+.2f}] | {x.d_vs_p:+.2f} [{x.d_vs_p_lo:+.2f}, "
                 f"{x.d_vs_p_hi:+.2f}] | {x.ratio_ixa:.3f} / {x.ratio_assessor:.3f} |")
    L += ["", "## Usefulness bar (D73)", ""]
    for k, v in b.items():
        L.append(f"- {k}: {json.dumps(v, default=float)}")
    (OUT / "d77.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    figure()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "smoke", "run", "eval"])
    a = ap.parse_args()
    dict(prep=prep, smoke=smoke, run=run, eval=evaluate)[a.step]()


if __name__ == "__main__":
    main()
