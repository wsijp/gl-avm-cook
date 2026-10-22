"""Sale-prediction test: fit model P on all 38 townships and predict their test sales.

Order: Lake (72) first (the runtime check below), then the other townships by number of training sales,
largest first. For each: county2025_fit.py (fit_struct.py with 94 months; 4 chains, 1,000 + 1,000, mode start, seed 1)
-> county2025_predict.py. At most 3 fits at a time, every step under `nice -n 15`; predictions one at a time.
Restart-safe as struct_run.py: a run with a `prediction` block is skipped; a fit with report.json but no prediction is
only predicted. A fit that fails mechanically (non-zero exit) is rerun once with seed 2 under <run>_reseed; poor
diagnostics are not a reason to rerun.

Runtime rule (fixed before fitting): no fit is launched while public/runs/v2/county2025_eval/STOP exists. The driver writes it
(1) if Lake has not finished (prediction included) 9 h after its launch, or (2) when Lake finishes, if the elapsed
time so far + (Lake's full elapsed time x the training sales of the townships not yet launched / Lake's) / 3 exceeds
24 h. Running fits finish. Before each launch the driver waits until macOS reports at least 25% of memory free
(8 GB of 32; `memory_pressure`).

Usage (repo root): nohup caffeinate -is .venv/bin/python -I public/scripts/v2/county2025_run.py \
                   > notes/v2/logs/county2025_driver.log 2>&1 < /dev/null &
Options (the defaults are the published test's; other values were used for further tests not in this repository):
--prep DIR, --prefix NAME (runs <prefix>_t<T>),
--out DIR (runs.json and STOP), --towns T ... (a subset, e.g. a fallback's townships).
Writes public/runs/v2/county2025_eval/runs.json: town -> run, seed, status, runtime, diagnostics.
"""
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / ".venv/bin/python")
SCR = REPO / "public/scripts/v2"
RUNS = REPO / "public/runs/v2"
LOGS = REPO / "notes/v2/logs"
OUT = RUNS / "county2025_eval"
PREP = "public/data/v2/prep_county2025"
MAX_PARALLEL = 3
SEED = 1
STOP = OUT / "STOP"
LAKE_MAX_H, BUDGET_H, MIN_FREE = 9.0, 24.0, 25
T_START = time.time()


def free_percent():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    line = [x for x in out.splitlines() if "free percentage" in x]
    return int(line[-1].split(":")[-1].strip().rstrip("%")) if line else 100

pred_lock = threading.Lock()
status_lock = threading.Lock()
status = {}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def order():
    s = json.loads((REPO / PREP / "split_summary.json").read_text())
    towns = sorted(s["by_region"], key=lambda T: -s["by_region"][T]["train"])
    if TOWNS:
        return [T for T in towns if T in TOWNS]
    return ["72"] + [T for T in towns if T != "72"]


PREFIX = "county2025_P"
TOWNS = None


def options():
    global PREP, OUT, STOP, PREFIX, TOWNS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=PREP)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--out", default=str(OUT.relative_to(REPO)))
    ap.add_argument("--towns", nargs="*")
    a = ap.parse_args()
    PREP, PREFIX, TOWNS = a.prep, a.prefix, a.towns
    OUT = REPO / a.out
    STOP = OUT / "STOP"


def main():
    options()
    OUT.mkdir(parents=True, exist_ok=True)
    spath = OUT / "runs.json"
    if spath.exists():
        status.update(json.loads(spath.read_text()))

    def save(key=None, **kw):
        with status_lock:
            if key is not None:
                status.setdefault(key, {}).update(kw)
            spath.write_text(json.dumps(status, indent=1))

    def run(cmd, logfile):
        with open(logfile, "a") as f:
            f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            f.flush()
            return subprocess.run(["nice", "-n", "15"] + cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL).returncode

    split = json.loads((REPO / PREP / "split_summary.json").read_text())["by_region"]
    launched = set()

    def stop(reason):
        if not STOP.exists():
            STOP.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}\n")
            log(f"STOP: {reason}")

    def lake_watch():
        while True:
            time.sleep(60)
            s = status.get("72", {})
            if s.get("status") == "done" or "launched_at" not in s:
                if s.get("status") == "done":
                    return
                continue
            if time.time() - s["launched_at"] > LAKE_MAX_H * 3600:
                stop(f"Lake not finished {LAKE_MAX_H} h after launch: no further fits launched; report to Will")
                return

    def one(T):
        save(T, town=T, prep=PREP)
        t_launch = None
        for seed, rn in [(SEED, f"{PREFIX}_t{T}"), (SEED + 1, f"{PREFIX}_t{T}_reseed")]:
            rd = RUNS / rn
            rep = rd / "report.json"
            if not rep.exists():
                while free_percent() < MIN_FREE and not STOP.exists():
                    log(f"t{T}: waiting for memory ({free_percent()}% free < {MIN_FREE}%)")
                    time.sleep(120)
                if STOP.exists():
                    save(T, status="not launched (STOP)")
                    log(f"t{T}: not launched (STOP)")
                    return
                launched.add(T)
                t_launch = t_launch or time.time()
                log(f"t{T}: fit {rn} (seed {seed})")
                save(T, status=f"fitting {rn}", started=time.strftime("%Y-%m-%d %H:%M:%S"), launched_at=t_launch)
                rc = run([PY, "-I", str(SCR / "county2025_fit.py"), "--town", T, "--model", "P", "--prep", PREP,
                          "--seed", str(seed), "--name", rn], LOGS / f"county2025_fit_{rn}.log")
                if rc or not rep.exists():
                    save(T, failures=status[T].get("failures", []) + [dict(run=rn, seed=seed, returncode=rc)])
                    log(f"t{T}: FIT FAILED ({rn}, exit {rc})")
                    continue
            save(T, run=f"public/runs/v2/{rn}", seed=seed)
            if "prediction" not in json.loads(rep.read_text()):
                with pred_lock:
                    log(f"t{T}: predict {rn}")
                    if run([PY, "-I", str(SCR / "county2025_predict.py"), str(rd), "--prep", PREP],
                           LOGS / f"county2025_predict_{rn}.log"):
                        save(T, status="prediction failed")
                        log(f"t{T}: PREDICTION FAILED")
                        return
            r = json.loads(rep.read_text())
            d = r["diagnostics"]
            save(T, status="done", runtime_s=r["timing"]["runtime_s"], max_rhat=d["max_rhat"],
                 min_ess_bulk=d["min_ess_bulk"], min_ess_tail=d["min_ess_tail"], divergent=d["divergent"],
                 ebfmi_min=min(d["ebfmi"]), max_treedepth_share=d["max_treedepth_share"],
                 sum_to_zero_pass=r["sum_to_zero"]["all_pass"], pred_rhat_max=r["prediction"]["latent_split_rhat"]["max"],
                 n_train=r["n_train"], finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                 elapsed_s=round(time.time() - t_launch, 1) if t_launch else None)
            log(f"t{T}: done ({rn}; {r['timing']['runtime_s'] / 60:.1f} min fit; "
                f"{(time.time() - t_launch) / 60 if t_launch else float('nan'):.1f} min elapsed; max R-hat {d['max_rhat']:.3f}; "
                f"sum-to-zero {'pass' if r['sum_to_zero']['all_pass'] else 'FAIL'})")
            if T == "72" and t_launch:
                lake = time.time() - t_launch
                rest = sum(split[x]["train"] for x in split if x not in launched)
                proj = (time.time() - T_START + lake * rest / split["72"]["train"] / MAX_PARALLEL) / 3600
                save(T, projected_total_h=round(proj, 2))
                log(f"runtime rule: Lake {lake / 3600:.2f} h; {rest:,} training sales not yet launched; "
                    f"projected total {proj:.1f} h (budget {BUDGET_H} h)")
                if proj > BUDGET_H:
                    stop(f"projected total {proj:.1f} h exceeds {BUDGET_H} h: no further fits launched; report to Will")
            return
        save(T, status="fit failed twice")

    todo = order()
    log(f"county2025 driver: {len(todo)} townships, order {todo}, at most {MAX_PARALLEL} at a time")
    threading.Thread(target=lake_watch, daemon=True).start()
    with ThreadPoolExecutor(MAX_PARALLEL) as ex:
        for f in [ex.submit(one, T) for T in todo]:
            f.result()
    save()
    bad = [k for k, s in status.items() if s.get("status") != "done"]
    log(f"county2025 driver finished; not done: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
