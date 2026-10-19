"""Fit models S and/or P on all 30 township-years and value them (P: the published model; S: the simpler
one-block-surface model, not used).

For each township-year, in the order below (Thornton 37 first as the runtime pilot, then the 2023 south triad, then
the 2022 north triad): fit_struct.py (4 chains, 1,000 + 1,000, mode start, the given seed) -> value_v2.py (2023) or
ay2022/value_v2.py (2022), both unchanged. All S jobs are queued before all P jobs. At most 3 fits at a time, every
step under `nice -n 15`; valuations one at a time.

Restart-safe: a run with a report.json holding a `valuation` block is skipped; a fit with report.json but no
valuation is only valued (so earlier check fits on Orland, struct_S_t28 / struct_P_t28, are reused). A fit that fails
mechanically (non-zero exit) is rerun once with seed + 1 under <run>_reseed; poor diagnostics are not a reason to rerun.

Usage (repo root): nohup caffeinate -is .venv/bin/python -I public/scripts/v2/struct_run.py --models S P \
                   > struct_driver.log 2>&1 < /dev/null &
       --seed 2 --suffix _seed2: the second-seed check on the chosen model.
Writes public/runs/v2/struct_eval/runs<suffix>.json: "<model> <year> <T>" -> prep, run, seed, status, runtime, R-hat.
"""
import argparse
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
OUT = RUNS / "struct_eval"
SOUTH = ["37", "27", "30", "15", "32", "21", "12", "34", "31", "39", "11", "13", "14", "19", "28", "33", "36"]
NORTH = ["10", "16", "17", "18", "20", "22", "23", "24", "25", "26", "29", "35", "38"]
MAX_PARALLEL = 3

value_lock = threading.Lock()
status_lock = threading.Lock()
status = {}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def jobs(models):
    for m in models:
        for T in SOUTH:
            yield m, "2023", T, "public/data/v2/prep_c8fix" if T == "28" else "public/data/v2/prep_p7", ""
        for T in NORTH:
            yield m, "2022", T, "public/data/v2/prep_p8", "_ay2022"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=["S", "P"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spath = OUT / f"runs{a.suffix}.json"
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

    def one(model, year, T, prep, tag):
        key = f"{model} {year} {T}"
        save(key, model=model, year=year, town=T, prep=prep)
        value = SCR / ("value_v2.py" if year == "2023" else "ay2022/value_v2.py")
        for seed, rn in [(a.seed, f"struct_{model}_t{T}{tag}{a.suffix}"), (a.seed + 1, f"struct_{model}_t{T}{tag}{a.suffix}_reseed")]:
            rd = RUNS / rn
            rep = rd / "report.json"
            if not rep.exists():
                log(f"{key}: fit {rn} (seed {seed})")
                save(key, status=f"fitting {rn}")
                rc = run([PY, "-I", str(SCR / "fit_struct.py"), "--town", T, "--model", model, "--prep", prep,
                          "--seed", str(seed), "--name", rn], LOGS / f"struct_fit_{rn}.log")
                if rc or not rep.exists():
                    save(key, failures=status[key].get("failures", []) + [dict(run=rn, seed=seed, returncode=rc)])
                    log(f"{key}: FIT FAILED ({rn}, exit {rc})")
                    continue
            save(key, run=f"public/runs/v2/{rn}", seed=seed)
            if "valuation" not in json.loads(rep.read_text()):
                with value_lock:
                    log(f"{key}: value {rn}")
                    if run([PY, "-I", str(value), str(rd), "--prep", prep], LOGS / f"struct_value_{rn}.log"):
                        save(key, status="valuation failed")
                        log(f"{key}: VALUE FAILED")
                        return
            r = json.loads(rep.read_text())
            d = r["diagnostics"]
            save(key, status="done", runtime_s=r["timing"]["runtime_s"], max_rhat=d["max_rhat"],
                 min_ess_bulk=d["min_ess_bulk"], min_ess_tail=d["min_ess_tail"], divergent=d["divergent"],
                 ebfmi_min=min(d["ebfmi"]), sum_to_zero_pass=r["sum_to_zero"]["all_pass"],
                 value_rhat_max=r["valuation"]["latent_split_rhat"]["max"])
            log(f"{key}: done ({rn}; {r['timing']['runtime_s'] / 60:.1f} min; max R-hat {d['max_rhat']:.3f}; "
                f"sum-to-zero {'pass' if r['sum_to_zero']['all_pass'] else 'FAIL'})")
            return
        save(key, status="fit failed twice")

    todo = list(jobs(a.models))
    log(f"struct driver: {len(todo)} fits (models {a.models}, seed {a.seed}), at most {MAX_PARALLEL} at a time")
    with ThreadPoolExecutor(MAX_PARALLEL) as ex:
        for f in [ex.submit(one, *j) for j in todo]:
            f.result()
    save()
    bad = [k for k, s in status.items() if s.get("status") != "done"]
    log(f"struct driver finished; not done: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
