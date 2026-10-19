"""Median-centred MdAPE as a secondary measure.

centred MdAPE = 100 x median |r / median(r) - 1|, r = value / price (time-adjusted unless the sample says otherwise),
for each model, re-centred in every bootstrap resample; paired difference GL - Assessor. It removes each model's own
overall level from the evaluation sample itself, so it is descriptive; MdAPE stays the headline.

For every ratio study listed below it rebuilds the evaluation table and the 2,000 cluster-bootstrap resamples with
ratio_v2.py's own code and settings (read from the study's JSON: evaluation files, models, sample, rounding, seed,
resamples), checks that the rebuilt resamples reproduce the study's saved MdAPE bootstrap draws (.boot.npz) exactly,
and then computes the centred measure overall and per township.

Usage (repo root): .venv/bin/python -I public/scripts/v2/p7b_centred.py
Reads public/runs/v2/p7_eval/ratio_{pilots,confirmation,all}_{primary,rounded}.json and
public/runs/v2/p7b_eval/ratio_*_{primary,rounded}.json; writes public/runs/v2/p7b_eval/centred.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def mdape(v, p):
    return 100 * np.median(np.abs(v - p) / p)


def centred(v, p):
    r = v / p
    return 100 * np.median(np.abs(r / np.median(r) - 1))


def rebuild(cfg):
    """ratio_v2.py's evaluation table, model values and resamples (same code path, same seed)."""
    ev = pd.concat([pd.read_parquet(REPO / p).assign(town=Path(p).parent.name.lstrip("t")) for p in cfg["evals"]],
                   ignore_index=True)
    if cfg["sample"] == "q1_unadjusted":
        ev = ev[ev["q1"]].copy()
    ev["p"] = ev["adj_price"] if cfg["sample"] == "primary" else ev["price"]
    V = {}
    for m in cfg["models"]:
        name, path = m.split("=", 1)
        if name == "assessor":
            v = pd.read_parquet(REPO / path, columns=["meta_pin", "pred_pin_final_fmv_round"]).drop_duplicates("meta_pin")
            v = v.set_index("meta_pin")["pred_pin_final_fmv_round"]
        else:
            v = pd.read_parquet(REPO / path).set_index("meta_pin")["value_rounded" if cfg["rounded"] else "value"]
        ev[f"v:{name}"] = ev["meta_pin"].map(v).astype(float)
        assert ev[f"v:{name}"].notna().all()
        V[name] = None
    ev = ev.reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    strata = []
    for _, g in ev.groupby("town"):
        strata.append([ix.to_numpy() for _, ix in g.groupby("group_id").groups.items()])
    boots = []
    for _ in range(cfg["boot"]):
        idx = []
        for members in strata:
            pick = rng.integers(0, len(members), len(members))
            idx.append(np.concatenate([members[i] for i in pick]))
        boots.append(np.concatenate(idx))
    P = ev["p"].to_numpy()
    V = {n: ev[f"v:{n}"].to_numpy() for n in V}
    return ev, P, V, boots


def table(mask, P, V, boots, f):
    sub = np.flatnonzero(mask)
    point = {n: float(f(V[n][sub], P[sub])) for n in V}
    bs = {n: [] for n in V}
    for b in boots:
        bi = b[mask[b]]
        if len(bi) < 2:
            continue
        for n in V:
            bs[n].append(f(V[n][bi], P[bi]))
    out = dict(n=int(len(sub)), models={n: dict(point=point[n], lo90=float(np.quantile(bs[n], 0.05)),
                                                 hi90=float(np.quantile(bs[n], 0.95))) for n in V})
    d = np.array(bs["gl"]) - np.array(bs["assessor"])
    out["gl-assessor"] = dict(point=point["gl"] - point["assessor"], lo90=float(np.quantile(d, 0.05)),
                              hi90=float(np.quantile(d, 0.95)))
    return out, bs


def main():
    studies = {}
    for s in ["pilots", "confirmation", "all"]:
        for smp in ["primary", "rounded"]:
            studies[f"seed1_{s}_{smp}"] = REPO / f"public/runs/v2/p7_eval/ratio_{s}_{smp}.json"
    for p in sorted((REPO / "public/runs/v2/p7b_eval").glob("ratio_*.json")):
        studies[p.stem.removeprefix("ratio_")] = p
    out = {}
    for key, path in studies.items():
        cfg = json.loads(path.read_text())
        ev, P, V, boots = rebuild(cfg)
        allm = np.ones(len(ev), bool)
        # Check: the rebuilt resamples reproduce ratio_v2's saved MdAPE bootstrap draws exactly.
        _, bs = table(allm, P, V, boots, mdape)
        saved = np.load(path.with_suffix(".boot.npz"))
        for n in V:
            assert np.array_equal(np.array(bs[n]), saved[f"{n}:mdape"]), f"{key}: resamples differ ({n})"
        assert abs(cfg["overall"]["pairs"]["gl-assessor"]["mdape"]["point"] - (mdape(V["gl"], P) - mdape(V["assessor"], P))) < 1e-12
        res = dict(source=str(path.relative_to(REPO)), resamples_checked=True)
        res["overall"], _ = table(allm, P, V, boots, centred)
        if ev["town"].nunique() > 1:
            res["by_town"] = {t: table((ev["town"] == t).to_numpy(), P, V, boots, centred)[0]
                              for t in sorted(ev["town"].unique())}
        out[key] = res
        o = res["overall"]
        print(f"{key}: centred MdAPE gl {o['models']['gl']['point']:.2f} assessor {o['models']['assessor']['point']:.2f} "
              f"diff {o['gl-assessor']['point']:+.2f} [{o['gl-assessor']['lo90']:+.2f}, {o['gl-assessor']['hi90']:+.2f}]",
              flush=True)
        (REPO / "public/runs/v2/p7b_eval/centred.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
