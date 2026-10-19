"""Fit the earlier block GL model (or the house-effect comparator) to one prepared fit region.

Model: public/stan/v2/gl_v2.stan (block units) or gl_v2_house.stan (one effect per house on a kNN-6 house
graph within each neighbourhood; engineering comparator only). Data: the output of prep_v2.py.
Training months 2014-01 .. 2022-12 (Nt = 108). Regressors: prep's coded columns, centred on the training means,
through a thin QR decomposition. Spatial prior: sparse scaled ICAR with per-component sum-to-zero or, with
--dense, the dense whitened equivalent (a check).

Methods (--method):
  nuts        NUTS (the comparison fit). Chains start at the posterior mode (L-BFGS optimisation in the
              unconstrained space with the Jacobian, "mode-initialised"); --init random for Stan's random
              inits (except intercept and theta, see `base` below).
  optimize    the mode only (one point).
  laplace     the mode plus 1,000 draws from its Laplace (normal) approximation.
  pathfinder  multi-path Pathfinder (4 paths), 1,000 draws.
Wall-clock is recorded per stage (compile excluded; mode finding and sampling included in `runtime_s`).

Writes public/runs/v2/<name>/: report.json (settings, runtimes, diagnostics), draws.npz (per draw: intercept, beta,
mu, zeta, psi at the last training month, the spatial effect u in prep order, scalars), design.npz (what
value_v2.py needs to value every valued PIN at 1 January 2023: the last training month, December 2022, carried
flat), summary.csv (NUTS), chains/ (CmdStan CSV).

Usage (repo root): .venv/bin/python -I public/scripts/v2/fit_v2.py --town 28 --variant block --method nuts
                   [--chains 4 --warmup 1000 --samples 1000 --thin 1 --seed 1 --dense --init mode|random --name NAME]
"""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
STAN = {"block": REPO / "public/stan/v2/gl_v2.stan", "house": REPO / "public/stan/v2/gl_v2_house.stan"}
NT = 108
EPS_EIG = 1e-9
SCALARS = ["intercept", "sigma_mu", "sigma_zeta", "sigma_u", "sigma_psi", "sigEps", "nu"]


def laplacian(n, a, b):
    L = np.zeros((n, n))
    np.add.at(L, (a, b), -1.0)
    np.add.at(L, (b, a), -1.0)
    np.fill_diagonal(L, 0.0)
    L[np.diag_indices(n)] = -L.sum(axis=1)
    return L


def whitener(L):
    """A = U Lambda^{-1/2} over eigenvectors with non-zero eigenvalue (as in the GL paper): A A' = pinv(L); the null space
    (one constant per connected component) is removed, so the effect sums to zero in every component."""
    lam, U = np.linalg.eigh(L)
    keep = lam > EPS_EIG * max(1.0, lam.max())
    return U[:, keep] / np.sqrt(lam[keep])


def spatial_structure(ns, a, b, comp, dense):
    """Order nodes so components are contiguous; per-component scaling factor s_c = geometric mean of the
    diagonal of pinv(L_c); for --dense, the block-diagonal scaled whitener. Returns perm (new -> old), and the
    Stan inputs in the new order."""
    perm = np.lexsort((np.arange(ns), comp))
    inv = np.empty(ns, int)
    inv[perm] = np.arange(ns)
    a2, b2, comp2 = inv[a], inv[b], comp[perm]
    comps, start, size = np.unique(comp2, return_index=True, return_counts=True)
    inv_sqrt_scale = np.ones(ns)
    blocks, scales = [], []
    for s0, n in zip(start, size):
        if n == 1:
            blocks.append(np.ones((1, 1)))
            scales.append(1.0)
            continue
        m = (a2 >= s0) & (a2 < s0 + n)
        Lc = laplacian(n, a2[m] - s0, b2[m] - s0)
        lam, U = np.linalg.eigh(Lc)
        keep = lam > EPS_EIG * lam.max()
        assert (~keep).sum() == 1, "a component's graph must be connected"
        Ak = U[:, keep] / np.sqrt(lam[keep])
        sc = float(np.exp(np.mean(np.log(np.einsum("ij,ij->i", Ak, Ak)))))
        scales.append(sc)
        inv_sqrt_scale[s0:s0 + n] = 1 / np.sqrt(sc)
        if dense:
            blocks.append(Ak / np.sqrt(sc))
    A_u = None
    if dense:
        A_u = np.zeros((ns, sum(bk.shape[1] for bk in blocks)))
        r = c = 0
        for bk in blocks:
            A_u[r:r + bk.shape[0], c:c + bk.shape[1]] = bk
            r, c = r + bk.shape[0], c + bk.shape[1]
    st = dict(N_edges=len(a2), node1=(a2 + 1).tolist(), node2=(b2 + 1).tolist(), N_sp_comp=len(comps),
              sp_comp_start=(start + 1).tolist(), sp_comp_size=size.tolist(), inv_sqrt_scale=inv_sqrt_scale)
    return perm, inv, st, A_u, np.array(scales), size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", required=True)
    ap.add_argument("--variant", choices=["block", "house"], default="block")
    ap.add_argument("--method", choices=["nuts", "optimize", "laplace", "pathfinder"], default="nuts")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dense", action="store_true", help="dense whitened spatial prior (the D25 check)")
    ap.add_argument("--init", choices=["mode", "random"], default="mode")
    ap.add_argument("--prep", default="public/data/v2/prep")
    ap.add_argument("--runs", default="public/runs/v2")
    ap.add_argument("--name")
    ap.add_argument("--dump", help="write the Stan data (JSON) to this file and stop (debugging)")
    a = ap.parse_args()
    prep = Path(a.prep) / f"t{a.town}"
    name = a.name or f"gl_{a.variant}_t{a.town}_{a.method}" + ("_dense" if a.dense else "")
    run = Path(a.runs) / name
    if (run / "report.json").exists():
        raise SystemExit(f"{run} already has report.json; choose another --name or move it aside")
    if run.exists():
        run.rename(run.with_name(run.name + f".partial-{time.strftime('%Y%m%d%H%M%S')}"))
    run.mkdir(parents=True)
    print(f"=== fit_v2 {name}: town {a.town}, variant {a.variant}, method {a.method}, dense {a.dense}", flush=True)

    t = pd.read_parquet(prep / "train.parquet")
    r = pd.read_parquet(prep / "roll.parquet")
    coding = json.loads((prep / "coding.json").read_text())
    xcols = ["x:" + c for c in coding["columns"]]

    # ------------------------------------------------------------ spatial nodes
    if a.variant == "block":
        units = pd.read_parquet(prep / "units.parquet")
        E = pd.read_parquet(prep / "edges.parquet")
        ns, comp = len(units), units["comp"].to_numpy()
        node_t, node_r = t["node"].to_numpy(), r["node"].to_numpy()
    else:
        h = pd.read_parquet(prep / "houses.parquet")
        E = pd.read_parquet(prep / "house_edges.parquet")
        ns = len(h)
        comp = pd.factorize(h["nbhd_id"], sort=True)[0]
        hid = dict(zip(h["meta_pin"], h["house_idx"]))
        node_t, node_r = t["meta_pin"].map(hid).to_numpy(), r["meta_pin"].map(hid).to_numpy()
    assert not (pd.isna(node_t).any() or pd.isna(node_r).any())
    t0 = time.time()
    perm, inv, st, A_u, scales, sizes = spatial_structure(ns, E["a"].to_numpy(), E["b"].to_numpy(), comp, a.dense)
    t_struct = time.time() - t0
    print(f"spatial: {ns:,} nodes, {len(E):,} edges, {len(sizes)} components (largest {sizes.max()}, "
          f"{(sizes == 1).sum()} singletons); scaling factors {scales.min():.3f}-{scales.max():.3f}; {t_struct:.1f} s",
          flush=True)

    # ------------------------------------------------------------ neighbourhoods and time
    nbhds = sorted(pd.read_parquet(prep / "units.parquet")["nbhd_id"].unique())
    ne = pd.read_parquet(prep / "nbhd_edges.parquet")
    kid = {k: i for i, k in enumerate(nbhds)}
    L_comp = laplacian(len(nbhds), ne["nbhd_a"].map(kid).to_numpy(), ne["nbhd_b"].map(kid).to_numpy())
    A_comp = whitener(L_comp)
    Dt = np.eye(NT)[1:] - np.eye(NT)[:-1]
    A_t = whitener(Dt.T @ Dt)

    # ------------------------------------------------------------ regressors through QR
    X = t[xcols].to_numpy()
    xmean = X.mean(axis=0)
    Qm, Rm = np.linalg.qr(X - xmean)
    sq = np.sqrt(len(t) - 1)
    Qs, Rs = Qm * sq, Rm / sq
    R_inv = np.linalg.inv(Rs)

    data = dict(N=len(t), Nt=NT, K=len(nbhds), Ns=ns, Nr=len(xcols),
                sell=t["month"].astype(int).tolist(), nbhd=t["nbhd_id"].map(kid).add(1).astype(int).tolist(),
                node=(inv[node_t.astype(int)] + 1).tolist(), yVar=t["log_price"].to_numpy(), Q=Qs, R_inv=R_inv,
                N_time_latent=A_t.shape[1], A_t=A_t, N_comp_latent=A_comp.shape[1], A_comp=A_comp,
                dense=int(a.dense), N_sp_latent=(A_u.shape[1] if a.dense else 0),
                A_u=(A_u if a.dense else np.zeros((0, 0))), y_init=float(t["log_price"].mean()), **st)

    import cmdstanpy
    if a.dump:
        cmdstanpy.write_stan_json(a.dump, data)
        shutil.rmtree(run)
        return
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN[a.variant]))
    out_dir = run / "chains"
    timing = dict(spatial_structure_s=round(t_struct, 1))
    report = dict(name=name, town=a.town, variant=a.variant, method=a.method, dense=a.dense, init=a.init,
                  seed=a.seed, settings=dict(chains=a.chains, warmup=a.warmup, samples=a.samples, thin=a.thin), n_train=len(t), n_valued=len(r), n_spatial_nodes=ns, n_edges=len(E),
                  n_components=len(sizes), n_nbhds=len(nbhds), n_regressors=len(xcols),
                  prep_manifest=json.loads((prep / "prep_manifest.json").read_text())["outputs"],
                  coding_sha256=json.loads((prep / "prep_manifest.json").read_text())["coding_sha256"],
                  stan_file=str(STAN[a.variant].relative_to(REPO)),
                  versions=dict(cmdstan=cmdstanpy.cmdstan_version(), cmdstanpy=cmdstanpy.__version__,
                                python=platform.python_version()))

    # Start values: Stan's random inits put theta anywhere in (-2, 2), which R_inv maps to absurd coefficients
    # (columns with little variation have small R), so every method starts at intercept = mean log price,
    # theta = 0, and moderate scales; everything else at Stan's random inits.
    base = dict(intercept=data["y_init"], theta=np.zeros(len(xcols)).tolist(), sigEps=0.3, nu=5.0, sigma_mu=0.05,
                sigma_zeta=0.2, sigma_u=0.2, sigma_psi=0.05)
    t_all = time.time()
    mode = None
    if a.method in ("optimize", "laplace") or (a.method == "nuts" and a.init == "mode"):
        t0 = time.time()
        mode = model.optimize(data=data, jacobian=True, seed=a.seed, output_dir=str(out_dir / "mode"),
                              iter=20000, inits=base, show_console=False)
        timing["optimize_s"] = round(time.time() - t0, 1)
        print(f"mode found in {timing['optimize_s']} s", flush=True)
    if a.method == "nuts":
        inits = base
        if mode is not None:
            mv = mode.stan_variables()
            inits = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in mv.items() if k != "beta"}
        t0 = time.time()
        fit = model.sample(data=data, chains=a.chains, parallel_chains=a.chains, iter_warmup=a.warmup,
                           iter_sampling=a.samples, thin=a.thin, seed=a.seed, inits=inits, output_dir=str(out_dir),
                           show_progress=False)
        timing["sample_s"] = round(time.time() - t0, 1)
        print(f"sampled in {timing['sample_s'] / 60:.1f} min", flush=True)
    elif a.method == "laplace":
        t0 = time.time()
        fit = model.laplace_sample(data=data, mode=mode, jacobian=True, draws=1000, seed=a.seed,
                                   output_dir=str(out_dir / "laplace"))
        timing["laplace_s"] = round(time.time() - t0, 1)
    elif a.method == "pathfinder":
        t0 = time.time()
        fit = model.pathfinder(data=data, num_paths=4, draws=1000, seed=a.seed, output_dir=str(out_dir / "pathfinder"),
                               inits=base, show_console=False)
        timing["pathfinder_s"] = round(time.time() - t0, 1)
    else:
        fit = mode
    timing["runtime_s"] = round(time.time() - t_all, 1)
    report["timing"] = timing
    print(f"runtime {timing['runtime_s'] / 60:.1f} min (compile excluded)", flush=True)

    # ------------------------------------------------------------ draws needed for valuation
    v = fit.stan_variables()
    def arr(k):
        x = np.asarray(v[k], dtype=float)
        return x[None, ...] if a.method == "optimize" else x
    mu = arr("mu_raw") @ A_t.T * arr("sigma_mu")[:, None]
    zeta = arr("zeta_raw") @ A_comp.T * arr("sigma_zeta")[:, None]
    psi_last = np.einsum("kl,dlm,m->dk", A_comp, arr("psi_raw"), A_t[-1]) * arr("sigma_psi")[:, None]
    phi = arr("phi")
    u_new = (phi @ A_u.T if a.dense else phi * st["inv_sqrt_scale"]) * arr("sigma_u")[:, None]
    u = u_new[:, inv]                                   # back to prep order
    beta = arr("theta") @ R_inv.T
    np.savez_compressed(run / "draws.npz", intercept=arr("intercept"), beta=beta, mu=mu, zeta=zeta,
                        psi_last=psi_last, u=u, **{k: arr(k) for k in SCALARS if k != "intercept"})
    np.savez_compressed(run / "design.npz", pins=r["meta_pin"].to_numpy(str), X_c=r[xcols].to_numpy() - xmean,
                        nbhd=r["nbhd_id"].map(kid).to_numpy(), node=node_r.astype(int), xcols=np.array(xcols),
                        nbhds=np.array(nbhds))
    report["n_draws"] = int(beta.shape[0])
    report["scalars"] = {k: dict(mean=float(arr(k).mean()), sd=float(arr(k).std())) for k in SCALARS}

    if a.method == "nuts":
        t0 = time.time()
        summ = fit.summary(sig_figs=4)
        summ.to_csv(run / "summary.csv")
        sv = fit.method_variables()
        body = summ.loc[[i for i in summ.index if not i.startswith("lp__")]]
        report["diagnostics"] = dict(
            max_rhat=float(body["R_hat"].max()), min_ess_bulk=float(body["ESS_bulk"].min()),
            min_ess_tail=float(body["ESS_tail"].min()), divergent=int(sv["divergent__"].sum()),
            max_treedepth_share=float((sv["treedepth__"] >= 10).mean()), mean_treedepth=float(sv["treedepth__"].mean()),
            stepsize=[float(x) for x in np.unique(np.round(sv["stepsize__"], 5))],
            scalar_rhat={k: float(summ.loc[k, "R_hat"]) for k in SCALARS},
            scalar_ess_bulk={k: float(summ.loc[k, "ESS_bulk"]) for k in SCALARS})
        (run / "diagnose.txt").write_text(fit.diagnose())
        timing["summary_s"] = round(time.time() - t0, 1)
        print(json.dumps(report["diagnostics"], indent=1), flush=True)
    elif a.method == "pathfinder":
        report["diagnostics"] = dict(note="Pathfinder: no convergence diagnostics; draws are approximate")
    (run / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["scalars"], indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
