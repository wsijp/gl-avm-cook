"""Fit model S or P to one prepared fit region.

P is the published model (neighbourhood effect, block effects summing to zero within each neighbourhood,
neighbourhood x month trend); S is the simpler one-block-surface model (not used in the results).

Model: public/stan/v2/gl_struct.stan. Every location and time effect is the GL paper's whitened construction:
A = U Lambda^{-1/2} over the eigenvectors of a graph Laplacian with non-zero eigenvalue, exactly one null dimension per whitened graph (asserted),
so each effect sums to zero exactly over its graph and the overall level sits in the intercept.
  S  block field u over the township graph (all rook edges, the prep's edges.parquet); pieces joined by MST bridges.
  P  zeta over the neighbourhood graph (prep's nbhd_edges.parquet; pieces joined by MST bridges); u over one graph
     per neighbourhood (rook edges with both ends in it; fragments joined by MST bridges), block-diagonal whitener;
     psi = A_comp psi_raw A_t' (Kronecker product of the neighbourhood and time graphs).
Bridges (as in the GL paper's fitting code): complete graph over the pieces, weight = smallest distance between the TIGER
2020 block polygons of any unit of one piece and any unit of the other (EPSG:3435; a split block x neighbourhood
portion takes its parent block's polygon); its minimum spanning tree; one binary edge per tree edge between that
nearest unit pair (ties: lowest node numbers). For the neighbourhood graph the bridge joins the two units'
neighbourhoods. The block whitener is scaled per whitened graph to unit geometric-mean marginal variance;
A_comp and A_t are unscaled, as in fit_v2.py and the paper.
Only units holding a training sale enter Stan (rows of A); u at every unit is A z from the draws (the exact joint
prior; graph interpolation). S passes A dense, P as CSR.

Everything else as fit_v2.py (data, QR, NT = 108 months, start values, mode-initialised NUTS). Writes
public/runs/v2/<name>/ in fit_v2.py's layout, so value_v2.py runs unchanged (S writes zeta = psi_last = 0):
report.json (settings, runtimes, graph, sum-to-zero checks, rank-normalised diagnostics of every component),
draws.npz, design.npz, bridges.csv, diag.npz (per-quantity R-hat and ESS), chains/.

Usage (repo root): .venv/bin/python -I public/scripts/v2/fit_struct.py --town 28 --model S --prep public/data/v2/prep_c8fix
                   [--chains 4 --warmup 1000 --samples 1000 --seed 1 --name NAME --dump FILE]
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcmc_diag as md  # noqa: E402
from fit_v2 import NT, laplacian  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
STAN = REPO / "public/stan/v2/gl_struct.stan"
TIGER = REPO / "notes/v2/explore_blocks/tl_2020_17_tabblock20.zip"
EPS_EIG = 1e-9
FT = 1200 / 3937                                       # EPSG:3435 is in US survey feet; metres per foot


def whiten(L, scale):
    """A = U Lambda^{-1/2} over the non-null eigenvectors; exactly one null dimension asserted (a connected graph).
    With scale, divided by sqrt(s), s = geometric mean of diag(A A'). A one-node graph gives (1, 0)."""
    n = L.shape[0]
    if n == 1:
        return np.zeros((1, 0)), 1.0
    lam, U = np.linalg.eigh(L)
    keep = lam > EPS_EIG * max(1.0, lam.max())
    assert (~keep).sum() == 1, f"whitened graph must be connected ({(~keep).sum()} null dimensions)"
    A = U[:, keep] / np.sqrt(lam[keep])
    s = float(np.exp(np.mean(np.log(np.einsum("ij,ij->i", A, A))))) if scale else 1.0
    return A / np.sqrt(s), s


def pieces_of(nodes, edges):
    """connected pieces of the graph (nodes, edges), each a sorted node list, ordered by first node"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0])


def mst_bridges(pieces, geom):
    """pieces: lists of unit nodes; geom: node -> shapely polygon. As in the GL paper's fitting code: complete graph over the
    pieces weighted by the smallest polygon distance between their units, its minimum spanning tree, one bridge per
    tree edge between that nearest unit pair. Returns [(a, b, distance_m)] (empty for one piece)."""
    from shapely import STRtree
    if len(pieces) == 1:
        return []
    F = nx.Graph()
    for i in range(len(pieces)):
        gi = [geom[n] for n in pieces[i]]
        for j in range(i + 1, len(pieces)):
            gj = [geom[n] for n in pieces[j]]
            (qi, qj), d = STRtree(gj).query_nearest(gi, return_distance=True, all_matches=True)
            best = min(zip(np.round(d, 6), [pieces[i][k] for k in qi], [pieces[j][k] for k in qj]))
            F.add_edge(i, j, weight=float(best[0]), pair=(best[1], best[2]))
    # Kruskal on edges sorted by weight, ties in insertion order (piece numbers): deterministic
    T = nx.minimum_spanning_tree(F, algorithm="kruskal")
    out = [(e["pair"][0], e["pair"][1], e["weight"] * FT) for _, _, e in sorted(T.edges(data=True), key=lambda x: (x[0], x[1]))]
    assert len(out) == len(pieces) - 1
    return out


def unit_geometry(units, prep_manifest):
    """node -> TIGER 2020 block polygon (EPSG:3435) via the year's blocks_<year>_units.parquet (prep input)."""
    import geopandas as gpd
    upath = next(k for k in prep_manifest["inputs"] if k.endswith("_units.parquet"))
    U = pd.read_parquet(REPO / upath, columns=["unit_idx", "block_geoid"])
    gid = units["unit_idx"].map(dict(zip(U["unit_idx"], U["block_geoid"])))
    assert gid.notna().all()
    blk = gpd.read_file(f"zip://{TIGER}", where="COUNTYFP20 = '031'", engine="pyogrio", columns=["COUNTYFP20", "GEOID20"])
    blk = blk[blk["GEOID20"].isin(set(gid))].to_crs(3435)
    g = dict(zip(blk["GEOID20"], blk.geometry.values))
    assert all(x in g for x in gid)
    return {n: g[x] for n, x in zip(units["node"], gid)}, upath


def build_structure(model, units, E, ne, pman, keep_laplacians=False):
    """Graphs, MST bridges and whiteners of model S or P for one fit region (see the module docstring)."""
    t0 = time.time()
    ns = len(units)
    nbhds = sorted(units["nbhd_id"].unique())
    kid = {k: i for i, k in enumerate(nbhds)}
    nb_of = units["nbhd_id"].map(kid).to_numpy()
    geom, upath = unit_geometry(units, pman)
    ea_all, eb_all = E["a"].to_numpy(), E["b"].to_numpy()
    if model == "S":
        groups = [np.arange(ns)]
        group_edges = [list(zip(ea_all, eb_all))]
    else:
        groups = [np.flatnonzero(nb_of == k) for k in range(len(nbhds))]
        same = nb_of[ea_all] == nb_of[eb_all]
        kk = nb_of[ea_all[same]]
        group_edges = [list(zip(ea_all[same][kk == k], eb_all[same][kk == k])) for k in range(len(nbhds))]
    bridges, blocks, scales = [], [], []
    for gi, (nodes, ed) in enumerate(zip(groups, group_edges)):
        br = mst_bridges(pieces_of(nodes.tolist(), ed), geom)
        bridges += [dict(graph="block" if model == "S" else f"block:{nbhds[gi]}", a=x, b=y, length_m=round(d, 1),
                         unit_a=units["unit_id"][x], unit_b=units["unit_id"][y]) for x, y, d in br]
        loc = {n: i for i, n in enumerate(nodes)}
        ea = [loc[x] for x, y in ed] + [loc[x] for x, y, _ in br]
        eb = [loc[y] for x, y in ed] + [loc[y] for x, y, _ in br]
        Lk = laplacian(len(nodes), np.array(ea, int), np.array(eb, int))
        Ak, s = whiten(Lk, scale=True)
        blocks.append((nodes, Ak, Lk if keep_laplacians else None))
        scales.append(s)
    nz = sum(Ak.shape[1] for _, Ak, _ in blocks)
    rows, cols, vals, c0 = [], [], [], 0
    for nodes, Ak, _ in blocks:
        rr, cc = np.meshgrid(nodes, np.arange(c0, c0 + Ak.shape[1]), indexing="ij")
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(Ak.ravel())
        c0 += Ak.shape[1]
    A_u = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ns, nz))
    group_of = np.zeros(ns, int)
    for gi, (nodes, _, _) in enumerate(blocks):
        group_of[nodes] = gi
    # neighbourhood graph (P): the prep's adjacencies plus MST bridges between its pieces, the
    # bridge joining the neighbourhoods of the nearest unit pair across two pieces
    ka, kb = ne["nbhd_a"].map(kid).to_numpy(), ne["nbhd_b"].map(kid).to_numpy()
    if model == "P":
        npieces = pieces_of(list(range(len(nbhds))), list(zip(ka, kb)))
        upieces = [sorted(np.flatnonzero(np.isin(nb_of, pc)).tolist()) for pc in npieces]
        for x, y, d in mst_bridges(upieces, geom):
            ka, kb = np.append(ka, nb_of[x]), np.append(kb, nb_of[y])
            bridges.append(dict(graph="nbhd", a=x, b=y, length_m=round(d, 1), unit_a=units["unit_id"][x],
                                unit_b=units["unit_id"][y], nbhd_a=nbhds[nb_of[x]], nbhd_b=nbhds[nb_of[y]]))
        A_comp, _ = whiten(laplacian(len(nbhds), ka, kb), scale=False)
    else:
        A_comp = np.zeros((len(nbhds), 0))
    Dt = np.eye(NT)[1:] - np.eye(NT)[:-1]
    A_t, _ = whiten(Dt.T @ Dt, scale=False)
    return dict(nbhds=nbhds, kid=kid, nb_of=nb_of, A_u=A_u, blocks=blocks, scales=scales, nz=nz, group_of=group_of,
                n_block_graphs=len(blocks), A_comp=A_comp, nbhd_edges=(ka, kb), A_t=A_t, bridges=bridges,
                upath=upath, seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", required=True)
    ap.add_argument("--model", choices=["S", "P"], required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--runs", default="public/runs/v2")
    ap.add_argument("--name")
    ap.add_argument("--jitter", action="store_true", help="independently perturbed starts around the mode (check)")
    ap.add_argument("--dump", help="write the Stan data (JSON) to this file and stop (debugging)")
    a = ap.parse_args()
    prep = Path(a.prep) / f"t{a.town}"
    name = a.name or f"struct_{a.model}_t{a.town}"
    run = Path(a.runs) / name
    if (run / "report.json").exists():
        raise SystemExit(f"{run} already has report.json; choose another --name or move it aside")
    if run.exists():
        run.rename(run.with_name(run.name + f".partial-{time.strftime('%Y%m%d%H%M%S')}"))
    run.mkdir(parents=True)
    print(f"=== fit_struct {name}: town {a.town}, model {a.model}, prep {a.prep}", flush=True)

    t = pd.read_parquet(prep / "train.parquet")
    r = pd.read_parquet(prep / "roll.parquet")
    units = pd.read_parquet(prep / "units.parquet")
    E = pd.read_parquet(prep / "edges.parquet")
    ne = pd.read_parquet(prep / "nbhd_edges.parquet")
    coding = json.loads((prep / "coding.json").read_text())
    pman = json.loads((prep / "prep_manifest.json").read_text())
    xcols = ["x:" + c for c in coding["columns"]]
    ns = len(units)
    assert (units["node"].to_numpy() == np.arange(ns)).all()
    node_t, node_r = t["node"].to_numpy().astype(int), r["node"].to_numpy().astype(int)

    st = build_structure(model=a.model, units=units, E=E, ne=ne, pman=pman)
    nbhds, kid, nb_of, A_u, A_comp, A_t = st["nbhds"], st["kid"], st["nb_of"], st["A_u"], st["A_comp"], st["A_t"]
    nz, group_of, bridges, scales, t_struct = st["nz"], st["group_of"], st["bridges"], st["scales"], st["seconds"]
    pd.DataFrame(bridges, columns=["graph", "a", "b", "length_m", "unit_a", "unit_b", "nbhd_a", "nbhd_b"]).to_csv(
        run / "bridges.csv", index=False)
    print(f"graph: {ns:,} units in {st['n_block_graphs']} whitened block graph(s), latent {nz:,}; {len(bridges)} bridges"
          + (f" (longest {max(b['length_m'] for b in bridges):,.0f} m)" if bridges else "")
          + f"; neighbourhood latent {A_comp.shape[1]}; {t_struct:.1f} s", flush=True)

    # ------------------------------------------------------------ units holding a training sale (rows of A in Stan)
    obs_units = np.unique(node_t)
    obs_of = np.full(ns, -1)
    obs_of[obs_units] = np.arange(len(obs_units))
    A_obs = A_u[obs_units]
    use_csr = int(a.model == "P")

    X = t[xcols].to_numpy()
    xmean = X.mean(axis=0)
    Qm, Rm = np.linalg.qr(X - xmean)
    sq = np.sqrt(len(t) - 1)
    R_inv = np.linalg.inv(Rm / sq)
    has_nbhd = int(a.model == "P")
    data = dict(N=len(t), Nt=NT, K=len(nbhds), Ns_obs=len(obs_units), Nr=len(xcols), has_nbhd=has_nbhd,
                sell=t["month"].astype(int).tolist(), nbhd=t["nbhd_id"].map(kid).add(1).astype(int).tolist(),
                obs=(obs_of[node_t] + 1).tolist(), yVar=t["log_price"].to_numpy(), Q=Qm * sq, R_inv=R_inv,
                N_time_latent=A_t.shape[1], A_t=A_t, N_comp_latent=A_comp.shape[1], A_comp=A_comp, Nz=nz,
                use_csr=use_csr, A_dense=(np.zeros((0, 0)) if use_csr else A_obs.toarray()),
                nnz=(A_obs.nnz if use_csr else 0), csr_w=(A_obs.data if use_csr else np.zeros(0)),
                csr_v=((A_obs.indices + 1).tolist() if use_csr else []),
                csr_u=((A_obs.indptr + 1).tolist() if use_csr else []), y_init=float(t["log_price"].mean()))
    assert (t["nbhd_id"].map(kid).to_numpy() == nb_of[node_t]).all()

    import cmdstanpy
    if a.dump:
        cmdstanpy.write_stan_json(a.dump, data)
        shutil.rmtree(run)
        return
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN), stanc_options={"O1": True})
    out_dir = run / "chains"
    timing = dict(spatial_structure_s=round(t_struct, 1))
    report = dict(name=name, town=a.town, model=a.model, method="nuts", init="mode" + ("+jitter" if a.jitter else ""), seed=a.seed,
                  whitener_sha256=dict(A_u=hashlib.sha256(A_u.data.tobytes() + A_u.indices.tobytes() + A_u.indptr.tobytes()).hexdigest(),
                                       A_comp=hashlib.sha256(np.ascontiguousarray(A_comp).tobytes()).hexdigest(),
                                       A_t=hashlib.sha256(np.ascontiguousarray(A_t).tobytes()).hexdigest()),
                  settings=dict(chains=a.chains, warmup=a.warmup, samples=a.samples, thin=1), prep=a.prep,
                  n_train=len(t), n_valued=len(r), n_spatial_nodes=ns, n_units_with_training_sale=len(obs_units),
                  n_edges=len(E), n_bridges=len(bridges), n_block_graphs=st["n_block_graphs"], n_block_latent=nz,
                  block_scales=dict(min=float(min(scales)), max=float(max(scales))), n_nbhds=len(nbhds),
                  n_nbhd_latent=int(A_comp.shape[1]), n_regressors=len(xcols), units_file=st["upath"],
                  prep_manifest=pman["outputs"], coding_sha256=pman["coding_sha256"],
                  stan_file=str(STAN.relative_to(REPO)),
                  versions=dict(cmdstan=cmdstanpy.cmdstan_version(), cmdstanpy=cmdstanpy.__version__,
                                python=platform.python_version()))

    # start values as fit_v2.py, then NUTS from the mode
    base = dict(intercept=data["y_init"], theta=np.zeros(len(xcols)).tolist(), sigEps=0.3, nu=5.0, sigma_mu=0.05,
                sigma_u=0.2, sigma_zeta=[0.2] * has_nbhd, sigma_psi=[0.05] * has_nbhd)
    t_all = time.time()
    t0 = time.time()
    mode = model.optimize(data=data, jacobian=True, seed=a.seed, output_dir=str(out_dir / "mode"), iter=20000,
                          inits=base, show_console=False)
    timing["optimize_s"] = round(time.time() - t0, 1)
    print(f"mode found in {timing['optimize_s']} s", flush=True)
    mv = mode.stan_variables()
    inits = {k: (np.atleast_1d(x) if k in ("sigma_zeta", "sigma_psi") else np.asarray(x))   # array[1] in Stan
             for k, x in mv.items() if np.size(x)}
    if a.jitter:
        # independently perturbed starts around the mode (a convergence check): latent N(0, 1) coordinates + N(0, 0.5),
        # coefficients' theta + N(0, 0.1), positive scales x exp(N(0, 0.3)), nu clipped to its bounds
        rng = np.random.default_rng(1000 + a.seed)
        def jit(k, x):
            if k in ("mu_raw", "zeta_raw", "psi_raw", "z"):
                return x + rng.normal(0, 0.5, x.shape)
            if k in ("theta", "intercept"):
                return x + rng.normal(0, 0.1 if k == "theta" else 0.01, np.shape(x))
            if k == "nu":
                return float(np.clip(x * np.exp(rng.normal(0, 0.3)), 2.5, 39))
            return x * np.exp(rng.normal(0, 0.3, np.shape(x)))
        inits = [{k: np.asarray(jit(k, x)).tolist() for k, x in inits.items() if k not in ("beta",)}
                 for _ in range(a.chains)]
    else:
        inits = {k: x.tolist() for k, x in inits.items()}
    t0 = time.time()
    fit = model.sample(data=data, chains=a.chains, parallel_chains=a.chains, iter_warmup=a.warmup,
                       iter_sampling=a.samples, seed=a.seed, inits=inits, output_dir=str(out_dir), show_progress=False)
    timing["sample_s"] = round(time.time() - t0, 1)
    timing["runtime_s"] = round(time.time() - t_all, 1)
    report["timing"] = timing
    print(f"sampled in {timing['sample_s'] / 60:.1f} min; runtime {timing['runtime_s'] / 60:.1f} min", flush=True)

    # ------------------------------------------------------------ effects from the draws (chain-major)
    t0 = time.time()
    v = fit.stan_variables()
    nd = len(v["intercept"])
    sig = lambda k: (np.asarray(v[k]).reshape(nd, -1)[:, 0] if has_nbhd or k in ("sigma_mu", "sigma_u") else np.zeros(nd))
    mu_full = v["mu_raw"] @ A_t.T * v["sigma_mu"][:, None]
    u = np.asarray((A_u @ (v["z"] * v["sigma_u"][:, None]).T).T)                  # (draws, units), prep order
    zeta = (v["zeta_raw"] @ A_comp.T * sig("sigma_zeta")[:, None]) if has_nbhd else np.zeros((nd, len(nbhds)))
    if has_nbhd:
        psi = np.einsum("kl,dlm,tm->dkt", A_comp, v["psi_raw"], A_t) * sig("sigma_psi")[:, None, None]
    else:
        psi = np.zeros((nd, len(nbhds), NT))
    beta = v["theta"] @ R_inv.T
    np.savez_compressed(run / "draws.npz", intercept=v["intercept"], beta=beta, mu=mu_full, zeta=zeta,
                        psi_last=psi[:, :, -1], u=u, sigma_mu=v["sigma_mu"], sigma_u=v["sigma_u"],
                        sigma_zeta=sig("sigma_zeta"), sigma_psi=sig("sigma_psi"), sigEps=v["sigEps"], nu=v["nu"])
    np.savez_compressed(run / "design.npz", pins=r["meta_pin"].to_numpy(str), X_c=r[xcols].to_numpy() - xmean,
                        nbhd=r["nbhd_id"].map(kid).to_numpy(), node=node_r, xcols=np.array(xcols),
                        nbhds=np.array(nbhds))

    # ------------------------------------------------------------ sum-to-zero checks from the draws
    def worst(sums, abs_sums):
        """largest |sum of the effect| / max(1, sum of |effect|) over draws and groups (a relative test)"""
        return dict(max_rel=float((np.abs(sums) / np.maximum(1.0, abs_sums)).max()), max_abs_sum=float(np.abs(sums).max()))
    G = sp.csr_matrix((np.ones(ns), (group_of, np.arange(ns))))
    checks = dict(mu_over_months=worst(mu_full.sum(axis=1), np.abs(mu_full).sum(axis=1)),
                  u_over_township=worst(u.sum(axis=1), np.abs(u).sum(axis=1)))
    if has_nbhd:
        checks["u_within_each_nbhd"] = worst(np.asarray(G @ u.T), np.asarray(G @ np.abs(u).T))
        checks["zeta_over_nbhds"] = worst(zeta.sum(axis=1), np.abs(zeta).sum(axis=1))
        checks["psi_over_nbhds_each_month"] = worst(psi.sum(axis=1), np.abs(psi).sum(axis=1))
        checks["psi_over_months_each_nbhd"] = worst(psi.sum(axis=2), np.abs(psi).sum(axis=2))
    tol = 1e-8
    report["sum_to_zero"] = dict(tolerance_relative=tol, checks=checks,
                                 all_pass=bool(all(c["max_rel"] < tol for c in checks.values())))

    # ------------------------------------------------------------ rank-normalised diagnostics of every component
    c, m = a.chains, a.samples
    valued_units = np.unique(node_r)
    ks = np.arange(len(nbhds))
    q = {"intercept": v["intercept"][:, None], "beta": beta, "mu": mu_full, "u_valued_units": u[:, valued_units],
         "location_valued_units": (zeta[:, nb_of[valued_units]] + psi[:, nb_of[valued_units], -1] + u[:, valued_units]),
         "z": v["z"], "theta": v["theta"], "mu_raw": v["mu_raw"]}
    q["level_plus_location"] = q["location_valued_units"] + (v["intercept"] + mu_full[:, -1])[:, None]
    scal = ["sigma_mu", "sigma_u", "sigEps", "nu"] + (["sigma_zeta", "sigma_psi"] if has_nbhd else [])
    q["scalars"] = np.column_stack([sig(k) if k.startswith("sigma_z") or k.startswith("sigma_p") else v[k] for k in scal])
    if has_nbhd:
        q.update(zeta=zeta[:, ks], psi_last=psi[:, :, -1], psi_all=psi.reshape(nd, -1), zeta_raw=v["zeta_raw"],
                 psi_raw=v["psi_raw"].reshape(nd, -1))
    sv = fit.method_variables()                                                   # (draws per chain, chains)
    q["lp__"] = sv["lp__"].T.reshape(-1)[:, None]
    en = sv["energy__"].T                                                         # (chains, draws)
    ebfmi = (np.diff(en, axis=1) ** 2).mean(axis=1) / en.var(axis=1)
    diag, dsave = {}, {}
    for k, x in q.items():
        x = np.asarray(x, float)
        x = x[:, x.std(axis=0) > 1e-12].reshape(c, m, -1)
        rh, eb, et = md.rhat(x), md.ess_bulk(x), md.ess_tail(x)
        dsave.update({f"{k}:rhat": rh, f"{k}:ess_bulk": eb, f"{k}:ess_tail": et})
        diag[k] = dict(k=int(x.shape[2]), rhat_max=float(rh.max()), share_rhat_gt_1_01=float((rh > 1.01).mean()),
                       ess_bulk_min=float(eb.min()), ess_tail_min=float(et.min()))
    diag["scalars_each"] = {k: dict(rhat=float(dsave["scalars:rhat"][i]), ess_bulk=float(dsave["scalars:ess_bulk"][i]))
                            for i, k in enumerate(scal)}
    np.savez_compressed(run / "diag.npz", **dsave)
    report["diagnostics"] = dict(
        max_rhat=float(max(d["rhat_max"] for k, d in diag.items() if k != "scalars_each")),
        min_ess_bulk=float(min(d["ess_bulk_min"] for k, d in diag.items() if k != "scalars_each")),
        min_ess_tail=float(min(d["ess_tail_min"] for k, d in diag.items() if k != "scalars_each")),
        divergent=int(sv["divergent__"].sum()), max_treedepth_share=float((sv["treedepth__"] >= 10).mean()),
        ebfmi=[float(x) for x in ebfmi], mean_leapfrog=float(sv["n_leapfrog__"].mean()),
        mean_treedepth=float(sv["treedepth__"].mean()),
        stepsize=[float(x) for x in np.unique(np.round(sv["stepsize__"], 5))], by_component=diag,
        note="rank-normalised split R-hat and bulk/tail ESS (mcmc_diag.py) of every component; constant columns left out")
    (run / "diagnose.txt").write_text(fit.diagnose())
    timing["post_s"] = round(time.time() - t0, 1)
    report["n_draws"] = nd
    report["scalars"] = {k: dict(mean=float(q["scalars"][:, i].mean()), sd=float(q["scalars"][:, i].std()))
                         for i, k in enumerate(scal)}
    (run / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(sum_to_zero=report["sum_to_zero"], max_rhat=report["diagnostics"]["max_rhat"],
                          min_ess_bulk=report["diagnostics"]["min_ess_bulk"],
                          divergent=report["diagnostics"]["divergent"],
                          by_component={k: (d["rhat_max"], d["ess_bulk_min"]) for k, d in diag.items()
                                        if k != "scalars_each"}), indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
