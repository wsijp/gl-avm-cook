"""Model development: fit one model variant to one township prepared by loop_prep.py and predict its evaluation sales.

Model public/stan/v2/gl_loop.stan: model P (gl_struct.stan, has_nbhd = 1) term for term with every option off; option
--zf adds the flat neighbourhood level zf. Graphs, bridges and whiteners: fit_struct.build_structure, model P,
unchanged, with 88 training months (2016-01 .. 2023-04). A preparation whose nodes are flat buildings is joined
by MST bridges on point distance (the buildings' coordinates) instead of block polygons.

Settings: 4 chains x 500 warm-up + 500 draws; mode start with independently perturbed starts per chain
(fit_struct.py's --jitter rule: latent N(0, 1) coordinates + N(0, 0.5), theta + N(0, 0.1), intercept + N(0, 0.01),
scales x exp(N(0, 0.3)), nu clipped to [2.5, 39], sigEps to >= 0.0125); zf_raw is a latent coordinate, sigma_zf a scale.

Prediction of an evaluation sale (roll.parquet): posterior median of exp(latent),
  latent = intercept + x' beta + mu[Apr 2023] + zeta[k] + psi[k, Apr 2023] + u[node] (+ zf[k] for a flat, with --zf);
every time-varying term held at April 2023, the last training month.
Option --gcurve: the gap-premium curve g of gl_loop.stan over the prep's hist_gbin (14 bins of the
time since the latest earlier sale; 0 = none): A_g = fit_struct.whiten of the path-graph Laplacian on the bins (unscaled,
like A_t); g_raw is a latent coordinate and sigma_g a scale for the jitter; an evaluation sale with gbin > 0 adds
g[gbin] draw by draw.
Option --gexp: the decaying premium a * exp(-g / tau) of gl_loop.stan, g = years since the latest
earlier sale since 1999 (the prep's hist_months / 12; complete history, loop_prep.py --fullhist long --gapcode exp|expc),
0 for a sale without one; with a ramp: premium a (1 - exp(-g / delta)) exp(-g / tau), delta = GEXP_DELTA fixed, priors
a ~ N(0, GEXP_A_SD), tau ~ lognormal(log GEXP_TAU_MEDIAN[segment], GEXP_TAU_SD); gexp_a is jittered like theta, gexp_tau
like a scale; an evaluation sale adds the premium draw by draw.

Option --nvc: neighbourhood-varying coefficients of NVC_COLS (houses: log building area, log lot
size): each keeps its township coefficient in beta and gets a neighbourhood deviation v = A_comp nvc_raw sigma_nvc (the
neighbourhood graph's whitener, as zeta: deviations sum to zero over neighbourhoods), applied to the covariate centred at
its training mean; sigma_nvc ~ N+(0, 0.1); nvc_raw a latent coordinate and sigma_nvc a scale for the jitter; an
evaluation sale adds x_c' v[k] draw by draw.

Option --np: the neighbourhood-varying resale premium and flip-purchase discount (houses),
through the same gl_loop.stan term as --nvc (n_nvc = 2, v = A_comp nvc_raw sigma_nvc, sigma_nvc ~ N+(0, 0.1)), with RAW
loadings, not centred: column 1 c_i = I(an earlier sale since 2016) w(g_i), w = 1 for g <= 2 years, (1 + cos(pi (g - 2)
/ 2)) / 2 for 2 < g < 4, 0 from 4 (g = hist_months / 12); column 2 b_i = the sum of the two flip-purchase flags (training
rows; 0 for evaluation sales). The township coefficients of the gap bands and the flags stay in beta (the deviations
sum to zero over neighbourhoods). With --forecast the term (time-invariant) is added to Stan's lat_dev and lat_dev_apr
draw by draw here, after the Stan = numpy check.

Option --forecast: the forecasting time model of gl_loop.stan (has_fc): township path = damped
trend (ARIMA(1,1,0) with drift on the monthly increments) + a calendar-month effect smoothed on the 12-month cycle; psi
keeps its random walk. The evaluation sales' design (centred x through R^-1, nodes, neighbourhood, months after April
2023, premium-curve gap; never their prices) goes into the Stan data, and Stan's generated quantities give each sale's
latent value at its own month (lat_dev: the forecast paths, as posterior draws) and held at April 2023 (lat_dev_apr, the
previous rule, a diagnostic). The prediction is the posterior median of exp(lat_dev). Checked after sampling against the
same quantities assembled here in numpy from the draws (FC_CHECK_TOL). With --county (the Assessor's own 2025 split:
train to 20 October 2023, test on the later sales) the test sales of 20-31 October 2023 lie in the last training month
(horizon 0): their value is lat_dev_apr (every time term at that month), the forecast rule's value for it; later sales
take the forecast at their own month.

Writes public/runs/v2/<name>/: report.json (settings, runtime, graph, sum-to-zero checks, diagnostics), predictions.parquet
(the three sale keys, price, prediction, 5% / 95% quantiles, latent R-hat, is_flat), draws.npz (scalars, beta, the
latent predictions), chains/.

Usage (repo root): .venv/bin/python -I public/scripts/v2/loop_fit.py --town 74 --prep public/data/v2/prep_loop_p \
                   --name loop_P_s1_t74 [--zf] [--seed 1] [--dump FILE]
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_v2  # noqa: E402
import loop_fold as lf  # noqa: E402  (GL_FOLD selects the development split; default = the 2023 split, 88 months)

COUNTY = "--county" in sys.argv          # the Assessor's 2025 split, 94 months (2016-01 .. 2023-10); else 88
assert not (COUNTY and lf.FOLD != "2023"), "the county design is fold 1's"
NT = 94 if COUNTY else lf.NT
fit_v2.NT = NT
import fit_struct  # noqa: E402
import mcmc_diag as md  # noqa: E402

assert fit_struct.NT == NT
REPO = Path(__file__).resolve().parents[3]
STAN = REPO / "public/stan/v2/gl_loop.stan"
KEYS = ["meta_pin", "meta_sale_document_num", "meta_card_num"]
ZF_MIN_PINS, ZF_MIN_NBHDS, ZF_MIN_NBHD_PINS = 500, 5, 30          # zf eligibility (training flat PINs)


def point_geometry(units, prep_manifest):
    """node -> shapely point at the flat building's coordinates (EPSG:3435), for MST bridges on point distance."""
    import shapely
    return {n: shapely.Point(x, y) for n, x, y in zip(units["node"], units["x"], units["y"])}, "units.parquet x, y"


def zf_eligible(t):
    f = t[t["is_flat"]]
    per = f.groupby("nbhd_id")["meta_pin"].nunique()
    return bool(f["meta_pin"].nunique() >= ZF_MIN_PINS and (per >= ZF_MIN_NBHD_PINS).sum() >= ZF_MIN_NBHDS)


# premium with a ramp: a (1 - exp(-g / delta)) exp(-g / tau), delta fixed;
# a ~ N(0, 1); tau ~ lognormal(log tau0, 0.5), tau0 = the pooled training estimate of the segment (a grid search of the
# same ramped curve on the training sales: houses 5.25 y, flats 6.77 y; the grid values 5.253 and 6.770 rounded to 0.01)
GEXP_TAU_MEDIAN = {"house": 5.25, "flat": 6.77}
GEXP_TAU_SD, GEXP_A_SD, GEXP_DELTA = 0.5, 1.0, 0.25


NVC_COLS = {"house": ["x:h:log_bldg_sf", "x:h:log_land_sf"]}            # --nvc
NP_COLS = ["np:resale_taper", "np:flipbuy"]                             # --np: raw loadings, built by np_loadings
NP_FLAGS = ["x:h:flipbuy_lt12", "x:h:flipbuy_12_24"]


def np_taper(g):
    """w(g) = 1 for g <= 2 years, (1 + cos(pi (g - 2) / 2)) / 2 for 2 < g < 4, 0 from 4 (fixed, not tuned)."""
    g = np.asarray(g, float)
    return np.where(g <= 2, 1.0, np.where(g >= 4, 0.0, 0.5 * (1 + np.cos(np.pi * (g - 2) / 2))))


def np_loadings(d, flags=True):
    """The raw loadings (c_i, b_i) of a train (flags=True) or roll (flags=False: b = 0) frame."""
    sold = (d["hist_n_prev"] > 0).to_numpy()
    g = np.where(sold, d["hist_months"].to_numpy(float) / 12, np.inf)
    assert np.isfinite(g[sold]).all() and (g[sold] > 0).all()
    c = np.where(sold, np_taper(np.where(sold, g, 0.0)), 0.0)
    if flags:                                      # the standardised flags are > 0 exactly where the raw flag is 1
        F = (d[NP_FLAGS].to_numpy() > 0).astype(float)
        assert F.sum(axis=1).max() <= 1, "the two flip-purchase flags are mutually exclusive"
        b = F.sum(axis=1)
    else:
        b = np.zeros(len(d))
    return np.column_stack([c, b])

# forecasting time model priors; the empty parameters of the option when it is off
FC_DELTA_SD, FC_SIGMA_S_SD = 0.01, 0.05
FC_OFF_PARAMS = dict(fc_delta=[], fc_phi=[], s_raw=[], sigma_s=[])
FC_CHECK_TOL = 1e-4                     # Stan CSV output keeps 6 significant digits (latent values ~12)


def cycle_whitener(n=12):
    """A_c: the whitened Laplacian of the n-cycle (unscaled, like A_t): one null dimension, so A_c s sums to zero."""
    D = np.eye(n)[np.r_[1:n, 0]] - np.eye(n)
    return fit_struct.whiten(D.T @ D, scale=False)[0]


def fc_path_np(e, s, delta, phi):
    """numpy twin of gl_loop.stan's fc_incr + fc_cum, draw by draw: e draws x (Nt - 1); returns (increments, the
    uncentred trend)."""
    d = np.empty_like(e)
    d[:, 0] = delta + s / np.sqrt(1 - phi ** 2) * e[:, 0]
    for j in range(1, e.shape[1]):
        d[:, j] = delta + phi * (d[:, j - 1] - delta) + s * e[:, j]
    return d, np.concatenate([np.zeros((e.shape[0], 1)), np.cumsum(d, axis=1)], axis=1)


def gap_whitener(n):
    D = np.eye(n)[1:] - np.eye(n)[:-1]
    return fit_struct.whiten(D.T @ D, scale=False)[0]


def make_data(prep, zf, gcurve=False, gexp=False, nvc=False, forecast=False, np_=False):
    """Stan data for gl_loop.stan and what prediction needs (the published P construction, fit_struct.main)."""
    t = pd.read_parquet(prep / "train.parquet")
    r = pd.read_parquet(prep / "roll.parquet")
    units = pd.read_parquet(prep / "units.parquet")
    E = pd.read_parquet(prep / "edges.parquet")
    ne = pd.read_parquet(prep / "nbhd_edges.parquet")
    coding = json.loads((prep / "coding.json").read_text())
    pman = json.loads((prep / "prep_manifest.json").read_text())
    # a segment-only fit may lack sales in the last month (county test, Evanston flats: none in October 2023); the time
    # paths carry through such months (random walk), so require only the range and a sale in the last 3 months
    assert t["month"].between(1, NT).all() and t["month"].max() >= NT - 2, f"{NT} training months expected"
    if COUNTY:
        assert (r["date"] >= "2023-10-20").all() and len(pman["counts"]) > 0, "roll = the held-out test sales"
    else:
        assert (r["date"] >= lf.DEV_START).all() and (r["date"] < lf.HELD_OUT_START).all(), "roll = development sales only"
    xcols = ["x:" + c for c in coding["columns"]]
    ns = len(units)
    assert (units["node"].to_numpy() == np.arange(ns)).all()
    buildings = pman.get("node_type") == "building"
    if buildings:
        fit_struct.unit_geometry = point_geometry
    st = fit_struct.build_structure(model="P", units=units, E=E, ne=ne, pman=pman)
    node_t = t["node"].to_numpy().astype(int)
    obs_units = np.unique(node_t)
    obs_of = np.full(ns, -1)
    obs_of[obs_units] = np.arange(len(obs_units))
    A_obs = st["A_u"][obs_units]
    X = t[xcols].to_numpy()
    xmean = X.mean(axis=0)
    Qm, Rm = np.linalg.qr(X - xmean)
    sq = np.sqrt(len(t) - 1)
    R_inv = np.linalg.inv(Rm / sq)
    kid = st["kid"]
    is_flat = t["is_flat"].astype(int).to_numpy()
    if zf:
        assert zf_eligible(t), "zf requested for a township that is not eligible (LOOP_PLAN §8, A1 (2))"
        assert is_flat.any() and not is_flat.all()
    data = dict(N=len(t), Nt=NT, K=len(st["nbhds"]), Ns_obs=len(obs_units), Nr=len(xcols),
                sell=t["month"].astype(int).tolist(), nbhd=t["nbhd_id"].map(kid).add(1).astype(int).tolist(),
                obs=(obs_of[node_t] + 1).tolist(), yVar=t["log_price"].to_numpy(), Q=Qm * sq, R_inv=R_inv,
                N_time_latent=st["A_t"].shape[1], A_t=st["A_t"], N_comp_latent=st["A_comp"].shape[1], A_comp=st["A_comp"],
                Nz=st["nz"], nnz=A_obs.nnz, csr_w=A_obs.data, csr_v=(A_obs.indices + 1).tolist(),
                csr_u=(A_obs.indptr + 1).tolist(), y_init=float(t["log_price"].mean()),
                n_dseg=0, dseg=[0] * len(t), has_zf=int(zf), has_psif=0, is_flat=is_flat.tolist(),
                n_gbin=0, gbin=[0] * len(t), A_g=np.zeros((0, 0)),
                has_gexp=0, gexp_sold=[0] * len(t), gexp_years=[0.0] * len(t), gexp_tau_mu=0.0, gexp_tau_sd=1.0,
                gexp_a_sd=1.0, gexp_delta=0.0, n_nvc=0, X_nvc=np.zeros((len(t), 0)),
                has_fc=0, cal=[], A_c=np.zeros((0, 0)), fc_delta_sd=FC_DELTA_SD, fc_sigma_s_sd=FC_SIGMA_S_SD, H=0,
                cal_fc=[], N_dev=0, h_dev=[], nbhd_dev=[], Q_dev=np.zeros((0, len(xcols))), nnz_dev=0, csr_w_dev=[],
                csr_v_dev=[], csr_u_dev=[1], gexp_sold_dev=[], gexp_years_dev=[])
    if gcurve:
        nb = int(pman["loop_options"]["n_gbin"])
        gb = t["hist_gbin"].astype(int).to_numpy()
        assert gb.min() >= 0 and gb.max() <= nb and (gb > 0).any()
        assert not any("hist_gap" in c for c in xcols), "the curve is the only gap term (A7)"
        data.update(n_gbin=nb, gbin=gb.tolist(), A_g=gap_whitener(nb))
    if gexp:
        lo = pman["loop_options"]
        assert not gcurve and lo.get("fullhist") == "long" and lo.get("gapcode") == "exp" and lo.get("segment") in GEXP_TAU_MEDIAN
        assert not any("hist_" in c for c in xcols), "the premium curve is the only history term (A11)"
        seg = lo["segment"]
        sold = (t["hist_n_prev"] > 0).to_numpy()
        g = np.where(sold, t["hist_months"].to_numpy(float) / 12, 0.0)
        assert np.isfinite(g).all() and (g[sold] > 0).all()
        data.update(has_gexp=1, gexp_sold=sold.astype(int).tolist(), gexp_years=g.tolist(),
                    gexp_tau_mu=float(np.log(GEXP_TAU_MEDIAN[seg])), gexp_tau_sd=GEXP_TAU_SD, gexp_a_sd=GEXP_A_SD,
                    gexp_delta=GEXP_DELTA)
    nvc_cols, nvc_mean = [], np.zeros(0)
    h0 = np.zeros(len(r), bool)                                        # horizon-0 test sales (the Assessor's 2025 split)
    if nvc:
        seg = pman["loop_options"].get("segment")
        nvc_cols = NVC_COLS[seg]
        assert all(c in xcols for c in nvc_cols), nvc_cols
        nvc_mean = np.array([xmean[xcols.index(c)] for c in nvc_cols])
        data.update(n_nvc=len(nvc_cols), X_nvc=t[nvc_cols].to_numpy() - nvc_mean)
    if np_:                                                            # --np: raw loadings in the n_nvc term
        assert not nvc and pman["loop_options"].get("segment") == "house" and all(c in xcols for c in NP_FLAGS)
        assert "x:h:hist_sold" in xcols and "x:h:hist_gap_lt1y" in xcols, "the township gap bands stay in beta"
        nvc_cols = NP_COLS
        data.update(n_nvc=2, X_nvc=np_loadings(t))
    if forecast:                                                       # --forecast: evaluation-sale design, never prices
        assert not (zf or gcurve or nvc), "the forecast option is built for K's terms (A14; NP's term is added after)"
        assert data["N_time_latent"] == NT - 1
        hr = r["month"].astype(int).to_numpy() - NT
        if COUNTY:      # test sales of 20-31 Oct 2023 lie in the last training month (horizon 0); Stan gets
            h0 = hr == 0  # the placeholder h = 1 for them and their value is lat_dev_apr (every time term at month T)
            assert hr.min() >= 0, "test sales from the last training month on"
            hr = np.maximum(hr, 1)
        else:
            assert hr.min() >= 1, "development sales after the last training month"
        H = int(hr.max())
        node_r = r["node"].to_numpy().astype(int)
        A_r = st["A_u"][node_r].tocsr()
        Xr = r[xcols].to_numpy() - xmean
        cal = (np.arange(NT + H) % 12) + 1                              # month 1 = January 2016
        data.update(has_fc=1, cal=cal[:NT].tolist(), A_c=cycle_whitener(), H=H, cal_fc=cal[NT:].tolist(),
                    N_dev=len(r), h_dev=hr.tolist(), nbhd_dev=(r["nbhd_id"].map(kid).to_numpy() + 1).tolist(),
                    Q_dev=Xr @ R_inv, nnz_dev=A_r.nnz, csr_w_dev=A_r.data, csr_v_dev=(A_r.indices + 1).tolist(),
                    csr_u_dev=(A_r.indptr + 1).tolist(), gexp_sold_dev=[0] * len(r), gexp_years_dev=[0.0] * len(r))
        if gexp:
            sold_r = (r["hist_n_prev"] > 0).to_numpy()
            g_r = np.where(sold_r, r["hist_months"].to_numpy(float) / 12, 0.0)
            assert np.isfinite(g_r).all()
            data.update(gexp_sold_dev=sold_r.astype(int).tolist(), gexp_years_dev=g_r.tolist())
    assert (t["nbhd_id"].map(kid).to_numpy() == st["nb_of"][node_t]).all()
    assert data["N_comp_latent"] >= 1, "gl_loop.stan needs at least two neighbourhoods"
    return dict(data=data, t=t, r=r, units=units, E=E, st=st, xcols=xcols, xmean=xmean, coding=coding, pman=pman,
                obs_units=obs_units, buildings=buildings, nvc_cols=nvc_cols, nvc_mean=nvc_mean, h0=h0)


def jitter_inits(mv, chains, seed):
    """fit_struct.py's --jitter rule, extended to zf and the optional terms."""
    rng = np.random.default_rng(1000 + seed)
    arr1 = ("sigma_zeta", "sigma_psi", "sigma_zf", "sigma_g", "gexp_a", "gexp_tau", "fc_delta", "fc_phi", "sigma_s")
    base = {k: (np.atleast_1d(x) if k in arr1 else np.asarray(x)) for k, x in mv.items() if np.size(x)}

    def jit(k, x):
        if k in ("mu_raw", "zeta_raw", "psi_raw", "z", "zf_raw", "g_raw", "nvc_raw", "s_raw"):
            return x + rng.normal(0, 0.5, x.shape)
        if k in ("theta", "intercept", "gexp_a"):
            return x + rng.normal(0, 0.01 if k == "intercept" else 0.1, np.shape(x))
        if k == "fc_delta":                # a drift per month, jittered on its own scale
            return x + rng.normal(0, 0.002, np.shape(x))
        if k == "fc_phi":
            return np.clip(x + rng.normal(0, 0.1, np.shape(x)), -0.9, 0.9)
        if k == "nu":
            return float(np.clip(x * np.exp(rng.normal(0, 0.3)), 2.5, 39))
        if k == "sigEps":                  # lower bound 0.01: a mode at the floor must not start below it
            return float(max(x * np.exp(rng.normal(0, 0.3)), 0.0125))
        return x * np.exp(rng.normal(0, 0.3, np.shape(x)))
    return [{k: np.asarray(jit(k, x)).tolist() for k, x in base.items()} for _ in range(chains)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--zf", action="store_true")
    ap.add_argument("--gcurve", action="store_true", help="A7: gap-premium random walk over hist_gbin")
    ap.add_argument("--gexp", action="store_true", help="D62 (A10): decaying premium a exp(-g / tau)")
    ap.add_argument("--nvc", action="store_true", help="D65 (A12): neighbourhood-varying coefficients of NVC_COLS")
    ap.add_argument("--forecast", action="store_true", help="D67 (A14): damped trend + seasonal; forecasts in Stan")
    ap.add_argument("--np", action="store_true", help="D68 (A15): neighbourhood-varying resale premium and flip discount")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-treedepth", type=int, default=10, help="D70: NUTS max tree depth (default 10 = every loop fit)")
    ap.add_argument("--runs", default="public/runs/v2")
    ap.add_argument("--dump", help="write the Stan data (JSON) to this file and stop")
    ap.add_argument("--county", action="store_true", help="A4 (5): held-out design, 94 months; use --warmup 1000 --samples 1000")
    a = ap.parse_args()
    prep = REPO / a.prep / f"t{a.town}"
    run = REPO / a.runs / a.name
    if (run / "report.json").exists():
        raise SystemExit(f"{run} already has report.json; choose another --name or move it aside")
    print(f"=== loop_fit {a.name}: town {a.town}, prep {a.prep}, zf {a.zf}, seed {a.seed}", flush=True)
    M = make_data(prep, a.zf, a.gcurve, a.gexp, a.nvc, a.forecast, a.np)
    data, t, r, st = M["data"], M["t"], M["r"], M["st"]
    import cmdstanpy
    if a.dump:
        cmdstanpy.write_stan_json(a.dump, data)
        return
    if run.exists():
        run.rename(run.with_name(run.name + f".partial-{time.strftime('%Y%m%d%H%M%S')}"))
    run.mkdir(parents=True)
    print(f"graph: {len(M['units']):,} nodes ({'flat buildings' if M['buildings'] else 'block units'}) in "
          f"{st['n_block_graphs']} whitened graphs, latent {st['nz']:,}; {len(st['bridges'])} bridges; neighbourhoods "
          f"{data['K']}; train {data['N']:,}, regressors {data['Nr']}, development sales {len(r):,}; {st['seconds']:.1f} s",
          flush=True)
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN), stanc_options={"O1": True})
    t_all = time.time()
    base = dict(intercept=data["y_init"], theta=np.zeros(data["Nr"]).tolist(), sigEps=0.3, nu=5.0, sigma_mu=0.05,
                sigma_u=0.2, sigma_zeta=[0.2], sigma_psi=[0.05])
    if a.zf:
        base["sigma_zf"] = [0.1]
    if a.gcurve:
        base["sigma_g"] = [0.05]
    if a.gexp:
        base["gexp_a"], base["gexp_tau"] = [0.2], [float(np.exp(data["gexp_tau_mu"]))]
    if a.nvc or a.np:
        base["sigma_nvc"] = [0.05] * data["n_nvc"]
    if a.forecast:
        base.update(fc_delta=[0.0], fc_phi=[0.0], sigma_s=[0.01])
    t0 = time.time()
    mode = model.optimize(data=data, jacobian=True, seed=a.seed, output_dir=str(run / "chains/mode"), iter=20000,
                          inits=base, show_console=False)
    t_opt = time.time() - t0
    print(f"mode found in {t_opt:.0f} s", flush=True)
    inits = jitter_inits(mode.stan_variables(), a.chains, a.seed)
    t0 = time.time()
    fit = model.sample(data=data, chains=a.chains, parallel_chains=a.chains, iter_warmup=a.warmup,
                       iter_sampling=a.samples, seed=a.seed, inits=inits, output_dir=str(run / "chains"),
                       max_treedepth=a.max_treedepth,
                       show_progress=False)
    t_samp = time.time() - t0
    print(f"sampled in {t_samp / 60:.1f} min", flush=True)

    # ------------------------------------------------------------ effects and evaluation-sale predictions
    v = fit.stan_variables()
    nd = len(v["intercept"])
    A_t, A_comp, A_u = st["A_t"], st["A_comp"], st["A_u"]
    nb_of = st["nb_of"]
    if a.forecast:                                     # the fitted path from Stan, checked against numpy
        incr_np, tr_np = fc_path_np(v["mu_raw"], v["sigma_mu"].reshape(nd), v["fc_delta"].reshape(nd), v["fc_phi"].reshape(nd))
        season_np = v["s_raw"] @ data["A_c"].T * v["sigma_s"].reshape(nd)[:, None]
        mu = v["mu_fit"]
        mu_np = tr_np + season_np[:, np.array(data["cal"]) - 1]
        mu_np -= mu_np.mean(axis=1, keepdims=True)
        fc_err = dict(mu_fit=float(np.abs(mu - mu_np).max()),
                      season=float(np.abs(v["season"] - season_np).max()))
    else:
        mu = v["mu_raw"] @ A_t.T * v["sigma_mu"][:, None]
    zeta = v["zeta_raw"] @ A_comp.T * v["sigma_zeta"].reshape(nd)[:, None]
    psi_last = np.einsum("kl,dlm,m->dk", A_comp, v["psi_raw"], A_t[-1]) * v["sigma_psi"].reshape(nd)[:, None]
    zsc = v["z"] * v["sigma_u"][:, None]
    beta = v["theta"] @ data["R_inv"].T
    zf = (v["zf_raw"] @ A_comp.T * v["sigma_zf"].reshape(nd)[:, None]) if a.zf else np.zeros((nd, data["K"]))
    node_r = r["node"].to_numpy().astype(int)
    kr = r["nbhd_id"].map(st["kid"]).to_numpy()
    assert (kr == nb_of[node_r]).all()
    u_r = np.asarray((A_u[node_r] @ zsc.T).T)
    Xr = r[M["xcols"]].to_numpy() - M["xmean"]
    fr = r["is_flat"].astype(bool).to_numpy()
    lat = (v["intercept"] + mu[:, -1])[:, None] + beta @ Xr.T + zeta[:, kr] + psi_last[:, kr] + u_r + zf[:, kr] * fr
    lat_apr = None
    g = np.zeros((nd, 0))
    if a.gcurve:
        g = v["g_raw"] @ data["A_g"].T * v["sigma_g"].reshape(nd)[:, None]
        gr = r["hist_gbin"].astype(int).to_numpy()
        lat = lat + np.where(gr > 0, g[:, np.maximum(gr, 1) - 1], 0.0)
    nvc = np.zeros((nd, data["K"], 0))
    lat_nvc = np.zeros_like(lat)
    if a.nvc or a.np:                                                  # --nvc, --np: neighbourhood deviations, draw x K x j
        nvc = np.einsum("kl,dlj,dj->dkj", A_comp, v["nvc_raw"].reshape(nd, -1, data["n_nvc"]), v["sigma_nvc"].reshape(nd, -1))
        Xv = np_loadings(r, flags=False) if a.np else r[M["nvc_cols"]].to_numpy() - M["nvc_mean"]
        lat_nvc = np.einsum("dij,ij->di", nvc[:, kr, :], Xv)
        lat = lat + lat_nvc
    GX = np.array([0.25, 0.5, 1, 2, 4, 8, 15, 25])                     # years: the fitted curve, reported
    gexp_curve = np.zeros((nd, 0))
    if a.gexp:
        ga, gt = v["gexp_a"].reshape(nd), v["gexp_tau"].reshape(nd)
        sold_r = (r["hist_n_prev"] > 0).to_numpy()
        g_r = np.where(sold_r, r["hist_months"].to_numpy(float) / 12, 0.0)
        assert np.isfinite(g_r).all()
        dl = data["gexp_delta"]
        w_r = 1 - np.exp(-g_r / dl) if dl > 0 else np.ones_like(g_r)
        w_x = 1 - np.exp(-GX / dl) if dl > 0 else np.ones_like(GX)
        lat = lat + (sold_r * w_r)[None, :] * ga[:, None] * np.exp(-g_r[None, :] / gt[:, None])
        gexp_curve = ga[:, None] * w_x[None, :] * np.exp(-GX[None, :] / gt[:, None])
    if a.forecast:                                     # Stan's evaluation-sale values against numpy, then Stan's are used
        hr = np.array(data["h_dev"]) - 1
        # the --np term is not in Stan's lat_dev / lat_dev_apr: compared without it, added after the check
        fc_err["lat_dev_apr"] = float(np.abs(v["lat_dev_apr"] + lat_nvc - lat).max())
        lat_fc_np = lat - lat_nvc - (mu[:, -1:] + psi_last[:, kr]) + v["mu_fc"][:, hr] + v["psi_fc"][:, kr, hr]
        fc_err["lat_dev"] = float(np.abs(v["lat_dev"] - lat_fc_np).max())
        print(f"forecast checks (max |Stan - numpy|): {fc_err}", flush=True)
        assert max(fc_err.values()) < FC_CHECK_TOL, fc_err
        # the forecast innovations implied by Stan's mu_fc must be N(0, 1): trend increments
        # d_h from mu_fc less the season, then e_h = (d_h - delta - phi (d_{h-1} - delta)) / sigma_mu
        sea_fc = v["season"][:, np.array(data["cal_fc"]) - 1]
        trend = np.column_stack([mu[:, -1] - v["season"][:, data["cal"][-1] - 1], v["mu_fc"] - sea_fc])
        dd = np.column_stack([incr_np[:, -1], np.diff(trend, axis=1)])
        dl_, ph_ = v["fc_delta"].reshape(nd, 1), v["fc_phi"].reshape(nd, 1)
        eps = (dd[:, 1:] - dl_ - ph_ * (dd[:, :-1] - dl_)) / v["sigma_mu"][:, None]
        fc_innov = dict(mean=float(eps.mean()), sd=float(eps.std()))
        print(f"implied forecast innovations: {fc_innov}", flush=True)
        assert abs(fc_innov["mean"]) < 0.1 and abs(fc_innov["sd"] - 1) < 0.1, fc_innov
        lat_apr, lat = v["lat_dev_apr"] + lat_nvc, v["lat_dev"] + lat_nvc
        if M["h0"].any():                              # horizon 0 = the fitted value at the last month
            lat = np.where(M["h0"][None, :], lat_apr, lat)

    # ------------------------------------------------------------ checks and diagnostics
    def worst(x):
        return float((np.abs(x.sum(axis=1)) / np.maximum(1.0, np.abs(x).sum(axis=1))).max())
    u_all = np.asarray((A_u @ zsc[:: max(1, nd // 200)].T).T)                  # a thinned set of draws suffices
    G = sp.csr_matrix((np.ones(len(nb_of)), (st["group_of"], np.arange(len(nb_of)))))
    checks = dict(mu_over_months=worst(mu),
                  zeta_over_nbhds=worst(zeta),
                  u_within_each_nbhd=float((np.abs(G @ u_all.T) / np.maximum(1.0, G @ np.abs(u_all).T)).max()))
    if a.forecast:
        checks["season_over_calendar_months"] = worst(v["season"])
    if a.zf:
        checks["zf_over_nbhds"] = worst(zf)
    if a.gcurve:
        checks["g_over_bins"] = worst(g)
    for j in range(nvc.shape[2]):
        checks[f"nvc{j + 1}_over_nbhds"] = worst(nvc[:, :, j])
    c, m = a.chains, a.samples
    scal = (["sigma_mu", "sigma_u", "sigma_zeta", "sigma_psi", "sigEps", "nu"] + (["sigma_zf"] if a.zf else [])
            + (["sigma_g"] if a.gcurve else []) + (["gexp_a", "gexp_tau"] if a.gexp else [])
            + (["fc_delta", "fc_phi", "sigma_s"] if a.forecast else []))
    S = np.column_stack([np.asarray(v[k]).reshape(nd) for k in scal]
                        + ([np.asarray(v["sigma_nvc"]).reshape(nd, -1)] if (a.nvc or a.np) else []))
    scal = scal + [f"sigma_nvc[{c.split(':')[-1]}]" for c in M["nvc_cols"]]
    diag = {}
    comps = dict(scalars=S, latent_pred=lat, beta=beta, mu_last=mu[:, -1:])
    if a.forecast:
        comps.update(mu_fc=v["mu_fc"], season=v["season"])
    if a.gcurve:                                         # mixing of beta_sold + g[b]
        (seg, num), = M["coding"]["segments"].items()       # one segment per fit (separate house and flat models)
        js = M["xcols"].index(f"x:{seg[0]}:hist_sold")
        sd = num["numeric"]["hist_sold"]["sd"]
        comps["premium_sold_plus_g"] = beta[:, [js]] / sd + g
    if a.gexp:                                           # the fitted premium at GX years (+ c = hist_sold's coefficient, expc)
        sold_cols = [c for c in M["xcols"] if c.endswith(":hist_sold")]
        cc = np.zeros((nd, 1))
        if sold_cols:
            (seg, num), = M["coding"]["segments"].items()
            cc = beta[:, [M["xcols"].index(sold_cols[0])]] / num["numeric"]["hist_sold"]["sd"]
        comps["premium_gexp"] = gexp_curve + cc
    for k, x in comps.items():
        x = x[:, x.std(axis=0) > 1e-12].reshape(c, m, -1)
        rh, eb, et = md.rhat(x), md.ess_bulk(x), md.ess_tail(x)
        diag[k] = dict(k=int(x.shape[2]), rhat_max=float(rh.max()), share_rhat_gt_1_01=float((rh > 1.01).mean()),
                       ess_bulk_min=float(eb.min()), ess_tail_min=float(et.min()))
        if k == "scalars":
            diag["scalars_each"] = {s: dict(rhat=float(rh[i]), ess_bulk=float(eb[i]), ess_tail=float(et[i]),
                                            mean=float(S[:, i].mean()), sd=float(S[:, i].std())) for i, s in enumerate(scal)}
    sv = fit.method_variables()
    en = sv["energy__"].T
    ebfmi = (np.diff(en, axis=1) ** 2).mean(axis=1) / en.var(axis=1)

    pred = np.exp(lat)
    out = r[KEYS + ["date", "meta_sale_price", "meta_class", "meta_township_code", "nbhd_id", "is_flat",
                    "has_training_sale"]].copy()
    out["pred"] = np.median(pred, axis=0)
    out["pred_q05"], out["pred_q95"] = np.quantile(pred, [0.05, 0.95], axis=0)
    xx = lat.reshape(c, m, -1)
    out["latent_rhat"] = md.rhat(xx)
    if a.forecast:
        out["pred_apr"] = np.median(np.exp(lat_apr), axis=0)          # diagnostic: every time term held at April 2023
    out["town"] = a.town
    out.to_parquet(run / "predictions.parquet", index=False)
    np.savez_compressed(run / "draws.npz", scalars=S, scalar_names=np.array(scal), beta=beta, xcols=np.array(M["xcols"]),
                        latent=lat.astype(np.float32), mu=mu, zeta=zeta, zf=zf, g=g,
                        gexp_curve=gexp_curve, gexp_years=GX, nvc=nvc.astype(np.float32), nvc_cols=np.array(M["nvc_cols"]),
                        nbhds=np.array([str(x) for x in st["nbhds"]]),
                        **(dict(latent_apr=lat_apr.astype(np.float32), mu_fc=v["mu_fc"], season=v["season"],
                                psi_fc=v["psi_fc"].astype(np.float32), h_dev=np.array(data["h_dev"]))
                           if a.forecast else {}))
    ratio = out["pred"] / out["meta_sale_price"]
    rep = dict(name=a.name, town=a.town, prep=a.prep, zf=a.zf, gcurve=a.gcurve, gexp=a.gexp, nvc=M["nvc_cols"], np=a.np,
               forecast=(dict(delta_sd=FC_DELTA_SD, sigma_s_sd=FC_SIGMA_S_SD, H=data["H"], checks=fc_err, innovations=fc_innov,
                              **(dict(n_horizon0=int(M["h0"].sum())) if COUNTY else {}))
                         if a.forecast else None),
               gexp_prior=(dict(tau_median=float(np.exp(data["gexp_tau_mu"])), tau_sd_log=data["gexp_tau_sd"], a_sd=data["gexp_a_sd"],
                               delta=data["gexp_delta"]) if a.gexp else None), seed=a.seed, method="nuts", init="mode+jitter",
               settings=dict(chains=c, warmup=a.warmup, samples=m, max_treedepth=a.max_treedepth), stan_file=str(STAN.relative_to(REPO)),
               node_type="building" if M["buildings"] else "block", n_train=data["N"], n_dev=len(r),
               n_dev_flats=int(fr.sum()), n_nodes=len(M["units"]), n_nodes_with_training_sale=len(M["obs_units"]),
               n_block_graphs=st["n_block_graphs"], n_block_latent=st["nz"], n_bridges=len(st["bridges"]),
               n_nbhds=data["K"], n_regressors=data["Nr"], coding_sha256=M["pman"].get("coding_sha256"),
               prep_outputs=M["pman"]["outputs"],
               timing=dict(structure_s=round(st["seconds"], 1), optimize_s=round(t_opt, 1), sample_s=round(t_samp, 1),
                           runtime_s=round(time.time() - t_all + st["seconds"], 1)),
               sum_to_zero=dict(checks=checks, all_pass=bool(all(x < 1e-8 for x in checks.values()))),
               diagnostics=dict(divergent=int(sv["divergent__"].sum()), max_treedepth_share=float((sv["treedepth__"] >= a.max_treedepth).mean()),
                                ebfmi=[float(x) for x in ebfmi], by_component=diag,
                                note="rank-normalised split R-hat, bulk/tail ESS (mcmc_diag.py); targets R-hat <= 1.01, ESS >= 400"),
               dev=dict(mdape=float((ratio - 1).abs().median()), median_ratio=float(ratio.median())),
               versions=dict(cmdstan=cmdstanpy.cmdstan_version(), cmdstanpy=cmdstanpy.__version__,
                             python=platform.python_version()),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               stan_sha256=hashlib.sha256(STAN.read_bytes()).hexdigest())
    (run / "diagnose.txt").write_text(fit.diagnose())
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    shutil.rmtree(run / "chains/mode", ignore_errors=True)
    print(json.dumps(dict(dev=rep["dev"], sum_to_zero=rep["sum_to_zero"]["all_pass"], divergent=rep["diagnostics"]["divergent"],
                          rhat={k: d["rhat_max"] for k, d in diag.items() if k != "scalars_each"},
                          ess_bulk={k: d["ess_bulk_min"] for k, d in diag.items() if k != "scalars_each"},
                          runtime_min=rep["timing"]["runtime_s"] / 60), indent=1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
