"""Rank-normalised MCMC diagnostics, vectorised over many quantities (Vehtari, Gelman, Simpson, Carpenter and
Bürkner 2021, "Rank-normalization, folding, and localization: an improved R-hat", Bayesian Analysis 16(2)).

Follows the R package `posterior` (rhat, ess_bulk, ess_tail, mcse_quantile) and Stan's ESS code: split chains,
rank normalisation with average ranks and the (r - 3/8) / (S + 1/4) offset, Geyer's initial positive and monotone
sequence with the antithetic correction, tau floored at 1 / log10(S).

Arrays are x[chain, draw, quantity]; every function returns one value per quantity. Used by fit_struct.py and
struct_convergence.py; checked against CmdStan's own summary (R_hat, ESS_bulk, ESS_tail) on the scalar parameters.
"""
import numpy as np
from scipy.special import ndtri
from scipy.stats import beta, rankdata


def split_chains(x):
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, x.shape[1] - n:]], axis=0)


def z_scale(x):
    c, n, k = x.shape
    r = rankdata(x.reshape(c * n, k), axis=0, method="average")
    return ndtri((r - 0.375) / (c * n + 0.25)).reshape(c, n, k)


def rhat_basic(x):
    """Split chains already applied."""
    n = x.shape[1]
    W = x.var(axis=1, ddof=1).mean(axis=0)
    B = x.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((n - 1) / n * W + B) / W)


def ess_basic(x):
    """ESS of the mean of x (split chains already applied), Stan's/posterior's estimator."""
    m, n, k = x.shape
    yc = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(yc, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n, axis=1)[:, :n] / n          # biased autocovariance, (m, n, k)
    chain_var = acov[:, 0] * n / (n - 1)
    mean_var = chain_var.mean(axis=0)
    var_plus = mean_var * (n - 1) / n + (x.mean(axis=1).var(axis=0, ddof=1) if m > 1 else 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1 - (mean_var[None] - acov.mean(axis=0)) / var_plus[None]       # (n, k)
    rho[0] = 1.0                                                              # as posterior
    kmax = (n - 4) // 2                                                       # last pair the loop can reach
    P = rho[0:2 * kmax + 1:2] + rho[1:2 * kmax + 2:2]                         # pair sums P_0 .. P_kmax
    even = rho[0:2 * kmax + 1:2]
    stop = ~(P[1:] > 0)                                                       # P_k <= 0 or NaN, k >= 1
    has = stop.any(axis=0)
    k_end = np.where(has, stop.argmax(axis=0) + 1, kmax)                      # first pair not used
    Pm = np.minimum.accumulate(P, axis=0)                                     # Geyer's monotone sequence
    use = np.arange(P.shape[0])[:, None] < k_end[None]
    s = np.where(use, Pm, 0).sum(axis=0)
    last = even[k_end, np.arange(k)]
    last = np.where(has, np.maximum(last, 0), last)
    tau = -1 + 2 * s + last
    S = m * n
    tau = np.maximum(tau, 1 / np.log10(S))
    return S / tau


def rhat(x):
    """Rank-normalised split R-hat: the larger of the bulk and folded versions."""
    s = split_chains(x)
    med = np.median(x.reshape(-1, x.shape[2]), axis=0)
    return np.maximum(rhat_basic(z_scale(s)), rhat_basic(z_scale(split_chains(np.abs(x - med)))))


def ess_bulk(x):
    return ess_basic(z_scale(split_chains(x)))


def _ess_quantile(x, prob):
    q = np.quantile(x.reshape(-1, x.shape[2]), prob, axis=0)
    return ess_basic(split_chains((x <= q).astype(float)))


def ess_tail(x):
    return np.minimum(_ess_quantile(x, 0.05), _ess_quantile(x, 0.95))


def mcse_quantile(x, prob=0.5):
    """Monte Carlo standard error of the `prob` quantile (posterior::mcse_quantile)."""
    ess = _ess_quantile(x, prob)
    S = x.shape[0] * x.shape[1]
    a1 = beta.ppf(0.1586553, ess * prob + 1, ess * (1 - prob) + 1)
    a2 = beta.ppf(0.8413447, ess * prob + 1, ess * (1 - prob) + 1)
    srt = np.sort(x.reshape(S, -1), axis=0)
    i1 = np.clip(np.floor(a1 * S).astype(int), 1, S) - 1
    i2 = np.clip(np.ceil(a2 * S).astype(int), 1, S) - 1
    cols = np.arange(srt.shape[1])
    return (srt[i2, cols] - srt[i1, cols]) / 2
