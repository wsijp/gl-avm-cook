"""Vertical Equity Indicator (VEI), IAAO Standard on Ratio Studies, exposure draft May 2026, Appendix E.

Standalone. vei(value, price) follows Appendix E.1 steps 1-7:
  1. ratio = value / price (price may be time-adjusted)
  2. proxy = 0.5*price + 0.5*value/median(ratio)   [equal weight to price and value; see note below]
  3. sort by proxy; split ratios into percentile groups: 2 if n 20-50, 4 if 51-500, 10 if >= 501,
     with group boundaries by the R7 (default) or R6 percentile-rank method of Appendix E.3
  4. median ratio and 90% median confidence interval per group (Appendix D.2 for n > 30; D.4 binomial
     table for smaller groups is not implemented -- groups here are large)
  5. VEI = 100 * (median of last group - median of first group) / overall median ratio
  6-7. if |VEI| > 10 and the first and last groups' intervals do not overlap,
       VEI significance = 100 * (lower limit of the higher group - upper limit of the lower group) / median

Note on step 2: the draft prints "Proxy = (0.50*SP) + (AV/Median Ratio)", but its text asks for equal weight
to price and value, which needs 0.5 on both terms. The self-test confirms it: the equal-weight proxy
reproduces the draft's worked example (Tables 25 and 27: group sizes, all four group medians, VEI 48.53 and
51.43) exactly; the formula as printed gets the middle groups wrong. Equal weight is the default.
The self-test does not validate the intervals: the example's groups have 13-14 sales and need the D.4
binomial table, which is not implemented. Cook County's groups are all large, where D.2 applies.

Run `python3 public/paper/vei.py --selftest` to reproduce Appendix E.4.
"""
import math
import sys

import numpy as np


def group_bounds(n, k, method="R7"):
    """Last index (1-based) of each of k percentile groups, Appendix E.3."""
    hs = []
    for i in range(1, k + 1):
        p = i / k
        h = math.floor(p * (n - 1)) + 1 if method == "R7" else math.floor(p * (n + 1))
        hs.append(min(h, n))
    return hs


def median_ci(r, z=1.645):
    """90% median confidence interval by Appendix D.2 (rank counting from the median)."""
    r = np.sort(np.asarray(r, float))
    n = len(r)
    j = z * math.sqrt(n) / 2 + (0.5 if n % 2 == 0 else 0.0)
    j = math.ceil(j)
    if n % 2:                       # median at rank m (1-based)
        m = (n + 1) // 2
        lo, hi = m - j, m + j
    else:                           # median between ranks n/2 and n/2+1
        lo, hi = n // 2 - j + 1, n // 2 + j
    lo, hi = max(lo, 1), min(hi, n)
    return r[lo - 1], r[hi - 1]


def n_groups(n):
    if n < 20:
        return None
    return 2 if n <= 50 else (4 if n <= 500 else 10)


def vei(value, price, method="R7", proxy="equal", k=None):
    value, price = np.asarray(value, float), np.asarray(price, float)
    ratio = value / price
    med = float(np.median(ratio))
    if proxy == "equal":
        px = 0.5 * price + 0.5 * value / med
    else:                           # the formula exactly as printed
        px = 0.5 * price + value / med
    order = np.argsort(px, kind="stable")
    r_sorted, px_sorted = ratio[order], px[order]
    k = k or n_groups(len(ratio))
    bounds = group_bounds(len(ratio), k, method)
    groups, start = [], 0
    for b in bounds:
        g = r_sorted[start:b]
        lo, hi = median_ci(g)
        groups.append(dict(n=len(g), proxy_median=float(np.median(px_sorted[start:b])),
                           median=float(np.median(g)), lo=float(lo), hi=float(hi)))
        start = b
    first, last = groups[0], groups[-1]
    v = 100 * (last["median"] - first["median"]) / med
    overlap = not (first["hi"] < last["lo"] or last["hi"] < first["lo"])
    sig = None
    if abs(v) > 10 and not overlap:
        top, bot = (last, first) if last["median"] > first["median"] else (first, last)
        sig = 100 * (top["lo"] - bot["hi"]) / med
    return dict(vei=v, significance=sig, overlap=overlap, median=med, groups=groups)


# Appendix E.4 data: Section 6.1, Table 1 (assessed value, sale price), 54 observations
TABLE1 = [(240000, 690000), (144000, 296250), (392000, 787500), (199200, 372000), (340800, 574500),
          (472000, 795000), (336000, 559500), (284800, 465000), (436000, 693600), (191200, 298500),
          (225000, 350000), (194000, 295000), (481600, 732000), (338400, 495000), (164800, 237000),
          (252800, 352500), (945000, 1289000), (280000, 379000), (407000, 540000), (423000, 560000),
          (306800, 390000), (236800, 300000), (361000, 450000), (290400, 345000), (235200, 277500),
          (415000, 485000), (381000, 442000), (380000, 435000), (875000, 998000), (680000, 772500),
          (329000, 365000), (457000, 487000), (516000, 547500), (295200, 300000), (580000, 580000),
          (840000, 840000), (640000, 622500), (660000, 637500), (354000, 340000), (800000, 750000),
          (390000, 352500), (800000, 705000), (469000, 410000), (1000000, 859500), (920000, 787500),
          (527000, 440000), (800000, 648000), (786000, 630000), (496000, 388500), (1000000, 765000),
          (320000, 243750), (960000, 720000), (952000, 705000), (327200, 240000)]

# Appendix E.4, Table 25 (R7) and Table 27 (R6): group sizes, group medians, VEI
EXPECTED = {"R7": dict(n=[14, 13, 13, 14], med=[0.7280, 0.8620, 0.7537, 1.1491], vei=48.53),
            "R6": dict(n=[13, 14, 14, 13], med=[0.7172, 0.8588, 0.7545, 1.1635], vei=51.43)}


def selftest():
    av, sp = map(np.array, zip(*TABLE1))
    ok = True
    for proxy in ("equal", "as_printed"):
        for method, exp in EXPECTED.items():
            res = vei(av, sp, method=method, proxy=proxy)
            ns = [g["n"] for g in res["groups"]]
            meds = [round(g["median"], 4) for g in res["groups"]]
            match = ns == exp["n"] and np.allclose(meds, exp["med"], atol=6e-4) and \
                abs(res["vei"] - exp["vei"]) < 0.06
            print(f"proxy={proxy:10s} {method}: n={ns} medians={meds} VEI={res['vei']:.2f} "
                  f"(expected {exp['vei']}) median={res['median']:.4f} -> {'MATCH' if match else 'no match'}")
            ok &= match if proxy == "equal" else True
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
