"""The paper's Table 2, the evaluation sales: what the homes and prices are like, per test.

For each evaluation sample: number of sales; sale price as paid (median and interquartile range); building area,
land area and age (medians); share of two- to six-flat buildings (Assessor classes 211 and 212).
Characteristics come from the Assessor's own files: the 2023 and 2022 assessment data (1 January tests) and the 2025
training file (county test). Reads stored outputs and the downloaded inputs only; writes paper/table_data.json and
paper/table_data.tex.

Usage, from the working folder that holds this repository as public/:
    .venv/bin/python -I public/paper/table_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # the working folder (this file is public/paper/table_data.py)
CH = REPO / "public/runs/v2/struct_eval/chosen"
OUT = HERE / "table_data.tex"
FLATS = {"211", "212"}
CHARS = ["meta_pin", "meta_card_num", "meta_class", "char_bldg_sf", "char_land_sf", "char_yrblt"]


def summarise(price, bldg, land, age, cls):
    q = lambda x: np.nanpercentile(x, [25, 50, 75])
    p25, p50, p75 = q(price)
    return dict(n=int(len(price)), price_p25=float(p25), price_median=float(p50), price_p75=float(p75),
                bldg_sf=float(np.nanmedian(bldg)), land_sf=float(np.nanmedian(land)), age=float(np.nanmedian(age)),
                share_flats=float(np.mean([str(c) in FLATS for c in cls])),
                missing_chars=int(np.isnan(bldg).sum() + np.isnan(age).sum()))


def jan(year):
    head = json.loads((CH / f"ratio_{year}_primary.json").read_text())
    ev = pd.concat([pd.read_parquet(REPO / p) for p in head["evals"]], ignore_index=True)
    a = pd.read_parquet(REPO / f"ccao/inputs/{year}/assessment_data.parquet", columns=CHARS)
    a = a.sort_values("meta_card_num").drop_duplicates("meta_pin")  # eligible homes have one building record
    m = ev.merge(a, on="meta_pin", how="left")
    assert len(m) == len(ev)
    age = int(year) - m["char_yrblt"].astype(float)
    return summarise(m["price"].to_numpy(float), m["char_bldg_sf"].to_numpy(float),
                     m["char_land_sf"].to_numpy(float), age.to_numpy(float), m["meta_class"])


def county():
    t = pd.read_parquet(REPO / "public/runs/v2/county2025_ixa_eval/test_sales.parquet")
    tr = pd.read_parquet(REPO / "ccao/inputs/2025/training_data.parquet",
                         columns=CHARS + ["meta_sale_document_num"])
    tr = tr.drop_duplicates(["meta_pin", "meta_sale_document_num", "meta_card_num"])
    m = t.merge(tr.drop(columns=["meta_class"]), on=["meta_pin", "meta_sale_document_num", "meta_card_num"],
                how="left")
    assert len(m) == len(t)
    age = pd.to_datetime(m["date"]).dt.year - m["char_yrblt"].astype(float)
    return summarise(m["meta_sale_price"].to_numpy(float), m["char_bldg_sf"].to_numpy(float),
                     m["char_land_sf"].to_numpy(float), age.to_numpy(float), m["meta_class"])


def main():
    res = {"1 January 2023": jan("2023"), "1 January 2022": jan("2022"), "County test": county()}
    (HERE / "table_data.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)
    cols = list(res)
    k = lambda x: f"{x / 1000:,.0f}"
    rows = [
        ("Sales", [f"{res[c]['n']:,}" for c in cols]),
        ("Sale price, median (\\$000)", [k(res[c]["price_median"]) for c in cols]),
        ("Sale price, interquartile range (\\$000)", [f"{k(res[c]['price_p25'])}--{k(res[c]['price_p75'])}" for c in cols]),
        ("Building area, median (sq ft)", [f"{res[c]['bldg_sf']:,.0f}" for c in cols]),
        ("Land area, median (sq ft)", [f"{res[c]['land_sf']:,.0f}" for c in cols]),
        ("Age, median (years)", [f"{res[c]['age']:.0f}" for c in cols]),
        ("Two- to six-flat buildings (\\%)", [f"{100 * res[c]['share_flats']:.0f}" for c in cols]),
    ]
    body = "\n".join(f"{name} & " + " & ".join(v) + " \\\\" for name, v in rows)
    OUT.write_text(r"""\begin{table}[htbp]
\centering\small
\caption{The evaluation sales.}\label{tab:data}
\begin{tabular}{@{}lrrr@{}}
\toprule
 & 1 January 2023 & 1 January 2022 & County test \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\par\smallskip\footnotesize\raggedright Sale prices as paid, not adjusted for time. Characteristics as recorded by the Assessor; age at the
valuation date (1 January tests) or in the year of sale (county test). The 1 January tests cover the southern and
western suburbs (2023) and the northern suburbs (2022); the county test covers all 38 townships.
\end{table}
""")
    print("written", OUT)


if __name__ == "__main__":
    main()
