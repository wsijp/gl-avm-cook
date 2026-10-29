"""Write the paper's Table 2 in LaTeX (paper/table2.tex) from paper/table2.json (made by table2.py).
No numbers typed by hand.

Layout: two rows per test (hedonic model; Assessor's model), differences on the hedonic row.
Usage, from the working folder: python3 public/paper/make_table2_tex.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "table2.tex"
LABEL = {"A. 1 January 2023": "1 January 2023", "A. 1 January 2022": "1 January 2022",
         "B. County, all": "County, all", "B. County, City": "Chicago", "B. County, suburbs": "Suburbs"}


def num(x, f):
    return f.format(x).replace("-", "$-$")


def pt(d, f="{:+.2f}"):
    return num(d['point'], f)


def iv(d, f="{:+.2f}"):
    return f"[{num(d['lo'], f)}, {num(d['hi'], f)}]"


def main():
    r = json.loads((HERE / "table2.json").read_text())
    rows = []
    for k, v in r.items():
        if k == "A. 1 January 2023":
            rows.append("\\multicolumn{10}{@{}l}{\\textit{A. 1 January assessment tests}}\\\\[2pt]")
        if k == "B. County, all":
            rows.append("\\midrule\n\\multicolumn{10}{@{}l}{\\textit{B. County sale-prediction test}}\\\\[2pt]")
        g, a = v["gl"], v["assessor"]
        rows.append(f"{LABEL[k]} & {v['n']:,} & Hedonic & {g['mdape']:.2f} & {g['median_ratio']:.3f} & {g['cod']:.1f} & "
                    f"{num(g['vei'], '{:.1f}')} & {pt(v['d_mdape'])} & {pt(v['d_lf_mdape'])} & "
                    f"{pt(v['d_vei'], '{:+.1f}')} \\\\")
        rows.append(f" & & Assessor & {a['mdape']:.2f} & {a['median_ratio']:.3f} & {a['cod']:.1f} & "
                    f"{num(a['vei'], '{:.1f}')} & {iv(v['d_mdape'])} & {iv(v['d_lf_mdape'])} & "
                    f"{iv(v['d_vei'], '{:+.1f}')} \\\\[3pt]")
    body = "\n".join(rows)
    OUT.write_text(r"""\begin{table}[htbp]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Headline results, and differences (hedonic minus Assessor) with paired 90\% intervals (in brackets, second row).}\label{tab:results}
\begin{tabular}{@{}lrlrrrrccc@{}}
\toprule
 & & & & & & & \multicolumn{3}{c}{Difference, hedonic minus Assessor} \\
\cmidrule(l){8-10}
Test & Sales & Model & \makecell[r]{MdAPE\\(\%)} & \makecell[r]{Median\\ratio} & COD & VEI & MdAPE & \makecell{MdAPE,\\level removed} & VEI \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\par\smallskip\footnotesize\raggedright MdAPE: median absolute percentage error. Median ratio: median of value / price.
COD: coefficient of dispersion. VEI: vertical equity indicator (IAAO 2026 draft); negative means regressive,
acceptable range $-10$ to $+10$. Level removed: each model divided by its own median ratio on the same sales
(retrospective). Values before rounding; 1 January sale prices adjusted to 1 January. Differences are computed from unrounded figures, so they may differ in the last digit from the rounded values shown.
\end{table}
""")
    print("written", OUT)


if __name__ == "__main__":
    main()
