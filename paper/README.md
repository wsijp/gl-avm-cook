# Scripts for the paper's tables

These scripts compute the numbers in the paper's headline table and its regressivity results ("An open Bayesian spatio-temporal hedonic model against a county assessor's production model: evidence from Cook County"). They read stored outputs only: the evaluation sales, the hedonic model's values and predictions, and the Assessor's model's values and predictions, all written by the repository's own steps (`public/reproduce.sh all` and `public/reproduce.sh county final`). They fit nothing and change nothing outside this folder.

| Script | What it makes | Output |
|---|---|---|
| `table2.py` | Table 2: accuracy, level, uniformity and regressivity for both models in each test, and the differences between them with 90% intervals | `table2.json`, `table2.md` |
| `make_table2_tex.py` | Table 2 in LaTeX, from `table2.json` | `table2.tex` |
| `vei_cook.py` | The VEI of each model with its significance, and the PRD, quoted in the text | `vei_results.json` |
| `vei.py` | The VEI calculation, used by the other scripts; `--selftest` reproduces the worked example of the IAAO draft standard | — |

The stored outputs in this folder are the ones quoted in the paper.

## How to run

From the working folder that holds this repository as `public/` (see the main README), after the main steps have run:

```bash
.venv/bin/python -I public/paper/vei.py --selftest   # seconds
.venv/bin/python -I public/paper/table2.py           # about a minute; 2,000 bootstrap draws
.venv/bin/python -I public/paper/vei_cook.py         # seconds
.venv/bin/python -I public/paper/make_table2_tex.py
```

`table2.py` first checks that its MdAPE, median ratio and COD equal the values stored by the repository's evaluation steps, and stops if they do not. The bootstrap uses a fixed seed, so a rerun gives the same intervals.

## The tests

- **1 January assessment tests:** values at 1 January 2023 (south and west suburbs) and 1 January 2022 (north suburbs), compared with sales during the year, whose prices are adjusted to 1 January with a price index. The hedonic model is the published model; the Assessor's model is our rebuild, before its values are rounded.
- **County sale-prediction test:** the Assessor's own test of its 2025 model (38,177 sales from October 2023 to December 2024), for the whole county, Chicago and the suburbs. The hedonic model is the sale-prediction version; the Assessor's predictions are before rounding.

## The measures

The ratio of a home is its value (or predicted price) divided by its sale price.

- **MdAPE (median absolute percentage error):** the median of |ratio − 1|, in per cent: the typical error.
- **Median ratio:** the median of the ratios; 1.00 means the values are at the right overall level, 0.95 about 5% low.
- **COD (coefficient of dispersion):** the average absolute difference between each ratio and the median ratio, as a percentage of the median ratio; lower means more uniform values.
- **VEI (vertical equity indicator):** from the IAAO's draft *Standard on Ratio Studies* (exposure draft, May 2026, Appendix E). Sales are sorted by a proxy that gives equal weight to price and value and split into groups (ten groups for more than 500 sales); the VEI is the median ratio of the dearest group minus that of the cheapest, as a percentage of the overall median ratio. Negative values mean the model is regressive: cheaper homes are valued higher relative to their prices than dearer homes. The draft accepts −10 to +10. When the VEI is outside that range and the 90% intervals of the two end groups do not overlap, its significance is the gap between those intervals, again as a percentage of the median ratio.
- **PRD (price-related differential):** the mean ratio divided by total value over total sale price; above 1 suggests regressivity. It is the measure of the IAAO's 2013 standard.
- **MdAPE with the level removed:** the MdAPE after dividing each model's values by its own median ratio on the same sales. It shows how much of a difference in MdAPE comes from the overall level. Neither model could make this correction in advance, since it uses the test sales.

Differences are the hedonic model minus the Assessor's model. Their 90% intervals come from a paired bootstrap: within each township, whole clusters of sales are drawn with replacement (valuation groups in the 1 January tests, properties in the county test), the same draw for both models, 2,000 times. These intervals can differ slightly in the second decimal from those of the repository's own evaluation steps, which use their own random draws.

In the JSON and Markdown outputs, `gl` and "GL" denote the hedonic model. In `table2.json`, `lf_mdape` is the MdAPE with the level removed, and `unadjusted` repeats the 1 January comparison against prices as paid, without adjustment to 1 January.

## The VEI self-test

The draft prints the proxy as 0.50 × price + value ÷ median ratio, but its text asks for equal weight on price and value, which needs 0.5 on both terms. With equal weight, `vei.py` reproduces the draft's worked example exactly (group sizes, the four group medians, and VEI 48.53 and 51.43 for the two percentile methods); with the formula as printed, the middle groups do not match. The worked example's groups are too small for the interval method used here, so the self-test does not check the intervals; every group in the Cook County tests is large enough.
