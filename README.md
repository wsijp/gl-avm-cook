# An open house valuation model, validated on Cook County IL

This repository contains the code of the **GL model**, an open Bayesian house valuation model, and a comparison of it with the Cook County Assessor’s own valuation model in Cook County, Illinois.

![Value of a standard home per census block across Cook County, including Chicago](figures/location_map.png)

<p align="center"><sub>Value of one standard home by census block across Cook County in October 2023, estimated by the GL model (<a href="#about-the-map">about the map</a>).</sub></p>

*Status: first public release; results are preliminary.*

## Part 1: the GL model and its code

The GL model is the spatio-temporal house price model published by [Sijp and Francke (2026) in *Real Estate Economics*](https://onlinelibrary.wiley.com/doi/full/10.1111/1540-6229.70062). It is a Bayesian hedonic regression fitted in [Stan](https://mc-stan.org): the log sale price is explained by the home’s characteristics, such as size and age, a monthly price index, neighbourhood price levels, neighbourhood price paths over time, and fine-grained location effects for each 2020 census block. Blocks with few sales borrow information from neighbouring blocks. “GL” stands for graph Laplacian, the mathematical device that connects those neighbours. Every value splits exactly into these parts (characteristics, price index, neighbourhood, block), so it can be explained term by term; the smoothing decides how much each part borrows from neighbouring areas, not whether it can be read.

The model code can be used as a house valuation model more generally, fitted to other housing markets with sales, home characteristics and a spatial layout. A [related working paper (Sijp 2026, SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7345003) compares the GL model with machine-learning models on Sydney house sales.

### Getting started

The repository runs two models on the same public data: the GL model, whose code is here, and, for comparison, the Cook County Assessor’s own valuation model. The Assessor publishes that model’s code ([ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm)) but not its fitted runs, so the scripts here rebuild it from their code and public inputs for each year, run it unchanged in Docker, and do not redistribute it; the rebuild reproduces about three quarters of the Assessor’s published values exactly. [REPRODUCE.md](REPRODUCE.md) lists the prerequisites, the commands for each stage (including the Docker memory setting for the Assessor’s model), their runtimes and where every result is written. The scripts expect this repository to be in a subfolder named `public` inside a working folder:

```bash
mkdir work && cd work
git clone https://github.com/wsijp/gl-avm-cook.git public
public/reproduce.sh setup   # Python environment and CmdStan 2.39.0
public/reproduce.sh all     # download, prepare, rebuild the Assessor's model, fit, evaluate
```

On an Apple M5 laptop with 32 GB of memory, running three townships at a time, the GL fits take about 3½ hours for the whole county (38 townships) and about 1 hour 40 minutes for the 30 townships of the 1 January tests.

### What is where

| File or folder | Purpose |
|---|---|
| [`stan/v2/gl_struct.stan`](stan/v2/gl_struct.stan) | The GL model |
| [`scripts/v2/`](scripts/v2/) | Python scripts for data preparation, fitting, valuation, evaluation and figures |
| [`scripts/v2/ay2022/`](scripts/v2/ay2022/) | Preparation and valuation scripts adapted for 2022 |
| [`assessor/`](assessor/) | Dockerfiles and the script that fetches and runs the Assessor’s model |
| [`figures/`](figures/) | Results figures |
| [`reproduce.sh`](reproduce.sh) | Runs the reproduction steps |
| [`REPRODUCE.md`](REPRODUCE.md) | Commands, software versions, runtimes and output locations |
| [`requirements.txt`](requirements.txt), [`inputs.sha256`](inputs.sha256) | Python dependencies and checksums for downloaded inputs |
| [`LICENSE`](LICENSE), [`CITATION.cff`](CITATION.cff) | Licence and citation information |

No data are included. The scripts download public inputs and create `data/` and `runs/` folders, which are ignored by git.

## Part 2: comparison with the Assessor’s model

We compare GL with the Assessor’s model in two settings: predicting later sale prices across Cook County, including Chicago, and valuing suburban homes at 1 January before that year’s sales are known.

**In predicting sale prices across the county, the Assessor’s model is more accurate, mostly in Chicago.** On the Assessor’s own test of its 2025 model (38,177 sales from October 2023 to December 2024), the typical error (MdAPE) of the GL predictions is 15.11% against 14.01% for the Assessor’s model: 1.10 percentage points higher (90% interval +0.96 to +1.23). The gap is 2.13 points in Chicago (+1.83 to +2.39) and 0.36 points in the suburbs (+0.25 to +0.49).

**In valuing homes at 1 January, the GL values sit at the right overall level and the Assessor’s about 5% low, which makes the GL about one point more accurate than our rebuild of the Assessor’s model.** Once each model’s level is corrected in advance, there is no clear difference in typical error.

### Results at a glance

- **MdAPE (median absolute percentage error):** the typical absolute gap between value and sale price, as a percentage of price; lower is better.
- **Median ratio:** the median of value ÷ sale price; 1.00 means the right overall level and 0.95 means about 5% low.
- **COD (coefficient of dispersion):** the average distance of ratios from their median, as a percentage of that median; lower means more uniform values.
- **PRD (price-related differential):** mean ratio ÷ (total value ÷ total sale price); above 1 indicates regressivity: cheaper homes receive higher values relative to their sale prices than dearer homes.

| Test and area | Model | Sales | MdAPE, % | Median ratio | COD | PRD |
|---|---|---:|---:|---:|---:|---:|
| **Sale prediction, whole county,** Oct 2023 – Dec 2024 | GL | 38,177 | 15.11 | 0.942 | 22.6 | 1.073 |
| | Assessor’s 2025 model | | 14.01 | 0.950 | 22.6 | 1.087 |
| ↳ Chicago | GL | 15,677 | 19.43 | 0.934 | 29.5 | 1.098 |
| | Assessor’s 2025 model | | 17.30 | 0.964 | 29.2 | 1.127 |
| ↳ Suburbs | GL | 22,500 | 12.76 | 0.945 | 17.9 | 1.055 |
| | Assessor’s 2025 model | | 12.40 | 0.945 | 17.7 | 1.059 |
| **1 January assessment, south and west suburbs,** 2023 | GL | 11,356 | 14.60 | 0.988 | 22.8 | 1.074 |
| | Assessor, our rebuild | | 15.67 | 0.944 | 23.6 | 1.072 |
| **1 January assessment, north suburbs,** 2022 | GL | 10,825 | 9.52 | 0.993 | 12.9 | 1.032 |
| | Assessor, our rebuild | | 10.39 | 0.954 | 13.3 | 1.028 |

Compare models within each test; the tests use different sales, dates and evaluation rules. In the sale-prediction test, values are compared with the prices actually paid; in the 1 January tests, with sale prices adjusted to 1 January. Chicago and the suburbs are parts of the county row, not further tests.

### Predicting sales across Cook County

The Assessor tests each year’s model by training it on the earlier 90% of its cleaned sales and predicting the latest 10%. We used its test for the 2025 model: training sales from January 2016 to 20 October 2023 and 38,177 test sales from 20 October 2023 to 27 December 2024, in all 38 townships, including Chicago’s eight. GL was fitted to each township’s training sales and predicted every test sale with its price index held at October 2023, the last training month; it learns nothing about prices after that. The Assessor’s predictions are those of its 2025 model, run unchanged on the same training sales.

The two models do not use the same information. Both receive each home’s characteristics from the Assessor’s 2025 data and the number of its other sales in the four years before the sale. The Assessor’s model, fitted across the whole county, also uses 68 further predictors (95 in all), among them location codes, distances to amenities and other surrounding-area measures, census statistics, parcel shape and the sale date. GL instead estimates location from each township’s own sales, by neighbourhood and census block.

GL is less accurate county-wide, by 1.10 MdAPE points, and the gap is mostly in Chicago (+2.13), where GL’s MdAPE is higher in seven of the eight townships, clearly so in six. In the suburbs the two models are close (+0.36): the difference is within half a point either way in 14 of the 30 suburban townships, GL is clearly more accurate (the 90% interval lies below zero) in Hanover and Schaumburg, and clearly less accurate in 9, most of all Rich, Calumet and Cicero. The largest gaps are in lower-priced areas where GL’s predictions fall well below the prices (median ratio 0.87–0.91 in Calumet, Rich, Hyde Park, Lake and Cicero), but the two dearest Chicago townships, Lake View and North Chicago, also show gaps of 2.65 points. The gap is not only a difference in overall level: after each model’s predictions are divided by its own median ratio on the test sales, GL’s MdAPE is still 1.16 points higher (+1.04 to +1.31). COD is the same for both models (22.6), and GL’s predictions are slightly less regressive (PRD 1.073 against 1.087).

Why the Assessor’s model does better in Chicago has not been tested; its additional location, surrounding-area, census and sale-date predictors are obvious candidates. This is a new comparison on an existing test set: an earlier version of this project used the same test sales for 14 townships with an earlier version of GL (where GL was 0.27 points less accurate), and what it learned informed later model development. The scripts of this test are not included in this release.

### Valuing homes at 1 January

An assessor must value every home on the roll at a fixed date, 1 January in Cook County, before the year’s sales are known. We value the same homes (single-family houses, two- to six-flat buildings and town houses) at the same date, from the same published training data, as the Assessor’s model, and check both against sales later in the year. Our benchmark is our rebuild of the Assessor’s model from its public code and data; the values the Assessor published are compared in the details below.

**As delivered, the GL values are more accurate than our rebuild of the Assessor’s model in both years.** Their typical error is about one percentage point lower: 14.60% against 15.67% in 2023 (difference −1.07 points, 90% interval −1.34 to −0.78) and 9.52% against 10.39% in 2022 (−0.87, −1.06 to −0.69). What an assessor looks for:

- **Level:** the GL values are at the right overall level (median ratio 0.99); our rebuild of the Assessor’s values is about 5% low (0.94–0.95).
- **Uniformity:** the GL values are slightly more uniform (COD lower by 0.7 points in 2023 and 0.3 in 2022).
- **Regressivity:** the two models are equally regressive (PRD 1.07 in 2023, 1.03 in 2022). In 2023 neither meets the guidance of the International Association of Assessing Officers (IAAO); in 2022 both are at its upper edge (details below).

**Once each model’s overall level is corrected in advance, there is no clear difference in typical error.** The correction divides each model’s values by its own median ratio on the previous October–December sales, using no prices after 1 January. GL minus rebuild MdAPE is then +0.13 points in 2023 (90% interval −0.18 to +0.34) and 0.00 in 2022 (−0.18 to +0.22). The GL’s advantage as delivered is therefore its level.

The areas are the parts of the county the Assessor revalued in each year: 17 townships of the southern and western suburbs in 2023 (11,356 evaluation sales) and 13 of the northern suburbs in 2022 (10,825). Chicago cannot be tested this way because the Assessor’s model inputs for its reassessment years (2021 and 2024) are not public; the sale-prediction test covers it.

## Details for expert readers

### Sale-prediction test: details

**Split.** The test uses the Assessor’s 2025 training file: sales from January 2016 to December 2024, less sales of parcels with more than one building record and sales the Assessor flagged as outliers (381,763 sales). Sorted by sale date, the first 343,586 train both models and the last 38,177 are the test, as in the Assessor’s own pipeline; our test sales are exactly the Assessor’s.

**GL.** The published model, unchanged except that its price index covers 94 months, January 2016 to October 2023. Its regressors are the 26 home characteristics among the 2025 model’s predictors, coded as in the 1 January tests, plus the previous-sale count; there is no proration rate, because the 2025 model has none, and prorated sales are kept (624 training, 38 test). The census-block units are those of the 1 January tests; 702 homes not placed there (775 sales) are placed by their map coordinates. Each test sale is predicted with the characteristics of its own sale record and with the township and neighbourhood price paths held at October 2023; the prediction is the median over 4,000 posterior draws. The 38 fits took 3 hours 26 minutes, three at a time.

**Previous-sale count.** This is the Assessor’s count of the home’s other sales in the four years before the sale. For a test sale it can include an earlier sale in the test period, so the predictions are not all made from what was known in October 2023; the prices of those sales enter neither model.

**The Assessor’s model.** Its 2025 model, from the Assessor’s published container image (pinned by digest), trained on the training sales by its own pipeline. We compare its initial, unrounded predictions for the test sales.

**Intervals.** Differences are paired, GL minus the Assessor on the same sales. The 90% intervals use 2,000 redraws of homes within each township, keeping every sale of a drawn home, and hold the fitted models fixed. Township intervals are pointwise: across 38 townships, a few that exclude zero can arise by chance.

**Diagnostics.** The gap is smaller for sales in the first three months after the cutoff (October–December 2023: +0.51 points [+0.16, +0.88]) than later (January–June 2024 +1.26, July–December 2024 +1.11). Both models’ median ratios fall from about 1.00 to 0.92–0.93 over the test period; this comparison does not separate rising prices from changes in which homes sold. The gap is larger for homes that also sold in the training period (+1.49 [+1.24, +1.73]) than for the others (+0.90 [+0.76, +1.05]); why is not established.

**Convergence.** All 38 fits finished at the first attempt. The largest R-hat of any parameter is 1.035, and of any test sale’s prediction 1.005. The sampler reported divergent transitions in seven small townships (157 in all, 113 of 4,000 iterations in Norwood Park); leaving those townships out gives +1.07 points [+0.94, +1.20] county-wide, +2.10 in Chicago and +0.33 in the suburbs.

![GL minus Assessor MdAPE by township, sale-prediction test](figures/county_by_township.png)

### About the maps

The map at the top shows GL’s estimated value of the same standard home in each census block across Cook County, including Chicago, in October 2023, relative to the median block, on one colour scale. It comes from the sale-prediction fits. Some colour steps at township boundaries may reflect the separate township fits. Blank areas have no mapped estimate; thin light lines mark township boundaries and the dashed line the City of Chicago. The rectangle marks the enlargement of West Chicago township (right), which runs from the Near West Side to the city’s western edge.

For each census block, the map shows the log value of one standard home in the last training month $T$ (October 2023), $m = \alpha + x_\ast^\top \beta + \mu_T + \zeta_k + \psi_{kT} + u_s$ (posterior median; notation as in “Model details”), as $\exp(m - \tilde m) - 1$, with $\tilde m$ the median over the blocks shown (about $341,000). Every predictor is fixed: the standard home $x_\ast$ has the median of each numerical characteristic and the most common level of each categorical one over all 38 townships’ training sales (65 years old, about 1,490 square feet of building on 5,200 of land, three bedrooms, two full bathrooms, six rooms) and a previous-sale count of 0, the median; it is coded as in each township. Colours are therefore comparable across townships. The colour scale ends at ±1 log point (−63% to +172%); 2.7% of blocks lie beyond it. Where a census block is split between neighbourhoods, the map shows the mean of its parts weighted by their numbers of homes.

The second map shows the same quantity from the 1 January fits of the 30 suburban townships: the value at 1 January 2022 in the north and 1 January 2023 in the south and west, for a standard home with the median characteristics of those townships’ training sales (57 years old, about 1,460 square feet of building on 7,400 of land, three bedrooms). For that map we checked the steps at neighbourhood boundaries against sales within 400 m on either side, adjusted for characteristics and sale month by a simple regression (sales of the last three training years). Within townships its steps track the sales’ (correlation 0.79 over 750 boundaries, with the map’s step outside the sales’ 90% interval at 12% of them); across township lines, where separately fitted townships meet, less closely (0.64 over 104 boundaries, 28% outside). These results describe the second map; the check has not been repeated for the whole-county fits. The check’s script is not included in this release.

![Value of a standard home per census block in the 30 suburban townships of the 1 January tests](figures/location_map_assessment.png)

<p align="center"><sub>The 1 January fits: value of one standard home by census block, relative to the median block. Grey: Chicago, not covered by those tests. The rectangle marks the zoom on Evanston (right).</sub></p>

### 1 January tests: homes and sales covered

In 2023 we valued 336,532 homes across 17 southern and western townships, checking them against 11,356 later sales. In 2022 we valued 280,424 homes across 13 northern townships, checking them against 10,825 later sales.

Eligible homes have one building record, map coordinates and a building not shared with another parcel. They are in the Assessor’s classes 202–212, 234, 278 and 295: single-family houses, two- to six-flat buildings, and town and row houses. Condominiums are excluded. The Assessor treats some homes as identical units, such as the town houses of one development; both models give every home in such a group the same value.

Both models use the Assessor’s published training sales: 2014–2022 for the 2023 valuation and 2013–2021 for 2022. GL is fitted separately to each township’s sales; the Assessor’s model uses sales across the county, including Chicago. We apply the Assessor’s training filters for each year: both exclude parcels with multiple building records, while the 2023 filter also excludes sales flagged as outliers.

A later sale is used for evaluation only if:

- the Assessor did not flag it as an outlier and it covers one parcel;
- the home’s class and specified recorded characteristics are unchanged between the Assessor’s 1 January roll and its record of the home in the year of sale;
- no home-improvement exemption was active at the sale;
- it is the home’s first eligible sale that year.

The characteristics checked are building and land area, rooms, bedrooms, full and half bathrooms, garage size, basement, attic finish and year built. We apply no ratio-based trimming.

Neither model was fitted to the evaluation sales.

### Public sources

- **Model inputs:** the Assessor’s training sales, assessment rolls, groups of identical units and land rates, from its public bucket at `https://ccao-data-public-us-east-1.s3.amazonaws.com/models/inputs/res/<year>/<file>`. Inputs are downloaded for 2022, 2023 and 2025. In the 1 January tests, the 2025 training file supplies the later sales used for evaluation and the price index, and trains neither model. In the sale-prediction test it is the data: both models train on its earlier 90% and are tested on its latest 10%.
- **Parcel sales:** [Assessor – Parcel Sales](https://datacatalog.cookcountyil.gov/d/wvhk-k5uv), used to exclude sales covering multiple parcels.
- **Published values:** [Assessor – Assessed Values](https://datacatalog.cookcountyil.gov/d/uzyt-m557), using the first-pass values mailed to owners. We multiply assessed values by 10 to put them on the market-value scale.
- **Census blocks:** US Census Bureau TIGER/Line [2020 Illinois census blocks](https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_17_tabblock20.zip).
- **Assessor model code:** [ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm), at tags `2023-assessment-year` and `2022-assessment-year` (1 January tests); for the sale-prediction test, the Assessor’s published container image of its 2025 model.
- **Map:** basemap tiles © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.

### 1 January tests: time adjustment, rounding and uncertainty

Sale prices are adjusted back to 1 January using a separate monthly price index for each study area (not the GL model’s own index), fitted to sales from the assessment year and the preceding year. The index controls for neighbourhood, property class, building and land area, and age. Its 1 January level is the average of the December and January log-price effects. Homes are divided into five fixed groups; each group’s adjustments use an index fitted to the other four, so a home’s own sale never enters its adjustment.

The Assessor’s final model values are rounded down (above $100,000, to the $10,000 below). The comparison uses those final values; rounding is part of the values being compared. GL values are not rounded.

Differences are paired—GL minus the other model on the same sales—and computed from unrounded statistics, so they can differ slightly from differences of the table’s entries. The 90% intervals use 2,000 bootstrap redraws, keeping homes in an identical-unit group together and resampling separately within townships. They reflect uncertainty about which sales were observed, holding the fitted models, price index and correction factors fixed.

The level correction of Part 2 comes in two versions. The one quoted there uses the October–December sale prices as they are, so it uses no prices after 1 January (we have not checked whether every sale record was already available then). The figure below uses those prices time-adjusted to 1 January, with the index above, which is estimated partly from later sales; it gives GL minus rebuild +0.01 points [−0.19, +0.25] in 2023 and −0.01 [−0.22, +0.20] in 2022. “Level-free” MdAPE instead divides each model’s values by its own median ratio on the evaluation sales themselves; it is an after-the-fact diagnostic.

![Typical error before and after level correction](figures/accuracy_level.png)

### Model details

For sale *i* of a home with characteristics $x_i$, in month $t_i$, Assessor neighbourhood $k_i$ and census-block unit $s_i$, the model is

$$\log p_i = \alpha + x_i^\top \beta + \mu_{t_i} + \zeta_{k_i} + \psi_{k_i t_i} + u_{s_i} + \varepsilon_i, \qquad \varepsilon_i \sim t_\nu(0, \sigma_\varepsilon),$$

with $\mu$ the township’s monthly price index over the training months (108 in the 1 January tests, 94 in the sale-prediction test), $\zeta$ the neighbourhood levels, $\psi$ the neighbourhood price paths and $u$ the block effects. In the 1 January tests, the regressors $x_i$ are the Assessor’s 27 home characteristics plus its proration rate (the share of a building’s value assigned to the parcel), which is 1 for every home valued there; in the sale-prediction test, they are the 2025 model’s 26 home characteristics plus the previous-sale count. Areas enter as logs, year built as age, numerical characteristics standardised and categorical ones as indicators; this coding is learned from each township’s training sales only (see [`prep_v2.py`](scripts/v2/prep_v2.py)), and the regressors are centred.

Each effect has a whitened graph-Laplacian prior, as in Sijp and Francke (2026). For a graph with Laplacian $L = U \Lambda U^\top$, let the whitener $A = U_+ \Lambda_+^{-1/2}$ over the eigenvectors with non-zero eigenvalue; an effect is $\sigma A z$ with $z \sim N(0, I)$, so its prior covariance is $\sigma^2 L^+$. $A_T$, $A_K$ and $A_u$ are the whiteners of the month, neighbourhood and block graphs. Dropping the null space makes each effect sum to zero over its graph, leaving the intercept $\alpha$ to carry the overall price level.

- $\mu = \sigma_\mu A_T z_\mu$ over the path graph of months: a first-order random walk.
- $\zeta = \sigma_\zeta A_K z_\zeta$ over the neighbourhood graph.
- $\psi = \sigma_\psi A_K Z_\psi A_T^\top$ with independent standard-normal entries in $Z_\psi$, so $\mathrm{vec}(\psi)$ has prior covariance $\sigma_\psi^2 (L_T^+ \otimes L_K^+)$: each neighbourhood’s price path is a random walk, smoothed across neighbouring neighbourhoods.
- $u = \sigma_u A_u z_u$ with one graph per neighbourhood, so block effects sum to zero within each neighbourhood. Each neighbourhood’s whitener is scaled to unit geometric-mean marginal variance. Blocks without training sales are estimated jointly with the rest, through their neighbours in this prior.

Blocks crossing Assessor neighbourhood boundaries are split. Blocks within a neighbourhood are connected when they share a boundary line, and neighbourhoods are connected when their blocks share one. Disconnected parts are joined by a minimum spanning tree using distances between block outlines.

Priors (normal distributions given by mean and standard deviation): $\alpha \sim N(\bar y, 10)$ with $\bar y$ the mean training log price; $\beta \sim N(0, 1)$, sampled through a QR decomposition of the regressors; half-normal scales $\sigma_\mu, \sigma_\psi \sim N^+(0, 0.1)$, $\sigma_\zeta, \sigma_u \sim N^+(0, 0.5)$ and, for the Student-t scale, $\sigma_\varepsilon \sim N^+(0, 0.25)$ with $\sigma_\varepsilon \ge 0.01$; degrees of freedom $\nu \sim \text{Gamma}(2, 0.1)$ (shape, rate) on $[2, 40]$.

The model is fitted separately to each township using four chains with 1,000 warm-up and 1,000 sampling iterations each. For 1 January valuation, the township and neighbourhood price paths ($\mu$ and $\psi$) are held at their December values; in the sale-prediction test, at their October 2023 values. Each home’s value is the median of its fitted prices across 4,000 posterior draws, without simulated sale-price error. For groups the Assessor treats as identical units, member prices are averaged within each draw before taking the median.

A simpler model with one block graph per township and no neighbourhood effects was also fitted. A rule fixed before fitting retained that simpler model unless the upper end of the 90% interval for its MdAPE disadvantage, on both years’ evaluation sales together, exceeded +0.5 percentage points. Its disadvantage was +0.81 points [+0.69, +0.93], so the model with neighbourhood effects was retained.

### 1 January tests: regressivity and appraisal guidance

The GL and rebuilt Assessor values are similarly regressive. Price-related bias (PRB) is roughly the proportional change in the ratio when a home’s value doubles (the slope of the relative ratio on the base-2 logarithm of a proxy averaging value and price). It is −0.124 for GL and −0.129 for the rebuild in 2023, and −0.034 and −0.024 in 2022. Negative values indicate higher ratios for cheaper homes.

The IAAO’s *Standard on Ratio Studies* (2013) suggests a COD of 5.0–15.0 for single-family homes, 5.0–10.0 in newer or fairly similar areas, and 5.0–20.0 for two- to four-unit buildings; a PRD of 0.98–1.03; and a PRB between −0.05 and +0.05. On single-family sales alone, both models fall outside all three ranges in the 2023 area. In 2022, both fall within the COD and PRB ranges and just above the PRD range (single-family PRD 1.034 for GL and 1.031 for the rebuild). Unlike many assessment ratio studies, ours does not trim outlying ratios.

![Median ratio by sale-price decile](figures/price_deciles.png)

Grouping by sale price can exaggerate apparent regressivity because a home that happens to sell cheaply has both a low price and a high ratio. PRB is less affected because of its proxy. The shaded 0.90–1.10 band is IAAO guidance for overall appraisal level, shown for reference, rather than a standard for price groups.

### 1 January tests: multi-family homes and township differences

For two- to six-flat buildings, GL’s typical error is higher than the rebuild’s: by +1.53 percentage points [+0.12, +2.49] in 2023 and +1.41 [+0.002, +2.69] in 2022. Correcting the overall level does not remove this gap. GL sees relatively few such sales within most townships, while the Assessor’s model learns from the whole county, including Chicago. A variant coding the number of flats numerically instead of categorically did not close the gap; its scripts are not included.

Some townships favour the Assessor’s model, including Oak Park, where GL’s MdAPE is +1.67 points higher [+0.37, +2.80] in 2023.

![GL minus rebuilt Assessor results by township](figures/by_township.png)

### 1 January tests: published values and the rebuild

The values the Assessor published, on the same evaluation sales as the headline table:

| Area and year | Values | MdAPE, % | Median ratio | COD | PRD |
|---|---|---:|---:|---:|---:|
| South and west, 2023 | GL | 14.60 | 0.988 | 22.8 | 1.074 |
| | Assessor, our rebuild | 15.67 | 0.944 | 23.6 | 1.072 |
| | Assessor, published | 15.73 | 0.942 | 23.5 | 1.061 |
| North, 2022 | GL | 9.52 | 0.993 | 12.9 | 1.032 |
| | Assessor, our rebuild | 10.39 | 0.954 | 13.3 | 1.028 |
| | Assessor, published | 9.55 | 0.966 | 12.7 | 1.028 |

In 2022, the Assessor’s published values are more accurate than our rebuild for homes sold early in the year. For January–April sales, MdAPE is 7.9% for published values against 10.3% for the rebuild; for September–December sales, both are 10.5%. This suggests that the published values for those homes reflect information from those sales, which neither model had at 1 January. For comparing methods, our rebuild is therefore the fairer benchmark; the published values are shown for completeness.

Our rebuild runs the Assessor’s tagged training and valuation code unchanged, using public inputs. It is not the production run. It reproduces published values exactly for 72% of 362,957 homes with both values in the 2023 area and 76% of 294,805 in 2022, including homes excluded from our valuation sample.

Bloom and Thornton differ more noticeably in 2023. In Bloom, rebuilt values are a median 9% above published values and only 27% match within $1 on the assessed-value scale, against 86% in the whole 2023 area; in Thornton, 68% match. Excluding both townships leaves the GL’s MdAPE advantage over the rebuild almost unchanged: −1.03 points [−1.35, −0.78].

Published 2023 values are less regressive than both models overall. Excluding Bloom and Thornton, published and rebuilt values have the same PRD, 1.054, compared with GL’s 1.061.

### 1 January tests: earlier sales and the sale-history check

Both models value homes that also sold during training lower relative to their later prices than homes without an earlier sale. The median-ratio gap is about 0.12 in 2023 and 0.07 in 2022.

To check how much GL depends on a home’s own earlier sales, we selected 704 homes at random from the 2023 evaluation sales with earlier sales, removed their earlier sales and refitted all 17 townships. Their values rose by a median 0.65% [0.59%, 0.73%], within the ±1% band fixed in advance for this median. The median size of the change, 0.93%, is larger than the 0.11% median difference between two random seeds for the same homes. The check required 17 further fits; its scripts are not included in this release.

### Limitations

Both tests are retrospective. The 1 January tests compare two years in two different areas; year and area change together, so their effects cannot be separated. Condominiums are not covered, and Chicago only in the sale-prediction test. The 1 January evaluation sales were also used to choose between GL and its simpler version (see “Model details”); the sale-prediction test sales were used by an earlier version of this project (Part 2).

The 1 January results depend on adjusting sale prices to the valuation date. Without adjustment, GL’s MdAPE advantage over the rebuild grows to −2.05 points in 2023 and −2.15 in 2022. Prices rose during both years; our interpretation is that unadjusted prices favour the model with the higher overall level.

In the 1 January tests, all neighbourhood effects, block effects and home values meet R-hat ≤ 1.01 across the 30 townships (the four chains agree). Some scale parameters reach 1.015. The sampler reported divergent transitions, a sign of difficult geometry, in five township fits: 116 of 4,000 sampling iterations in Norwood Park in 2022, and fewer than 15 in each of the other four. These fits were recorded rather than refitted. A second random seed gave a similar GL minus rebuild MdAPE difference on both years’ sales together, −0.98 against −0.97 points.

Exact reproduction is not guaranteed. Open-data queries are live and their contents can change; software and compiler differences may also affect results. [REPRODUCE.md](REPRODUCE.md) records downloads, checksums and reproduction limits. The sale-prediction test, the sale-history check and the number-of-flats variant are outside the released workflow.

## Licences and terms

- **Repository code:** MIT licence ([LICENSE](LICENSE)).
- **Assessor model code:** AGPL-3.0. It is not included or redistributed here. [`assessor/rebuild.sh`](assessor/rebuild.sh) fetches it from [ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm) at the year’s tag and runs it unchanged, using our Dockerfiles following each tag’s build recipe.
- **Data:** none are included. Check each source’s terms before reuse. The Assessor offers its model inputs for public download; we found no stated licence for them. We also found no stated licence for the parcel-sales and assessed-values datasets; the portal links to Cook County’s general [terms of use](https://www.cookcountyil.gov/terms-use). Census blocks are published by the US Census Bureau and, as US federal government works, are not subject to copyright in the United States.
- **Map:** the basemap of the location map is © OpenStreetMap contributors, available under the [Open Database Licence](https://www.openstreetmap.org/copyright).

## References

- Sijp, W. P. and Francke, M. K. (2026). Graph-Laplacian modeling of spatiotemporal effects for house price estimation. *Real Estate Economics* (volume and pages not yet assigned). <https://doi.org/10.1111/1540-6229.70062>
- Sijp, W. (2026). Scale and Local Information in Housing Markets: A Graph-Laplacian AVM Benchmarked Against Machine Learning. SSRN working paper. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7345003>
- International Association of Assessing Officers (2013). *Standard on Ratio Studies*. Kansas City, MO: IAAO.

## How to cite

For the model, cite Sijp and Francke (2026). For this code and its results, cite this repository. [`CITATION.cff`](CITATION.cff) holds both: GitHub’s “Cite this repository” button shows the model paper as the preferred citation, and the file also describes the repository itself.
