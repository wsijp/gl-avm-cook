# Reproducing the results

This file lists the steps from the public inputs to the tables and figures of the [README](README.md), with the
commands, what each step does, how long it took us and what it needs; "Which README results are covered" below says
which results they reproduce and which they do not. [`reproduce.sh`](reproduce.sh) runs the same steps:
`public/reproduce.sh all` for the 1 January results (after preparing Docker's memory as described under "The
Assessor's model"), `public/reproduce.sh county` for the sale-prediction test of the published model,
`public/reproduce.sh final` for the final sale-prediction model's test (after `county`), or one step at a time by name.

## What we used, and what you need first
- **Hardware:** one Apple M5 laptop (10 cores, 32 GB of memory), macOS. Allow about 25 GB of free disk space, plus the
  Docker virtual machine's disk (below).
- **Python 3.14** (we used 3.14.7; other 3.14 releases are untested), installed beforehand (for example from
  python.org or with Homebrew, `brew install python@3.14`), available as `python3.14`. `setup` creates the environment `.venv` and installs the packages and versions in
  [`requirements.txt`](requirements.txt).
- **A C++ compiler and `make`**, which CmdStan needs (on macOS the Xcode command line tools,
  `xcode-select --install`; on Linux `g++` and `make`). `setup` installs **CmdStan 2.39.0** into
  `~/.cmdstan/cmdstan-2.39.0` and checks that it is the version used; `reproduce.sh` points cmdstanpy at it through
  the `CMDSTAN` environment variable. If you set `CMDSTAN` yourself, it must point at an existing CmdStan 2.39
  installation (`setup` checks the version). The model is compiled on its first use.
- **Docker** able to run linux/amd64 images, for the Assessor's model. On Apple silicon we used colima with Rosetta
  emulation: `colima start --cpu 8 --memory 16 --disk 60 --vm-type vz --vz-rosetta` (more memory for one step, see
  "The Assessor's model").
- `git`, `curl`, `rsync`, `shasum` and `bash`.

## Folder layout
The Python and Stan code is exactly the code that produced the published results (only their comments were tidied
for publication), with two exceptions: `parcel_sales.py` gained an `--out` option afterwards (its default is the
folder we used), and the final model's figure scripts (`final_figures.py`, `final_map.py`) were written for this
repository from the saved results. The shell scripts (`reproduce.sh`, `assessor/rebuild.sh`, `assessor/run2025.sh`) run the commands
we ran, gathered for this release; `assessor/run2025.sh` repeats our 2025 run of the Assessor's model (same image,
input and command) but was not itself used for it. The scripts expect to run from a working folder that holds this repository as a subfolder named `public`. Everything else is created beside it:

```
work/                     run every command from here
  public/                 this repository
  .venv/                  Python environment (the scripts call .venv/bin/python)
  ccao/inputs/<year>/     the Assessor's input files (downloaded)
  ccao/model-res-avm-<year>/, ccao/runs/<year>_rebuild/   the Assessor's code and our runs of it
  ccao/runs/2025_full/    the Assessor's 2025 model, run for the sale-prediction test
  notes/v2/logs/, notes/v2/figures/, notes/v2/explore_blocks/   logs, working figures, the census-block file
```
Prepared data go to `public/data/` and fits and results to `public/runs/`; both are ignored by git.

```
mkdir work && cd work
git clone https://github.com/wsijp/gl-avm-cook.git public
public/reproduce.sh setup                      # folders, Python environment, CmdStan 2.39.0
```

## Names used in scripts and folders
The scripts keep the labels used while the work was done. They mean:

| label | meaning |
|---|---|
| model **P** (`struct_P_t<code>`, `--model P`) | the published GL model |
| model **S** (`struct_S_t<code>`) | the simpler model without neighbourhood effects (not used) |
| `struct_` | the model structure in which every effect sums to zero (`fit_struct.py`, `stan/v2/gl_struct.stan`) |
| `d38_` | the correction made in advance |
| `equity_` | regressivity measures and the published values |
| `p2_`, `p3_`, `p4_`, `p7b_`, `p8_` prefixes; folders `prep_p7`, `prep_c8fix`, `prep_p8` | the stage of the work in which a script or folder was first written (`p3_`: checking sales and price index; `p4_`: the Assessor's values; `p8_`: the 2022 year). The folder names have no further meaning. |
| `_ay2022`, `ay2022/` | assessment year 2022 |
| `county2025_` | the sale-prediction test on the Assessor's 2025 split |
| `loop_` (`loop_prep.py`, `loop_fit.py`, `loop_fold.py`, `stan/v2/gl_loop.stan`) | the sale-prediction model, the published model with optional terms, developed in a series of comparisons on earlier sales |
| **H1** (`county2025_H1_t<code>`, `prep_county2025_h1`) | the published model fitted to houses and flats together, with sale-history indicators (one copy for houses, one for flats) |
| **H2** (`H2h` houses, `H2f` flats; `salehist_county.py`, `prep_county2025_h2h`, `_h2f`) | the first sale-history version: separate house and flat models with each home's sale history since 2016 |
| **IXa** (`IXah`, `prep_loop_ixah`; `d70_prep.py`) | the final house model: H2's houses with flip-purchase flags (`k1h`, `d63_prep.py`), neighbourhood-varying resale terms (`--np`), age interactions and the forecasting price index (`--forecast`) |
| **FCf** (`prep_county_fcf`) | the final flat model: flats with sale history since 1999 (`--fullhist`), the fading premium (`--gexp`) and the forecasting price index |
| `d77_county.py`, `county_d77/`, `county2025_ixa_eval/` | the final model's test (preparations, fits, evaluation) |
| `salehist`, `salehist_full`, `parcel_sales` | sale histories: from the Assessor's 2025 file (since 2016) and from the county's Parcel Sales table (since 1999) |
| township codes, e.g. `t28` | the Assessor's two-digit township codes (28 = Orland) |

## Steps
Times are wall-clock on the machine above.

| step | command (from `work/`) | what it does | time |
|---|---|---|---|
| 1 | `public/reproduce.sh download` | Downloads the Assessor's input files for 2023, 2022 and 2025 (about 1.1 GB) and the census-block file (0.3 GB), and checks every file against [`inputs.sha256`](inputs.sha256). A file that fails the check must be deleted before downloading again. | network |
| 2 | `public/reproduce.sh opendata` | Queries the Cook County open data portal for the parcel sales of 2023 and 2022 and the Assessor's published values (see "Live downloads" below). | a few minutes |
| 3 | `public/reproduce.sh data` | For each year: the valued homes, the census blocks and their map, the sales used to check accuracy, the price index and the time adjustment, and one prepared folder per township (data and characteristic coding). | about 10 minutes |
| 4 | `public/reproduce.sh assessor` | Rebuilds the Assessor's model for each year and runs it ([`assessor/rebuild.sh`](assessor/rebuild.sh); details and memory needs below), then compares our rebuild with the published values. | about 1 hour plus downloads |
| 5 | `public/reproduce.sh fit` | Fits the GL model in each of the 30 townships (17 for 2023, 13 for 2022), at most three at a time, and values every home at 1 January (`scripts/v2/struct_run.py`, calling `fit_struct.py` and `value_v2.py`). | about 1 hour 40 minutes (4 hours of fitting in total) |
| 6 | `public/reproduce.sh evaluate` | Every measure and interval of the 1 January tests in the README, both years, their figures in `public/figures/`, and the remaining 1 January numbers. | about 30 minutes |
| 7 | `public/reproduce.sh county` (optional, not part of `all`; after steps 1 and 3) | The sale-prediction test of the published model: runs the Assessor's 2025 model, fits the published GL in all 38 townships and predicts the test sales, then its table and by-township figure (see "The sale-prediction test"). | about 3½ hours (3 h 26 min of fitting at three at a time) |
| 8 | `public/reproduce.sh final` (optional; after step 7) | The final model's sale-prediction test: downloads the Parcel Sales table, prepares the sale histories, fits the first sale-history version (which the final model reuses), then the final model's 56 fits, the README's sale-prediction table, its two figures and the map at the top (see "The final model's sale-prediction test"). | about 22 hours (6 h + 16 h of fitting at three at a time) |

`all` runs steps setup and 1–6. The `assessor` step runs the 2022 assess stage straight after the others, so with
`all` give the Docker machine the larger memory and swap described under "The Assessor's model" before you start (or
run the steps one by one). Beyond `all`:
- **The simpler model** ("0.81 points higher [+0.69, +0.93]"): `public/reproduce.sh simpler` before `evaluate` (about
  9 hours of fitting in total; 3–4 hours at three at a time); `evaluate` then writes the comparison to
  `public/runs/v2/struct_eval/selection.json`.
- **The second seed** ("−0.98 against −0.97"), after `evaluate`:
  ```
  .venv/bin/python -I public/scripts/v2/struct_run.py --models P --seed 2 --suffix _seed2
  .venv/bin/python -I public/scripts/v2/struct_chosen.py --model P --runs runs_seed2.json --tag _seed2
  ```
  The pooled GL − rebuild MdAPE difference is in `public/runs/v2/struct_eval/chosen_seed2/ratio_pooled_primary.json`
  (pair `gl-assessor`: −0.976); the first seed's is in `chosen/ratio_pooled_primary.json` (−0.974).
- **The sale-prediction tests:** `public/reproduce.sh county`, then `public/reproduce.sh final`; see "The sale-prediction
  test" and "The final model's sale-prediction test" below.
- **Not included in this repository:** the sale-history check of the README ("Further checks"; 17 further fits on
  training data with the selected homes' earlier sales removed) and the number-of-flats variant mentioned under the
  multi-family results. Their numbers cannot be reproduced from this release.

Step 3 in detail, for 2023 (2022 uses the scripts in `scripts/v2/ay2022/`, the same code with the year's dates, files
and the Assessor's 2022 training filter):
1. `scripts/v2/p2_population.py`: the roll inventory of homes in the model classes.
2. `scripts/v2/blocks.py`: places every home and training sale in a 2020 census block, splits blocks that straddle
   two Assessor neighbourhoods, and builds the map of adjacent blocks.
3. `scripts/v2/p3_eligibility.py`: the sales used to check accuracy, with every rule's flag.
4. `scripts/v2/p3_time_index.py`: the monthly price index and each checking sale's adjustment to 1 January.
5. `scripts/v2/prep_v2.py --town <code>` for each township: training sales, valued homes, checking sales and the
   coding of the characteristics. Orland's folder is `public/data/v2/prep_c8fix/`, the others' `public/data/v2/prep_p7/`
   (2022: `public/data/v2/prep_p8/`), because the fit driver reads them from there. The 2022 checking sales are also
   copied to `public/runs/v2/p8_eval/keys/prep/`, where the evaluation scripts read them.

Step 6 in detail: `struct_chosen.py --model P` (the headline measures, calling `ratio_v2.py`, which computes every
measure and interval), `struct_convergence.py`, `d38_level.py` and `d38_unadjusted.py` (the correction made in advance,
with and without time adjustment), `d72_unrounded.py` (the headline comparison with both models' values before
rounding, the rounded-alike comparison and the correction in advance on unrounded values), `equity_eval.py` and `equity_figure.py` (the published values, PRD and PRB, the
decile figure), `readme_figures.py`, `readme_numbers.py` and `fig_location_map.py --set assessment` (the map of the 1 January fits,
`figures/location_map_assessment.png`; it downloads about 70 OpenStreetMap tiles once, into
`public/data/v2/osm_tiles/`). The README's sale-prediction results, its figures and the whole-county map at the top
come from steps 7 (`county`) and 8 (`final`), below.

## The sale-prediction test
`public/reproduce.sh county` runs, after `download` (which also fetches the 2025 input), `opendata` and `data` (`data`
needs the open-data files and builds the census-block units the county step reuses), so from a fresh working folder:
`public/reproduce.sh setup download opendata data county`. It stops the Docker machine (colima) before the GL fits:
1. `public/assessor/run2025.sh`: clones the Assessor's code at the tag `2025-assessment-year` (commit
   `f292e7b3a31cd01df5950b3b6bc54e848666a33c`) into `ccao/model-res-avm-2025/`, places the 2025 input in its `input/`
   folder, pulls the Assessor's own image for that tag pinned by digest
   (`ghcr.io/ccao-data/model-res-avm@sha256:faba881c593c4743a7873d720eff9b8a980d46f4b2088d7e598fefcf17215b7b`; no
   build), and runs `pipeline/01-train.R` unchanged into `ccao/runs/2025_full/output/`. Training took us 7 minutes,
   plus the image download. Its predictions for its own test
   sales are in `output/test_card/model_test_card.parquet`, column `pred_card_initial_fmv`. Ours had sha256
   `3954f1505c1772e791dad2a1b271f3f94119192635fa89976b76506b7963e60a`; whether a run on another machine matches it
   bit for bit is untested. The test sales themselves (their keys) depend only on the input and the Assessor's split,
   and step 2 checks that they match.
2. `scripts/v2/county2025_prep.py`: the Assessor's split of the 2025 input (381,763 sales; the last 38,177 are the
   test, checked against the Assessor's test sales), placement in the census-block units, and one prepared folder per
   township in `public/data/v2/prep_county2025/`. About 3 minutes.
3. `scripts/v2/county2025_run.py` (calling `county2025_fit.py`, which is `fit_struct.py` with 94 training months, and
   `county2025_predict.py`): fits model P in all 38 townships, at most three at a time, and predicts every test sale.
   3 hours 26 minutes for us (9.2 hours of fitting in all; the longest fits were Lake, about 2 hours, and Jefferson
   and West Chicago, 1 hour 5–20 minutes each).
   Before each fit the driver waits for 25% of memory to be free, which it reads with macOS's `memory_pressure`; as
   written it runs on macOS only.
4. `scripts/v2/county2025_eval.py`: both models' MdAPE, median ratio, COD and PRD and the paired 90% intervals, in
   `public/runs/v2/county2025_eval/summary.csv` and `summary.json` (per-township convergence in `convergence.csv`).
   Then `county2025_figure.py` (the by-township figure, copied to `public/figures/county_by_township.png`, linked
   from the README's details). (`fig_location_map.py --set county2025` draws the same kind of map from these fits; the
   README's map now comes from the final model, step 8.)

`county2025_run.py` and `county2025_eval.py` gained options after our run (for further tests not in this
repository); their defaults, which `reproduce.sh` uses, run the published test unchanged.

## The final model's sale-prediction test
`public/reproduce.sh final` runs after `county` (it reuses the county step's split, census-block placement, the
Assessor's 2025 predictions and the published model's results). Fits run at most three at a time, each under `nice`,
with the same memory rule as the county step (macOS only as written). Times are ours on the M5:
1. `scripts/v2/parcel_sales.py`, three times: the county's Parcel Sales table (all classes, dates from 1999 on; buyer
   and seller names are not requested) for the 38 townships, in the three downloads our scripts read
   (`public/data/v2/parcel_sales/sales_t<codes>.parquet`, with a `.json` record of each query). A live query: see
   "Live downloads". The largest download (30 townships, 1.46 million records) took us 5½ minutes.
2. `scripts/v2/loop_prep.py --county`, three times: the preparations of the first sale-history version (`prep_county2025_h1`,
   `_h2h`, `_h2f`): the Assessor's split, each sale's history from the 2025 file strictly before its date, the coding of
   the characteristics for houses and flats separately, and the flat buildings' graph. About 3 minutes each.
3. `scripts/v2/salehist_county.py run --arm H2`, then `eval --arm H2`: the first sale-history version's 76 fits (38
   house fits, 18 flat fits, and 20 joint fits whose flats the final model reuses), and its evaluation. About 6 hours
   for us (17.8 hours of fitting in all). The final step needs these fits: the 20 joint fits value the flats of
   the townships with too few flat sales, the house and flat fits' runtimes order the final fits, and their predictions
   are a comparison column of the final evaluation.
4. `scripts/v2/d77_county.py prep`: the final house preparations (the first version's houses plus the flip-purchase
   flags, `d63_prep.py`, and the age interactions, `d70_prep.py`) and the final flat preparations (`loop_prep.py
   --county --fullhist long --gapcode exp`: sale history since 1999). Under a minute.
5. `scripts/v2/d77_county.py run`: the final model's 56 fits (`loop_fit.py --county`; 38 house fits
   `county2025_IXah_t<code>` with `--forecast --np`, 18 flat fits `county2025_FCf_t<code>` with `--forecast --gexp`;
   four chains of 1,000 + 1,000 iterations, seed 1). 15 hours 53 minutes for us (47.4 hours of fitting in all; Lake's
   house fit alone 11.2 hours, West Chicago's flats 4.3, Jefferson's houses 4.9).
6. `scripts/v2/d77_county.py eval`: both models' measures and the paired intervals, as in the county step
   (`public/runs/v2/county2025_ixa_eval/summary.csv`), the sale-history groups (`salehist.md`), houses and flats and
   the comparisons with the earlier versions (`d77.md`, `d77_groups.csv`), and the usefulness bar (`d77.json`). A few
   minutes.
7. `scripts/v2/final_figures.py` (the README's by-township and price-index figures) and `scripts/v2/final_map.py` (the
   map at the top). The map rebuilds each house fit's block effects from its saved chains (`chains/*.csv`) and checks
   that the rebuild reproduces every test sale's fitted October 2023 value; about 2 minutes.

The first sale-history version's fits (step 3) were made on 20–21 October 2026 with earlier versions of `loop_prep.py`,
`loop_fit.py` and `gl_loop.stan`; the released versions added options and kept the earlier ones' code paths, but we have
not refitted step 3 with them. Rebuilt with the released `loop_prep.py`, step 2's three preparations were identical to
ours in every data file (only the manifests' option lists and script hashes differ). The final model's preparations and
fits (steps 4–6) were made with the released code. `d63_prep.py` and `d70_prep.py` copy a paragraph of their own
description into each preparation's `coding.json`; the released descriptions were reworded for publication, so those
notes (and the file hashes) differ from ours, not the data.

## Where each published number is written
All under `public/runs/v2/`:

| README item | file |
|---|---|
| **1 January headline, both models before rounding**: MdAPE, median ratio, COD, PRD, GL − rebuild differences, subgroups (multi-family, earlier sale), townships (Oak Park); rounded alike; corrected in advance (unrounded); without time adjustment; without Bloom and Thornton | `struct_eval/chosen/d72/summary.json`, `d72/ratio_<year>_primary.json` (unrounded), `d72/ratio_<year>_rounded.json`, `d72/ratio_<year>_unadjusted.json`, `d72/ratio_2023_excl_12_37.json` |
| the same against the rebuild's rounded values (the comparison used before 22 October 2026; the three figures still drawn that way) | `struct_eval/chosen/ratio_<year>_primary.json` (`overall`, `subgroups`, `by_town`, `own_sale_gap`) |
| corrected in advance with time adjustment; level-free | `struct_eval/chosen/d38/summary.json` |
| corrected in advance without time adjustment | `struct_eval/chosen/d38/unadjusted.json` |
| published values: all measures, GL − published, PRB, deciles | `struct_eval/chosen/equity/summary.json`, `deciles_<year>.csv` |
| without time adjustment ("roughly doubles") | `struct_eval/chosen/ratio_<year>_unadjusted.json` |
| single-family IAAO measures; without Bloom and Thornton; exact matches with the published values | `struct_eval/chosen/readme_numbers.json` |
| convergence (R-hat, divergent transitions) | `struct_eval/convergence_P.json`, `.csv` |
| the simpler model (optional) | `struct_eval/selection.json` |
| sale-prediction test of the published model: MdAPE, median ratio, COD, PRD, differences and intervals (county, City, suburbs, townships, horizons, earlier sale, level-free) | `county2025_eval/summary.csv`, `summary.json` |
| sale-prediction test of the published model: convergence, runtimes | `county2025_eval/convergence.csv`, `runs.json` |
| final model's sale-prediction test: MdAPE, median ratio, COD, PRD, differences and intervals (county, City, suburbs, townships, horizons, level-free, without flagged townships) | `county2025_ixa_eval/summary.csv`, `summary.json` |
| final model: houses and flats, old never-sold Chicago houses, comparison with the earlier versions, usefulness bar | `county2025_ixa_eval/d77.md`, `d77_groups.csv`, `d77.json` |
| final model: homes sold before or not, gaps since the previous sale | `county2025_ixa_eval/salehist.md`, `salehist.csv` |
| final model: convergence, runtimes, sources of each township's predictions | `county2025_ixa_eval/convergence.csv`, `runs.json`; each fit's `report.json` |

## Which README results are covered
| README result | command | covered |
|---|---|---|
| Sale-prediction test: table, county / Chicago / suburbs, township results, diagnostics, convergence, usefulness bar | `reproduce.sh final` (after `county`) | yes |
| Figures `final_by_township.png`, `final_price_index.png` and the map at the top (`location_map.png`) | `reproduce.sh final` | yes |
| The published model on the sale-prediction test (+1.10 points) and its figure `county_by_township.png` | `reproduce.sh county` | yes |
| The first sale-history version (+0.59 points) | `reproduce.sh final` (step 3) | yes |
| The second, boosted correction model tried and not adopted | none | no: scripts not included |
| 1 January tests: accuracy, level, uniformity, regressivity, intervals, both years | `reproduce.sh all` (`evaluate`) | yes |
| Corrected in advance; level-free; without time adjustment | `reproduce.sh all` (`evaluate`) | yes |
| Published values, PRB, price deciles, IAAO measures, multi-family and earlier-sale subgroups | `reproduce.sh all` (`evaluate`) | yes |
| Figures `accuracy_level.png`, `by_township.png`, `price_deciles.png`, `location_map_assessment.png` | `reproduce.sh all` (`evaluate`) | yes |
| Exact matches of our rebuild with the published values | `reproduce.sh all` (`assessor`, `evaluate`) | yes |
| The simpler model ("0.81 points higher") | `reproduce.sh simpler`, then `evaluate` | yes |
| The second seed ("−0.98 against −0.97") | the two commands above | yes |
| Map boundary check (correlations 0.79 and 0.64) | none | no: script not included |
| Sale-history check (704 homes, +0.65%) | none | no: scripts not included |
| Number-of-flats variant | none | no: scripts not included |

"Covered" means that the commands that produced the result are here; it is not a promise of identical numbers (see
"Live downloads" and "What to expect").

## What has been tested of this workflow
From a fresh copy of this repository in an empty working folder, we ran `setup`, the input check of `download` (on
copies of our input files, so nothing was fetched) and `data` for both years (with copies of our open-data files, so
the live queries were not repeated): every prepared data file was identical to ours, apart from three timing records
and, in 32 manifests, the scripts' own hashes and creation times. We then fitted and valued one township (River Forest) with model P: its
values were identical to ours. Not tested from a fresh copy: the downloads and live queries, the Docker rebuilds of
2023 and 2022 and the 2025 run (`assessor/`), the full fits, `evaluate`, and the `county` step. The Python scripts and
the model file are the ones that produced the results; the shell wrappers (`reproduce.sh`, `assessor/rebuild.sh`,
`assessor/run2025.sh`) were assembled for this repository from the commands we ran and have not been run end to end.

For the final model's step (`final`): from a fresh copy of this repository we ran the Parcel Sales download for one
township (River Forest, `--towns 33`, into a separate folder): its 6,498 records were identical to ours, and the sale
histories built from it were identical to ours for all 90 of River Forest's test sales. The step's preparations of the
first sale-history version, rebuilt with the released code, were identical to ours (see "The final model's
sale-prediction test"). Not tested: the `final` step end to end, and its fits from a fresh copy.
## The Assessor's model
`public/assessor/rebuild.sh <year>` does, for 2023 and for 2022:
1. **fetch:** clones [ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm) at the tag
   `<year>-assessment-year` and checks the commit (2023: `32542556332433790ad5e84aed1a49dbaeb8fdc3`; 2022:
   `7b026b20f10f91db7cbded92528512b1de89df9f`). Their code is AGPL-3.0 and is not part of this repository.
2. **build:** builds a Docker image from [`assessor/Dockerfile.2023`](assessor/Dockerfile.2023) or
   [`Dockerfile.2022`](assessor/Dockerfile.2022). The tags have no Dockerfile; ours follow each tag's own CI recipe
   (its R version, system packages and `renv::restore()` of their lockfile), with the base image pinned by digest and
   R packages taken as precompiled binaries from Posit Package Manager snapshots (2023-03-01 for 2023, 2022-04-01 for
   2022) at the lockfile's exact versions. The Dockerfiles' headers list every departure. Our successful builds took
   10–12 minutes each; earlier failed attempts had already downloaded part of what they need, so a first build may
   take longer.
3. **train** and **assess:** runs `pipeline/01-train.R` and `pipeline/02-assess.R` unchanged, with the year's input
   files mounted as `input/` (no database ingest, no upload). About 5–8 minutes per stage. The values are in
   `ccao/runs/<year>_rebuild/output/assessment_pin/model_assessment_pin.parquet`, column `pred_pin_final_fmv_round`.

`rebuild.sh` will not overwrite an earlier run: before any step runs, it stops if `ccao/runs/<year>_rebuild/`
already holds the outputs or log of a requested stage. A finished run is a reference result: to repeat it, use a fresh working folder. Only a
failed run of your own (for example a stage that ran out of memory) should be moved aside before running it again. `run2025.sh` (see "The sale-prediction test") does the same for
`ccao/runs/2025_full/`.

**Memory for the 2022 assess stage.** With 16 GB the 2022 assess stage was stopped for lack of memory, also at 22 GB
without swap. It finished with 22 GB of memory and a 24 GB swap file inside the virtual machine. With colima, before
`public/assessor/rebuild.sh 2022 assess`:
```
colima stop && colima start --cpu 8 --memory 22 --disk 60 --vm-type vz --vz-rosetta
colima ssh -- sudo sh -c 'fallocate -l 24G /mnt/lima-colima/swapfile && chmod 600 /mnt/lima-colima/swapfile \
  && mkswap /mnt/lima-colima/swapfile && swapon /mnt/lima-colima/swapfile'
```
These are the standard commands for this; we created our swap file by hand in the same folder, and did not run
them in exactly this form. The 60 GB virtual disk held both images and the swap file. Remove the
swap file and restart colima at 16 GB afterwards if you wish.

**Checking a rebuild.** Our output files (`model_assessment_pin.parquet`) had sha256
`d9ef45186c9f14c7dcbc9451299c9635d8906f420c99852818111f72f37222ea` (2023) and
`ad311061aeb50d841b26ed144164a9a55fc390c4fd15cad1bd43e5dae1ccbdeb` (2022). We have not tested whether a rebuild on
another machine reproduces them bit for bit (their model is a LightGBM model, run here under x86 emulation), and we do
not include our rebuilt values (this repository contains no data). What you can check is the agreement
with the Assessor's published values: `readme_numbers.json` reports the share of homes whose rebuilt value equals
10 × the published assessed value exactly (72.0% in 2023 and 76.1% in 2022 for us).

## Live downloads
The parcel sales and published values are live queries of the county's open data portal, which can change. The first four
were fetched on 18 October 2026 (Australian Eastern Daylight Time), the Parcel Sales files of the final model on the
dates shown; each fetch also writes a `.json` record of its query,
time, row count and sha256:

| file (in `public/data/v2/`) | rows | sha256 |
|---|---|---|
| `p3_opendata_sales_2023.parquet` | 79,132 | `c93d7df6e0188433ea55b7a9191240487f0a96e88e513102f2211cf4f8df6224` |
| `p4_published_2023_south.parquet` | 527,671 | `fbb7eeb0a1ff6a4f04adf59e998bc5497581a75d1fa0cdf4a3e5268ac43141e2` |
| `ay2022/p8_opendata_sales_2022.parquet` | 101,398 | `95143eda9171b59f7f9605b66605213de15dfa4aac7fcad17b2f1b7e2442c62f` |
| `ay2022/p8_published_2022_north.parquet` | 453,915 | `54991a902250df86d106934d8fb70816eeab37f976c58481ea1c23be21fe16a0` |
| `parcel_sales/sales_t77-73-76-74-31-11.parquet` (fetched 2026-10-20 UTC) | 816,129 | `2b4202aab649e33b251a7a4755e586198f5e4b5e0bd93f9d3d1888cba971fddc` |
| `parcel_sales/sales_t72-70.parquet` (fetched 2026-10-21 UTC) | 412,245 | `aa6c550470921a603bf7cb62222b90b6a4979508805a0649adb6fae21d33ee7e` |
| `parcel_sales/sales_t10-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30-32-33-34-35-36-37-38-39-71-75.parquet` (fetched 2026-10-23 UTC) | 1,460,437 | `9e42d161322c9452f0fbd54ea535673501a1ac520abeef12715b23fa32a42866` |

**An exact reproduction of our sample and results is therefore not guaranteed:** if a fresh fetch differs from ours
(compare its sha256 with the table), the results may differ too, and we cannot bound by how much. We do not
redistribute our copies (we found no licence stated for them).

## What to expect
- The fits use fixed random seeds, so a rerun with the same software should give the same values or very nearly so;
  across machines and compilers small differences in the draws are possible. Between two different seeds the pooled
  headline difference moved by 0.002 points (−0.974 and −0.976; −0.97 and −0.98 when rounded).
- The 2022 scripts in `scripts/v2/ay2022/` are the 2023 scripts with the 2022 dates, file names and north-suburb
  labels, plus four differences: the 2022 training filter, columns missing from the 2022 files read as missing,
  missing group identifiers valued as single homes, and a count check that only holds in 2023. To see every
  difference, compare a 2022 script with its 2023 source (`diff public/scripts/v2/prep_v2.py
  public/scripts/v2/ay2022/prep_v2.py`).
- Every script states its inputs, outputs and arguments in its first lines. Some file and folder names keep the labels
  used while the work was done (see "Names used in scripts and folders"), and a few messages and records written by
  the scripts still use them; they do not affect what the code does.
- The sale-history check in the README ("Further checks") is not part of `reproduce.sh`: it needs a further 17 fits
  on reduced training data, and its scripts are not included here.
