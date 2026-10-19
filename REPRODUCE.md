# Reproducing the results

This file lists every step from the public inputs to the tables and figures of the [README](README.md), with the
commands, what each step does, how long it took us and what it needs. [`reproduce.sh`](reproduce.sh) runs the same
steps: `public/reproduce.sh all` for the main results (after preparing Docker's memory as described under "The
Assessor's model"), or one step at a time by name.

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
The scripts' code is exactly the code that produced the published results (only their comments were tidied for
publication), and they expect to run from a working folder that holds this repository as a subfolder named `public`. Everything else is created beside it:

```
work/                     run every command from here
  public/                 this repository
  .venv/                  Python environment (the scripts call .venv/bin/python)
  ccao/inputs/<year>/     the Assessor's input files (downloaded)
  ccao/model-res-avm-<year>/, ccao/runs/<year>_rebuild/   the Assessor's code and our runs of it
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
| 6 | `public/reproduce.sh evaluate` | Every measure and interval of the README, both years, the figures in `public/figures/`, and the remaining README numbers. | about 30 minutes |

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
with and without time adjustment), `equity_eval.py` and `equity_figure.py` (the published values, PRD and PRB, the
decile figure), `readme_figures.py`, `readme_numbers.py` and `fig_location_map.py` (the map; it downloads about 70
OpenStreetMap tiles once, into `public/data/v2/osm_tiles/`).

## Where each published number is written
All under `public/runs/v2/`:

| README item | file |
|---|---|
| MdAPE, median ratio, COD, PRD of the GL and our rebuild; GL − rebuild differences; subgroups (multi-family, earlier sale), townships | `struct_eval/chosen/ratio_<year>_primary.json` (`overall`, `subgroups`, `by_town`, `own_sale_gap`) |
| corrected in advance with time adjustment; level-free | `struct_eval/chosen/d38/summary.json` |
| corrected in advance without time adjustment | `struct_eval/chosen/d38/unadjusted.json` |
| published values: all measures, GL − published, PRB, deciles | `struct_eval/chosen/equity/summary.json`, `deciles_<year>.csv` |
| without time adjustment ("roughly doubles") | `struct_eval/chosen/ratio_<year>_unadjusted.json` |
| single-family IAAO measures; without Bloom and Thornton; exact matches with the published values | `struct_eval/chosen/readme_numbers.json` |
| convergence (R-hat, divergent transitions) | `struct_eval/convergence_P.json`, `.csv` |
| the simpler model (optional) | `struct_eval/selection.json` |

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
The parcel sales and published values are live queries of the county's open data portal, which can change. Ours were
fetched on 18 October 2026 (Australian Eastern Daylight Time); each fetch also writes a `.json` record of its query,
time, row count and sha256:

| file (in `public/data/v2/`) | rows | sha256 |
|---|---|---|
| `p3_opendata_sales_2023.parquet` | 79,132 | `c93d7df6e0188433ea55b7a9191240487f0a96e88e513102f2211cf4f8df6224` |
| `p4_published_2023_south.parquet` | 527,671 | `fbb7eeb0a1ff6a4f04adf59e998bc5497581a75d1fa0cdf4a3e5268ac43141e2` |
| `ay2022/p8_opendata_sales_2022.parquet` | 101,398 | `95143eda9171b59f7f9605b66605213de15dfa4aac7fcad17b2f1b7e2442c62f` |
| `ay2022/p8_published_2022_north.parquet` | 453,915 | `54991a902250df86d106934d8fb70816eeab37f976c58481ea1c23be21fe16a0` |

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
