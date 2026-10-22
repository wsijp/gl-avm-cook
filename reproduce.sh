#!/usr/bin/env bash
# Reproduce the published results from public sources: see REPRODUCE.md for what each step does, its runtime and
# what it needs.
#
#   public/reproduce.sh <step> [<step> ...]
#   steps, in order: setup download opendata data assessor fit evaluate      (all: every step in this order)
#   optional: simpler   (fits the simpler model without neighbourhood effects; ~9 hours of fitting; run before
#                        evaluate, which then also writes the comparison)
#             county    (the sale-prediction test: the Assessor's 2025 model, then the GL in all 38 townships,
#                        ~3.5 hours at three fits at a time; after download, opendata and data; not part of all)
#
# Run from the working folder that holds this repository as public/. Python is called as .venv/bin/python (the scripts
# start their own subprocesses with that path). Outputs: ccao/ (the Assessor's inputs, code and runs), public/data/,
# public/runs/ (prepared data, fits and results; not part of the repository), notes/v2/ (logs and figures).
set -euo pipefail
[ -f public/reproduce.sh ] || { echo "run from the folder that contains public/" >&2; exit 2; }
PY=.venv/bin/python
S=public/scripts/v2
SOUTH="11 12 13 14 15 19 21 27 30 31 32 33 34 36 37 39"   # 2023 south and west townships except Orland (28)
NORTH="10 16 17 18 20 22 23 24 25 26 29 35 38"            # 2022 north townships
BUCKET=https://ccao-data-public-us-east-1.s3.amazonaws.com/models/inputs/res
export CMDSTAN=${CMDSTAN:-$HOME/.cmdstan/cmdstan-2.39.0}   # the CmdStan version we used (cmdstanpy reads CMDSTAN)
run() { echo "+ $*"; nice -n 15 "$@"; }
fetch() {   # fetch URL FILE: skip if FILE exists; download to a temporary name and rename only on success
  [ -f "$2" ] || { curl --fail -L -o "$2.part" "$1" && mv "$2.part" "$2"; }
}

free_docker_memory() {   # the Docker machine and the GL fits must not share memory (REPRODUCE.md)
  if command -v colima > /dev/null && colima status > /dev/null 2>&1; then
    echo "+ colima stop   (frees the Docker machine's memory before the GL fits)"; colima stop
  elif command -v docker > /dev/null && docker info > /dev/null 2>&1; then
    echo "NOTE: Docker is running; stop it (or its virtual machine) before the GL fits to free its memory" >&2
  fi
}

step_setup() {
  mkdir -p notes/v2/logs notes/v2/figures notes/v2/explore_blocks ccao/inputs/2023 ccao/inputs/2022 ccao/inputs/2025
  [ -x $PY ] || python3.14 -m venv .venv
  $PY -m pip install -r public/requirements.txt
  [ -d "$CMDSTAN" ] || $PY -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.39.0')"
  $PY -c "import cmdstanpy; v = cmdstanpy.cmdstan_version(); print('CmdStan', v); assert v == (2, 39), 'CmdStan 2.39 expected'"
}

step_download() {
  for y in 2023 2022; do
    for f in training_data assessment_data complex_id_data land_nbhd_rate_data land_site_rate_data; do
      fetch $BUCKET/$y/$f.parquet ccao/inputs/$y/$f.parquet
    done
  done
  fetch $BUCKET/2025/training_data.parquet ccao/inputs/2025/training_data.parquet
  fetch https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_17_tabblock20.zip \
    notes/v2/explore_blocks/tl_2020_17_tabblock20.zip
  shasum -a 256 -c public/inputs.sha256
}

step_opendata() {   # live queries of the county's open data; a fresh download may differ from ours (REPRODUCE.md)
  run $PY -I $S/p3_fetch_opendata_sales.py public/data/v2
  run $PY -I $S/p4_fetch_published_2023.py public/data/v2
  run $PY -I $S/ay2022/p3_fetch_opendata_sales.py public/data/v2/ay2022
  run $PY -I $S/ay2022/p4_fetch_published_2022.py public/data/v2/ay2022
}

step_data() {
  z=notes/v2/explore_blocks/tl_2020_17_tabblock20.zip
  # 2023: valued homes, census blocks and their map, checking sales, price index, one prepared folder per township
  run $PY -I $S/p2_population.py ccao/inputs public/data/v2
  run $PY -I $S/blocks.py $z ccao/inputs public/data/v2
  run $PY -I $S/p3_eligibility.py ccao/inputs public/data/v2
  run $PY -I $S/p3_time_index.py ccao/inputs public/data/v2 notes/v2/figures
  for T in $SOUTH; do run $PY -I $S/prep_v2.py --town $T --out public/data/v2/prep_p7; done
  run $PY -I $S/prep_v2.py --town 28 --out public/data/v2/prep_c8fix   # Orland: the fit driver reads it from here
  # 2022
  mkdir -p public/data/v2/ay2022
  run $PY -I $S/ay2022/blocks.py $z ccao/inputs public/data/v2/ay2022
  run $PY -I $S/ay2022/p3_eligibility.py ccao/inputs public/data/v2/ay2022
  run $PY -I $S/ay2022/p3_time_index.py ccao/inputs public/data/v2/ay2022 notes/v2/figures
  for T in $NORTH; do
    run $PY -I $S/ay2022/prep_v2.py --town $T --data public/data/v2/ay2022 --out public/data/v2/prep_p8
    # the evaluation scripts read the 2022 checking sales from this folder
    mkdir -p public/runs/v2/p8_eval/keys/prep/t$T
    cp public/data/v2/prep_p8/t$T/eval.parquet public/runs/v2/p8_eval/keys/prep/t$T/
  done
}

step_assessor() {
  public/assessor/rebuild.sh 2023
  public/assessor/rebuild.sh 2022 fetch build train
  # the 2022 assess stage needs about 22 GB of memory plus swap in the Docker machine (REPRODUCE.md)
  public/assessor/rebuild.sh 2022 assess
  run $PY -I $S/p4_external_check.py ccao public/data/v2
  run $PY -I $S/ay2022/p4_external_check.py ccao public/data/v2/ay2022
}

step_fit() {
  free_docker_memory
  # the published model (P) in every township: fit, then value every home at 1 January
  run $PY -I $S/struct_run.py --models P > notes/v2/logs/struct_driver.log 2>&1
}

step_simpler() {
  free_docker_memory
  run $PY -I $S/struct_run.py --models S >> notes/v2/logs/struct_driver.log 2>&1
}

step_county() {
  # the sale-prediction test: the Assessor's 2025 model on its own split (public/assessor/run2025.sh), then the GL
  # fitted to each township's training sales and predicting its test sales; the README's table, figure and top map
  public/assessor/run2025.sh
  free_docker_memory
  run $PY -I $S/county2025_prep.py
  run $PY -I $S/county2025_run.py > notes/v2/logs/county2025_driver.log 2>&1
  run $PY -I $S/county2025_eval.py
  run $PY -I $S/county2025_figure.py
  cp notes/v2/figures/county2025_mdape_difference.png public/figures/county_by_township.png
  run $PY -I $S/fig_location_map.py --set county2025   # the map at the top of the README
}

step_evaluate() {
  if [ -f public/runs/v2/struct_S_t37/report.json ]; then
    run $PY -I $S/struct_eval.py              # the simpler model against P (writes selection.json)
  fi
  # headline measures and intervals, both years (public/runs/v2/struct_eval/chosen/)
  run $PY -I $S/struct_chosen.py --model P
  run $PY -I $S/struct_convergence.py --model P
  run $PY -I $S/d38_level.py --structure      # correction made in advance
  run $PY -I $S/d38_unadjusted.py --structure
  run $PY -I $S/equity_eval.py --structure    # PRD, published values, price deciles
  run $PY -I $S/equity_figure.py --structure
  cp notes/v2/figures/structure_equity_deciles.png public/figures/price_deciles.png
  run $PY -I $S/readme_figures.py --structure # public/figures/
  run $PY -I $S/fig_location_map.py --set assessment   # the 1 January map (downloads OpenStreetMap tiles once)
  run $PY -I $S/readme_numbers.py
}

steps=("$@")
[ ${#steps[@]} -gt 0 ] || { sed -n '2,14p' "$0"; exit 2; }
[ "${steps[0]}" = all ] && steps=(setup download opendata data assessor fit evaluate)
for s in "${steps[@]}"; do
  echo "=== $s $(date '+%Y-%m-%d %H:%M:%S')"
  "step_$s"
done
