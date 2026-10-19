#!/usr/bin/env bash
# Rebuild and run the Cook County Assessor's residential model for one assessment year, unchanged.
#
#   public/assessor/rebuild.sh <year> [fetch] [build] [train] [assess]      (year 2023 or 2022; default: all four)
#
# Run from the working folder that holds this repository as public/ (see REPRODUCE.md); the year's inputs must be in
# ccao/inputs/<year>/. Steps:
#   fetch   clone https://github.com/ccao-data/model-res-avm at the tag <year>-assessment-year into
#           ccao/model-res-avm-<year>/ and check the tag's commit;
#   build   copy the checkout (without .git) to ccao/runs/<year>_rebuild/build/context/ and build the image
#           gl-avm/ccao-res-<year>:local from public/assessor/Dockerfile.<year> (10-12 minutes each for us, see
#           REPRODUCE.md);
#   train   pipeline/01-train.R, inputs mounted read-only at input/, outputs in ccao/runs/<year>_rebuild/output/;
#   assess  pipeline/02-assess.R. The values are in
#           ccao/runs/<year>_rebuild/output/assessment_pin/model_assessment_pin.parquet, column pred_pin_final_fmv_round.
# Only the train and assess stages run: no database ingest and no upload (their params.yaml is not changed).
# Their code is licensed AGPL-3.0; it is fetched from their repository, not included here.
# Needs Docker able to run linux/amd64 images (on Apple silicon: colima start --cpu 8 --memory 16 --disk 60
# --vm-type vz --vz-rosetta). The 2022 assess stage needs more memory than 16 GB: we gave the virtual machine 22 GB of
# memory and a 24 GB swap file. Logs: ccao/runs/<year>_rebuild/{build,train,assess}.log.
set -euo pipefail
year=${1:?usage: rebuild.sh <2023|2022> [fetch] [build] [train] [assess]}
shift
case $year in
  2023) commit=32542556332433790ad5e84aed1a49dbaeb8fdc3 ;;
  2022) commit=7b026b20f10f91db7cbded92528512b1de89df9f ;;
  *) echo "year must be 2023 or 2022" >&2; exit 2 ;;
esac
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(fetch build train assess)
[ -f public/assessor/Dockerfile.$year ] || { echo "run from the folder that contains public/" >&2; exit 2; }
code=ccao/model-res-avm-$year
run=ccao/runs/${year}_rebuild
img=gl-avm/ccao-res-$year:local
mkdir -p "$run"

for step in "${steps[@]}"; do
  case $step in
    fetch)
      [ -d "$code/.git" ] || git clone --branch "$year-assessment-year" --depth 1 https://github.com/ccao-data/model-res-avm "$code"
      [ "$(git -C "$code" rev-parse HEAD)" = "$commit" ] || { echo "$code is not at commit $commit" >&2; exit 1; }
      ;;
    build)
      mkdir -p "$run/build"
      rsync -a --delete --exclude .git "$code/" "$run/build/context/"
      cp public/assessor/Dockerfile.$year "$run/build/Dockerfile"
      ( cd "$run/build" && DOCKER_BUILDKIT=0 docker build --platform linux/amd64 -t "$img" -f Dockerfile . ) \
        2>&1 | tee -a "$run/build.log"
      docker image inspect "$img" --format 'image {{.Id}}' | tee "$run/image_used.txt"
      ;;
    train|assess)
      [ -d "$run/output" ] || cp -R "$code/output/" "$run/output/"
      script=pipeline/01-train.R
      [ "$step" = assess ] && script=pipeline/02-assess.R
      docker run --rm --platform linux/amd64 --cpus 8 \
        -v "$PWD/ccao/inputs/$year:/model-res-avm/input:ro" \
        -v "$PWD/$run/output:/model-res-avm/output" \
        "$img" Rscript "$script" 2>&1 | tee "$run/$step.log"
      ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
