#!/usr/bin/env bash
# Run the Cook County Assessor's 2025 residential model, unchanged, from its published container image, to get its
# predictions for its own held-out test sales (the README's sale-prediction test).
#
#   public/assessor/run2025.sh [fetch] [pull] [train]      (default: all three)
#
# Run from the working folder that holds this repository as public/ (see REPRODUCE.md); the 2025 input must be in
# ccao/inputs/2025/training_data.parquet (public/reproduce.sh download). Steps:
#   fetch  clone https://github.com/ccao-data/model-res-avm at the tag 2025-assessment-year into
#          ccao/model-res-avm-2025/, check the tag's commit, and place the input in the clone's input/ folder
#          (checked against public/inputs.sha256);
#   pull   pull the Assessor's image for that tag, pinned by digest (it holds their code and R packages; nothing is
#          built);
#   train  pipeline/01-train.R, with the clone's input/ mounted at input/ and ccao/runs/2025_full/output/ (seeded with
#          the tag's empty output folders) at output/. About 7 minutes for us. The test-sale predictions are in
#          ccao/runs/2025_full/output/test_card/model_test_card.parquet, column pred_card_initial_fmv, keyed by
#          meta_pin, meta_sale_document_num and meta_card_num; scripts/v2/county2025_prep.py checks that its test sales
#          are exactly these keys, and scripts/v2/county2025_eval.py reads the predictions.
# Only the training stage runs, with the image's params.yaml as it is (run type "test"): no database ingest, no upload.
# Their code is licensed AGPL-3.0; it is fetched from their repository and image, not included here.
# Needs Docker able to run linux/amd64 images (as for rebuild.sh). Log: ccao/runs/2025_full/train.log.
# It will not overwrite an earlier run: train stops if ccao/runs/2025_full/output/ already holds any file (other than
# the empty folders' .gitkeep) or train.log exists. To repeat a finished run, use a fresh working
# folder; move a run aside only if it failed.
set -euo pipefail
commit=f292e7b3a31cd01df5950b3b6bc54e848666a33c
img=ghcr.io/ccao-data/model-res-avm@sha256:faba881c593c4743a7873d720eff9b8a980d46f4b2088d7e598fefcf17215b7b
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(fetch pull train)
[ -f public/assessor/run2025.sh ] || { echo "run from the folder that contains public/" >&2; exit 2; }
code=ccao/model-res-avm-2025
run=ccao/runs/2025_full
input=ccao/inputs/2025/training_data.parquet

# refuse to write into an earlier run, before any step runs
for step in "${steps[@]}"; do
  case $step in
    fetch|pull) ;;
    train)
      if [ -e "$run/train.log" ] || { [ -d "$run/output" ] && [ -n "$(find "$run/output" -type f ! -name .gitkeep | head -1)" ]; }; then
        echo "STOP: $run already holds a run (train.log or files in output/); a finished run is a reference: repeat it in a fresh working folder (move aside only a failed run)" >&2; exit 1
      fi ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done

for step in "${steps[@]}"; do
  case $step in
    fetch)
      [ -d "$code/.git" ] || git clone --branch 2025-assessment-year --depth 1 https://github.com/ccao-data/model-res-avm "$code"
      [ "$(git -C "$code" rev-parse HEAD)" = "$commit" ] || { echo "$code is not at commit $commit" >&2; exit 1; }
      want=$(awk -v f="$input" '$2 == f {print $1}' public/inputs.sha256)
      [ -f "$code/input/training_data.parquet" ] || cp "$input" "$code/input/training_data.parquet"
      [ "$(shasum -a 256 "$code/input/training_data.parquet" | cut -d' ' -f1)" = "$want" ] \
        || { echo "$code/input/training_data.parquet does not match public/inputs.sha256" >&2; exit 1; }
      ;;
    pull)
      docker pull --platform linux/amd64 "$img"
      ;;
    train)
      mkdir -p "$run"
      [ -d "$run/output" ] || cp -R "$code/output/" "$run/output/"
      docker run --rm --platform linux/amd64 \
        -v "$PWD/$code/input:/model-res-avm/input" \
        -v "$PWD/$run/output:/model-res-avm/output" \
        "$img" Rscript pipeline/01-train.R 2>&1 | tee "$run/train.log"
      ;;
  esac
done
