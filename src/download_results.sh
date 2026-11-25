#!/usr/bin/env bash
# Download saved model scores (results/scores/) and filter simulations (results/filter_sim__*.parquet)
# from the GitHub release, so every table can be checked without a GPU.
set -e
cd "$(dirname "$0")/.."
URL=https://github.com/AhmedDatori/progress-aware-spoiler-filter/releases/download/v1.0
curl -fL --progress-bar -o results_scores.zip "$URL/results_scores.zip"
unzip -oq results_scores.zip && rm results_scores.zip
echo "done: $(ls results/scores | wc -l) score files"
