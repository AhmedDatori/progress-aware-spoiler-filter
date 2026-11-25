#!/usr/bin/env bash
# Download the spoiler-matching dataset (CC-BY-SA-3.0) into data/raw/
set -e
mkdir -p "$(dirname "$0")/../data/raw" && cd "$(dirname "$0")/../data/raw"
B=https://huggingface.co/datasets/bobotran/spoiler-matching/resolve/main
for f in matching/with_autolabels/{summaries,train,val,test}.json filtering/handlabeled/{train,dev,test}.csv; do
  mkdir -p "$(dirname "$f")"; curl -sL -o "$f" "$B/$f"
done
echo done
