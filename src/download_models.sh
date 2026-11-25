#!/usr/bin/env bash
# Download our fine-tuned models (fp16) from the GitHub release into models/
#   v6_bi              stage-1 bi-encoder        (nomic-ai/modernbert-embed-base, fine-tuned)
#   v6_rr48_s2         stage-2 reranker          (Alibaba-NLP/gte-reranker-modernbert-base, fine-tuned)
#   relevance_ft_base  "about the story?" classifier
set -e
cd "$(dirname "$0")/.." && mkdir -p models
URL=https://github.com/AhmedDatori/progress-aware-spoiler-filter/releases/download/v1.0
for m in v6_bi v6_rr48_s2 relevance_ft_base; do
  echo "downloading $m"
  curl -fL --progress-bar -o "models/$m.zip" "$URL/$m.zip"
  unzip -oq "models/$m.zip" -d models && rm "models/$m.zip"
done
echo "done: $(ls models)"
