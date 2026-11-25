"""Two-stage matching: a fast bi-encoder picks the top-K episodes, a cross-encoder re-scores them.

A cross-encoder reads (comment, summary text) TOGETHER, so every comment word can look at every summary
word: slower, usually more precise. Long summaries are handled two ways:
  full   - whole summary in one pass (8k-context rerankers)
  chunks - best ~200-word chunk per episode (MaxP)

Rerankers
  gte_rr     Alibaba gte-reranker-modernbert-base (149M), a normal cross-encoder
  bge_rr_m3  BAAI bge-reranker-v2-m3 (568M), a normal cross-encoder
  qwen3_rr_0.6b / qwen3_rr_4b  Qwen3-Reranker (June 2025): a small LLM asked "does this document match? yes/no";
             the score is P(yes)

Usage: python rerank.py <first-stage tag> <reranker name> <mode full|chunks> [K] [split]
"""
import sys
import time

import numpy as np
import torch

from common import append_result, chunk_text, evaluate, load_scores, load_split, load_summaries, save_scores

RERANKERS = {
    "gte_rr": ("Alibaba-NLP/gte-reranker-modernbert-base", 8192),
    "bge_rr_m3": ("BAAI/bge-reranker-v2-m3", 8192),
    "qwen3_rr_0.6b": ("Qwen/Qwen3-Reranker-0.6B", 8192),
    "qwen3_rr_4b": ("Qwen/Qwen3-Reranker-4B", 8192),
}
first, rr_name, mode = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
split = sys.argv[5] if len(sys.argv) > 5 else "test"
hf_or_path, max_len = RERANKERS.get(rr_name, (rr_name, 8192))


class Qwen3Reranker:
    """Qwen3-Reranker scoring as described on its model card: prompt -> logits of 'yes' vs 'no'."""
    INSTRUCT = "Given a comment from a TV show discussion thread, judge whether the episode summary passage is what the comment talks about"
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
              "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")

    @torch.no_grad()
    def predict(self, pairs, batch_size=32, **_):
        out = []
        for k in range(0, len(pairs), batch_size):
            texts = [f"{self.PREFIX}<Instruct>: {self.INSTRUCT}\n<Query>: {q}\n<Document>: {d}{self.SUFFIX}"
                     for q, d in pairs[k:k + batch_size]]
            b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
            last = self.model(**b, logits_to_keep=1).logits[:, -1, :]    # left padding: last position = answer slot
            out.append(torch.log_softmax(last[:, [self.no, self.yes]].float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)


summ = load_summaries()
df = load_split(split)
base = load_scores(first, split)
if rr_name.startswith("qwen3_rr"):
    model = Qwen3Reranker(hf_or_path)
else:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(hf_or_path, device="cuda", max_length=max_len if mode == "full" else 384,
                         model_kwargs={"dtype": torch.float16}, trust_remote_code=True)

t0 = time.time()
pairs, owner = [], []  # owner = (comment index, episode index)
for i, (show, comment) in enumerate(df[["show", "comment"]].itertuples(index=False)):
    top = np.argsort(-np.asarray(base[i], float))[:K]
    for j in top:
        text = summ[show][1][j]
        docs = [text] if mode == "full" else chunk_text(text)
        for d in docs:
            pairs.append((comment, d))
            owner.append((i, j))
print(f"{len(pairs):,} pairs", flush=True)
# sort by length so batches have similar lengths (much faster)
order = np.argsort([len(p[1]) for p in pairs])
pred = np.empty(len(pairs), dtype=float)
bs = 4 if mode == "full" else 64
pred[order] = model.predict([pairs[k] for k in order], batch_size=bs, show_progress_bar=False)

scores = [np.asarray(s, float) - 1e6 for s in base]  # non-reranked episodes stay below
best = {}
for (i, j), p in zip(owner, pred):
    best[(i, j)] = max(best.get((i, j), -1e9), p)
for (i, j), p in best.items():
    scores[i][j] = 1e6 + p
tag = f"{first}+{rr_name.split('/')[-1]}_{mode}_k{K}"
save_scores(tag, split, scores)
row = evaluate(df, scores, tag) | {"split": split, "secs": round(time.time() - t0, 1)}
append_result(row)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
