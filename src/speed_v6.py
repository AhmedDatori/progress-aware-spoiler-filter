"""Speed of the full v6 matcher on the laptop GPU (RTX 4050 6 GB), same GPU as speed.py, all 3,105 test comments.

v6 = stage 1 (bi-encoder + BM25 over all episodes) -> stage 2 (reranker on the top K episodes, M chunks each).
Summary chunks are encoded once and cached (they do not change), so that time is reported separately.
Compared with the Longformer estimate in speed.json (every comment x every episode, up to 4,096 tokens).

Usage: python speed_v6.py <bi-encoder tag> <reranker tag> <K> <M> <BM25 weight>   e.g. python speed_v6.py v6_bi v6_rr24_s0 10 2 0.3
"""
import json
import sys
import time

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import RESULTS, ROOT, chunk_text, load_split, load_summaries, tokenize
from fuse import z

bi_tag, rr_tag, K, M, w_bm25 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
summ = load_summaries()
test = load_split("test")
shows = sorted(test.show.unique())


def sync():
    torch.cuda.synchronize()
    return time.time()


# ---- one-time index: chunk texts, chunk embeddings, BM25 index
bi = SentenceTransformer(str(ROOT / "models" / bi_tag), device="cuda").half()
bi.max_seq_length = 384
bi.encode(["warm up"] * 64)
t0 = sync()
idx = {}
for s in shows:
    docs, owner = [], []
    for i, t in enumerate(summ[s][1]):
        for c in chunk_text(t):
            docs.append(c)
            owner.append(i)
    owner = np.array(owner)
    e = bi.encode(["search_document: " + d for d in docs], batch_size=64, normalize_embeddings=True)
    idx[s] = (docs, e, [np.where(owner == i)[0] for i in range(len(summ[s][1]))],
              BM25Okapi([tokenize(d) for d in docs]), owner)
t_index = sync() - t0

# ---- stage 1 for all test comments
t0 = sync()
q = bi.encode(["search_query: " + c for c in test.comment], batch_size=128, normalize_embeddings=True)
cands = []
for i, (s, c) in enumerate(test[["show", "comment"]].itertuples(index=False)):
    docs, e, ep_chunks, bm, owner = idx[s]
    sims = e @ q[i]
    dense = np.array([sims[ix].max() for ix in ep_chunks])
    b = np.full(len(ep_chunks), -1e9)
    np.maximum.at(b, owner, bm.get_scores(tokenize(c)))
    first = z(dense) + w_bm25 * z(b)
    for j in np.argsort(-first)[:K]:
        for k in ep_chunks[j][np.argsort(-sims[ep_chunks[j]])][:M]:
            cands.append((c, docs[k]))
t_stage1 = sync() - t0

# ---- stage 2: reranker on K x M pairs per comment
tok = AutoTokenizer.from_pretrained(ROOT / "models" / rr_tag)
rr = AutoModelForSequenceClassification.from_pretrained(ROOT / "models" / rr_tag).cuda().half().eval()
order = np.argsort([len(a) + len(b) for a, b in cands])
with torch.no_grad():
    enc = tok(["warm up"] * 8, ["warm up"] * 8, return_tensors="pt").to("cuda")
    rr(**enc)
    t0 = sync()
    for k in range(0, len(cands), 128):
        batch = [cands[i] for i in order[k:k + 128]]
        enc = tok([a for a, _ in batch], [b for _, b in batch], truncation=True, max_length=384, padding=True,
                  return_tensors="pt").to("cuda")
        rr(**enc)
    t_stage2 = sync() - t0

old = json.load(open(RESULTS / "speed.json"))
params = sum(p.numel() for p in bi.parameters()) + sum(p.numel() for p in rr.parameters())
out = old | {
    "v6_K": K, "v6_M": M, "v6_params_M": params / 1e6, "v6_pairs": len(cands),
    "v6_index_secs_once": t_index, "v6_stage1_secs": t_stage1, "v6_stage2_secs": t_stage2,
    "v6_all_test_secs": t_stage1 + t_stage2, "v6_comments_per_sec": len(test) / (t_stage1 + t_stage2),
    "v6_speedup_vs_longformer": old["longformer_all_test_secs_est"] / (t_stage1 + t_stage2),
}
json.dump(out, open(RESULTS / "speed.json", "w"), indent=2)
json.dump({k: v for k, v in out.items() if k.startswith("v6")}, open(RESULTS / f"speed_v6_K{K}_M{M}.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k.startswith("v6") or k.startswith("longformer_all")}, indent=2))
