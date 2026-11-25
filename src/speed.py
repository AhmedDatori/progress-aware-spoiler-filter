"""Speed and size comparison on the same laptop GPU (RTX 4050 6 GB, fp16).

Ours (bi-encoder): summaries are encoded ONCE and cached; each new comment needs one short pass.
Paper's best (Longformer cross-encoder): every (comment, episode summary) pair needs its own
long pass (up to 4,096 tokens). We time a random sample of real pairs and extrapolate.
Weights do not matter for speed, so the public longformer-base-4096 checkpoint is used.
"""
import json
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import RESULTS, ROOT, chunk_text, load_split, load_summaries

summ = load_summaries()
test = load_split("test")
n_pairs = int(sum(len(summ[s][0]) for s in test.show))
out = {"test_comments": len(test), "test_pairs_comment_x_episode": n_pairs}


def params(m):
    return sum(p.numel() for p in m.parameters())


# ---- ours
m = SentenceTransformer(str(ROOT / "models/ft_nomic_24k"), device="cuda").half()
m.max_seq_length = 256
docs = [c for s in sorted(test.show.unique()) for t in summ[s][1] for c in chunk_text(t)]
m.encode(["warm up"] * 64)
torch.cuda.synchronize(); t0 = time.time()
m.encode(["search_document: " + d for d in docs], batch_size=64)
torch.cuda.synchronize(); t_docs = time.time() - t0
t0 = time.time()
m.encode(["search_query: " + c for c in test.comment], batch_size=128)
torch.cuda.synchronize(); t_q = time.time() - t0
lat = []
for c in test.comment.sample(200, random_state=0):
    t0 = time.time(); m.encode(["search_query: " + c]); torch.cuda.synchronize(); lat.append(time.time() - t0)
out["ours_params_M"] = params(m) / 1e6
out["ours_summary_index_secs_once"] = t_docs
out["ours_all_test_comments_secs"] = t_q
out["ours_comments_per_sec"] = len(test) / t_q
out["ours_single_comment_ms"] = 1000 * float(np.median(lat))
del m; torch.cuda.empty_cache()

# ---- Longformer cross-encoder (paper's best model type)
tok = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
lf = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=1).cuda().half().eval()
rng = np.random.default_rng(0)
sample = []
for i in rng.choice(len(test), 48, replace=False):
    show, comment = test.show[i], test.comment[i]
    j = rng.integers(len(summ[show][1]))
    sample.append((comment, summ[show][1][j]))
lens = [len(tok(c, s, truncation=True, max_length=4096)["input_ids"]) for c, s in sample]
with torch.no_grad():
    b = tok([sample[0][0]] * 2, [sample[0][1]] * 2, truncation=True, max_length=4096, padding=True, return_tensors="pt").to("cuda")
    lf(**b)
    torch.cuda.synchronize(); t0 = time.time()
    for k in range(0, len(sample), 2):
        c, s = zip(*sample[k:k + 2])
        b = tok(list(c), list(s), truncation=True, max_length=4096, padding=True, return_tensors="pt").to("cuda")
        lf(**b)
    torch.cuda.synchronize(); t_lf = time.time() - t0
per_pair = t_lf / len(sample)
avg_eps = n_pairs / len(test)
out["longformer_params_M"] = params(lf) / 1e6
out["longformer_median_pair_tokens"] = float(np.median(lens))
out["longformer_secs_per_pair"] = per_pair
out["longformer_all_test_secs_est"] = per_pair * n_pairs
out["longformer_comments_per_sec_est"] = 1 / (per_pair * avg_eps)
out["longformer_single_comment_ms_est"] = 1000 * per_pair * avg_eps
out["speedup_all_test"] = out["longformer_all_test_secs_est"] / t_q
json.dump(out, open(RESULTS / "speed.json", "w"), indent=2)
print(json.dumps(out, indent=2))
