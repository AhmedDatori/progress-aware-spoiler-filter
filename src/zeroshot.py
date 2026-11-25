"""Off-the-shelf embedding models (no training): embed comments and summaries, rank by cosine.

Two ways to handle long summaries:
  full   - feed the whole summary (only for long-context models)
  chunks - split into ~200-word windows and take the best window per episode (MaxP)
"""
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import append_result, chunk_text, evaluate, load_split, load_summaries, save_scores

QWEN_Q = ("Instruct: Given a comment from a TV show discussion thread, retrieve the episode summary passage "
          "that the comment is about\nQuery:")
# name -> (hf id, max tokens, query prefix, doc prefix)
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 256, "", ""),
    "bge_base": ("BAAI/bge-base-en-v1.5", 512, "Represent this sentence for searching relevant passages: ", ""),
    "gte_modernbert": ("Alibaba-NLP/gte-modernbert-base", 8192, "", ""),
    "nomic_modernbert": ("nomic-ai/modernbert-embed-base", 8192, "search_query: ", "search_document: "),
    "bge_m3": ("BAAI/bge-m3", 8192, "", ""),
    # 2025 models (Qwen3-Embedding, June 2025): instruction in front of the query, nothing in front of documents
    "qwen3_emb_0.6b": ("Qwen/Qwen3-Embedding-0.6B", 8192, QWEN_Q, ""),
    "qwen3_emb_4b": ("Qwen/Qwen3-Embedding-4B", 8192, QWEN_Q, ""),
    "qwen3_emb_8b": ("Qwen/Qwen3-Embedding-8B", 8192, QWEN_Q, ""),
}

split = sys.argv[1] if len(sys.argv) > 1 else "test"
DEFAULT = ["minilm", "bge_base", "gte_modernbert", "nomic_modernbert", "bge_m3"]   # the v1 sweep
# batch sizes (queries, chunks, full summaries); the big Qwen models need small batches on a 24 GB GPU
BATCH = {"qwen3_emb_4b": (16, 8, 1), "qwen3_emb_8b": (8, 4, 1)}
names = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT
summ = load_summaries()
df = load_split(split)
shows_needed = sorted(df.show.unique())

for name in names:
    hf, max_len, qp, dp = MODELS[name]
    t0 = time.time()
    qwen = name.startswith("qwen")   # Qwen needs bf16 (fp16 can overflow) and left padding (last-token pooling)
    model = SentenceTransformer(hf, device="cuda", trust_remote_code=True,
                                model_kwargs={"dtype": torch.bfloat16 if qwen else torch.float16},
                                tokenizer_kwargs={"padding_side": "left"} if qwen else None)
    model.max_seq_length = max_len
    bq, bc, bf = BATCH.get(name, (64, 32, 2 if qwen else 4))
    q = model.encode([qp + c for c in df.comment], batch_size=bq, normalize_embeddings=True, convert_to_numpy=True)
    modes = ["chunks"] + (["full"] if max_len >= 4096 else [])
    for mode in modes:
        ep_emb = {}
        for show in shows_needed:
            eps, texts = summ[show]
            if mode == "full":
                docs, owner = texts, list(range(len(texts)))
            else:
                docs, owner = [], []
                for i, t in enumerate(texts):
                    for c in chunk_text(t):
                        docs.append(c)
                        owner.append(i)
            e = model.encode([dp + d for d in docs], batch_size=bf if mode == "full" else bc,
                             normalize_embeddings=True, convert_to_numpy=True)
            ep_emb[show] = (e, np.array(owner), len(texts))
        scores = []
        for i, show in enumerate(df.show):
            e, owner, n = ep_emb[show]
            s = e @ q[i]
            ep_scores = np.full(n, -1e9)
            np.maximum.at(ep_scores, owner, s)
            scores.append(ep_scores)
        tag = f"{name}_{mode}"
        save_scores(tag, split, scores)
        row = evaluate(df, scores, tag) | {"split": split, "secs": round(time.time() - t0, 1)}
        append_result(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    del model
    torch.cuda.empty_cache()
