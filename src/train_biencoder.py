"""Fine-tune a bi-encoder (comment encoder = summary-chunk encoder) for spoiler matching.

Training examples are built automatically:
  positive       = the chunk of the correct episode's summary most similar to the comment
                   (chosen with the untrained model, since we do not know which part the comment is about)
  hard negatives = the best chunk of each of the K most similar WRONG episodes of the same show
Loss: (Cached) MultipleNegativesRankingLoss: each comment must pick its positive among all chunks
in the batch (its own hard negatives + everyone else's chunks).
At test time: MaxP over chunks, exactly as in zeroshot.py.

Examples
  python train_biencoder.py 24000                                   # v2 (19 Nov recipe)
  python train_biencoder.py 0 --max-len 384 --negs 3 --batch 128    # v4: all comments (after the cap), new recipe
  python train_biencoder.py 0 --max-len 384 --negs 3 --batch 128 --clean 0.3 --rel-model ../models/relevance_ft
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from common import ROOT, append_result, chunk_text, evaluate, load_split, load_summaries, save_scores, warmup

# what each base model expects in front of a comment (query) and a summary chunk (document)
PROMPTS = {
    "nomic-ai/modernbert-embed-base": ("search_query: ", "search_document: "),
    "Qwen/Qwen3-Embedding-0.6B": ("Instruct: Given a comment from a TV show discussion thread, retrieve the "
                                  "episode summary passage that the comment is about\nQuery:", ""),
}

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int, nargs="?", default=24000, help="training comments; 0 = all (after the per-show cap)")
ap.add_argument("tag", nargs="?", default=None, help="name of the saved model / score files")
ap.add_argument("--base", default="nomic-ai/modernbert-embed-base")
ap.add_argument("--max-len", type=int, default=256, help="token limit for comments and chunks")
ap.add_argument("--negs", type=int, default=1, help="hard negatives per comment")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--mini-batch", type=int, default=16, help="GradCache chunk size (memory only, not the result)")
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap", type=int, default=15000, help="max comments per show (Game of Thrones has 127k)")
ap.add_argument("--clean", type=float, default=0.0, help="drop comments with P(about plot) below this; 0 = off")
ap.add_argument("--rel-model", default=str(ROOT / "models" / "relevance_ft"), help="classifier used by --clean")
ap.add_argument("--exclude-shows", default="", help="comma list of shows left out of training; 'val' = the 4 val shows "
                "(they are also in train.json, so without this val rewards memorising those shows)")
args = ap.parse_args()

QP, DP = PROMPTS[args.base]
TAG = args.tag or f"ft_nomic_{args.n // 1000}k"
OUT = ROOT / "models" / TAG
print("config:", vars(args) | {"tag": TAG}, flush=True)
summ = load_summaries()


def show_chunks(model, shows):
    out = {}
    for show in shows:
        docs, owner = [], []
        for i, t in enumerate(summ[show][1]):
            for c in chunk_text(t):
                docs.append(c)
                owner.append(i)
        e = model.encode([DP + d for d in docs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        out[show] = (docs, e, np.array(owner), len(summ[show][1]))
    return out


def score(model, df, chunks=None):
    chunks = chunks or show_chunks(model, sorted(df.show.unique()))
    q = model.encode([QP + c for c in df.comment], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    scores = []
    for i, show in enumerate(df.show):
        _, e, owner, n = chunks[show]
        s = np.full(n, -1e9)
        np.maximum.at(s, owner, e @ q[i])
        scores.append(s)
    return scores


# ---- 1. sample training comments (cap per show so one show does not dominate)
train = load_split("train")
excl = sorted(load_split("val").show.unique()) if args.exclude_shows == "val" else [x for x in args.exclude_shows.split(",") if x]
if excl:
    print(f"left out of training: {excl} ({train.show.isin(excl).sum():,} comments)", flush=True)
    train = train[~train.show.isin(excl)]
train = pd.concat([g.sample(min(len(g), args.cap), random_state=args.seed) for _, g in train.groupby("show")])
if args.n:
    train = train.sample(min(args.n, len(train)), random_state=args.seed)
used_idx = train.index.tolist()  # rows of train.json used here (the reranker trains on OTHER comments)
train = train.reset_index(drop=True)

# ---- 1b. optional: drop comments the plot-relevance classifier thinks are off-topic (auto-labels are noisy)
if args.clean > 0:
    from relevance import predict_relevance
    p = predict_relevance(args.rel_model, train.comment.tolist())
    keep = p >= args.clean
    print(f"clean: dropped {int((~keep).sum()):,} of {len(train):,} comments ({(~keep).mean():.1%}) with P(plot) < {args.clean}",
          flush=True)
    train = train[keep].reset_index(drop=True)
    torch.cuda.empty_cache()
print("train comments:", len(train), train.show.value_counts().to_dict(), flush=True)

# ---- 2. mine the positive and K hard-negative chunks with the untrained model
t0 = time.time()
model = SentenceTransformer(args.base, device="cuda", tokenizer_kwargs={"padding_side": "left"} if "Qwen" in args.base else None)
model.max_seq_length = args.max_len
ch = show_chunks(model, sorted(train.show.unique()))
q = model.encode([QP + c for c in train.comment], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
rows = {"anchor": [], "positive": []} | {f"negative_{k + 1}": [] for k in range(args.negs)}
summ_eps = {s: summ[s][0] for s in summ}
for i, (show, ep, comment) in enumerate(train.itertuples(index=False)):
    docs, e, owner, n = ch[show]
    gold = summ_eps[show].index(ep)
    sims = e @ q[i]
    gmask = owner == gold
    p = np.where(gmask)[0][np.argmax(sims[gmask])]
    # wrong-episode chunks from most to least similar; keep the best chunk of each of the first K episodes
    order = np.where(~gmask)[0][np.argsort(-sims[~gmask])]
    hard, seen = [], set()
    for j in order:
        if owner[j] not in seen:
            seen.add(owner[j])
            hard.append(j)
            if len(hard) == args.negs:
                break
    rows["anchor"].append(QP + comment)
    rows["positive"].append(DP + docs[p])
    for k, j in enumerate(hard):
        rows[f"negative_{k + 1}"].append(DP + docs[j])
print(f"mined pairs in {time.time() - t0:.0f}s", flush=True)
ds = Dataset.from_dict(rows)

# ---- 3. train
bf16 = torch.cuda.is_bf16_supported()
targs = SentenceTransformerTrainingArguments(
    output_dir=str(OUT / "ckpt"), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
    learning_rate=args.lr, **warmup(0.1), fp16=not bf16, bf16=bf16, batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=25, save_strategy="no", report_to="none", seed=args.seed,
)
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
t0 = time.time()
trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
trainer.train()
train_secs = time.time() - t0
print(f"trained in {train_secs:.0f}s", flush=True)
model.save(str(OUT))
json.dump(used_idx, open(OUT / "train_idx.json", "w"))
json.dump(trainer.state.log_history, open(OUT / "train_log.json", "w"), indent=1)  # loss curve for the notebook
json.dump(vars(args) | {"tag": TAG, "train_comments": len(train), "train_secs": round(train_secs)},
          open(OUT / "train_config.json", "w"), indent=1)

# ---- 4. evaluate on val (seen-style shows) and test (unseen shows)
model.half() if not bf16 else model.bfloat16()
for split in ["val", "test"]:
    df = load_split(split)
    t0 = time.time()
    s = score(model, df)
    save_scores(TAG, split, s)
    row = evaluate(df, s, TAG) | {"split": split, "secs": round(time.time() - t0, 1)}
    append_result(row)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
