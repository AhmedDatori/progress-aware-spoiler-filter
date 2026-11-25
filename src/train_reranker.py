"""Stage 2: fine-tune a small cross-encoder reranker, then score the top episodes of val and test.

Full matcher (both models together < 300M parameters):
  stage 1  bi-encoder (149M) + BM25 score every episode of the show -> keep the top K episodes
  stage 2  this reranker (149M) reads (comment, chunk) TOGETHER for the best M chunks of each kept episode
           (best = most similar to the comment under the bi-encoder); episode score = best chunk score (MaxP)

Why a cross-encoder: the bi-encoder turns comment and chunk into one vector EACH, then compares the vectors,
so details get lost. A cross-encoder puts both texts in one input, so every comment word can attend to every
summary word ("the guy with the scar" <-> "Zuko"). It is slower, so it only sees the top K episodes.

Training data (built automatically, listwise):
  for each training comment, a group of G chunks: 1 positive + G-1 hard negatives
    positive  = the chunk of the correct episode that the bi-encoder finds most similar
    negatives = the best chunk of each of the G-1 highest-ranked WRONG episodes (same show, similar text)
  the model scores the G chunks, softmax over the group, loss = -log P(positive)   (cross-entropy)
Training comments never come from the val shows (val stays honest) and are not the comments the bi-encoder
was trained on (on those it is too good, so its "hard" negatives would be too easy).

Output: models/<tag>/ and results/scores/<tag>__{val,test}.pkl = reranker score of the top M chunks of the top K
episodes for every comment. combine_rerank.py then picks K, M and the stage-1 weight on val.

Usage
  python train_reranker.py v6_bi v6_first v6_rr_s0 --seed 0
  python train_reranker.py v6_bi v6_first gte_rr_zeroshot --no-train          # same pipeline, untrained reranker
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ROOT, SCORES, chunk_text, load_scores, load_split, load_summaries

QP, DP = "search_query: ", "search_document: "  # prompts of the nomic bi-encoder

ap = argparse.ArgumentParser()
ap.add_argument("bi", help="stage-1 bi-encoder (models/<bi>), used to pick chunks and mine negatives")
ap.add_argument("first", help="stage-1 score tag for val/test (e.g. bi-encoder + BM25 mix)")
ap.add_argument("tag", help="name of the saved reranker / score files")
ap.add_argument("--base", default="Alibaba-NLP/gte-reranker-modernbert-base", help="starting checkpoint (149M)")
ap.add_argument("--n", type=int, default=24000, help="training comments")
ap.add_argument("--group", type=int, default=8, help="chunks per training group: 1 positive + group-1 negatives")
ap.add_argument("--batch", type=int, default=16, help="groups per optimizer step")
ap.add_argument("--micro", type=int, default=4, help="groups per forward pass (memory only)")
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-len", type=int, default=384, help="tokens for comment + chunk")
ap.add_argument("--cap", type=int, default=15000, help="max comments per show")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kmax", type=int, default=15, help="episodes re-scored per comment at test time (max tried)")
ap.add_argument("--mmax", type=int, default=4, help="chunks re-scored per episode (max tried)")
ap.add_argument("--splits", default="val,test")
ap.add_argument("--no-train", action="store_true", help="skip training: score with the base checkpoint")
args = ap.parse_args()
print("config:", vars(args), flush=True)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
OUT = ROOT / "models" / args.tag
summ = load_summaries()

# ---- stage-1 bi-encoder: chunk embeddings per show
bi = SentenceTransformer(str(ROOT / "models" / args.bi), device="cuda")
bi.max_seq_length = args.max_len
bi.half()


def show_chunks(shows):
    """show -> (chunk texts, chunk embeddings, chunk indices of each episode)"""
    out = {}
    for show in shows:
        docs, owner = [], []
        for i, t in enumerate(summ[show][1]):
            for c in chunk_text(t):
                docs.append(c)
                owner.append(i)
        e = bi.encode([DP + d for d in docs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        owner = np.array(owner)
        out[show] = (docs, e, [np.where(owner == i)[0] for i in range(len(summ[show][1]))])
    return out


def encode_comments(comments):
    return bi.encode([QP + c for c in comments], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)


tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).cuda()


def batch_logits(pairs):
    enc = tok([c for c, _ in pairs], [d for _, d in pairs], truncation=True, max_length=args.max_len,
              padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**enc).logits.float().view(-1)


@torch.no_grad()
def predict(pairs, bs=128):
    model.eval()
    order = np.argsort([len(c) + len(d) for c, d in pairs])  # similar lengths per batch = less padding
    out = np.empty(len(pairs))
    for k in range(0, len(pairs), bs):
        idx = order[k:k + bs]
        out[idx] = batch_logits([pairs[i] for i in idx]).cpu().numpy()
    return out


# ---- 1. training data
if not args.no_train:
    t0 = time.time()
    train = load_split("train")
    val_shows = set(load_split("val").show)
    used = set(json.load(open(ROOT / "models" / args.bi / "train_idx.json")))
    train = train[~train.show.isin(val_shows) & ~train.index.isin(used)]
    train = pd.concat([g.sample(min(len(g), args.cap), random_state=args.seed) for _, g in train.groupby("show")])
    train = train.sample(min(args.n, len(train)), random_state=args.seed).reset_index(drop=True)
    print("train comments:", len(train), train.show.value_counts().to_dict(), flush=True)
    ch = show_chunks(sorted(train.show.unique()))
    q = encode_comments(train.comment.tolist())
    groups = []  # (comment, [positive chunk, negative chunks...])
    for i, (show, ep, comment) in enumerate(train.itertuples(index=False)):
        docs, e, ep_chunks = ch[show]
        sims = e @ q[i]
        best = [idx[np.argmax(sims[idx])] for idx in ep_chunks]   # best chunk of each episode
        ep_score = np.array([sims[b] for b in best])
        gold = summ[show][0].index(ep)
        wrong = [j for j in np.argsort(-ep_score) if j != gold][:args.group - 1]
        if len(wrong) == args.group - 1:
            groups.append((comment, [docs[best[gold]]] + [docs[best[j]] for j in wrong]))
    print(f"{len(groups):,} groups mined in {time.time() - t0:.0f}s", flush=True)

    # ---- 2. train (listwise cross-entropy: the positive is always position 0 of its group)
    steps = int(len(groups) * args.epochs) // args.batch
    order = np.concatenate([rng.permutation(len(groups)) for _ in range(int(np.ceil(args.epochs)))])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    log, run_loss, run_acc = [], [], []
    bi.to("cpu")  # free GPU memory for training (6 GB laptop); back on the GPU for scoring
    torch.cuda.empty_cache()
    t0 = time.time()
    model.train()
    for step in range(steps):
        ids = order[step * args.batch:(step + 1) * args.batch]
        opt.zero_grad()
        for k in range(0, len(ids), args.micro):
            mb = [groups[i] for i in ids[k:k + args.micro]]
            logits = batch_logits([(c, d) for c, ds in mb for d in ds]).view(len(mb), args.group)
            target = torch.zeros(len(mb), dtype=torch.long, device="cuda")
            loss = F.cross_entropy(logits, target)
            (loss * len(mb) / len(ids)).backward()
            run_loss.append(loss.item())
            run_acc.append((logits.argmax(1) == 0).float().mean().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            log.append({"step": step + 1, "loss": float(np.mean(run_loss)), "acc": float(np.mean(run_acc)),
                        "secs": round(time.time() - t0)})
            print(log[-1], flush=True)
            run_loss, run_acc = [], []
    train_secs = time.time() - t0
    print(f"trained in {train_secs:.0f}s", flush=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    json.dump(log, open(OUT / "train_log.json", "w"), indent=1)
    json.dump(vars(args) | {"train_comments": len(train), "groups": len(groups), "train_secs": round(train_secs)},
              open(OUT / "train_config.json", "w"), indent=1)

# ---- 3. score the top kmax episodes (top mmax chunks each) of every val / test comment
model = model.to(torch.bfloat16)
bi.to("cuda")
for split in args.splits.split(","):
    t0 = time.time()
    df = load_split(split)
    first = load_scores(args.first, split)
    ch = show_chunks(sorted(df.show.unique()))
    q = encode_comments(df.comment.tolist())
    pairs, owner = [], []  # owner = (comment, episode, chunk rank within the episode)
    for i, (show, comment) in enumerate(df[["show", "comment"]].itertuples(index=False)):
        docs, e, ep_chunks = ch[show]
        sims = e @ q[i]
        for j in np.argsort(-np.asarray(first[i], float))[:args.kmax]:
            idx = ep_chunks[j]
            for r, c in enumerate(idx[np.argsort(-sims[idx])][:args.mmax]):
                pairs.append((comment, docs[c]))
                owner.append((i, j, r))
    pred = predict(pairs)
    out = [dict() for _ in range(len(df))]  # out[i][episode] = scores of its chunks, best bi-encoder chunk first
    for (i, j, r), p in zip(owner, pred):
        out[i].setdefault(int(j), []).append(float(p))
    secs = time.time() - t0
    pickle.dump(out, open(SCORES / f"{args.tag}__{split}.pkl", "wb"))
    print(f"{split}: {len(pairs):,} pairs scored in {secs:.0f}s", flush=True)
