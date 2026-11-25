"""Demo: the full progress-aware spoiler filter on one comment.

You give: a show, the last episode you watched, a comment.
It answers: is the comment about the plot? which episode is it about? should it be hidden from you?

Steps (the same models and settings as the results in the report):
  1. P(plot)      fine-tuned classifier (models/relevance_ft_base)
  2. episodes     v6 matcher: bi-encoder + BM25 keep the top 15 episodes, the reranker reads the 4 best
                  summary chunks of each; final = z(rerank) + 1.0 * z(stage 1); softmax with temperature T
                  (T = 1.2512, fitted on val in filter_eval.py) turns scores into probabilities
  3. P(spoiler)   = P(plot) x P(the comment is about an episode AFTER the one you watched)
                  hide the comment if P(spoiler) > 0.5

Usage
  python demo.py attack_on_titan 80 "Zeke got shot by Magath and fell off the wall"
  python demo.py --examples          # a few fixed examples
  python demo.py --shows             # list shows and their episode numbers
Runs on the laptop GPU (or CPU, slower). Needs models/v6_bi, models/v6_rr48_s2, models/relevance_ft_base.
"""
import sys

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, chunk_text, load_summaries, tokenize
from fuse import z

K, M, A, W_BM25, T, MAX_LEN = 15, 4, 1.0, 0.3, 1.2512, 384
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32


class SpoilerFilter:
    def __init__(self):
        self.summ = load_summaries()
        self.bi = SentenceTransformer(str(ROOT / "models/v6_bi"), device=DEV, model_kwargs={"dtype": DT})
        self.bi.max_seq_length = MAX_LEN
        self.rr_tok = AutoTokenizer.from_pretrained(ROOT / "models/v6_rr48_s2")
        self.rr = AutoModelForSequenceClassification.from_pretrained(ROOT / "models/v6_rr48_s2", dtype=DT).to(DEV).eval()
        self.rel_tok = AutoTokenizer.from_pretrained(ROOT / "models/relevance_ft_base")
        self.rel = AutoModelForSequenceClassification.from_pretrained(ROOT / "models/relevance_ft_base", dtype=DT).to(DEV).eval()
        self.index = {}

    def show_index(self, show):
        """Chunks, chunk vectors and BM25 index of one show (made once, then cached)."""
        if show not in self.index:
            docs, owner = [], []
            for i, t in enumerate(self.summ[show][1]):
                for c in chunk_text(t):
                    docs.append(c)
                    owner.append(i)
            owner = np.array(owner)
            e = self.bi.encode(["search_document: " + d for d in docs], batch_size=32, normalize_embeddings=True)
            self.index[show] = (docs, owner, e, BM25Okapi([tokenize(d) for d in docs]))
        return self.index[show]

    @torch.no_grad()
    def p_plot(self, comments):
        b = self.rel_tok(comments, truncation=True, max_length=128, padding=True, return_tensors="pt").to(DEV)
        return torch.softmax(self.rel(**b).logits.float(), -1)[:, 1].cpu().numpy()

    @torch.no_grad()
    def episode_probs(self, show, comment):
        docs, owner, e, bm = self.show_index(show)
        n = len(self.summ[show][0])
        sims = e @ self.bi.encode("search_query: " + comment, normalize_embeddings=True)
        dense = np.array([sims[owner == i].max() for i in range(n)])
        lex = np.full(n, -1e9)
        np.maximum.at(lex, owner, bm.get_scores(tokenize(comment)))
        first = z(dense) + W_BM25 * z(lex)                     # stage 1
        top = np.argsort(-first)[:K]
        pairs, who = [], []
        for j in top:                                          # stage 2: M best chunks of each top episode
            idx = np.where(owner == j)[0]
            for c in idx[np.argsort(-sims[idx])][:M]:
                pairs.append((comment, docs[c]))
                who.append(j)
        b = self.rr_tok([p for p, _ in pairs], [d for _, d in pairs], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(DEV)
        out = self.rr(**b).logits.float().view(-1).cpu().numpy()
        best = np.array([out[np.array(who) == j].max() for j in top])   # MaxP
        final = first - 1e6
        final[top] = z(best) + A * z(first[top])
        return softmax(final / T)

    def check(self, show, watched, comment):
        eps = self.summ[show][0]
        p = self.episode_probs(show, comment)
        p_plot = float(self.p_plot([comment])[0])
        p_later = float(p[[i for i, e in enumerate(eps) if e > watched]].sum())
        p_spoiler = p_plot * p_later
        top = np.argsort(-p)[:3]
        return {"p_plot": p_plot, "top_episodes": [(eps[i], float(p[i])) for i in top], "p_later": p_later,
                "p_spoiler": p_spoiler, "hide": p_spoiler > 0.5}


def show_result(show, watched, comment, r):
    print(f"\n{show}, you watched up to episode {watched}")
    print(f'  comment: "{comment}"')
    print(f"  about the plot?          {r['p_plot']:.0%}")
    print("  most likely episodes:    " + ", ".join(f"ep {e} ({q:.0%})" for e, q in r["top_episodes"]))
    print(f"  about a later episode?   {r['p_later']:.0%}")
    print(f"  P(spoiler for you)       {r['p_spoiler']:.0%}  ->  {'HIDE' if r['hide'] else 'show'}")


EXAMPLES = [  # real test comments; right episodes: Yelena comment = ep 77, Korra comment = ep 18
    ("attack_on_titan", 70, "The emotional whiplash Yelena experienced from seeing her God in Zeke light up those "
     "Marley blimps to watching said God get sniped by Magath and tumble off the Wall"),
    ("attack_on_titan", 85, "The emotional whiplash Yelena experienced from seeing her God in Zeke light up those "
     "Marley blimps to watching said God get sniped by Magath and tumble off the Wall"),
    ("attack_on_titan", 70, "This episode was so good, the animation was amazing"),
    ("legend_of_korra", 10, "So Korra lost her memory, right?"),
]

if __name__ == "__main__":
    if "--shows" in sys.argv:
        for s, (eps, _) in load_summaries().items():
            print(f"{s:32s} episodes {eps[0]}..{eps[-1]} ({len(eps)})")
        sys.exit()
    f = SpoilerFilter()
    items = EXAMPLES if "--examples" in sys.argv or len(sys.argv) < 4 else [(sys.argv[1], int(sys.argv[2]), " ".join(sys.argv[3:]))]
    for show, watched, comment in items:
        show_result(show, watched, comment, f.check(show, watched, comment))
