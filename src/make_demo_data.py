"""Make the data for report/demo.html: real test comments with the filter's real outputs.

For each test show we take a fixed random sample of short hand-labelled comments (plot and non-plot),
run the full filter (demo.py) on them and save P(plot) and P(episode) for every episode.
The page then decides hide/show for any viewer position without running a model.
"""
import json

import pandas as pd

from common import RESULTS, ROOT
from demo import SpoilerFilter
from relevance import TEST_SHOWS, load_relevance

f = load_relevance()
te = f[f.show.isin(TEST_SHOWS)]
te = te[te.comment.str.split().str.len().between(6, 35)]
flt = SpoilerFilter()
out = {}
for show in sorted(TEST_SHOWS):
    g = te[te.show == show]
    pick = pd.concat([g[g.label == 1].sample(36, random_state=1), g[g.label == 0].sample(14, random_state=1)])
    pick = pick.sample(frac=1, random_state=2)
    eps = flt.summ[show][0]
    pp = flt.p_plot(pick.comment.tolist())
    rows = []
    for (c, ep, lab), p_plot in zip(pick[["comment", "episode", "label"]].itertuples(index=False), pp):
        p = flt.episode_probs(show, c)
        rows.append({"c": c, "ep": int(ep), "plot": int(lab), "pp": round(float(p_plot), 3),
                     "pe": [round(float(x), 4) for x in p]})
    out[show] = {"episodes": [int(e) for e in eps], "comments": rows}
    print(show, len(rows))
json.dump(out, open(RESULTS / "demo_comments.json", "w"))
