"""Add the semantic features (sem_dist_mean, sem_dist_sd, sdfd, sdfd_1to1).

Reads outputs/emb_<emb_tag>.npz and overwrites outputs/features_task{A,B1,B2}.parquet
in place, so every later script uses whichever embedding was run last. Also
prints example matches, a many-to-one absorption check and group means.

Run:  python scripts/06_semantic_features.py [emb_tag]
      emb_tag: word2vec_google_news_300 (default) or gpt2
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from sdfd import features as F, metrics
from sdfd.vocab import CachedEmbedder

OUT = ROOT / "outputs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "word2vec_google_news_300"


def main():
    emb = CachedEmbedder(OUT / f"emb_{TAG}.npz")
    print(f"embedding: {TAG}  ({len(emb.words)} words, {emb.dim} dims)\n")

    for task in ["A", "B1", "B2"]:
        p = OUT / f"features_task{task}.parquet"
        df = pd.read_parquet(p)
        df = F.add_semantic_features(df, emb, tau=0.0)  # plain: raw cosine
        df.to_parquet(p)
        print(f"  task {task:<2} {len(df):>5} pairs  +sem_dist_mean, sem_dist_sd, sdfd")

    a = pd.read_parquet(OUT / "features_taskA.parquet")

    # ---------------- the important bit ----------------
    print("\n" + "=" * 74)
    print("ACTUAL MATCHES -- read these, not the score")
    print("=" * 74)
    for label, sub in [("HC", a[a.y == 0]), ("ProbAD", a[a.y == 1])]:
        r = sub.iloc[5]
        print(f"\n--- {label}: subject {r.id}, visits {r.listnum_1} -> {r.listnum_2} ---")
        t = metrics.match_table(list(r.L1)[:12], list(r.L2)[:12], emb)
        print(t.to_string(index=False))
        bad = [(x.w1, x.w2, x.cosine) for x in t.itertuples() if not x.exact]
        print(f"  non-exact matches: {bad}")

    # absorption: how often do several L1 words grab the same L2 word?
    print("\n" + "=" * 74)
    print("ABSORPTION CHECK")
    print("=" * 74)
    coll, tot = [], 0
    for r in a.itertuples():
        l1, l2 = list(r.L1), list(r.L2)
        if not l1 or not l2:
            continue
        sim = emb.similarity_matrix(l1, l2)
        j = np.nanargmax(np.nan_to_num(sim, nan=-1.0), axis=1)
        counts = np.bincount(j, minlength=len(l2))
        coll.append(counts.max())
        tot += 1
    coll = np.array(coll)
    print(f"  max L1-words mapping to one L2-word, per pair:")
    print(f"     median {np.median(coll):.0f}   mean {coll.mean():.2f}   "
          f"90th pct {np.percentile(coll, 90):.0f}   max {coll.max()}")
    print(f"  pairs where 4+ words collapse onto one: "
          f"{(coll >= 4).sum()} of {tot} ({100*(coll >= 4).mean():.1f}%)")

    # ---------------- group differences ----------------
    print("\n" + "=" * 74)
    print("FEATURES BY GROUP (Task A)")
    print("=" * 74)
    cols = [c for c in F.MODEL_FEATURES if c in a.columns]
    summ = a.groupby(a.y.map({0: "HC", 1: "ProbAD"}))[cols].mean().T
    summ["diff"] = summ["HC"] - summ["ProbAD"]
    summ["welch_t"] = [
        stats.ttest_ind(a.loc[a.y == 0, c].dropna(),
                        a.loc[a.y == 1, c].dropna(), equal_var=False).statistic
        for c in summ.index
    ]
    print(summ.round(3).to_string())

    # ---------------- confounds ----------------
    print("\n" + "=" * 74)
    print("CONFOUND CHECKS")
    print("=" * 74)
    print(f"  corr(dfd,  list_length) = {a.dfd.corr(a.list_length):+.3f}")
    print(f"  corr(sdfd, list_length) = {a.sdfd.corr(a.list_length):+.3f}")
    print(f"  corr(dfd,  sdfd)        = {a.dfd.corr(a.sdfd):+.3f}"
          f"   <- collinearity, drives the joint-scramble decision")
    print("\n  full correlation matrix:")
    print(a[cols].corr().round(2).to_string())


if __name__ == "__main__":
    main()
