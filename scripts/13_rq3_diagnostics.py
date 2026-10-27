"""Embedding comparison diagnostics on Task A.

Splits sDfD into its exact-match and semantic-match parts, checks how often the
two embeddings pick the same partner for non-exact matches, and times each
embedding. Writes outputs/rq3_decomposition.csv.

Run:  python scripts/13_rq3_diagnostics.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sdfd import metrics
from sdfd.vocab import CachedEmbedder

OUT = ROOT / "outputs"


def decompose(df, emb, one_to_one: bool):
    """Split each pair's sDfD into exact-match and semantic-match contributions."""
    exact_tot, sem_tot, n_exact, n_sem = 0.0, 0.0, 0, 0
    per_pair = []

    for r in df.itertuples():
        l1, l2 = list(r.L1), list(r.L2)
        if not l1 or not l2:
            continue
        sim = np.nan_to_num(emb.similarity_matrix(l1, l2), nan=0.0)

        if one_to_one:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-sim)
        else:
            cols = np.argmax(sim, axis=1)
            rows = np.arange(len(l1))

        w = sim[rows, cols]
        shift = np.abs(rows - cols)
        term = w * shift
        is_exact = np.array([l1[i] == l2[j] for i, j in zip(rows, cols)])

        e, s = float(term[is_exact].sum()), float(term[~is_exact].sum())
        exact_tot += e
        sem_tot += s
        n_exact += int(is_exact.sum())
        n_sem += int((~is_exact).sum())
        if e + s > 0:
            per_pair.append(e / (e + s))

    total = exact_tot + sem_tot
    return {
        "exact_contribution": exact_tot / total if total else np.nan,
        "semantic_contribution": sem_tot / total if total else np.nan,
        "n_exact_matches": n_exact,
        "n_semantic_matches": n_sem,
        "median_pair_exact_share": float(np.median(per_pair)) if per_pair else np.nan,
    }


def main():
    caches = {p.stem.replace("emb_", ""): p for p in sorted(OUT.glob("emb_*.npz"))}
    if not caches:
        sys.exit("no embedding caches in outputs/")
    df = pd.read_parquet(OUT / "features_taskA.parquet")

    print("=" * 74)
    print("1. WHERE DOES sDfD'S VALUE COME FROM?")
    print("=" * 74)
    print("  exact matches score 1.0 in EVERY embedding, so any agreement between")
    print("  embeddings on that portion is guaranteed, not evidence of robustness\n")

    rows = []
    for tag, path in caches.items():
        emb = CachedEmbedder(path)
        for name, o2o in [("sdfd", False), ("sdfd_1to1", True)]:
            d = decompose(df, emb, o2o)
            rows.append({"embedding": tag, "metric": name, **d})
            print(f"  {tag} / {name}")
            print(f"     contribution from EXACT matches    : "
                  f"{d['exact_contribution']:.1%}")
            print(f"     contribution from SEMANTIC matches : "
                  f"{d['semantic_contribution']:.1%}")
            print(f"     match counts: {d['n_exact_matches']:,} exact, "
                  f"{d['n_semantic_matches']:,} semantic")
            print(f"     median per-pair exact share        : "
                  f"{d['median_pair_exact_share']:.1%}\n")

    pd.DataFrame(rows).to_csv(OUT / "rq3_decomposition.csv", index=False)

    print("=" * 74)
    print("2. DO THE EMBEDDINGS DISAGREE WHERE IT MATTERS?")
    print("=" * 74)
    tags = list(caches)
    if len(tags) >= 2:
        a, b = CachedEmbedder(caches[tags[0]]), CachedEmbedder(caches[tags[1]])
        shared = [w for w in a.words if w in b]
        agree_exact, agree_argmax, n = 0, 0, 0
        for r in df.itertuples():
            l1, l2 = list(r.L1), list(r.L2)
            if not l1 or not l2:
                continue
            ja = np.argmax(np.nan_to_num(a.similarity_matrix(l1, l2), nan=-1), axis=1)
            jb = np.argmax(np.nan_to_num(b.similarity_matrix(l1, l2), nan=-1), axis=1)
            same = ja == jb
            ex = np.array([l1[i] == l2[ja[i]] for i in range(len(l1))])
            agree_exact += int((same & ex).sum())
            agree_argmax += int((same & ~ex).sum())
            n += int((~ex).sum())
        print(f"  comparing {tags[0]} vs {tags[1]}")
        print(f"     shared vocabulary: {len(shared)} words")
        print(f"     NON-EXACT positions where both pick the SAME L2 word: "
              f"{agree_argmax}/{n} ({100*agree_argmax/max(n,1):.1f}%)")
        print("     (low agreement here + near-identical downstream scores means")
        print("      the exact-match backbone is carrying the metric)\n")

    print("=" * 74)
    print("3. COMPUTE COST")
    print("=" * 74)
    for tag, path in caches.items():
        emb = CachedEmbedder(path)
        t0 = time.perf_counter()
        for r in df.head(200).itertuples():
            metrics.sdfd(list(r.L1), list(r.L2), emb)
        dt = time.perf_counter() - t0
        print(f"  {tag}")
        print(f"     dimensions        : {emb.dim}")
        print(f"     cache size        : {path.stat().st_size/1024:.0f} KB")
        print(f"     sDfD for 200 pairs: {dt*1000:.0f} ms "
              f"({dt/200*1e6:.0f} us per pair)")
    print("\n  one-off build cost (from the cache scripts):")
    print("     word2vec-google-news-300 : 1.6 GB download, ~3.6 GB RAM to load")
    print("     gpt2                     : 474 MB download, ~0.5 GB RAM to load")
    print("  both are paid once; downstream work reads the small cache only")

    print(f"\nwrote rq3_decomposition.csv")


if __name__ == "__main__":
    main()
