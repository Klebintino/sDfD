"""Build the Word2Vec embedding cache for the animal vocabulary.

Downloads the model through gensim (word2vec-google-news-300 is ~1.6 GB) and
writes outputs/emb_<model>.npz with just the animal vectors (~520 KB), which
every later script reads instead.

Run:  python scripts/05_build_cache.py [model_name]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collections

from sdfd import data, normalise, vocab as V

RAW = ROOT / "data" / "raw"
XLSX = RAW / "ucsd_fluency_for_snafu_20180518.xlsx"
CSV = RAW / "ucsd_fluency.csv"
OUT = ROOT / "outputs"

MODEL = sys.argv[1] if len(sys.argv) > 1 else "word2vec-google-news-300"


def main():
    import gensim.downloader as api

    sess = data.load_sessions(XLSX)
    sess = normalise.merge_into_sessions(sess, normalise.load_normalised(RAW, CSV))
    animals = normalise.load_animal_list(RAW)

    counts = collections.Counter()
    for ws in sess.unique_words:
        for w in ws:
            counts[w] += 1
    words = sorted(counts)

    print(f"vocabulary: {len(words)} words, {sum(counts.values()):,} spoken tokens")
    print(f"loading {MODEL} (once) ...")
    kv = api.load(MODEL)
    print(f"  {len(kv.index_to_key):,} entries\n")

    tag = MODEL.replace("-", "_")
    path = OUT / f"emb_{tag}.npz"
    plan = V.build_cache(words, animals, kv, path)

    by_method = collections.Counter(m for m, _ in plan.values())
    tok = collections.Counter()
    for w, (m, _) in plan.items():
        tok[m] += counts[w]
    total = sum(counts.values())

    print("=" * 66)
    print("RESOLUTION")
    print("=" * 66)
    for m in ["direct", "capitalised", "compound", "UNRESOLVED"]:
        print(f"  {m:<13} {by_method[m]:>4} words ({by_method[m]/len(words):>5.1%})"
              f"   {tok[m]:>6} tokens ({tok[m]/total:>6.2%})")

    comp = [(w, p) for w, (m, p) in plan.items() if m == "compound"]
    print(f"\n  compound splits ({len(comp)}), most frequent first:")
    for w, p in sorted(comp, key=lambda x: -counts[x[0]])[:12]:
        print(f"     {w:<20} -> {' + '.join(p):<26} x{counts[w]}")

    unres = sorted((w for w, (m, _) in plan.items() if m == "UNRESOLVED"),
                   key=lambda w: -counts[w])
    print(f"\n  unresolved ({len(unres)}), most frequent first:")
    print("    ", unres[:20])

    size = path.stat().st_size / 1024
    print(f"\nwrote {path.name}  ({size:.0f} KB)")
    print("Downstream scripts load this, not the full model.")


if __name__ == "__main__":
    main()
