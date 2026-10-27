"""Report how much of the normalised vocabulary Word2Vec can resolve, and how.

Read-only diagnostic: loads word2vec-google-news-300 through gensim and prints
coverage by strategy (direct, capitalised, compound split, unresolved).

Run:  python scripts/05a_check_coverage.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collections
import re

from sdfd import data, normalise

RAW = ROOT / "data" / "raw"
XLSX = RAW / "ucsd_fluency_for_snafu_20180518.xlsx"
CSV = RAW / "ucsd_fluency.csv"


def split_compound(word: str, vocab: set) -> list[str] | None:
    """Try to split a concatenated compound into two known words.

    'arabianhorse' -> ['arabian', 'horse'].  Returns None if no split works.
    """
    for i in range(3, len(word) - 2):
        a, b = word[:i], word[i:]
        if a in vocab and b in vocab:
            return [a, b]
    return None


def main():
    import gensim.downloader as api

    print("loading word2vec-google-news-300 (this takes a minute)...")
    kv = api.load("word2vec-google-news-300")
    print(f"  loaded {len(kv.index_to_key):,} words\n")

    sess = data.load_sessions(XLSX)
    sess = normalise.merge_into_sessions(sess, normalise.load_normalised(RAW, CSV))

    counts = collections.Counter()
    for ws in sess.unique_words:
        for w in ws:
            counts[w] += 1
    vocab = sorted(counts)

    # lowercase view of the embedding vocabulary, for compound splitting
    kv_lower = {k.lower() for k in kv.index_to_key if k.islower() or k.isalpha()}

    buckets = collections.defaultdict(list)
    for w in vocab:
        if w in kv:
            buckets["direct"].append(w)
        elif w.capitalize() in kv:
            buckets["capitalised"].append(w)
        elif split_compound(w, kv_lower):
            buckets["compound_split"].append(w)
        else:
            buckets["UNRESOLVED"].append(w)

    print("=" * 68)
    print(f"COVERAGE over {len(vocab)} normalised words")
    print("=" * 68)
    for k in ["direct", "capitalised", "compound_split", "UNRESOLVED"]:
        ws = buckets[k]
        tokens = sum(counts[w] for w in ws)
        total_tokens = sum(counts.values())
        print(f"  {k:<16} {len(ws):>4} words ({len(ws)/len(vocab):>5.1%})"
              f"   {tokens:>6} tokens ({tokens/total_tokens:>5.1%})")

    print("\n  compound splits found (sample):")
    for w in buckets["compound_split"][:12]:
        print(f"     {w:<22} -> {split_compound(w, kv_lower)}")

    print("\n  UNRESOLVED, by how often they appear:")
    unres = sorted(buckets["UNRESOLVED"], key=lambda w: -counts[w])
    for w in unres[:25]:
        print(f"     {w:<22} x{counts[w]}")
    once = [w for w in unres if counts[w] == 1]
    print(f"\n  {len(once)} of the {len(unres)} unresolved appear only ONCE in the "
          f"whole dataset.")

    total_tokens = sum(counts.values())
    unres_tokens = sum(counts[w] for w in buckets["UNRESOLVED"])
    print(f"\n  bottom line: {unres_tokens} of {total_tokens} spoken words "
          f"({unres_tokens/total_tokens:.2%}) cannot be embedded.")


if __name__ == "__main__":
    main()
