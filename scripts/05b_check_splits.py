"""List every word a vocabulary-based compound splitter would resolve.

Read-only diagnostic: prints each split sorted by the shorter fragment's
length, marking a minimum-fragment-length rule (MIN_PART).

Run:  python scripts/05b_check_splits.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collections

from sdfd import data, normalise

RAW = ROOT / "data" / "raw"
XLSX = RAW / "ucsd_fluency_for_snafu_20180518.xlsx"
CSV = RAW / "ucsd_fluency.csv"

MIN_PART = 4  # the rule under test


def all_splits(word: str, vocab: set) -> list[tuple[str, str]]:
    """Every way the word splits into two in-vocabulary parts."""
    out = []
    for i in range(2, len(word) - 1):
        a, b = word[:i], word[i:]
        if a in vocab and b in vocab:
            out.append((a, b))
    return out


def main():
    import gensim.downloader as api

    print("loading word2vec-google-news-300...")
    kv = api.load("word2vec-google-news-300")
    kv_lower = {k.lower() for k in kv.index_to_key if k.isalpha()}
    print(f"  {len(kv.index_to_key):,} words\n")

    sess = data.load_sessions(XLSX)
    sess = normalise.merge_into_sessions(sess, normalise.load_normalised(RAW, CSV))
    counts = collections.Counter()
    for ws in sess.unique_words:
        for w in ws:
            counts[w] += 1

    rows = []
    for w in sorted(counts):
        if w in kv or w.capitalize() in kv:
            continue
        sp = all_splits(w, kv_lower)
        if sp:
            # prefer the most balanced split
            best = max(sp, key=lambda ab: min(len(ab[0]), len(ab[1])))
            rows.append((w, best, min(len(best[0]), len(best[1])), counts[w], len(sp)))

    rows.sort(key=lambda r: (r[2], r[0]))

    print("=" * 72)
    print(f"ALL {len(rows)} WORDS THE SPLITTER RESOLVES")
    print(f"sorted by shorter-fragment length. proposed rule: keep if >= {MIN_PART}")
    print("=" * 72)
    print(f"{'word':<22} {'split':<28} {'min':>3} {'freq':>5} {'alts':>4}  rule")
    shown = None
    for w, (a, b), mn, freq, nalt in rows:
        if mn != shown:
            print("-" * 72)
            shown = mn
        verdict = "KEEP" if mn >= MIN_PART else "drop"
        print(f"{w:<22} {a + ' + ' + b:<28} {mn:>3} {freq:>5} {nalt:>4}  {verdict}")

    keep = [r for r in rows if r[2] >= MIN_PART]
    drop = [r for r in rows if r[2] < MIN_PART]
    print("=" * 72)
    print(f"rule would KEEP {len(keep)} words "
          f"({sum(r[3] for r in keep)} tokens), DROP {len(drop)} words "
          f"({sum(r[3] for r in drop)} tokens)")
    print("\nRead the two sections. If the KEEP block is mostly real compounds and")
    print("the drop block is mostly mangled misspellings, the rule works. If not,")
    print(f"try a different MIN_PART (edit the constant at the top of this file).")


if __name__ == "__main__":
    main()
