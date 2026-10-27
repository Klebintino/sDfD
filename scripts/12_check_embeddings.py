"""Sanity-check every cached embedding in outputs/emb_*.npz.

Prints anisotropy (mean pairwise cosine), cosines for known related and
unrelated animal pairs, and Cohen's d between the two pair sets.

Run:  python scripts/12_check_embeddings.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import itertools

import numpy as np

from sdfd.vocab import CachedEmbedder

OUT = ROOT / "outputs"

# pairs a human would call clearly related / clearly unrelated
RELATED = [("dog", "cat"), ("horse", "pony"), ("lion", "tiger"),
           ("cow", "bull"), ("mouse", "rat"), ("frog", "toad"),
           ("wolf", "fox"), ("whale", "dolphin"), ("eagle", "hawk"),
           ("sheep", "goat")]
UNRELATED = [("dog", "whale"), ("cat", "eagle"), ("horse", "shark"),
             ("mouse", "elephant"), ("frog", "lion"), ("cow", "penguin"),
             ("wolf", "goldfish"), ("bear", "butterfly"), ("snake", "cow"),
             ("rabbit", "octopus")]


def report(tag: str, path: Path):
    emb = CachedEmbedder(path)
    V = emb.vectors
    print("=" * 68)
    print(f"{tag}   ({len(emb.words)} words, {emb.dim} dims)")
    print("=" * 68)

    # 1. anisotropy -- sample pairs rather than all n^2
    rng = np.random.default_rng(0)
    idx = rng.choice(len(V), size=min(400, len(V)), replace=False)
    S = V[idx] @ V[idx].T
    off = S[~np.eye(len(idx), dtype=bool)]
    print(f"\n  1. ANISOTROPY")
    print(f"     mean pairwise cosine over random pairs : {off.mean():+.3f}")
    print(f"     sd                                     : {off.std():.3f}")
    print(f"     5th / 95th percentile                  : "
          f"{np.percentile(off, 5):+.3f} / {np.percentile(off, 95):+.3f}")
    verdict = ("ISOTROPIC (healthy)" if abs(off.mean()) < 0.15
               else "ANISOTROPIC -- vectors share a common direction")
    print(f"     -> {verdict}")

    # 2. known pairs
    def cos(a, b):
        va, vb = emb.vector(a), emb.vector(b)
        return None if va is None or vb is None else float(va @ vb)

    print(f"\n  2. KNOWN PAIRS")
    rel = [(a, b, cos(a, b)) for a, b in RELATED]
    unr = [(a, b, cos(a, b)) for a, b in UNRELATED]
    rel = [(a, b, c) for a, b, c in rel if c is not None]
    unr = [(a, b, c) for a, b, c in unr if c is not None]
    print("     related:")
    for a, b, c in rel:
        print(f"        {a + '/' + b:<22} {c:+.3f}")
    print("     unrelated:")
    for a, b, c in unr:
        print(f"        {a + '/' + b:<22} {c:+.3f}")

    # 3. discrimination
    r = np.array([c for _, _, c in rel])
    u = np.array([c for _, _, c in unr])
    pooled = np.sqrt((r.var(ddof=1) + u.var(ddof=1)) / 2)
    d = (r.mean() - u.mean()) / pooled if pooled > 0 else 0.0
    overlap = sum(1 for x in u if x > r.min())
    print(f"\n  3. DISCRIMINATION")
    print(f"     related   mean {r.mean():+.3f}")
    print(f"     unrelated mean {u.mean():+.3f}")
    print(f"     gap {r.mean() - u.mean():+.3f}   Cohen's d {d:.2f}")
    print(f"     unrelated pairs scoring above the WEAKEST related pair: "
          f"{overlap}/{len(u)}")
    if d > 2:
        print("     -> separates related from unrelated cleanly")
    elif d > 1:
        print("     -> separates, but with overlap")
    else:
        print("     -> POOR separation; cosine here carries little signal")
    print()


def main():
    caches = sorted(OUT.glob("emb_*.npz"))
    if not caches:
        sys.exit("no embedding caches found in outputs/")
    for p in caches:
        report(p.stem.replace("emb_", ""), p)

    print("=" * 68)
    print("READING THIS")
    print("=" * 68)
    print("  A high mean pairwise cosine means the space is anisotropic: every")
    print("  vector points roughly the same way, so cosine values are squeezed")
    print("  into a narrow band and small differences carry the signal.")
    print()
    print("  What matters for sDfD is test 3, not test 1. An anisotropic space")
    print("  can still work if related pairs sit consistently above unrelated")
    print("  ones. If Cohen's d is much lower for one embedding, that embedding")
    print("  genuinely is worse for this task -- not a pipeline bug.")


if __name__ == "__main__":
    main()
