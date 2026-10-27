"""Resolve the animal vocabulary to vectors and cache them as a small .npz.

The resolution plan is decided once (from Word2Vec) and the same word set is
used for every embedding, so embedding comparisons are not about coverage.

Resolution order:
    direct       word is in the embedding as-is
    capitalised  "Chihuahua" etc.
    compound     [modifier][animal] -> mean of the two part vectors. The tail
                 must be in animals_list.txt (checking against the embedding
                 vocabulary gives spurious splits like guine + apig).
    UNRESOLVED   no vector, for every embedding
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def split_modifier_animal(word: str, animals: set[str]) -> tuple[str, str] | None:
    """Split [modifier][animal] where the tail is a known animal.

    Prefers the longest valid tail: 'mountainlion' -> ('mountain', 'lion').
    Accepts a plural tail ('redsquirrels' -> 'red' + 'squirrels').
    """
    best = None
    for i in range(2, len(word) - 1):
        head, tail = word[:i], word[i:]
        if tail in animals or (tail.endswith("s") and tail[:-1] in animals):
            if best is None or len(tail) > len(best[1]):
                best = (head, tail)
    return best


def resolve_vocabulary(vocab: list[str], animals: set[str], kv) -> dict:
    """Decide how each word resolves. Returns {word: (method, parts)}."""
    plan = {}
    for w in vocab:
        if w in kv:
            plan[w] = ("direct", [w])
        elif w.capitalize() in kv:
            plan[w] = ("capitalised", [w.capitalize()])
        else:
            sp = split_modifier_animal(w, animals)
            if sp and all(p in kv or p.capitalize() in kv for p in sp):
                parts = [p if p in kv else p.capitalize() for p in sp]
                plan[w] = ("compound", parts)
            else:
                plan[w] = ("UNRESOLVED", [])
    return plan


def build_cache(vocab: list[str], animals: set[str], kv, out_path: str | Path) -> dict:
    """Extract vectors under the resolution plan and save to .npz."""
    plan = resolve_vocabulary(vocab, animals, kv)

    words, vecs = [], []
    for w, (method, parts) in plan.items():
        if method == "UNRESOLVED":
            continue
        v = np.mean([np.asarray(kv[p], dtype=np.float32) for p in parts], axis=0)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        words.append(w)
        vecs.append(v / n)  # unit-normalised, so cosine is a dot product

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        words=np.array(words),
        vectors=np.vstack(vecs).astype(np.float32),
        methods=np.array([plan[w][0] for w in words]),
    )
    return plan


class CachedEmbedder:
    """Word vectors from a cached .npz (no gensim needed).

    Unresolved words return None, and similarities involving them are NaN.
    """

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.words = list(z["words"])
        self.vectors = z["vectors"]
        self.methods = dict(zip(self.words, z["methods"]))
        self._idx = {w: i for i, w in enumerate(self.words)}
        self.dim = self.vectors.shape[1]

    def __contains__(self, word: str) -> bool:
        return word in self._idx

    def vector(self, word: str):
        i = self._idx.get(word)
        return None if i is None else self.vectors[i]

    def matrix(self, words: list[str]) -> np.ndarray:
        """(n, d), unit-normalised. Unresolved rows are NaN."""
        out = np.full((len(words), self.dim), np.nan, dtype=np.float32)
        for i, w in enumerate(words):
            j = self._idx.get(w)
            if j is not None:
                out[i] = self.vectors[j]
        return out

    def similarity_matrix(self, l1: list[str], l2: list[str]) -> np.ndarray:
        return self.matrix(l1) @ self.matrix(l2).T
