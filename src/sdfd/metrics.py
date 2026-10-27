"""DfD (Maboudian et al. 2022) and its semantic extension, sDfD.

Both operate on a visit pair of deduplicated word lists (L1, L2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Original DfD
# --------------------------------------------------------------------------

def dfd(l1: list[str], l2: list[str]) -> float:
    """Distance from Diagonal: sum over exact matches of |i - j|.

    Lists are deduplicated, so exact matching is one-to-one.
    """
    pos2 = {w: j for j, w in enumerate(l2)}
    return float(sum(abs(i - pos2[w]) for i, w in enumerate(l1) if w in pos2))


def n_overlap(l1: list[str], l2: list[str]) -> int:
    return len(set(l1) & set(l2))


# --------------------------------------------------------------------------
# sDfD
# --------------------------------------------------------------------------

def sigma_raw(s: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """Raw cosine above tau, no rescaling (the default)."""
    return np.where(s > tau, s, 0.0)


def sigma_gate(s: np.ndarray, tau: float) -> np.ndarray:
    """Threshold only; surviving matches keep their full cosine. Range [tau, 1]."""
    return np.where(s > tau, s, 0.0)


def sigma_hinge(s: np.ndarray, tau: float) -> np.ndarray:
    """Hinge: max(0, s - tau) rescaled to [0, 1]."""
    return np.maximum(0.0, s - tau) / (1.0 - tau)


def sdfd(
    l1: list[str],
    l2: list[str],
    embedder,
    tau: float = 0.0,
    sigma_fn=sigma_raw,
    normalise_by_length: bool = False,
) -> float:
    """Semantic Distance from Diagonal.

    Each L1 word is matched to its most similar L2 word (argmax, so several L1
    words can share one L2 word), and the positional shift is weighted by sigma.
    Asymmetric: sdfd(L1, L2) != sdfd(L2, L1).
    """
    if not l1 or not l2:
        return 0.0
    sim = embedder.similarity_matrix(l1, l2)          # (n1, n2)
    jstar = np.nanargmax(np.nan_to_num(sim, nan=-1.0), axis=1)
    best = sim[np.arange(len(l1)), jstar]
    best = np.nan_to_num(best, nan=0.0)
    weights = sigma_fn(best, tau)
    shifts = np.abs(np.arange(len(l1)) - jstar)
    total = float(np.sum(weights * shifts))
    return total / len(l1) if normalise_by_length else total


def sdfd_assignment(
    l1: list[str],
    l2: list[str],
    embedder,
    tau: float = 0.0,
    sigma_fn=sigma_raw,
    normalise_by_length: bool = False,
) -> float:
    """sDfD with one-to-one matching (Hungarian / optimal assignment).

    Each L2 word is used at most once, and the pairing maximises total
    similarity. With unequal lengths, min(n1, n2) words are matched and the
    rest contribute nothing. Symmetric, and one-to-one like DfD.
    """
    if not l1 or not l2:
        return 0.0
    from scipy.optimize import linear_sum_assignment

    sim = np.nan_to_num(embedder.similarity_matrix(l1, l2), nan=0.0)
    # maximise total similarity == minimise its negative
    rows, cols = linear_sum_assignment(-sim)

    best = sim[rows, cols]
    weights = sigma_fn(best, tau)
    shifts = np.abs(rows - cols)
    total = float(np.sum(weights * shifts))
    return total / len(l1) if normalise_by_length else total


def match_table(l1: list[str], l2: list[str], embedder, tau: float = 0.0,
                sigma_fn=sigma_raw) -> pd.DataFrame:
    """Return the argmax pairings behind sdfd(), one row per L1 word, for inspection."""
    sim = embedder.similarity_matrix(l1, l2)
    jstar = np.nanargmax(np.nan_to_num(sim, nan=-1.0), axis=1)
    best = np.nan_to_num(sim[np.arange(len(l1)), jstar], nan=0.0)
    w = sigma_fn(best, tau)
    shifts = np.abs(np.arange(len(l1)) - jstar)
    return pd.DataFrame(
        {
            "i": np.arange(len(l1)),
            "w1": l1,
            "j*": jstar,
            "w2": [l2[j] for j in jstar],
            "cosine": np.round(best, 3),
            "sigma": np.round(w, 3),
            "|i-j*|": shifts,
            "term": np.round(w * shifts, 3),
            "exact": [a == l2[j] for a, j in zip(l1, jstar)],
        }
    )
