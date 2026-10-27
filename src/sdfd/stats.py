"""Significance testing for model comparisons on pooled out-of-fold predictions.

Each visit pair has one held-out prediction, and the two models being
compared are scored on the same pairs.

  delong_test         DeLong test for two correlated AUCs (fast version of
                      Sun & Xu 2014). Treats pairs as independent, so it is
                      somewhat anti-conservative here.
  paired_permutation  Subject-level paired permutation: randomly swap each
                      subject's predictions between the two models. Primary test.
  cluster_bootstrap   Subject-level bootstrap, for confidence intervals.
  holm                Holm-Bonferroni correction across comparisons.

Permutation and bootstrap work for both AUC and R^2.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC via the Mann-Whitney rank formula (ties get average ranks)."""
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = _st.rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def score_fn(kind: str):
    return fast_auc if kind == "auc" else r2


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    return _st.rankdata(x)  # average ranks for ties == midranks


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """Two-sided DeLong test for AUC(a) - AUC(b) on the same cases."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    preds = np.vstack([p_a, p_b])                       # (2, N)
    k = preds.shape[0]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        x, yy = preds[r, pos], preds[r, neg]
        tx[r] = _midrank(x)
        ty[r] = _midrank(yy)
        tz[r] = _midrank(np.concatenate([x, yy]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n                           # structural components
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n

    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = np.sqrt(var) if var > 0 else np.nan
    z = diff / se if se and se > 0 else np.nan
    p = float(2 * _st.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "se": float(se), "z": float(z), "p": p,
            "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se}


def delong_single_ci(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """AUC and DeLong 95% CI for a single model."""
    yv = np.asarray(y)
    pos, neg = yv == 1, yv == 0
    m, n = int(pos.sum()), int(neg.sum())
    x, yy = p[pos], p[neg]
    tx, ty = _midrank(x), _midrank(yy)
    tz = _midrank(np.concatenate([x, yy]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    se = np.sqrt(np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n)
    a = float(tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n)
    return a, a - 1.96 * se, a + 1.96 * se


# --------------------------------------------------------------------------
# Subject-level resampling
# --------------------------------------------------------------------------

def _subject_index(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq = np.unique(groups)
    rows = [np.flatnonzero(groups == g) for g in uniq]
    return uniq, rows


def paired_permutation(y, p_a, p_b, groups, kind: str = "auc",
                       n_perm: int = 5000, seed: int = 42) -> dict:
    """Subject-level paired permutation test for score(a) - score(b).

    Two-sided p-value with the +1 correction, so it can never be exactly 0:
    p = (1 + #{|null diff| >= |observed diff|}) / (n_perm + 1).
    """
    f = score_fn(kind)
    y = np.asarray(y); p_a = np.asarray(p_a, float); p_b = np.asarray(p_b, float)
    if kind == "auc":
        # AUC only depends on ranks, so use percentile ranks: AUCs are unchanged
        # but the swapped scores share a scale (naive Bayes gives extreme probabilities)
        p_a = _st.rankdata(p_a) / len(p_a)
        p_b = _st.rankdata(p_b) / len(p_b)
    obs = f(y, p_a) - f(y, p_b)
    _, rows = _subject_index(np.asarray(groups))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        flip = rng.random(len(rows)) < 0.5
        a, bb = p_a.copy(), p_b.copy()
        if flip.any():
            idx = np.concatenate([rows[i] for i in np.flatnonzero(flip)])
            a[idx], bb[idx] = p_b[idx], p_a[idx]
        null[b] = f(y, a) - f(y, bb)
    p = (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / (n_perm + 1)
    return {"diff": float(obs), "p": float(p)}


def cluster_bootstrap(y, preds: dict, groups, kind: str = "auc",
                      n_boot: int = 2000, seed: int = 42) -> dict:
    """Subject-level bootstrap of every model's score.

    Returns {name: array of n_boot scores}. Differences between models are
    computed from the SAME resamples, so their CI respects the pairing.
    """
    f = score_fn(kind)
    y = np.asarray(y)
    _, rows = _subject_index(np.asarray(groups))
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        pick = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[i] for i in pick])
        for k, p in preds.items():
            out[k][b] = f(y[idx], np.asarray(p)[idx])
    return out


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


# --------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------

def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
