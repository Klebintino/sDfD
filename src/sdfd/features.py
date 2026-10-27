"""Feature extraction: one row per visit pair.

Features are computed on the second list of the pair (L2), except length_diff
and the recurrence metrics, which compare both lists.

    1  list_length        unique words in L2
    2  length_diff        n2 - n1
    3  n_repeats          perseverations in L2            [snafu]
    4  n_intrusions       out-of-category words in L2     [snafu]
    5  dfd                Distance from Diagonal          [ours]
    6  (dropped)          n_clusters always equals n_switches + 1, so it is
                          kept in the table but left out of MODEL_FEATURES
    7  cluster_size       mean cluster size in L2         [snafu]
    8  n_switches         cluster switches in L2          [snafu]
    9  sem_dist_mean      mean cosine distance between consecutive words in L2
    10 sem_dist_sd        sd of those distances
    11 sdfd, sdfd_1to1    Semantic Distance from Diagonal [ours]

Features 9-11 require an embedding and are added by `add_semantic_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

# Features 1-8 need no embedding.
STRUCTURAL_FEATURES = [
    "list_length",
    "length_diff",
    "n_repeats",
    "n_intrusions",
    "dfd",
    "cluster_size",
    "n_switches",
]

SEMANTIC_FEATURES = ["sem_dist_mean", "sem_dist_sd", "sdfd", "sdfd_1to1"]

# Columns used by the models (n_clusters excluded).
MODEL_FEATURES = STRUCTURAL_FEATURES + SEMANTIC_FEATURES


def build_structural(pairs: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    """Attach features 1-8 to each visit pair.

    `sess` must already carry the snafu columns (n_persev, n_intrusions,
    cluster_size, n_switches, n_clusters) -- see normalise.merge_into_sessions.
    """
    snafu_cols = ["n_persev", "n_intrusions", "n_intrusions_scheme", "cluster_size",
                  "n_switches", "n_clusters"]
    missing = [c for c in snafu_cols if c not in sess.columns]
    if missing:
        raise ValueError(f"session table missing snafu columns: {missing}")

    l2 = sess[["id", "listnum"] + snafu_cols].rename(columns={"listnum": "listnum_2"})
    out = pairs.merge(l2, on=["id", "listnum_2"], how="left", validate="many_to_one")

    out["list_length"] = out["n2"].astype(float)
    out["length_diff"] = (out["n2"] - out["n1"]).astype(float)
    out["n_repeats"] = out["n_persev"].astype(float)
    out["dfd"] = [float(metrics.dfd(list(r.L1), list(r.L2))) for r in out.itertuples()]

    return out.drop(columns=["n_persev"])


def add_semantic_features(
    df: pd.DataFrame,
    embedder,
    tau: float = 0.0,
    sigma_fn=None,
    normalise_by_length: bool = False,
) -> pd.DataFrame:
    """Attach features 9-11. Defaults: tau=0, raw cosine, no length normalisation."""
    if sigma_fn is None:
        sigma_fn = metrics.sigma_raw

    df = df.copy()
    means, sds, sdfds, sdfd1 = [], [], [], []

    for r in df.itertuples():
        l2 = list(r.L2)
        # features 9/10: cosine DISTANCE between consecutive words in L2
        if len(l2) >= 2:
            V = embedder.matrix(l2)
            consec = np.array(
                [float(V[i] @ V[i + 1]) for i in range(len(l2) - 1)], dtype=float
            )
            d = 1.0 - consec
            d = d[~np.isnan(d)]
            means.append(float(d.mean()) if d.size else np.nan)
            sds.append(float(d.std(ddof=0)) if d.size else np.nan)
        else:
            means.append(np.nan)
            sds.append(np.nan)

        kw = dict(tau=tau, sigma_fn=sigma_fn,
                  normalise_by_length=normalise_by_length)
        sdfds.append(metrics.sdfd(list(r.L1), l2, embedder, **kw))
        sdfd1.append(metrics.sdfd_assignment(list(r.L1), l2, embedder, **kw))

    df["sem_dist_mean"] = means
    df["sem_dist_sd"] = sds
    df["sdfd"] = sdfds
    df["sdfd_1to1"] = sdfd1
    return df


def feature_matrix(df: pd.DataFrame, cols: list[str] | None = None):
    """Return (X, y, groups) ready for sklearn, with NaNs median-filled."""
    cols = cols or [c for c in MODEL_FEATURES if c in df.columns]
    X = df[cols].astype(float).copy()
    X = X.fillna(X.median(numeric_only=True))
    return X.to_numpy(), df["y"].to_numpy(), df["id"].to_numpy(), cols
