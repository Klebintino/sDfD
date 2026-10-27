"""Visit-pair construction and frozen subject-level cross-validation splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_pairs(sess: pd.DataFrame, require_adjacent_listnum: bool = False) -> pd.DataFrame:
    """Every consecutive pair of sessions per subject.

    Each pair is labelled by the diagnosis and scores at the second visit.
    `listnum` has gaps for some subjects; `listnum_gap` records the step size.
    Set `require_adjacent_listnum=True` to keep only gap == 1 pairs.
    """
    rows = []
    for sid, g in sess.sort_values(["id", "listnum"]).groupby("id"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.loc[i], g.loc[i + 1]
            rows.append(
                {
                    "id": sid,
                    "listnum_1": a["listnum"],
                    "listnum_2": b["listnum"],
                    "listnum_gap": b["listnum"] - a["listnum"],
                    "L1": a["unique_words"],
                    "L2": b["unique_words"],
                    "L2_with_repeats": b["words"],
                    "dx_1": a["dx"],
                    "dx": b["dx"],            # label = diagnosis at second visit
                    "dx_raw": b["dx_raw"],
                    "MMSE": b["MMSE"],
                    "DRS": b["DRS"],
                    "SEX": b["SEX"],
                    "EDUC": b["EDUC"],
                    "n1": a["n_unique"],
                    "n2": b["n_unique"],
                    "n_repeats_2": b["n_repeats"],
                }
            )
    pairs = pd.DataFrame(rows)
    if require_adjacent_listnum:
        pairs = pairs[pairs["listnum_gap"] == 1].reset_index(drop=True)
    return pairs


def task_a(pairs: pd.DataFrame) -> pd.DataFrame:
    """Binary classification: pairs whose second visit is HC or ProbAD."""
    out = pairs[pairs["dx"].isin(["HC", "ProbAD"])].copy()
    out["y"] = (out["dx"] == "ProbAD").astype(int)
    return out.reset_index(drop=True)


def task_b(pairs: pd.DataFrame, score: str = "MMSE") -> pd.DataFrame:
    """Regression: AD-spectrum pairs with a valid score on `score`."""
    spectrum = ["HC", "MCI", "AtRisk", "ProbAD"]
    out = pairs[pairs["dx"].isin(spectrum) & pairs[score].notna()].copy()
    out["y"] = out[score].astype(float)
    return out.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 42, stratify_on: str | None = "y"
) -> list[dict]:
    """Subject-level CV folds, so no subject's pairs straddle train and test."""
    groups = df["id"].to_numpy()
    if stratify_on is not None and stratify_on in df:
        y = df[stratify_on].to_numpy()
        if np.issubdtype(y.dtype, np.floating):  # regression -> bin for stratification
            y = pd.qcut(y, q=4, labels=False, duplicates="drop")
    else:
        y = np.zeros(len(df))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, te) in enumerate(splitter.split(df, y, groups)):
        folds.append({"fold": k, "train_idx": tr.tolist(), "test_idx": te.tolist()})
    return folds


def save_splits(folds: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(folds))


def load_splits(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def assert_no_subject_leak(df: pd.DataFrame, folds: list[dict]) -> None:
    for f in folds:
        tr = set(df.iloc[f["train_idx"]]["id"])
        te = set(df.iloc[f["test_idx"]]["id"])
        assert not (tr & te), f"subject leak in fold {f['fold']}: {tr & te}"
