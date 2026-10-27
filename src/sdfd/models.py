"""Model families, evaluation, and feature interrogation.

Four model families, evaluated on the frozen subject-level folds. Feature
importance comes from SHAP and from scrambling (permutation), with optional
joint scrambling for correlated features such as dfd and sdfd.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def model_families(seed: int = 42) -> dict:
    """The four families. Scaled where the model needs it."""
    fams = {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1),
        "naive_bayes": Pipeline([
            ("scale", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        fams["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1)
    except ImportError:
        print("  (xgboost not installed -- skipping that family)")
    return fams


def _scores(y, pred, prob) -> dict:
    return {
        "auc": roc_auc_score(y, prob),
        "f1_macro": f1_score(y, pred, average="macro"),
        "kappa": cohen_kappa_score(y, pred),
        "precision_ad": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_ad": recall_score(y, pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def cross_validate(model, X, y, groups, folds, return_models: bool = False):
    """Evaluate on the frozen folds. Returns per-fold scores and OOF predictions."""
    from sklearn.base import clone

    rows, fitted = [], []
    oof_prob = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1)

    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        # xgboost needs the imbalance handled explicitly
        if m.__class__.__name__ == "XGBClassifier":
            pos = max((y[tr] == 1).sum(), 1)
            m.set_params(scale_pos_weight=float((y[tr] == 0).sum() / pos))
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[te], oof_pred[te] = prob, pred
        rows.append({"fold": f["fold"], **_scores(y[te], pred, prob)})
        if return_models:
            fitted.append((m, tr, te))

    out = pd.DataFrame(rows)
    return (out, oof_prob, oof_pred, fitted) if return_models else (out, oof_prob, oof_pred)


def baseline_scores(X, y, groups, folds) -> pd.DataFrame:
    """Majority-class (prior) baseline."""
    return cross_validate(DummyClassifier(strategy="prior"), X, y, groups, folds)[0]


def scramble_importance(model, X, y, groups, folds, cols, seed=42,
                        joint_groups: list[list[str]] | None = None) -> pd.DataFrame:
    """Permutation importance on held-out folds, refitting per fold.

    For each feature: shuffle it in the TEST fold only, re-score, record the
    drop in AUC. A large drop means the model was relying on it.

    joint_groups lets correlated features be shuffled together, e.g.
    [["dfd", "sdfd"]] -- otherwise each covers for the other.
    """
    from sklearn.base import clone

    rng = np.random.default_rng(seed)
    targets = [[c] for c in cols] + (joint_groups or [])
    rows = []

    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        if m.__class__.__name__ == "XGBClassifier":
            pos = max((y[tr] == 1).sum(), 1)
            m.set_params(scale_pos_weight=float((y[tr] == 0).sum() / pos))
        m.fit(X[tr], y[tr])
        base = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])

        for group in targets:
            idx = [cols.index(c) for c in group]
            drops = []
            for _ in range(5):                      # average over 5 shuffles
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, idx] = Xp[perm][:, idx]       # shuffle jointly, keeping
                                                    # within-group correlation
                drops.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
            rows.append({"feature": "+".join(group), "fold": f["fold"],
                         "auc_drop": float(np.mean(drops))})

    return (pd.DataFrame(rows).groupby("feature")["auc_drop"]
            .agg(["mean", "std"]).sort_values("mean", ascending=False))


def regression_families(seed: int = 42) -> dict:
    """Regression counterparts of the four families, for Tasks B1 and B2.

    Bayesian ridge stands in for naive Bayes, which has no regressor.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge, Ridge

    fams = {
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=seed)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2,
            random_state=seed, n_jobs=-1),
        "bayesian_ridge": Pipeline([
            ("scale", StandardScaler()),
            ("reg", BayesianRidge()),
        ]),
    }
    try:
        from xgboost import XGBRegressor
        fams["xgboost"] = XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=seed, n_jobs=-1)
    except ImportError:
        print("  (xgboost not installed -- skipping that family)")
    return fams


def _reg_scores(y, pred) -> dict:
    import warnings

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import ConstantInputWarning, pearsonr

    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "pearson": _safe_pearson(y, pred),
    }


def _safe_pearson(y, pred) -> float:
    """0.0 for a constant prediction (the mean baseline), rather than a warning."""
    import warnings

    from scipy.stats import pearsonr

    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pearsonr(y, pred)[0])


def cross_validate_reg(model, X, y, groups, folds):
    """Regression counterpart of cross_validate. Returns scores and OOF preds."""
    from sklearn.base import clone

    rows = []
    oof = np.full(len(y), np.nan)
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        oof[te] = pred
        rows.append({"fold": f["fold"], **_reg_scores(y[te], pred)})
    return pd.DataFrame(rows), oof


def baseline_reg(X, y, groups, folds) -> pd.DataFrame:
    """Baseline that always predicts the training mean (R^2 <= 0 is no better)."""
    from sklearn.dummy import DummyRegressor
    return cross_validate_reg(DummyRegressor(strategy="mean"), X, y, groups, folds)[0]


def scramble_importance_reg(model, X, y, groups, folds, cols, seed=42,
                            joint_groups: list[list[str]] | None = None):
    """Permutation importance for regression. Drop in R^2 when shuffled."""
    from sklearn.base import clone
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    targets = [[c] for c in cols] + (joint_groups or [])
    rows = []

    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        m.fit(X[tr], y[tr])
        base = r2_score(y[te], m.predict(X[te]))
        for group in targets:
            idx = [cols.index(c) for c in group]
            drops = []
            for _ in range(5):
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, idx] = Xp[perm][:, idx]
                drops.append(base - r2_score(y[te], m.predict(Xp)))
            rows.append({"feature": "+".join(group), "fold": f["fold"],
                         "r2_drop": float(np.mean(drops))})

    return (pd.DataFrame(rows).groupby("feature")["r2_drop"]
            .agg(["mean", "std"]).sort_values("mean", ascending=False))


def shap_importance_reg(model, X, y, folds, cols):
    """Mean |SHAP| per feature for regression models."""
    try:
        import shap
    except ImportError:
        return None
    from sklearn.base import clone

    tree_like = ("RandomForestRegressor", "XGBRegressor")
    acc = []
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        m.fit(X[tr], y[tr])
        if m.__class__.__name__ in tree_like:
            sv = shap.TreeExplainer(m).shap_values(X[te])
        else:
            bg = shap.kmeans(X[tr], 20)
            sv = shap.KernelExplainer(m.predict, bg).shap_values(
                X[te], nsamples=100, silent=True)
        acc.append(np.abs(sv).mean(axis=0))

    return (pd.DataFrame({"feature": cols, "mean_abs_shap": np.mean(acc, axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .set_index("feature"))


def shap_importance(model, X, y, folds, cols) -> pd.DataFrame | None:
    """Mean |SHAP| per feature, averaged over folds. None if shap is missing."""
    try:
        import shap
    except ImportError:
        return None
    from sklearn.base import clone

    tree_like = ("RandomForestClassifier", "XGBClassifier")
    acc = []
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        if m.__class__.__name__ == "XGBClassifier":
            pos = max((y[tr] == 1).sum(), 1)
            m.set_params(scale_pos_weight=float((y[tr] == 0).sum() / pos))
        m.fit(X[tr], y[tr])

        if m.__class__.__name__ in tree_like:
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(X[te])
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            bg = shap.kmeans(X[tr], 20)
            ex = shap.KernelExplainer(lambda d: m.predict_proba(d)[:, 1], bg)
            sv = ex.shap_values(X[te], nsamples=100, silent=True)
        acc.append(np.abs(sv).mean(axis=0))

    return (pd.DataFrame({"feature": cols, "mean_abs_shap": np.mean(acc, axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .set_index("feature"))
