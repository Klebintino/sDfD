"""Out-of-fold SHAP beeswarm plots for all four model families, per task.

Each fold model explains its held-out pairs. Dot position is the SHAP value,
colour is the feature value (blue = low, vermillion = high). Explainers: Tree
(random forest, XGBoost), Linear on standardised features (logistic, ridge,
Bayesian ridge), Kernel with a 20-point k-means background (naive Bayes,
approximate). Units differ by explainer and are shown on each panel.

Run:  python scripts/18_shap_beeswarm.py [--tasks A,B1,B2]
Writes outputs/figures/shap_beeswarm_task{A,B1,B2}.png
       outputs/shap_oof_values_task*.parquet
       outputs/shap_oof_ranks_task*.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone

from sdfd import features as F, models as M, pairs as P

OUT = ROOT / "outputs"
FIG = OUT / "figures"
RECURRENCE = {"dfd", "sdfd", "sdfd_1to1"}
# Okabe-Ito blue -> light grey -> vermillion, matching the paper figures (script 19)
CMAP = LinearSegmentedColormap.from_list("okabe_ito_div", ["#0072B2", "#d9d9d9", "#D55E00"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "mathtext.fontset": "dejavuserif"})
LABELS = {"list_length": "List length", "length_diff": "Length difference",
          "n_repeats": "Repetitions", "n_intrusions": "Intrusions", "dfd": "DfD",
          "cluster_size": "Mean cluster size", "n_switches": "Switches",
          "sem_dist_mean": "Mean consecutive distance", "sem_dist_sd": "SD of consecutive distance",
          "sdfd": "sDfD (many-to-one)", "sdfd_1to1": "sDfD (one-to-one)"}
TITLES = {"logistic": "Logistic regression", "random_forest": "Random forest",
          "naive_bayes": "Naive Bayes", "xgboost": "XGBoost", "ridge": "Ridge",
          "bayesian_ridge": "Bayesian ridge"}


def oof_shap(task, fam, model, X, y, folds):
    import shap

    is_reg = task != "A"
    sv = np.full(X.shape, np.nan)
    units = None
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        m = clone(model)
        if m.__class__.__name__ == "XGBClassifier":
            pos = max((y[tr] == 1).sum(), 1)
            m.set_params(scale_pos_weight=float((y[tr] == 0).sum() / pos))
        m.fit(X[tr], y[tr])
        name = m.__class__.__name__
        if name in ("RandomForestClassifier", "XGBClassifier",
                    "RandomForestRegressor", "XGBRegressor"):
            vals = shap.TreeExplainer(m).shap_values(X[te])
            if isinstance(vals, list):
                vals = vals[1]
            elif np.ndim(vals) == 3:
                vals = vals[:, :, 1]
            units = {"RandomForestClassifier": "P(ProbAD)",
                     "XGBClassifier": "log-odds ProbAD"}.get(name, "score points")
        elif fam == "naive_bayes":
            bg = shap.kmeans(X[tr], 20)
            ex = shap.KernelExplainer(lambda d: m.predict_proba(d)[:, 1], bg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals = ex.shap_values(X[te], nsamples=100, silent=True)
            units = "P(ProbAD)"
        else:  # scaled linear pipeline: logistic / ridge / bayesian ridge
            scaler, est = m[0], m[-1]
            bg = scaler.transform(X[tr])
            ex = shap.LinearExplainer(
                est, shap.maskers.Independent(bg, max_samples=len(bg)))
            vals = ex.shap_values(scaler.transform(X[te]))
            if isinstance(vals, list):
                vals = vals[-1]
            units = "log-odds ProbAD" if not is_reg else "score points"
        sv[te] = np.asarray(vals)
    assert not np.isnan(sv).any()
    return sv, units


def beeswarm(ax, sv, X, cols, title, units, rng):
    order = np.argsort(np.abs(sv).mean(axis=0))          # least important at bottom
    for row, j in enumerate(order):
        v = sv[:, j]
        pct = (rankdata(X[:, j]) - 1) / max(len(v) - 1, 1)
        # density-aware jitter: wider where dots pile up
        hist, edges = np.histogram(v, bins=40)
        dens = hist[np.clip(np.digitize(v, edges) - 1, 0, len(hist) - 1)] / max(hist.max(), 1)
        jit = rng.uniform(-1, 1, len(v)) * 0.38 * np.sqrt(dens)
        ax.scatter(v, row + jit, c=pct, cmap=CMAP, s=4, alpha=0.7,
                   linewidths=0, rasterized=True)
    ax.axvline(0, color="#888", lw=0.7, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS.get(cols[j], cols[j]) for j in order], fontsize=8)
    for lab, j in zip(ax.get_yticklabels(), order):
        if cols[j] in RECURRENCE:
            lab.set_fontweight("bold")
    ax.set_xlabel(f"SHAP value ({units})", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="A,B1,B2")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for task in args.tasks.split(","):
        df = pd.read_parquet(OUT / f"features_task{task}.parquet")
        folds = P.load_splits(OUT / f"splits_task{task}.json")
        X, y, groups, cols = F.feature_matrix(df)
        fams = M.regression_families() if task != "A" else M.model_families()

        fig, axes = plt.subplots(2, 2, figsize=(12, 9.5),
                                 gridspec_kw={"wspace": 0.55, "hspace": 0.35})
        store, ranks = {}, {}
        for ax, (fam, model) in zip(axes.ravel(), fams.items()):
            print(f"  task {task}: {fam} ...", flush=True)
            sv, units = oof_shap(task, fam, model, X, y, folds)
            store[fam] = sv
            mean_abs = np.abs(sv).mean(axis=0)
            ranks[fam] = pd.Series(rankdata(-mean_abs, method="min"), index=cols).astype(int)
            beeswarm(ax, sv, X, cols, TITLES.get(fam, fam), units, rng)

        sm = plt.cm.ScalarMappable(cmap=CMAP)
        cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
        cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
        cb.set_label("feature value (percentile)", fontsize=8)
        path = FIG / f"shap_beeswarm_task{task}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        wide = pd.DataFrame({f"{fam}|{c}": store[fam][:, j]
                             for fam in store for j, c in enumerate(cols)})
        wide.insert(0, "id", df.id.values)
        wide.to_parquet(OUT / f"shap_oof_values_task{task}.parquet")
        rk = pd.DataFrame(ranks)
        rk.to_csv(OUT / f"shap_oof_ranks_task{task}.csv")
        print(f"\n  TASK {task} mean |SHAP| ranks (1 = most important):")
        print(rk.to_string())
        print(f"  wrote {path}\n")


if __name__ == "__main__":
    main()
