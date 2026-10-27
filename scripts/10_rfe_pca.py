"""RFE on the features, then PCA with SHAP / scrambling / RFE on the components.

Variance rank is not predictive rank, so the components are ranked by how well
they predict, and the recurrence metrics' loadings are read off the most
predictive ones. Writes outputs/pca_loadings_task<task>.csv and
pca_importance_task<task>.csv.

Run:  python scripts/10_rfe_pca.py [task]     task in {A, B1, B2}, default A
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

from sdfd import features as F, models as M, pairs as P

OUT = ROOT / "outputs"
TASK = sys.argv[1] if len(sys.argv) > 1 else "A"
IS_REG = TASK != "A"
METRICS = ["dfd", "sdfd", "sdfd_1to1"]


def main():
    df = pd.read_parquet(OUT / f"features_task{TASK}.parquet")
    folds = P.load_splits(OUT / f"splits_task{TASK}.json")
    X, y, groups, cols = F.feature_matrix(df)
    score = "r2" if IS_REG else "roc_auc"

    print("=" * 76)
    print(f"TASK {TASK}: RFE AND PCA   ({len(df)} pairs, {len(cols)} features)")
    print("=" * 76)

    # ---------------------------------------------------------------- RFE
    print("\n" + "=" * 76)
    print("RECURSIVE FEATURE ELIMINATION")
    print("=" * 76)
    print("  drop the weakest feature, refit, repeat; CV picks the subset size\n")

    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=5)
    estimators = (
        {"ridge": Ridge(alpha=1.0),
         "random_forest": RandomForestRegressor(
             n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)}
        if IS_REG else
        {"logistic": LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=42),
         "random_forest": RandomForestClassifier(
             n_estimators=200, min_samples_leaf=2, class_weight="balanced",
             random_state=42, n_jobs=-1)}
    )

    Xs = StandardScaler().fit_transform(X)
    for name, est in estimators.items():
        sel = RFECV(est, step=1, cv=cv, scoring=score, min_features_to_select=1,
                    n_jobs=-1)
        sel.fit(Xs, y, groups=groups)
        kept = [c for c, k in zip(cols, sel.support_) if k]
        ranks = dict(zip(cols, sel.ranking_))
        print(f"  {name}: optimal subset = {sel.n_features_} features")
        print(f"     kept    : {', '.join(kept)}")
        dropped = [c for c in cols if c not in kept]
        print(f"     dropped : {', '.join(dropped) if dropped else '(none)'}")
        for m in METRICS:
            if m in ranks:
                mark = "KEPT" if m in kept else f"dropped (rank {ranks[m]})"
                print(f"     {m:<12} {mark}")
        curve = sel.cv_results_["mean_test_score"]
        print(f"     score by subset size: "
              f"{np.round(curve, 3)}")
        print()

    # ---------------------------------------------------------------- PCA
    print("=" * 76)
    print("PCA, THEN INTERPRETABILITY ON THE COMPONENTS")
    print("=" * 76)

    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=len(cols), random_state=42).fit(scaler.transform(X))
    Z = pca.transform(scaler.transform(X))
    ev = pca.explained_variance_ratio_
    comp_names = [f"PC{i+1}" for i in range(len(cols))]

    print("\n  variance explained:")
    print("     " + "  ".join(f"{n}:{v:.1%}" for n, v in zip(comp_names, ev)))
    print(f"     cumulative to 90%: {int(np.searchsorted(np.cumsum(ev), 0.90) + 1)}"
          f" components")

    fams = M.regression_families() if IS_REG else M.model_families()

    # interrogate the components under every family
    scr_all, shap_all, perf = {}, {}, {}
    for fam_name, model in fams.items():
        if IS_REG:
            sc, _ = M.cross_validate_reg(model, Z, y, groups, folds)
            perf[fam_name] = sc.r2.mean()
            scr = M.scramble_importance_reg(model, Z, y, groups, folds, comp_names)
            sh = M.shap_importance_reg(model, Z, y, folds, comp_names)
        else:
            sc, _, _ = M.cross_validate(model, Z, y, groups, folds)
            perf[fam_name] = sc.auc.mean()
            scr = M.scramble_importance(model, Z, y, groups, folds, comp_names)
            sh = M.shap_importance(model, Z, y, folds, comp_names)
        scr_all[fam_name] = scr["mean"]
        if sh is not None:
            shap_all[fam_name] = sh["mean_abs_shap"]

    unit = "R2" if IS_REG else "AUC"
    print(f"\n  models trained on all {len(cols)} components:")
    for k, v in perf.items():
        print(f"     {k:<16} {unit} {v:.3f}")

    print("\n  IS VARIANCE THE SAME AS PREDICTIVENESS?")
    scr_tbl = pd.DataFrame(scr_all).loc[comp_names]
    scr_rank = scr_tbl.rank(ascending=False)
    tbl = pd.DataFrame({"var_explained": ev,
                        "var_rank": pd.Series(ev, index=comp_names)
                                      .rank(ascending=False).values},
                       index=comp_names)
    for fam in scr_rank.columns:
        tbl[f"scr_{fam[:8]}"] = scr_rank[fam].astype(int)
    if shap_all:
        sh_rank = pd.DataFrame(shap_all).loc[comp_names].rank(ascending=False)
        for fam in sh_rank.columns:
            tbl[f"shap_{fam[:8]}"] = sh_rank[fam].astype(int)
    print(tbl.round(4).to_string())

    rank_cols = [c for c in tbl.columns if c.startswith(("scr_", "shap_"))]
    tbl["pred_rank_mean"] = tbl[rank_cols].mean(axis=1)
    rho = tbl["var_rank"].corr(tbl["pred_rank_mean"], method="spearman")
    print(f"\n  Spearman(variance rank, mean predictive rank) = {rho:+.3f}")
    print("  (well below 1 => high-variance components are NOT the predictive")
    print("   ones, which is exactly the point about PCA being unsupervised)")

    # ---- RFE on the components ----
    print("\n  RFE ON THE COMPONENTS")
    for name, est in estimators.items():
        sel = RFECV(est, step=1, cv=cv, scoring=score,
                    min_features_to_select=1, n_jobs=-1)
        sel.fit(Z, y, groups=groups)
        kept = [c for c, k in zip(comp_names, sel.support_) if k]
        print(f"     {name}: keeps {sel.n_features_} components -> "
              f"{', '.join(kept)}")
        dropped = [c for c in comp_names if c not in kept]
        print(f"        dropped: {', '.join(dropped) if dropped else '(none)'}")

    top = tbl["pred_rank_mean"].sort_values().head(3).index.tolist()
    print(f"\n  most PREDICTIVE components (mean rank across families): {top}")
    load = pd.DataFrame(pca.components_.T, index=cols, columns=comp_names)
    print("\n  loadings of the recurrence metrics there:")
    print(load.loc[[c for c in METRICS if c in cols], top].round(3).to_string())
    print("\n  strongest loading in each:")
    for c in top:
        srt = load[c].abs().sort_values(ascending=False)
        print(f"     {c} ({ev[comp_names.index(c)]:.1%} of variance): "
              + ", ".join(f"{i} ({load.loc[i, c]:+.2f})" for i in srt.head(4).index))

    load.to_csv(OUT / f"pca_loadings_task{TASK}.csv")
    tbl.to_csv(OUT / f"pca_importance_task{TASK}.csv")
    print(f"\nwrote pca_loadings_task{TASK}.csv, pca_importance_task{TASK}.csv")


if __name__ == "__main__":
    main()
