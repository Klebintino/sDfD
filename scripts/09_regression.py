"""Tasks B1 (MMSE) and B2 (DRS): train the four regressors, check predictions by
diagnosis group, and rank features with SHAP and scrambling.

Writes outputs/model_results_task<task>.csv, ranks_scrambling_task<task>.csv,
ranks_shap_task<task>.csv.

Run:  python scripts/09_regression.py [task] [--fast]     task in {B1, B2}
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sdfd import features as F, models as M, pairs as P

OUT = ROOT / "outputs"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
TASK = args[0] if args else "B1"
FAST = "--fast" in sys.argv
SCORE = {"B1": "MMSE", "B2": "DRS"}[TASK]
TREE = ("random_forest", "xgboost")


def main():
    df = pd.read_parquet(OUT / f"features_task{TASK}.parquet")
    folds = P.load_splits(OUT / f"splits_task{TASK}.json")
    X, y, groups, cols = F.feature_matrix(df)
    fams = M.regression_families()

    print("=" * 76)
    print(f"TASK {TASK}: predicting {SCORE}")
    print(f"  {len(df)} pairs, {df.id.nunique()} subjects, {len(cols)} features")
    print(f"  {SCORE}: mean {y.mean():.2f}, sd {y.std():.2f}, "
          f"range {y.min():.0f}-{y.max():.0f}")
    print(f"  diagnosis mix: {dict(df.dx.value_counts())}")
    print("=" * 76)

    base = M.baseline_reg(X, y, groups, folds)
    print(f"\n  MEAN BASELINE (predict the training mean, always)")
    print(f"     RMSE {base.rmse.mean():.3f}   MAE {base.mae.mean():.3f}   "
          f"R2 {base.r2.mean():.3f}")
    print("     every model below must beat this to mean anything")

    print("\nMODEL PERFORMANCE  (mean over 5 subject-level folds)")
    perf, oofs = {}, {}
    for name, model in fams.items():
        sc, oof = M.cross_validate_reg(model, X, y, groups, folds)
        perf[name] = sc.mean()
        oofs[name] = oof
        print(f"  {name:<16} RMSE {sc.rmse.mean():.3f}+-{sc.rmse.std():.3f}   "
              f"MAE {sc.mae.mean():.3f}   R2 {sc.r2.mean():+.3f}   "
              f"r {sc.pearson.mean():.3f}")
    pd.DataFrame(perf).T.to_csv(OUT / f"model_results_task{TASK}.csv")

    best = pd.DataFrame(perf).T.r2.idxmax()
    print(f"\n  best by R2: {best}")

    # ---- graded sensitivity check ----
    print("\n" + "=" * 76)
    print("GRADED SENSITIVITY: predicted score by true diagnosis group")
    print("=" * 76)
    chk = df[["dx"]].copy()
    chk["actual"] = y
    chk["predicted"] = oofs[best]
    g = chk.groupby("dx")[["actual", "predicted"]].agg(["mean", "count"])
    print(g.round(2).to_string())
    # order groups by their actual mean score (AtRisk is below MCI here), then
    # check the predictions follow the same order
    obs = chk.groupby("dx")["actual"].mean().sort_values(ascending=False)
    order = list(obs.index)
    means = [chk.loc[chk.dx == d, "predicted"].mean() for d in order]
    print(f"\n  groups ordered by TRUE mean {SCORE}: {order}")
    print(f"     actual    {[round(v, 2) for v in obs.values]}")
    print(f"     predicted {[round(m, 2) for m in means]}")
    mono = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    print(f"  predictions follow the same ordering? {mono}")
    print("  (graded sensitivity: if MCI/at-risk predictions fall between HC and")
    print("   ProbAD, the model is tracking severity, not a binary distinction)")

    # systematic bias per group -- the ceiling effect shows up here
    print(f"\n  prediction error by group (predicted - actual):")
    for d in order:
        sub = chk[chk.dx == d]
        bias = (sub.predicted - sub.actual).mean()
        flag = "   <- overestimates function" if bias > 1 else ""
        print(f"     {d:<8} n={len(sub):>4}  bias {bias:+.2f}{flag}")

    # ---- interrogation, all families ----
    joint = [g for g in [["dfd", "sdfd"], ["dfd", "sdfd", "sdfd_1to1"]]
             if all(c in cols for c in g)]
    scr_all, shap_all = {}, {}

    for name, model in fams.items():
        print(f"\n{'=' * 76}\nINTERROGATION: {name}\n{'=' * 76}")
        scr = M.scramble_importance_reg(model, X, y, groups, folds, cols,
                                        joint_groups=joint)
        scr_all[name] = scr["mean"]
        print("  scrambling (R2 drop), top 6:")
        for f, r in scr.head(6).iterrows():
            print(f"     {f:<22} {r['mean']:+.4f} +- {r['std']:.4f}")

        if FAST and name not in TREE:
            print("  SHAP: skipped (--fast)")
            continue
        sh = M.shap_importance_reg(model, X, y, folds, cols)
        if sh is not None:
            shap_all[name] = sh["mean_abs_shap"]
            print("  SHAP, top 6:")
            for f, r in sh.head(6).iterrows():
                print(f"     {f:<22} {r['mean_abs_shap']:.4f}")

    print(f"\n{'=' * 76}")
    print("CONSISTENCY ACROSS FAMILIES  (rank, 1 = most important)")
    print("=" * 76)
    for label, store in [("SCRAMBLING", scr_all), ("SHAP", shap_all)]:
        if not store:
            continue
        ranks = pd.DataFrame(store).rank(ascending=False)
        ranks = ranks.loc[[c for c in cols if c in ranks.index]]
        print(f"\n  {label}")
        print("    " + f"{'feature':<22}" +
              "".join(f"{c[:11]:>13}" for c in ranks.columns))
        for f in ranks.index:
            row = "".join(f"{int(ranks.loc[f, c]):>13}" for c in ranks.columns)
            star = "  <--" if f in ("dfd", "sdfd", "sdfd_1to1") else ""
            print(f"    {f:<22}{row}{star}")
        ranks.to_csv(OUT / f"ranks_{label.lower()}_task{TASK}.csv")

    print(f"\nwrote model_results_task{TASK}.csv, ranks_*.csv")


if __name__ == "__main__":
    main()
