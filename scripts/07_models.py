"""Task A (HC vs ProbAD): train the four classifiers and rank features with
SHAP and scrambling (including joint dfd/sdfd scrambling) in each family.

Writes outputs/model_results_taskA.csv, ranks_scrambling_taskA.csv, ranks_shap_taskA.csv.
Use 09_regression.py for B1/B2.

Run:  python scripts/07_models.py A [--fast]
      --fast skips the slow KernelExplainer SHAP for non-tree families
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
TASK = args[0] if args else "A"
FAST = "--fast" in sys.argv
if TASK != "A":
    sys.exit(f"07_models.py is Task A only (classifiers, AUC); got task '{TASK}'. "
             f"For B1/B2 run: python scripts/09_regression.py {TASK}")

TREE = ("random_forest", "xgboost")


def rank_of(series, name):
    order = list(series.sort_values(ascending=False).index)
    return order.index(name) + 1 if name in order else None


def main():
    df = pd.read_parquet(OUT / f"features_task{TASK}.parquet")
    folds = P.load_splits(OUT / f"splits_task{TASK}.json")
    X, y, groups, cols = F.feature_matrix(df)
    fams = M.model_families()

    print("=" * 76)
    print(f"TASK {TASK}   {len(df)} pairs, {df.id.nunique()} subjects, "
          f"{len(cols)} features")
    print(f"  class balance: {(y == 0).sum()} / {(y == 1).sum()} "
          f"({100 * y.mean():.1f}% positive)")
    base = M.baseline_scores(X, y, groups, folds)
    print(f"  majority-class baseline: accuracy {base.accuracy.mean():.3f}, "
          f"AUC {base.auc.mean():.3f}")
    print("=" * 76)

    print("\nMODEL PERFORMANCE  (mean over 5 subject-level folds)")
    perf = {}
    for name, model in fams.items():
        sc, _, _ = M.cross_validate(model, X, y, groups, folds)
        perf[name] = sc.mean()
        print(f"  {name:<15} AUC {sc.auc.mean():.3f}+-{sc.auc.std():.3f}   "
              f"F1 {sc.f1_macro.mean():.3f}   kappa {sc.kappa.mean():.3f}   "
              f"recall_AD {sc.recall_ad.mean():.3f}")
    pd.DataFrame(perf).T.to_csv(OUT / f"model_results_task{TASK}.csv")

    joint = [g for g in [["dfd", "sdfd"], ["dfd", "sdfd", "sdfd_1to1"]]
             if all(c in cols for c in g)]

    scr_all, shap_all = {}, {}
    for name, model in fams.items():
        print(f"\n{'=' * 76}\nINTERROGATION: {name}\n{'=' * 76}")

        scr = M.scramble_importance(model, X, y, groups, folds, cols,
                                    joint_groups=joint)
        scr_all[name] = scr["mean"]
        print("  scrambling (AUC drop), top 6:")
        for f, r in scr.head(6).iterrows():
            print(f"     {f:<22} {r['mean']:+.4f} +- {r['std']:.4f}")

        if FAST and name not in TREE:
            print("  SHAP: skipped (--fast)")
            continue
        sh = M.shap_importance(model, X, y, folds, cols)
        if sh is None:
            print("  SHAP: shap not installed")
            continue
        shap_all[name] = sh["mean_abs_shap"]
        print("  SHAP (mean |SHAP|), top 6:")
        for f, r in sh.head(6).iterrows():
            print(f"     {f:<22} {r['mean_abs_shap']:.4f}")

    print(f"\n{'=' * 76}")
    print("CONSISTENCY ACROSS MODEL FAMILIES  (rank, 1 = most important)")
    print("=" * 76)

    for label, store in [("SCRAMBLING", scr_all), ("SHAP", shap_all)]:
        if not store:
            continue
        tbl = pd.DataFrame(store)
        ranks = tbl.rank(ascending=False)
        ranks = ranks.loc[[c for c in cols if c in ranks.index]]
        print(f"\n  {label}")
        print("    " + f"{'feature':<22}" +
              "".join(f"{c[:11]:>13}" for c in ranks.columns))
        for f in ranks.index:
            row = "".join(f"{int(ranks.loc[f, c]):>13}" for c in ranks.columns)
            star = "  <--" if f.startswith("sdfd") or f == "dfd" else ""
            print(f"    {f:<22}{row}{star}")
        ranks.to_csv(OUT / f"ranks_{label.lower()}_task{TASK}.csv")

    half = len(cols) // 2
    print(f"\n  success criterion: ranks in the top half (<= {half}) "
          f"on BOTH methods")
    for m in ["dfd", "sdfd", "sdfd_1to1"]:
        if m not in cols:
            continue
        bits = []
        for label, store in [("scr", scr_all), ("shap", shap_all)]:
            for fam, s in store.items():
                r = rank_of(s[[c for c in cols if c in s.index]], m)
                if r:
                    bits.append(f"{fam[:4]}/{label} {r}")
        print(f"    {m:<12} {', '.join(bits)}")

    print(f"\nwrote model_results_task{TASK}.csv, ranks_*.csv")


if __name__ == "__main__":
    main()
