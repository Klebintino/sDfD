"""Compare dfd, sdfd and sdfd_1to1 in isolation, in every model family.

The three metrics are highly correlated, so they are compared one at a time:
  ALONE   each metric (and list_length) as the only feature
  ADDED   structural baseline (no recurrence metric) + one metric
Writes outputs/ablation_alone_task<task>.csv and ablation_added_task<task>.csv.

Run:  python scripts/08_ablation.py [task]     task in {A, B1, B2}, default A
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from sdfd import features as F, models as M, pairs as P

OUT = ROOT / "outputs"
TASK = sys.argv[1] if len(sys.argv) > 1 else "A"
METRICS = ["dfd", "sdfd", "sdfd_1to1"]


def main():
    df = pd.read_parquet(OUT / f"features_task{TASK}.parquet")
    folds = P.load_splits(OUT / f"splits_task{TASK}.json")
    _, y, groups, cols = F.feature_matrix(df)

    # Task A is classification (AUC); B1/B2 are regression (R^2)
    IS_REG = TASK != "A"
    fams = M.regression_families() if IS_REG else M.model_families()
    SCORE = "R2" if IS_REG else "AUC"

    metrics = [m for m in METRICS if m in cols]
    baseline_cols = [c for c in cols
                     if c in F.STRUCTURAL_FEATURES and c not in METRICS]

    print("=" * 78)
    print(f"TASK {TASK}: ISOLATED COMPARISON, ALL MODEL FAMILIES")
    print("=" * 78)
    print(f"  baseline ({len(baseline_cols)} features): {', '.join(baseline_cols)}")

    def score_of(model, feats):
        X, _, _, _ = F.feature_matrix(df, feats)
        if IS_REG:
            sc, _ = M.cross_validate_reg(model, X, y, groups, folds)
            return sc.r2.values
        sc, _, _ = M.cross_validate(model, X, y, groups, folds)
        return sc.auc.values

    alone_tbl, added_tbl = {}, {}

    for fam_name, model in fams.items():
        print(f"\n{'-' * 78}\n{fam_name}\n{'-' * 78}")

        print("  ALONE (single feature)")
        alone = {}
        for m in metrics + ["list_length"]:
            a = score_of(model, [m])
            alone[m] = a
            tag = "   <- best structural feature" if m == "list_length" else ""
            print(f"     {m:<14} {SCORE} {a.mean():+.3f}   "
                  f"folds {np.round(a, 3)}{tag}")
        alone_tbl[fam_name] = {k: v.mean() for k, v in alone.items()}

        for a, b in [("sdfd", "dfd"), ("sdfd_1to1", "dfd"), ("sdfd_1to1", "sdfd")]:
            if a in alone and b in alone:
                d = alone[a] - alone[b]
                p = stats.ttest_rel(alone[a], alone[b]).pvalue
                print(f"       {a} - {b}: {d.mean():+.3f}, "
                      f"wins {int((d > 0).sum())}/5, p={p:.3f}")

        print("  ADDED (baseline + one metric)")
        base = score_of(model, baseline_cols)
        print(f"     {'baseline':<14} {SCORE} {base.mean():+.3f}")
        added = {"baseline": base.mean()}
        for m in metrics:
            a = score_of(model, baseline_cols + [m])
            d = a - base
            p = stats.ttest_rel(a, base).pvalue
            added[m] = a.mean()
            print(f"     {'+ ' + m:<14} {SCORE} {a.mean():+.3f}   "
                  f"delta {d.mean():+.4f}, wins {int((d > 0).sum())}/5, p={p:.3f}")
        added_tbl[fam_name] = added

    print(f"\n{'=' * 78}")
    print(f"SUMMARY: {SCORE} of each metric ALONE, by family")
    print("=" * 78)
    t = pd.DataFrame(alone_tbl)
    print(t.round(3).to_string())
    print("\n  best metric per family:")
    for fam in t.columns:
        sub = t.loc[metrics, fam]
        print(f"     {fam:<15} {sub.idxmax()} ({sub.max():.3f})")

    print(f"\n{'=' * 78}")
    print(f"SUMMARY: {SCORE} gain over baseline, by family")
    print("=" * 78)
    g = pd.DataFrame(added_tbl)
    gain = g.loc[metrics] - g.loc["baseline"]
    print(gain.round(4).to_string())

    t.to_csv(OUT / f"ablation_alone_task{TASK}.csv")
    g.to_csv(OUT / f"ablation_added_task{TASK}.csv")
    print(f"\nwrote ablation_alone_task{TASK}.csv, ablation_added_task{TASK}.csv")


if __name__ == "__main__":
    main()
