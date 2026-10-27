"""Significance tests for the metric comparisons, on pooled out-of-fold predictions.

Each comparison (see COMPARISONS) uses a subject-level paired permutation test
(primary), a subject-level bootstrap CI and, for AUC, DeLong's test. Permutation
p-values are Holm-corrected across the 12 task x family cells. Scores are AUC
for Task A and R^2 for B1/B2.

Run:  python scripts/16_significance.py [--n-perm 5000] [--n-boot 2000] [--tasks A,B1,B2]
Writes outputs/significance_scores.csv     (every configuration, score + 95% CI)
       outputs/significance_tests.csv      (every comparison)
       outputs/oof_predictions_task*.parquet  (pooled OOF predictions, for reuse)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sdfd import features as F, models as M, pairs as P, stats as S

OUT = ROOT / "outputs"
METRICS = ["dfd", "sdfd", "sdfd_1to1"]
BASELINE = ["list_length", "length_diff", "n_repeats", "n_intrusions",
            "cluster_size", "n_switches"]

CONFIGS = {
    # name: feature list
    "dfd": ["dfd"],
    "sdfd": ["sdfd"],
    "sdfd_1to1": ["sdfd_1to1"],
    "list_length": ["list_length"],
    "baseline": BASELINE,
    "baseline+dfd": BASELINE + ["dfd"],
    "baseline+sdfd": BASELINE + ["sdfd"],
    "baseline+sdfd_1to1": BASELINE + ["sdfd_1to1"],
    "full": F.MODEL_FEATURES,
}

COMPARISONS = [
    # (block, a, b)  -- tests score(a) - score(b)
    ("alone", "sdfd_1to1", "dfd"),
    ("alone", "sdfd", "dfd"),
    ("alone", "sdfd_1to1", "sdfd"),
    ("alone", "list_length", "sdfd_1to1"),
    ("alone", "list_length", "dfd"),
    ("added", "baseline+dfd", "baseline"),
    ("added", "baseline+sdfd", "baseline"),
    ("added", "baseline+sdfd_1to1", "baseline"),
]


def oof_predictions(task: str):
    df = pd.read_parquet(OUT / f"features_task{task}.parquet")
    folds = P.load_splits(OUT / f"splits_task{task}.json")
    is_reg = task != "A"
    fams = M.regression_families() if is_reg else M.model_families()
    _, y, groups, _ = F.feature_matrix(df)

    preds, foldmean = {}, {}
    for fam, model in fams.items():
        for cname, cols in CONFIGS.items():
            X, _, _, _ = F.feature_matrix(df, cols)
            if is_reg:
                sc, oof = M.cross_validate_reg(model, X, y, groups, folds)
                foldmean[(fam, cname)] = sc.r2.mean()
            else:
                sc, oof, _ = M.cross_validate(model, X, y, groups, folds)
                foldmean[(fam, cname)] = sc.auc.mean()
            assert not np.isnan(oof).any(), "every pair must get one OOF prediction"
            preds[(fam, cname)] = oof
    return df, y, groups, preds, foldmean, list(fams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tasks", default="A,B1,B2")
    args = ap.parse_args()

    score_rows, test_rows = [], []
    for task in args.tasks.split(","):
        t0 = time.time()
        kind = "r2" if task != "A" else "auc"
        unit = "R2" if kind == "r2" else "AUC"
        df, y, groups, preds, foldmean, fams = oof_predictions(task)
        print("=" * 78)
        print(f"TASK {task}  ({len(df)} pairs, {len(np.unique(groups))} subjects)  "
              f"score = pooled out-of-fold {unit}")
        print("=" * 78)

        # save OOF predictions for reuse (figures, later checks)
        oof_df = df[["id", "listnum_1", "listnum_2", "y"]].copy()
        for (fam, cname), p in preds.items():
            oof_df[f"{fam}|{cname}"] = p
        oof_df.to_parquet(OUT / f"oof_predictions_task{task}.parquet")

        for fam in fams:
            fam_preds = {c: preds[(fam, c)] for c in CONFIGS}
            boot = S.cluster_bootstrap(y, fam_preds, groups, kind=kind,
                                       n_boot=args.n_boot, seed=42)
            for c in CONFIGS:
                lo, hi = S.ci(boot[c])
                score_rows.append({
                    "task": task, "family": fam, "config": c, "metric": unit,
                    "score_pooled": S.score_fn(kind)(y, fam_preds[c]),
                    "ci_lo": lo, "ci_hi": hi,
                    "score_foldmean": foldmean[(fam, c)],
                })
            for block, a, b in COMPARISONS:
                pa, pb = fam_preds[a], fam_preds[b]
                perm = S.paired_permutation(y, pa, pb, groups, kind=kind,
                                            n_perm=args.n_perm, seed=42)
                dlo, dhi = S.ci(boot[a] - boot[b])
                row = {"task": task, "family": fam, "block": block,
                       "a": a, "b": b, "metric": unit,
                       "score_a": S.score_fn(kind)(y, pa),
                       "score_b": S.score_fn(kind)(y, pb),
                       "diff": perm["diff"], "boot_ci_lo": dlo, "boot_ci_hi": dhi,
                       "perm_p": perm["p"],
                       "foldmean_diff": foldmean[(fam, a)] - foldmean[(fam, b)]}
                if kind == "auc":
                    d = S.delong_test(y, pa, pb)
                    row.update({"delong_z": d["z"], "delong_p": d["p"]})
                test_rows.append(row)
        print(f"  done in {time.time() - t0:.0f}s")

    scores = pd.DataFrame(score_rows)
    tests = pd.DataFrame(test_rows)
    # Holm across the 12 task x family cells of each comparison
    tests["perm_p_holm"] = np.nan
    for (a, b), idx in tests.groupby(["a", "b"]).groups.items():
        tests.loc[idx, "perm_p_holm"] = S.holm(tests.loc[idx, "perm_p"].values)
    if "delong_p" in tests:
        tests["delong_p_holm"] = np.nan
        for (a, b), idx in tests[tests.metric == "AUC"].groupby(["a", "b"]).groups.items():
            tests.loc[idx, "delong_p_holm"] = S.holm(tests.loc[idx, "delong_p"].values)

    scores.to_csv(OUT / "significance_scores.csv", index=False)
    tests.to_csv(OUT / "significance_tests.csv", index=False)

    # ---------------- readable summary ----------------
    def star(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    for block, a, b in COMPARISONS:
        sub = tests[(tests.a == a) & (tests.b == b)]
        print(f"\n{'-' * 78}\n{block.upper()}: {a}  minus  {b}\n{'-' * 78}")
        print(f"  {'task':<5}{'family':<16}{'diff':>8}  {'95% CI (subject bootstrap)':>28}"
              f"  {'perm p':>8} {'Holm':>7}  {'DeLong p':>9}")
        for r in sub.itertuples():
            dl = f"{r.delong_p:9.4f}" if r.metric == "AUC" else f"{'n/a':>9}"
            print(f"  {r.task:<5}{r.family:<16}{r.diff:+8.3f}  "
                  f"[{r.boot_ci_lo:+.3f}, {r.boot_ci_hi:+.3f}]{'':>12}"
                  f"  {r.perm_p:8.4f} {r.perm_p_holm:7.4f} {star(r.perm_p_holm):>3} {dl}")
        n_sig = int((sub.perm_p_holm < 0.05).sum())
        n_pos = int((sub["diff"] > 0).sum())
        print(f"  -> positive in {n_pos}/{len(sub)} cells; significant after Holm in "
              f"{n_sig}/{len(sub)}")

    print("\nReading guide: 'perm p' is the subject-level paired permutation test (primary).")
    print("'Holm' corrects it across the 12 task x family cells. DeLong treats the pairs")
    print("as independent and is shown because it is the standard; expect it to be smaller.")
    print("\nwrote significance_scores.csv, significance_tests.csv, oof_predictions_task*.parquet")


if __name__ == "__main__":
    main()
