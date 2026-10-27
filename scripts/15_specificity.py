"""Check how the non-AD dementia pairs (excluded from all tasks) behave.

1. Score them with the Task A (HC vs ProbAD) fold models.
2. Compare their feature means with HC and ProbAD.
3. Rerun Task B with them included.
Qualitative only: 19 pairs from 10 subjects. Writes outputs/specificity_non_ad.csv.

Run:  python scripts/15_specificity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.base import clone

from sdfd import data, features as F, models as M, normalise, pairs as P

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"

CLEAR_NON_AD = {"LEWYBODYVARNT", "DLB", "FTD", "PARKINSONSDISEASEW/DEMENTIA"}
AMBIGUOUS = {"MIXED/VASCULAR+AD", "NCPD", "OTHERCNSDX", "PSEUDODEMENTIA/DEPRESSION"}


def main():
    sess = data.load_sessions(RAW / "ucsd_fluency_for_snafu_20180518.xlsx")
    sess = normalise.merge_into_sessions(
        sess, normalise.load_normalised(RAW, RAW / "ucsd_fluency.csv"))
    pr = P.build_pairs(sess)
    allf = F.build_structural(pr, sess)

    # non-AD pairs are not in any task table, so compute their semantic features here
    from sdfd.vocab import CachedEmbedder
    caches = sorted(OUT.glob("emb_*.npz"))
    pref = [c for c in caches if "word2vec" in c.name] or caches
    emb = CachedEmbedder(pref[0])
    print(f"  (semantic features computed with {pref[0].stem.replace('emb_', '')})")
    allf = F.add_semantic_features(allf, emb, tau=0.0)

    other = allf[allf.dx == "Other"].copy()
    other["subgroup"] = np.where(other.dx_raw.isin(CLEAR_NON_AD),
                                 "clear non-AD", "ambiguous")

    print("=" * 74)
    print("THE NON-AD GROUP")
    print("=" * 74)
    print(f"  {len(other)} visit pairs from {other.id.nunique()} subjects, "
          f"{other.dx_raw.nunique()} distinct labels")
    print("\n  by label:")
    for lbl, n in other.dx_raw.value_counts().items():
        grp = "clear non-AD" if lbl in CLEAR_NON_AD else "ambiguous"
        print(f"     {lbl:<32} {n:>2}   ({grp})")
    print(f"\n  MMSE: mean {other.MMSE.mean():.1f} "
          f"(HC {allf[allf.dx=='HC'].MMSE.mean():.1f}, "
          f"ProbAD {allf[allf.dx=='ProbAD'].MMSE.mean():.1f})")

    # ---------------------------------------------------------- test 1
    print("\n" + "=" * 74)
    print("1. WHERE DOES AN AD MODEL PUT THEM?")
    print("=" * 74)
    taskA = pd.read_parquet(OUT / "features_taskA.parquet")
    folds = P.load_splits(OUT / "splits_taskA.json")
    X, y, groups, cols = F.feature_matrix(taskA)
    Xo_df = other.assign(y=1)
    Xo = Xo_df[cols].astype(float)
    # fill any residual gaps from the TRAINING distribution, not from 19 rows
    Xo = Xo.fillna(pd.DataFrame(X, columns=cols).median()).to_numpy()

    print("  P(ProbAD) assigned to each group, averaged over the 5 fold models:\n")
    for fam_name, model in M.model_families().items():
        probs = []
        for f in folds:
            m = clone(model)
            tr = np.array(f["train_idx"])
            if m.__class__.__name__ == "XGBClassifier":
                pos = max((y[tr] == 1).sum(), 1)
                m.set_params(scale_pos_weight=float((y[tr] == 0).sum() / pos))
            m.fit(X[tr], y[tr])
            probs.append(m.predict_proba(Xo)[:, 1])
        p = np.mean(probs, axis=0)
        other[f"p_{fam_name}"] = p

        hc_p = np.mean([clone(model).fit(X[np.array(f["train_idx"])],
                                         y[np.array(f["train_idx"])])
                        .predict_proba(X[np.array(f["test_idx"])])[:, 1]
                        [y[np.array(f["test_idx"])] == 0].mean() for f in folds])
        ad_p = np.mean([clone(model).fit(X[np.array(f["train_idx"])],
                                         y[np.array(f["train_idx"])])
                        .predict_proba(X[np.array(f["test_idx"])])[:, 1]
                        [y[np.array(f["test_idx"])] == 1].mean() for f in folds])
        print(f"  {fam_name}")
        print(f"     held-out HC      {hc_p:.3f}")
        print(f"     held-out ProbAD  {ad_p:.3f}")
        for sub in ["clear non-AD", "ambiguous"]:
            v = p[(other.subgroup == sub).to_numpy()]
            if len(v):
                print(f"     {sub:<16} {v.mean():.3f}   "
                      f"(n={len(v)}, {int((v > 0.5).sum())} classified ProbAD)")
        print()

    pcols = [c for c in other.columns if c.startswith("p_")]
    other["p_mean"] = other[pcols].mean(axis=1)
    print("  per pair, averaged across families:")
    print(other[["id", "dx_raw", "subgroup", "MMSE", "p_mean"]]
          .sort_values("p_mean", ascending=False).round(3).to_string(index=False))

    # ---------------------------------------------------------- test 2
    print("\n" + "=" * 74)
    print("2. DO THEIR FEATURES LOOK LIKE ProbAD?")
    print("=" * 74)
    feats = [c for c in F.MODEL_FEATURES if c in allf.columns]
    ref = allf[allf.dx.isin(["HC", "ProbAD"])].groupby("dx")[feats].mean().T
    ref["non_AD"] = other[feats].mean()
    ref["closer_to"] = np.where(
        (ref["non_AD"] - ref["HC"]).abs() < (ref["non_AD"] - ref["ProbAD"]).abs(),
        "HC", "ProbAD")
    print(ref.round(3).to_string())
    n_ad = int((ref.closer_to == "ProbAD").sum())
    print(f"\n  features closer to ProbAD: {n_ad}/{len(ref)}")

    # ---------------------------------------------------------- test 3
    print("\n" + "=" * 74)
    print("3. DOES INCLUDING THEM CHANGE TASK B?")
    print("=" * 74)
    for task, score in [("B1", "MMSE"), ("B2", "DRS")]:
        base = pd.read_parquet(OUT / f"features_task{task}.parquet")
        fl = P.load_splits(OUT / f"splits_task{task}.json")
        Xb, yb, gb, cb = F.feature_matrix(base)
        reg = M.regression_families()["random_forest"]
        sc, _ = M.cross_validate_reg(reg, Xb, yb, gb, fl)

        extra = other[other[score].notna()].copy()
        extra["y"] = extra[score]
        combo = pd.concat([base, extra[base.columns.intersection(extra.columns)]],
                          ignore_index=True)
        Xc, yc, gc, _ = F.feature_matrix(combo, cb)
        fc = P.make_splits(combo, n_splits=5, seed=42)
        sc2, _ = M.cross_validate_reg(reg, Xc, yc, gc, fc)

        print(f"  {task} ({score}), random forest")
        print(f"     excluding non-AD (current) : R2 {sc.r2.mean():+.3f}   "
              f"n={len(base)}")
        print(f"     including non-AD           : R2 {sc2.r2.mean():+.3f}   "
              f"n={len(combo)}")
        print(f"     difference                 : {sc2.r2.mean()-sc.r2.mean():+.4f}")

    other.to_csv(OUT / "specificity_non_ad.csv", index=False)
    print(f"\nwrote specificity_non_ad.csv")
    print("\n  NOTE: n=19 pairs from 10 subjects across 6 conditions. Qualitative")
    print("  check only -- state the sample size whenever this is reported.")


if __name__ == "__main__":
    main()
