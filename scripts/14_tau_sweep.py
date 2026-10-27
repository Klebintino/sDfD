"""Sweep the similarity threshold tau and the weighting function sigma.

Scores sdfd and sdfd_1to1 as single features (logistic for A, ridge for B) at
each setting, computed from the embedding cache, and reports the share of
surviving matches that are exact (sDfD approaches DfD as tau rises).

sigma variants:
    raw    sigma = cosine, no rescaling            (default)
    gate   sigma = cosine if > tau else 0          range [tau, 1]
    hinge  sigma = max(0, s - tau) / (1 - tau)     range [0, 1]

Writes outputs/tau_sweep_task<task>_<emb_tag>.csv.

Run:  python scripts/14_tau_sweep.py [task] [emb_tag]
      defaults: task A, emb word2vec_google_news_300
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sdfd import features as F, metrics, models as M, pairs as P
from sdfd.vocab import CachedEmbedder

OUT = ROOT / "outputs"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
TASK = args[0] if args else "A"
TAG = args[1] if len(args) > 1 else "word2vec_google_news_300"
IS_REG = TASK != "A"

TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
SIGMAS = {"raw": metrics.sigma_raw,
          "gate": metrics.sigma_gate,
          "hinge": metrics.sigma_hinge}


def exact_share(df, emb, tau, one_to_one):
    """Fraction of surviving matches that are identical words."""
    ex, tot = 0, 0
    for r in df.itertuples():
        l1, l2 = list(r.L1), list(r.L2)
        if not l1 or not l2:
            continue
        sim = np.nan_to_num(emb.similarity_matrix(l1, l2), nan=0.0)
        if one_to_one:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-sim)
        else:
            cols, rows = np.argmax(sim, axis=1), np.arange(len(l1))
        keep = sim[rows, cols] > tau
        for i, j, k in zip(rows, cols, keep):
            if k:
                tot += 1
                ex += int(l1[i] == l2[j])
    return ex / tot if tot else np.nan


def main():
    emb = CachedEmbedder(OUT / f"emb_{TAG}.npz")
    df = pd.read_parquet(OUT / f"features_task{TASK}.parquet")
    folds = P.load_splits(OUT / f"splits_task{TASK}.json")
    _, y, groups, _ = F.feature_matrix(df)

    fams = M.regression_families() if IS_REG else M.model_families()
    model = fams["ridge" if IS_REG else "logistic"]
    unit = "R2" if IS_REG else "AUC"

    def score(vals):
        X = np.asarray(vals, dtype=float).reshape(-1, 1)
        X = np.nan_to_num(X, nan=float(np.nanmedian(X)))
        if IS_REG:
            sc, _ = M.cross_validate_reg(model, X, y, groups, folds)
            return sc.r2.values
        sc, _, _ = M.cross_validate(model, X, y, groups, folds)
        return sc.auc.values

    print("=" * 78)
    print(f"TASK {TASK} / {TAG}: tau and sigma sweep")
    print(f"  single-feature {unit}, {len(df)} pairs, frozen folds, "
          f"{'ridge' if IS_REG else 'logistic'}")
    print("=" * 78)

    # reference points
    dfd_vals = [metrics.dfd(list(r.L1), list(r.L2)) for r in df.itertuples()]
    dfd_s = score(dfd_vals)
    print(f"\n  reference: plain DfD {unit} {dfd_s.mean():+.3f}")

    rows = []
    for name, o2o in [("sdfd", False), ("sdfd_1to1", True)]:
        fn = metrics.sdfd_assignment if o2o else metrics.sdfd
        print(f"\n{'-' * 78}\n{name}\n{'-' * 78}")
        print(f"  {'sigma':<7}{'tau':>6}{'':3}{unit:>8}{'':4}"
              f"{'exact% of matches':>19}   note")
        for sig_name, sig_fn in SIGMAS.items():
            for tau in TAUS:
                # raw and gate are the same function; run raw only at tau=0
                if sig_name == "raw" and tau > 0:
                    continue
                if sig_name == "hinge" and tau >= 0.9:
                    continue
                vals = [fn(list(r.L1), list(r.L2), emb, tau=tau, sigma_fn=sig_fn)
                        for r in df.itertuples()]
                s = score(vals)
                ex = exact_share(df, emb, tau, o2o)
                note = ""
                if ex > 0.9:
                    note = "<- almost pure DfD"
                elif ex > 0.8:
                    note = "<- mostly exact matches"
                rows.append({"metric": name, "sigma": sig_name, "tau": tau,
                             "score": s.mean(), "score_sd": s.std(),
                             "exact_frac": ex})
                print(f"  {sig_name:<7}{tau:>6.2f}{'':3}{s.mean():>+8.3f}{'':4}"
                      f"{ex:>18.1%}   {note}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"tau_sweep_task{TASK}_{TAG}.csv", index=False)

    print(f"\n{'=' * 78}\nBEST CONFIGURATIONS\n{'=' * 78}")
    for name in ["sdfd", "sdfd_1to1"]:
        sub = res[res.metric == name]
        best = sub.loc[sub.score.idxmax()]
        plain = sub[(sub.sigma == "raw") & (sub.tau == 0)].iloc[0]
        print(f"\n  {name}")
        print(f"     plain (raw, tau=0) : {unit} {plain.score:+.3f}   "
              f"exact {plain.exact_frac:.1%}")
        print(f"     best in sweep      : {unit} {best.score:+.3f}   "
              f"sigma={best.sigma}, tau={best.tau:.2f}, "
              f"exact {best.exact_frac:.1%}")
        print(f"     gain over plain    : {best.score - plain.score:+.4f}")
        if best.exact_frac > 0.85:
            print("     WARNING: the best setting keeps almost only exact matches,")
            print("     so it is approaching DfD rather than tuning sDfD.")

    print("\n  how flat is the tau curve? (threshold sensitivity)")
    for name in ["sdfd", "sdfd_1to1"]:
        for sig in ["gate", "hinge"]:
            sub = res[(res.metric == name) & (res.sigma == sig)]
            if len(sub) > 1:
                print(f"     {name:<10} {sig:<6} range "
                      f"{sub.score.min():+.3f} to {sub.score.max():+.3f}   "
                      f"spread {sub.score.max() - sub.score.min():.3f}")

    print(f"\nwrote tau_sweep_task{TASK}_{TAG}.csv")


if __name__ == "__main__":
    main()
