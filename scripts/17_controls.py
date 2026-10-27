"""Controls: is sDfD's advantage over DfD semantic, or list length?

A. Shuffled embedding: real Word2Vec vectors randomly reassigned to words, then
   sdfd / sdfd_1to1 recomputed and scored alone against the real version.
B. Length decomposition: correlations with list length; each metric residualised
   on n1, n2, n1^2, n2^2, n1*n2 inside each training fold and scored alone (raw,
   scaled by the length-predicted spread, and model-free); and lengths alone vs
   lengths + metric, with linear and quadratic length baselines.
C. Shuffled labels: the full model refitted on permuted outcomes (should be chance).

Run:  python scripts/17_controls.py [--n-emb 100] [--n-emb-tree 20] [--n-label 50]
                                    [--n-label-tree 20] [--n-boot 2000]
                                    [--tasks A,B1,B2] [--families all|linear]
Writes outputs/controls_shuffled_embedding.csv
       outputs/controls_length_correlations.csv
       outputs/controls_length_residual.csv
       outputs/controls_length_added.csv
       outputs/controls_shuffled_label.csv
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from sdfd import features as F, metrics, models as M, pairs as P, stats as S
from sdfd.vocab import CachedEmbedder

OUT = ROOT / "outputs"
EMB_TAG = "word2vec_google_news_300"
LINEAR = {"logistic", "ridge"}


class ShuffledEmbedder:
    """Real vectors, randomly reassigned to words. Same interface as CachedEmbedder."""

    def __init__(self, base: CachedEmbedder, seed: int):
        perm = np.random.default_rng(seed).permutation(len(base.words))
        self._idx = dict(base._idx)
        self.vectors = base.vectors[perm]
        self.dim = base.dim

    def matrix(self, words):
        out = np.full((len(words), self.dim), np.nan, dtype=np.float32)
        for i, w in enumerate(words):
            j = self._idx.get(w)
            if j is not None:
                out[i] = self.vectors[j]
        return out

    def similarity_matrix(self, l1, l2):
        return self.matrix(l1) @ self.matrix(l2).T


def load(task):
    df = pd.read_parquet(OUT / f"features_task{task}.parquet")
    folds = P.load_splits(OUT / f"splits_task{task}.json")
    _, y, groups, _ = F.feature_matrix(df)
    df["n1"] = df.L1.apply(len).astype(float)
    df["n2"] = df.L2.apply(len).astype(float)
    df["n1_sq"], df["n2_sq"], df["n1_x_n2"] = df.n1 ** 2, df.n2 ** 2, df.n1 * df.n2
    return df, folds, y, groups


LENGTH_LIN = ["n1", "n2"]
LENGTH_QUAD = ["n1", "n2", "n1_sq", "n2_sq", "n1_x_n2"]


def families(task, which):
    fams = M.regression_families() if task != "A" else M.model_families()
    if which == "linear":
        fams = {k: v for k, v in fams.items() if k in LINEAR}
    return fams


def cv_oof(task, model, X, y, groups, folds):
    if task == "A":
        sc, oof, _ = M.cross_validate(model, X, y, groups, folds)
        return sc.auc.mean(), oof
    sc, oof = M.cross_validate_reg(model, X, y, groups, folds)
    return sc.r2.mean(), oof


# --------------------------------------------------------------------------
# A. shuffled-embedding null
# --------------------------------------------------------------------------
TREE = {"random_forest", "xgboost"}


def shuffled_embedding(tasks, n_emb, fam_which, n_emb_tree):
    base = CachedEmbedder(OUT / f"emb_{EMB_TAG}.npz")
    data = {t: load(t) for t in tasks}

    # compute every shuffled metric once per unique visit pair (B1/B2 overlap A)
    keys = {}
    for t, (df, *_rest) in data.items():
        for r in df.itertuples():
            keys[(r.id, r.listnum_1, r.listnum_2)] = (list(r.L1), list(r.L2))
    keylist = list(keys)

    shuffled = {"sdfd": np.empty((n_emb, len(keylist))),
                "sdfd_1to1": np.empty((n_emb, len(keylist)))}
    t0 = time.time()
    for s in range(n_emb):
        emb = ShuffledEmbedder(base, seed=1000 + s)
        for k, key in enumerate(keylist):
            l1, l2 = keys[key]
            shuffled["sdfd"][s, k] = metrics.sdfd(l1, l2, emb)
            shuffled["sdfd_1to1"][s, k] = metrics.sdfd_assignment(l1, l2, emb)
        if s == 0:
            print(f"    (one shuffle of {len(keylist)} pairs took {time.time() - t0:.1f}s)")
    pos = {key: k for k, key in enumerate(keylist)}

    rows = []
    for t in tasks:
        df, folds, y, groups = data[t]
        cols_idx = np.array([pos[(r.id, r.listnum_1, r.listnum_2)] for r in df.itertuples()])
        for fam, model in families(t, fam_which).items():
            for metric in ["sdfd", "sdfd_1to1"]:
                real, _ = cv_oof(t, model, df[[metric]].to_numpy(float), y, groups, folds)
                ref, _ = cv_oof(t, model, df[["dfd"]].to_numpy(float), y, groups, folds)
                n_here = min(n_emb_tree, n_emb) if fam in TREE else n_emb
                null = np.array([
                    cv_oof(t, model, shuffled[metric][s, cols_idx].reshape(-1, 1),
                           y, groups, folds)[0]
                    for s in range(n_here)])
                lo, hi = np.quantile(null, [0.025, 0.975])
                p = (1 + int(np.sum(null >= real))) / (n_here + 1)
                rows.append({"task": t, "family": fam, "metric": metric,
                             "score": "R2" if t != "A" else "AUC",
                             "dfd": ref, "real": real, "shuffled_mean": null.mean(),
                             "shuffled_lo": lo, "shuffled_hi": hi,
                             "gain_over_dfd": real - ref,
                             "gain_from_semantics": real - null.mean(),
                             "p_real_beats_shuffled": p, "n_shuffles": n_here})
                r = rows[-1]
                print(f"  {t:<3}{fam:<15}{metric:<10} dfd {ref:.3f}  real {real:.3f}  "
                      f"shuffled {null.mean():.3f} [{lo:.3f},{hi:.3f}]  "
                      f"semantic gain {r['gain_from_semantics']:+.3f}  p={p:.3f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B. length decomposition
# --------------------------------------------------------------------------
def length_terms(df):
    n1, n2 = df.n1.to_numpy(), df.n2.to_numpy()
    return np.column_stack([n1, n2, n1 ** 2, n2 ** 2, n1 * n2])


def length_analysis(tasks, fam_which, n_boot):
    corr_rows, resid_rows, added_rows = [], [], []
    for t in tasks:
        df, folds, y, groups = load(t)
        kind = "r2" if t != "A" else "auc"
        L = length_terms(df)
        overlap = np.array([len(set(a) & set(b)) for a, b in zip(df.L1, df.L2)], float)

        # B1 descriptive
        for m in ["dfd", "sdfd", "sdfd_1to1"]:
            v = df[m].to_numpy(float)
            corr_rows.append({
                "task": t, "metric": m,
                "r_n1": np.corrcoef(v, df.n1)[0, 1], "r_n2": np.corrcoef(v, df.n2)[0, 1],
                "r_n1xn2": np.corrcoef(v, df.n1 * df.n2)[0, 1],
                "rho_n1xn2": spearmanr(v, df.n1 * df.n2)[0],
                "r_overlap": np.corrcoef(v, overlap)[0, 1],
                "r2_on_length_terms": LinearRegression().fit(L, v).score(L, v)})

        fams = families(t, fam_which)
        for fam, model in fams.items():
            # B2 residual alone (residualise inside each fold)
            preds, preds_s, direct = {}, {}, {}

            def fit_predict(xtr, xte, tr):
                mm = clone(model)
                if mm.__class__.__name__ == "XGBClassifier":
                    pos_ = max((y[tr] == 1).sum(), 1)
                    mm.set_params(scale_pos_weight=float((y[tr] == 0).sum() / pos_))
                mm.fit(xtr, y[tr])
                return mm.predict_proba(xte)[:, 1] if t == "A" else mm.predict(xte)

            for m in ["dfd", "sdfd", "sdfd_1to1"]:
                v = df[m].to_numpy(float)
                oof = np.full(len(y), np.nan)
                oof_s = np.full(len(y), np.nan)
                res_oof = np.full(len(y), np.nan)
                for f in folds:
                    tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
                    lr = LinearRegression().fit(L[tr], v[tr])
                    rtr = v[tr] - lr.predict(L[tr])
                    rte = v[te] - lr.predict(L[te])
                    res_oof[te] = rte
                    oof[te] = fit_predict(rtr.reshape(-1, 1), rte.reshape(-1, 1), tr)
                    # the residual's spread grows with length, so divide by the
                    # length-predicted spread (fitted on the training fold)
                    sd = LinearRegression().fit(L[tr], np.abs(rtr))
                    floor = 0.05 * np.mean(np.abs(rtr))
                    str_ = rtr / np.clip(sd.predict(L[tr]), floor, None)
                    ste = rte / np.clip(sd.predict(L[te]), floor, None)
                    oof_s[te] = fit_predict(str_.reshape(-1, 1), ste.reshape(-1, 1), tr)
                preds[m] = oof
                preds_s[m] = oof_s
                # model-free score with the direction fixed in advance (lower DfD =
                # more impaired): -residual for Task A, the residual for B1/B2
                direct[m] = -res_oof if t == "A" else res_oof
            boot = S.cluster_bootstrap(y, preds, groups, kind=kind, n_boot=n_boot)
            boot_s = S.cluster_bootstrap(y, preds_s, groups, kind=kind, n_boot=n_boot)
            dkind = "auc" if t == "A" else "pearson"
            for m, p in preds.items():
                lo, hi = S.ci(boot[m])
                slo, shi = S.ci(boot_s[m])
                # direct score + subject bootstrap CI
                d = direct[m]
                if t == "A":
                    dval = S.fast_auc(y, d)
                    dboot = S.cluster_bootstrap(y, {"d": d}, groups, kind="auc",
                                                n_boot=n_boot)["d"]
                else:
                    dval = float(np.corrcoef(d, y)[0, 1])
                    _, rws = S._subject_index(groups)
                    rng = np.random.default_rng(42)
                    dboot = np.empty(n_boot)
                    for b in range(n_boot):
                        idx = np.concatenate([rws[i] for i in rng.integers(0, len(rws), len(rws))])
                        dboot[b] = np.corrcoef(d[idx], y[idx])[0, 1]
                dlo, dhi = S.ci(dboot)
                resid_rows.append({"task": t, "family": fam, "metric": m,
                                   "score": kind.upper(),
                                   "scaled_residual_alone": S.score_fn(kind)(y, preds_s[m]),
                                   "scaled_ci_lo": slo, "scaled_ci_hi": shi,
                                   "residual_alone": S.score_fn(kind)(y, p),
                                   "ci_lo": lo, "ci_hi": hi,
                                   "chance": 0.5 if t == "A" else 0.0,
                                   "direct_measure": dkind.upper(),
                                   "direct": dval, "direct_ci_lo": dlo,
                                   "direct_ci_hi": dhi,
                                   "direct_chance": 0.5 if t == "A" else 0.0})

            # B3 lengths only vs lengths + metric, with linear (n1, n2) and
            # quadratic (n1, n2, n1^2, n2^2, n1*n2) length baselines
            for base_name, base_cols in [("linear", LENGTH_LIN), ("quadratic", LENGTH_QUAD)]:
                cfg = {"lengths": base_cols}
                for m in ["dfd", "sdfd", "sdfd_1to1"]:
                    cfg[f"lengths+{m}"] = base_cols + [m]
                lp = {c: cv_oof(t, model, df[cols].to_numpy(float), y, groups, folds)[1]
                      for c, cols in cfg.items()}
                boot = S.cluster_bootstrap(y, lp, groups, kind=kind, n_boot=n_boot)
                for m in ["dfd", "sdfd", "sdfd_1to1"]:
                    a = f"lengths+{m}"
                    perm = S.paired_permutation(y, lp[a], lp["lengths"], groups, kind=kind,
                                                n_perm=2000)
                    lo, hi = S.ci(boot[a] - boot["lengths"])
                    added_rows.append({"task": t, "family": fam, "length_baseline": base_name,
                                       "metric": m, "score": kind.upper(),
                                       "lengths": S.score_fn(kind)(y, lp["lengths"]),
                                       "lengths_plus_metric": S.score_fn(kind)(y, lp[a]),
                                       "diff": perm["diff"], "ci_lo": lo, "ci_hi": hi,
                                       "perm_p": perm["p"]})
            print(f"  {t:<3}{fam:<15} done", flush=True)
    return pd.DataFrame(corr_rows), pd.DataFrame(resid_rows), pd.DataFrame(added_rows)


# --------------------------------------------------------------------------
# C. shuffled-label null
# --------------------------------------------------------------------------
def shuffled_labels(tasks, n_label, fam_which, n_label_tree):
    rows = []
    for t in tasks:
        df, folds, y, groups = load(t)
        X, _, _, _ = F.feature_matrix(df)
        rng = np.random.default_rng(7)
        for fam, model in families(t, fam_which).items():
            real, _ = cv_oof(t, model, X, y, groups, folds)
            n_here = min(n_label_tree, n_label) if fam in TREE else n_label
            null = np.array([cv_oof(t, model, X, rng.permutation(y), groups, folds)[0]
                             for _ in range(n_here)])
            lo, hi = np.quantile(null, [0.025, 0.975])
            rows.append({"task": t, "family": fam, "score": "R2" if t != "A" else "AUC",
                         "real_full_model": real, "shuffled_mean": null.mean(),
                         "shuffled_lo": lo, "shuffled_hi": hi,
                         "p": (1 + int(np.sum(null >= real))) / (n_here + 1),
                         "n_shuffles": n_here})
            print(f"  {t:<3}{fam:<15} real {real:.3f}   shuffled labels "
                  f"{null.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-emb", type=int, default=100)
    ap.add_argument("--n-emb-tree", type=int, default=20,
                    help="shuffles for random forest / XGBoost (slow); linear families use --n-emb")
    ap.add_argument("--n-label", type=int, default=50)
    ap.add_argument("--n-label-tree", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tasks", default="A,B1,B2")
    ap.add_argument("--families", default="all", choices=["all", "linear"])
    ap.add_argument("--only", default="A,B,C", help="which analyses to run")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    only = args.only.split(",")

    if "A" in only:
        print("=" * 78)
        print(f"A. SHUFFLED-EMBEDDING NULL ({args.n_emb} shuffles of the word->vector mapping; "
              f"{min(args.n_emb, args.n_emb_tree)} for tree models)")
        print("=" * 78)
        a = shuffled_embedding(tasks, args.n_emb, args.families, args.n_emb_tree)
        a.to_csv(OUT / "controls_shuffled_embedding.csv", index=False)
        print(f"\n  semantic gain = real minus shuffled. Across all cells: "
              f"{a.gain_from_semantics.min():+.3f} to {a.gain_from_semantics.max():+.3f}; "
              f"gain over DfD {a.gain_over_dfd.min():+.3f} to {a.gain_over_dfd.max():+.3f}")
        print(f"  cells where real beats shuffled at p < 0.05: "
              f"{int((a.p_real_beats_shuffled < 0.05).sum())}/{len(a)}")

    if "B" in only:
        print("\n" + "=" * 78)
        print("B. LENGTH DECOMPOSITION")
        print("=" * 78)
        c, r, ad = length_analysis(tasks, args.families, args.n_boot)
        c.to_csv(OUT / "controls_length_correlations.csv", index=False)
        r.to_csv(OUT / "controls_length_residual.csv", index=False)
        ad.to_csv(OUT / "controls_length_added.csv", index=False)
        print("\n  B1. how much of each metric is list length?")
        print(c.round(3).to_string(index=False))
        print("\n  B2. each metric with BOTH list lengths regressed out, used alone")
        print(r.round(3).to_string(index=False))
        print("\n  B3. lengths (n1, n2) alone vs lengths + metric")
        print(ad.round(4).to_string(index=False))

    if "C" in only:
        print("\n" + "=" * 78)
        print(f"C. SHUFFLED-LABEL NULL (full 11-feature model, {args.n_label} shuffles)")
        print("=" * 78)
        s = shuffled_labels(tasks, args.n_label, args.families, args.n_label_tree)
        s.to_csv(OUT / "controls_shuffled_label.csv", index=False)

    print("\nwrote controls_*.csv")


if __name__ == "__main__":
    main()
