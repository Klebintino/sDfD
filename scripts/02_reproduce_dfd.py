"""Reproduce the HC vs ProbAD DfD result of Maboudian et al. (2022).

Uses their protocol (pure HC / pure ProbAD subjects, first pair only), with and
without snafu spelling normalisation, then repeats the comparison on all
consecutive Task A pairs. Writes outputs/taskA_dfd.parquet and
outputs/dfd_replication.png.

Run:  python scripts/02_reproduce_dfd.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sdfd import data, metrics, normalise, pairs as P

OUT = ROOT / "outputs"
RAW = ROOT / "data" / "raw"
XLSX = RAW / "ucsd_fluency_for_snafu_20180518.xlsx"
CSV = RAW / "ucsd_fluency.csv"

PAPER = {"n_hc": 77, "n_ad": 40,
         "dfd_hc": 43.53, "dfd_hc_sd": 27.99, "dfd_ad": 14.53, "dfd_ad_sd": 11.36,
         "t": 7.93, "len_hc": 19.55, "len_ad": 12.54}


def welch(hc, ad):
    r = stats.ttest_ind(hc, ad, equal_var=False)
    return r.statistic, r.pvalue, r.df


def maboudian(sess: pd.DataFrame) -> dict:
    """Maboudian protocol on one session table. Returns the summary numbers.

    Keeps HC/ProbAD sessions, drops subjects seen under both labels, then takes
    each subject's first pair.
    """
    hc_ad = sess[sess.dx.isin(["HC", "ProbAD"])].copy()
    traj = hc_ad.groupby("id").dx.apply(set)
    pure = {sid: next(iter(t)) for sid, t in traj.items() if len(t) == 1}
    pr = P.build_pairs(hc_ad[hc_ad.id.isin(pure)])
    first = pr.groupby("id", as_index=False).first()
    first["dx_pure"] = first.id.map(pure)
    first["DfD"] = [metrics.dfd(r.L1, r.L2) for r in first.itertuples()]

    hc, ad = first[first.dx_pure == "HC"], first[first.dx_pure == "ProbAD"]
    t, p, dof = welch(hc.DfD, ad.DfD)
    lt, _, _ = welch(hc.n2, ad.n2)
    return {
        "labelled": ((traj.apply(lambda s: "HC" in s)).sum(),
                     (traj.apply(lambda s: "ProbAD" in s)).sum(),
                     (traj.apply(len) > 1).sum()),
        "n_hc": len(hc), "n_ad": len(ad),
        "dfd_hc": hc.DfD.mean(), "dfd_hc_sd": hc.DfD.std(),
        "dfd_ad": ad.DfD.mean(), "dfd_ad_sd": ad.DfD.std(),
        "t": t, "p": p, "dof": dof,
        "len_hc": hc.n2.mean(), "len_ad": ad.n2.mean(), "len_t": lt,
        "first": first,
    }


def main():
    raw_sess = data.load_sessions(XLSX)
    if not CSV.exists():
        data.load_raw(XLSX).to_csv(CSV, index=False)
    norm = normalise.load_normalised(RAW, CSV)
    norm_sess = normalise.merge_into_sessions(raw_sess, norm)

    # ---------- A. Maboudian replication, both ways ----------
    u, n = maboudian(raw_sess), maboudian(norm_sess)

    print("=" * 72)
    print("A. MABOUDIAN PROTOCOL (pure subjects, first pair only)")
    print("=" * 72)
    print(f"  HC-labelled / ProbAD-labelled / both: {u['labelled'][0]} / "
          f"{u['labelled'][1]} / {u['labelled'][2]}   (paper: 97 / 61 / 19)")
    print(f"  snafu spelling corrections applied for 'normalised': "
          f"{norm.attrs['n_spelling_corrections']:,}\n")

    rows = [
        ("subjects HC", "n_hc", "{:.0f}"),
        ("subjects ProbAD", "n_ad", "{:.0f}"),
        ("DfD HC, mean", "dfd_hc", "{:.2f}"),
        ("DfD HC, SD", "dfd_hc_sd", "{:.2f}"),
        ("DfD ProbAD, mean", "dfd_ad", "{:.2f}"),
        ("DfD ProbAD, SD", "dfd_ad_sd", "{:.2f}"),
        ("DfD Welch t", "t", "{:.2f}"),
        ("list length HC", "len_hc", "{:.2f}"),
        ("list length ProbAD", "len_ad", "{:.2f}"),
    ]
    print(f"  {'':<20}{'un-normalised':>15}{'normalised':>13}{'paper':>9}")
    for label, key, fmt in rows:
        print(f"  {label:<20}{fmt.format(u[key]):>15}{fmt.format(n[key]):>13}"
              f"{fmt.format(PAPER[key]):>9}")
    for tag, r in [("un-normalised", u), ("normalised", n)]:
        print(f"\n  {tag}: Welch t({r['dof']:.2f}) = {r['t']:.2f}, p = {r['p']:.3g}; "
              f"ProbAD {'LOWER' if r['dfd_ad'] < r['dfd_hc'] else 'HIGHER (!! WRONG WAY)'}"
              f" than HC")
    print(f"\n  gap to paper, ProbAD mean: un-normalised {u['dfd_ad'] - PAPER['dfd_ad']:+.2f}, "
          f"normalised {n['dfd_ad'] - PAPER['dfd_ad']:+.2f}")
    print(f"  gap to paper, HC mean:     un-normalised {u['dfd_hc'] - PAPER['dfd_hc']:+.2f}, "
          f"normalised {n['dfd_hc'] - PAPER['dfd_hc']:+.2f}")

    # ---------- B. This project's protocol ----------
    print("\n" + "=" * 72)
    print("B. THIS PROJECT'S PROTOCOL (all consecutive pairs, Task A)")
    print("=" * 72)
    task = {}
    for tag, s in [("un-normalised", raw_sess), ("normalised", norm_sess)]:
        a = P.task_a(P.build_pairs(s))
        a["DfD"] = [metrics.dfd(r.L1, r.L2) for r in a.itertuples()]
        hc, ad = a.loc[a.y == 0, "DfD"], a.loc[a.y == 1, "DfD"]
        t, _, dof = welch(hc, ad)
        print(f"  {tag:<14} HC n={len(hc)} M={hc.mean():6.2f}   "
              f"ProbAD n={len(ad)} M={ad.mean():6.2f}   Welch t({dof:.1f}) = {t:.2f}")
        task[tag] = a
    print("  (pairs within a subject are correlated, so this t is anti-conservative;")
    print("   it is a sanity check on direction, not an inferential claim)")

    a = task["normalised"]
    a.to_parquet(OUT / "taskA_dfd.parquet")

    # ---------- plot (normalised, the version the project uses) ----------
    first = n["first"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (df_, gcol, title) in zip(
        axes,
        [(first, "dx_pure", "Maboudian protocol\n(first pair, pure subjects)"),
         (a.assign(g=a.y.map({0: "HC", 1: "ProbAD"})), "g",
          "This project\n(all consecutive pairs)")],
    ):
        groups = ["HC", "ProbAD"]
        ax.boxplot([df_.loc[df_[gcol] == g, "DfD"] for g in groups],
                   tick_labels=groups, widths=0.5, showfliers=False)
        rng = np.random.default_rng(0)
        for k, g in enumerate(groups, start=1):
            v = df_.loc[df_[gcol] == g, "DfD"].to_numpy()
            ax.scatter(k + rng.uniform(-0.08, 0.08, len(v)), v, alpha=0.25, s=12,
                       color="#2b6cb0" if g == "HC" else "#c05621")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("DfD")
    fig.suptitle("DfD is lower in Probable AD — replicating Maboudian et al. (2022), "
                 "snafu-normalised", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "dfd_replication.png", dpi=150)
    print(f"\nwrote {OUT/'taskA_dfd.parquet'} (normalised) and {OUT/'dfd_replication.png'}")


if __name__ == "__main__":
    main()
