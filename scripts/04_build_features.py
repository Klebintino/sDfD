"""Apply snafu normalisation and build features 1-8 (no embedding needed).

Writes outputs/features_task{A,B1,B2}.parquet, one row per visit pair, and
data/raw/ucsd_fluency.csv if it is missing. Script 06 adds features 9-11.

Run:  python scripts/04_build_features.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sdfd import data, features as F, normalise, pairs as P

RAW = ROOT / "data" / "raw"
XLSX = RAW / "ucsd_fluency_for_snafu_20180518.xlsx"
CSV = RAW / "ucsd_fluency.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    # snafu's loader wants a CSV; our source file is a CSV wrapped in .xlsx
    if not CSV.exists():
        data.load_raw(XLSX).to_csv(CSV, index=False)
        print(f"wrote {CSV}")

    sess = data.load_sessions(XLSX)
    norm = normalise.load_normalised(RAW, CSV)
    sess = normalise.merge_into_sessions(sess, norm)

    print("=" * 68)
    print("NORMALISATION (snafu spellfile)")
    print("=" * 68)
    print(f"  spelling corrections applied : {norm.attrs['n_spelling_corrections']:,}")
    print(f"  sessions                     : {len(sess):,}")
    print(f"  n_clusters == n_switches + 1 : "
          f"{bool(np.all(sess.n_clusters == sess.n_switches + 1))}  (so feature 6 dropped)")

    pr = P.build_pairs(sess)
    print(f"\n  visit pairs: {len(pr):,}")

    print("\n" + "=" * 68)
    print("FEATURES 1-8")
    print("=" * 68)

    for name, df_ in [
        ("A", P.task_a(pr)),
        ("B1", P.task_b(pr, "MMSE")),
        ("B2", P.task_b(pr, "DRS")),
    ]:
        feats = F.build_structural(df_, sess)
        keep = ["id", "listnum_1", "listnum_2", "dx", "y", "L1", "L2"] + \
               F.STRUCTURAL_FEATURES + ["n_clusters"]
        feats = feats[keep]
        path = OUT / f"features_task{name}.parquet"
        feats.to_parquet(path)
        print(f"  task {name:<2} {len(feats):>5} pairs -> {path.name}")

    # summary of the structural features on Task A, by group
    a = F.build_structural(P.task_a(pr), sess)
    print("\n" + "=" * 68)
    print("TASK A: feature means by group")
    print("=" * 68)
    summ = a.groupby(a.y.map({0: "HC", 1: "ProbAD"}))[F.STRUCTURAL_FEATURES].mean().T
    summ["difference"] = summ["HC"] - summ["ProbAD"]
    print(summ.round(3).to_string())
    print("\nfeatures 9-11 (semantic) need the embedding cache -- see scripts 05 and 06")


if __name__ == "__main__":
    main()
