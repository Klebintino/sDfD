"""Load and clean the raw data, build sessions and visit pairs, and write the
frozen subject-level CV splits. Prints a dataset profile.

Reads data/raw/ucsd_fluency_for_snafu_20180518.xlsx.
Writes outputs/sessions.parquet, outputs/pairs.parquet, outputs/splits_task{A,B1,B2}.json.

Run:  python scripts/01_ingest.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from sdfd import data, pairs as P

RAW = ROOT / "data" / "raw" / "ucsd_fluency_for_snafu_20180518.xlsx"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    raw = data.load_raw(RAW)
    clean = data.clean(raw)
    sess = data.sessions(clean)

    print("=" * 66)
    print("DATASET PROFILE")
    print("=" * 66)
    print(f"rows (words)            {len(raw):>8,}")
    print(f"junk tokens dropped     {clean.attrs['n_junk_dropped']:>8,}")
    print(f"subjects                {sess.id.nunique():>8,}")
    print(f"sessions                {len(sess):>8,}")
    print(f"sessions/subject        median {sess.groupby('id').size().median():.0f}, "
          f"range {sess.groupby('id').size().min()}-{sess.groupby('id').size().max()}")
    print(f"list length (unique)    mean {sess.n_unique.mean():.1f}, "
          f"range {sess.n_unique.min()}-{sess.n_unique.max()}")
    print(f"repeats total           {sess.n_repeats.sum():>8,} "
          f"across {(sess.n_repeats > 0).sum()} sessions")
    print(f"MMSE missing            {sess.MMSE.isna().sum():>8,} sessions")
    print(f"DRS missing             {sess.DRS.isna().sum():>8,} sessions")
    print("\ndiagnosis at session level:")
    print(sess.dx.value_counts().to_string())

    # subject trajectory types
    traj = sess.groupby("id").dx.apply(lambda s: tuple(sorted(set(s))))
    pure_hc = (traj == ("HC",)).sum()
    pure_ad = (traj == ("ProbAD",)).sum()
    print(f"\npure-HC subjects {pure_hc}  pure-ProbAD {pure_ad}  "
          f"transitioners {sess.id.nunique() - pure_hc - pure_ad}")

    # ---- pairs ----
    pr = P.build_pairs(sess)
    print("\n" + "=" * 66)
    print("VISIT PAIRS")
    print("=" * 66)
    print(f"all consecutive pairs   {len(pr):>8,} from {pr.id.nunique()} subjects")
    print(f"  with listnum gap > 1  {(pr.listnum_gap > 1).sum():>8,}")

    a = P.task_a(pr)
    print(f"Task A  (HC vs ProbAD)  {len(a):>8,} pairs from {a.id.nunique()} subjects "
          f"| HC {(a.y == 0).sum()}  ProbAD {(a.y == 1).sum()} "
          f"({(a.y == 0).sum() / max((a.y == 1).sum(), 1):.1f}:1)")
    for score in ("MMSE", "DRS"):
        b = P.task_b(pr, score)
        print(f"Task B  ({score:<4})          {len(b):>8,} pairs from {b.id.nunique()} subjects "
              f"| {dict(b.dx.value_counts())}")

    # ---- frozen splits ----
    print("\n" + "=" * 66)
    print("FROZEN SUBJECT-LEVEL CV SPLITS")
    print("=" * 66)
    for name, df_ in [("A", a), ("B1", P.task_b(pr, "MMSE")), ("B2", P.task_b(pr, "DRS"))]:
        folds = P.make_splits(df_, n_splits=5, seed=42)
        P.assert_no_subject_leak(df_, folds)
        P.save_splits(folds, OUT / f"splits_task{name}.json")
        sizes = [len(f["test_idx"]) for f in folds]
        print(f"task {name:<2} 5 folds, test sizes {sizes}, no subject leak  OK")

    sess.to_parquet(OUT / "sessions.parquet")
    pr.to_parquet(OUT / "pairs.parquet")
    print(f"\nwrote {OUT/'sessions.parquet'} and {OUT/'pairs.parquet'}")


if __name__ == "__main__":
    main()
