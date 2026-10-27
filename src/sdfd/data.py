"""Load the UCSD semantic fluency dataset (distributed with SNAFU).

`ucsd_fluency_for_snafu_20180518.xlsx` is really a CSV pasted into one
spreadsheet column, so we read the rows with openpyxl and re-parse them as CSV.

One row = one word. Columns:
    id       subject identifier
    listnum  session index for that subject (the only ordering; no dates)
    group    clinical diagnosis at that session
    SEX      1/2
    EDUC     years of education
    DRS      Dementia Rating Scale, 0-144
    MMSE     Mini-Mental State Exam, 0-30 (uses negative sentinels for missing)
    category always "animals"
    rank     1-indexed position of the word within the session
    item     the word
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

# The 21 raw `group` labels, collapsed into coarse diagnoses.
HC_LABELS = {"NC"}
PROBAD_LABELS = {"ProbAD"}
MCI_LABELS = {
    "MCI", "MCI-4a", "MCI-4b", "MCI-4c", "MCI-4d",
    "MILDNEUROCOGNITIVEDISORDER", "ImpairednoMCI", "AMNESTICSYNDROME",
}
ATRISK_LABELS = {"ATRISKFORAD", "ATRISKFORMID", "POSSIBLEAD"}
# Non-AD dementias. Excluded from all modelling tasks.
OTHER_DEMENTIA_LABELS = {
    "LEWYBODYVARNT", "DLB", "FTD", "PARKINSONSDISEASEW/DEMENTIA", "NCPD",
    "MIXED/VASCULAR+AD", "OTHERCNSDX", "PSEUDODEMENTIA/DEPRESSION",
}

# Tokens that are transcription artefacts rather than attempted animal names.
JUNK_TOKENS = {"q", "yo", "drom dry", "nan", ""}


def _coarse_group(raw: str) -> str:
    if raw in HC_LABELS:
        return "HC"
    if raw in PROBAD_LABELS:
        return "ProbAD"
    if raw in MCI_LABELS:
        return "MCI"
    if raw in ATRISK_LABELS:
        return "AtRisk"
    if raw in OTHER_DEMENTIA_LABELS:
        return "Other"
    return "Unknown"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the .xlsx-wrapped CSV into a tidy one-word-per-row frame."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True)
        lines = [r[0] for r in wb.active.iter_rows(values_only=True) if r[0] is not None]
        df = pd.read_csv(io.StringIO("\n".join(lines)))
    else:
        df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise words, drop junk tokens, mark missing clinical scores."""
    df = df.copy()

    df["item"] = (
        df["item"].astype(str).str.strip().str.lower().str.replace(r"[?!.]+$", "", regex=True)
    )
    df["is_junk"] = df["item"].isin(JUNK_TOKENS)
    df["is_multiword"] = df["item"].str.contains(" ")

    n_junk = int(df["is_junk"].sum())
    df = df[~df["is_junk"]].copy()

    # `rank` is NaN on a couple of rows; fall back to file order within session.
    df["rank"] = df.groupby(["id", "listnum"]).cumcount() + 1

    # MMSE uses negative sentinels for missing, and one impossible 97.
    df["MMSE"] = df["MMSE"].where(df["MMSE"].between(0, 30))
    df["DRS"] = df["DRS"].where(df["DRS"].between(0, 144))

    df["dx"] = df["group"].map(_coarse_group)

    df.attrs["n_junk_dropped"] = n_junk
    return df


def sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (subject, session), carrying the word list.

    `words` keeps every token in spoken order (repeats included).
    `unique_words` keeps first occurrences only; DfD and sDfD use this.
    """
    df = df.sort_values(["id", "listnum", "rank"])
    rows = []
    for (sid, listnum), g in df.groupby(["id", "listnum"], sort=True):
        words = g["item"].tolist()
        seen, uniq = set(), []
        for w in words:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        rows.append(
            {
                "id": sid,
                "listnum": listnum,
                "dx": g["dx"].iloc[0],
                "dx_raw": g["group"].iloc[0],
                "SEX": g["SEX"].iloc[0],
                "EDUC": g["EDUC"].iloc[0],
                "MMSE": g["MMSE"].iloc[0],
                "DRS": g["DRS"].iloc[0],
                "words": words,
                "unique_words": uniq,
                "n_words": len(words),
                "n_unique": len(uniq),
                "n_repeats": len(words) - len(uniq),
                "n_multiword": int(g["is_multiword"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["id", "listnum"]).reset_index(drop=True)


def load_sessions(path: str | Path) -> pd.DataFrame:
    """Raw file -> cleaned session table."""
    return sessions(clean(load_raw(path)))
