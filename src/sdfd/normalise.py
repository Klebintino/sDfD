"""Spelling normalisation and clustering features via snafu (Zemla et al.).

Uses snafu's animals spellfile for normalisation and the Troyer/Hills scheme
for clustering (features 6-8). Intrusions are counted against Maboudian et
al.'s animal list instead.

Install:  pip install git+https://github.com/AusterweilLab/snafu-py.git
(not `pip install snafu`, which is an unrelated package)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SPELLFILE = "animals_snafu_spellfile.csv"
SCHEMEFILE = "animals_troyer_hills_scheme.csv"
ANIMALLIST = "animals_list.txt"

# snafu's intrusions() counts words missing from the clustering scheme, which
# flags rare but valid animals. We count intrusions against Maboudian et al.'s
# 648-animal list instead (see load_animal_list).

# 'fluid': a word extends the current cluster if it shares any category with it
# (snafu's default). 'static': membership fixed by the first category.
CLUSTERTYPE = "fluid"


def _paths(raw_dir: str | Path) -> tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    return raw_dir / SPELLFILE, raw_dir / SCHEMEFILE


def _key(word: str) -> str:
    """Match the animal list's convention: lowercase, no spaces or hyphens."""
    return word.strip().lower().replace(" ", "").replace("-", "")


def load_animal_list(raw_dir: str | Path) -> set[str]:
    """Maboudian et al.'s 648-animal list, used to count intrusions (feature 4).

    Download:
    https://raw.githubusercontent.com/smaboudian/recurrence-analysis/main/animals_list.txt
    """
    p = Path(raw_dir) / ANIMALLIST
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- see README for the download URL.")
    return {_key(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def load_normalised(raw_dir: str | Path, csv_path: str | Path) -> pd.DataFrame:
    """Run the dataset through snafu and return one row per (id, listnum).

    Columns: id, listnum, words_norm (normalised, repeats kept),
             n_persev, n_intrusions, cluster_size, n_switches, n_clusters
    """
    import snafu

    spell, scheme = _paths(raw_dir)
    for p in (spell, scheme):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Download it from the snafu-py repo -- see README."
            )

    fd = snafu.load_fluency_data(
        str(csv_path),
        category="animals",
        spell=str(spell),
        scheme=str(scheme),
        removePerseverations=False,
        removeIntrusions=False,
        # strips trailing punctuation ("aardvark?" -> "aardvark") so real animals
        # are not misread as intrusions. Genuine junk ("q") still counts as one.
        removeNonAlphaChars=True,
    )

    lists = fd.labeledlists
    persev = snafu.perseverations(lists)
    intrus = snafu.intrusions(lists, str(scheme))
    csize = snafu.clusterSize(lists, str(scheme), clustertype=CLUSTERTYPE)
    nswitch = snafu.clusterSwitch(lists, str(scheme), clustertype=CLUSTERTYPE)
    clusters = snafu.findClusters(lists, str(scheme), clustertype=CLUSTERTYPE)

    rows = []
    for i, (sub, listnum) in enumerate(fd.listnums):
        # findClusters returns one size per cluster, so len() is the cluster count
        # (always n_switches + 1, hence n_clusters is not a model feature)
        cl = clusters[i]
        rows.append(
            {
                "id": int(sub),
                "listnum": int(listnum),
                "words_norm": list(lists[i]),
                "n_persev": float(persev[i]),
                "n_intrusions_scheme": float(intrus[i]),
                "cluster_size": float(csize[i]),
                "n_switches": float(nswitch[i]),
                "n_clusters": float(len(cl)),
            }
        )
    out = pd.DataFrame(rows)

    # feature 4: intrusions against the animal reference list, not the scheme
    animals = load_animal_list(raw_dir)
    out["n_intrusions"] = out["words_norm"].apply(
        lambda ws: float(sum(1 for w in ws if _key(w) not in animals))
    )

    # how many words snafu actually changed, for reporting
    n_corr = sum(len(c) for c in fd.spelling_corrections)
    out.attrs["n_spelling_corrections"] = n_corr
    return out


def merge_into_sessions(sess: pd.DataFrame, norm: pd.DataFrame) -> pd.DataFrame:
    """Attach normalised lists and snafu features to our session table.

    Replaces `words` / `unique_words` with the normalised versions, so DfD,
    sDfD, list length and repetition counts all inherit the fix.
    """
    m = sess.merge(norm, on=["id", "listnum"], how="left", validate="one_to_one")

    missing = m["words_norm"].isna().sum()
    if missing:
        raise ValueError(f"{missing} sessions did not match snafu output")

    m["words"] = m["words_norm"]
    m["unique_words"] = m["words_norm"].apply(lambda ws: list(dict.fromkeys(ws)))
    m["n_words"] = m["words"].apply(len)
    m["n_unique"] = m["unique_words"].apply(len)
    m["n_repeats"] = m["n_words"] - m["n_unique"]
    return m.drop(columns=["words_norm"])
