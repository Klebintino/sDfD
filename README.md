# sDfD: semantic Distance from Diagonal

Distance from Diagonal (DfD; Maboudian et al., 2022) compares two animal-fluency lists from the same person at
consecutive visits. It sums, over every word said at both visits, how far that word moved in position. sDfD extends
DfD by replacing exact word matching with embedding cosine similarity (Word2Vec, with the GPT-2 input embedding as a
comparison), so a word can also be matched to a similar word. Two versions are implemented: many-to-one (each word
takes its most similar partner) and one-to-one (optimal assignment). On the UCSD longitudinal fluency data, sDfD
beats DfD when each is used alone to separate healthy controls from probable Alzheimer's disease and to predict MMSE
and DRS. That advantage comes from list length, not meaning: it survives when the word vectors are shuffled
between words, and it disappears once list length is accounted for. Under held-out evaluation, neither metric adds
information beyond list length.

## Paper

Gokal, K. (2026). *Does semantic matching improve a longitudinal verbal fluency metric? A held-out evaluation of
distance from diagonal and its semantic extension.* BSc Honours research report, University of the Witwatersrand.

## Setup

Python 3.13.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

snafu is installed from GitHub (`pysnafu`, imported as `snafu`). Do not `pip install snafu`: that PyPI package is
unrelated. torch and transformers are only needed for script 11.

## Data

The input files are not included in this repository, because their sources do not state a licence that allows
redistribution. Put these four files in `data/raw/`:

| file | what it is | source |
|---|---|---|
| `ucsd_fluency_for_snafu_20180518.xlsx` | UCSD ADRC longitudinal animal-fluency dataset, one row per word (subject, session, diagnosis, MMSE, DRS). Really a CSV pasted into one spreadsheet column; `sdfd.data.load_raw` re-parses it. | Published by Zemla & Austerweil (2019) on the Open Science Framework: https://osf.io/j6qea/ (as cited by Maboudian et al., 2022). Download ucsd_fluency_for_snafu_20180518.xlsx into data/raw/. |
| `animals_snafu_spellfile.csv` | spelling and variant normalisation | [snafu-py `spellfiles/`](https://raw.githubusercontent.com/AusterweilLab/snafu-py/master/spellfiles/animals_snafu_spellfile.csv) |
| `animals_troyer_hills_scheme.csv` | Troyer/Hills category scheme, used for clustering and switching | [snafu-py `schemes/`](https://raw.githubusercontent.com/AusterweilLab/snafu-py/master/schemes/animals_troyer_hills_scheme.csv) |
| `animals_list.txt` | the 648-animal list of Maboudian et al. (2022), used to count intrusions | [smaboudian/recurrence-analysis](https://raw.githubusercontent.com/smaboudian/recurrence-analysis/main/animals_list.txt) |

The spellfile and scheme come from SNAFU (Zemla, Cao, Mueller & Austerweil; https://github.com/AusterweilLab/snafu-py).
The animal list comes from Maboudian et al. (2022), *Frontiers in Aging Neuroscience*,
https://doi.org/10.3389/fnagi.2022.810799. Script 04 writes `data/raw/ucsd_fluency.csv`, a plain-CSV copy that
snafu's loader needs.

## Running the pipeline

Run the scripts from the repository root, in this order. Each script prints what it finds. Everything is written to
`outputs/`.

| # | script | what it does | writes |
|---|---|---|---|
| 01 | `01_ingest.py` | loads and cleans the data, builds sessions and visit pairs, writes the frozen CV splits | `sessions.parquet`, `pairs.parquet`, `splits_task{A,B1,B2}.json` |
| 02 | `02_reproduce_dfd.py` | reproduces Maboudian et al.'s HC vs ProbAD DfD result, with and without normalisation | `taskA_dfd.parquet`, `dfd_replication.png` |
| 04 | `04_build_features.py` | snafu normalisation and features 1-8 (length, repeats, intrusions, DfD, clustering) | `features_task{A,B1,B2}.parquet` |
| 05 | `05_build_cache.py` | Word2Vec cache for the animal vocabulary | `emb_word2vec_google_news_300.npz` |
| 05a | `05a_check_coverage.py` | read-only: Word2Vec coverage of the vocabulary | (prints only) |
| 05b | `05b_check_splits.py` | read-only: audit of a vocabulary-based compound splitter | (prints only) |
| 06 | `06_semantic_features.py [emb_tag]` | adds sem_dist_mean, sem_dist_sd, sdfd, sdfd_1to1 (overwrites the feature tables in place) | `features_task*.parquet` |
| 07 | `07_models.py A [--fast]` | Task A: four classifiers, SHAP and scrambling importance | `model_results_taskA.csv`, `ranks_{shap,scrambling}_taskA.csv` |
| 08 | `08_ablation.py {A,B1,B2}` | each recurrence metric alone, and added to the structural baseline | `ablation_{alone,added}_task*.csv` |
| 09 | `09_regression.py {B1,B2} [--fast]` | Tasks B1 (MMSE) and B2 (DRS): four regressors, SHAP and scrambling | `model_results_taskB*.csv`, `ranks_*_taskB*.csv` |
| 10 | `10_rfe_pca.py {A,B1,B2}` | RFE, then PCA with importance computed on the components | `pca_{loadings,importance}_task*.csv` |
| 11 | `11_build_gpt2_cache.py` | GPT-2 input-embedding cache, with the same word set as Word2Vec | `emb_gpt2.npz` |
| 12 | `12_check_embeddings.py` | anisotropy and related/unrelated separation for each cache | (prints only) |
| 13 | `13_rq3_diagnostics.py` | exact vs semantic share of sDfD, embedding agreement, compute cost | `rq3_decomposition.csv` |
| 14 | `14_tau_sweep.py [task] [emb_tag]` | similarity threshold (tau) and weighting (sigma) sweep | `tau_sweep_task*_<emb_tag>.csv` |
| 15 | `15_specificity.py` | how the excluded non-AD dementia pairs behave (qualitative, n = 19) | `specificity_non_ad.csv` |
| 16 | `16_significance.py` | significance tests on pooled out-of-fold predictions | `significance_{scores,tests}.csv`, `oof_predictions_task*.parquet` |
| 17 | `17_controls.py` | shuffled-embedding null, length decomposition, shuffled-label null | `controls_*.csv` |
| 18 | `18_shap_beeswarm.py` | out-of-fold SHAP beeswarm plots | `figures/shap_beeswarm_task*.png`, `shap_oof_{values,ranks}_task*` |
| 19 | `19_paper_figures.py` | the paper's figures and LaTeX tables | `paper/figures/`, `paper/tables/` |

For example: `python scripts/07_models.py A`. Embedding tags are `word2vec_google_news_300` (the default) and `gpt2`.
There is no script 03.

Notes:
- Script 05 downloads word2vec-google-news-300 through gensim (about 1.6 GB, and about 4 GB of RAM to load). Later
  scripts read only the small cache. Scripts 05a and 05b also load this model.
- Script 11 downloads GPT-2 from Hugging Face (about 500 MB), needs torch and transformers, and needs the Word2Vec
  cache first.
- Script 06 overwrites the feature tables, so every later script uses whichever embedding was run through 06 last.

### GPT-2 comparison

Word2Vec is the primary embedding. The GPT-2 results in `outputs/gpt2_results/` were produced like this:

1. `python scripts/11_build_gpt2_cache.py`
2. `python scripts/06_semantic_features.py gpt2`
3. Run `07_models.py A` and `08_ablation.py` for A, B1 and B2. Copy the resulting CSVs into `outputs/gpt2_results/`.
4. `python scripts/06_semantic_features.py word2vec_google_news_300` to put the Word2Vec features back. Re-run any
   later script whose top-level outputs were overwritten in step 3.

Script 19 reads `outputs/gpt2_results/ablation_alone_task*.csv` for the embedding comparison table. Script 13 uses
both caches directly.

## Reproducing the paper's figures and tables

After scripts 01-15, run:

```
python scripts/16_significance.py
python scripts/17_controls.py
python scripts/18_shap_beeswarm.py
python scripts/19_paper_figures.py
```

Script 19 writes to `outputs/paper/`:

| paper | file |
|---|---|
| Figures 1-5 | `figures/fig1_matching`, `fig2_tau_sweep`, `fig3_single_feature`, `fig4_shuffled_embedding`, `fig5_length` (.pdf and .png) |
| Figure 6, Figures S1-S2 | `figures/fig_shap_task{A,B1,B2}.jpg` |
| Tables 3-6 | `tables/tab_models.tex`, `tab_alone.tex`, `tab_added.tex`, `tab_rq3.tex` |
| Supplementary tables | `tables/tab_supp_*.tex` |

## Included results

`outputs/` already holds the results from the run reported in the paper: the results CSVs, the figures and LaTeX
tables, the frozen CV splits and both embedding caches. `specificity_non_ad.csv` (script 15) is left
out because it contains participant-level rows. You can inspect them without rerunning
anything, and with the caches you can rerun from script 06 onward without downloading Word2Vec or GPT-2. The
feature tables (`.parquet`) are not included. Scripts 01, 04 and 06 regenerate them, which needs the data in
`data/raw/`. Rerunning a script overwrites its files in `outputs/`.

## Reproducibility notes

- Cross-validation uses frozen 5-fold subject-level splits (`StratifiedGroupKFold`, seed 42) in
  `outputs/splits_task*.json`. The folds are row positions in `features_task*.parquet`. Do not regenerate the splits
  without rebuilding the feature tables, or reorder or filter those tables, because the two must stay in step.
- XGBoost results can differ by up to about 0.01 between machines, because multi-threaded training is not
  deterministic.
- Rough runtimes on a 12-core laptop: most scripts take seconds to a few minutes. 16 takes about 4 minutes, 17
  takes about 20 minutes and 18 about 1 minute. 05 is dominated by the Word2Vec download.

## Licence

The code and the included results are released under the MIT licence (see `LICENSE`). The licence does not cover
the third-party data (the UCSD dataset, the SNAFU spellfile and scheme, and the Maboudian et al. animal list), which
remain under their owners' terms.

## Contact

Kaylan Gokal, 2724001@students.wits.ac.za, kaylangokal@gmail.com
