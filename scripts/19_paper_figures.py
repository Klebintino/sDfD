"""Build the paper's figures and LaTeX tables from the CSVs in outputs/.

Reads the outputs of scripts 02-18. Scores from scripts 16 and 17B are pooled
out-of-fold; the tau sweep (14) and shuffled-embedding control (17A) use the
mean of the five fold scores, so the two differ slightly.

Run:  python scripts/19_paper_figures.py
Writes outputs/paper/figures/*.pdf|png   outputs/paper/tables/*.tex
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sdfd import metrics
from sdfd.vocab import CachedEmbedder

OUT = ROOT / "outputs"
PAPER = OUT / "paper"
FIG = PAPER / "figures"
TAB = PAPER / "tables"

# ---- style: Okabe-Ito palette (colour-blind safe) + neutral grey; serif to match the paper ----
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
BLUE, ORANGE, AQUA = "#0072B2", "#D55E00", "#009E73"   # Okabe-Ito blue, vermillion, bluish green
STYLE = {  # metric -> (label, colour, marker)
    "dfd": ("DfD", MUTED, "o"),
    "sdfd": ("sDfD (many-to-one)", ORANGE, "s"),
    "sdfd_1to1": ("sDfD (one-to-one)", BLUE, "D"),
    "list_length": ("List length", AQUA, "^"),
}
TASKS = {"A": "Task A: HC vs ProbAD (AUC)", "B1": "Task B1: MMSE (R²)", "B2": "Task B2: DRS (R²)"}
FAMS = {"A": ["logistic", "random_forest", "naive_bayes", "xgboost"],
        "B": ["ridge", "random_forest", "bayesian_ridge", "xgboost"]}
FAMLAB = {"logistic": "Logistic", "random_forest": "Random forest", "naive_bayes": "Naive Bayes",
          "xgboost": "XGBoost", "ridge": "Ridge", "bayesian_ridge": "Bayesian ridge"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def fams_for(task):
    return FAMS["A"] if task == "A" else FAMS["B"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"  figure  {name}")


def write_tex(name, body):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"  table   {name}")


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def sgn3(x):
    s = f"{x:+.3f}"
    return s.replace("-", "$-$")


# =============================================================================
# Figure 1: how the three metrics match words (real ProbAD pair, subject 2087)
# =============================================================================
def fig_matching():
    a = pd.read_parquet(OUT / "features_taskA.parquet")
    r = a[(a.id == 2087) & (a.listnum_1 == 1)].iloc[0]
    n = 12
    l1, l2 = list(r.L1)[:n], list(r.L2)[:n]
    emb = CachedEmbedder(OUT / "emb_word2vec_google_news_300.npz")
    sim = np.nan_to_num(emb.similarity_matrix(l1, l2), nan=0.0)

    from scipy.optimize import linear_sum_assignment
    pairs_dfd = [(i, l2.index(w), 1.0) for i, w in enumerate(l1) if w in l2]
    jstar = sim.argmax(axis=1)
    pairs_m2o = [(i, int(jstar[i]), sim[i, jstar[i]]) for i in range(n)]
    rr, cc = linear_sum_assignment(-sim)
    pairs_121 = [(int(i), int(j), sim[i, j]) for i, j in zip(rr, cc)]

    def score(pairs):
        return sum(s * abs(i - j) for i, j, s in pairs)

    panels = [("(a) DfD: exact matches only", pairs_dfd, MUTED),
              ("(b) sDfD, many-to-one (argmax)", pairs_m2o, ORANGE),
              ("(c) sDfD, one-to-one (Hungarian)", pairs_121, BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.5))
    for ax, (title, pairs, col) in zip(axes, panels):
        ax.set_xlim(-0.9, 1.9); ax.set_ylim(n - 0.4, -1.4); ax.axis("off")
        ax.text(0, -1.0, "Visit 1", ha="center", fontsize=7.5, color=INK2, fontweight="bold")
        ax.text(1, -1.0, "Visit 2", ha="center", fontsize=7.5, color=INK2, fontweight="bold")
        for k in range(n):
            ax.text(-0.05, k, l1[k], ha="right", va="center", fontsize=6.8)
            ax.text(1.05, k, l2[k], ha="left", va="center", fontsize=6.8)
        for i, j, s in pairs:
            exact = l1[i] == l2[j]
            ax.plot([0, 1], [i, j], color=INK if exact else col,
                    lw=1.3 if exact else 0.6 + 1.4 * s, alpha=0.95 if exact else 0.35 + 0.6 * s,
                    ls="-" if exact else (0, (3, 1.5)), solid_capstyle="round")
        ax.set_title(f"{title}\nscore = {score(pairs):.1f}", fontsize=7.5, loc="center")
    save(fig, "fig1_matching")


# =============================================================================
# Figure 2: tau sweep
# =============================================================================
def fig_tau():
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.9), gridspec_kw={"height_ratios": [2.2, 1]},
                             sharex=True)
    for c, task in enumerate(["A", "B1", "B2"]):
        d = pd.read_csv(OUT / f"tau_sweep_task{task}_word2vec_google_news_300.csv")
        ax, axb = axes[0, c], axes[1, c]
        for metric, sig, ls in [("sdfd_1to1", "gate", "-"), ("sdfd_1to1", "hinge", (0, (3, 1.5))),
                                ("sdfd", "gate", "-")]:
            s = d[(d.metric == metric) & (d.sigma == sig)].sort_values("tau")
            lab, col, mk = STYLE[metric]
            ax.plot(s.tau, s.score, color=col, ls=ls, lw=1.4, marker=mk, ms=3,
                    label=f"{lab}, {'threshold only' if sig == 'gate' else 'threshold + rescaling'}")
        # DfD reference = score at tau 0.9 gate, one-to-one (identical to DfD)
        dfd_ref = d[(d.metric == "sdfd_1to1") & (d.sigma == "gate")].sort_values("tau").score.iloc[-1]
        ax.axhline(dfd_ref, color=MUTED, lw=1, ls=":")
        ax.text(0.0, dfd_ref, " DfD", va="bottom", fontsize=6.5, color=INK2)
        ax.set_title(TASKS[task]); ax.set_ylabel("Score (mean over folds)" if c == 0 else "")
        e = d[(d.metric == "sdfd_1to1") & (d.sigma == "gate")].sort_values("tau")
        axb.plot(e.tau, 100 * e.exact_frac, color=BLUE, lw=1.4)
        e2 = d[(d.metric == "sdfd") & (d.sigma == "gate")].sort_values("tau")
        axb.plot(e2.tau, 100 * e2.exact_frac, color=ORANGE, lw=1.4)
        axb.set_ylim(30, 102); axb.set_xlabel("Threshold τ")
        axb.set_ylabel("% matches\nthat are exact" if c == 0 else "")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig2_tau_sweep")


# =============================================================================
# Figure 3: single-feature performance with 95% CIs (pooled OOF)
# =============================================================================
def fig_alone():
    sc = pd.read_csv(OUT / "significance_scores.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.9), sharey=True)
    order = ["dfd", "sdfd", "sdfd_1to1", "list_length"]
    offs = np.linspace(-0.27, 0.27, 4)
    for ax, task in zip(axes, ["A", "B1", "B2"]):
        fams = fams_for(task)
        for k, m in enumerate(order):
            lab, col, mk = STYLE[m]
            for y, fam in enumerate(fams):
                r = sc[(sc.task == task) & (sc.family == fam) & (sc.config == m)].iloc[0]
                yy = y + offs[k]
                ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=col, lw=1.2, solid_capstyle="round")
                ax.plot(r.score_pooled, yy, marker=mk, ms=4.5, color=col, mec="white", mew=0.6,
                        label=lab if y == 0 else None)
        ax.set_yticks(range(4)); ax.set_yticklabels([FAMLAB[f] if task == "A" else
                                                     FAMLAB[f] for f in fams])
        ax.invert_yaxis(); ax.set_title(TASKS[task]); ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC" if task == "A" else "R²")
    axes[0].set_yticklabels(["Logistic /\nRidge", "Random\nforest", "Naive Bayes /\nBayes ridge",
                             "XGBoost"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_single_feature")


# =============================================================================
# Figure 4: shuffled-embedding control
# =============================================================================
def fig_shuffle():
    d = pd.read_csv(OUT / "controls_shuffled_embedding.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.2), sharey=True)
    for ax, task in zip(axes, ["A", "B1", "B2"]):
        fams = fams_for(task)
        rows = [(f, m) for f in fams for m in ["sdfd", "sdfd_1to1"]]
        for y, (fam, m) in enumerate(rows):
            r = d[(d.task == task) & (d.family == fam) & (d.metric == m)].iloc[0]
            lab, col, mk = STYLE[m]
            ax.plot([r.shuffled_lo, r.shuffled_hi], [y, y], color="#c9c8c3", lw=5,
                    solid_capstyle="butt", label="Shuffled embedding, 95% range" if y == 0 else None)
            ax.plot(r.shuffled_mean, y, marker="|", ms=8, color=INK2, mew=1.2,
                    label="Shuffled embedding, mean" if y == 0 else None)
            ax.plot(r.real, y, marker=mk, ms=4.5, color=col, mec="white", mew=0.6,
                    label=f"{lab}, real embedding" if y in (0, 1) else None)
            ax.plot(r.dfd, y, marker="o", ms=3.5, color=MUTED, mfc="white", mew=1,
                    label="DfD (exact matching)" if y == 0 else None)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{FAMLAB[f]}" if m == "sdfd" else "" for f, m in rows])
        for y in range(1, len(rows), 2):
            ax.axhline(y + 0.5, color=GRID, lw=0.6)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_title(TASKS[task]); ax.set_xlabel("AUC" if task == "A" else "R²")
    axes[0].set_yticklabels([("Logistic / Ridge" if i == 0 else
                              ["", "Random forest", "Naive Bayes /\nBayes ridge", "XGBoost"][i // 2])
                             if i % 2 == 0 else "" for i in range(8)])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig4_shuffled_embedding")


# =============================================================================
# Figure 5: list length
# =============================================================================
def fig_length():
    a = pd.read_parquet(OUT / "features_taskA.parquet")
    n1, n2 = a.L1.apply(len), a.L2.apply(len)
    prod = n1 * n2
    res = pd.read_csv(OUT / "controls_length_residual.csv")
    fig = plt.figure(figsize=(6.8, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.38)
    for k, m in enumerate(["dfd", "sdfd_1to1"]):
        ax = fig.add_subplot(gs[0, k])
        for g, col, lab in [(0, BLUE, "HC"), (1, ORANGE, "ProbAD")]:
            s = a.y == g
            ax.scatter(prod[s], a[m][s], s=5, color=col, alpha=0.45, lw=0, label=lab)
        r = np.corrcoef(prod, a[m])[0, 1]
        ax.set_title(f"({'ab'[k]}) {STYLE[m][0]}: r = {r:.2f}")
        ax.set_xlabel("List length product  n₁ × n₂"); ax.set_ylabel(STYLE[m][0])
        if k == 0:
            ax.legend(frameon=False, markerscale=2.5, loc="upper left")
    ax = fig.add_subplot(gs[0, 2])
    t = res[res.task == "A"]
    order = ["dfd", "sdfd", "sdfd_1to1"]
    for y, m in enumerate(order):
        lab, col, mk = STYLE[m]
        r0 = t[t.metric == m].iloc[0]
        ax.plot([r0.direct_ci_lo, r0.direct_ci_hi], [y - 0.15, y - 0.15], color=col, lw=1.2)
        ax.plot(r0.direct, y - 0.15, marker=mk, ms=4.5, color=col, mec="white", mew=0.6,
                label="Model-free" if y == 0 else None)
        for j, fam in enumerate(FAMS["A"]):
            rf = t[(t.metric == m) & (t.family == fam)].iloc[0]
            ax.plot(rf.scaled_residual_alone, y + 0.12 + 0.06 * (j - 1.5), marker=mk, ms=3,
                    color=col, mfc="white", mew=0.8,
                    label="Fitted model, one per family" if (y == 0 and j == 0) else None)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(range(3)); ax.set_yticklabels([STYLE[m][0].replace(" (", "\n(") for m in order])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0.35, 0.65)
    ax.set_xlabel("AUC with length removed"); ax.set_title("(c) What is left after length")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1,
              fontsize=6.5, handlelength=1.2)
    save(fig, "fig5_length")


# =============================================================================
# Figure 6 (+ supplementary): SHAP beeswarms from script 18
# =============================================================================
def fig_shap():
    FIG.mkdir(parents=True, exist_ok=True)
    for t in ["A", "B1", "B2"]:
        src = OUT / "figures" / f"shap_beeswarm_task{t}.png"
        if src.exists():
            # smaller JPG copy so LaTeX compiles faster
            from PIL import Image
            im = Image.open(src).convert("RGB")
            im = im.resize((int(im.width * 0.75), int(im.height * 0.75)), Image.LANCZOS)
            im.save(FIG / f"fig_shap_task{t}.jpg", quality=88, optimize=True)
            print(f"  figure  fig_shap_task{t}.jpg (from script 18)")


# =============================================================================
# Tables
# =============================================================================
def tab_models():
    sc = pd.read_csv(OUT / "significance_scores.csv")
    lines = []
    for task in ["A", "B1", "B2"]:
        cells = []
        for fam in fams_for(task):
            r = sc[(sc.task == task) & (sc.family == fam) & (sc.config == "full")].iloc[0]
            cells.append(f"{f3(r.score_pooled)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]")
        lines.append(f"{TASKS[task].split(':')[0]} & {'AUC' if task == 'A' else '$R^2$'} & "
                     + " & ".join(cells) + r" \\")
    body = r"""\begin{tabular}{llcccc}
\toprule
Task & Score & Logistic / Ridge & Random forest & Naive Bayes / Bayes. ridge & XGBoost \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_models", body)


def tab_alone():
    t = pd.read_csv(OUT / "significance_tests.csv")
    comps = [("sdfd_1to1", "dfd"), ("sdfd", "dfd"), ("sdfd_1to1", "sdfd"),
             ("list_length", "sdfd_1to1"), ("list_length", "dfd")]
    rows = []
    for task in ["A", "B1", "B2"]:
        for k, fam in enumerate(fams_for(task)):
            cells = []
            for a, b in comps:
                r = t[(t.task == task) & (t.family == fam) & (t.a == a) & (t.b == b)].iloc[0]
                star = r"$^{*}$" if r.perm_p_holm < 0.05 else ""
                cells.append(sgn3(r["diff"]) + star)
            lead = (r"\multirow{4}{*}{" + task + "}") if k == 0 else ""
            rows.append(f"{lead} & {FAMLAB[fam]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if task != "B2" else "")
    body = r"""\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{3}{c}{Recurrence metrics compared} & \multicolumn{2}{c}{List length vs.} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Task & Family & 1:1 $-$ DfD & m:1 $-$ DfD & 1:1 $-$ m:1 & $-$ 1:1 & $-$ DfD \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_alone", body)


def tab_added():
    sc = pd.read_csv(OUT / "significance_scores.csv")
    t = pd.read_csv(OUT / "significance_tests.csv")
    la = pd.read_csv(OUT / "controls_length_added.csv")
    rows = []
    for task in ["A", "B1", "B2"]:
        for k, fam in enumerate(fams_for(task)):
            g = lambda c: sc[(sc.task == task) & (sc.family == fam) & (sc.config == c)].iloc[0].score_pooled
            cells = [f3(g("baseline"))]
            for m in ["dfd", "sdfd", "sdfd_1to1"]:
                r = t[(t.task == task) & (t.family == fam) & (t.a == f"baseline+{m}")].iloc[0]
                star = r"$^{*}$" if r.perm_p_holm < 0.05 else ""
                cells.append(sgn3(r["diff"]) + star)
            q = la[(la.task == task) & (la.family == fam) & (la.length_baseline == "quadratic")
                   & (la.metric == "dfd")].iloc[0].lengths
            cells += [f3(q), f3(g("full"))]
            lead = (r"\multirow{4}{*}{" + task + "}") if k == 0 else ""
            rows.append(f"{lead} & {FAMLAB[fam]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if task != "B2" else "")
    body = r"""\begin{tabular}{llcccccc}
\toprule
 & & & \multicolumn{3}{c}{Change when added to baseline} & & \\
\cmidrule(lr){4-6}
Task & Family & Baseline & $+$DfD & $+$sDfD m:1 & $+$sDfD 1:1 & Length terms only & All 11 features \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_added", body)


def tab_rq3():
    w = {t: pd.read_csv(OUT / f"ablation_alone_task{t}.csv", index_col=0) for t in ["A", "B1", "B2"]}
    g = {t: pd.read_csv(OUT / "gpt2_results" / f"ablation_alone_task{t}.csv", index_col=0)
         for t in ["A", "B1", "B2"]}
    rows = []
    for task in ["A", "B1", "B2"]:
        for k, fam in enumerate(fams_for(task)):
            cells = []
            for m in ["sdfd", "sdfd_1to1"]:
                a_, b_ = w[task].loc[m, fam], g[task].loc[m, fam]
                cells += [f3(a_), f3(b_), sgn3(b_ - a_)]
            lead = (r"\multirow{4}{*}{" + task + "}") if k == 0 else ""
            rows.append(f"{lead} & {FAMLAB[fam]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if task != "B2" else "")
    body = r"""\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{sDfD many-to-one} & \multicolumn{3}{c}{sDfD one-to-one} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Task & Family & Word2Vec & GPT-2 & Diff. & Word2Vec & GPT-2 & Diff. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_rq3", body)


# ---- supplementary tables -------------------------------------------------
def tab_supp_tests():
    t = pd.read_csv(OUT / "significance_tests.csv")
    name = {"sdfd_1to1": "sDfD 1:1", "sdfd": "sDfD m:1", "dfd": "DfD", "list_length": "Length",
            "baseline": "Baseline", "baseline+dfd": "Base.+DfD", "baseline+sdfd": "Base.+m:1",
            "baseline+sdfd_1to1": "Base.+1:1"}
    rows = []
    for r in t.itertuples():
        dl = f"{r.delong_p:.4f}" if r.metric == "AUC" else "--"
        rows.append(f"{r.task} & {FAMLAB[r.family]} & {name[r.a]} $-$ {name[r.b]} & {sgn3(r.diff)} & "
                    f"[{sgn3(r.boot_ci_lo)}, {sgn3(r.boot_ci_hi)}] & {r.perm_p:.4f} & "
                    f"{r.perm_p_holm:.4f} & {dl}" + r" \\")
    body = r"""\begin{longtable}{lllcccccc}
\caption{All model comparisons on pooled out-of-fold predictions. CI: subject-level bootstrap (2{,}000 resamples). Perm.~$p$: subject-level paired permutation test (5{,}000 permutations). Holm: corrected across the 12 task $\times$ family cells of each comparison. DeLong $p$ treats visit pairs as independent and applies to AUC only.}\label{tab:supp_tests}\\
\toprule
Task & Family & Comparison & Diff. & 95\% CI & Perm.~$p$ & Holm $p$ & DeLong $p$ \\
\midrule
\endfirsthead
\toprule
Task & Family & Comparison & Diff. & 95\% CI & Perm.~$p$ & Holm $p$ & DeLong $p$ \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}"""
    write_tex("tab_supp_tests", body)


def tab_supp_length():
    c = pd.read_csv(OUT / "controls_length_correlations.csv")
    rows = []
    nm = {"dfd": "DfD", "sdfd": "sDfD m:1", "sdfd_1to1": "sDfD 1:1"}
    for r in c.itertuples():
        rows.append(f"{r.task} & {nm[r.metric]} & {f3(r.r_n1)} & {f3(r.r_n2)} & {f3(r.r_n1xn2)} & "
                    f"{f3(r.rho_n1xn2)} & {f3(r.r2_on_length_terms)}" + r" \\")
    body = r"""\begin{tabular}{llccccc}
\toprule
Task & Metric & $r$ with $n_1$ & $r$ with $n_2$ & $r$ with $n_1 n_2$ & $\rho$ with $n_1 n_2$ & $R^2$ on length terms \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_supp_length_corr", body)

    res = pd.read_csv(OUT / "controls_length_residual.csv")
    rows = []
    for r in res.itertuples():
        rows.append(f"{r.task} & {FAMLAB[r.family]} & {nm[r.metric]} & "
                    f"{f3(r.direct)} [{f3(r.direct_ci_lo)}, {f3(r.direct_ci_hi)}] & "
                    f"{f3(r.scaled_residual_alone)} [{f3(r.scaled_ci_lo)}, {f3(r.scaled_ci_hi)}] & "
                    f"{f3(r.residual_alone)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]" + r" \\")
    body = r"""\begin{longtable}{lllccc}
\caption{Each recurrence metric after regressing out both list lengths ($n_1, n_2, n_1^2, n_2^2, n_1 n_2$; fitted within training folds). Model-free: the residual scored directly in the direction predicted by \citet{maboudian2022} (AUC for Task A, Pearson $r$ for Tasks B1/B2; identical across families). Scaled: a model fitted on the residual divided by its length-predicted spread (primary). Raw: a model fitted on the unscaled residual, which retains length information through its variance. Chance is 0.5 (AUC) or 0 ($R^2$, $r$).}\label{tab:supp_residual}\\
\toprule
Task & Family & Metric & Model-free & Scaled residual & Raw residual \\
\midrule
\endfirsthead
\toprule
Task & Family & Metric & Model-free & Scaled residual & Raw residual \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}"""
    write_tex("tab_supp_residual", body)

    la = pd.read_csv(OUT / "controls_length_added.csv")
    rows = []
    for r in la.itertuples():
        rows.append(f"{r.task} & {FAMLAB[r.family]} & {r.length_baseline} & {nm[r.metric]} & "
                    f"{f3(r.lengths)} & {f3(r.lengths_plus_metric)} & {sgn3(r.diff)} & "
                    f"[{sgn3(r.ci_lo)}, {sgn3(r.ci_hi)}] & {r.perm_p:.3f}" + r" \\")
    body = r"""\begin{longtable}{llllccccc}
\caption{List-length terms alone versus list-length terms plus one recurrence metric. Linear: $n_1, n_2$. Quadratic: $n_1, n_2, n_1^2, n_2^2, n_1 n_2$. Pooled out-of-fold scores; CI from a subject-level bootstrap; $p$ from a subject-level paired permutation test (2{,}000 permutations, uncorrected).}\label{tab:supp_length_added}\\
\toprule
Task & Family & Length terms & Metric & Length only & $+$ metric & Diff. & 95\% CI & $p$ \\
\midrule
\endfirsthead
\toprule
Task & Family & Length terms & Metric & Length only & $+$ metric & Diff. & 95\% CI & $p$ \\
\midrule
\endhead
""" + "\n".join(rows) + r"""
\bottomrule
\end{longtable}"""
    write_tex("tab_supp_length_added", body)


def tab_supp_shuffle_label():
    s = pd.read_csv(OUT / "controls_shuffled_label.csv")
    rows = [f"{r.task} & {FAMLAB[r.family]} & {f3(r.real_full_model)} & {f3(r.shuffled_mean)} & "
            f"[{f3(r.shuffled_lo)}, {f3(r.shuffled_hi)}] & {r.n_shuffles}" + r" \\" for r in s.itertuples()]
    body = r"""\begin{tabular}{llcccc}
\toprule
Task & Family & Real labels & Shuffled labels (mean) & 95\% range & Shuffles \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_supp_shuffled_label", body)


def tab_supp_shap():
    rows = []
    for task in ["A", "B1", "B2"]:
        k = pd.read_csv(OUT / f"shap_oof_ranks_task{task}.csv", index_col=0)
        for i, feat in enumerate(k.index):
            lead = (r"\multirow{11}{*}{" + task + "}") if i == 0 else ""
            rows.append(f"{lead} & \\texttt{{{feat.replace('_', chr(92) + '_')}}} & "
                        + " & ".join(str(v) for v in k.loc[feat].values) + r" \\")
        rows.append(r"\midrule" if task != "B2" else "")
    body = r"""\begin{tabular}{llcccc}
\toprule
Task & Feature & Logistic / Ridge & Random forest & Naive Bayes / Bayes. ridge & XGBoost \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_supp_shap_ranks", body)


def tab_supp_pca():
    rows = []
    for task in ["A", "B1", "B2"]:
        imp = pd.read_csv(OUT / f"pca_importance_task{task}.csv", index_col=0)
        rank_cols = [c for c in imp.columns if c.startswith(("scr_", "shap_"))]
        imp["mean_rank"] = imp[rank_cols].mean(axis=1)
        for pc in ["PC1", "PC4", "PC8"]:
            r = imp.loc[pc]
            rows.append(f"{task} & {pc} & {100 * r.var_explained:.1f}\\% & {int(r.var_rank)} & "
                        f"{r.mean_rank:.1f} & {int(r[rank_cols].min())}--{int(r[rank_cols].max())}"
                        + r" \\")
    body = r"""\begin{tabular}{llcccc}
\toprule
Task & Component & Variance explained & Variance rank & Mean predictive rank & Range \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}"""
    write_tex("tab_supp_pca", body)


def main():
    print("building paper figures and tables")
    fig_matching(); fig_tau(); fig_alone(); fig_shuffle(); fig_length(); fig_shap()
    tab_models(); tab_alone(); tab_added(); tab_rq3()
    tab_supp_tests(); tab_supp_length(); tab_supp_shuffle_label(); tab_supp_shap(); tab_supp_pca()
    print(f"\nwrote {PAPER}")


if __name__ == "__main__":
    main()
