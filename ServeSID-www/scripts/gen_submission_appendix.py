#!/usr/bin/env python3
"""Create compact single-PDF WWW appendix: all 540 original SD cells and
the existing sensitivity/cost tables, without duplicating body means.
"""
from pathlib import Path
from gen_tables import (
    DATASETS, STUDENTS, METHOD_ORDER, ABLATION_ORDER, MAIN_SD,
    SUPPLIED_ABLATION_SD,
)

ROOT = Path(__file__).resolve().parents[1]
SUP = (ROOT / "supplement.tex").read_text(encoding="utf-8")
OUT = ROOT / "Figures" / "submission_appendix.tex"
METHODS = list(METHOD_ORDER) + list(ABLATION_ORDER)
MEASURES = ("R@5", "N@5", "R@10", "N@10")


def tex_row(cells):
    return " & ".join(cells) + r" \\"


def extract_tabular(label):
    start = SUP.index(r"\label{" + label + "}")
    a = SUP.index(r"\begin{tabular}", start)
    b = SUP.index(r"\end{tabular}", a) + len(r"\end{tabular}")
    return SUP[a:b]


lines = [
    "% Automatically generated; regenerate with scripts/gen_submission_appendix.py",
    r"\section{Full Three-Seed Standard Deviations}",
    r"\label{sec:appendix-sd}",
    "All values below are standard deviations across seeds 42, 43, and 44.",
    "Methods are evaluated using the same next-item metrics as Tables 1--2.",
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{Complete standard deviations for the main comparisons and ablation experiments on all three datasets. Each group of four columns reports R@5, N@5, R@10, and N@10 for one student backbone. Shared Base KD and Full entries are displayed once.}",
    r"\label{tab:submission-sds}",
    r"\fontsize{6.25}{7.5}\selectfont",
    r"\setlength{\tabcolsep}{1.8pt}",
    r"\renewcommand{\arraystretch}{1.0}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll*{12}{r}@{}}",
    r"\toprule",
    tex_row([
        "Dataset", "Method / ablation",
        r"\multicolumn{4}{c}{MLP}",
        r"\multicolumn{4}{c}{GRU}",
        r"\multicolumn{4}{c}{CNN}",
    ]),
    tex_row(["", ""] + list(MEASURES) * 3),
    r"\midrule",
]
count = 0
for di, ds in enumerate(DATASETS):
    if di:
        lines.append(r"\midrule")
    for mi, method in enumerate(METHODS):
        if mi == len(METHOD_ORDER):
            lines.append(r"\addlinespace[1pt]")
        cells = [ds if mi == 0 else "", method]
        for st in STUDENTS:
            for k in range(4):
                key = (ds, st, method, k)
                val = (MAIN_SD if method in METHOD_ORDER
                       else SUPPLIED_ABLATION_SD)[key]
                cells.append(val)
                count += 1
        lines.append(tex_row(cells))
lines += [
    r"\bottomrule", r"\end{tabular*}", r"\end{table*}",
]
if count != 468:
    raise AssertionError(f"Expected 468 nonduplicate SD cells, got {count}")


def minipage(title, label, width="0.49", notes=""):
    txt = [
        r"\begin{minipage}[t]{" + width + r"\textwidth}",
        r"\centering",
        r"\textbf{" + title + r"}",
        r"\par\vspace{2pt}",
        extract_tabular(label),
    ]
    if notes:
        txt.extend([r"\par\scriptsize " + notes])
    txt.append(r"\end{minipage}")
    return "\n".join(txt)


lines += [
    r"\section{Sensitivity to Training Hyperparameters}",
    r"\label{sec:appendix-sensitivity}",
    "We vary $M$, $B$, $\\beta$, and $\\tau$ on the MLP student.",
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{Candidate-support and outsider-rank sensitivity for MLP students. The $M$ sweep uses HServe-only; the $B$ sweep uses HServe with direct Plain Pair and ten negatives.}",
    r"\fontsize{6.8}{8}\selectfont",
    minipage(r"(a) Candidate support $M$", "tab:sup-sensitive-m"),
    r"\hfill",
    minipage(r"(b) Outsider-rank bound $B$", "tab:sup-sensitive-b"),
    r"\end{table*}",
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{Correction weight and temperature sensitivity. The $\beta$ sweep evaluates the DA-CDRS weight and the $\tau$ sweep HServe-only distribution temperature.}",
    r"\fontsize{7.1}{8.5}\selectfont",
    minipage(r"(a) DA correction coefficient $\beta$", "tab:sup-sensitive-beta"),
    r"\hfill",
    minipage(r"(b) HServe temperature $\tau$", "tab:sup-sensitive-tau"),
    r"\end{table*}",
    r"\section{Offline and Online Cost}",
    r"\label{sec:appendix-cost}",
    "The Beauty cost audit uses $M=64$, $B=40$, and online beam width "
    "$W=50$. Student throughput includes frozen T5 history encoding. "
    "Reported time per request is batch-amortized.",
    r"\begin{table*}[!t]",
    r"\centering",
    r"\caption{Measured Beauty training and serving cost. Per-epoch "
    r"training time excludes periodic beam validation; full training "
    r"wall time includes it. Student throughput uses batch 32; "
    r"the TIGER reference uses batch 64. Phase times are in ms/request.}",
    r"\fontsize{7.1}{8.4}\selectfont",
    minipage(r"(a) Offline training (M=64)", "tab:sup-cost-train",
             width="0.33"),
    r"\hfill",
    minipage(r"(b) Online throughput (req/s)", "tab:sup-cost-online",
             width="0.29"),
    r"\hfill",
    minipage(r"(c) Serving phase times", "tab:sup-cost-phases",
             width="0.35"),
    r"\end{table*}",
    "The Beauty teacher cache requires 26.2 minutes for export, "
    "105.4 MB for the beam dump, 8.5 MB for Top-$K$ candidates, "
    "and 5.7 GB for shared frozen-encoder representations. "
    "Toys and Video Games teacher export times are 26.7 and "
    "38.6 minutes, respectively (the latter uses four shards).",
]
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Generated {OUT}: {count} nonduplicate SD entries across 3 datasets")
