#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate LaTeX tables for the ServeSID WWW submission from transcribed results.

Data source: latest author-provided metrics (Beauty/Toys/VG, MLP/GRU/CNN):
  - main results: all methods x 3 students x 3 datasets (R@5/N@5/R@10/N@10; Jac kept from earlier runs)
  - ablation: 7 arms x 3 students x 3 datasets (R@5/N@5/R@10/N@10)

Outputs (into ../Figures/):
  - table_base_vs.tex        (Table 1: base KD vs ServeSID, NDCG@10)
  - table_ablation.tex       (Table 2: ablation arms, NDCG@10)
  - table_main_results.tex   (full all-methods table, currently unused in main)
  - table_supp_main.tex      (supplementary: full metrics per dataset)
  - table_supp_ablation.tex  (supplementary: full ablation metrics per dataset)
"""

import os

DATASETS = ["Beauty", "Toys", "VG"]
DS_SHORT = ["Beauty", "Toys", "VG"]
STUDENTS = ["MLP", "GRU", "CNN"]

TEACHER = {  # beam-20 reference
    "Beauty": [0.0428, 0.0283, 0.0677, 0.0363],
    "Toys":   [0.0405, 0.0254, 0.0648, 0.0332],
    "VG":     [0.0439, 0.0281, 0.0695, 0.0363],
}

# method -> dataset -> student -> [R@5, N@5, R@10, N@10, Jac@5, Jac@10]
MAIN = {
    "SID-MLP (base KD)": {
        "Beauty": {
            "MLP": [0.0429, 0.0282, 0.0660, 0.0356, 0.0113, 0.0154],
            "GRU": [0.0416, 0.0281, 0.0656, 0.0358, 0.0091, 0.0122],
            "CNN": [0.0380, 0.0254, 0.0580, 0.0319, 0.0075, 0.0103],
        },
        "Toys": {
            "MLP": [0.0374, 0.0235, 0.0626, 0.0315, 0.0062, 0.0089],
            "GRU": [0.0367, 0.0233, 0.0565, 0.0296, 0.0045, 0.0066],
            "CNN": [0.0328, 0.0210, 0.0533, 0.0276, 0.0038, 0.0056],
        },
        "VG": {
            "MLP": [0.0588, 0.0383, 0.0922, 0.0488, 0.0100, 0.0131],
            "GRU": [0.0591, 0.0389, 0.0935, 0.0499, 0.0088, 0.0117],
            "CNN": [0.0553, 0.0358, 0.0870, 0.0460, 0.0092, 0.0121],
        },
    },
    "ServeSID (ours)": {
        "Beauty": {
            "MLP": [0.0442, 0.0292, 0.0679, 0.0369, 0.0118, 0.0154],
            "GRU": [0.0441, 0.0291, 0.0682, 0.0368, 0.0116, 0.0151],
            "CNN": [0.0437, 0.0288, 0.0686, 0.0368, 0.0114, 0.0150],
        },
        "Toys": {
            "MLP": [0.0408, 0.0254, 0.0644, 0.0330, 0.0062, 0.0087],
            "GRU": [0.0388, 0.0250, 0.0650, 0.0334, 0.0060, 0.0086],
            "CNN": [0.0398, 0.0250, 0.0639, 0.0328, 0.0059, 0.0083],
        },
        "VG": {
            "MLP": [0.0604, 0.0396, 0.0944, 0.0505, 0.0094, 0.0122],
            "GRU": [0.0607, 0.0400, 0.0954, 0.0512, 0.0092, 0.0121],
            "CNN": [0.0605, 0.0399, 0.0944, 0.0508, 0.0092, 0.0121],
        },
    },
    "SmartGR": {
        "Beauty": {
            "MLP": [0.0434, 0.0286, 0.0668, 0.0363, 0.0115, 0.0155],
            "GRU": [0.0426, 0.0286, 0.0669, 0.0361, 0.0092, 0.0125],
            "CNN": [0.0374, 0.0250, 0.0594, 0.0320, 0.0075, 0.0106],
        },
        "Toys": {
            "MLP": [0.0381, 0.0241, 0.0623, 0.0319, 0.0065, 0.0089],
            "GRU": [0.0382, 0.0242, 0.0604, 0.0314, 0.0045, 0.0068],
            "CNN": [0.0329, 0.0206, 0.0520, 0.0267, 0.0041, 0.0061],
        },
        "VG": {
            "MLP": [0.0593, 0.0388, 0.0929, 0.0496, 0.0100, 0.0130],
            "GRU": [0.0594, 0.0391, 0.0933, 0.0500, 0.0089, 0.0118],
            "CNN": [0.0549, 0.0359, 0.0870, 0.0462, 0.0091, 0.0121],
        },
    },
    "RD": {
        "Beauty": {
            "MLP": [0.0422, 0.0281, 0.0657, 0.0357, 0.0098, 0.0141],
            "GRU": [0.0433, 0.0285, 0.0671, 0.0362, 0.0096, 0.0138],
            "CNN": [0.0419, 0.0274, 0.0649, 0.0348, 0.0092, 0.0135],
        },
        "Toys": {
            "MLP": [0.0380, 0.0234, 0.0615, 0.0309, 0.0054, 0.0080],
            "GRU": [0.0381, 0.0245, 0.0614, 0.0318, 0.0051, 0.0077],
            "CNN": [0.0380, 0.0241, 0.0603, 0.0312, 0.0047, 0.0076],
        },
        "VG": {
            "MLP": [0.0575, 0.0372, 0.0927, 0.0485, 0.0086, 0.0116],
            "GRU": [0.0595, 0.0388, 0.0938, 0.0498, 0.0083, 0.0113],
            "CNN": [0.0580, 0.0373, 0.0921, 0.0483, 0.0081, 0.0112],
        },
    },
    "RRD": {
        "Beauty": {
            "MLP": [0.0430, 0.0287, 0.0659, 0.0361, 0.0125, 0.0154],
            "GRU": [0.0422, 0.0281, 0.0664, 0.0359, 0.0116, 0.0143],
            "CNN": [0.0420, 0.0281, 0.0649, 0.0354, 0.0118, 0.0145],
        },
        "Toys": {
            "MLP": [0.0396, 0.0245, 0.0625, 0.0319, 0.0061, 0.0084],
            "GRU": [0.0381, 0.0245, 0.0622, 0.0322, 0.0057, 0.0076],
            "CNN": [0.0378, 0.0239, 0.0585, 0.0306, 0.0056, 0.0080],
        },
        "VG": {
            "MLP": [0.0592, 0.0390, 0.0918, 0.0496, 0.0094, 0.0114],
            "GRU": [0.0596, 0.0393, 0.0892, 0.0490, 0.0090, 0.0105],
            "CNN": [0.0591, 0.0389, 0.0884, 0.0483, 0.0090, 0.0109],
        },
    },
    "RCE-KD": {
        "Beauty": {
            "MLP": [0.0434, 0.0286, 0.0666, 0.0363, 0.0112, 0.0151],
            "GRU": [0.0432, 0.0286, 0.0671, 0.0361, 0.0104, 0.0140],
            "CNN": [0.0409, 0.0276, 0.0651, 0.0354, 0.0094, 0.0124],
        },
        "Toys": {
            "MLP": [0.0387, 0.0244, 0.0631, 0.0324, 0.0061, 0.0087],
            "GRU": [0.0380, 0.0246, 0.0622, 0.0324, 0.0052, 0.0075],
            "CNN": [0.0364, 0.0232, 0.0587, 0.0303, 0.0043, 0.0066],
        },
        "VG": {
            "MLP": [0.0593, 0.0388, 0.0926, 0.0497, 0.0097, 0.0126],
            "GRU": [0.0595, 0.0394, 0.0938, 0.0503, 0.0091, 0.0119],
            "CNN": [0.0589, 0.0387, 0.0919, 0.0493, 0.0092, 0.0120],
        },
    },
    "LOHRec": {
        "Beauty": {
            "MLP": [0.0414, 0.0278, 0.0639, 0.0350, 0.0087, 0.0116],
            "GRU": [0.0376, 0.0251, 0.0576, 0.0315, 0.0072, 0.0088],
            "CNN": [0.0361, 0.0242, 0.0542, 0.0300, 0.0081, 0.0101],
        },
        "Toys": {
            "MLP": [0.0362, 0.0224, 0.0596, 0.0299, 0.0054, 0.0072],
            "GRU": [0.0317, 0.0201, 0.0510, 0.0263, 0.0035, 0.0048],
            "CNN": [0.0320, 0.0198, 0.0514, 0.0260, 0.0035, 0.0051],
        },
        "VG": {
            "MLP": [0.0584, 0.0377, 0.0908, 0.0481, 0.0096, 0.0127],
            "GRU": [0.0538, 0.0346, 0.0818, 0.0436, 0.0079, 0.0101],
            "CNN": [0.0464, 0.0297, 0.0711, 0.0376, 0.0069, 0.0091],
        },
    },
    "GKD": {
        "Beauty": {
            "MLP": [0.0428, 0.0286, 0.0668, 0.0363, 0.0115, 0.0157],
            "GRU": [0.0416, 0.0281, 0.0656, 0.0358, 0.0091, 0.0122],
            "CNN": [0.0380, 0.0254, 0.0580, 0.0319, 0.0075, 0.0103],
        },
        "Toys": {
            "MLP": [0.0373, 0.0237, 0.0627, 0.0318, 0.0065, 0.0091],
            "GRU": [0.0382, 0.0243, 0.0613, 0.0316, 0.0048, 0.0072],
            "CNN": [0.0328, 0.0210, 0.0533, 0.0276, 0.0038, 0.0056],
        },
        "VG": {
            "MLP": [0.0593, 0.0388, 0.0929, 0.0496, 0.0101, 0.0131],
            "GRU": [0.0597, 0.0393, 0.0937, 0.0502, 0.0089, 0.0118],
            "CNN": [0.0569, 0.0370, 0.0896, 0.0475, 0.0093, 0.0124],
        },
    },
    "DistiLLM-2": {
        "Beauty": {
            "MLP": [0.0428, 0.0287, 0.0666, 0.0361, 0.0117, 0.0158],
            "GRU": [0.0416, 0.0281, 0.0656, 0.0358, 0.0091, 0.0122],
            "CNN": [0.0380, 0.0254, 0.0580, 0.0319, 0.0075, 0.0103],
        },
        "Toys": {
            "MLP": [0.0375, 0.0238, 0.0614, 0.0315, 0.0066, 0.0092],
            "GRU": [0.0367, 0.0233, 0.0565, 0.0296, 0.0045, 0.0066],
            "CNN": [0.0328, 0.0210, 0.0533, 0.0276, 0.0038, 0.0056],
        },
        "VG": {
            "MLP": [0.0594, 0.0387, 0.0927, 0.0494, 0.0100, 0.0131],
            "GRU": [0.0596, 0.0392, 0.0939, 0.0503, 0.0093, 0.0121],
            "CNN": [0.0587, 0.0384, 0.0914, 0.0489, 0.0097, 0.0127],
        },
    },
}

METHOD_ORDER = [
    "SID-MLP (base KD)", "SmartGR", "RD", "RRD", "RCE-KD",
    "LOHRec", "GKD", "DistiLLM-2", "ServeSID (ours)",
]

# ablation: arm -> dataset -> student -> [R@5, N@5, R@10, N@10]
ABLATION = {
    "d4-only KL": {
        "Beauty": {
            "MLP": [0.0431, 0.0283, 0.0663, 0.0359],
            "GRU": [0.0422, 0.0283, 0.0660, 0.0360],
            "CNN": [0.0402, 0.0266, 0.0625, 0.0348],
        },
        "Toys": {
            "MLP": [0.0379, 0.0237, 0.0628, 0.0317],
            "GRU": [0.0369, 0.0237, 0.0594, 0.0313],
            "CNN": [0.0354, 0.0225, 0.0566, 0.0305],
        },
        "VG": {
            "MLP": [0.0589, 0.0384, 0.0925, 0.0491],
            "GRU": [0.0593, 0.0390, 0.0937, 0.0502],
            "CNN": [0.0570, 0.0370, 0.0888, 0.0488],
        },
    },
    "DA-CDRS only": {
        "Beauty": {
            "MLP": [0.0435, 0.0286, 0.0672, 0.0362],
            "GRU": [0.0430, 0.0286, 0.0673, 0.0364],
            "CNN": [0.0414, 0.0273, 0.0652, 0.0359],
        },
        "Toys": {
            "MLP": [0.0389, 0.0246, 0.0638, 0.0324],
            "GRU": [0.0377, 0.0241, 0.0622, 0.0323],
            "CNN": [0.0370, 0.0236, 0.0598, 0.0322],
        },
        "VG": {
            "MLP": [0.0595, 0.0388, 0.0932, 0.0498],
            "GRU": [0.0601, 0.0394, 0.0943, 0.0507],
            "CNN": [0.0586, 0.0380, 0.0914, 0.0498],
        },
    },
    "HServe only": {
        "Beauty": {
            "MLP": [0.0436, 0.0287, 0.0670, 0.0363],
            "GRU": [0.0431, 0.0286, 0.0671, 0.0364],
            "CNN": [0.0418, 0.0276, 0.0656, 0.0360],
        },
        "Toys": {
            "MLP": [0.0390, 0.0245, 0.0633, 0.0323],
            "GRU": [0.0379, 0.0243, 0.0630, 0.0325],
            "CNN": [0.0375, 0.0238, 0.0606, 0.0320],
        },
        "VG": {
            "MLP": [0.0597, 0.0390, 0.0935, 0.0499],
            "GRU": [0.0600, 0.0395, 0.0945, 0.0506],
            "CNN": [0.0591, 0.0386, 0.0923, 0.0500],
        },
    },
        "HServe + Uniform": {
        "Beauty": {
            "MLP": [0.0437, 0.0286, 0.0673, 0.0364],
            "GRU": [0.0437, 0.0287, 0.0674, 0.0365],
            "CNN": [0.0423, 0.0280, 0.0669, 0.0362],
        },
        "Toys": {
            "MLP": [0.0395, 0.0248, 0.0640, 0.0325],
            "GRU": [0.0381, 0.0247, 0.0642, 0.0328],
            "CNN": [0.0382, 0.0242, 0.0630, 0.0323],
        },
        "VG": {
            "MLP": [0.0599, 0.0391, 0.0938, 0.0500],
            "GRU": [0.0602, 0.0397, 0.0950, 0.0510],
            "CNN": [0.0596, 0.0390, 0.0935, 0.0502],
        },
    },
}

ABLATION_ORDER = [
    "d4-only KL", "DA-CDRS only", "HServe only",
    "HServe + Uniform",
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures")
OUT_DIR = os.path.normpath(OUT_DIR)


def fmt(v):
    return "{:.4f}".format(v)


def fmt_3(v):
    return "{:.3f}".format(v)


def base_vs_table():
    lines = []
    lines.append("% AUTO-GENERATED by scripts/gen_tables.py -- do not edit by hand.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{2mm}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\begin{tabular}{lllcccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Student & Method & R@5 & N@5 & R@10 & N@10 \\\\")
    lines.append("\\midrule")
    for di, ds in enumerate(DATASETS):
        for si, st in enumerate(STUDENTS):
            b = MAIN["SID-MLP (base KD)"][ds][st]
            s = MAIN["ServeSID (ours)"][ds][st]
            dsname = ds if si == 0 else ""
            lines.append(dsname + " & " + st + " & Base KD & "
                         + " & ".join(fmt(x) for x in b[:4]) + " \\\\")
            lines.append(" & & \\method & "
                         + " & ".join(fmt(x) for x in s[:4]) + " \\\\")
        if di < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Full next-item recommendation metrics of the base-KD students and \\method across datasets and architectures.}")
    lines.append("\\label{tab:main-vs}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def ablation_table():
    lines = []
    lines.append("% AUTO-GENERATED by scripts/gen_tables.py -- do not edit by hand.")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{0.8mm}")
    lines.append("\\renewcommand{\\arraystretch}{1.0}")
    lines.append("\\begin{tabular}{l" + "c" * (len(STUDENTS) * 4) + "}")
    lines.append("\\toprule")
    lines.append("Arm & \\multicolumn{4}{c}{MLP} & \\multicolumn{4}{c}{GRU} & \\multicolumn{4}{c}{CNN} \\\\")
    lines.append(" & R@5 & N@5 & R@10 & N@10 & R@5 & N@5 & R@10 & N@10 & R@5 & N@5 & R@10 & N@10 \\\\")
    lines.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
    lines.append("\\midrule")
    for ds in DATASETS:
        lines.append("\\multicolumn{13}{l}{\\emph{" + ds + "}} \\\\")
        cells = ["SID-MLP (base KD)"]
        for st in STUDENTS:
            v = MAIN["SID-MLP (base KD)"][ds][st]
            cells += [fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3])]
        lines.append(" & ".join(cells) + " \\\\")
        for arm in ABLATION_ORDER:
            cells = [arm]
            for st in STUDENTS:
                v = ABLATION[arm][ds][st]
                cells += [fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3])]
            lines.append(" & ".join(cells) + " \\\\")
        cells = ["\\textbf{\\method (full)}"]
        for st in STUDENTS:
            v = MAIN["ServeSID (ours)"][ds][st]
            cells += ["\\textbf{" + fmt(v[0]) + "}", "\\textbf{" + fmt(v[1]) + "}",
                      "\\textbf{" + fmt(v[2]) + "}", "\\textbf{" + fmt(v[3]) + "}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Ablation of HServe and DA-CDRS. Structural variants compare hierarchical distribution alignment, depth-wise supervision, and their combination.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main_results_table():
    """Main results table: Dataset | Category | Method | MLP/GRU/CNN metrics.
    Category blocks are colored (trad blue / gen yellow / SID green), ServeSID
    row is coral, best per column bold, second-best per column gray (ties included)."""
    trad = ["RD", "RRD", "RCE-KD"]
    gen = ["GKD", "DistiLLM-2"]
    sid = ["SID-MLP (base KD)", "SmartGR", "LOHRec", "ServeSID (ours)"]
    cat_defs = [
        ("catTrad", 3, "Traditional Recommendation KD", trad),
        ("catGen", 2, "General Autoregressive / LLM KD", gen),
        ("catSID", 4, "SID-/Identifier-Aware Methods", sid),
    ]
    lines = []
    lines.append("% AUTO-GENERATED by scripts/gen_tables.py -- do not edit by hand.")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{2.6pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.10}")
    lines.append("\\begin{tabular}{@{}lll" + "r" * (len(STUDENTS) * 4) + "@{}}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Category} & \\multirow{2}{*}{Method}")
    for _ in STUDENTS:
        lines.append(" & \\multicolumn{4}{c}{" + _ + "}")
    lines.append(" \\\\")
    lines.append("\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}\\cmidrule(lr){12-15}")
    lines.append(" & & & R@5 & N@5 & R@10 & N@10 & R@5 & N@5 & R@10 & N@10 & R@5 & N@5 & R@10 & N@10 \\\\")
    lines.append("\\midrule")

    def cell_value(v, best_vals, second_vals):
        if abs(v - best_vals) < 1e-9:
            return "\\textbf{" + fmt(v) + "}"
        if abs(v - second_vals) < 1e-9:
            return "\\cellcolor{secondCell}" + fmt(v)
        return fmt(v)

    nrows_per_ds = 9
    for di, ds in enumerate(DATASETS):
        rows = []
        row_idx = 0
        for cat_color, cat_n, cat_label, methods in cat_defs:
            for mi, m in enumerate(methods):
                cells = []
                for st in STUDENTS:
                    v = MAIN[m][ds][st]
                    for k in range(4):
                        vals = sorted({MAIN[mm][ds][st][k] for mm in METHOD_ORDER}, reverse=True)
                        best = vals[0]
                        second = vals[1] if len(vals) > 1 else vals[0]
                        cells.append(cell_value(v[k], best, second))
                body = " & ".join(cells)
                if m == "ServeSID (ours)":
                    row = "\\rowcolor{bestRow} & & \\textbf{\\method} & " + body + " \\\\"
                else:
                    if row_idx == 0:
                        dcell = "\\cellcolor{datasetgray}\\multirow{" + str(nrows_per_ds) + "}{*}{\\textbf{" + ds + "}}"
                    else:
                        dcell = "\\cellcolor{datasetgray}"
                    if mi == 0:
                        ccell = "\\cellcolor{" + cat_color + "}\\multirow{" + str(cat_n) + "}{*}{\\parbox{1.9cm}{\\raggedright " + cat_label + "}}"
                    else:
                        ccell = "\\cellcolor{" + cat_color + "}"
                    row = dcell + " & " + ccell + " & " + m.replace("_", "\\_") + " & " + body + " \\\\"
                rows.append(row)
                row_idx += 1
        lines.extend(rows)
        if di < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Overall Top-$K$ recommendation performance. Methods are grouped by family across three datasets and three student architectures. "
                 "Bold highlights the best result in each column; gray marks second-best results (ties included).}")
    lines.append("\\label{tab:main-all}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def supp_main_tables():
    """One table per dataset: methods x students, showing R@5/N@5/R@10/N@10."""
    blocks = []
    for ds in DATASETS:
        L = []
        L.append("\\begin{table*}[t]")
        L.append("\\centering")
        L.append("\\footnotesize")
        L.append("\\setlength{\\tabcolsep}{1.1mm}")
        L.append("\\renewcommand{\\arraystretch}{1.05}")
        L.append("\\begin{tabular}{l" + "c" * (len(STUDENTS) * 4) + "}")
        L.append("\\toprule")
        h1 = "Method"
        for st in STUDENTS:
            h1 += " & \\multicolumn{4}{c}{" + st + "}"
        L.append(h1 + " \\\\")
        h2 = ""
        for _ in STUDENTS:
            h2 += " & R@5 & N@5 & R@10 & N@10"
        L.append(h2 + " \\\\")
        L.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
        L.append("\\midrule")
        L.append("\\multicolumn{13}{l}{\\emph{Teacher TIGER reference (beam-20):}} \\\\")
        t = TEACHER[ds]
        L.append("TIGER & " + " & ".join([fmt_3(t[0]), fmt_3(t[1]), fmt_3(t[2]), fmt_3(t[3])] * len(STUDENTS)) + " \\\\")
        L.append("\\midrule")
        for m in METHOD_ORDER:
            cells = [m.replace("_", "\\_")]
            for st in STUDENTS:
                v = MAIN[m][ds][st]
                cells += [fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3])]
            if m == "ServeSID (ours)":
                L.append("\\textbf{\\method} & " + " & ".join(cells[1:]) + " \\\\")
            else:
                L.append(" & ".join(cells) + " \\\\")
        L.append("\\bottomrule")
        L.append("\\end{tabular}")
        L.append("\\caption{Full utility metrics on " + ds + " for all methods and students.}")
        L.append("\\label{tab:supp-utility-" + ds.replace(" ", "") + "}")
        L.append("\\end{table*}")
        blocks.append("\n".join(L))

        J = []
        J.append("\\begin{table*}[t]")
        J.append("\\centering")
        J.append("\\footnotesize")
        J.append("\\setlength{\\tabcolsep}{2mm}")
        J.append("\\renewcommand{\\arraystretch}{1.05}")
        J.append("\\begin{tabular}{l" + "c" * (len(STUDENTS) * 2) + "}")
        J.append("\\toprule")
        jh1 = "Method"
        for st in STUDENTS:
            jh1 += " & \\multicolumn{2}{c}{" + st + "}"
        J.append(jh1 + " \\\\")
        jh2 = ""
        for _ in STUDENTS:
            jh2 += " & Jac@5 & Jac@10"
        J.append(jh2 + " \\\\")
        J.append("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
        J.append("\\midrule")
        for m in METHOD_ORDER:
            cells = [m.replace("_", "\\_")]
            for st in STUDENTS:
                v = MAIN[m][ds][st]
                cells += [fmt(v[4]), fmt(v[5])]
            if m == "ServeSID (ours)":
                J.append("\\textbf{\\method} & " + " & ".join(cells[1:]) + " \\\\")
            else:
                J.append(" & ".join(cells) + " \\\\")
        J.append("\\bottomrule")
        J.append("\\end{tabular}")
        J.append("\\caption{Teacher-agreement (Jaccard with the TIGER beam-20 top-K list) on " + ds + " for all methods and students.}")
        J.append("\\label{tab:supp-jac-" + ds.replace(" ", "") + "}")
        J.append("\\end{table*}")
        blocks.append("\n".join(J))
    return "\n\n".join(blocks)


def supp_ablation_tables():
    blocks = []
    for ds in DATASETS:
        L = []
        L.append("\\begin{table*}[t]")
        L.append("\\centering")
        L.append("\\footnotesize")
        L.append("\\setlength{\\tabcolsep}{1.1mm}")
        L.append("\\renewcommand{\\arraystretch}{1.05}")
        L.append("\\begin{tabular}{l" + "c" * (len(STUDENTS) * 4) + "}")
        L.append("\\toprule")
        h1 = "Arm"
        for st in STUDENTS:
            h1 += " & \\multicolumn{4}{c}{" + st + "}"
        L.append(h1 + " \\\\")
        h2 = ""
        for _ in STUDENTS:
            h2 += " & R@5 & N@5 & R@10 & N@10"
        L.append(h2 + " \\\\")
        L.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}")
        L.append("\\midrule")
        cells = ["SID-MLP (base KD)"]
        for st in STUDENTS:
            v = MAIN["SID-MLP (base KD)"][ds][st]
            cells += [fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3])]
        L.append(" & ".join(cells) + " \\\\")
        for arm in ABLATION_ORDER:
            cells = [arm]
            for st in STUDENTS:
                v = ABLATION[arm][ds][st]
                cells += [fmt(v[0]), fmt(v[1]), fmt(v[2]), fmt(v[3])]
            L.append(" & ".join(cells) + " \\\\")
        cells = ["\\textbf{\\method (full)}"]
        for st in STUDENTS:
            v = MAIN["ServeSID (ours)"][ds][st]
            cells += ["\\textbf{" + fmt(v[0]) + "}", "\\textbf{" + fmt(v[1]) + "}",
                      "\\textbf{" + fmt(v[2]) + "}", "\\textbf{" + fmt(v[3]) + "}"]
        L.append(" & ".join(cells) + " \\\\")
        L.append("\\bottomrule")
        L.append("\\end{tabular}")
        L.append("\\caption{Full ablation metrics on " + ds + " for all students.}")
        L.append("\\label{tab:supp-ablation-" + ds.replace(" ", "") + "}")
        L.append("\\end{table*}")
        blocks.append("\n".join(L))
    return "\n\n".join(blocks)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "table_base_vs.tex"), "w") as f:
        f.write(base_vs_table() + "\n")
    with open(os.path.join(OUT_DIR, "table_ablation.tex"), "w") as f:
        f.write(ablation_table() + "\n")
    with open(os.path.join(OUT_DIR, "table_main_results.tex"), "w") as f:
        f.write(main_results_table() + "\n")
    with open(os.path.join(OUT_DIR, "table_supp_main.tex"), "w") as f:
        f.write(supp_main_tables() + "\n")
    with open(os.path.join(OUT_DIR, "table_supp_ablation.tex"), "w") as f:
        f.write(supp_ablation_tables() + "\n")
    print("Wrote tables to", OUT_DIR)


if __name__ == "__main__":
    main()
