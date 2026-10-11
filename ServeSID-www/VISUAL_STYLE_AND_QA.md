# ServeSID visual style and reproducible QA (WWW 2027)

This document adapts principles from six publicly available paper-layout Skills to the existing ACM `acmart` two-column review manuscript. It records **presentation** conventions, not new experimental evidence. Official venue requirements and the paper's own ground truth always override generic Skill recommendations.

## Applied guidance

- **polish-tables-figures** (ShaishavMaisuria): keep `booktabs`, no vertical table rules; double-column figures use `figure*` and `\textwidth` fractions; use ACM `\Description` alt text; inspect overfull warnings and grayscale readability.
- **paper-figures-tables** (DELONG-L): preserve every reported numerical result, source record, legend meaning, and axis scale. Use vector PDF/SVG and inspect actual paper placement rather than only full-size SVG.
- **paper-figure** (wbopan): render → compare at final ACM size → inspect text/leader-line collisions → revise → export again. Do not copy its generic fonts/template over ACM.
- **latex-typesetting-advisor** (AMICW): local changes to float typography and captions only; do not override the `acmart` geometry, baseline grid, headings, or global body font.
- **latex-document-skill** (ndpvt-web): run BibTeX and repeated LaTeX passes, inspect log warnings and preview the resulting PDF.
- **research-paper-figures** (CDUTAKL): each figure must support a stated claim; check diagram source-of-truth, distinguish measurements from illustrative cases, and retain editable vector sources.

## Figure inventory and source of truth

| Figure | Scientific role | LaTeX wrapper | Editable figure source |
|---|---|---|---|
| 1 | Teacher/student probability-mass shifts on shared SID prefixes | `Figures/fig_mech1.tex` | `Figures/ServeSID_Figure1_Hierarchy_only.svg` |
| 2 | Local-vs-candidate KL; preferred-prefix disagreement and explicit VG counterexample | `Figures/fig_stat_h1.tex` | `Figures/figure_no_title_background_vector.svg` |
| 3 | Directional multi-depth residual statistics | `Figures/fig_stat_d12.tex` | `Figures/depthwise_residuals_no_header_bg_vector.svg` |
| 4 | Illustrative teacher replacement at different SID depths | `Figures/fig_mech2.tex` | `Figures/image2.svg` |
| 5 | Method architecture overview: **placeholder** until the author's full method figure arrives | `Figures/fig_method_overview.tex` | editable TikZ fallback; optional `Figures/main_method.pdf` |

- Figures 1–4 are exported from the SVGs above to corresponding PDF files by GitHub Actions. Edit the **SVG**, not only the generated PDF, to avoid regeneration undoing changes.
- Figure 2 has deliberately truncated vertical axes. Always preserve both break indicators and disclosed 0.06 / 10% starting values. Never let explanation cards or arrows obscure the plotted bars.
- Figure 3's MLP/GRU/CNN colors are complemented by labels, but some in-plot numerical labels remain small at final ACM size. A comprehensive redesign of those numbers should use a larger panel or a supplementary numeric table, not silent deletion or rounding.
- Figure 4 contains many short labels illustrating one case. The final method diagram is still pending; do not use a fabricated schematic as verified implementation evidence.
- Figures use muted blue/orange or green for semantic roles; avoid creating interpretation that relies solely on hue.

## Tables

- Table 1: `Figures/table_main_results.tex` (8 methods × 3 datasets) and Table 2: `Figures/table_ablation.tex` (7 arms × 3 datasets).
- **Both are generated from** `scripts/gen_tables.py`. Any table typography change must update the generator and committed `.tex` consistently.
- Preserve 4-decimal means, 5-decimal reported SDs, grouping, and tie-aware highlighting. The single source for the supplied ablation statistics is `data/ablation_user_supplied_mean_sd.csv`; never silently alter numerical entries.
- Color is supportive only: best is bold, second-best gray, full ServeSID coral, and group labels show the method family in words.
- Provenance (2026-10-11): the author confirms 288 main-table and 252 ablation mean/SD entries as real results. Raw per-seed logs and checkpoint IDs remain to be linked for independent replication. The main and ablation SDs disagree for 67 of 72 duplicate Base-KD / Full metrics; keep both sources unchanged until reconciled.

## Mandatory checks after any layout change

1. Compile `main.tex` and `supplement.tex` with GitHub Actions.
2. Confirm the body-end label is on page 8 or earlier; total pages do not exceed the current WWW ceiling; no table or figure is deferred after the main body.
3. Inspect **rendered** page 4 (Figures 1–2), page 5 (Figures 3–4), page 6 (overview placeholder), and pages 7–8 (tables) at both normal PDF width and zoomed-in view.
4. Check any warnings in the LaTeX log, especially math overflow, float overlaps, citation errors and SVG-to-PDF font rendering.
5. Ensure `\Description` explains each figure's content rather than merely restating its caption; verify that exact figure data, units, counterexamples and axes remain unchanged.
6. If a graphic is below readable print size, first improve the **inner plot layout**, then its placement; do not distort `acmart`'s global body typography.
