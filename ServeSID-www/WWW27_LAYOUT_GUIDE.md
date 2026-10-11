# WWW 2027 research-track layout checklist (ServeSID)

## Authoritative submission rules
Source: https://www2027.thewebconf.org/research-track-papers/
- Long-paper main body: at most 8 pages; references and optional appendix after the body, total PDF at most 12 pages.
- The first 8 pages must be self-contained.
- Use `\\documentclass[sigconf, anonymous, review]{acmart}` during double-blind review.
- Follow standard ACM margins, fonts, and two-column layout; do not squeeze the manuscript by overriding ACM's text-area geometry.

## Supplementary GitHub skills reviewed (guidance, not conference rules)
- Venue-first paper revision: https://github.com/ZoeLinUTS/Academic-paper-revision/blob/master/SKILL.md
- LaTeX compiling and PDF review: https://github.com/ndpvt-web/latex-document-skill/blob/main/SKILL.md
- Publication-grade figure contrast and integrity: https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-visualization/SKILL.md

## Current layout decisions
- ACM official table captions, above tables; eliminate duplicate 12 pt in-table headings.
- Preserve original result values, and keep family colors in Table 1 and configuration colors in Table 2.
- All diagrams and statistics should remain vector PDF/SVG when possible, with legible labels at final printed size.
- Review-mode metadata should say WWW '27, Dublin, Ireland rather than the stock Conference '17 text.
- The `Figures/fig_method_overview.tex` float provides a grounded TikZ overview. Replace fallback with your actual vector `Figures/main_method.pdf` without changing LaTeX references.
- The GitHub Actions workflow builds both PDFs, verifies the References boundary and 12-page total, and renders pages 1, 4, 7 and 8 for visual review.
- Do not automatically suppress technical content or shrink ACM font/line spacing to hit the limit.

## Before submission (scientific validity, separate from typesetting)
- Verify that the author-confirmed 288 main and 252 ablation mean/SD records correspond to the `42/43/44` run artifacts, especially the corrected no-direct-pair-anchor Full objective. There are 67/72 duplicate Base-KD / Full SD discrepancies between main and ablation summaries awaiting reconciliation.
- Supply the final main method figure PDF, inspect it in the compiled paper, and check all figure captions/references.
- Require run-level metrics, checkpoint IDs, dataset split manifests and matching configurations before claiming measured statistical significance or fully reproducible improvements.
- Verify all bib entries, anonymity and metadata, accessible color contrast, and supporting diagnostic logs.
