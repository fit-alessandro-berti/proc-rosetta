# Compilation record

Date: 2 September 2026

## Scope

This was a LaTeX build and visual-layout check only. No model evaluation, training, corpus generation, data reproduction, or reproducibility test was run.

## Toolchain

- pdfTeX 3.141592653-2.6-1.40.26 (TeX Live 2025)
- BibTeX 0.99d
- Official MDPI class `Definitions/mdpi.cls`, version 31 August 2026
- `Definitions/mdpi.bst`

The local minimal TeX installation lacked `verse.sty` and the `repstopdf` wrapper required by the official MDPI class. For the build check only, `texlive-humanities` was extracted to a temporary directory and Ghostscript created temporary PDF conversions of the official EPS logos. Those temporary dependencies and conversions were removed after verification.

## Build sequence

1. `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
2. `bibtex main`
3. two further `pdflatex` passes for `main.tex`
4. two `pdflatex` passes for `supplementary.tex`
5. two `pdflatex` passes for `cover_letter.tex`

## Result

- `main.pdf`: 35 A4 pages
- `supplementary.pdf`: 6 A4 pages
- `cover_letter.pdf`: 1 A4 page
- All three final builds exited successfully.
- The final logs contained no undefined citations, undefined references, duplicate labels, LaTeX/package errors, overfull boxes, or underfull boxes.
- The remaining font-substitution messages originate from the official MDPI class and do not affect legibility.
- Every page was rasterized with Poppler and visually inspected. Tables, figures, screenshots, headers, footers, page numbers, and section transitions render without clipping or overlap.
