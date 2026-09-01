# ProcRosetta — Process Science submission bundle

This directory contains the submission-ready journal manuscript, cover letter,
Springer Nature source dependencies, and the machine-readable records used in
the article.

## Primary files

- `main.tex` — core manuscript source and appendices. Section 4 and Appendix A
  contain the self-contained, code-current data-generation specification; no
  detached generation document is required to interpret the corpus.
- `figures/*.tex` — the TikZ/PGFPlots figures included directly by
  `main.tex`.
- `references.bib` — bibliography.
- `sn-jnl.cls` and `sn-basic.bst` — official Springer Nature LaTeX template v3.1
  (December 2024).
- `cover_letter.tex` — journal-specific cover-letter source.
- `evaluation_evidence.json` — machine-readable scope, point estimates,
  uncertainty intervals, qualitative results, report hashes, and limitations.
- `checkpoint_provenance.json` — evaluated model configuration, training
  provenance, and hashes.
- `curriculum_manifest.json` — resolved corpus manifest for the evaluated run.
- `SUBMISSION_CHECKLIST.md` — final portal and archival checks.

Generated manuscript and cover-letter PDFs and submission archives are
intentionally not versioned. Build them locally when needed for submission.

## Build

The manuscript uses only repository-native LaTeX and vector TikZ/PGFPlots
sources. From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
```

A manual manuscript build is equivalent:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The verified build environment uses pdfTeX 1.40.25 and LatexMk 4.83. The final
log contains no undefined citations/references and no overfull manuscript
boxes.

## Reproduction record

The exact corpus manifest SHA-256 is:

```text
9a48ee723f8a717c01677d203b2df84841c1a7b0d04b4389ce681aff91cbe1db
```

The evaluated code revision is:

```text
52a92f09f07ddc6b9b30463c5216973db7c76f83
```

The quantitative paper claims are deliberately scoped to the deterministic,
family-complete fixed evaluation subset: 16 families (64 rows) per curriculum,
four decode sources, beam width 2, maximum decode length 512, and 16 behavior
simulation traces. The complete test split contains 512 families per
curriculum. `evaluation_evidence.json` records this distinction explicitly.

## Reproducing data, training, and evaluation

Run these commands from the repository root:

```bash
./sample.py \
  --data-dir data \
  --train-families 4096 \
  --validation-families 512 \
  --test-families 512 \
  --activity-vocab-size 30 \
  --traces-per-sample 128

./train.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.pt \
  --epochs 100 \
  --batch-size 128

./test.py \
  --data-dir data \
  --curriculum complex \
  --checkpoint checkpoints/proc_rosetta.best.pt \
  --json
```

The training command has a nominal 100-epoch horizon; the reported run stopped
after 17 completed epochs and selected epoch 11 as its balanced checkpoint.
