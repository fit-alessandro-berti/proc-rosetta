# Applied Sciences adaptation changelog

This directory is a separate submission version. The original manuscript in `paper/` was not overwritten.

## Editorial repositioning

- Replaced the generation-led title with a title centered on behavior-supervised alignment and cross-artifact retrieval.
- Rewrote the abstract, opening motivation, scope, research questions, contribution list, discussion, and conclusion around repository search and a retrieve--propose--validate workflow.
- Made grammar-constrained process-tree generation a secondary candidate-proposal capability.
- Repeatedly separated syntactic validity, process-tree-to-Petri-net convertibility, and empirical behavioral fidelity.
- Retained and foregrounded the negative generation and external-log findings.

## Scientific presentation

- Expanded related work into process-model similarity and repository search, representation learning, discovery, multimodal learning, constrained prediction, and synthetic benchmarks.
- Added a conservative positioning table using only references already present in the verified manuscript bibliography; unsupported capabilities are marked `N/R`.
- Added formal task definitions, an explicit behavioral-relation hierarchy, proof sketches for grammar masking, deterministic completion, source-alphabet restriction, representational conversion, and non-implication of behavioral equivalence.
- Added asymptotic complexity analysis and the recorded parameter count, while explicitly withholding an empirical scalability claim.
- Reordered results so representation alignment and cross-artifact retrieval precede candidate generation.
- Renamed the two-log analysis as an external-log generation stress test.
- Revised result captions to disclose evaluation unit, sample size, duplicate handling, direction of improvement, and fixed-subset scope.
- Added explicit TODO notices for requested experiments that were not run.

## Template and organization

- Converted the separate submission version to the official MDPI ACS-style LaTeX template distributed on 1 September 2026, configured for `applsci` and `article`.
- Converted citations to numbered MDPI syntax and replaced Springer-specific front and back matter with MDPI fields.
- Added the special-issue cover letter and a retrieve--propose--validate workflow figure.
- Moved generator schematics, interface screenshots, detailed commands, full training coefficients and history, the running example, and complete serialized examples to `supplementary.tex`.
- Condensed the main implementation section to one workflow figure and a short reference-implementation paragraph.
- Preserved authors, affiliations, ORCIDs, funding, the Celonis disclosure, and the generative-AI disclosure.

## Evidence and artifacts

- Preserved all numerical results from the existing manuscript and its recorded evidence files; no value was replaced or extrapolated.
- Copied the existing bibliography, figures, screenshots, curriculum manifest, checkpoint provenance, and recorded evaluation JSON files needed by the submission source.
- Per the author's instruction, no evaluation, training, corpus-generation, or reproducibility test was run. No new evaluation artifact was created.
