# Unresolved items before submission

The manuscript marks unavailable scientific evidence explicitly instead of inventing results. The following items require author action if they are to be claimed in the submitted paper.

## Experiments not run in this adaptation

- Full held-out retrieval over all 1,536 test families.
- Deduplicated process-level retrieval with one artifact of each type per family.
- A common, non-neural cross-modal behavioral-signature retrieval baseline evaluated on the same galleries.
- Recall@5, median rank, MAP or nDCG where appropriate, and family-level paired bootstrap intervals for retrieval.
- Train--test overlap audit using normalized trees, canonical signatures, finite languages, bounded behavioral signatures, and activity-renaming-normalized structures.
- Independent semantic tests beyond descriptors overlapping the training supervision.
- Hardware-qualified encoding, indexing, query, memory, and beam-decoding measurements.
- Retrained behavior-supervision ablations and multiple random seeds.
- A heterogeneous external-data retrieval application with independently evaluated conformance.

These items were intentionally not executed because the author requested that no reproducibility tests be performed.

## Metadata and archival decisions

- Replace the mutable `latest` corpus and checkpoint URLs with immutable, licensed releases or DOI-backed archives when available.
- Add the code and data licenses and verify the provenance/license wording for each external event log.
- Supply a dependency lock or immutable container digest if one is created; the present repository metadata is not described as a lock file.
- Confirm the final volume, issue, article number, received/revised/accepted dates, and any Special Issue metadata requested by the editorial system.
- Confirm the final corresponding-author details, affiliations, ORCIDs, contribution taxonomy, funding wording, and Celonis conflict disclosure.

## Editorial verification

- Recheck every row of the related-work comparison table against the full cited publication during final author review.
- Decide whether the explicit red TODO notices should remain in the submitted manuscript or whether unsupported RQ2/scalability wording should be reduced further.
- Verify the Special Issue selection in the MDPI submission system; the advertised deadline is 20 October 2026.
