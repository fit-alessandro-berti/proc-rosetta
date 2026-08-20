# Distribution artifacts

`data_rosetta_latest.tar.gz` is the schema-v5 corpus. All 10,240 rows use
`pm4py-fold-v1` semantic normalization and contain separate source-legal
`tree`, `trace`, and `petri` decoder targets.

`checkpoint_rosetta_latest.tar.gz` is retained only as a legacy architecture-v2
training artifact. It uses the former GRU model and 31-row activity embeddings;
it is not compatible with the v5 latent Transformer and must not be used to
benchmark this implementation. Produce a replacement with `train.py` and the
schema-v5 corpus. Checkpoint metadata cannot safely migrate learned weights
across these architecture and tensor-shape changes.
