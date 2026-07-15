# proc-rosetta

`proc-rosetta` is a research prototype for multimodal process-mining
representation learning. It learns encoders for event logs, process trees, and
Petri nets, aligns them in one latent process-behavior space, and decodes latent
vectors back into grammar-valid process trees. Decoded trees can then be
converted to Petri nets with PM4Py.

Process trees remain the output language, but the input corpus now includes
matched non-block and alternative Petri-net representations. Each generated
behavior family has a canonical tree, shared log views, multiple Petri-net
realizations, and a visible-trace-language equivalence certificate.

## Research Questions

The code is organized around three practical research questions.

1. Can heterogeneous process-mining artifacts be embedded into a shared latent
   space?
   The project trains separate tree, trace, and Petri-net encoders whose latent
   means should agree for artifacts representing the same process behavior.

2. Can a learned model translate logs or Petri nets into valid process trees?
   The shared decoder is grammar-masked, so it can only generate syntactically
   valid process-tree token sequences. The resulting trees are checked for
   PM4Py Petri-net convertibility.

3. How do learned models compare with classical process-mining baselines?
   `test.py` reports embedding-quality metrics, decode-quality metrics, and a
   process-discovery comparison against PM4Py Inductive Miner using
   alignment-based fitness, alignment-based precision, and F1.

The implementation should be read as a first-stage feasibility study, not as a
claim that the neural model replaces mature discovery algorithms on real logs.

## What Is Implemented

The synthetic training pipeline creates behavior families and flattens them into
grouped multimodal rows:

```text
one canonical behavior -> shared master trace pool -> one or more log views
                       -> duplicate activity / silent-routing nets
                       -> concurrent / explicit-interleaving nets
                       -> block / non-free-choice M-pattern nets
                       -> canonical / isomorphically renamed random nets
                       -> optional exact prefix tries and tau refinements
```

Controlled motifs are embedded in a configurable shared sequence context
(`motif_context_size`) so matched experiments are not limited to tiny nets.

Each sample contains:

```text
tree:        canonical process tree
traces:      tuple of activity-label traces
petri_graph: typed Petri-net graph with markings
equivalence_id: behavior ID shared by every equivalent row
model_variant_id / representation_kind / log_view_id
equivalence_level and validation metadata
```

The model contains:

```text
process tree tokens -> TreeEncoder ------.
event-log traces    -> TraceEncoder -----+-> shared latent z -> GrammarTreeDecoder
Petri graph         -> PetriGraphEncoder-'
```

The training objective combines:

- tree-to-tree reconstruction;
- trace-to-tree translation;
- Petri-to-tree translation;
- latent mean alignment between modalities;
- symmetric cross-modal contrastive alignment;
- weak KL regularization.

The contrastive objective is multi-positive: every row with the same behavior
ID is a positive, and the training loader keeps multiple family views in the
same batch. The Petri encoder also embeds visible transition labels.

The default activity vocabulary supports `A0` through `A29`. If an old
checkpoint was trained with fewer activity labels, retrain it before using logs
with more activities.

## Installation

From a checkout:

```bash
python -m pip install -e .
```

The root scripts also work without installation as long as dependencies are
available in the active environment:

```bash
./sample.py --help
./train.py --help
./test.py --help
```

PM4Py is used for process-tree conversion, Petri-net handling, event-log IO,
Inductive Miner, and alignment-based conformance metrics.

## Retraining A Checkpoint

Start by recreating synthetic training, validation, and test splits:

```bash
./sample.py \
  --data-dir data \
  --train-count 8192 \
  --validation-count 1024 \
  --test-count 1024 \
  --max-activities 30 \
  --traces-per-sample 128
```

This creates:

```text
data/
  metadata.json
  training/samples.jsonl
  validation/samples.jsonl
  test/samples.jsonl
```

These defaults correspond to 4,096 independent training behavior families and
512 families in each evaluation split with the standard two representations
per family. Under balanced motif weights that gives 1,024 training families and
128 validation/test families per motif. The larger evaluation splits are
intentional: they keep per-motif and per-representation estimates meaningful
for the more heterogeneous family, sampling, and noise strata.

Counts still refer to flattened samples, preserving the earlier command shape.
Consecutive rows are alternate representations of one behavior and a behavior
never crosses split boundaries. Useful generator controls include:

```bash
./sample.py --preset smoke --train-count 32 --validation-count 8 --test-count 8
./sample.py --train-families 4096 --validation-families 512 --test-families 512
./sample.py --preset equivalence_train --log-views-per-behavior 2
./sample.py --preset nonblock_ood
./sample.py --preset noise_ood
./sample.py --motif-weights duplicate_vs_silent=1,m_nonfreechoice=1
./sample.py --generator-config configs/behavior_families.json
./sample.py --generator isolated  # legacy isolated triples
```

Behavior-family splits use deterministic class quotas rather than independent
random motif draws. By default, strict coverage requires at least 8 behavior
families per positive-weight motif in training and 4 per motif in validation
and test. Counts must describe complete families so every representation slot
receives the same coverage. Change the thresholds in the nested
`class_coverage.min_families_per_motif` configuration or uniformly from the
command line:

```bash
./sample.py --min-families-per-motif 12
```

An infeasible strict request fails before existing data is replaced. Tiny
diagnostic datasets can opt into `--class-coverage-mode best_effort`; their
manifest records any motif or representation-slot deficits. Every split's
metadata includes planned and actual motif-family counts, flattened motif ×
representation counts, and a machine-checkable `meets_minimum` result.

Additional evaluation presets include `iid_behavior`, `equivalence_seen`,
`equivalence_unseen`, `scale_ood`, `sampling_ood`, and `loops_bounded`.

Generator JSON uses nested `motifs`, `class_coverage`, `representations`,
`logs`, and `validation` objects. Metadata records deterministic seeds, transformations,
structural statistics, and exact/bounded language certificates.
Log modes include uniform, resampled, long-tail, sparse, incomplete, and noisy;
noisy logs retain exact edit provenance.

Train the model:

```bash
./train.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.pt \
  --epochs 100 \
  --batch-size 32
```

Training writes:

```text
checkpoints/proc_rosetta.pt       # latest completed epoch
checkpoints/proc_rosetta.best.pt  # best validation-loss epoch
checkpoints/training_metrics.csv  # per-epoch metrics
```

Useful training controls:

```bash
./train.py --quiet
./train.py --device cuda
./train.py --latent-dim 256 --hidden-dim 256
./train.py --dropout 0.2 --weight-decay 1e-4
./train.py --views-per-family 2
./train.py --no-group-aware-batches
```

The tokenizer size is fixed by the data metadata stored when `sample.py` runs.
For logs with many activities, generate data with a sufficiently large
`--max-activities` and retrain the checkpoint.

## Evaluating A Checkpoint

Evaluate the held-out synthetic test split:

```bash
./test.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.best.pt
```

For machine-readable output:

```bash
./test.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.best.pt \
  --json > report.json
```

The test report includes:

- neural test losses;
- greedy decode quality from tree, trace, Petri, and fused latent vectors;
- process-discovery quality comparing `proc_rosetta_trace_mu` with PM4Py
  Inductive Miner using alignment fitness, precision, and F1;
- behavioral distance summaries over the test logs;
- cross-modal retrieval metrics;
- behavior-family cosine, retrieval, equivalence margin, and distance by
  representation pair;
- learned embedding quality versus deterministic log and Petri baselines;
- PM4Py Petri-net Node2Vec/Word2Vec baseline when available.

The PM4Py Petri embedding baseline can be slow. Disable it when iterating:

```bash
./test.py --skip-pm4py-petri-embedding
```

## Using A Checkpoint On External Files

The `scripts/` directory contains direct utilities for external artifacts:

```bash
scripts/print_embeddings.py input.xes --checkpoint checkpoints/proc_rosetta.best.pt
scripts/print_embeddings.py input.pnml --checkpoint checkpoints/proc_rosetta.best.pt
scripts/print_embeddings.py input.ptml --checkpoint checkpoints/proc_rosetta.best.pt
```

Supported input formats:

- `.xes`: event log;
- `.pnml`: Petri net;
- `.ptml`: process tree.

By default, XES logs use `concept:name` as the activity key and
`case:concept:name` as the case id key. Override these if needed:

```bash
scripts/print_embeddings.py log.xes \
  --checkpoint checkpoints/proc_rosetta.best.pt \
  --activity-key activity \
  --case-id-key case_id
```

Convert an event log to a decoded process tree:

```bash
scripts/xes_to_ptml.py \
  log.xes \
  decoded.ptml \
  --checkpoint checkpoints/proc_rosetta.best.pt
```

Convert a Petri net to a decoded process tree:

```bash
scripts/pnml_to_ptml.py \
  model.pnml \
  decoded.ptml \
  --checkpoint checkpoints/proc_rosetta.best.pt
```

Encode and decode an existing process tree:

```bash
scripts/decode_ptml.py \
  input.ptml \
  decoded.ptml \
  --checkpoint checkpoints/proc_rosetta.best.pt
```

External logs are canonicalized internally to `A0`, `A1`, ... in first-seen
order. The XES and PTML decoding scripts restore original activity labels by
default; pass `--keep-canonical-labels` to keep the canonical labels. The Petri
graph encoder uses graph structure, node types, markings, and visible transition
labels; decoded PNML outputs use the model's canonical activity labels.

If a decoded model is invalid or the decoder does not emit `<eos>` within the
decode limit, the conversion script exits with a clear error instead of writing
an unusable `.ptml` file.

## Practical Limits

- The decoder still emits process trees for non-block Petri inputs. Behavioral
  decode agreement is therefore more meaningful than exact tree syntax for
  representation-ambiguous families.
- The default script limits are `--max-traces 128`, `--max-trace-length 128`,
  and `--max-petri-nodes 512` where Petri inputs are accepted.
  Increase them for larger logs if memory allows.
- A checkpoint can only encode activity labels covered by its tokenizer. The
  current default is 30 activities, but older checkpoints may support fewer.
- The model decodes process trees, not arbitrary Petri nets. Petri nets are
  produced through deterministic PM4Py tree-to-net conversion.
- PM4Py alignment metrics can be computationally expensive on larger logs and
  models.

## Project Layout

- `src/proc_rosetta/tree.py`: immutable process-tree representation.
- `src/proc_rosetta/pm4py_bridge.py`: PM4Py conversion, playout, and Petri graph
  extraction.
- `src/proc_rosetta/synthetic.py`: synthetic paired sample generation.
- `src/proc_rosetta/tokenizers.py`: process-tree and activity tokenizers plus
  grammar masks.
- `src/proc_rosetta/data.py`: JSONL datasets and batch collation.
- `src/proc_rosetta/models.py`: PyTorch encoders, latent projections, decoder,
  and multimodal model.
- `src/proc_rosetta/losses.py`: reconstruction, translation, KL, latent
  alignment, and contrastive losses.
- `src/proc_rosetta/training.py`: training, validation, checkpointing, and
  loading utilities.
- `src/proc_rosetta/benchmarks.py`: test report, embedding baselines,
  decode-quality metrics, discovery-quality metrics, and report formatting.
- `src/proc_rosetta/cli.py`: `sample`, `train`, and `test` command
  implementations.
- `scripts/`: external-file utilities for embeddings and `.xes`/`.pnml`/`.ptml`
  process-tree decoding.

## Reproducibility Notes

`sample.py` uses a Python random seed for synthetic tree generation. PM4Py
playout behavior can still depend on the installed PM4Py version. The training
checkpoint stores the model weights, training configuration, and synthetic data
configuration needed to reconstruct the tokenizers and model dimensions.
