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
   selectable token-based replay or process-tree footprint fitness, precision,
   and F1.

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

Every controlled motif is inserted at a random point in a 4–12-node structural
SEQ/XOR/AND context. The same context is composed with each Petri representation,
and duplicate complete-language signatures are rejected.

Motif weights favor ordinary random trees (`ordinary_tree=0.75`, the three
controlled motifs approximately `0.0833` each) so most training logs carry realistic variant
diversity and alphabet sizes. `min_activities` (default 8) enforces an
alphabet floor per behavior, matching real event logs, and every stored
sample is relabeled to `A0, A1, ...` in first-seen trace order — the same
canonicalization external XES logs receive at inference time — so the trace,
tree, and Petri labels of a row always agree with the inference-time scheme.

Each sample contains:

```text
tree:        canonical process tree
traces:      tuple of activity-label traces
petri_graph: typed Petri-net graph with markings
equivalence_id: behavior ID shared by every equivalent row
exact_behavior_id / exact_trace_language_id: SHA-256 of certified normalized language
partial_order_id / structural_motif_id
behavior_signature: bounded 128-D language and footprint signature
model_variant_id / representation_kind / log_view_id
equivalence_level and validation metadata
```

The model contains:

```text
process tree tokens -> 3-layer prefix Transformer --------.
event-log traces    -> 1-layer biGRU + 1-layer set encoder -+-> 6 x 192 source memory
Petri graph         -> 5-layer edge-aware residual GNN ----'       |
                                                                cross-attention
normalized 96-D semantic latent <---- each encoder              3-layer tree decoder
             `-> disposable 2-layer hard-contrastive head
```

The training objective combines:

- tree-to-tree reconstruction;
- trace-to-tree translation;
- Petri-to-tree translation;
- all-positive supervised contrast across all six modality directions;
- within-modality view invariance;
- soft behavior-neighborhood matching;
- positive-only cross-modal cosine alignment on the semantic latent;
- variance/covariance anti-collapse regularization.

Semantic content is deterministic: supervised decoding does not sample a VAE,
KL is explicitly unsupported, and decoder memory is separate from the
contrastive head. Strong positives require a certified exact language and matching
partial-order identity. Bounded families participate only in soft behavior
geometry. The decoder cross-attends on every layer and uses an activity-copy
distribution scored against contextual occurrence states aggregated per source
activity, rather than copying from an unconditioned vocabulary alone.

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

## Multimodal Process Studio

The repository includes a four-view Streamlit application for interactive
artifact inspection, shared-latent analysis, translation, and checkpoint-history review:

```bash
streamlit run streamlit_app.py
```

The application reads trusted server-side checkpoints from `checkpoints/` by
default. Set `PROC_ROSETTA_CHECKPOINT_DIR` to use another configured directory.
Checkpoint upload is intentionally disabled: PyTorch checkpoint files are
treated as trusted executable content, so browser users may select only files
already installed by the server operator.

### Docker with NGINX

The Compose stack builds Streamlit as an unprivileged application container and
publishes it only through an NGINX reverse proxy. During the image build, a
local `checkpoints/` directory containing `.pt` files is copied into the image.
If no local checkpoint is present, the build downloads and safely extracts
`https://www.alessandroberti.it/checkpoint_rosetta_latest.tar.gz` instead.

```bash
docker compose up --build --detach
```

Open `http://localhost:8080`. To publish another host port:

```bash
PROC_ROSETTA_PORT=80 docker compose up --build --detach
```

Stop the stack with `docker compose down`. NGINX forwards Streamlit's WebSocket
connection and accepts artifact uploads up to the application's 50 MiB limit.
The fallback URL can be replaced with `PROC_ROSETTA_CHECKPOINT_URL`; set
`PROC_ROSETTA_CHECKPOINT_SHA256` as well to require an exact archive checksum.
Because checkpoints are baked into the image, rebuild it when they change.

The studio provides:

- pre-inference XES, PTML, and PNML previews;
- explicit canonical-label maps, sampling, clipping, vocabulary, arity, and
  node-limit diagnostics;
- a process-group workspace with automatic encoding after artifact import;
- cosine similarity, PCA, same-group connections, agreement, and nearest-neighbor views;
- checkpoint-specific synthetic reference galleries, fused group means, and exploratory latent interpolation;
- token-by-token grammar-masked decoding with separate EOS, syntax, arity,
  vocabulary, and Petri-conversion validation;
- experimental weighted fusion and reproducible VAE latent-sampling galleries;
- PTML, derived PNML, embedding, validation-report, and complete-workspace exports;
- checkpoint configuration and training-history inspection.

External PNML inference canonicalizes visible transition labels, supplies their
token IDs to the Petri encoder, constrains decoding to that source alphabet,
and restores original labels in normalized exports. The studio shows a warning
only when a legacy checkpoint lacks compatible trained transition-label
embeddings; such checkpoints need retraining for meaningful PNML activity-copy
quality.

## Retraining A Checkpoint

Start by recreating synthetic training, validation, and test splits:

```bash
./sample.py \
  --data-dir data \
  --train-families 4096 \
  --validation-families 512 \
  --test-families 512 \
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

Generated trees and modality-specific reconstruction targets carry the
`pm4py-fold-v1` normalization marker. To upgrade an existing split archive
without regenerating its sampled behavior, run:

```bash
scripts/migrate_dataset_normalization.py data
```

The migration folds each semantic tree once, rebuilds source-legal tree, trace,
and Petri decoder targets, and writes the current schema metadata. Checkpoints
from an older model architecture still require retraining; changing their
metadata cannot make incompatible tensor shapes valid.

`sample.py` shows per-split `tqdm` progress on stderr while triplets are
generated. Pass `--quiet` to suppress the progress bars.
Pass `--multiprocessing` to generate behavior families concurrently with
`max(1, N - 1)` worker processes, where `N` is the available logical CPU count.

These are also the command defaults: 4,096 independent training behavior
families and 512 families in each evaluation split. With two representations
and two fixed log views per family, they produce 16,384/2,048/2,048 flattened
rows. Under balanced motif weights that gives 1,024 training families and 128
validation/test families per motif. The larger evaluation splits are
intentional: they keep per-motif and per-representation estimates meaningful
for the more heterogeneous family, sampling, and noise strata.

The legacy `--train-count`, `--validation-count`, and `--test-count` flags still
accept flattened row counts but print a prominent deprecation warning. Family
counts are primary so behavior diversity cannot be mistaken for duplicated
views. Consecutive rows are alternate representations of one behavior and a
behavior never crosses split boundaries. Useful generator controls include:

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
The gated remediation presets are `stage_a_tiny_overfit`,
`stage_b_exact_alignment`, `stage_c_behavior_geometry`, and
`stage_d_observation_curriculum`.

Generator JSON uses nested `motifs`, `class_coverage`, `representations`,
`logs`, and `validation` objects. Metadata records deterministic seeds, transformations,
structural statistics, and exact/bounded language certificates.
Log modes include uniform, resampled, long-tail, sparse, incomplete, and noisy;
noisy logs retain exact edit provenance.
The default corpus creates two independent clean views (`uniform_variants` and
`resampled`) and asserts that exact
behavior signatures are disjoint across train, validation, and test. Training
collation fails on over-length traces instead of silently truncating them.

Train the model:

```bash
./train.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.pt \
  --epochs 100 \
  --batch-size 128
```

Training writes:

```text
checkpoints/proc_rosetta.pt       # latest completed epoch
checkpoints/proc_rosetta.best.pt  # best validation-loss epoch (testing default)
checkpoints/proc_rosetta.best_loss.pt
checkpoints/proc_rosetta.best_trace.pt
checkpoints/proc_rosetta.best_edit.pt
checkpoints/proc_rosetta.best_latent.pt
checkpoints/training_metrics.csv  # per-epoch metrics
```

Every strict objective improvement is checkpointed. Learning-rate scheduling
and early stopping default to the stable validation `trace_to_tree` loss (or
can use `reconstruction_composite`/`loss`) with the same absolute
`--min-delta`. `train_from_data_dir()` restores `.best.pt` before returning;
`--no-restore-best-weights` disables that return-time step without changing the
latest resume checkpoint.

An EMA starts at epoch 3 with decay 0.995. Ordinary and EMA validation metrics
are both recorded, and the better scheduler-monitor result supplies the epoch's
candidate checkpoints. Use `--no-use-ema` for an ablation.

Resume an interrupted run from the latest completed epoch:

```bash
./train.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.pt \
  --epochs 100 \
  --resume
```

With `--resume`, `--epochs` is the total target epoch count. Resume uses the
checkpoint's accumulated history and training state, and the metrics CSV is
synchronized with that history before training continues. Checkpoints created
before resume-state support can still be continued from their model weights,
but their optimizer momentum and exact random state cannot be recovered.

Scheduled-sampling policy can be changed while resuming. For example, continue
an epoch-19 checkpoint without scheduled sampling using
`--scheduled-sampling-max 0`. Overrides to the maximum, start epoch, or ramp
length are logged and recorded in the first new checkpoint-history row. Other
training-configuration differences remain rejected on resume.

Useful training controls:

```bash
./train.py --quiet
./train.py --device cuda
./train.py --latent-dim 96 --hidden-dim 192 --memory-tokens 6
./train.py --semantic-latent-mode deterministic
./train.py --trace-encoder-dropout 0.2 --decoder-dropout 0.2 --projection-dropout 0.2
./train.py --tree-encoder-dropout 0.12 --petri-encoder-dropout 0.12
./train.py --weight-decay 5e-4 --label-smoothing 0.04
./train.py --activity-remap-probability 0.5 --views-per-family 2
./train.py --training-stage a  # disables metric objectives for the tiny overfit gate
./train.py --no-group-aware-batches
```

Training, testing, and external-file scripts default to `cuda` when available,
then `mps` when available, and otherwise `cpu`. Pass `--device cpu` to force CPU.

The staged run sequence is intentionally gated:

1. Generate 64 families with `--preset stage_a_tiny_overfit --train-families 64`
   and train with `--training-stage a`. Training-family trace exact must reach
   95%, while shuffled and zero source memories remain at or below 10%.
2. Use `stage_b_exact_alignment` with `--training-stage b`. The logged gate
   requires zero false negatives, effective rank above 32 for the default
   96-D head, and exact-behavior Recall@1 of at least 90%.
3. Enable `stage_c_behavior_geometry` / `--training-stage c` and inspect the
   logged behavior-distance Spearman correlation.
4. Add sparse, incomplete, long-tail, and noisy observations only with
   `stage_d_observation_curriculum` / `--training-stage d` (six views including
   the two clean baselines).

Expensive per-encoder gradient diagnostics are disabled by default. Enable them
for a diagnostic run with `--gradient-diagnostics-interval 10` (or another
positive interval), then inspect both metric/reconstruction norm ratios and the
recorded reconstruction/exact, reconstruction/geometry, and exact/geometry
gradient cosines for every encoder.

By default, reconstruction trains alone for epochs 1–2, exact/within contrastive
and anti-collapse terms ramp across epochs 3–6, and soft behavior geometry ramps
across epochs 5–10. Hard contrastive loss uses its own projection head at
temperature 0.3; the downstream semantic latent is aligned with a smaller
positive-only cosine term.

The tokenizer size is fixed by the data metadata stored when `sample.py` runs.
For logs with many activities, generate data with a sufficiently large
`--max-activities` and retrain the checkpoint.

## Evaluating A Checkpoint

Evaluate the held-out synthetic test split:

```bash
./test.py \
  --data-dir data
```

Testing resolves `checkpoints/proc_rosetta.pt` to `.best.pt` by default. To
evaluate the most recently completed epoch instead, pass
`--checkpoint-selection latest`; an explicitly named objective checkpoint is
used as-is.

The command prints its evaluation plan and `tqdm` progress bars to stderr,
including the total/completed conformance checks, decode evaluations,
behavioral pairs, baseline feature sets, and optional Petri embeddings. This
keeps stdout available for the final report. Use `--quiet` to disable progress
output.

Token-based replay on Petri nets is the default. To use footprint fitness and
precision instead, computed directly from the log and discovered process tree:

```bash
./test.py --conformance-method footprints
```

Footprint mode does not compute model footprints from converted Petri nets.

For machine-readable output:

```bash
./test.py \
  --data-dir data \
  --checkpoint checkpoints/proc_rosetta.best.pt \
  --json > report.json
```

The test report includes:

- neural test losses;
- grammar-constrained, length-normalized beam decode quality from tree, trace,
  Petri, and fused sources;
- process-discovery quality comparing `proc_rosetta_trace_mu` with PM4Py
  Inductive Miner using the selected token-based replay or footprint fitness,
  precision, and F1;
- behavioral distance summaries over the test logs;
- exact-behavior, partial-order, and analogy-neighborhood retrieval metrics;
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
order. The XES, PTML, and PNML decoding scripts restore original activity labels by
default; pass `--keep-canonical-labels` to keep the canonical labels. The
external PNML tensor construction includes visible transition-label IDs, so the
activity-copy head can preserve the canonicalized PNML activity inventory.

All three conversion scripts restrict visible output labels to the complete
source alphabet and avoid repeated visible labels by default. Use
`--no-constrain-source-activities` or `--no-avoid-duplicate-transitions` only
for legacy comparisons or latent-space experiments. Raw decoder tokens remain
available for diagnostics; PTML export uses the source-sanitized, semantically
folded tree.

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
- `src/proc_rosetta/losses.py`: structurally weighted translation, exact and
  soft behavioral metric objectives, and anti-collapse losses.
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
checkpoint stores the model weights, every loss weight and temperature,
deterministic/stochastic setting, training configuration, and synthetic data
configuration needed to reconstruct the tokenizers and model dimensions.
