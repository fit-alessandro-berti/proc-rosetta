# proc-rosetta

`proc-rosetta` is a first-stage implementation of a multimodal process-mining
embedding model. It uses PyTorch for neural encoders/decoders and pm4py for
process-tree conversion, Petri-net handling, and trace simulation.

The implemented scope follows the recommended staged design from the project
brief:

- process trees are the central generative object;
- traces/logs, process trees, and Petri nets have separate structural encoders;
- all encoders project into a shared process-behavior latent space;
- a grammar-masked process-tree decoder reconstructs or translates back to a
  valid process-tree token grammar;
- decoded process trees can be deterministically converted to Petri nets with
  pm4py;
- synthetic paired triples provide the initial training data.

This is intentionally block-structured first. Direct arbitrary Petri-net
decoding is left for a later stage because validity, soundness, and graph
matching are materially harder.

## Quick start

Run the root command scripts directly from a checkout. No editable install is
required as long as the Python dependencies are already available in the active
environment.

```bash
./sample.py
./train.py
./test.py
```

`sample.py` recreates a local split dataset:

```text
data/
  metadata.json
  training/samples.jsonl
  validation/samples.jsonl
  test/samples.jsonl
```

`train.py` reads `data/training`, prints training and validation metrics each
epoch, and saves a checkpoint to `checkpoints/proc_rosetta.pt`. `test.py` loads
that checkpoint and evaluates `data/test` with a human-readable benchmark
report.

Training progress bars and debug messages are printed to stderr, while per-epoch
metrics stay as JSON lines on stdout. Use `./train.py --quiet` to suppress the
debug/progress output.

You can control the generated data and training run:

```bash
./sample.py --train-count 4000 --validation-count 512 --test-count 512 --traces-per-sample 24
./train.py --epochs 30 --batch-size 64 --checkpoint checkpoints/proc_rosetta.pt
./test.py --checkpoint checkpoints/proc_rosetta.pt
```

The `test.py` report includes:

- neural loss metrics on the test split;
- behavioral distance summaries across test logs;
- cross-modal retrieval for the learned tree, trace, and Petri latent vectors;
- nearest-neighbor and distance-correlation statistics for each embedding;
- deterministic event-log baselines: activity counts, trace variants,
  directly-follows, eventually-follows, and pm4py case features;
- deterministic Petri structural-count baselines;
- pm4py's Petri-net Node2Vec/Word2Vec embedding from Colonna et al.,
  "Process mining embeddings: Learning vector representations for Petri nets".
- direct agreement between pm4py's Petri embedding geometry and ProcRosetta's
  fused latent geometry, including pairwise distance Spearman correlation and
  nearest-neighbor overlap.

The pm4py Petri embedding baseline uses `gensim`. Its runtime can be tuned:

```bash
./test.py --petri-embedding-dim 32 --petri-num-walks 3 --petri-walk-length 12 --petri-epochs 3
```

For machine-readable output:

```bash
./test.py --json
```

For package-style installation, the `proc-rosetta sample`, `proc-rosetta train`,
and `proc-rosetta test` console commands remain available after
`python -m pip install -e .`.

## Architecture

```text
traces/logs  -> TraceEncoder ----.
process tree -> TreeEncoder -----+-> shared latent z -> GrammarTreeDecoder
Petri net    -> PetriGraphEncoder'
                                      |
                                      v
                              pm4py tree-to-net
```

The model optimizes tree reconstruction and cross-modal tree translation:

```text
tree  -> z -> tree
trace -> z -> tree
net   -> z -> tree
```

It also adds latent alignment between equivalent modalities. The first version
uses deterministic conversion from process tree to Petri net rather than a
direct Petri-net decoder, which keeps generated models valid for
block-structured behavior.

## Project layout

- `src/proc_rosetta/tree.py`: process-tree data model and canonicalization.
- `src/proc_rosetta/pm4py_bridge.py`: pm4py conversion, simulation, Petri graph
  extraction.
- `src/proc_rosetta/synthetic.py`: synthetic paired triples.
- `src/proc_rosetta/tokenizers.py`: activity/tree tokenization and grammar masks.
- `src/proc_rosetta/data.py`: dataset and batch collation.
- `src/proc_rosetta/models.py`: PyTorch encoders, latent projections, decoder,
  and multimodal model.
- `src/proc_rosetta/losses.py`: reconstruction, cross-modal, KL, and alignment
  losses, including cross-modal contrastive alignment.
- `src/proc_rosetta/behavior.py`: trace-variant, directly-follows, and
  trace-length behavioral distances.
- `src/proc_rosetta/benchmarks.py`: rich test-set embedding comparisons and
  event-log/Petri-net baseline reports.
- `src/proc_rosetta/training.py`: training loop utilities.
- `src/proc_rosetta/cli.py`: sample and train commands.
- `sample.py`: root command for generating synthetic process triples without
  installation.
- `train.py`: root command for training without installation.
- `test.py`: root command for checkpoint evaluation on the test split without
  installation.

## Notes

The embedding is label-name invariant by default for synthetic data: activity
names are canonicalized to `A0`, `A1`, ... while preserving repeated activity
identity. Commutative process-tree operators (`XOR`, `AND`) canonicalize child
order, so equivalent child permutations map to the same structural form.
