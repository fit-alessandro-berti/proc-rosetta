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

```bash
python -m pip install -e ".[dev]"
proc-rosetta sample --count 2 --traces-per-sample 4 --seed 7
proc-rosetta train --samples 32 --epochs 1 --batch-size 8 --traces-per-sample 6
```

Or run without installing:

```bash
PYTHONPATH=src python -m proc_rosetta.cli sample --count 1
PYTHONPATH=src python -m proc_rosetta.cli train --samples 16 --epochs 1
```

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
- `src/proc_rosetta/training.py`: training loop utilities.
- `src/proc_rosetta/cli.py`: sample and train commands.

## Notes

The embedding is label-name invariant by default for synthetic data: activity
names are canonicalized to `A0`, `A1`, ... while preserving repeated activity
identity. Commutative process-tree operators (`XOR`, `AND`) canonicalize child
order, so equivalent child permutations map to the same structural form.
