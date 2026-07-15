import torch
from torch.utils.data import DataLoader

from proc_rosetta.data import ProcessBatchCollator, SyntheticProcessDataset
from proc_rosetta.losses import (
    cross_modal_contrastive_loss,
    multimodal_tree_loss,
    sequence_cross_entropy,
)
from proc_rosetta.models import LatentDistribution
from proc_rosetta.models import ProcRosettaModel
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer
from proc_rosetta.training import TrainConfig, train_synthetic


def test_model_forward_and_loss():
    synthetic_config = SyntheticConfig(max_depth=2, max_activities=5, traces_per_sample=3)
    tree_tokenizer = TreeTokenizer(max_activities=5, max_arity=3)
    activity_tokenizer = ActivityTokenizer(max_activities=5)
    dataset = SyntheticProcessDataset(3, config=synthetic_config, seed=3)
    collator = ProcessBatchCollator(tree_tokenizer, activity_tokenizer)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collator)))
    model = ProcRosettaModel(tree_tokenizer, activity_tokenizer, latent_dim=16, hidden_dim=32)

    outputs = model(batch, deterministic=True)
    losses = multimodal_tree_loss(outputs, batch["tree_tokens"])

    assert outputs["tree_logits"]["tree"].shape[:2] == batch["tree_tokens"][:, :-1].shape
    assert torch.isfinite(losses["loss"])


def test_label_smoothing_ignores_grammar_masked_logits():
    logits = torch.tensor([[[2.0, -1e9, 0.0], [0.0, 2.0, -1e9]]])
    targets = torch.tensor([[0, 1]])

    loss = sequence_cross_entropy(logits, targets, label_smoothing=0.1)

    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_train_synthetic_smoke():
    train_config = TrainConfig(samples=4, epochs=1, batch_size=2, latent_dim=8, hidden_dim=16, seed=11)
    synthetic_config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=2)

    _, history = train_synthetic(train_config=train_config, synthetic_config=synthetic_config)

    assert len(history) == 1
    assert history[0]["loss"] > 0


def test_multi_positive_contrastive_loss_does_not_make_family_views_negatives():
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    dists = {
        "tree": LatentDistribution(values, torch.zeros_like(values)),
        "petri": LatentDistribution(values, torch.zeros_like(values)),
    }
    diagonal_only = cross_modal_contrastive_loss(dists)
    family_positive = cross_modal_contrastive_loss(
        dists, positive_mask=torch.ones((2, 2), dtype=torch.bool)
    )

    assert diagonal_only > 0
    assert torch.allclose(family_positive, torch.zeros_like(family_positive))


def test_petri_batch_contains_visible_transition_label_ids():
    synthetic_config = SyntheticConfig(
        max_activities=6,
        traces_per_sample=2,
        motif_weights={"duplicate_vs_silent": 1.0},
    )
    tree_tokenizer = TreeTokenizer(max_activities=6, max_arity=3)
    activity_tokenizer = ActivityTokenizer(max_activities=6)
    dataset = SyntheticProcessDataset(2, config=synthetic_config, seed=3)
    batch = ProcessBatchCollator(tree_tokenizer, activity_tokenizer)(dataset.samples)

    assert batch["petri"]["transition_label_ids"].gt(0).any()
    assert batch["positive_mask"].all()
