import torch
import pytest
from dataclasses import replace
from torch.utils.data import DataLoader

from proc_rosetta.data import BatchConfig, ProcessBatchCollator, SyntheticProcessDataset
from proc_rosetta.losses import (
    cross_modal_contrastive_loss,
    multimodal_tree_loss,
    sequence_cross_entropy,
)
from proc_rosetta.models import LatentDistribution
from proc_rosetta.models import ProcRosettaModel
from proc_rosetta.families import flatten_behavior_family, generate_behavior_family
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer
from proc_rosetta.training import (
    BehaviorFamilyBatchSampler,
    TrainConfig,
    _different_behavior_permutation,
    loss_weights_from_checkpoint,
    stage_acceptance_report,
    train_synthetic,
)
from types import SimpleNamespace


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
    assert outputs["dists"]["trace"].memory.shape == (3, 8, 32)
    assert outputs["dists"]["trace"].activity_memory.shape == (3, 5, 32)
    assert outputs["dists"]["tree"].activity_memory.shape == (3, 5, 32)
    assert outputs["dists"]["petri"].activity_memory.shape == (3, 5, 32)
    assert torch.allclose(
        outputs["dists"]["trace"].sample(deterministic=False),
        outputs["dists"]["trace"].mu,
    )
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
    # Every positive contributes its own log probability; two equally likely
    # positives therefore produce the irreducible log(2) normalization term.
    assert torch.allclose(family_positive, torch.log(torch.tensor(2.0)))


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


def test_strict_trace_collation_rejects_silent_prefix_truncation():
    config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=1)
    sample = SyntheticProcessDataset(1, config=config, seed=3).samples[0]
    too_long = replace(sample, traces=(("A0",) * 129,))
    collator = ProcessBatchCollator(
        TreeTokenizer(max_activities=4),
        ActivityTokenizer(max_activities=4),
        BatchConfig(max_trace_length=128, strict_trace_lengths=True),
    )

    with pytest.raises(ValueError, match="strict mode"):
        collator([too_long])


def test_strict_trace_collation_rejects_silent_trace_set_truncation():
    config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=1)
    sample = SyntheticProcessDataset(1, config=config, seed=3).samples[0]
    too_many = replace(sample, traces=tuple(("A0",) for _ in range(129)))
    collator = ProcessBatchCollator(
        TreeTokenizer(max_activities=4),
        ActivityTokenizer(max_activities=4),
        BatchConfig(max_traces=128, strict_trace_lengths=True),
    )

    with pytest.raises(ValueError, match="trace-set size"):
        collator([too_many])


def test_strong_positive_mask_distinguishes_partial_order_semantics():
    config = SyntheticConfig(
        max_activities=8,
        traces_per_sample=2,
        log_views_per_behavior=2,
        motif_weights={"concurrent_vs_interleaved": 1.0},
    )
    samples = flatten_behavior_family(
        generate_behavior_family(
            config,
            0,
            seed=9,
            split="training",
            motif="concurrent_vs_interleaved",
        ),
        max_activities=8,
    )
    batch = ProcessBatchCollator(
        TreeTokenizer(max_activities=8), ActivityTokenizer(max_activities=8)
    )(samples)

    assert len(set(batch["exact_behavior_ids"])) == 1
    left = next(index for index, sample in enumerate(samples) if sample.partial_order_id.endswith("concurrent"))
    right = next(index for index, sample in enumerate(samples) if sample.partial_order_id.endswith("interleaved"))
    assert not batch["positive_mask"][left, right]
    assert batch["analogy_mask"][left, right]
    assert not batch["contrastive_candidate_mask"][left, right]
    signature_cosine = torch.nn.functional.cosine_similarity(
        batch["behavior_signatures"][left].unsqueeze(0),
        batch["behavior_signatures"][right].unsqueeze(0),
    ).item()
    assert 0.8 < signature_cosine < 0.999


def test_effective_contrastive_batch_contains_32_distinct_behaviors():
    samples = [
        SimpleNamespace(equivalence_id=f"behavior-{family}")
        for family in range(40)
        for _ in range(8)
    ]
    sampler = BehaviorFamilyBatchSampler(
        samples,
        batch_size=128,
        views_per_family=4,
        shuffle=False,
    )
    first_batch = next(iter(sampler))

    assert len(first_batch) == 128
    assert len({samples[index].equivalence_id for index in first_batch}) == 32


def test_source_ablation_deranges_adjacent_views_of_the_same_behavior():
    identifiers = [f"behavior-{family}" for family in range(4) for _ in range(2)]

    permutation = _different_behavior_permutation(
        identifiers, len(identifiers), torch.device("cpu")
    )

    assert all(
        identifiers[row] != identifiers[int(permutation[row])]
        for row in range(len(identifiers))
    )


def test_stage_a_acceptance_retains_auditable_measurements():
    report = stage_acceptance_report(
        "a",
        {
            "trace_canonical_exact": 0.96,
            "shuffled_trace_canonical_exact": 0.02,
            "zero_trace_canonical_exact": 0.0,
        },
        semantic_dimension=128,
    )

    assert report["evaluated"] is True
    assert report["passed"] is True
    assert report["measurements"]["training_trace_canonical_exact"] == 0.96


def test_checkpoint_loss_payload_is_authoritative():
    config = TrainConfig(trace_to_tree_weight=2.0)

    restored = loss_weights_from_checkpoint(
        {"loss_weights": {"trace_to_tree": 7.5, "label_smoothing": 0.02}},
        config,
    )

    assert restored.trace_to_tree == 7.5
    assert restored.label_smoothing == 0.02


def test_activity_remapping_is_semantics_preserving_and_family_consistent():
    synthetic_config = SyntheticConfig(
        max_activities=6,
        traces_per_sample=2,
        motif_weights={"duplicate_vs_silent": 1.0},
    )
    tree_tokenizer = TreeTokenizer(max_activities=6, max_arity=3)
    activity_tokenizer = ActivityTokenizer(max_activities=6)
    dataset = SyntheticProcessDataset(2, config=synthetic_config, seed=3)
    plain = ProcessBatchCollator(tree_tokenizer, activity_tokenizer)(dataset.samples)
    augmented = ProcessBatchCollator(
        tree_tokenizer,
        activity_tokenizer,
        activity_remap_probability=1.0,
        seed=7,
    )(dataset.samples)

    assert augmented["positive_mask"].all()
    assert not torch.equal(plain["tree_tokens"], augmented["tree_tokens"])
    assert torch.equal(augmented["tree_tokens"][0], augmented["tree_tokens"][1])
    assert torch.equal(
        augmented["traces"]["tokens"][0], augmented["traces"]["tokens"][1]
    )
    for row in range(2):
        tree_ids = set(augmented["tree_tokens"][row].tolist())
        visible_activity_ids = set(
            augmented["traces"]["tokens"][row]
            [augmented["traces"]["tokens"][row].gt(0)]
            .tolist()
        )
        for activity_id in visible_activity_ids:
            activity_name = activity_tokenizer.tokens[activity_id]
            assert tree_tokenizer.token_to_id[activity_name] in tree_ids
        tree_tokenizer.decode_tree(augmented["tree_tokens"][row].tolist())
