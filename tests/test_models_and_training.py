import torch
import pytest
from dataclasses import replace
from torch.utils.data import DataLoader

from proc_rosetta.data import (
    BatchConfig,
    ProcessBatchCollator,
    SplitCounts,
    SyntheticProcessDataset,
    recreate_data_splits,
)
from proc_rosetta.losses import (
    cross_modal_contrastive_loss,
    multimodal_tree_loss,
    positive_latent_alignment_loss,
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
    build_lr_scheduler,
    build_model,
    build_optimizer,
    checkpoint_selection_key,
    loss_weights_from_checkpoint,
    loss_weights_from_config,
    replay_scheduler_history,
    stage_acceptance_report,
    train_synthetic,
    train_from_data_dir,
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
    assert outputs["dists"]["trace"].memory.shape == (3, 6, 32)
    assert outputs["dists"]["trace"].activity_memory.shape == (3, 5, 32)
    assert outputs["dists"]["tree"].activity_memory.shape == (3, 5, 32)
    assert outputs["dists"]["petri"].activity_memory.shape == (3, 5, 32)
    assert torch.allclose(
        outputs["dists"]["trace"].sample(deterministic=False),
        outputs["dists"]["trace"].mu,
    )
    assert torch.isfinite(losses["loss"])
    assert set(outputs["contrastive_embeddings"]) == {"tree", "trace", "petri"}

    model.zero_grad(set_to_none=True)
    losses["exact_contrastive"].backward()
    assert any(parameter.grad is not None for parameter in model.contrastive_head.parameters())
    assert all(parameter.grad is None for parameter in model.tree_decoder.parameters())

    model.zero_grad(set_to_none=True)
    sampled_outputs = model(
        batch,
        deterministic=True,
        scheduled_sampling_probability=1.0,
    )
    assert all(
        torch.isfinite(logits).any(dim=-1).all()
        for logits in sampled_outputs["tree_logits"].values()
    )


def test_positive_latent_alignment_is_real_and_uses_family_views():
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    swapped = values.flip(0)
    dists = {
        "tree": LatentDistribution(values, torch.zeros_like(values)),
        "trace": LatentDistribution(swapped, torch.zeros_like(swapped)),
    }

    diagonal = positive_latent_alignment_loss(dists)
    all_views = positive_latent_alignment_loss(
        dists, torch.ones((2, 2), dtype=torch.bool)
    )

    assert diagonal == pytest.approx(1.0)
    assert all_views == pytest.approx(0.5)


def test_metric_objective_ramps_replace_hard_stage_switches():
    config = TrainConfig()

    epoch_two = loss_weights_from_config(config, epoch=2)
    epoch_three = loss_weights_from_config(config, epoch=3)
    epoch_six = loss_weights_from_config(config, epoch=6)
    epoch_ten = loss_weights_from_config(config, epoch=10)

    assert epoch_two.exact_contrastive == 0.0
    assert epoch_three.exact_contrastive == pytest.approx(0.30 / 4)
    assert epoch_six.exact_contrastive == pytest.approx(0.30)
    assert epoch_three.soft_behavior_geometry == 0.0
    assert epoch_ten.soft_behavior_geometry == pytest.approx(0.25)


def test_default_capacity_and_adamw_parameter_groups_are_regularized():
    config = TrainConfig(device="cpu")
    model = build_model(config, SyntheticConfig(), torch.device("cpu"))
    optimizer = build_optimizer(model, config)

    assert sum(parameter.numel() for parameter in model.parameters()) < 7_500_000
    assert config.hidden_dim == 192
    assert config.latent_dim == 96
    assert config.tree_encoder_layers == 3
    assert config.trace_event_layers == 1
    assert config.trace_set_layers == 1
    assert {group["weight_decay"] for group in optimizer.param_groups} == {
        0.0,
        config.weight_decay,
    }
    no_decay_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for parameter in group["params"]
    }
    assert all(
        id(parameter) in no_decay_ids
        for name, parameter in model.named_parameters()
        if name.endswith(".bias") or parameter.ndim < 2
    )


def test_label_smoothing_ignores_grammar_masked_logits():
    logits = torch.tensor([[[2.0, -1e9, 0.0], [0.0, 2.0, -1e9]]])
    targets = torch.tensor([[0, 1]])

    loss = sequence_cross_entropy(logits, targets, label_smoothing=0.1)

    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_checkpoint_selection_is_strictly_lexicographic():
    baseline = {
        "checkpoint_selection_primary_exact": 0.90,
        "checkpoint_selection_edit_score": 0.90,
        "checkpoint_selection_recall_at_1": 0.90,
        "checkpoint_selection_spearman": 0.90,
    }
    lower_exact = {**baseline, "checkpoint_selection_primary_exact": 0.8999999}
    lower_exact.update(
        checkpoint_selection_edit_score=1.0,
        checkpoint_selection_recall_at_1=1.0,
        checkpoint_selection_spearman=1.0,
    )
    tied_exact_better_edit = {
        **baseline,
        "checkpoint_selection_edit_score": 0.91,
        "checkpoint_selection_recall_at_1": 0.0,
        "checkpoint_selection_spearman": -1.0,
    }

    assert checkpoint_selection_key(lower_exact) < checkpoint_selection_key(baseline)
    assert checkpoint_selection_key(tied_exact_better_edit) > checkpoint_selection_key(
        baseline
    )
    legacy_metrics = {
        "ordinary_trace_canonical_exact": 0.75,
        "trace_normalized_tree_edit": 0.20,
        "exact_behavior_recall_at_1": 0.60,
        "behavior_distance_spearman": 0.40,
    }
    assert checkpoint_selection_key(legacy_metrics) == (0.75, 0.80, 0.60, 0.40)


def test_lr_scheduler_tracks_validation_loss_instead_of_discovery_score():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    config = TrainConfig(lr_patience=0, lr_factor=0.5)
    scheduler = build_lr_scheduler(optimizer, config)

    history = [
        {
            "validation": {
                "loss": 1.0,
                "checkpoint_selection_score": 0.1,
            }
        },
        {
            "validation": {
                "loss": 1.1,
                "checkpoint_selection_score": 0.9,
            }
        },
    ]
    replay_scheduler_history(scheduler, history)

    assert scheduler.mode == "min"
    assert scheduler.threshold == config.min_delta
    assert scheduler.threshold_mode == "abs"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_train_synthetic_smoke():
    train_config = TrainConfig(samples=4, epochs=1, batch_size=2, latent_dim=8, hidden_dim=16, seed=11)
    synthetic_config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=2)

    _, history = train_synthetic(train_config=train_config, synthetic_config=synthetic_config)

    assert len(history) == 1
    assert history[0]["loss"] > 0


def test_train_from_data_dir_restores_best_weights_before_return(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    checkpoint = tmp_path / "model.pt"
    recreate_data_splits(
        data_dir,
        counts=SplitCounts(training=1, validation=1, test=1),
        config=SyntheticConfig(
            generator="isolated",
            max_depth=2,
            max_activities=4,
            min_activities=1,
            traces_per_sample=1,
            class_coverage_mode="best_effort",
        ),
        show_progress=False,
    )

    def fake_train_epoch(model, *args, epoch=None, **kwargs):
        with torch.no_grad():
            next(model.parameters()).add_(float(epoch))
        return {
            "loss": float(epoch),
            "tree_reconstruction": 0.1,
            "trace_to_tree": 0.2,
            "petri_to_tree": 0.1,
        }

    def fake_evaluate_epoch(model, *args, epoch=None, **kwargs):
        value = float(epoch)
        return {
            "loss": value,
            "tree_reconstruction": value,
            "trace_to_tree": value,
            "petri_to_tree": value,
            "trace_normalized_tree_edit": value / 10.0,
            "exact_behavior_recall_at_1": 1.0 / value,
            "behavior_distance_spearman": 1.0 / value,
            "effective_rank": 1.0 / value,
            "trace_canonical_exact": 1.0 / value,
            "checkpoint_selection_score": 1.0 / value,
            "checkpoint_selection_primary_exact": 1.0 / value,
            "checkpoint_selection_edit_score": 1.0 - value / 10.0,
            "checkpoint_selection_recall_at_1": 1.0 / value,
            "checkpoint_selection_spearman": 1.0 / value,
        }

    monkeypatch.setattr("proc_rosetta.training.train_epoch", fake_train_epoch)
    monkeypatch.setattr("proc_rosetta.training.evaluate_epoch", fake_evaluate_epoch)
    model, history = train_from_data_dir(
        data_dir=data_dir,
        checkpoint_path=checkpoint,
        metrics_csv_path=tmp_path / "metrics.csv",
        train_config=TrainConfig(
            epochs=2,
            batch_size=1,
            hidden_dim=16,
            latent_dim=8,
            memory_tokens=2,
            decoder_layers=1,
            tree_encoder_layers=1,
            trace_event_layers=1,
            trace_set_layers=1,
            petri_message_passing_steps=1,
            use_ema=False,
            device="cpu",
        ),
        show_progress=False,
    )

    best = torch.load(
        checkpoint.with_name("model.best.pt"),
        map_location="cpu",
        weights_only=False,
    )["model_state_dict"]
    latest = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    returned = model.state_dict()
    assert len(history) == 2
    assert all(torch.equal(returned[name], best[name]) for name in returned)
    assert any(not torch.equal(returned[name], latest[name]) for name in returned)


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


def test_batch_sampler_groups_exact_partial_order_positives_without_orphans():
    samples = [
        SimpleNamespace(
            equivalence_id="broad-family",
            exact_behavior_id="exact",
            partial_order_id=partial,
        )
        for partial, count in (("concurrent", 5), ("interleaved", 4))
        for _ in range(count)
    ]
    sampler = BehaviorFamilyBatchSampler(
        samples,
        batch_size=5,
        views_per_family=4,
        shuffle=False,
    )

    batches = list(sampler)
    for batch in batches:
        partial_counts = {
            partial: sum(samples[index].partial_order_id == partial for index in batch)
            for partial in {samples[index].partial_order_id for index in batch}
        }
        assert all(count >= 2 for count in partial_counts.values())


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
