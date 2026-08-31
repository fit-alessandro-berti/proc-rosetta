import torch
import pytest
from copy import deepcopy
from dataclasses import asdict, replace
from torch.utils.data import DataLoader

from proc_rosetta.data import (
    BatchConfig,
    ProcessBatchCollator,
    SplitCounts,
    SyntheticProcessDataset,
    recreate_data_splits,
)
from proc_rosetta.losses import (
    LossWeights,
    cross_modal_contrastive_loss,
    duplicate_activity_probability_loss,
    expected_tree_complexity_loss,
    hierarchical_metric_loss,
    multimodal_tree_loss,
    positive_latent_alignment_loss,
    semantic_exact_contrastive_loss,
    sequence_policy_feasibility_losses,
    sequence_cross_entropy,
    tree_complexity_token_costs,
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
    _folded_tree_statistics,
    _summarize_discovery_metrics,
    adaptive_scheduled_sampling_probability,
    beam_minimum_risk_loss,
    build_lr_scheduler,
    build_model,
    build_optimizer,
    balanced_validation_components,
    checkpoint_selection_key,
    curriculum_batch_plan,
    final_curriculum_checkpoint_key,
    epoch_snapshot_directory,
    loss_weights_from_checkpoint,
    loss_weights_from_config,
    metric_spec,
    replay_scheduler_history,
    save_epoch_snapshot,
    select_validation_candidate,
    stage_acceptance_report,
    structural_curriculum_for_epoch,
    train_synthetic,
    train_from_data_dir,
    train_config_from_checkpoint,
    validate_resume_configuration,
)
from proc_rosetta.tree import ProcessTreeNode
from types import SimpleNamespace


def test_structural_curriculum_schedule_retains_replay_and_is_deterministic():
    assert structural_curriculum_for_epoch(1, 100)["weights"] == {
        "simple": 1.0,
        "medium": 0.0,
        "complex": 0.0,
    }
    assert structural_curriculum_for_epoch(50, 100)["weights"] == {
        "simple": 0.10,
        "medium": 0.20,
        "complex": 0.70,
    }
    first = curriculum_batch_plan(
        {"simple": 0.10, "medium": 0.20, "complex": 0.70},
        10,
        seed=13,
        epoch=50,
    )
    assert first == curriculum_batch_plan(
        {"simple": 0.10, "medium": 0.20, "complex": 0.70},
        10,
        seed=13,
        epoch=50,
    )
    assert {level: first.count(level) for level in set(first)} == {
        "simple": 1,
        "medium": 2,
        "complex": 7,
    }
    assert structural_curriculum_for_epoch(
        20,
        200,
        minimum_stage="medium",
    )["name"] == "medium"
    assert structural_curriculum_for_epoch(
        50,
        200,
        minimum_stage="complex",
    )["weights"] == {
        "simple": 0.10,
        "medium": 0.20,
        "complex": 0.70,
    }


def test_batched_incremental_beam_matches_individual_rows(monkeypatch):
    torch.manual_seed(17)
    tree_tokenizer = TreeTokenizer(max_activities=4, max_arity=3)
    model = ProcRosettaModel(
        tree_tokenizer,
        ActivityTokenizer(max_activities=4),
        latent_dim=8,
        hidden_dim=16,
        decoder_layers=1,
        dropout=0.0,
        memory_tokens=2,
    ).eval()
    latent = torch.randn(3, 8)
    allowed = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [True, False, False, True],
        ]
    )

    def fail_on_full_prefix(*args, **kwargs):
        raise AssertionError("beam search must use cached incremental decoding")

    monkeypatch.setattr(model.tree_decoder, "next_token_scores", fail_on_full_prefix)
    batched = model.tree_decoder.decode_beam_candidates(
        latent,
        max_length=12,
        beam_size=3,
        allowed_activity_mask=allowed,
        completion_policy="bounded",
    )
    individual = [
        model.tree_decoder.decode_beam_candidates(
            latent[row : row + 1],
            max_length=12,
            beam_size=3,
            allowed_activity_mask=allowed[row : row + 1],
            completion_policy="bounded",
        )[0]
        for row in range(latent.shape[0])
    ]

    assert [[tokens for tokens, _ in rows] for rows in batched] == [
        [tokens for tokens, _ in rows] for rows in individual
    ]
    for batch_rows, individual_rows in zip(batched, individual):
        assert [score for _, score in batch_rows] == pytest.approx(
            [score for _, score in individual_rows], abs=1e-5
        )
        for tokens, _ in batch_rows:
            assert tree_tokenizer.eos_id in tokens


def test_model_forward_and_loss(monkeypatch):
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
    assert set(outputs["contrastive_embeddings"]) == {
        "tree", "trace", "petri", "fused"
    }
    assert set(outputs["decoder_inputs"]) == {"tree", "trace", "petri", "fused"}
    assert outputs["dists"]["fused"].memory is None
    assert torch.allclose(
        outputs["fused_decoder_source"],
        torch.stack(
            [outputs["dists"][name].mu for name in ("tree", "trace", "petri")]
        ).mean(dim=0),
    )
    assert torch.isfinite(losses["fused_to_tree"])
    for name in ("tree", "trace", "petri", "fused"):
        assert torch.equal(
            outputs["decoder_inputs"][name],
            outputs["decoder_targets"][name][:, :-1],
        )

    model.zero_grad(set_to_none=True)
    losses["exact_contrastive"].backward()
    assert any(parameter.grad is not None for parameter in model.contrastive_head.parameters())
    assert all(parameter.grad is None for parameter in model.tree_decoder.parameters())

    model.zero_grad(set_to_none=True)
    decoder_forward_calls = 0
    decoder_inputs_seen = []
    incremental_grad_modes = []
    original_forward = model.tree_decoder.forward
    original_incremental_step = model.tree_decoder._incremental_step

    def counted_forward(*args, **kwargs):
        nonlocal decoder_forward_calls
        decoder_forward_calls += 1
        decoder_inputs_seen.append(args[1].detach().clone())
        return original_forward(*args, **kwargs)

    def checked_incremental_step(*args, **kwargs):
        incremental_grad_modes.append(torch.is_grad_enabled())
        return original_incremental_step(*args, **kwargs)

    monkeypatch.setattr(model.tree_decoder, "forward", counted_forward)
    monkeypatch.setattr(
        model.tree_decoder,
        "_incremental_step",
        checked_incremental_step,
    )
    sampled_outputs = model(
        batch,
        deterministic=True,
        scheduled_sampling_probability=1.0,
    )
    sampled_losses = multimodal_tree_loss(
        outputs=sampled_outputs,
        tree_tokens=batch["tree_tokens"],
    )
    sampled_losses["loss"].backward()

    assert decoder_forward_calls == 2
    assert torch.equal(
        decoder_inputs_seen[0],
        torch.cat(
            [
                sampled_outputs["decoder_inputs"][name]
                for name in ("tree", "trace", "petri")
            ],
            dim=0,
        ),
    )
    assert torch.equal(
        decoder_inputs_seen[1],
        sampled_outputs["decoder_inputs"]["fused"],
    )
    assert incremental_grad_modes and not any(incremental_grad_modes)
    assert all(
        torch.isfinite(logits).any(dim=-1).all()
        for logits in sampled_outputs["tree_logits"].values()
    )
    assert torch.isfinite(sampled_losses["loss"])
    assert any(parameter.grad is not None for parameter in model.tree_decoder.parameters())


def test_fused_reconstruction_backpropagates_to_every_encoder_and_decoder():
    config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=2)
    tokenizer = TreeTokenizer(max_activities=4, max_arity=3)
    activity_tokenizer = ActivityTokenizer(max_activities=4)
    batch = ProcessBatchCollator(tokenizer, activity_tokenizer)(
        SyntheticProcessDataset(2, config=config, seed=19).samples
    )
    model = ProcRosettaModel(
        tokenizer,
        activity_tokenizer,
        latent_dim=8,
        hidden_dim=16,
        decoder_layers=1,
        tree_encoder_layers=1,
        memory_tokens=2,
    )
    fused_loss = multimodal_tree_loss(
        model(batch),
        batch["decoder_targets"],
    )["fused_to_tree"]

    fused_loss.backward()

    for module in (
        model.tree_encoder,
        model.trace_encoder,
        model.petri_encoder,
        model.tree_decoder,
    ):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def _peaked_logits(
    tokenizer: TreeTokenizer,
    token_names: list[str],
) -> torch.Tensor:
    logits = torch.full((1, len(token_names), tokenizer.vocab_size), -20.0)
    for position, name in enumerate(token_names):
        logits[0, position, tokenizer.token_to_id[name]] = 20.0
    return logits


def test_expected_complexity_penalizes_operator_probability_at_leaf_position():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    targets = torch.tensor(
        [[tokenizer.bos_id, tokenizer.token_to_id["A0"], tokenizer.eos_id]]
    )
    leaf_logits = _peaked_logits(tokenizer, ["A0", "<eos>"])
    operator_logits = _peaked_logits(tokenizer, ["SEQ", "<eos>"])

    leaf_loss = expected_tree_complexity_loss(leaf_logits, targets, tokenizer)
    operator_loss = expected_tree_complexity_loss(operator_logits, targets, tokenizer)

    assert operator_loss > leaf_loss


def test_tree_complexity_loss_charges_high_arity_more_than_binary_arity():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    costs = tree_complexity_token_costs(tokenizer, torch.device("cpu"))
    targets = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["SEQ"],
            tokenizer.token_to_id["ARITY_2"],
            tokenizer.token_to_id["A0"],
            tokenizer.token_to_id["A1"],
            tokenizer.eos_id,
        ]]
    )
    binary = _peaked_logits(
        tokenizer,
        ["SEQ", "ARITY_2", "A0", "A1", "<eos>"],
    )
    high_arity = _peaked_logits(
        tokenizer,
        ["SEQ", "ARITY_4", "A0", "A1", "<eos>"],
    )

    assert costs[tokenizer.token_to_id["ARITY_2"]] == 0.0
    assert (
        costs[tokenizer.token_to_id["ARITY_4"]]
        > costs[tokenizer.token_to_id["ARITY_2"]]
    )
    assert expected_tree_complexity_loss(
        high_arity,
        targets,
        tokenizer,
    ) > expected_tree_complexity_loss(binary, targets, tokenizer)


def test_duplicate_loss_ignores_first_occurrence_and_penalizes_repetition():
    tokenizer = TreeTokenizer(max_activities=2)
    targets = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["A0"],
            tokenizer.token_to_id["A1"],
            tokenizer.eos_id,
        ]]
    )
    decoder_inputs = targets[:, :-1]
    first_only = _peaked_logits(tokenizer, ["A0", "A1", "<eos>"])
    repeated = _peaked_logits(tokenizer, ["A0", "A0", "<eos>"])

    first_loss = duplicate_activity_probability_loss(
        first_only,
        decoder_inputs,
        targets,
        tokenizer,
    )
    repeated_loss = duplicate_activity_probability_loss(
        repeated,
        decoder_inputs,
        targets,
        tokenizer,
    )

    assert first_loss == pytest.approx(0.0, abs=1e-8)
    assert repeated_loss > first_loss


def test_intended_duplicate_is_discounted_but_other_repeat_is_not():
    tokenizer = TreeTokenizer(max_activities=2)
    targets = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["A0"],
            tokenizer.token_to_id["A1"],
            tokenizer.token_to_id["A0"],
            tokenizer.eos_id,
        ]]
    )
    decoder_inputs = targets[:, :-1]
    intended = _peaked_logits(tokenizer, ["A0", "A1", "A0", "<eos>"])
    other_repeat = _peaked_logits(tokenizer, ["A0", "A1", "A1", "<eos>"])

    intended_loss = duplicate_activity_probability_loss(
        intended,
        decoder_inputs,
        targets,
        tokenizer,
        required_duplicate_fraction=0.15,
    )
    other_loss = duplicate_activity_probability_loss(
        other_repeat,
        decoder_inputs,
        targets,
        tokenizer,
        required_duplicate_fraction=0.15,
    )

    assert intended_loss == pytest.approx(other_loss.item() * 0.15, rel=1e-5)


def test_duplicate_loss_uses_sampled_decoder_prefixes():
    tokenizer = TreeTokenizer(max_activities=2)
    targets = torch.tensor(
        [[tokenizer.bos_id, tokenizer.token_to_id["A1"], tokenizer.eos_id]]
    )
    teacher_inputs = targets[:, :-1]
    sampled_inputs = teacher_inputs.clone()
    sampled_inputs[0, 0] = tokenizer.token_to_id["A0"]
    logits = _peaked_logits(tokenizer, ["A0", "<eos>"])

    teacher_loss = duplicate_activity_probability_loss(
        logits,
        teacher_inputs,
        targets,
        tokenizer,
    )
    sampled_loss = duplicate_activity_probability_loss(
        logits,
        sampled_inputs,
        targets,
        tokenizer,
    )

    assert sampled_loss > teacher_loss


def test_structure_auxiliary_losses_have_finite_nonzero_logit_gradients():
    torch.manual_seed(7)
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    targets = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["SEQ"],
            tokenizer.token_to_id["ARITY_2"],
            tokenizer.token_to_id["A0"],
            tokenizer.token_to_id["A0"],
            tokenizer.eos_id,
        ]]
    )
    logits = torch.randn(
        1,
        targets.shape[1] - 1,
        tokenizer.vocab_size,
        requires_grad=True,
    )
    complexity = expected_tree_complexity_loss(logits, targets, tokenizer)
    duplicate = duplicate_activity_probability_loss(
        logits,
        targets[:, :-1],
        targets,
        tokenizer,
    )

    complexity_gradient = torch.autograd.grad(
        complexity,
        logits,
        retain_graph=True,
    )[0]
    duplicate_gradient = torch.autograd.grad(duplicate, logits)[0]

    for gradient in (complexity_gradient, duplicate_gradient):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_scheduled_sampling_masks_illegal_targets_and_stops_finished_rows(monkeypatch):
    tree_tokenizer = TreeTokenizer(max_activities=3, max_arity=3)
    model = ProcRosettaModel(
        tree_tokenizer,
        ActivityTokenizer(max_activities=3),
        latent_dim=8,
        hidden_dim=16,
        decoder_layers=1,
    )
    tokens = tree_tokenizer.token_to_id
    stacked_targets = torch.tensor(
        [
            [
                tree_tokenizer.bos_id,
                tokens["SEQ"],
                tokens["ARITY_2"],
                tokens["A0"],
                tokens["A1"],
                tree_tokenizer.eos_id,
            ]
        ]
    )
    source = LatentDistribution(
        mu=torch.zeros(1, 8),
        logvar=torch.zeros(1, 8),
    )
    positions = []

    def forced_incremental_step(input_token, position, memory, layer_caches, **kwargs):
        positions.append(position)
        logits = torch.full(
            (input_token.shape[0], tree_tokenizer.vocab_size),
            -torch.inf,
        )
        if position == 0:
            logits[:, tokens["SEQ"]] = 0.0
            logits[:, tokens["A0"]] = 1.0
        else:
            logits[:, tree_tokenizer.eos_id] = 1.0
        caches = [
            torch.zeros(input_token.shape[0], position + 1, 16)
            for _ in model.tree_decoder.decoder.layers
        ]
        return logits, caches

    monkeypatch.setattr(
        model.tree_decoder,
        "_incremental_step",
        forced_incremental_step,
    )
    sampled_input, loss_targets = model._scheduled_sampling_inputs(
        source,
        stacked_targets[:, :-1],
        None,
        stacked_targets=stacked_targets,
        probability=1.0,
        input_token_dropout=0.0,
    )

    assert positions == [0, 1]
    assert sampled_input.tolist() == [
        [
            tree_tokenizer.bos_id,
            tokens["A0"],
            tree_tokenizer.eos_id,
            tree_tokenizer.pad_id,
            tree_tokenizer.pad_id,
        ]
    ]
    assert loss_targets.tolist() == [
        [
            tree_tokenizer.bos_id,
            tokens["SEQ"],
            tree_tokenizer.pad_id,
            tree_tokenizer.pad_id,
            tree_tokenizer.pad_id,
            tree_tokenizer.pad_id,
        ]
    ]


def test_scheduled_sampling_skips_incremental_decoder_when_no_rows_are_selected(
    monkeypatch,
):
    tree_tokenizer = TreeTokenizer(max_activities=2, max_arity=3)
    model = ProcRosettaModel(
        tree_tokenizer,
        ActivityTokenizer(max_activities=2),
        latent_dim=8,
        hidden_dim=16,
        decoder_layers=1,
    )
    target = torch.tensor(
        [
            tree_tokenizer.encode_tree(ProcessTreeNode.activity("A0")),
            tree_tokenizer.encode_tree(ProcessTreeNode.activity("A1")),
        ]
    )
    source = LatentDistribution(
        mu=torch.zeros(2, 8),
        logvar=torch.zeros(2, 8),
    )
    monkeypatch.setattr(
        torch,
        "rand",
        lambda size, **kwargs: torch.ones(size, device=kwargs.get("device")),
    )
    monkeypatch.setattr(
        model.tree_decoder,
        "_incremental_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("incremental decoder should be skipped")
        ),
    )

    sampled_input, loss_targets = model._scheduled_sampling_inputs(
        source,
        target[:, :-1],
        None,
        stacked_targets=target,
        probability=0.01,
        input_token_dropout=0.0,
    )

    assert torch.equal(sampled_input, target[:, :-1])
    assert torch.equal(loss_targets, target)


def test_beam_risk_caps_rows_and_batches_decode_by_modality(monkeypatch):
    batch_size = 6
    latent_dim = 4
    decode_batch_sizes = []

    class Decoder:
        @staticmethod
        def decode_beam_candidates(source, **kwargs):
            source_tensor = source.mu if isinstance(source, LatentDistribution) else source
            decode_batch_sizes.append(source_tensor.shape[0])
            return [[] for _ in range(source_tensor.shape[0])]

    model = SimpleNamespace(tree_decoder=Decoder())
    distributions = {
        name: LatentDistribution(
            mu=torch.zeros(batch_size, latent_dim, requires_grad=True),
            logvar=torch.zeros(batch_size, latent_dim),
        )
        for name in ("tree", "trace", "petri", "fused")
    }
    batch = {
        "samples": [object() for _ in range(batch_size)],
        "decoder_targets": {
            name: torch.ones(batch_size, 2, dtype=torch.long)
            for name in ("tree", "trace", "petri", "fused")
        },
        "source_activity_masks": {
            name: torch.ones(batch_size, 2, dtype=torch.bool)
            for name in ("tree", "trace", "petri", "fused")
        },
    }

    loss, diagnostics = beam_minimum_risk_loss(
        model,
        {"dists": distributions},
        batch,
        beam_size=3,
        max_decode_length=32,
        max_rows=2,
    )

    assert decode_batch_sizes == [2, 2, 2, 2]
    assert loss == 0.0
    assert all(value == 0.0 for value in diagnostics.values())


def test_resume_allows_and_reports_scheduled_sampling_policy_overrides():
    checkpoint_config = TrainConfig(device="cpu")
    synthetic_config = SyntheticConfig()
    checkpoint = {
        "train_config": asdict(checkpoint_config),
        "synthetic_config": synthetic_config.to_dict(),
    }
    requested = replace(
        checkpoint_config,
        epochs=checkpoint_config.epochs + 1,
        scheduled_sampling_max=0.0,
    )

    overrides = validate_resume_configuration(
        checkpoint,
        requested,
        synthetic_config,
    )

    assert overrides == {
        "scheduled_sampling_max": {
            "checkpoint": checkpoint_config.scheduled_sampling_max,
            "requested": 0.0,
        }
    }
    with pytest.raises(ValueError, match="learning_rate"):
        validate_resume_configuration(
            checkpoint,
            replace(requested, learning_rate=1e-3),
            synthetic_config,
        )


def test_epoch_snapshot_archives_checkpoint_and_matching_metrics(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"epoch-one")
    metrics = checkpoint.parent / "training_metrics.csv"
    history = [
        {
            "epoch": 1,
            "training": {"loss": 1.0},
            "validation": {"loss": 1.5},
            "generalization_gap": {"loss": 0.5},
            "learning_rate": 1e-3,
            "epoch_seconds": 2.0,
            "best_validation_loss": 1.5,
            "is_best": True,
            "epochs_without_improvement": 0,
        }
    ]

    archived_checkpoint, archived_metrics = save_epoch_snapshot(
        checkpoint,
        metrics,
        history,
        epoch=1,
    )

    assert epoch_snapshot_directory(checkpoint, 1) == checkpoint.parent / "00001"
    assert archived_checkpoint == checkpoint.parent / "00001" / "model.pt"
    assert archived_checkpoint.read_bytes() == b"epoch-one"
    assert archived_metrics == checkpoint.parent / "00001" / "training_metrics.csv"
    assert archived_metrics.read_text(encoding="utf-8").splitlines()[1].startswith("1,")

    checkpoint.write_bytes(b"epoch-two")
    second_checkpoint, _ = save_epoch_snapshot(
        checkpoint,
        metrics,
        [*history, {**history[0], "epoch": 2}],
        epoch=2,
    )
    assert second_checkpoint.read_bytes() == b"epoch-two"
    assert archived_checkpoint.read_bytes() == b"epoch-one"
    with pytest.raises(ValueError, match="epoch >= 1"):
        epoch_snapshot_directory(checkpoint, 0)


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


def test_structure_regularization_weights_ramp_after_delayed_start():
    config = TrainConfig(
        tree_complexity_weight=0.03,
        duplicate_activity_weight=0.01,
        structure_regularization_start_epoch=5,
        structure_regularization_ramp_epochs=5,
    )

    before = loss_weights_from_config(config, epoch=4)
    first = loss_weights_from_config(config, epoch=5)
    full = loss_weights_from_config(config, epoch=9)

    assert before.tree_complexity == 0.0
    assert before.duplicate_activity == 0.0
    assert first.tree_complexity == pytest.approx(0.03 / 5)
    assert first.duplicate_activity == pytest.approx(0.01 / 5)
    assert full.tree_complexity == pytest.approx(0.03)
    assert full.duplicate_activity == pytest.approx(0.01)
    with pytest.raises(ValueError, match="tree_complexity_weight"):
        TrainConfig(tree_complexity_weight=-0.01)
    with pytest.raises(ValueError, match="duplicate_activity_weight"):
        TrainConfig(duplicate_activity_weight=-0.01)


def test_zero_structure_weights_preserve_previous_total_exactly():
    torch.manual_seed(21)
    tokenizer = TreeTokenizer(max_activities=2)
    targets = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["A0"],
            tokenizer.eos_id,
        ]]
    )
    embeddings = torch.randn(1, 4)
    dists = {
        name: LatentDistribution(embeddings.clone(), torch.zeros_like(embeddings))
        for name in ("tree", "trace", "petri")
    }
    logits = {
        name: torch.randn(1, 2, tokenizer.vocab_size)
        for name in ("tree", "trace", "petri")
    }
    outputs = {
        "tree_logits": logits,
        "dists": dists,
        "decoder_targets": {name: targets for name in logits},
        "decoder_inputs": {name: targets[:, :-1] for name in logits},
    }
    weights = LossWeights(
        fused_to_tree=0.0,
        fused_subset_to_tree=0.0,
        deployment_to_tree=0.0,
        semantic_exact_contrastive=0.0,
        semantic_memory_contrastive=0.0,
        hierarchical_metric=0.0,
        observed_behavior_regression=0.0,
        observed_behavior_ranking=0.0,
        eos_calibration=0.0,
        generated_length=0.0,
        unresolved_open_slots=0.0,
        completion_feasibility=0.0,
        observation_view_consistency=0.0,
        tree_complexity=0.0,
        duplicate_activity=0.0,
    )

    losses = multimodal_tree_loss(
        outputs,
        targets,
        weights=weights,
        tokenizer=tokenizer,
    )
    legacy_outputs = dict(outputs)
    legacy_outputs.pop("decoder_inputs")
    legacy_losses = multimodal_tree_loss(
        legacy_outputs,
        targets,
        weights=weights,
        tokenizer=tokenizer,
    )
    previous_total = (
        weights.tree_reconstruction * losses["tree_reconstruction"]
        + weights.trace_to_tree * losses["trace_to_tree"]
        + weights.petri_to_tree * losses["petri_to_tree"]
        + weights.exact_contrastive * losses["exact_contrastive"]
        + weights.within_modality_contrastive
        * losses["within_modality_contrastive"]
        + weights.soft_behavior_geometry * losses["soft_behavior_geometry"]
        + weights.variance * losses["variance"]
        + weights.covariance * losses["covariance"]
        + weights.latent_alignment * losses["latent_alignment"]
    )

    assert torch.equal(losses["loss"], previous_total)
    assert torch.equal(
        legacy_losses["duplicate_activity"],
        losses["duplicate_activity"],
    )


def test_policy_feasibility_losses_are_differentiable():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    target = torch.tensor(
        [
            tokenizer.encode_tree(
                ProcessTreeNode.seq(
                    ProcessTreeNode.activity("A0"),
                    ProcessTreeNode.activity("A1"),
                ),
                canonicalize=False,
            )
        ]
    )
    logits = torch.zeros(
        (1, target.shape[1] - 1, tokenizer.vocab_size),
        requires_grad=True,
    )

    unresolved, feasibility = sequence_policy_feasibility_losses(
        {"tree": logits},
        {"tree": target[:, :-1]},
        {"tree": target},
        tokenizer,
    )
    (unresolved + feasibility).backward()

    assert unresolved >= 0.0
    assert feasibility > 0.0
    assert logits.grad is not None and logits.grad.abs().sum() > 0.0


def test_policy_feasibility_penalizes_sampled_prefix_that_exhausts_budget():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    tokens = tokenizer.token_to_id
    target = torch.tensor(
        [
            [
                tokenizer.bos_id,
                tokens["SEQ"],
                tokens["ARITY_2"],
                tokens["A0"],
                tokens["A1"],
                tokenizer.eos_id,
            ]
        ]
    )
    sampled_input = torch.tensor(
        [
            [
                tokenizer.bos_id,
                tokens["SEQ"],
                tokens["ARITY_4"],
                tokens["A0"],
                tokens["A1"],
            ]
        ]
    )
    logits = torch.zeros(
        (1, sampled_input.shape[1], tokenizer.vocab_size),
        requires_grad=True,
    )

    unresolved, feasibility = sequence_policy_feasibility_losses(
        {"tree": logits},
        {"tree": sampled_input},
        {"tree": target},
        tokenizer,
    )
    (unresolved + feasibility).backward()

    assert torch.isfinite(feasibility)
    assert feasibility > 0.0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_policy_feasibility_uses_unmasked_targets_for_completion_budget():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    tokens = tokenizer.token_to_id
    target = torch.tensor(
        [
            [
                tokenizer.bos_id,
                tokens["SEQ"],
                tokens["ARITY_2"],
                tokens["A0"],
                tokens["A1"],
                tokenizer.eos_id,
            ]
        ]
    )
    loss_target = target.clone()
    loss_target[0, 3] = tokenizer.pad_id
    logits = torch.full(
        (1, target.shape[1] - 1, tokenizer.vocab_size),
        -torch.inf,
    )
    for position, token_id in enumerate(target[0, 1:]):
        logits[0, position, token_id] = 0.0

    _, feasibility = sequence_policy_feasibility_losses(
        {"tree": logits},
        {"tree": target[:, :-1]},
        {"tree": loss_target},
        tokenizer,
        budget_targets={"tree": target},
    )

    assert feasibility == 0.0


def test_scheduled_sampling_adapts_to_raw_decode_exposure_gap():
    config = TrainConfig(
        device="cpu",
        scheduled_sampling_max=0.20,
        scheduled_sampling_start_epoch=5,
        scheduled_sampling_ramp_epochs=10,
    )
    metrics = {
        "trace_to_tree": 0.05,
        "decode_quality": {
            "methods": {
                "proc_rosetta_trace_mu": {
                    "terminated_rate": 0.25,
                    "mean_normalized_token_edit_distance": 0.75,
                }
            }
        },
    }

    baseline = adaptive_scheduled_sampling_probability(config, 6, None)
    adapted = adaptive_scheduled_sampling_probability(config, 6, metrics)

    assert baseline < adapted <= config.scheduled_sampling_max


def test_unbounded_diagnostics_do_not_influence_checkpoint_selection():
    deployment = {
        "methods": {
            "proc_rosetta_trace_mu": {
                "terminated_rate": 1.0,
                "valid_tree_rate": 1.0,
                "exact_tree_match_rate": 0.5,
                "petri_conversion_rate": 1.0,
                "behavior_eval_success_rate": 1.0,
                "mean_behavior_l1": 0.2,
                "mean_normalized_token_edit_distance": 0.25,
            }
        }
    }
    poor_diagnostic = {
        "methods": {
            "proc_rosetta_trace_mu": {
                "terminated_rate": 0.0,
                "valid_tree_rate": 0.0,
                "exact_tree_match_rate": 0.0,
                "petri_conversion_rate": 0.0,
                "behavior_eval_success_rate": 0.0,
                "mean_behavior_l1": 2.0,
                "mean_normalized_token_edit_distance": 1.0,
            }
        }
    }
    strong_diagnostic = deepcopy(deployment)

    poor = balanced_validation_components(
        {
            "deployment_decode_quality": deployment,
            "diagnostic_unbounded_decode_quality": poor_diagnostic,
        }
    )
    strong = balanced_validation_components(
        {
            "deployment_decode_quality": deployment,
            "diagnostic_unbounded_decode_quality": strong_diagnostic,
        }
    )

    assert poor["decode_score"] == strong["decode_score"]
    assert poor["balanced_score"] == strong["balanced_score"]
    assert poor["all_hard_gates_pass"] == strong["all_hard_gates_pass"]


def test_legacy_checkpoint_defaults_new_structure_objectives_to_zero():
    legacy_config = asdict(TrainConfig(device="cpu"))
    for name in (
        "tree_complexity_weight",
        "duplicate_activity_weight",
        "structure_regularization_start_epoch",
        "structure_regularization_ramp_epochs",
    ):
        legacy_config.pop(name)
    checkpoint = {
        "train_config": legacy_config,
        "loss_weights": {
            "tree_reconstruction": 0.5,
            "trace_to_tree": 2.0,
            "petri_to_tree": 0.5,
        },
    }

    restored_config = train_config_from_checkpoint(checkpoint, "cpu")
    restored_weights = loss_weights_from_checkpoint(checkpoint, restored_config)

    assert restored_config.tree_complexity_weight == 0.0
    assert restored_config.duplicate_activity_weight == 0.0
    assert restored_weights.tree_complexity == 0.0
    assert restored_weights.duplicate_activity == 0.0


def test_discovery_structure_metrics_use_folded_decoded_trees():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=3)
    target_ids = tokenizer.encode_tree(
        ProcessTreeNode.seq(
            ProcessTreeNode.activity("A0"),
            ProcessTreeNode.activity("A1"),
        ),
        canonicalize=False,
    )
    decoded_ids = tokenizer.encode_tree(
        ProcessTreeNode.seq(
            ProcessTreeNode.activity("A0"),
            ProcessTreeNode.activity("A1"),
            ProcessTreeNode.activity("A0"),
        ),
        canonicalize=False,
    )
    target = _folded_tree_statistics(target_ids, tokenizer)
    decoded = _folded_tree_statistics(decoded_ids, tokenizer)
    assert target is not None and decoded is not None
    row = {
        "motif": "ordinary_tree",
        "ordinary": True,
        "loop": False,
        "exact": False,
        "normalized_edit": 0.25,
        "raw_exact": False,
        "raw_normalized_edit": 0.25,
        "deployment_exact": False,
        "deployment_normalized_edit": 0.25,
        "source_ablation": False,
        "token_accuracy": {
            "operator": (1, 1),
            "arity": (1, 1),
            "activity_copy": (2, 2),
        },
        "target_size": target[0],
        "target_depth": target[1],
        "target_duplicate_count": target[2],
        "decoded_size": decoded[0],
        "decoded_depth": decoded[1],
        "decoded_duplicate_count": decoded[2],
        "decoded_has_duplicates": decoded[3],
    }

    metrics = _summarize_discovery_metrics([row], {}, [], [], [])

    assert metrics["trace_decoded_mean_size"] == 4.0
    assert metrics["trace_decoded_mean_depth"] == 2.0
    assert metrics["trace_decoded_mean_duplicate_count"] == 1.0
    assert metrics["trace_decoded_duplicate_rate"] == 1.0
    assert metrics["trace_decoded_mean_size_delta"] == 1.0
    assert metrics["trace_decoded_mean_depth_delta"] == 0.0
    assert metrics["trace_decoded_mean_duplicate_count_delta"] == 1.0


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


def test_checkpoint_selection_uses_balanced_quality_before_loss():
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

    assert checkpoint_selection_key(lower_exact) > checkpoint_selection_key(baseline)
    assert checkpoint_selection_key(tied_exact_better_edit) < checkpoint_selection_key(
        baseline
    )
    legacy_metrics = {
        "ordinary_trace_canonical_exact": 0.75,
        "trace_normalized_tree_edit": 0.20,
        "exact_behavior_recall_at_1": 0.60,
        "behavior_distance_spearman": 0.40,
    }
    assert checkpoint_selection_key(legacy_metrics)[1] == pytest.approx(
        balanced_validation_components(legacy_metrics)["balanced_score"]
    )


def test_ordinary_and_ema_use_the_same_balanced_selection_key():
    ordinary = {
        "checkpoint_selection_primary_exact": 0.70,
        "checkpoint_selection_edit_score": 0.70,
        "checkpoint_selection_recall_at_1": 0.70,
        "checkpoint_selection_spearman": 0.20,
        "loss": 0.1,
    }
    ema = {
        **ordinary,
        "checkpoint_selection_primary_exact": 0.80,
        "checkpoint_selection_recall_at_1": 0.80,
        "loss": 1.0,
    }

    selected, weight_kind = select_validation_candidate(ordinary, ema)

    assert selected is ema
    assert weight_kind == "ema"
    assert checkpoint_selection_key(ema) > checkpoint_selection_key(ordinary)


@pytest.mark.parametrize(
    "name",
    ("mean_behavior_l1", "mean_rank", "nearest_neighbor_behavior_l1"),
)
def test_metric_registry_marks_distance_and_rank_metrics_lower_is_better(name):
    assert metric_spec(name).direction == "min"


def test_final_curriculum_checkpoint_uses_robust_macro_and_regression_gate():
    complex_metrics = {
        "checkpoint_selection_primary_exact": 0.80,
        "checkpoint_selection_edit_score": 0.70,
        "checkpoint_selection_recall_at_1": 0.60,
        "checkpoint_selection_spearman": 0.50,
    }
    macro_metrics = {
        "checkpoint_selection_primary_exact": 0.75,
        "checkpoint_selection_edit_score": 0.65,
        "checkpoint_selection_recall_at_1": 0.55,
        "checkpoint_selection_spearman": 0.45,
    }
    better_macro = {**macro_metrics, "checkpoint_selection_primary_exact": 0.76}
    worse_complex = {**complex_metrics, "checkpoint_selection_primary_exact": 0.79}

    baseline = final_curriculum_checkpoint_key(complex_metrics, macro_metrics)
    assert final_curriculum_checkpoint_key(complex_metrics, better_macro) > baseline
    assert final_curriculum_checkpoint_key(worse_complex, better_macro) > baseline
    assert final_curriculum_checkpoint_key(
        complex_metrics,
        better_macro,
        regression_within_tolerance=False,
    )[0] == 0.0


def test_lr_scheduler_tracks_balanced_validation_score():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    config = TrainConfig(lr_patience=0, lr_factor=0.5)
    scheduler = build_lr_scheduler(optimizer, config)

    history = [
        {
            "validation": {
                "loss": 1.0,
                "trace_canonical_exact": 0.1,
                "trace_normalized_tree_edit": 0.9,
                "exact_behavior_recall_at_1": 0.1,
                "behavior_distance_spearman": -0.8,
            }
        },
        {
            "validation": {
                "loss": 1.1,
                "trace_canonical_exact": 0.9,
                "trace_normalized_tree_edit": 0.1,
                "exact_behavior_recall_at_1": 0.9,
                "behavior_distance_spearman": 0.8,
            }
        },
    ]
    replay_scheduler_history(scheduler, history)

    assert scheduler.mode == "max"
    assert scheduler.threshold == config.min_delta
    assert scheduler.threshold_mode == "abs"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


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
            validation_audit_enabled=False,
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


def test_semantic_retrieval_loss_operates_without_projection_head():
    semantic = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]],
        requires_grad=True,
    )
    dists = {
        name: LatentDistribution(semantic, torch.zeros_like(semantic))
        for name in ("tree", "trace", "petri")
    }
    exact = torch.tensor(
        [[True, True, False], [True, True, False], [False, False, True]]
    )
    candidates = torch.ones_like(exact)

    loss = semantic_exact_contrastive_loss(
        dists,
        exact,
        candidates,
        temperature=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert semantic.grad is not None and semantic.grad.abs().sum() > 0


def test_hierarchical_metric_consumes_analogy_mask_and_has_finite_gradients():
    embedding = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    dists = {"tree": LatentDistribution(embedding, torch.zeros_like(embedding))}
    masks = []
    for index in range(4):
        mask = torch.zeros((5, 5), dtype=torch.bool)
        mask[0, index + 1] = mask[index + 1, 0] = True
        masks.append(mask)

    loss = hierarchical_metric_loss(dists, *masks)
    loss.backward()

    assert torch.isfinite(loss) and loss > 0
    assert embedding.grad is not None and torch.isfinite(embedding.grad).all()


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

    assert 120 <= len(first_batch) <= 128
    assert len({samples[index].equivalence_id for index in first_batch}) >= 25


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
    mixed_relation_batch = False
    for batch in batches:
        partial_counts = {
            partial: sum(samples[index].partial_order_id == partial for index in batch)
            for partial in {samples[index].partial_order_id for index in batch}
        }
        assert max(partial_counts.values()) >= 2
        mixed_relation_batch |= len(partial_counts) > 1
    assert mixed_relation_batch


def test_relation_aware_sampler_length_matches_expanded_batches():
    samples = [
        SimpleNamespace(
            equivalence_id=f"family-{family}",
            exact_behavior_id=f"exact-{family}-{variant // 2}",
            partial_order_id=f"partial-{variant % 2}",
        )
        for family in range(6)
        for variant in range(4)
    ]
    sampler = BehaviorFamilyBatchSampler(
        samples,
        batch_size=5,
        views_per_family=2,
        shuffle=True,
    )

    expected_length = len(sampler)
    batches = list(sampler)

    assert expected_length == len(batches)
    assert all(1 <= len(batch) <= 5 for batch in batches)


def test_signature_hard_negative_search_scales_without_quadratic_matrix():
    samples = [
        SimpleNamespace(
            equivalence_id=f"family-{index // 4}",
            behavior_signature=tuple(
                ((index * 17 + dimension * 13) % 101) / 101.0
                for dimension in range(16)
            ),
        )
        for index in range(4096)
    ]

    sampler = BehaviorFamilyBatchSampler(
        samples,
        batch_size=128,
        views_per_family=2,
        shuffle=False,
    )

    assert len(sampler.hard_negatives) == len(samples)
    assert all(
        negative is not None
        and samples[index].equivalence_id != samples[negative].equivalence_id
        for index, negative in enumerate(sampler.hard_negatives)
    )


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
