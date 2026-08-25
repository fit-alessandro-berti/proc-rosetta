from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import random
from types import SimpleNamespace

import pytest
import torch

from proc_rosetta.artifact_io import (
    ArtifactModality,
    parse_artifact,
    prepare_artifact_for_model,
)
from proc_rosetta.data import ProcessBatchCollator
from proc_rosetta.inference import (
    ArtifactEncodingResult,
    build_decode_result,
    combine_encoding_decode_evidence,
    decode_guaranteed,
    decode_latent_iter,
    petri_graph_to_tensors,
    simulate_decoded_behavior,
)
from proc_rosetta.losses import LossWeights
from proc_rosetta.models import (
    DecodeConstraints,
    LatentDistribution,
    PetriGraphEncoder,
    ProcRosettaModel,
)
from proc_rosetta.pm4py_bridge import (
    fold_process_tree,
    prepare_tree_for_model,
    simulate_traces,
    tree_to_petri_net,
)
from proc_rosetta.synthetic import ProcessSample, decoder_target_trees_for_sample
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer
from proc_rosetta.training import TrainConfig, gradient_norm_diagnostics
from proc_rosetta.tree import ProcessTreeNode, sanitize_activity_labels
from proc_rosetta_ui.decoding_service import _decode_cache_key
from proc_rosetta_ui.export_service import process_tree_ptml
from scripts._common import (
    add_decode_constraint_arguments,
    decode_tree_from_latent,
    encode_traces_distribution,
)


def _model(max_activities: int = 4) -> ProcRosettaModel:
    return ProcRosettaModel(
        TreeTokenizer(max_activities=max_activities, max_arity=3),
        ActivityTokenizer(max_activities=max_activities),
        latent_dim=8,
        hidden_dim=16,
        decoder_layers=1,
        dropout=0.0,
        memory_tokens=2,
    ).eval()


def test_ptml_is_folded_before_local_tree_validation(tmp_path):
    import pm4py
    from pm4py.objects.process_tree.obj import Operator, ProcessTree

    root = ProcessTree(operator=Operator.SEQUENCE)
    child = ProcessTree(label="only")
    child.parent = root
    root.children = [child]
    path = tmp_path / "unary.ptml"
    pm4py.write_ptml(root, str(path))

    parsed = parse_artifact(path)

    assert parsed.tree == ProcessTreeNode.activity("only")
    assert parsed.source_metadata["source_tree_size_before_fold"] == 2
    assert parsed.source_metadata["source_tree_size_after_fold"] == 1
    assert parsed.source_metadata["fold_changed"] is True


def test_semantic_fold_is_idempotent_and_model_view_remains_bounded():
    tree = ProcessTreeNode.seq(
        ProcessTreeNode.tau(),
        ProcessTreeNode.seq(
            *(ProcessTreeNode.activity(f"A{index}") for index in range(5))
        ),
    )

    normalized = prepare_tree_for_model(tree, maximum_arity=3)

    assert fold_process_tree(normalized.semantic_tree) == normalized.semantic_tree
    assert len(normalized.semantic_tree.children) == 5
    assert max(len(node.children) for node in _walk(normalized.model_tree)) <= 3
    TreeTokenizer(max_activities=5, max_arity=3).encode_tree(
        normalized.model_tree,
        canonicalize=False,
    )


def test_semantic_fold_preserves_representative_visible_behavior():
    tree = ProcessTreeNode.seq(
        ProcessTreeNode.tau(),
        ProcessTreeNode.seq(
            ProcessTreeNode.activity("A0"),
            ProcessTreeNode.activity("A1"),
        ),
        ProcessTreeNode.tau(),
    )

    before = {tuple(trace) for trace in simulate_traces(tree, 20)}
    after = {tuple(trace) for trace in simulate_traces(fold_process_tree(tree), 20)}

    assert before == after == {("A0", "A1")}


def test_sanitization_replaces_illegal_and_duplicate_labels_before_folding():
    tree = ProcessTreeNode.seq(
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.activity("A2"),
    )

    sanitized = sanitize_activity_labels(
        tree,
        allowed_labels={"A0"},
        avoid_duplicates=True,
    )

    assert sanitized.out_of_source_activities_replaced == 1
    assert sanitized.duplicate_activities_replaced == 1
    assert fold_process_tree(sanitized.tree) == ProcessTreeNode.activity("A0")


def test_shared_constraint_mask_keeps_tau_and_masks_illegal_or_used_activities():
    model = _model(3)
    tokenizer = model.tree_tokenizer
    prefix = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["SEQ"],
            tokenizer.token_to_id["ARITY_2"],
            tokenizer.token_to_id["A0"],
        ]]
    )
    grammar = tokenizer.valid_next_token_masks(prefix)
    mask = model.tree_decoder.decode_constraint_mask(
        prefix,
        grammar_mask=grammar,
        allowed_activity_mask=torch.tensor([[True, True, False]]),
        avoid_duplicate_activity_labels=True,
    )[0, -1]
    next_mask = model.tree_decoder.decode_constraint_mask(
        torch.tensor([[tokenizer.bos_id]]),
        grammar_mask=tokenizer.valid_next_token_masks(
            torch.tensor([[tokenizer.bos_id]])
        ),
        allowed_activity_mask=torch.tensor([[False, False, False]]),
        avoid_duplicate_activity_labels=True,
    )[0, -1]

    assert not mask[tokenizer.token_to_id["A0"]]
    assert not next_mask[tokenizer.token_to_id["A0"]]
    assert next_mask[tokenizer.token_to_id["TAU"]]

    legacy_mask = model.tree_decoder.decode_constraint_mask(
        prefix,
        grammar_mask=grammar,
        allowed_activity_mask=torch.tensor([[True, True, False]]),
        avoid_duplicate_activity_labels=False,
    )[0, -1]
    assert legacy_mask[tokenizer.token_to_id["A0"]]


def test_penalize_duplicate_policy_softens_search_score_without_hard_masking():
    model = _model(3)
    tokenizer = model.tree_tokenizer
    prefix = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.token_to_id["SEQ"],
            tokenizer.token_to_id["ARITY_2"],
            tokenizer.token_to_id["A0"],
        ]]
    )

    scores = model.tree_decoder.next_token_scores(
        torch.zeros(1, 8),
        prefix,
        avoid_duplicate_activity_labels=True,
        duplicate_policy="penalize",
        completion_policy="prefix_only",
    )
    repeated_id = tokenizer.token_to_id["A0"]
    unused_id = tokenizer.token_to_id["A1"]

    assert scores.effective_mask[0, repeated_id]
    assert scores.search_scores[0, repeated_id].item() == pytest.approx(
        scores.base_log_probs[0, repeated_id].item() - 0.75
    )
    assert torch.equal(
        scores.search_scores[0, unused_id],
        scores.base_log_probs[0, unused_id],
    )


def test_greedy_beam_and_progressive_decoding_respect_source_and_duplicates():
    torch.manual_seed(4)
    model = _model(3)
    tokenizer = model.tree_tokenizer
    latent = torch.randn(1, 8)
    allowed = torch.tensor([[True, False, False]])
    constraints = DecodeConstraints(allowed_activity_slots=allowed)

    greedy = model.tree_decoder.decode_greedy(
        latent,
        max_length=24,
        constraints=constraints,
    )[0].tolist()
    beams = model.tree_decoder.decode_beam_candidates(
        latent,
        max_length=24,
        beam_size=20,
        constraints=constraints,
    )[0]
    steps = list(
        decode_latent_iter(
            model,
            latent,
            max_length=24,
            allowed_activity_slots=[True, False, False],
        )
    )

    for row in [greedy, *(tokens for tokens, _ in beams)]:
        names = [tokenizer.tokens[token] for token in row]
        assert "A1" not in names and "A2" not in names
        assert names.count("A0") <= 1
    assert all(step.chosen_token not in {"A1", "A2"} for step in steps)
    assert sum(step.chosen_token == "A0" for step in steps) <= 1


def test_build_result_preserves_raw_and_normalized_views_and_restores_last(tmp_path):
    model = _model(3)
    tokenizer = model.tree_tokenizer
    raw_names = ["<bos>", "SEQ", "ARITY_3", "A0", "A0", "A2", "<eos>"]
    result = build_decode_result(
        model,
        [0.0] * 8,
        source_artifact_ids=["source"],
        source_modalities=[ArtifactModality.PROCESS_TREE],
        latent_source="test",
        token_ids=[tokenizer.token_to_id[name] for name in raw_names],
        canonical_mapping={"Original": "A0"},
        allowed_activity_slots=[True, False, False],
    )

    assert result.raw_token_names == raw_names
    assert str(result.raw_tree) == "SEQ(A0, A0, A2)"
    assert result.tree == ProcessTreeNode.activity("A0")
    assert result.restored_tree == ProcessTreeNode.activity("Original")
    assert result.model_normalized_token_names == ["<bos>", "A0", "<eos>"]
    assert result.out_of_source_activities_replaced == 1
    assert result.duplicate_activities_replaced == 1
    assert result.source_alphabet_respected and result.duplicate_free
    assert result.output_fold_idempotent

    exported = tmp_path / "normalized.ptml"
    exported.write_bytes(process_tree_ptml(result))
    assert parse_artifact(exported).tree == result.restored_tree
    assert {
        label for label in result.petri_net.graph.transition_labels if label is not None
    } == {"Original"}
    assert set(simulate_decoded_behavior(result, num_traces=10)) == {("Original",)}


def test_fused_source_evidence_uses_union_of_allowed_activities():
    left = _encoding(
        "left",
        allowed=[True, False, False],
        copy=[True, False, False],
        activity_memory=[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    right = _encoding(
        "right",
        allowed=[False, True, False],
        copy=[False, True, False],
        activity_memory=[[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    )

    allowed, copy, memory = combine_encoding_decode_evidence([left, right])

    assert allowed == [True, True, False]
    assert copy == [True, True, False]
    assert memory == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def test_command_line_duplicate_constraint_is_opt_in():
    parser = argparse.ArgumentParser()
    add_decode_constraint_arguments(parser)

    assert parser.parse_args([]).avoid_duplicate_transitions is False
    assert (
        parser.parse_args(["--avoid-duplicate-transitions"]).avoid_duplicate_transitions
        is True
    )


def test_trace_script_encoding_preserves_decoder_copy_evidence():
    model = _model(3)

    distribution = encode_traces_distribution(
        model,
        (("A0", "A1", "A0"),),
        torch.device("cpu"),
        max_traces=4,
        max_trace_length=8,
    )

    assert isinstance(distribution, LatentDistribution)
    assert distribution.activity_mask.tolist() == [[True, True, False]]
    assert distribution.activity_memory is not None
    assert distribution.activity_memory.shape == (1, 3, model.tree_decoder.hidden_dim)


def test_command_line_decode_forwards_full_distribution_and_allows_duplicates():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=3)
    captured: dict[str, object] = {}

    class Decoder:
        def decode_greedy(self, source, **kwargs):
            captured["source"] = source
            captured["constraints"] = kwargs["constraints"]
            return torch.tensor(
                [[tokenizer.bos_id, tokenizer.token_to_id["A0"], tokenizer.eos_id]]
            )

    model = SimpleNamespace(tree_tokenizer=tokenizer, tree_decoder=Decoder())
    distribution = LatentDistribution(
        mu=torch.zeros(1, 8),
        logvar=torch.zeros(1, 8),
        activity_mask=torch.tensor([[True, False]]),
        activity_memory=torch.zeros(1, 2, 16),
    )

    tree, _ = decode_tree_from_latent(
        model,
        distribution,
        max_decode_length=8,
        require_petri_convertible=False,
        canonical_mapping={"activity": "A0"},
    )

    assert tree == ProcessTreeNode.activity("A0")
    assert captured["source"] is distribution
    constraints = captured["constraints"]
    assert isinstance(constraints, DecodeConstraints)
    assert constraints.avoid_duplicate_activity_labels is False


def test_pnml_preparation_is_label_aware_and_exposes_allowed_slots():
    model = _model(8)
    parsed = parse_artifact("scripts/files/running-example.pnml")
    prepared = prepare_artifact_for_model(
        parsed,
        max_activities=8,
        max_arity=3,
    )
    tensors = petri_graph_to_tensors(
        prepared.model_input,
        torch.device("cpu"),
        model.activity_tokenizer,
    )

    assert prepared.canonical_mapping
    assert prepared.model_input_summary["visible_labels_used_by_encoder"] is True
    assert tensors["transition_label_ids"].gt(1).any()


def test_modality_targets_are_folded_bounded_and_source_legal():
    tree = ProcessTreeNode.seq(
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.activity("A1"),
    )
    graph = tree_to_petri_net(tree).graph
    targets = decoder_target_trees_for_sample(tree, (("A0",),), graph)
    tokenizer = TreeTokenizer(max_activities=2, max_arity=3)

    assert targets["trace"] == ProcessTreeNode.activity("A0")
    for target in targets.values():
        tokenizer.encode_tree(target, canonicalize=False)


def test_vectorized_grammar_masks_match_reference_for_random_prefixes():
    tokenizer = TreeTokenizer(max_activities=4, max_arity=4)
    generator = torch.Generator().manual_seed(7)
    prefixes = torch.randint(
        tokenizer.vocab_size,
        (64, 24),
        generator=generator,
    )

    assert torch.equal(
        tokenizer.valid_next_token_masks(prefixes),
        tokenizer.valid_next_token_masks_reference(prefixes),
    )


def test_bounded_completion_masks_exact_boundaries_and_operator_arities():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=4)
    tokens = tokenizer.token_to_id

    shortest = tokenizer.next_token_mask(
        [tokenizer.bos_id],
        remaining_tokens=2,
    )
    assert shortest[tokens["TAU"]]
    assert shortest[tokens["A0"]]
    assert not shortest[tokens["SEQ"]]

    operator_boundary = tokenizer.next_token_mask(
        [tokenizer.bos_id],
        remaining_tokens=5,
    )
    assert operator_boundary[tokens["SEQ"]]
    assert not tokenizer.next_token_mask(
        [tokenizer.bos_id],
        remaining_tokens=4,
    )[tokens["SEQ"]]

    pending_boundary = tokenizer.next_token_mask(
        [tokenizer.bos_id, tokens["SEQ"]],
        remaining_tokens=4,
    )
    assert pending_boundary[tokens["ARITY_2"]]
    assert not pending_boundary[tokens["ARITY_3"]]

    class TernaryXorTokenizer(TreeTokenizer):
        def legal_arities(self, operator: str | int) -> tuple[int, ...]:
            name = self.tokens[int(operator)] if isinstance(operator, int) else operator
            return (3,) if name == "XOR" else super().legal_arities(operator)

    specialized = TernaryXorTokenizer(max_activities=2, max_arity=4)
    specialized_tokens = specialized.token_to_id
    root_boundary = specialized.next_token_mask(
        [specialized.bos_id],
        remaining_tokens=5,
    )
    assert root_boundary[specialized_tokens["SEQ"]]
    assert not root_boundary[specialized_tokens["XOR"]]
    xor_arity = specialized.next_token_mask(
        [specialized.bos_id, specialized_tokens["XOR"]],
        remaining_tokens=5,
    )
    assert xor_arity[specialized_tokens["ARITY_3"]]
    assert not xor_arity[specialized_tokens["ARITY_2"]]


def test_every_bounded_candidate_leaves_a_fitting_completion():
    tokenizer = TreeTokenizer(max_activities=3, max_arity=4)
    pending_values = [
        0,
        *(tokenizer.token_to_id[name] for name in tokenizer.operator_tokens),
    ]
    for open_slots in range(5):
        for pending in pending_values:
            if pending and open_slots < 0:
                continue
            open_nodes = torch.tensor([open_slots])
            pending_operator = torch.tensor([pending])
            grammar = tokenizer.prefix_grammar_mask(open_nodes, pending_operator)[0]
            for remaining in range(1, 11):
                completion = tokenizer.completion_feasibility_mask(
                    open_nodes,
                    pending_operator,
                    remaining,
                )[0]
                for token_id in torch.where(grammar)[0].tolist():
                    next_open = open_nodes.clone()
                    next_pending = pending_operator.clone()
                    tokenizer.advance_grammar_state(
                        torch.tensor([token_id]),
                        next_open,
                        next_pending,
                        torch.tensor([True]),
                    )
                    minimum_after = (
                        0
                        if token_id == tokenizer.eos_id
                        else int(
                            tokenizer.minimum_tokens_to_finish(
                                next_open,
                                next_pending,
                            )[0]
                        )
                    )
                    assert bool(completion[token_id]) == (
                        minimum_after <= remaining - 1
                    )


def _expansion_biased_model() -> ProcRosettaModel:
    model = _model(2)
    tokenizer = model.tree_tokenizer
    decoder = model.tree_decoder
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        structural_ids = decoder.structural_token_ids.tolist()
        loop_output = structural_ids.index(tokenizer.token_to_id["LOOP"])
        decoder.structural_output.bias[loop_output] = 20.0
        arity_ids = decoder.arity_token_ids.tolist()
        arity_output = arity_ids.index(tokenizer.token_to_id["ARITY_3"])
        decoder.arity_output.bias[arity_output] = 20.0
    return model.eval()


def test_bounded_runaway_decode_terminates_and_reports_interventions():
    model = _expansion_biased_model()
    tokenizer = model.tree_tokenizer
    latent = torch.zeros(1, 8)

    raw = model.tree_decoder.decode_greedy(
        latent,
        max_length=8,
        completion_policy="prefix_only",
        constrain_to_source_activities=False,
        avoid_duplicate_activity_labels=False,
    )[0].tolist()
    bounded = model.tree_decoder.decode_greedy(
        latent,
        max_length=8,
        completion_policy="bounded",
        constrain_to_source_activities=False,
        avoid_duplicate_activity_labels=False,
    )[0].tolist()
    steps = list(
        decode_latent_iter(
            model,
            latent,
            max_length=8,
            completion_policy="bounded",
            constrain_to_source_activities=False,
            avoid_duplicate_activity_labels=False,
        )
    )

    assert tokenizer.eos_id not in raw
    assert tokenizer.eos_id in bounded
    tokenizer.decode_tree(bounded)
    assert any(step.budget_mask_active for step in steps)
    assert any(step.argmax_overridden for step in steps)
    overridden = next(step for step in steps if step.argmax_overridden)
    assert overridden.selected_pre_budget_probability < 1.0
    assert overridden.selected_conditional_probability >= overridden.selected_pre_budget_probability
    result = build_decode_result(
        model,
        latent,
        source_artifact_ids=["synthetic"],
        source_modalities=[ArtifactModality.PROCESS_TREE],
        latent_source="test",
        token_ids=bounded,
        steps=steps,
        completion_policy="bounded",
        max_length=8,
    )
    assert result.decoder_configuration["completion_policy"] == "bounded"
    assert result.budget_intervention_steps > 0
    assert result.argmax_override_steps > 0
    assert result.minimum_completion_slack is not None
    assert result.minimum_completion_slack >= 0
    assert result.raw_unresolved_open_slots == 0


def test_bounded_decoding_has_cross_path_decision_parity():
    model = _expansion_biased_model()
    tokenizer = model.tree_tokenizer
    latent = torch.zeros(1, 8)
    arguments = {
        "max_length": 8,
        "completion_policy": "bounded",
        "constrain_to_source_activities": False,
        "avoid_duplicate_activity_labels": False,
    }

    incremental = model.tree_decoder.decode_greedy(latent, **arguments)[0].tolist()
    model.train()
    full_prefix = model.tree_decoder.decode_greedy(latent, **arguments)[0].tolist()
    model.eval()
    beam = model.tree_decoder.decode_beam_candidates(
        latent,
        beam_size=1,
        **arguments,
    )[0][0][0]
    progressive = [tokenizer.bos_id]
    progressive.extend(
        step.chosen_token_id
        for step in decode_latent_iter(model, latent, **arguments)
    )

    assert incremental == full_prefix == beam == progressive


def test_every_bounded_decode_emits_eos_and_converts_to_petri():
    torch.manual_seed(19)
    model = _model(3)
    tokenizer = model.tree_tokenizer
    latent = torch.randn(12, 8)
    for max_length in (3, 7, 12):
        decoded = model.tree_decoder.decode_greedy(
            latent,
            max_length=max_length,
            completion_policy="bounded",
            constrain_to_source_activities=False,
            avoid_duplicate_activity_labels=False,
        )
        for token_ids in decoded.tolist():
            assert tokenizer.eos_id in token_ids
            tree = tokenizer.decode_tree(token_ids)
            tree_to_petri_net(tree)


def test_budget_mask_does_not_renormalize_beam_scores():
    model = _expansion_biased_model()
    tokenizer = model.tree_tokenizer
    prefix = torch.tensor([[tokenizer.bos_id]])
    scores = model.tree_decoder.next_token_scores(
        torch.zeros(1, 8),
        prefix,
        completion_policy="bounded",
        remaining_tokens=2,
    )
    removed_operator = tokenizer.token_to_id["LOOP"]
    selected_leaf = tokenizer.token_to_id["TAU"]

    assert scores.prefix_grammar_mask[0, removed_operator]
    assert not scores.completion_mask[0, removed_operator]
    assert torch.equal(
        scores.search_scores[0, selected_leaf],
        scores.base_log_probs[0, selected_leaf],
    )
    surviving_mass = scores.search_scores[0].exp().sum()
    assert 0 < surviving_mass < 1


def test_prefix_only_forward_is_unchanged_and_bounded_contract_is_explicit():
    model = _model(2)
    tokenizer = model.tree_tokenizer
    latent = torch.zeros(1, 8)
    prefix = torch.tensor([[tokenizer.bos_id]])

    historical = model.tree_decoder(latent, prefix)
    explicit = model.tree_decoder(
        latent,
        prefix,
        completion_policy="prefix_only",
    )
    assert torch.equal(historical, explicit)

    with pytest.raises(ValueError, match="remaining token budget"):
        model.tree_decoder(latent, prefix, completion_policy="bounded")
    with pytest.raises(ValueError, match="requires grammar masking"):
        model.tree_decoder(
            latent,
            prefix,
            apply_grammar_mask=False,
            completion_policy="bounded",
            remaining_tokens=2,
        )
    with pytest.raises(ValueError, match="max_length >= 3"):
        model.tree_decoder.decode_greedy(latent, max_length=2)
    with pytest.raises(ValueError, match="valid grammar prefix"):
        tokenizer.next_token_mask(
            [tokenizer.bos_id, tokenizer.eos_id],
            remaining_tokens=2,
        )


def test_shortest_completion_and_strict_sequence_contract():
    tokenizer = TreeTokenizer(max_activities=2, max_arity=3)
    sequence_id = tokenizer.token_to_id["SEQ"]
    completion = tokenizer.shortest_completion(0, sequence_id)

    assert completion == [
        tokenizer.token_to_id["ARITY_2"],
        tokenizer.token_to_id["TAU"],
        tokenizer.token_to_id["TAU"],
        tokenizer.eos_id,
    ]
    token_ids = [tokenizer.bos_id, sequence_id, *completion]
    tokenizer.validate_complete_tree_sequence(token_ids, token_budget=len(token_ids))
    with pytest.raises(ValueError, match="missing EOS"):
        tokenizer.validate_complete_tree_sequence(token_ids[:-1], token_budget=len(token_ids))
    with pytest.raises(ValueError, match="tree exceeds token budget"):
        tokenizer.validate_complete_tree_sequence(token_ids, token_budget=len(token_ids) - 1)
    with pytest.raises(ValueError, match="non-padding token after EOS"):
        tokenizer.validate_complete_tree_sequence(
            [*token_ids, tokenizer.token_to_id["TAU"]],
            token_budget=len(token_ids) + 1,
        )


def test_budget_three_forced_closure_never_consults_neural_decoder(monkeypatch):
    model = _model(2)
    tokenizer = model.tree_tokenizer

    def fail_if_scored(*args, **kwargs):
        raise AssertionError("boundary closure must not score the neural decoder")

    monkeypatch.setattr(
        model.tree_decoder,
        "_incremental_next_token_scores",
        fail_if_scored,
    )
    greedy = model.tree_decoder.decode_guaranteed(
        torch.zeros(1, 8),
        total_token_budget_including_bos_eos=3,
    )
    beam = model.tree_decoder.decode_guaranteed(
        torch.zeros(1, 8),
        total_token_budget_including_bos_eos=3,
        beam_size=3,
    )
    progressive = [tokenizer.bos_id]
    progressive.extend(
        step.chosen_token_id
        for step in decode_latent_iter(model, torch.zeros(1, 8), max_length=3)
    )
    expected = [tokenizer.bos_id, tokenizer.token_to_id["TAU"], tokenizer.eos_id]

    assert greedy[0].tolist() == beam[0].tolist() == progressive == expected


def test_guaranteed_api_hides_policy_and_emergency_fallback_is_explicit(monkeypatch):
    assert "completion_policy" not in inspect.signature(decode_guaranteed).parameters
    assert "completion_policy" not in inspect.signature(
        _model(2).tree_decoder.decode_guaranteed
    ).parameters

    model = _model(2)
    tokenizer = model.tree_tokenizer
    checkpoint = SimpleNamespace(model=model, device=torch.device("cpu"))

    def fail_decode(*args, **kwargs):
        raise RuntimeError("injected neural failure")

    monkeypatch.setattr(model.tree_decoder, "decode_beam", fail_decode)
    result = decode_guaranteed(
        checkpoint,
        torch.zeros(1, 8),
        source_artifact_ids=["test"],
        source_modalities=[ArtifactModality.PROCESS_TREE],
        latent_source="test",
        total_token_budget_including_bos_eos=8,
    )

    assert result.raw_token_ids == [
        tokenizer.bos_id,
        tokenizer.token_to_id["TAU"],
        tokenizer.eos_id,
    ]
    assert result.hard_structural_success
    assert result.fallback_used
    assert result.fallback_reason == "RuntimeError: injected neural failure"
    assert result.forced_closure_used
    assert not result.neural_decode_without_fallback


def test_token_budget_rejects_minimum_and_positional_capacity_violations():
    model = _model(2)
    with pytest.raises(ValueError, match="fewer than 3 tokens"):
        model.tree_decoder.decode_guaranteed(
            torch.zeros(1, 8),
            total_token_budget_including_bos_eos=2,
        )
    with pytest.raises(ValueError, match="positional capacity"):
        model.tree_decoder.decode_guaranteed(
            torch.zeros(1, 8),
            total_token_budget_including_bos_eos=(
                model.tree_decoder.maximum_supported_decode_length + 1
            ),
        )


def test_sparse_and_dense_petri_paths_match_outputs_and_gradients():
    torch.manual_seed(2)
    tokenizer = ActivityTokenizer(max_activities=3)
    dense = PetriGraphEncoder(
        tokenizer,
        semantic_dim=4,
        hidden_dim=8,
        message_passing_steps=2,
        dropout=0.0,
        memory_tokens=2,
    ).eval()
    sparse = deepcopy(dense)
    node_types = torch.tensor([[0, 1, 0], [0, 2, 0]])
    markings = torch.randn(2, 3, 2)
    node_mask = torch.ones(2, 3, dtype=torch.bool)
    labels = torch.tensor([[0, 2, 0], [0, 0, 0]])
    adjacency = torch.zeros(2, 2, 3, 3)
    edge_rows = [(0, 0, 1, 0), (0, 1, 2, 1), (1, 0, 1, 0), (1, 1, 2, 1)]
    for batch, source, target, kind in edge_rows:
        adjacency[batch, kind, source, target] = 1
    edge_index = torch.tensor(
        [[batch * 3 + source, batch * 3 + target] for batch, source, target, _ in edge_rows]
    ).T
    edge_types = torch.tensor([kind for _, _, _, kind in edge_rows])

    dense_output = dense(node_types, markings, adjacency, node_mask, labels)
    sparse_output = sparse(
        node_types,
        markings,
        adjacency,
        node_mask,
        labels,
        edge_index,
        edge_types,
    )
    dense_output.pre_normalized.sum().backward()
    sparse_output.pre_normalized.sum().backward()

    assert torch.allclose(dense_output.mu, sparse_output.mu, atol=1e-6)
    for left, right in zip(dense.parameters(), sparse.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-5)


def test_stacked_decoder_matches_three_separate_calls_in_eval_mode():
    tree = ProcessTreeNode.seq(
        ProcessTreeNode.activity("A0"), ProcessTreeNode.activity("A1")
    )
    graph = tree_to_petri_net(tree).graph
    sample = ProcessSample(
        tree=tree,
        traces=(("A0", "A1"),),
        petri_graph=graph,
        equivalence_id="one",
        decoder_target_trees=decoder_target_trees_for_sample(
            tree, (("A0", "A1"),), graph
        ),
    )
    model = _model(2)
    batch = ProcessBatchCollator(model.tree_tokenizer, model.activity_tokenizer)([sample])
    outputs = model(batch)

    for name in ("tree", "trace", "petri"):
        separate = model.tree_decoder(
            outputs["dists"][name],
            batch["decoder_targets"][name][:, :-1],
            allowed_activity_mask=batch["source_activity_masks"][name],
        )
        assert torch.allclose(outputs["tree_logits"][name], separate, atol=1e-6)


def test_incremental_decoder_cache_matches_full_prefix_evaluation():
    torch.manual_seed(8)
    model = _model(3)
    decoder = model.tree_decoder
    tokenizer = model.tree_tokenizer
    latent = torch.randn(2, 8)
    memory, _ = decoder.source_memory(latent)
    prefix = torch.tensor(
        [
            [tokenizer.bos_id, tokenizer.token_to_id["SEQ"], tokenizer.token_to_id["ARITY_2"]],
            [tokenizer.bos_id, tokenizer.token_to_id["XOR"], tokenizer.token_to_id["ARITY_3"]],
        ]
    )
    caches = [None] * len(decoder.decoder.layers)
    for position in range(prefix.shape[1]):
        hidden, caches = decoder._incremental_hidden(
            prefix[:, position], position, memory, caches
        )
        incremental = decoder._project_hidden(hidden)[:, 0]
        full = decoder(
            latent,
            prefix[:, : position + 1],
            apply_grammar_mask=False,
        )[:, -1]
        assert torch.allclose(incremental, full, atol=1e-6)


def test_expensive_gradient_instrumentation_is_opt_in():
    config = TrainConfig()

    assert config.gradient_diagnostics_interval == 0
    assert not config.use_pcgrad


def test_gradient_diagnostics_do_not_change_optimizer_updates():
    torch.manual_seed(12)
    reference = SimpleNamespace(
        tree_encoder=torch.nn.Linear(3, 2),
        trace_encoder=torch.nn.Linear(3, 2),
        petri_encoder=torch.nn.Linear(3, 2),
    )
    diagnosed = deepcopy(reference)
    inputs = torch.randn(4, 3)

    def update(model, *, diagnose):
        optimizer = torch.optim.SGD(
            [
                parameter
                for encoder in (
                    model.tree_encoder,
                    model.trace_encoder,
                    model.petri_encoder,
                )
                for parameter in encoder.parameters()
            ],
            lr=0.05,
        )
        reconstructions = {
            name: getattr(model, f"{name}_encoder")(inputs).square().mean()
            for name in ("tree", "trace", "petri")
        }
        metric = sum(
            getattr(model, f"{name}_encoder")(inputs).mean()
            for name in ("tree", "trace", "petri")
        )
        losses = {
            "tree_reconstruction": reconstructions["tree"],
            "trace_to_tree": reconstructions["trace"],
            "petri_to_tree": reconstructions["petri"],
            "exact_contrastive": metric,
            "within_modality_contrastive": metric * 0.5,
            "soft_behavior_geometry": metric * 0.25,
            "variance": metric * 0.125,
            "covariance": metric * 0.0625,
        }
        weights = LossWeights()
        total = (
            weights.tree_reconstruction * losses["tree_reconstruction"]
            + weights.trace_to_tree * losses["trace_to_tree"]
            + weights.petri_to_tree * losses["petri_to_tree"]
            + weights.exact_contrastive * losses["exact_contrastive"]
            + weights.within_modality_contrastive * losses["within_modality_contrastive"]
            + weights.soft_behavior_geometry * losses["soft_behavior_geometry"]
            + weights.variance * losses["variance"]
            + weights.covariance * losses["covariance"]
        )
        if diagnose:
            diagnostics = gradient_norm_diagnostics(model, losses, weights)
            for modality in ("tree", "trace", "petri"):
                for name in (
                    "reconstruction_exact_gradient_cosine",
                    "reconstruction_soft_geometry_gradient_cosine",
                    "exact_soft_geometry_gradient_cosine",
                ):
                    assert -1.0 <= diagnostics[f"{name}_{modality}"] <= 1.0
        total.backward()
        optimizer.step()

    update(reference, diagnose=False)
    update(diagnosed, diagnose=True)

    for left_encoder, right_encoder in (
        (reference.tree_encoder, diagnosed.tree_encoder),
        (reference.trace_encoder, diagnosed.trace_encoder),
        (reference.petri_encoder, diagnosed.petri_encoder),
    ):
        for left, right in zip(left_encoder.parameters(), right_encoder.parameters()):
            assert torch.equal(left, right)


def test_decode_cache_key_separates_source_alphabet_and_duplicate_policy():
    arguments = {
        "checkpoint_identifier": "checkpoint",
        "latent": [0.1, 0.2],
        "max_length": 32,
        "beam_size": 5,
        "mapping": {"original": "A0"},
        "copy_slots": [True, False],
        "activity_memory": [[1.0], [0.0]],
        "constrain_to_source_activities": True,
        "artifact_ids": ["source"],
        "latent_source": "test",
    }
    baseline = _decode_cache_key(
        **arguments,
        allowed_slots=[True, False],
        avoid_duplicate_activity_labels=True,
    )
    different_alphabet = _decode_cache_key(
        **arguments,
        allowed_slots=[False, True],
        avoid_duplicate_activity_labels=True,
    )
    duplicates_allowed = _decode_cache_key(
        **arguments,
        allowed_slots=[True, False],
        avoid_duplicate_activity_labels=False,
    )
    raw_policy = _decode_cache_key(
        **arguments,
        allowed_slots=[True, False],
        avoid_duplicate_activity_labels=True,
        completion_policy="prefix_only",
    )

    assert len({baseline, different_alphabet, duplicates_allowed, raw_policy}) == 4


def _encoding(
    artifact_id: str,
    *,
    allowed: list[bool],
    copy: list[bool],
    activity_memory: list[list[float]],
) -> ArtifactEncodingResult:
    return ArtifactEncodingResult(
        artifact_id=artifact_id,
        artifact_name=artifact_id,
        modality=ArtifactModality.EVENT_LOG,
        checkpoint_identifier="checkpoint",
        source_metadata={},
        preprocessing_metadata={},
        canonical_mapping={},
        model_input_summary={},
        mu=[0.0],
        logvar=[0.0],
        attention_weights=None,
        embedding_seconds=0.0,
        allowed_activity_slots=allowed,
        copy_activity_slots=copy,
        activity_memory=activity_memory,
    )


def _walk(tree: ProcessTreeNode):
    yield tree
    for child in tree.children:
        yield from _walk(child)
