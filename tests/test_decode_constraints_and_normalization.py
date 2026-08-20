from __future__ import annotations

from copy import deepcopy
import random

import torch

from proc_rosetta.artifact_io import (
    ArtifactModality,
    parse_artifact,
    prepare_artifact_for_model,
)
from proc_rosetta.data import ProcessBatchCollator
from proc_rosetta.inference import build_decode_result, decode_latent_iter, petri_graph_to_tensors
from proc_rosetta.models import DecodeConstraints, PetriGraphEncoder, ProcRosettaModel
from proc_rosetta.pm4py_bridge import (
    fold_process_tree,
    prepare_tree_for_model,
    tree_to_petri_net,
)
from proc_rosetta.synthetic import ProcessSample, decoder_target_trees_for_sample
from proc_rosetta.tokenizers import ActivityTokenizer, TreeTokenizer
from proc_rosetta.training import TrainConfig
from proc_rosetta.tree import ProcessTreeNode, sanitize_activity_labels


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


def test_build_result_preserves_raw_and_normalized_views_and_restores_last():
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


def test_gradient_diagnostics_are_disabled_by_default():
    assert TrainConfig().gradient_diagnostics_interval == 0


def _walk(tree: ProcessTreeNode):
    yield tree
    for child in tree.children:
        yield from _walk(child)
