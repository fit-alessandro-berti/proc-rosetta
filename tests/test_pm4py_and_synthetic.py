import random

import pytest

from proc_rosetta.pm4py_bridge import petri_graph_to_net, simulate_traces, tree_to_petri_net
from proc_rosetta.synthetic import (
    ProcessSample,
    SyntheticConfig,
    generate_process_tree,
    generate_sample,
)
from proc_rosetta.families import enumerate_visible_language, generate_behavior_family
from proc_rosetta.tree import NodeKind, ProcessTreeNode


def test_operator_probability_defaults_and_configuration_round_trip():
    config = SyntheticConfig()

    assert config.operator_probabilities == {
        "seq": 0.25,
        "xor": 0.25,
        "and": 0.25,
        "loop": 0.25,
    }
    assert config.root_operator_probabilities == {
        "seq": 0.7,
        "xor": 0.1,
        "and": 0.1,
        "loop": 0.1,
    }
    assert SyntheticConfig.from_dict(config.to_dict()) == config


def test_root_and_non_root_operator_probabilities_are_independent():
    config = SyntheticConfig(
        max_depth=2,
        max_activities=2,
        min_activities=2,
        max_arity=2,
        curriculum_phase=2,
        leaf_probability=0.0,
        min_tree_depth=1,
        min_tree_size=1,
        operator_probabilities={"xor": 1.0},
        root_operator_probabilities={"sequence": 1.0},
    )

    tree = generate_process_tree(config, random.Random(17))

    assert tree.kind is NodeKind.SEQ
    assert tree.children
    assert all(child.kind is NodeKind.XOR for child in tree.children)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({"seq": 0.5, "xor": 0.4}, "sum to 1.0"),
        ({"seq": 0.5, "unknown": 0.5}, "unknown operator"),
        ({"seq": 1.1, "xor": -0.1}, "non-negative"),
    ],
)
def test_operator_probability_validation(probabilities, message):
    with pytest.raises(ValueError, match=message):
        SyntheticConfig(operator_probabilities=probabilities)


def test_operator_probabilities_require_a_curriculum_eligible_choice():
    with pytest.raises(ValueError, match="enabled in curriculum phase 2"):
        SyntheticConfig(
            curriculum_phase=2,
            root_operator_probabilities={"loop": 1.0},
        )


def test_tree_to_petri_net_and_trace_simulation():
    tree = ProcessTreeNode.seq(ProcessTreeNode.activity("A0"), ProcessTreeNode.activity("A1"))

    petri = tree_to_petri_net(tree)
    traces = simulate_traces(tree, num_traces=3)

    assert petri.graph.num_nodes > 0
    assert petri.graph.num_edges > 0
    assert traces == [["A0", "A1"], ["A0", "A1"], ["A0", "A1"]]

    restored = petri_graph_to_net(petri.graph)
    assert restored.graph == petri.graph


def test_generate_sample_contains_all_modalities():
    sample = generate_sample(SyntheticConfig(traces_per_sample=4, max_depth=2), equivalence_id="x")

    assert sample.equivalence_id == "x"
    assert sample.tree.size() >= 2
    assert len(sample.traces) == 4
    assert sample.petri_graph.num_nodes > 0

    restored = ProcessSample.from_dict(sample.to_dict())
    assert restored.to_dict() == sample.to_dict()


def test_default_generator_uses_two_child_loops_and_larger_logs():
    sample = generate_sample(SyntheticConfig(), equivalence_id="x")

    loop_nodes = []

    def visit(node):
        if node.kind is NodeKind.LOOP:
            loop_nodes.append(node)
        for child in node.children:
            visit(child)

    visit(sample.tree)

    assert len(sample.traces) == 128
    assert sample.tree.max_depth() >= 4
    assert loop_nodes
    assert all(len(node.children) == 2 for node in loop_nodes)


def test_exact_behavior_family_motifs_share_language_and_ids():
    motif_pairs = {
        "duplicate_vs_silent": {"duplicate_prefix", "silent_routing"},
        "concurrent_vs_interleaved": {"parallel", "explicit_interleaving"},
        "m_nonfreechoice": {"canonical_block_pm4py", "m_nonfreechoice"},
    }
    for motif, expected_kinds in motif_pairs.items():
        config = SyntheticConfig(
            max_activities=6,
            traces_per_sample=4,
            motif_weights={motif: 1.0},
        )
        family = generate_behavior_family(config, 0, seed=7, split="test")

        assert family.equivalence_certificate.status == "exact"
        assert len(family.exact_behavior_id) == 64
        assert family.behavior_id == family.exact_behavior_id
        assert len(family.behavior_signature) == 128
        assert family.equivalence_certificate.semantics == "visible_complete_trace_language"
        assert {variant.representation_kind for variant in family.model_variants} == expected_kinds
        assert all(view.traces for view in family.log_views)
        assert all(variant.variant_id.startswith(family.behavior_id) for variant in family.model_variants)
        canonical = tree_to_petri_net(family.canonical_tree.canonicalize_activity_labels())
        canonical_language, completed = enumerate_visible_language(
            canonical.net,
            canonical.initial_marking,
            canonical.final_marking,
        )
        assert completed
        assert canonical_language == set(family.clean_trace_pool)
        if motif == "m_nonfreechoice":
            nonfree = next(
                variant
                for variant in family.model_variants
                if variant.representation_kind == "m_nonfreechoice"
            )
            assert nonfree.structural_statistics["free_choice_violation_count"] > 0
            assert nonfree.structural_statistics["nonblock_reason"] == "non_free_choice_m_pattern"


def test_controlled_families_receive_unique_random_structural_contexts():
    config = SyntheticConfig(
        max_activities=12,
        traces_per_sample=4,
        motif_weights={"duplicate_vs_silent": 1.0},
    )
    families = [
        generate_behavior_family(
            config, index, seed=17, split="training", motif="duplicate_vs_silent"
        )
        for index in range(3)
    ]

    assert len({family.exact_behavior_id for family in families}) == 3
    assert all(
        4 <= int(family.metadata["structural_context"]["node_count"]) <= 12
        for family in families
    )
    assert all(family.metadata["structural_context"]["operators"] for family in families)


def test_stage_d_curriculum_includes_each_observation_regime():
    config = SyntheticConfig.preset("stage_d_observation_curriculum")

    assert config.log_views_per_behavior == 6
    assert set(config.log_view_modes) == {
        "uniform_variants",
        "resampled",
        "sparse",
        "incomplete",
        "long_tail",
        "noisy",
    }
