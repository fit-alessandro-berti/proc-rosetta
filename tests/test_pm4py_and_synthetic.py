import random

import pytest

from proc_rosetta.pm4py_bridge import petri_graph_to_net, simulate_traces, tree_to_petri_net
from proc_rosetta.synthetic import (
    CURRICULUM_LEVELS,
    ProcessSample,
    SyntheticConfig,
    accepts_profile,
    complexity_profile,
    config_for_curriculum,
    generate_process_tree,
    generate_process_tree_with_provenance,
    generate_sample,
)
from proc_rosetta.families import (
    allocate_operator_quotas,
    enumerate_visible_language,
    family_generation_plan,
    flatten_behavior_family,
    generate_behavior_family,
)
from proc_rosetta.data import operator_probability_audit, sample_statistics
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


def test_default_complexity_profiles_form_a_small_ordered_curriculum():
    def profile_bounds(level):
        profile = complexity_profile(level)
        return (
            profile.max_depth,
            profile.min_tree_depth,
            profile.min_tree_size,
            profile.max_tree_size,
            profile.min_generated_activities,
            profile.max_generated_activities,
        )

    assert {level: profile_bounds(level) for level in CURRICULUM_LEVELS} == {
        "simple": (2, 2, 3, 6, 2, 4),
        "medium": (3, 2, 7, 11, 3, 7),
        "complex": (4, 3, 12, 18, 5, 12),
    }


@pytest.mark.parametrize("level", CURRICULUM_LEVELS)
def test_operator_probabilities_do_not_change_across_curricula(level):
    config = config_for_curriculum(SyntheticConfig(), level)
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


@pytest.mark.parametrize("level", CURRICULUM_LEVELS)
def test_generated_final_folded_tree_satisfies_profile(level):
    config = config_for_curriculum(SyntheticConfig(), level)
    tree, provenance = generate_process_tree_with_provenance(
        config, random.Random(101)
    )
    assert accepts_profile(tree, complexity_profile(level))
    assert provenance["folded_complexity"]["tree_size"] == tree.size()
    assert tree.kind is not NodeKind.ACTIVITY


def test_root_operator_quota_is_exact_for_multiple_of_ten():
    config = config_for_curriculum(
        SyntheticConfig(
            motif_weights={"ordinary_tree": 1.0},
            min_families_per_motif={"training": 0, "validation": 0, "test": 0},
        ),
        "simple",
    )
    plan, quotas = family_generation_plan(1000, config, 13, "test")
    assert quotas == {"seq": 700, "xor": 100, "and": 100, "loop": 100}
    assert len(plan) == 1000
    assert allocate_operator_quotas(11, config.root_operator_probabilities) == {
        "seq": 8,
        "xor": 1,
        "and": 1,
        "loop": 1,
    }


def test_operator_audit_uses_configured_probabilities_from_provenance():
    config = config_for_curriculum(
        SyntheticConfig(
            motif_weights={"ordinary_tree": 1.0},
            operator_probabilities={"xor": 1.0},
            root_operator_probabilities={"seq": 1.0},
        ),
        "simple",
    )
    family = generate_behavior_family(
        config,
        0,
        seed=31,
        split="test",
        motif="ordinary_tree",
        forced_root_operator=NodeKind.SEQ,
    )
    audit = operator_probability_audit(
        [flatten_behavior_family(family, max_activities=config.max_activities)[0]]
    )

    assert audit["expected_root_operator_probabilities"] == {
        "seq": 1.0,
        "xor": 0.0,
        "and": 0.0,
        "loop": 0.0,
    }
    assert audit["expected_non_root_operator_probabilities"] == {
        "seq": 0.0,
        "xor": 1.0,
        "and": 0.0,
        "loop": 0.0,
    }


def test_family_statistics_do_not_duplicate_cross_product_rows():
    config = config_for_curriculum(
        SyntheticConfig(
            traces_per_sample=2,
            log_views_per_behavior=2,
            variants_per_behavior=2,
            motif_weights={"duplicate_vs_silent": 1.0},
        ),
        "simple",
    )
    family = generate_behavior_family(
        config,
        0,
        seed=37,
        split="test",
        motif="duplicate_vs_silent",
    )
    rows = flatten_behavior_family(family, max_activities=config.max_activities)
    statistics = sample_statistics(rows)

    assert len(rows) == 4
    assert statistics["complexity_distributions"]["tree_size"]["count"] == 1
    assert statistics["complexity_distributions"]["petri_node_count"]["count"] == 2
    assert statistics["complexity_distributions"]["trace_length"]["count"] == 4


def test_sequence_padding_no_longer_changes_forced_root():
    config = config_for_curriculum(SyntheticConfig(), "simple")
    tree = generate_process_tree(
        config,
        random.Random(7),
        forced_root_operator=NodeKind.XOR,
    )
    assert tree.kind is NodeKind.XOR
    assert len(tree.unique_activity_labels()) >= config.min_activities


def test_isomorphic_loop_family_has_strong_positive_id():
    config = config_for_curriculum(
        SyntheticConfig(
            motif_weights={"ordinary_tree": 1.0},
            max_trace_length=4,
        ),
        "simple",
    )
    family = generate_behavior_family(
        config,
        0,
        seed=19,
        split="training",
        motif="ordinary_tree",
        forced_root_operator=NodeKind.LOOP,
    )
    rows = flatten_behavior_family(family, max_activities=config.max_activities)
    assert family.equivalence_certificate.status == "isomorphic"
    assert all(row.strong_behavior_id == family.behavior_id for row in rows)
    assert max(map(len, family.clean_trace_pool)) <= config.max_trace_length


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


def test_trace_simulation_bounds_loops_even_when_rng_always_continues():
    class ContinueLoopRandom(random.Random):
        def random(self):
            return 0.0

    tree = ProcessTreeNode.loop(
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.activity("A1"),
    )

    traces = simulate_traces(
        tree,
        num_traces=3,
        max_trace_length=5,
        rng=ContinueLoopRandom(7),
    )

    assert traces == [["A0", "A1", "A0", "A1", "A0"]] * 3
    assert max(map(len, traces)) == 5


def test_trace_simulation_is_reproducible_with_a_local_rng():
    tree = ProcessTreeNode.loop(
        ProcessTreeNode.xor(
            ProcessTreeNode.activity("A0"),
            ProcessTreeNode.activity("A1"),
        ),
        ProcessTreeNode.activity("A2"),
    )

    first = simulate_traces(tree, 20, rng=random.Random(41))
    second = simulate_traces(tree, 20, rng=random.Random(41))

    assert first == second
    assert all(len(trace) <= 128 for trace in first)


def test_generate_sample_contains_all_modalities():
    sample = generate_sample(SyntheticConfig(traces_per_sample=4, max_depth=2), equivalence_id="x")

    assert sample.equivalence_id == "x"
    assert sample.tree.size() >= 2
    assert len(sample.traces) == 4
    assert sample.petri_graph.num_nodes > 0

    restored = ProcessSample.from_dict(sample.to_dict())
    assert restored.to_dict() == sample.to_dict()


def test_phase_three_enables_but_does_not_require_two_child_loops():
    sample = generate_sample(
        SyntheticConfig(
            operator_probabilities={"seq": 0.34, "xor": 0.33, "and": 0.33},
            root_operator_probabilities={"seq": 1.0},
        ),
        equivalence_id="x",
    )

    loop_nodes = []

    def visit(node):
        if node.kind is NodeKind.LOOP:
            loop_nodes.append(node)
        for child in node.children:
            visit(child)

    visit(sample.tree)

    assert len(sample.traces) == 128
    assert sample.tree.max_depth() >= complexity_profile("complex").min_tree_depth
    assert not loop_nodes


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
