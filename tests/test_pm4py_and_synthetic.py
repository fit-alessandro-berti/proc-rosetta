from proc_rosetta.pm4py_bridge import petri_graph_to_net, simulate_traces, tree_to_petri_net
from proc_rosetta.synthetic import ProcessSample, SyntheticConfig, generate_sample
from proc_rosetta.families import enumerate_visible_language, generate_behavior_family
from proc_rosetta.tree import ProcessTreeNode


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
