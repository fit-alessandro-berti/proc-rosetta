import sys
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from proc_rosetta.behavior import behavioral_distance
from proc_rosetta.benchmarks import (
    activity_count_features,
    alignment_f1_score,
    discovery_quality_report,
    decode_target_tree,
    decode_with_beam,
    behavior_matrices,
    evaluate_single_decode,
    ValidationAuditConfig,
    validation_audit_report,
    validation_split_hash,
    evaluate_inductive_miner_discovery,
    evaluate_proc_rosetta_discovery,
    evaluate_embedding_method,
    fitness_precision_f1_score,
    footprint_fitness_precision,
    format_human_test_report,
    levenshtein_distance,
    observation_quality_report,
    retrieval_metrics,
    rich_test_report,
    summarize_discovery_quality,
    token_based_replay_fitness_precision,
    trim_tree_token_sequence,
)
from proc_rosetta.tokenizers import TreeTokenizer
from proc_rosetta.tree import ProcessTreeNode
from proc_rosetta.data import ProcessBatchCollator, SyntheticProcessDataset
from proc_rosetta.models import ProcRosettaModel
from proc_rosetta.synthetic import SyntheticConfig
from proc_rosetta.tokenizers import ActivityTokenizer


def test_report_rejects_mixed_curriculum_samples():
    samples = [
        SimpleNamespace(complexity_level="simple"),
        SimpleNamespace(complexity_level="medium"),
    ]
    with pytest.raises(ValueError, match="other than 'simple'"):
        rich_test_report(
            checkpoint_path="unused.pt",
            data_dir="unused",
            samples=samples,
            curriculum="simple",
        )


def test_embedding_method_report_contains_behavior_alignment():
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    behavior = np.asarray(
        [
            [0.0, 0.2, 1.5],
            [0.2, 0.0, 1.0],
            [1.5, 1.0, 0.0],
        ]
    )

    report = evaluate_embedding_method(embeddings, behavior)

    assert report["available"] is True
    assert report["vector_statistics"]["count"] == 3
    assert report["nearest_neighbor_behavior"]["count"] == 3
    assert "spearman_embedding_distance_vs_behavior_l1" in report["behavior_alignment"]


def test_retrieval_metrics_and_trace_features():
    embeddings = np.eye(3)
    traces = [["A0", "A1"], ["A0", "A2"]]

    metrics = retrieval_metrics(embeddings, embeddings)
    features = activity_count_features(traces)

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["mrr"] == 1.0
    assert features[("activity", "A0")] == 0.5


def test_retrieval_accepts_any_candidate_with_the_same_exact_behavior_id():
    query = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    candidates = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    metrics = retrieval_metrics(
        query,
        candidates,
        query_labels=["same", "same"],
        candidate_labels=["same", "same"],
    )

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["recall_at_5"] == 1.0


def test_decode_token_edit_helpers():
    tokenizer = TreeTokenizer(max_activities=4)
    tokens = [
        tokenizer.bos_id,
        tokenizer.token_to_id["A0"],
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.token_to_id["A1"],
    ]

    assert trim_tree_token_sequence(tokens, tokenizer) == tokens[:3]
    assert levenshtein_distance([1, 2, 3], [1, 4, 3, 5]) == 2


def _decode_sample(tree, *, targets=None, traces=None):
    return SimpleNamespace(
        tree=tree,
        traces=traces or [[label for label in tree.activity_labels()]],
        petri_graph=SimpleNamespace(
            transition_labels=tuple(tree.activity_labels()),
        ),
        decoder_target_trees=targets or {
            name: tree for name in ("tree", "trace", "petri")
        },
        equivalence_id="family",
        metadata={},
    )


def test_evaluate_single_decode_uses_explicit_source_target(monkeypatch):
    tokenizer = TreeTokenizer(max_activities=3)
    original = ProcessTreeNode.seq(
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.activity("A1"),
    )
    trace_target = ProcessTreeNode.activity("A0")
    sample = _decode_sample(
        original,
        targets={"tree": original, "trace": trace_target, "petri": original},
    )
    simulated = []

    def simulate(tree, num_traces):
        simulated.append(tree)
        return [[str(tree)]]

    monkeypatch.setattr("proc_rosetta.benchmarks.simulate_traces", simulate)
    monkeypatch.setattr("proc_rosetta.benchmarks.tree_to_petri_net", lambda tree: object())
    tokens = tokenizer.encode_tree(trace_target, canonicalize=False)

    row = evaluate_single_decode(
        SimpleNamespace(tree_tokenizer=tokenizer),
        sample,
        tokens,
        source_name="trace",
        target_tree=trace_target,
        include_original_tree_metrics=True,
    )

    assert row["source_name"] == "trace"
    assert row["exact_tree_match"] is True
    assert row["original_tree_exact_match"] is False
    assert row["normalized_token_edit_distance"] == 0.0
    assert simulated[0] == trace_target
    assert row["tree_size"] == trace_target.size()


def test_semantic_and_deployment_duplicate_policies_use_matching_targets():
    duplicate = ProcessTreeNode.seq(
        ProcessTreeNode.activity("A0"),
        ProcessTreeNode.activity("A0"),
    )
    sample = _decode_sample(duplicate)

    semantic = decode_target_tree(sample, "tree", avoid_duplicates=False)
    deployment = decode_target_tree(sample, "tree", avoid_duplicates=True)

    assert semantic.activity_labels() == ("A0", "A0")
    assert deployment.activity_labels() == ("A0",)


def test_raw_beam_policy_allows_semantic_duplicates():
    calls = []

    class Decoder:
        def decode_beam(self, source, **kwargs):
            calls.append(kwargs)
            return torch.tensor([[1, 2]])

    decode_with_beam(
        Decoder(),
        torch.zeros((1, 2)),
        max_length=4,
        completion_policy="prefix_only",
        avoid_duplicate_activity_labels=False,
    )
    decode_with_beam(
        Decoder(),
        torch.zeros((1, 2)),
        max_length=4,
        completion_policy="bounded",
        avoid_duplicate_activity_labels=True,
    )

    assert calls[0]["avoid_duplicate_activity_labels"] is False
    assert calls[1]["avoid_duplicate_activity_labels"] is True


def test_behavior_matrix_cache_is_invalidated_when_validation_rows_change(tmp_path):
    config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=2)
    first = SyntheticProcessDataset(2, config=config, seed=31).samples
    second = SyntheticProcessDataset(2, config=config, seed=32).samples

    first_matrices = behavior_matrices(first, cache_dir=tmp_path)
    second_matrices = behavior_matrices(second, cache_dir=tmp_path)

    assert validation_split_hash(first) != validation_split_hash(second)
    assert len(list(tmp_path.glob("behavior-*.npz"))) == 2
    assert first_matrices["mean_l1"].shape == second_matrices["mean_l1"].shape


def test_behavior_matrices_match_pairwise_behavioral_distance():
    config = SyntheticConfig(max_depth=2, max_activities=4, traces_per_sample=3)
    samples = SyntheticProcessDataset(4, config=config, seed=33).samples

    matrices = behavior_matrices(samples)

    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            expected = behavioral_distance(
                samples[left].traces,
                samples[right].traces,
            )
            for name in (
                "mean_l1",
                "variant_l1",
                "directly_follows_l1",
                "length_l1",
            ):
                assert matrices[name][left, right] == pytest.approx(expected[name])


def test_validation_audit_has_no_checkpoint_or_split_path_surface(monkeypatch):
    parameters = inspect.signature(validation_audit_report).parameters
    assert "checkpoint_path" not in parameters
    assert "data_dir" not in parameters
    assert "split" not in parameters

    monkeypatch.setattr(
        "proc_rosetta.training.load_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validation audit must not load a checkpoint")
        ),
    )
    config = SyntheticConfig(max_depth=1, max_activities=3, traces_per_sample=1)
    samples = SyntheticProcessDataset(1, config=config, seed=41).samples
    tokenizer = TreeTokenizer(max_activities=3, max_arity=3)
    model = ProcRosettaModel(
        tokenizer,
        ActivityTokenizer(max_activities=3),
        latent_dim=4,
        hidden_dim=8,
        decoder_layers=1,
        tree_encoder_layers=1,
        memory_tokens=1,
    )

    report = validation_audit_report(
        model,
        samples,
        "complex",
        loss_metrics={"loss": 1.0},
        epoch=1,
        config=ValidationAuditConfig(
            decode_interval=2,
            full_interval=10,
            decode_family_count=1,
            discovery_family_count=1,
            beam_size=1,
            max_decode_length=8,
        ),
        batch_size=1,
        device="cpu",
    )

    assert report["split"] == "validation"
    assert "decode_quality" in report and "cross_modal_retrieval" in report


def test_discovery_quality_summary_helpers():
    rows = [
        {
            "model_discovered": True,
            "conformance_evaluable": True,
            "fitness": 1.0,
            "precision": 0.5,
            "f1": alignment_f1_score(1.0, 0.5),
            "error": None,
        },
        {
            "model_discovered": False,
            "conformance_evaluable": False,
            "fitness": None,
            "precision": None,
            "f1": None,
            "error": "decode:ValueError: invalid",
        },
    ]

    summary = summarize_discovery_quality(rows)

    assert alignment_f1_score(1.0, 0.5) == 0.666667
    assert fitness_precision_f1_score(1.0, 0.5) == 0.666667
    assert summary["count"] == 2
    assert summary["model_discovered_rate"] == 0.5
    assert summary["conformance_evaluable_rate"] == 0.5
    assert summary["token_replay_evaluable_rate"] == 0.5
    assert summary["mean_fitness"] == 1.0
    assert summary["mean_precision"] == 0.5
    assert summary["mean_f1"] == 0.666667
    assert summary["token_replay_error_count"] == 1

    footprint_summary = summarize_discovery_quality(rows, conformance_method="footprints")
    assert footprint_summary["footprint_evaluable_rate"] == 0.5
    assert footprint_summary["footprint_error_count"] == 1


def test_discovery_reports_reranked_f1_against_beam_oracle(monkeypatch):
    class Decoder:
        @staticmethod
        def decode_beam_candidates(source, **kwargs):
            return [[([1], 0.0), ([2], -1.0)] for _ in range(len(source.mu))]

    class Tokenizer:
        def decode_tree(self, token_ids):
            raise ValueError("force language-model reranking")

    class Model:
        tree_tokenizer = Tokenizer()
        activity_tokenizer = object()
        tree_decoder = Decoder()

        def eval(self):
            return None

        def to(self, device):
            return None

        def encode_traces(self, traces):
            return SimpleNamespace(mu=torch.zeros((len(traces), 1)))

    sample = SimpleNamespace(
        equivalence_id="family",
        metadata={"motif": "ordinary_tree", "observation_quality": "full"},
    )
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.ProcessBatchCollator",
        lambda *args: lambda samples: {"traces": torch.zeros((len(samples), 1))},
    )

    def discovery_row(model, sample, token_ids, **kwargs):
        f1 = 1.0 if token_ids == [2] else 0.5
        return {
            "model_discovered": True,
            "conformance_evaluable": True,
            "fitness": f1,
            "precision": f1,
            "f1": f1,
            "error": None,
        }

    monkeypatch.setattr(
        "proc_rosetta.benchmarks.evaluate_proc_rosetta_discovery", discovery_row
    )
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.evaluate_inductive_miner_discovery",
        lambda *args, **kwargs: discovery_row(None, None, [2]),
    )

    report = discovery_quality_report(Model(), [sample], batch_size=1, device="cpu")
    neural = report["methods"]["proc_rosetta_trace_mu"]

    assert neural["mean_f1"] == 0.5
    assert neural["mean_beam_oracle_f1"] == 1.0
    assert neural["mean_beam_oracle_f1_gap"] == 0.5


def test_observation_quality_report_includes_decode_and_exact_retrieval():
    samples = [
        SimpleNamespace(
            strong_behavior_id=f"exact-{index}",
            exact_behavior_id=f"exact-{index}",
            metadata={"observation_quality": quality},
        )
        for index, quality in enumerate(("full", "sparse", "noisy"))
    ]
    embeddings = {
        "proc_rosetta_trace_mu": np.eye(3),
        "proc_rosetta_tree_mu": np.eye(3),
    }

    report = observation_quality_report(
        samples,
        embeddings,
        {"full": {"exact_tree_match_rate": 0.75}},
    )

    assert report["full"]["exact_tree_match_rate"] == 0.75
    assert all(
        values["exact_retrieval_recall_at_1"] == 1.0
        for values in report.values()
    )


def test_footprint_fitness_and_precision_use_log_and_process_tree_footprints(monkeypatch):
    import pm4py
    from pm4py.algo.conformance.footprints import algorithm as footprint_conformance
    from pm4py.algo.conformance.footprints.util import evaluation as footprint_evaluation

    process_tree = object()
    discovered_from = []
    conformance_calls = []
    fitness_calls = []
    precision_calls = []

    def discover_footprints(value):
        discovered_from.append(value)
        return {"source": "log" if len(discovered_from) == 1 else "tree"}

    def conformance(log_footprints, tree_footprints, variant):
        conformance_calls.append((log_footprints, tree_footprints, variant))
        return {"result": "conformance"}

    def fitness(log_footprints, tree_footprints, conformance_result):
        fitness_calls.append((log_footprints, tree_footprints, conformance_result))
        return 0.75

    def precision(log_footprints, tree_footprints):
        precision_calls.append((log_footprints, tree_footprints))
        return 0.5

    monkeypatch.setattr(pm4py, "discover_footprints", discover_footprints)
    monkeypatch.setattr(footprint_conformance, "apply", conformance)
    monkeypatch.setattr(footprint_evaluation, "fp_fitness", fitness)
    monkeypatch.setattr(footprint_evaluation, "fp_precision", precision)

    result = footprint_fitness_precision([["A", "B"]], process_tree)

    assert result == (0.75, 0.5)
    assert list(discovered_from[0]["concept:name"]) == ["A", "B"]
    assert discovered_from[1] is process_tree
    assert conformance_calls == [
        (
            {"source": "log"},
            {"source": "tree"},
            footprint_conformance.Variants.LOG_EXTENSIVE,
        )
    ]
    assert fitness_calls == [
        ({"source": "log"}, {"source": "tree"}, {"result": "conformance"})
    ]
    assert precision_calls == [({"source": "log"}, {"source": "tree"})]


def test_proc_rosetta_footprints_score_the_process_tree_without_petri_conversion(monkeypatch):
    decoded_tree = object()
    pm4py_tree = object()
    scored_trees = []

    class Tokenizer:
        eos_id = 2

        def decode_tree(self, token_ids):
            return decoded_tree

    def score_footprints(traces, tree):
        scored_trees.append(tree)
        return 0.8, 0.6

    monkeypatch.setattr("proc_rosetta.benchmarks.to_pm4py_tree", lambda tree: pm4py_tree)
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.tree_to_petri_net",
        lambda tree: (_ for _ in ()).throw(AssertionError("Petri conversion must not run")),
    )
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.footprint_fitness_precision",
        score_footprints,
    )

    row = evaluate_proc_rosetta_discovery(
        SimpleNamespace(tree_tokenizer=Tokenizer()),
        SimpleNamespace(traces=[["A", "B"]]),
        [1, 2],
        conformance_method="footprints",
    )

    assert scored_trees == [pm4py_tree]
    assert row["model_discovered"] is True
    assert row["conformance_evaluable"] is True
    assert row["fitness"] == 0.8
    assert row["precision"] == 0.6
    assert row["f1"] == 0.685714


def test_inductive_miner_footprints_score_its_tree_without_petri_conversion(monkeypatch):
    pm4py_tree = object()
    scored_trees = []

    monkeypatch.setitem(
        sys.modules,
        "pm4py",
        SimpleNamespace(
            discover_process_tree_inductive=lambda log, **kwargs: pm4py_tree,
        ),
    )

    def score_footprints(traces, tree):
        scored_trees.append(tree)
        return 0.9, 0.7

    monkeypatch.setattr(
        "proc_rosetta.benchmarks.footprint_fitness_precision",
        score_footprints,
    )

    row = evaluate_inductive_miner_discovery(
        SimpleNamespace(traces=[["A", "B"]]),
        conformance_method="footprints",
    )

    assert scored_trees == [pm4py_tree]
    assert row["model_discovered"] is True
    assert row["conformance_evaluable"] is True
    assert row["fitness"] == 0.9
    assert row["precision"] == 0.7


def test_token_based_replay_fitness_and_precision_use_pm4py_token_apis(monkeypatch):
    calls = []

    def fitness(log, net, initial_marking, final_marking, **kwargs):
        calls.append(("fitness", log, net, initial_marking, final_marking, kwargs))
        return {"log_fitness": 0.8}

    def precision(log, net, initial_marking, final_marking, **kwargs):
        calls.append(("precision", log, net, initial_marking, final_marking, kwargs))
        return 0.6

    monkeypatch.setitem(
        sys.modules,
        "pm4py",
        SimpleNamespace(
            fitness_token_based_replay=fitness,
            precision_token_based_replay=precision,
        ),
    )

    result = token_based_replay_fitness_precision(
        [["A", "B"]],
        net="net",
        initial_marking="initial",
        final_marking="final",
    )

    assert result == (0.8, 0.6)
    assert [call[0] for call in calls] == ["fitness", "precision"]
    assert list(calls[0][1]["concept:name"]) == ["A", "B"]
    assert calls[0][2:5] == ("net", "initial", "final")
    assert calls[0][5] == {
        "activity_key": "concept:name",
        "timestamp_key": "time:timestamp",
        "case_id_key": "case:concept:name",
    }
    assert calls[1][5] == calls[0][5]


def test_discovery_progress_counts_both_methods_for_every_sample(monkeypatch):
    progress_details = {}
    updates = []
    postfixes = []

    class RecordingProgress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def update(self, count=1):
            updates.append(count)

        def set_postfix(self, *args, **kwargs):
            postfixes.append(args[0])

    def recording_progress_bar(total, enabled, **kwargs):
        progress_details.update(total=total, enabled=enabled, **kwargs)
        return RecordingProgress()

    class Model:
        tree_tokenizer = object()
        activity_tokenizer = object()

        class tree_decoder:
            @staticmethod
            def decode_greedy(latent, **kwargs):
                return torch.ones((len(latent), 1), dtype=torch.long)

        def eval(self):
            return None

        def to(self, device):
            return None

        def encode_traces(self, traces):
            return SimpleNamespace(mu=torch.zeros((len(traces), 1)))

    successful_row = {
        "model_discovered": True,
        "conformance_evaluable": True,
        "fitness": 1.0,
        "precision": 1.0,
        "f1": 1.0,
        "error": None,
    }
    monkeypatch.setattr("proc_rosetta.benchmarks.progress_bar", recording_progress_bar)
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.ProcessBatchCollator",
        lambda *args: lambda samples: {"traces": torch.zeros((len(samples), 1))},
    )
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.evaluate_proc_rosetta_discovery",
        lambda *args, **kwargs: successful_row.copy(),
    )
    monkeypatch.setattr(
        "proc_rosetta.benchmarks.evaluate_inductive_miner_discovery",
        lambda *args, **kwargs: successful_row.copy(),
    )

    report = discovery_quality_report(
        Model(),
        [object(), object(), object()],
        batch_size=2,
        device="cpu",
        show_progress=True,
    )

    assert progress_details == {
        "total": 6,
        "enabled": True,
        "desc": "Discovery replays",
        "unit": "replay",
    }
    assert sum(updates) == 6
    assert postfixes[0]["remaining"] == 5
    assert postfixes[-1]["remaining"] == 0
    assert report["methods"]["proc_rosetta_trace_mu"]["count"] == 3
    assert report["methods"]["inductive_miner"]["count"] == 3


def test_human_report_mentions_method_comparison():
    report = {
        "split": "test",
        "sample_count": 2,
        "loss_metrics": {
            "loss": 1.0,
            "tree_reconstruction": 0.3,
            "trace_to_tree": 0.3,
            "petri_to_tree": 0.3,
            "contrastive": 0.1,
            "kl": 0.2,
        },
        "behavioral_distance_summary": {
            "mean": 1.2,
            "min": 0.4,
            "max": 2.0,
            "pair_count": 1,
            "std": 0.0,
        },
        "method_ranking": [
            {
                "method": "proc_rosetta_fused_mu",
                "behavior_spearman": 0.7,
                "nearest_neighbor_behavior_l1": 0.5,
                "improvement_over_random": 0.7,
            }
        ],
        "embedding_methods": {
            "proc_rosetta_fused_mu": {"vector_statistics": {"dimension": 8}, "available": True}
        },
        "method_comparisons_against_proc_rosetta_fused_mu": {
            "reference_method": "proc_rosetta_fused_mu",
            "available": True,
            "comparisons": {
                "pm4py_colonna_petri_node2vec": {
                    "pairwise_distance_spearman_agreement": 0.8,
                    "top1_neighbor_overlap": 0.5,
                    "behavior_spearman_delta_vs_reference": 0.1,
                    "nearest_neighbor_behavior_l1_delta_vs_reference": -0.2,
                }
            },
        },
        "cross_modal_retrieval": {
            "tree_to_trace": {"top1_accuracy": 1.0, "mrr": 1.0, "mean_rank": 1.0}
        },
        "decode_quality": {
            "methods": {
                "proc_rosetta_tree_mu": {
                    "terminated_rate": 1.0,
                    "valid_tree_rate": 1.0,
                    "exact_tree_match_rate": 0.5,
                    "petri_conversion_rate": 1.0,
                    "mean_behavior_l1": 0.2,
                    "mean_normalized_token_edit_distance": 0.1,
                },
                "proc_rosetta_fused_mu": {
                    "terminated_rate": 1.0,
                    "valid_tree_rate": 1.0,
                    "exact_tree_match_rate": 0.5,
                    "petri_conversion_rate": 1.0,
                    "mean_behavior_l1": 0.2,
                    "mean_normalized_token_edit_distance": 0.1,
                },
            }
        },
        "discovery_quality": {
            "methods": {
                "proc_rosetta_trace_mu": {
                    "model_discovered_rate": 1.0,
                    "token_replay_evaluable_rate": 1.0,
                    "mean_fitness": 0.8,
                    "mean_precision": 0.6,
                    "mean_f1": 0.685714,
                },
                "inductive_miner": {
                    "model_discovered_rate": 1.0,
                    "token_replay_evaluable_rate": 1.0,
                    "mean_fitness": 1.0,
                    "mean_precision": 0.9,
                    "mean_f1": 0.947368,
                },
            }
        },
    }

    text = format_human_test_report(report)

    assert "ProcRosetta Test Report" in text
    assert "Decode quality" in text
    assert "Process discovery quality" in text
    assert "Inductive Miner" in text
    assert "pm4py Petri Node2Vec vs ProcRosetta fused" in text
    assert "behavior rho" in text
