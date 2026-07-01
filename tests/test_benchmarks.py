import numpy as np

from proc_rosetta.benchmarks import (
    activity_count_features,
    evaluate_embedding_method,
    format_human_test_report,
    levenshtein_distance,
    retrieval_metrics,
    trim_tree_token_sequence,
)
from proc_rosetta.tokenizers import TreeTokenizer


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
    }

    text = format_human_test_report(report)

    assert "ProcRosetta Test Report" in text
    assert "Decode quality" in text
    assert "pm4py Petri Node2Vec vs ProcRosetta fused" in text
    assert "behavior rho" in text
