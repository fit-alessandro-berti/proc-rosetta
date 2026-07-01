from proc_rosetta.behavior import (
    behavioral_distance,
    directly_follows_distribution,
    trace_length_distribution,
    trace_variant_distribution,
)


def test_behavioral_distance_is_zero_for_identical_trace_multisets():
    traces = [["A0", "A1"], ["A0", "A2"]]

    distance = behavioral_distance(traces, traces)

    assert distance["mean_l1"] == 0.0
    assert trace_variant_distribution(traces)[("A0", "A1")] == 0.5
    assert trace_length_distribution(traces)[2] == 1.0
    assert directly_follows_distribution(traces)[("<start>", "A0")] == 1 / 3


def test_behavioral_distance_detects_control_flow_change():
    reference = [["A0", "A1"], ["A0", "A1"]]
    candidate = [["A1", "A0"], ["A1", "A0"]]

    distance = behavioral_distance(reference, candidate)

    assert distance["variant_l1"] > 0.0
    assert distance["directly_follows_l1"] > 0.0
    assert distance["length_l1"] == 0.0
