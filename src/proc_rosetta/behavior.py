from __future__ import annotations

from collections import Counter
from math import log
from typing import Hashable, Iterable, Mapping, Sequence

Trace = Sequence[str]
Distribution = dict[Hashable, float]


def _normalize(counter: Counter[Hashable]) -> Distribution:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def trace_variant_distribution(traces: Iterable[Trace]) -> Distribution:
    return _normalize(Counter(tuple(trace) for trace in traces))


def trace_length_distribution(traces: Iterable[Trace]) -> Distribution:
    return _normalize(Counter(len(trace) for trace in traces))


def activity_frequency_distribution(traces: Iterable[Trace]) -> Distribution:
    return _normalize(Counter(activity for trace in traces for activity in trace))


def start_activity_distribution(traces: Iterable[Trace]) -> Distribution:
    return _normalize(Counter(trace[0] for trace in traces if trace))


def end_activity_distribution(traces: Iterable[Trace]) -> Distribution:
    return _normalize(Counter(trace[-1] for trace in traces if trace))


def directly_follows_distribution(
    traces: Iterable[Trace],
    include_boundaries: bool = True,
) -> Distribution:
    counts: Counter[Hashable] = Counter()
    for trace in traces:
        events = list(trace)
        if include_boundaries:
            events = ["<start>", *events, "<end>"]
        counts.update(zip(events, events[1:]))
    return _normalize(counts)


def l1_distribution_distance(left: Mapping[Hashable, float], right: Mapping[Hashable, float]) -> float:
    keys = set(left) | set(right)
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def jensen_shannon_divergence(left: Mapping[Hashable, float], right: Mapping[Hashable, float]) -> float:
    keys = set(left) | set(right)
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}
    return 0.5 * _kl_divergence(left, midpoint, keys) + 0.5 * _kl_divergence(right, midpoint, keys)


def behavioral_distance(reference: Iterable[Trace], candidate: Iterable[Trace]) -> dict[str, float]:
    reference = list(reference)
    candidate = list(candidate)
    variant_l1 = l1_distribution_distance(
        trace_variant_distribution(reference),
        trace_variant_distribution(candidate),
    )
    directly_follows_l1 = l1_distribution_distance(
        directly_follows_distribution(reference),
        directly_follows_distribution(candidate),
    )
    length_l1 = l1_distribution_distance(
        trace_length_distribution(reference),
        trace_length_distribution(candidate),
    )
    activity_l1 = l1_distribution_distance(
        activity_frequency_distribution(reference),
        activity_frequency_distribution(candidate),
    )
    start_activity_l1 = l1_distribution_distance(
        start_activity_distribution(reference),
        start_activity_distribution(candidate),
    )
    end_activity_l1 = l1_distribution_distance(
        end_activity_distribution(reference),
        end_activity_distribution(candidate),
    )
    reference_mean_length = sum(map(len, reference)) / max(len(reference), 1)
    candidate_mean_length = sum(map(len, candidate)) / max(len(candidate), 1)
    mean_trace_length_difference = abs(reference_mean_length - candidate_mean_length)
    return {
        "variant_l1": variant_l1,
        "directly_follows_l1": directly_follows_l1,
        "length_l1": length_l1,
        "activity_frequency_l1": activity_l1,
        "start_activity_l1": start_activity_l1,
        "end_activity_l1": end_activity_l1,
        "mean_trace_length_difference": mean_trace_length_difference,
        # Preserve the established aggregate used by benchmark reports while
        # exposing the newly added activity component separately.
        "mean_l1": (variant_l1 + directly_follows_l1 + length_l1) / 3.0,
        "aggregate_distribution_l1": (
            variant_l1 + directly_follows_l1 + length_l1 + activity_l1
        )
        / 4.0,
    }


def _kl_divergence(
    left: Mapping[Hashable, float],
    right: Mapping[Hashable, float],
    keys: Iterable[Hashable],
) -> float:
    total = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        if p > 0.0 and q > 0.0:
            total += p * log(p / q)
    return total
