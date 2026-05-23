"""
tests/test_models.py
--------------------
Unit tests for dmf/core/models.py — AnalysisReport dataclass.

Coverage:
  - Correct initialization of all required fields.
  - raw_metadata defaults to {} when omitted.
  - to_dict() returns a plain dict with matching values.
  - to_json() returns a valid, parseable JSON string.
  - to_json() output contains all six expected field keys.
  - to_json() correctly round-trips a non-trivial raw_metadata payload.
  - to_dict() / to_json() produce deep copies (mutations don't affect the original).
"""

import json

import pytest

from dmf.models.analysis import (
    AnalysisReport,
    InteractionProvenance,
    InteractionSignals,
    MemoryLineage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_report() -> AnalysisReport:
    """AnalysisReport with every required field set; raw_metadata omitted."""
    return AnalysisReport(
        info_density=0.45,
        sentiment_abs=0.72,
        entity_count=3,
        is_system_prompt=False,
        latency_ms=12.5,
    )


@pytest.fixture()
def full_report() -> AnalysisReport:
    """AnalysisReport with all fields, including a rich raw_metadata payload."""
    return AnalysisReport(
        info_density=0.60,
        sentiment_abs=0.10,
        entity_count=7,
        is_system_prompt=True,
        latency_ms=8.3,
        topic_identity="preference|favorite|dish",
        topic_value="risotto",
        is_query_like=True,
        is_ack_like=True,
        raw_metadata={
            "pos_counts": {"NOUN": 5, "VERB": 3, "ADJ": 1, "PROPN": 2},
            "token_count": 22,
            "raw_tokens": ["The", "DMF", "engine", "processes", "text"],
        },
        provenance=InteractionProvenance(
            role="user",
            source_turn=11,
            is_user_correction=True,
            is_constraint=True,
        ),
        signals=InteractionSignals(
            is_current_state=True,
            is_preference=True,
            has_replacement=True,
            temporal_markers=["current"],
            cue_phrases=["favorite"],
        ),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_all_required_fields_are_stored_correctly(minimal_report: AnalysisReport):
    """Every required field must be stored exactly as provided."""
    assert minimal_report.info_density == 0.45
    assert minimal_report.sentiment_abs == 0.72
    assert minimal_report.entity_count == 3
    assert minimal_report.is_system_prompt is False
    assert minimal_report.latency_ms == 12.5


def test_raw_metadata_defaults_to_empty_dict_when_omitted(minimal_report: AnalysisReport):
    """raw_metadata must default to {} — not share state across instances."""
    assert minimal_report.raw_metadata == {}


def test_provenance_defaults_to_empty_object_when_omitted(minimal_report: AnalysisReport):
    """provenance must default to an empty structured object."""
    assert minimal_report.provenance == InteractionProvenance()


def test_signals_defaults_to_empty_object_when_omitted(minimal_report: AnalysisReport):
    """signals must default to an empty structured object."""
    assert minimal_report.signals == InteractionSignals()


def test_raw_metadata_default_is_not_shared_across_instances():
    """Each instance must receive its own dict, not the same mutable object."""
    report_a = AnalysisReport(
        info_density=0.1, sentiment_abs=0.0, entity_count=0,
        is_system_prompt=False, latency_ms=1.0,
    )
    report_b = AnalysisReport(
        info_density=0.2, sentiment_abs=0.0, entity_count=0,
        is_system_prompt=False, latency_ms=1.0,
    )
    report_a.raw_metadata["key"] = "value"
    # Mutating report_a's metadata must not affect report_b
    assert "key" not in report_b.raw_metadata


def test_is_system_prompt_stores_true_correctly(full_report: AnalysisReport):
    """is_system_prompt=True must be stored as the boolean True (not truthy)."""
    assert full_report.is_system_prompt is True


def test_entity_count_stores_zero_correctly():
    """entity_count=0 is a valid value and must not be treated as falsy."""
    report = AnalysisReport(
        info_density=0.0, sentiment_abs=0.0, entity_count=0,
        is_system_prompt=False, latency_ms=0.0,
    )
    assert report.entity_count == 0


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_returns_a_plain_dict(minimal_report: AnalysisReport):
    """to_dict() must return a plain dict, not an AnalysisReport instance."""
    result = minimal_report.to_dict()
    assert isinstance(result, dict)


def test_to_dict_contains_all_fields(minimal_report: AnalysisReport):
    """to_dict() must expose every AnalysisReport field as a top-level key."""
    expected_keys = {
        "info_density", "sentiment_abs", "entity_count",
        "is_system_prompt", "latency_ms", "semantic_divergence",
        "survival_score", "status", "raw_metadata", "provenance", "signals",
        "topic_identity", "topic_value", "is_query_like", "is_ack_like",
    }
    assert set(minimal_report.to_dict().keys()) == expected_keys


def test_to_dict_values_match_report_fields(full_report: AnalysisReport):
    """to_dict() values must match the originating report fields exactly."""
    d = full_report.to_dict()
    assert d["info_density"] == full_report.info_density
    assert d["sentiment_abs"] == full_report.sentiment_abs
    assert d["entity_count"] == full_report.entity_count
    assert d["is_system_prompt"] == full_report.is_system_prompt
    assert d["latency_ms"] == full_report.latency_ms
    assert d["raw_metadata"] == full_report.raw_metadata
    assert d["provenance"]["role"] == "user"
    assert d["provenance"]["source_turn"] == 11
    assert d["provenance"]["is_user_correction"] is True
    assert d["signals"]["is_current_state"] is True
    assert d["signals"]["is_preference"] is True
    assert d["signals"]["cue_phrases"] == ["favorite"]
    assert d["topic_identity"] == "preference|favorite|dish"
    assert d["topic_value"] == "risotto"
    assert d["is_query_like"] is True
    assert d["is_ack_like"] is True


def test_to_dict_produces_a_deep_copy(full_report: AnalysisReport):
    """Mutating the to_dict() result must not alter the original report."""
    d = full_report.to_dict()
    d["raw_metadata"]["injected_key"] = "should_not_propagate"
    assert "injected_key" not in full_report.raw_metadata


# ---------------------------------------------------------------------------
# to_json()
# ---------------------------------------------------------------------------

def test_to_json_returns_a_string(minimal_report: AnalysisReport):
    """to_json() must return a str, not bytes or dict."""
    assert isinstance(minimal_report.to_json(), str)


def test_to_json_is_valid_json(minimal_report: AnalysisReport):
    """to_json() output must be parseable by json.loads() without raising."""
    try:
        json.loads(minimal_report.to_json())
    except json.JSONDecodeError as exc:
        pytest.fail(f"to_json() produced invalid JSON: {exc}")


def test_to_json_contains_all_field_keys(minimal_report: AnalysisReport):
    """Parsed to_json() output must contain all AnalysisReport field keys."""
    parsed = json.loads(minimal_report.to_json())
    expected_keys = {
        "info_density", "sentiment_abs", "entity_count",
        "is_system_prompt", "latency_ms", "semantic_divergence",
        "survival_score", "status", "raw_metadata", "provenance", "signals",
        "topic_identity", "topic_value", "is_query_like", "is_ack_like",
    }
    assert set(parsed.keys()) == expected_keys


def test_to_json_values_round_trip_correctly(full_report: AnalysisReport):
    """Values must survive a to_json() → json.loads() round-trip unchanged."""
    parsed = json.loads(full_report.to_json())
    assert parsed["info_density"] == full_report.info_density
    assert parsed["sentiment_abs"] == full_report.sentiment_abs
    assert parsed["entity_count"] == full_report.entity_count
    assert parsed["is_system_prompt"] == full_report.is_system_prompt
    assert parsed["latency_ms"] == full_report.latency_ms
    assert parsed["topic_identity"] == "preference|favorite|dish"
    assert parsed["topic_value"] == "risotto"
    assert parsed["is_query_like"] is True
    assert parsed["is_ack_like"] is True


def test_to_json_round_trips_non_trivial_raw_metadata(full_report: AnalysisReport):
    """A complex raw_metadata dict must survive JSON round-trip without data loss."""
    parsed = json.loads(full_report.to_json())
    assert parsed["raw_metadata"]["pos_counts"] == {"NOUN": 5, "VERB": 3, "ADJ": 1, "PROPN": 2}
    assert parsed["raw_metadata"]["token_count"] == 22
    assert parsed["raw_metadata"]["raw_tokens"] == [
        "The", "DMF", "engine", "processes", "text"
    ]


def test_to_json_round_trips_structured_provenance(full_report: AnalysisReport):
    """Structured provenance must survive JSON round-trip unchanged."""
    parsed = json.loads(full_report.to_json())
    assert parsed["provenance"] == {
        "role": "user",
        "source_turn": 11,
        "is_user_correction": True,
        "is_preference_update": False,
        "is_constraint": True,
        "derived_from_model": False,
        "corrected_by_user": False,
    }


def test_to_json_round_trips_structured_signals(full_report: AnalysisReport):
    """Structured conversational signals must survive JSON round-trip unchanged."""
    parsed = json.loads(full_report.to_json())
    assert parsed["signals"] == {
        "is_current_state": True,
        "is_past_state": False,
        "is_preference": True,
        "is_constraint": False,
        "is_correction": False,
        "has_negation": False,
        "has_replacement": True,
        "operational_weight": 0.0,
        "personal_relevance": 0.0,
        "quantitative_relevance": 0.0,
        "task_relevance": 0.0,
        "temporal_markers": ["current"],
        "cue_phrases": ["favorite"],
    }


def test_interaction_signals_defaults_are_empty_and_serializable() -> None:
    signals = InteractionSignals()
    assert json.loads(json.dumps(signals.__dict__)) == {
        "is_current_state": False,
        "is_past_state": False,
        "is_preference": False,
        "is_constraint": False,
        "is_correction": False,
        "has_negation": False,
        "has_replacement": False,
        "operational_weight": 0.0,
        "personal_relevance": 0.0,
        "quantitative_relevance": 0.0,
        "task_relevance": 0.0,
        "temporal_markers": [],
        "cue_phrases": [],
    }


def test_memory_lineage_defaults_to_empty_relationship_lists() -> None:
    lineage = MemoryLineage()
    assert lineage.supersedes == []
    assert lineage.conflicts_with == []
    assert lineage.corrects == []
    assert lineage.invalidates == []


def test_memory_lineage_round_trips_via_dataclasses_asdict() -> None:
    lineage = MemoryLineage(
        supersedes=["record:1"],
        conflicts_with=["record:2"],
        corrects=["record:3"],
        invalidates=["record:4"],
    )
    assert json.loads(json.dumps(lineage.__dict__)) == {
        "supersedes": ["record:1"],
        "conflicts_with": ["record:2"],
        "corrects": ["record:3"],
        "invalidates": ["record:4"],
    }


def test_memory_lineage_to_dict_matches_dataclass_shape() -> None:
    lineage = MemoryLineage(corrects=["record:3"])
    assert lineage.to_dict() == {
        "supersedes": [],
        "conflicts_with": [],
        "corrects": ["record:3"],
        "invalidates": [],
    }


def test_memory_lineage_has_relationships_detects_non_empty_edges() -> None:
    assert MemoryLineage().has_relationships() is False
    assert MemoryLineage(invalidates=["record:4"]).has_relationships() is True
