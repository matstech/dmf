# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from dmf.memory.answerability_rerank import (
    AnswerabilityFeatureExtractor,
    AnswerabilityRanker,
    rerank_answerable_evidence,
)
from dmf.memory.query_understanding import parse_query_frame
from dmf.models.analysis import InteractionProvenance
from dmf.models.memory import (
    MemoryCard,
    MemoryCardProvenance,
    MemoryCardTimeAnchor,
    MemoryCardValidity,
    RetrievedEvidence,
)
from dmf.models.raw_ltm import RawLTMRecord


def _card(
    *,
    card_id: str = "card:record:1:0",
    source_record_id: str = "record:1",
    subject: str = "Alice",
    predicate: str = "prefer",
    object_value: str = "green tea",
    kind: str = "preference",
    status: str = "active",
    relative_order: int = 1,
) -> MemoryCard:
    return MemoryCard(
        card_id=card_id,
        kind=kind,
        subject=subject,
        predicate=predicate,
        object=object_value,
        time_anchor=MemoryCardTimeAnchor(
            relative_order=relative_order,
            turn_id=relative_order,
        ),
        validity=MemoryCardValidity(status=status),
        provenance=MemoryCardProvenance(
            source_record_id=source_record_id,
            speaker_role="user",
            source_turn=relative_order,
        ),
        confidence=0.9,
        surface_forms=[f"{subject} prefers {object_value}"],
    )


def _card_evidence(
    card: MemoryCard,
    *,
    semantic_score: float | None = 0.4,
    symbolic_score: float | None = 0.8,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=card.card_id,
        evidence_type="card",
        source="test",
        semantic_score=semantic_score,
        symbolic_score=symbolic_score,
        temporal_score=float(card.time_anchor.relative_order or 0),
        answerability_features={
            "kind": card.kind,
            "validity_status": card.validity.status,
        },
        render_payload={"card": card.to_dict()},
        provenance={
            "source_record_id": card.provenance.source_record_id,
            "speaker_role": card.provenance.speaker_role,
            "source_turn": card.provenance.source_turn,
            "card_id": card.card_id,
        },
    )


def _raw_evidence(
    *,
    evidence_id: str,
    text: str,
    role: str = "user",
    semantic_score: float | None = 0.5,
    temporal_score: float | None = 1.0,
) -> RetrievedEvidence:
    record = RawLTMRecord(
        record_id=evidence_id,
        interaction_id=int(temporal_score or 1),
        role=role,
        text=text,
        created_at=float(temporal_score or 1),
        provenance=InteractionProvenance(
            role=role,
            source_turn=int(temporal_score or 1),
        ),
    )
    return RetrievedEvidence(
        evidence_id=evidence_id,
        evidence_type="raw_turn",
        source="test",
        semantic_score=semantic_score,
        temporal_score=temporal_score,
        render_payload={"record": record.to_dict()},
        provenance={
            "source_record_id": evidence_id,
            "speaker_role": role,
            "source_turn": record.provenance.source_turn,
        },
    )


def test_feature_extractor_scores_entity_subject_predicate_and_chain() -> None:
    query = parse_query_frame("What drink does Alice prefer?")
    diagnostics = AnswerabilityFeatureExtractor().extract(
        query,
        _card_evidence(_card()),
        baseline_rank=2,
    )

    assert diagnostics.features["semantic_similarity"] == 0.4
    assert diagnostics.features["entity_overlap"] > 0
    assert diagnostics.features["subject_match"] == 1.0
    assert diagnostics.features["predicate_match"] == 1.0
    assert diagnostics.features["answer_span_likelihood"] == 1.0
    assert diagnostics.features["evidence_chain_availability"] == 1.0
    assert diagnostics.baseline_rank == 2


def test_ranker_promotes_more_answerable_candidate_over_similarity_only() -> None:
    query = parse_query_frame("What drink does Alice prefer?")
    similarity_only = _raw_evidence(
        evidence_id="record:2",
        text="Bob mentioned green tea during a trip.",
        semantic_score=0.95,
    )
    answerable = _card_evidence(_card(), semantic_score=0.35)

    ranked = rerank_answerable_evidence(query, [similarity_only, answerable])

    assert [candidate.evidence_id for candidate in ranked] == [
        "card:record:1:0",
        "record:2",
    ]
    assert ranked[0].answerability_features["answerability_score"] > ranked[
        1
    ].answerability_features["answerability_score"]


def test_ranker_penalizes_topic_only_time_candidate_without_predicate_support() -> None:
    query = parse_query_frame("When did Alice paint a mural?")
    topic_only = _card_evidence(
        _card(
            kind="event",
            predicate="mentions_event",
            object_value="Alice plays clarinet and started when young.",
        ),
        semantic_score=0.85,
        symbolic_score=0.9,
    )
    specific = _card_evidence(
        _card(
            card_id="card:record:2:0",
            source_record_id="record:2",
            kind="event",
            predicate="mentions_event",
            object_value="Alice painted a mural after visiting the beach last week.",
        ),
        semantic_score=0.35,
        symbolic_score=0.7,
    )

    ranked = rerank_answerable_evidence(query, [topic_only, specific])
    diagnostics = {
        item.evidence_id: item.answerability_features["answerability_diagnostics"]
        for item in ranked
    }

    assert [candidate.evidence_id for candidate in ranked] == [
        "card:record:2:0",
        "card:record:1:0",
    ]
    assert diagnostics["card:record:1:0"]["penalties"]["topic_only_penalty"] == 1.0
    assert (
        diagnostics["card:record:2:0"]["features"]["answer_span_likelihood"]
        > diagnostics["card:record:1:0"]["features"]["answer_span_likelihood"]
    )


def test_ranker_preserves_baseline_order_when_features_do_not_discriminate() -> None:
    query = parse_query_frame("Tell me anything")
    first = RetrievedEvidence(evidence_id="a", evidence_type="raw_turn", source="baseline")
    second = RetrievedEvidence(evidence_id="b", evidence_type="raw_turn", source="baseline")

    ranked = AnswerabilityRanker().rerank(query, [first, second])

    assert [candidate.evidence_id for candidate in ranked] == ["a", "b"]
    assert ranked[0].answerability_features["answerability_diagnostics"][
        "baseline_rank"
    ] == 0


def test_stale_and_ack_social_penalties_lower_candidates() -> None:
    query = parse_query_frame("What does Alice currently prefer?")
    active = _card_evidence(_card(kind="current_state"))
    stale = _card_evidence(
        _card(
            card_id="card:record:2:0",
            source_record_id="record:2",
            status="superseded",
        )
    )
    ack = _raw_evidence(
        evidence_id="record:3",
        text="Thanks, got it.",
        role="assistant",
    )

    ranked = rerank_answerable_evidence(query, [stale, ack, active])
    diagnostics = {
        item.evidence_id: item.answerability_features["answerability_diagnostics"]
        for item in ranked
    }

    assert ranked[0].evidence_id == "card:record:1:0"
    assert diagnostics["card:record:2:0"]["penalties"]["stale_fact_penalty"] == 1.0
    assert diagnostics["record:3"]["penalties"]["ack_social_penalty"] == 1.0


def test_temporal_and_current_compatibility_distinguish_query_intent() -> None:
    current_query = parse_query_frame("What is Alice's current location?")
    historical_query = parse_query_frame("What was Alice's previous location?")
    current = _card_evidence(
        _card(kind="current_state", predicate="live", object_value="Rome")
    )
    old = _card_evidence(
        _card(
            card_id="card:record:2:0",
            source_record_id="record:2",
            kind="current_state",
            predicate="live",
            object_value="Paris",
            status="superseded",
        )
    )

    current_features = AnswerabilityFeatureExtractor().extract(current_query, current)
    old_for_current = AnswerabilityFeatureExtractor().extract(current_query, old)
    old_for_history = AnswerabilityFeatureExtractor().extract(historical_query, old)

    assert current_features.features["current_state_compatibility"] == 1.0
    assert old_for_current.features["temporal_compatibility"] == 0.0
    assert old_for_history.features["temporal_compatibility"] == 1.0


def test_future_temporal_query_penalizes_past_event_candidates() -> None:
    query = parse_query_frame("When is Mira planning on going camping?")
    past = _raw_evidence(
        evidence_id="record:past",
        text="Mira went camping last weekend.",
        temporal_score=1.0,
    )
    planned = _raw_evidence(
        evidence_id="record:future",
        text="Mira is planning to go camping next month.",
        temporal_score=2.0,
    )

    ranked = rerank_answerable_evidence(query, [past, planned])
    diagnostics = {
        item.evidence_id: item.answerability_features["answerability_diagnostics"]
        for item in ranked
    }

    assert [candidate.evidence_id for candidate in ranked] == [
        "record:future",
        "record:past",
    ]
    assert (
        diagnostics["record:future"]["features"]["temporal_compatibility"]
        > diagnostics["record:past"]["features"]["temporal_compatibility"]
    )


def test_past_temporal_query_penalizes_future_plan_candidates() -> None:
    query = parse_query_frame("When did Nora give a speech at a school?")
    planned = _raw_evidence(
        evidence_id="record:future",
        text="Nora will give a speech at the school next week.",
        temporal_score=2.0,
    )
    completed = _raw_evidence(
        evidence_id="record:past",
        text="Nora gave a speech at the school yesterday.",
        temporal_score=1.0,
    )

    ranked = rerank_answerable_evidence(query, [planned, completed])
    diagnostics = {
        item.evidence_id: item.answerability_features["answerability_diagnostics"]
        for item in ranked
    }

    assert [candidate.evidence_id for candidate in ranked] == [
        "record:past",
        "record:future",
    ]
    assert diagnostics["record:future"]["features"]["temporal_compatibility"] == 0.0
    assert diagnostics["record:past"]["features"]["temporal_compatibility"] == 1.0


def test_answerability_diagnostics_are_json_serializable() -> None:
    query = parse_query_frame("What does Alice prefer?")
    ranked = rerank_answerable_evidence(query, [_card_evidence(_card())])

    payload = json.loads(json.dumps(ranked[0].to_dict()))
    diagnostics = payload["answerability_features"]["answerability_diagnostics"]

    assert isinstance(diagnostics["score"], float)
    assert diagnostics["features"]["subject_match"] == 1.0
    assert diagnostics["baseline_key"]["symbolic_score"] == 0.8
