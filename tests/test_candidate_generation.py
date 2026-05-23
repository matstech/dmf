from __future__ import annotations

import json

import numpy as np

from dmf.memory.candidate_generation import (
    CandidateGenerationConfig,
    CandidateGenerator,
    CardSemanticRetriever,
    CardSymbolicRetriever,
    DeterministicCardSemanticRetriever,
    HardFilterContext,
    RawLexicalRetriever,
    RawSemanticRetriever,
    apply_hard_filters,
    merge_dedupe_candidates,
)
from dmf.memory.query_understanding import parse_query_frame
from dmf.memory.temporal_memory import TemporalMemory
from dmf.models.analysis import AnalysisReport, InteractionProvenance, InteractionSignals
from dmf.models.ltm_hook import LTMHook
from dmf.models.memory import (
    MemoryCard,
    MemoryCardProvenance,
    MemoryCardTimeAnchor,
    MemoryCardValidity,
    RetrievedEvidence,
)
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit


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
        time_anchor=MemoryCardTimeAnchor(relative_order=relative_order, turn_id=relative_order),
        validity=MemoryCardValidity(status=status),
        provenance=MemoryCardProvenance(
            source_record_id=source_record_id,
            speaker_role="user",
            source_turn=relative_order,
        ),
        confidence=0.8,
        surface_forms=[f"{subject} prefers {object_value}"],
    )


def _raw_record(
    *,
    record_id: str = "record:10",
    text: str = "Alice prefers green tea in the afternoon.",
    interaction_id: int = 10,
    role: str = "user",
) -> RawLTMRecord:
    return RawLTMRecord(
        record_id=record_id,
        interaction_id=interaction_id,
        role=role,
        text=text,
        created_at=float(interaction_id),
        provenance=InteractionProvenance(role=role, source_turn=interaction_id),
    )


class _FakeHook(LTMHook):
    def __init__(self, hits: list[RawRecallHit]) -> None:
        self.hits = hits
        self.seen_k: int | None = None

    def archive(self, entry) -> None:  # noqa: ANN001
        pass

    def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:
        self.seen_k = k
        return self.hits[:k]


def test_retrieved_evidence_serialization_round_trips() -> None:
    evidence = RetrievedEvidence(
        evidence_id="card:record:1:0",
        evidence_type="card",
        source="card_symbolic",
        semantic_score=0.2,
        symbolic_score=1.0,
        temporal_score=3.0,
        answerability_features={"kind": "preference"},
        render_payload={"card": _card().to_dict()},
        provenance={"channels": ["card_symbolic"], "source_record_id": "record:1"},
    )

    parsed = RetrievedEvidence.from_dict(json.loads(json.dumps(evidence.to_dict())))

    assert parsed == evidence


def test_card_semantic_retrieval_uses_deterministic_provisional_overlap() -> None:
    query = parse_query_frame("What drink does Alice prefer?")
    cards = [
        _card(object_value="green tea"),
        _card(
            card_id="card:record:2:0",
            source_record_id="record:2",
            subject="Bob",
            object_value="jazz",
        ),
    ]

    results = DeterministicCardSemanticRetriever().retrieve(query, cards, k=1)

    assert [result.evidence_id for result in results] == ["card:record:1:0"]
    assert results[0].source == "card_semantic"
    assert results[0].semantic_score is not None
    assert results[0].render_payload["card"]["object"] == "green tea"


def test_card_symbolic_lookup_matches_subject_and_predicate() -> None:
    query = parse_query_frame("What does Alice prefer?")

    results = CardSymbolicRetriever().retrieve(query, [_card()], k=5)

    assert [result.evidence_id for result in results] == ["card:record:1:0"]
    assert results[0].source == "card_symbolic"
    assert results[0].symbolic_score is not None


def test_card_symbolic_lookup_rejects_topic_only_card_for_specific_event_query() -> None:
    query = parse_query_frame("When did Alice paint a mural?")
    generic = _card(
        card_id="card:record:2:0",
        source_record_id="record:2",
        kind="event",
        predicate="mentions_event",
        object_value="Alice plays clarinet and started when young.",
    )
    specific = _card(
        card_id="card:record:3:0",
        source_record_id="record:3",
        kind="event",
        predicate="mentions_event",
        object_value="Alice painted a mural after visiting the beach last week.",
    )

    results = CardSymbolicRetriever().retrieve(query, [generic, specific], k=5)

    assert [result.evidence_id for result in results] == ["card:record:3:0"]
    assert results[0].symbolic_score is not None


def test_raw_semantic_retrieval_adapts_existing_ltm_hook() -> None:
    record = _raw_record()
    hook = _FakeHook([RawRecallHit(record=record, similarity_score=0.91)])
    query = parse_query_frame("Alice tea", query_embedding=[1.0, 0.0])

    results = RawSemanticRetriever(hook).retrieve(query, k=3)

    assert hook.seen_k == 3
    assert [result.evidence_id for result in results] == ["record:10"]
    assert results[0].source == "raw_semantic"
    assert results[0].semantic_score == 0.91


def test_raw_lexical_retrieval_searches_raw_turn_text() -> None:
    query = parse_query_frame("Which tea does Alice prefer?")
    records = [
        _raw_record(),
        _raw_record(record_id="record:11", text="Bob booked flights.", interaction_id=11),
    ]

    results = RawLexicalRetriever(records).retrieve(query, k=5)

    assert [result.evidence_id for result in results] == ["record:10"]
    assert results[0].source == "raw_lexical"
    assert results[0].symbolic_score is not None


def test_raw_lexical_retrieval_searches_speaker_role() -> None:
    query = parse_query_frame("When did Nora paint a mural?")
    records = [
        _raw_record(
            record_id="record:1",
            text="Image: query painting mural. The image shows a mural.",
            role="nora",
        ),
        _raw_record(
            record_id="record:2",
            text="Leo painted a mural after visiting the beach.",
            role="leo",
        ),
    ]

    results = RawLexicalRetriever(records).retrieve(query, k=1)

    assert [result.evidence_id for result in results] == ["record:1"]


def test_merge_dedupe_combines_channel_provenance_and_scores() -> None:
    card = _card()
    query = parse_query_frame("What does Alice prefer?")
    semantic = DeterministicCardSemanticRetriever().retrieve(query, [card], k=1)[0]
    symbolic = CardSymbolicRetriever().retrieve(query, [card], k=1)[0]

    merged = merge_dedupe_candidates([semantic, symbolic])

    assert len(merged) == 1
    assert merged[0].evidence_id == card.card_id
    assert merged[0].semantic_score is not None
    assert merged[0].symbolic_score is not None
    assert merged[0].provenance["channels"] == ["card_semantic", "card_symbolic"]


def test_hard_filters_drop_active_invalid_entity_and_type_mismatches() -> None:
    query = parse_query_frame("What does Alice prefer?")
    candidates = [
        CardSymbolicRetriever().retrieve(query, [_card(source_record_id="record:active")], k=1)[0],
        CardSymbolicRetriever().retrieve(
            query,
            [_card(card_id="card:record:2:0", source_record_id="record:2", status="superseded")],
            k=1,
        )[0],
        DeterministicCardSemanticRetriever().retrieve(
            query,
            [
                _card(
                    card_id="card:record:3:0",
                    source_record_id="record:3",
                    subject="Bob",
                )
            ],
            k=1,
        )[0],
        RetrievedEvidence(
            evidence_id="record:4",
            evidence_type="raw_turn",
            source="raw_lexical",
            render_payload={
                "record": _raw_record(
                    record_id="record:4",
                    text="Bob likes jazz.",
                ).to_dict()
            },
            provenance={"channels": ["raw_lexical"], "source_record_id": "record:4"},
        ),
        CardSymbolicRetriever().retrieve(query, [_card(source_record_id="record:kept")], k=1)[0],
    ]

    pool = apply_hard_filters(
        candidates,
        query=query,
        context=HardFilterContext(
            active_record_ids={"record:active"},
            allowed_evidence_types={"card"},
        ),
    )

    assert [candidate.provenance["source_record_id"] for candidate in pool.candidates] == [
        "record:kept"
    ]
    assert [item["reason"] for item in pool.suppressed] == [
        "duplicate_active_context",
        "invalidated_or_superseded_mismatch",
        "entity_mismatch",
        "evidence_type_mismatch",
    ]


def test_hard_entity_filter_accepts_speaker_role_match() -> None:
    query = parse_query_frame("When did Nora paint a mural?")
    candidate = RawLexicalRetriever(
        [
            _raw_record(
                record_id="record:nora",
                text="Image: query painting mural.",
                role="nora",
            )
        ]
    ).retrieve(query, k=1)[0]

    pool = apply_hard_filters(
        [candidate],
        query=query,
        context=HardFilterContext(),
    )

    assert [item.evidence_id for item in pool.candidates] == ["record:nora"]
    assert pool.suppressed == []


def test_candidate_generator_builds_serializable_multi_channel_pool() -> None:
    record = _raw_record(record_id="record:2", text="Alice also likes oolong tea.")
    hook = _FakeHook([RawRecallHit(record=record, similarity_score=0.7)])
    query = parse_query_frame("What tea does Alice prefer?", query_embedding=[0.1, 0.2])
    generator = CandidateGenerator(
        cards=[_card()],
        ltm_hook=hook,
        raw_records=[record],
        config=CandidateGenerationConfig(
            card_prefetch_k=5,
            raw_prefetch_k=5,
            symbolic_lookup_k=5,
        ),
    )

    pool = generator.generate(query)
    payload = json.loads(json.dumps(pool.to_dict()))

    assert {candidate["evidence_type"] for candidate in payload["candidates"]} == {
        "card",
        "raw_turn",
    }
    assert any(
        set(candidate["provenance"]["channels"]) == {"card_semantic", "card_symbolic"}
        for candidate in payload["candidates"]
    )
    assert any(
        set(candidate["provenance"]["channels"]) == {"raw_semantic", "raw_lexical"}
        for candidate in payload["candidates"]
    )


class _HookWithSearchCards:
    """Fake LTM hook that exposes search_cards (Chroma path)."""

    def __init__(self, hits: list[RawRecallHit]) -> None:
        self._hits = hits
        self.search_cards_called = False
        self.seen_k: int | None = None

    def archive(self, entry) -> None:  # noqa: ANN001
        pass

    def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:
        return []

    def search_cards(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:
        self.search_cards_called = True
        self.seen_k = k
        return self._hits[:k]


def test_card_semantic_retriever_uses_hook_search_cards_when_embedding_present() -> None:
    record = _raw_record()
    hit = RawRecallHit(record=record, similarity_score=0.85)
    hook = _HookWithSearchCards([hit])
    retriever = CardSemanticRetriever(ltm_hook=hook)
    query = parse_query_frame("What does Alice prefer?", query_embedding=[0.1, 0.2])

    results = retriever.retrieve(query, cards=[], k=5)

    assert hook.search_cards_called is True
    assert len(results) == 1
    assert results[0].evidence_id == record.record_id
    assert results[0].source == "card_semantic"


def test_card_semantic_retriever_falls_back_to_lexical_when_no_hook() -> None:
    retriever = CardSemanticRetriever(ltm_hook=None)
    query = parse_query_frame("What does Alice prefer?", query_embedding=[0.1, 0.2])
    cards = [_card()]

    results = retriever.retrieve(query, cards=cards, k=5)

    assert len(results) == 1
    assert results[0].evidence_id == "card:record:1:0"
    assert results[0].source == "card_semantic"


def test_card_semantic_retriever_falls_back_to_lexical_when_no_embedding() -> None:
    hook = _HookWithSearchCards([])
    retriever = CardSemanticRetriever(ltm_hook=hook)
    # No query_embedding → falls back to lexical
    query = parse_query_frame("What does Alice prefer?")
    cards = [_card()]

    results = retriever.retrieve(query, cards=cards, k=5)

    assert hook.search_cards_called is False
    assert len(results) == 1
    assert results[0].evidence_id == "card:record:1:0"


def test_deterministic_card_semantic_retriever_alias_is_card_semantic_retriever() -> None:
    """Backward-compat alias should point to the same class."""
    assert DeterministicCardSemanticRetriever is CardSemanticRetriever


def test_candidate_generator_passes_ltm_hook_to_card_semantic_retriever() -> None:
    record = _raw_record()
    hit = RawRecallHit(record=record, similarity_score=0.9)
    hook = _HookWithSearchCards([hit])
    query = parse_query_frame("Alice tea", query_embedding=[0.1, 0.2])

    generator = CandidateGenerator(
        cards=[],
        ltm_hook=hook,
        raw_records=[],
        config=CandidateGenerationConfig(
            card_prefetch_k=5,
            raw_prefetch_k=0,
            symbolic_lookup_k=0,
            enable_card_semantic=True,
            enable_card_symbolic=False,
            enable_raw_semantic=False,
            enable_raw_lexical=False,
        ),
    )
    generator.generate(query)

    assert hook.search_cards_called is True
