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

import numpy as np

from dmf.memory.evidence_assembly import (
    EvidenceAssemblyConfig,
    apply_final_cutoff,
    assemble_final_evidence,
    expand_card_evidence,
    render_evidence_context,
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


def _record(
    record_id: str,
    text: str,
    *,
    interaction_id: int,
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


def _card(
    *,
    card_id: str = "card:record:2:0",
    source_record_id: str = "record:2",
    status: str = "active",
    supersedes: list[str] | None = None,
    superseded_by: list[str] | None = None,
    corrects: list[str] | None = None,
    invalidates: list[str] | None = None,
) -> MemoryCard:
    return MemoryCard(
        card_id=card_id,
        kind="current_state",
        subject="Alice",
        predicate="lives in",
        object="Rome",
        time_anchor=MemoryCardTimeAnchor(relative_order=2, turn_id=2),
        validity=MemoryCardValidity(
            status=status,
            supersedes=supersedes or [],
            superseded_by=superseded_by or [],
            corrects=corrects or [],
            invalidates=invalidates or [],
        ),
        provenance=MemoryCardProvenance(
            source_record_id=source_record_id,
            speaker_role="user",
            source_turn=2,
        ),
        confidence=0.9,
        surface_forms=["Alice lives in Rome"],
    )


def _card_evidence(card: MemoryCard, *, score: float = 1.0) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=card.card_id,
        evidence_type="card",
        source="test",
        semantic_score=score,
        answerability_features={
            "kind": card.kind,
            "validity_status": card.validity.status,
        },
        render_payload={"card": card.to_dict()},
        provenance={
            "source_record_id": card.provenance.source_record_id,
            "card_id": card.card_id,
        },
    )


def _raw_evidence(record: RawLTMRecord) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=record.record_id,
        evidence_type="raw_turn",
        source="test",
        render_payload={"record": record.to_dict()},
        provenance={"source_record_id": record.record_id},
    )


class _FakeHook(LTMHook):
    def __init__(self, hits: list[RawRecallHit]) -> None:
        self.hits = hits

    def archive(self, entry) -> None:  # noqa: ANN001
        pass

    def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:
        return self.hits[:k]


def test_final_cutoff_is_applied_after_rerank_order() -> None:
    candidates = [
        _card_evidence(_card(card_id=f"card:record:{idx}:0"), score=1.0 - idx / 10)
        for idx in range(4)
    ]

    final = apply_final_cutoff(candidates, final_recall_limit=2)

    assert [item.evidence_id for item in final] == [
        "card:record:0:0",
        "card:record:1:0",
    ]


def test_card_expansion_adds_source_supersession_and_neighbors() -> None:
    records = [
        _record("record:1", "Alice used to live in Paris.", interaction_id=1),
        _record("record:2", "Alice lives in Rome now.", interaction_id=2),
        _record("record:3", "She corrected the old address.", interaction_id=3),
    ]
    card = _card(supersedes=["record:1"], superseded_by=["record:3"])

    expanded = expand_card_evidence(
        [_card_evidence(card)],
        raw_records=records,
        query=parse_query_frame("Where did Alice previously live?"),
        config=EvidenceAssemblyConfig(
            max_support_turns_per_card=4,
            include_neighbor_turns=True,
        ),
    )

    support = expanded[0].render_payload["support_records"]
    assert expanded[0].evidence_type == "card_backed_turn"
    assert [(item["relation"], item["record"]["record_id"]) for item in support] == [
        ("source", "record:2"),
        ("historical_link", "record:1"),
        ("supersession_link", "record:3"),
    ]
    assert expanded[0].provenance["support_record_ids"] == [
        "record:2",
        "record:1",
        "record:3",
    ]


def test_current_query_omits_superseded_history_but_keeps_supersession_turn() -> None:
    records = [
        _record("record:1", "Alice used to live in Paris.", interaction_id=1),
        _record("record:2", "Alice lives in Rome now.", interaction_id=2),
        _record("record:3", "Alice corrected the address again.", interaction_id=3),
    ]
    card = _card(supersedes=["record:1"], superseded_by=["record:3"])

    expanded = assemble_final_evidence(
        [_card_evidence(card)],
        raw_records=records,
        query=parse_query_frame("Where does Alice currently live?"),
        config=EvidenceAssemblyConfig(final_recall_limit=1, max_support_turns_per_card=3),
    )

    support_ids = [
        item["record"]["record_id"]
        for item in expanded[0].render_payload["support_records"]
    ]
    assert support_ids == [
        "record:2",
        "record:3",
    ]


def test_optional_local_neighbors_are_included_when_enabled() -> None:
    records = [
        _record("record:1", "Alice mentioned moving plans.", interaction_id=1),
        _record("record:2", "Alice lives in Rome now.", interaction_id=2),
        _record("record:3", "Bob asked a follow-up question.", interaction_id=3),
    ]

    expanded = expand_card_evidence(
        [_card_evidence(_card())],
        raw_records=records,
        config=EvidenceAssemblyConfig(
            max_support_turns_per_card=3,
            include_neighbor_turns=True,
        ),
    )

    support = expanded[0].render_payload["support_records"]
    assert [(item["relation"], item["record"]["record_id"]) for item in support] == [
        ("source", "record:2"),
        ("neighbor_before", "record:1"),
        ("neighbor_after", "record:3"),
    ]


def test_renderer_outputs_structured_and_raw_support_without_scores() -> None:
    record = _record("record:2", "Alice lives in Rome now.", interaction_id=2)
    expanded = expand_card_evidence(
        [_card_evidence(_card())],
        raw_records=[record],
    )

    rendered = render_evidence_context(expanded)

    assert "=== STRUCTURED EVIDENCE ===" in rendered
    assert "[R1] user | time=2 | current" in rendered
    assert "Alice lives in Rome" in rendered
    assert "=== RAW SUPPORTING EVIDENCE ===" in rendered
    assert "[S1] source | turn=2" in rendered
    assert "Alice lives in Rome now." in rendered
    assert "semantic_score" not in rendered
    assert "answerability_score" not in rendered


def test_expanded_evidence_serialization_round_trips() -> None:
    record = _record("record:2", "Alice lives in Rome now.", interaction_id=2)
    expanded = expand_card_evidence([_card_evidence(_card())], raw_records=[record])[0]

    payload = json.loads(json.dumps(expanded.to_dict()))
    parsed = RetrievedEvidence.from_dict(payload)

    assert parsed == expanded


def test_raw_winner_passes_through_and_renders_as_support() -> None:
    record = _record("record:9", "Alice likes green tea.", interaction_id=9)
    raw = _raw_evidence(record)

    final = assemble_final_evidence(
        [raw, _card_evidence(_card())],
        raw_records=[record],
        config=EvidenceAssemblyConfig(final_recall_limit=1),
    )
    rendered = render_evidence_context(final, include_structured_evidence=False)

    assert final == [raw]
    assert "=== STRUCTURED EVIDENCE ===" not in rendered
    assert "Alice likes green tea." in rendered


def test_raw_winner_expands_local_neighbors_when_enabled() -> None:
    records = [
        _record("record:8", "Bob asked which tea Alice likes.", interaction_id=8),
        _record("record:9", "Alice likes green tea.", interaction_id=9),
        _record("record:10", "Bob said he would buy some.", interaction_id=10),
    ]
    raw = _raw_evidence(records[1])

    final = assemble_final_evidence(
        [raw],
        raw_records=records,
        config=EvidenceAssemblyConfig(
            final_recall_limit=1,
            include_neighbor_turns=True,
        ),
    )
    support = final[0].render_payload["support_records"]
    rendered = render_evidence_context(final)

    assert final[0].evidence_type == "raw_turn"
    assert final[0].provenance["support_record_ids"] == [
        "record:9",
        "record:8",
        "record:10",
    ]
    assert [(item["relation"], item["record"]["record_id"]) for item in support] == [
        ("neighbor_before", "record:8"),
        ("neighbor_after", "record:10"),
    ]
    assert "[R1] user | time=9" in rendered
    assert "[S1] prev | turn=8" in rendered
    assert "[S2] next | turn=10" in rendered


def test_raw_winner_renders_as_numbered_recalled_item_when_structured_section_enabled() -> None:
    record = _record("record:9", "Alice likes green tea.", interaction_id=9)
    raw = _raw_evidence(record)

    rendered = render_evidence_context([raw])

    assert "=== STRUCTURED EVIDENCE ===" in rendered
    assert "[R1] user | time=9" in rendered
    assert "Alice likes green tea." in rendered
    assert "=== RAW SUPPORTING EVIDENCE ===" not in rendered


def test_legacy_raw_retrieval_path_is_unchanged_by_assembly_components() -> None:
    record = _record("record:7", "Alice booked a train to Rome.", interaction_id=7)
    tm = TemporalMemory(ltm_hook=_FakeHook([RawRecallHit(record=record, similarity_score=0.9)]))
    tm._nlp_engine = _Analyzer()  # noqa: SLF001

    recalled = tm.rerank_contextualized_recall_candidates(
        tm.contextualize_raw_recall_hits(
            tm.get_raw_recall_hits(np.array([1.0, 0.0]), k=1)
        )
    )

    assert [candidate.record.record_id for candidate in recalled] == ["record:7"]
    assert all(not hasattr(candidate, "render_payload") for candidate in recalled)


class _Analyzer:
    def analyze_interaction(
        self,
        text: str,
        is_system: bool = False,  # noqa: ARG002
    ) -> AnalysisReport:
        return AnalysisReport(
            info_density=0.8,
            entity_count=1,
            sentiment_abs=0.1,
            semantic_divergence=0.1,
            is_system_prompt=False,
            latency_ms=0.0,
            signals=InteractionSignals(personal_relevance=1.0),
            raw_metadata={"text": text},
        )
