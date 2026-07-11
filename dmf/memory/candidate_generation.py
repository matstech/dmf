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

"""Deterministic multi-channel candidate generation for future reranking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from dmf.memory.lexical_normalization import lemma_terms, normalize_text
from dmf.memory.constants import (
    CARD_SEMANTIC_CHANNEL,
    CARD_SYMBOLIC_CHANNEL,
    COMMON_STOP_WORDS,
    QUERY_SPECIFIC_STOP_WORDS,
    RAW_LEXICAL_CHANNEL,
    RAW_SEMANTIC_CHANNEL,
)
from dmf.models.constants import (
    CARD_KIND_CONSTRAINT,
    CARD_KIND_CURRENT_STATE,
    CARD_KIND_EVENT,
    CARD_KIND_FACT,
    CARD_KIND_PREFERENCE,
    CARD_STATUS_INVALIDATED,
    CARD_STATUS_SUPERSEDED,
    EVIDENCE_TYPE_RAW_TURN,
    QUERY_ANSWER_TYPE_PERSON,
    QUERY_ANSWER_TYPE_PLACE,
    QUERY_ANSWER_TYPE_TIME,
    QUERY_ANSWER_TYPE_UNKNOWN,
    QUERY_CURRENTNESS_HISTORICAL,
    QUERY_CURRENTNESS_MIXED,
)
from dmf.models.ltm_hook import CardSearchLTMHook, LTMHook
from dmf.models.memory import MemoryCard, QueryFrame, RetrievedEvidence
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit


_STOP_WORDS = COMMON_STOP_WORDS
_QUERY_SPECIFIC_STOP_WORDS = QUERY_SPECIFIC_STOP_WORDS


class RawRecordSource(Protocol):
    """Source that can expose raw records for lexical candidate generation.

    Args:
        None.

    Returns:
        Protocol implementer for raw-record reads.

    Raises:
        Backend-specific exceptions may surface from concrete implementations.
    """

    def read_all(self) -> list[RawLTMRecord]:
        """Return raw records ordered by archival insertion.

        Returns:
            Raw records in insertion order.

        Raises:
            Backend-specific exceptions may surface from concrete stores.
        """


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Prefetch and channel gates for the phase-3 candidate pool.

    These settings do not change the public final recall limit. They only
    control this standalone pool generator, which remains opt-in in phase 3.

    Args:
        card_prefetch_k: Maximum semantic card candidates requested.
        raw_prefetch_k: Maximum raw candidates requested per raw channel.
        symbolic_lookup_k: Maximum symbolic card candidates requested.
        enable_card_semantic: Enables semantic card retrieval.
        enable_card_symbolic: Enables symbolic card lookup.
        enable_raw_semantic: Enables raw vector retrieval.
        enable_raw_lexical: Enables raw lexical retrieval.

    Returns:
        Immutable candidate-generation configuration.

    Raises:
        None.
    """

    card_prefetch_k: int = 32
    raw_prefetch_k: int = 16
    symbolic_lookup_k: int = 16
    enable_card_semantic: bool = True
    enable_card_symbolic: bool = True
    enable_raw_semantic: bool = True
    enable_raw_lexical: bool = True


@dataclass(frozen=True)
class HardFilterContext:
    """Inputs used by preliminary hard filters before answerability rerank.

    Args:
        active_record_ids: Source records currently active in working memory.
        suppressed_record_ids: Source records already suppressed upstream.
        allowed_evidence_types: Optional evidence-type allowlist.

    Returns:
        Immutable hard-filter context.

    Raises:
        None.
    """

    active_record_ids: set[str] = field(default_factory=set)
    suppressed_record_ids: set[str] = field(default_factory=set)
    allowed_evidence_types: set[str] | None = None


@dataclass(frozen=True)
class CandidatePool:
    """Diagnostic output from multi-channel candidate generation.

    Args:
        candidates: Evidence candidates that survived preliminary filtering.
        suppressed: Diagnostic payloads for candidates removed by hard filters.

    Returns:
        Immutable candidate-pool payload.

    Raises:
        None.
    """

    candidates: list[RetrievedEvidence]
    suppressed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable candidate-pool diagnostic payload.

        Returns:
            Dictionary containing candidates and suppression diagnostics.

        Raises:
            None.
        """
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "suppressed": list(self.suppressed),
        }


class CardSemanticRetriever:
    """Card semantic retriever with Chroma-backed path and lexical fallback.

    When an LTM hook with a ``search_cards`` method is available and the query
    carries an embedding, real vector search over the card index is used.
    Otherwise the original deterministic token-overlap path is preserved as a
    fallback, ensuring this retriever works without any vector store.

    Args:
        ltm_hook: Optional hook exposing ``search_cards``.

    Returns:
        Retriever instance.

    Raises:
        None.
    """

    def __init__(self, ltm_hook=None) -> None:  # noqa: ANN001
        self._ltm_hook = ltm_hook

    def retrieve(
        self,
        query: QueryFrame,
        cards: Sequence[MemoryCard],
        *,
        k: int,
    ) -> list[RetrievedEvidence]:
        """Return card evidence from vector search or lexical fallback.

        Args:
            query: Parsed query frame.
            cards: In-memory cards used by the lexical fallback.
            k: Maximum number of results.

        Returns:
            Retrieved card evidence ordered by channel relevance.

        Raises:
            Backend-specific exceptions may surface from ``search_cards``.
        """
        if (
            self._ltm_hook is not None
            and isinstance(self._ltm_hook, CardSearchLTMHook)
            and query.query_embedding is not None
        ):
            hits = self._ltm_hook.search_cards(query.query_embedding, k=k)
            return [_raw_hit_evidence(hit, source=CARD_SEMANTIC_CHANNEL) for hit in hits]
        # The fallback keeps retrieval available for file-only stores while
        # preserving deterministic behavior in tests.
        query_tokens = _query_tokens(query)
        scored: list[tuple[float, int, RetrievedEvidence]] = []
        for index, card in enumerate(cards):
            score = _lexical_overlap_score(query_tokens, _tokens(_card_text(card)))
            if score <= 0.0:
                continue
            scored.append((
                score,
                index,
                _card_evidence(
                    card,
                    source=CARD_SEMANTIC_CHANNEL,
                    semantic_score=score,
                ),
            ))
        return [item for _, _, item in sorted(scored, key=lambda row: (-row[0], row[1]))[:k]]


# Backward compatibility keeps downstream imports stable during the retriever
# rename.
DeterministicCardSemanticRetriever = CardSemanticRetriever


class CardSymbolicRetriever:
    """Lookup cards by query entities, subjects, predicates, and answer type.

    Args:
        None.

    Returns:
        Retriever instance.

    Raises:
        None.
    """

    def retrieve(
        self,
        query: QueryFrame,
        cards: Sequence[MemoryCard],
        *,
        k: int,
    ) -> list[RetrievedEvidence]:
        """Return symbolic card matches for a parsed query.

        Args:
            query: Parsed query frame.
            cards: Candidate cards to score.
            k: Maximum number of results.

        Returns:
            Retrieved card evidence ordered by symbolic score.

        Raises:
            None.
        """
        scored: list[tuple[float, int, RetrievedEvidence]] = []
        for index, card in enumerate(cards):
            score = _symbolic_card_score(query, card)
            if score <= 0.0:
                continue
            scored.append((
                score,
                index,
                _card_evidence(
                    card,
                    source=CARD_SYMBOLIC_CHANNEL,
                    symbolic_score=score,
                ),
            ))
        return [item for _, _, item in sorted(scored, key=lambda row: (-row[0], row[1]))[:k]]


class RawSemanticRetriever:
    """Adapter over the existing raw semantic LTM surface.

    Args:
        ltm_hook: Hook used for raw vector retrieval.

    Returns:
        Retriever instance.

    Raises:
        None.
    """

    def __init__(self, ltm_hook: LTMHook) -> None:
        self._ltm_hook = ltm_hook

    def retrieve(
        self,
        query: QueryFrame,
        *,
        k: int,
    ) -> list[RetrievedEvidence]:
        """Return raw evidence from semantic LTM search.

        Args:
            query: Parsed query frame. Queries without embeddings return no
                semantic candidates.
            k: Maximum number of results.

        Returns:
            Retrieved raw-turn evidence.

        Raises:
            Backend-specific exceptions may surface from ``search_raw``.
        """
        if query.query_embedding is None:
            return []
        hits = self._ltm_hook.search_raw(query.query_embedding, k=k)
        return [_raw_hit_evidence(hit, source=RAW_SEMANTIC_CHANNEL) for hit in hits]


class RawLexicalRetriever:
    """Deterministic lexical lookup over raw archived turns.

    Args:
        records: Iterable raw records or a source exposing ``read_all``.

    Returns:
        Retriever instance.

    Raises:
        None.
    """

    def __init__(self, records: Iterable[RawLTMRecord] | RawRecordSource) -> None:
        self._records = records

    def retrieve(
        self,
        query: QueryFrame,
        *,
        k: int,
    ) -> list[RetrievedEvidence]:
        """Return raw evidence by deterministic lexical overlap.

        Args:
            query: Parsed query frame.
            k: Maximum number of results.

        Returns:
            Retrieved raw-turn evidence ordered by lexical overlap.

        Raises:
            Backend-specific exceptions may surface from ``records.read_all``.
        """
        query_tokens = _query_tokens(query)
        scored: list[tuple[float, int, RetrievedEvidence]] = []
        for index, record in enumerate(self._read_records()):
            score = _lexical_overlap_score(
                query_tokens,
                _tokens(_raw_record_search_text(record)),
            )
            if score <= 0.0:
                continue
            scored.append((
                score,
                index,
                _raw_record_evidence(
                    record,
                    source=RAW_LEXICAL_CHANNEL,
                    symbolic_score=score,
                ),
            ))
        return [item for _, _, item in sorted(scored, key=lambda row: (-row[0], row[1]))[:k]]

    def _read_records(self) -> list[RawLTMRecord]:
        if hasattr(self._records, "read_all"):
            return list(self._records.read_all())
        return list(self._records)


class CandidateGenerator:
    """Opt-in orchestration for phase-3 multi-channel candidate generation.

    Args:
        cards: Structured cards available to card channels.
        ltm_hook: Optional LTM hook for raw and card semantic retrieval.
        raw_records: Raw records available to lexical retrieval and assembly.
        config: Optional channel and prefetch configuration.

    Returns:
        Generator instance.

    Raises:
        None.
    """

    def __init__(
        self,
        *,
        cards: Sequence[MemoryCard] = (),
        ltm_hook: LTMHook | None = None,
        raw_records: Iterable[RawLTMRecord] | RawRecordSource = (),
        config: CandidateGenerationConfig | None = None,
    ) -> None:
        self._cards = cards
        self._ltm_hook = ltm_hook
        self._raw_records = raw_records
        self._config = config or CandidateGenerationConfig()
        self._card_semantic = CardSemanticRetriever(ltm_hook=ltm_hook)
        self._card_symbolic = CardSymbolicRetriever()
        self._raw_lexical = RawLexicalRetriever(raw_records)

    def generate(
        self,
        query: QueryFrame,
        *,
        hard_filter_context: HardFilterContext | None = None,
    ) -> CandidatePool:
        """Return merged, deduped, preliminarily filtered evidence candidates.

        Args:
            query: Parsed query frame.
            hard_filter_context: Optional context for preliminary filtering.

        Returns:
            Candidate pool with kept and suppressed diagnostics.

        Raises:
            Backend-specific exceptions may surface from retrieval backends.
        """
        channel_results: list[RetrievedEvidence] = []
        if self._config.enable_card_semantic:
            channel_results.extend(
                self._card_semantic.retrieve(
                    query,
                    self._cards,
                    k=self._config.card_prefetch_k,
                )
            )
        if self._config.enable_card_symbolic:
            channel_results.extend(
                self._card_symbolic.retrieve(
                    query,
                    self._cards,
                    k=self._config.symbolic_lookup_k,
                )
            )
        if self._config.enable_raw_semantic and self._ltm_hook is not None:
            channel_results.extend(
                RawSemanticRetriever(self._ltm_hook).retrieve(
                    query,
                    k=self._config.raw_prefetch_k,
                )
            )
        if self._config.enable_raw_lexical:
            channel_results.extend(
                self._raw_lexical.retrieve(query, k=self._config.raw_prefetch_k)
            )

        merged = merge_dedupe_candidates(channel_results)
        return apply_hard_filters(
            merged,
            query=query,
            context=hard_filter_context or HardFilterContext(),
        )


def merge_dedupe_candidates(
    candidates: Iterable[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    """Merge duplicate evidence while preserving contributing channel provenance.

    Args:
        candidates: Evidence candidates from all enabled retrieval channels.

    Returns:
        Deduplicated evidence sorted by deterministic candidate score.

    Raises:
        None.
    """
    merged: dict[tuple[str, str], RetrievedEvidence] = {}
    order: list[tuple[str, str]] = []
    for candidate in candidates:
        key = (candidate.evidence_type, candidate.evidence_id)
        current = merged.get(key)
        if current is None:
            merged[key] = _with_channel(candidate)
            order.append(key)
            continue
        merged[key] = _merge_evidence(current, candidate)
    return sorted(
        (merged[key] for key in order),
        key=_candidate_sort_key,
        reverse=True,
    )


def apply_hard_filters(
    candidates: Iterable[RetrievedEvidence],
    *,
    query: QueryFrame,
    context: HardFilterContext,
) -> CandidatePool:
    """Apply phase-3 hard filters without answerability reranking.

    Args:
        candidates: Evidence candidates after channel merge.
        query: Parsed query frame.
        context: Preliminary filter context.

    Returns:
        Candidate pool with kept evidence and suppression diagnostics.

    Raises:
        None.
    """
    kept: list[RetrievedEvidence] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = _hard_filter_reason(candidate, query, context)
        if reason is not None:
            suppressed.append({"reason": reason, "candidate": candidate.to_dict()})
            continue
        kept.append(candidate)
    return CandidatePool(candidates=kept, suppressed=suppressed)


def _hard_filter_reason(
    candidate: RetrievedEvidence,
    query: QueryFrame,
    context: HardFilterContext,
) -> str | None:
    record_id = _source_record_id(candidate)
    if record_id in context.active_record_ids:
        return "duplicate_active_context"
    if record_id in context.suppressed_record_ids:
        return "invalidated_or_superseded_mismatch"
    if (
        context.allowed_evidence_types is not None
        and candidate.evidence_type not in context.allowed_evidence_types
    ):
        return "evidence_type_mismatch"
    if _is_invalid_for_currentness(candidate, query):
        return "invalidated_or_superseded_mismatch"
    if _has_hard_entity_mismatch(candidate, query):
        return "entity_mismatch"
    return None


def _card_evidence(
    card: MemoryCard,
    *,
    source: str,
    semantic_score: float | None = None,
    symbolic_score: float | None = None,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=card.card_id,
        evidence_type="card",
        source=source,
        semantic_score=semantic_score,
        symbolic_score=symbolic_score,
        temporal_score=_temporal_score(card.time_anchor.relative_order),
        answerability_features={
            "kind": card.kind,
            "validity_status": card.validity.status,
            "confidence": card.confidence,
        },
        render_payload={"card": card.to_dict()},
        provenance={
            "channels": [source],
            "source_record_id": card.provenance.source_record_id,
            "speaker_role": card.provenance.speaker_role,
            "source_turn": card.provenance.source_turn,
            "card_id": card.card_id,
        },
    )


def _raw_hit_evidence(hit: RawRecallHit, *, source: str) -> RetrievedEvidence:
    return _raw_record_evidence(
        hit.record,
        source=source,
        semantic_score=_raw_similarity_score(hit),
        rank_hint=hit.rank_hint,
    )


def _raw_record_evidence(
    record: RawLTMRecord,
    *,
    source: str,
    semantic_score: float | None = None,
    symbolic_score: float | None = None,
    rank_hint: int | None = None,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=record.record_id,
        evidence_type=EVIDENCE_TYPE_RAW_TURN,
        source=source,
        semantic_score=semantic_score,
        symbolic_score=symbolic_score,
        temporal_score=_temporal_score(record.interaction_id),
        answerability_features={},
        render_payload={"record": record.to_dict()},
        provenance={
            "channels": [source],
            "source_record_id": record.record_id,
            "speaker_role": record.role,
            "source_turn": record.provenance.source_turn,
            "rank_hint": rank_hint,
        },
    )


def _merge_evidence(
    left: RetrievedEvidence,
    right: RetrievedEvidence,
) -> RetrievedEvidence:
    channels = _dedupe_strings([
        *_provenance_channels(left),
        *_provenance_channels(right),
        left.source,
        right.source,
    ])
    return RetrievedEvidence(
        evidence_id=left.evidence_id,
        evidence_type=left.evidence_type,
        source="+".join(channels),
        semantic_score=_max_optional(left.semantic_score, right.semantic_score),
        symbolic_score=_max_optional(left.symbolic_score, right.symbolic_score),
        temporal_score=_max_optional(left.temporal_score, right.temporal_score),
        answerability_features={
            **left.answerability_features,
            **right.answerability_features,
        },
        render_payload=left.render_payload or right.render_payload,
        provenance={
            **left.provenance,
            **right.provenance,
            "channels": channels,
        },
    )


def _with_channel(candidate: RetrievedEvidence) -> RetrievedEvidence:
    if "channels" in candidate.provenance:
        return candidate
    return RetrievedEvidence(
        evidence_id=candidate.evidence_id,
        evidence_type=candidate.evidence_type,
        source=candidate.source,
        semantic_score=candidate.semantic_score,
        symbolic_score=candidate.symbolic_score,
        temporal_score=candidate.temporal_score,
        answerability_features=candidate.answerability_features,
        render_payload=candidate.render_payload,
        provenance={**candidate.provenance, "channels": [candidate.source]},
    )


def _candidate_sort_key(candidate: RetrievedEvidence) -> tuple[float, float, float]:
    return (
        candidate.semantic_score or 0.0,
        candidate.symbolic_score or 0.0,
        candidate.temporal_score or 0.0,
    )


def _raw_similarity_score(hit: RawRecallHit) -> float | None:
    if hit.similarity_score is not None:
        return float(hit.similarity_score)
    if hit.distance is not None:
        return 1.0 - float(hit.distance)
    return None


def _symbolic_card_score(query: QueryFrame, card: MemoryCard) -> float:
    card_text = _card_text(card)
    subject_match = _card_subject_match(query, card)
    entity_overlap = _card_entity_overlap(query, card_text)
    predicate_match = _card_predicate_match(query, card, card_text)
    specificity_overlap = _card_specificity_overlap(query, card)
    answer_type_bonus = (
        1.0
        if query.answer_type != QUERY_ANSWER_TYPE_UNKNOWN and _answer_type_matches_card(query, card)
        else 0.0
    )
    has_specific_query = bool(query.predicate_focus or _query_specific_tokens(query))

    if has_specific_query and predicate_match == 0.0 and specificity_overlap == 0.0:
        return 0.0

    weighted_score = (
        subject_match * 0.9
        + entity_overlap * 0.55
        + predicate_match * 1.2
        + specificity_overlap * 1.35
        + answer_type_bonus * 0.2
    )
    if _is_generic_topic_card(card) and max(predicate_match, specificity_overlap) < 0.5:
        weighted_score -= 0.25
    if subject_match > 0.0 and entity_overlap > 0.0 and max(
        predicate_match,
        specificity_overlap,
    ) <= 0.0:
        weighted_score -= 0.4
    score = max(0.0, min(1.0, weighted_score / 4.2))
    if has_specific_query and score < 0.2:
        return 0.0
    return round(score, 6)


def _answer_type_matches_card(query: QueryFrame, card: MemoryCard) -> bool:
    if query.answer_type == QUERY_ANSWER_TYPE_PERSON:
        return card.kind == "relation"
    if query.answer_type == QUERY_ANSWER_TYPE_PLACE:
        return card.predicate in {"live", "location", "address"} or card.kind == CARD_KIND_EVENT
    if query.answer_type == QUERY_ANSWER_TYPE_TIME:
        return card.time_anchor.absolute_time is not None or card.kind == CARD_KIND_EVENT
    return card.kind in {
        CARD_KIND_FACT,
        CARD_KIND_PREFERENCE,
        CARD_KIND_CURRENT_STATE,
        CARD_KIND_CONSTRAINT,
    }


def _card_subject_match(query: QueryFrame, card: MemoryCard) -> float:
    subjects = [_normalize(value) for value in query.subject_focus if value]
    if not subjects:
        return 0.0
    card_subject = _normalize(card.subject)
    card_object = _normalize(card.object)
    for subject in subjects:
        if not subject or subject == "user":
            continue
        if subject == card_subject:
            return 1.0
        if subject in card_object:
            return 0.7
    return 0.0


def _card_entity_overlap(query: QueryFrame, card_text: str) -> float:
    entities = [_normalize(value) for value in query.entities if value]
    if not entities:
        return 0.0
    normalized_text = _normalize(card_text)
    hits = sum(1 for value in entities if value and value in normalized_text)
    return min(1.0, hits / len(entities)) if hits else 0.0


def _card_predicate_match(
    query: QueryFrame,
    card: MemoryCard,
    card_text: str,
) -> float:
    predicates = _predicate_terms(query.predicate_focus)
    if not predicates:
        return 0.0
    card_tokens = _focus_tokens(" ".join([card.predicate, card.kind, card_text]))
    if predicates & card_tokens:
        return 1.0
    return 0.0


def _card_specificity_overlap(query: QueryFrame, card: MemoryCard) -> float:
    query_tokens = _query_specific_tokens(query)
    if not query_tokens:
        return 0.0
    card_tokens = _focus_tokens(" ".join([card.predicate, card.object, *card.surface_forms]))
    overlap = query_tokens & card_tokens
    if not overlap:
        return 0.0
    return min(1.0, len(overlap) / len(query_tokens))


def _is_generic_topic_card(card: MemoryCard) -> bool:
    return card.kind in {CARD_KIND_EVENT, CARD_KIND_FACT} and _normalize(card.predicate) in {
        "mentions_event",
        CARD_KIND_FACT,
    }


def _is_invalid_for_currentness(candidate: RetrievedEvidence, query: QueryFrame) -> bool:
    if query.historical_vs_current in {QUERY_CURRENTNESS_HISTORICAL, QUERY_CURRENTNESS_MIXED}:
        return False
    status = candidate.answerability_features.get("validity_status")
    return status in {CARD_STATUS_INVALIDATED, CARD_STATUS_SUPERSEDED}


def _has_hard_entity_mismatch(candidate: RetrievedEvidence, query: QueryFrame) -> bool:
    if not query.entities and not query.subject_focus:
        return False
    candidate_text = _candidate_text(candidate)
    if not candidate_text:
        return False
    required = [
        _normalize(value)
        for value in [*query.entities, *query.subject_focus]
        if len(_normalize(value)) >= 3 and _normalize(value) != "user"
    ]
    if not required:
        return False
    return not any(value in _normalize(candidate_text) for value in required)


def _source_record_id(candidate: RetrievedEvidence) -> str:
    value = candidate.provenance.get("source_record_id")
    return str(value) if value is not None else candidate.evidence_id


def _provenance_channels(candidate: RetrievedEvidence) -> list[str]:
    channels = candidate.provenance.get("channels", [])
    if not isinstance(channels, list):
        return []
    return [str(channel) for channel in channels]


def _candidate_text(candidate: RetrievedEvidence) -> str:
    payload = candidate.render_payload
    card_payload = payload.get("card")
    if isinstance(card_payload, dict):
        pieces = [
            card_payload.get("kind"),
            card_payload.get("subject"),
            card_payload.get("predicate"),
            card_payload.get("object"),
            *card_payload.get("surface_forms", []),
        ]
        return " ".join(str(piece) for piece in pieces if piece)
    record_payload = payload.get("record")
    if isinstance(record_payload, dict):
        return _raw_record_payload_search_text(record_payload)
    return ""


def _card_text(card: MemoryCard) -> str:
    pieces = [
        card.kind,
        card.subject,
        card.predicate,
        card.object,
        *card.surface_forms,
    ]
    return " ".join(piece for piece in pieces if piece)


def _raw_record_search_text(record: RawLTMRecord) -> str:
    return " ".join(
        piece
        for piece in [
            record.role,
            record.provenance.role,
            record.text,
        ]
        if piece
    )


def _raw_record_payload_search_text(record: dict[str, Any]) -> str:
    provenance = record.get("provenance", {})
    provenance_role = (
        provenance.get("role")
        if isinstance(provenance, dict)
        else None
    )
    return " ".join(
        str(piece)
        for piece in [
            record.get("role"),
            provenance_role,
            record.get("text"),
        ]
        if piece
    )


def _query_tokens(query: QueryFrame) -> set[str]:
    return _tokens(
        " ".join([
            query.query_text,
            *query.entities,
            *query.subject_focus,
            *query.predicate_focus,
        ])
    )


def _query_specific_tokens(query: QueryFrame) -> set[str]:
    blocked = {
        token
        for value in [*query.entities, *query.subject_focus]
        for token in lemma_terms(value, extra_stop_words=_QUERY_SPECIFIC_STOP_WORDS)
        if token
    }
    query_tokens = _focus_tokens(" ".join([query.query_text, *query.predicate_focus]))
    return {
        token
        for token in query_tokens
        if token
        and token not in blocked
        and token not in _QUERY_SPECIFIC_STOP_WORDS
    }


def _tokens(text: str) -> set[str]:
    return lemma_terms(text, extra_stop_words=_STOP_WORDS)


def _focus_tokens(text: str) -> set[str]:
    return lemma_terms(
        text,
        extra_stop_words=_QUERY_SPECIFIC_STOP_WORDS,
        content_only=True,
    )


def _lexical_overlap_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(query_tokens)


def _normalize(value: str) -> str:
    return normalize_text(value)


def _predicate_terms(values: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        terms.update(_focus_tokens(value))
    return {term for term in terms if term}


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _temporal_score(order: int | None) -> float | None:
    if order is None:
        return None
    return float(order)
