"""Deterministic answerability-aware reranking for retrieved evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dmf.memory.constants import (
    COMMON_STOP_WORDS,
    QUERY_SPECIFIC_STOP_WORDS,
    STALE_TEMPORAL_CUES,
)
from dmf.memory.lexical_normalization import lemma_terms, normalize_text
from dmf.memory.temporal_interpretation import (
    TemporalInterpretation,
    interpret_temporal_text,
    interpretation_from_query_frame_filters,
)
from dmf.models.constants import (
    CARD_KIND_CURRENT_STATE,
    CARD_STATUS_ACTIVE,
    CARD_STATUS_CORRECTED,
    CARD_STATUS_SUPERSEDED,
    QUERY_ANSWER_TYPE_BOOLEAN,
    QUERY_ANSWER_TYPE_PERSON,
    QUERY_ANSWER_TYPE_PLACE,
    QUERY_ANSWER_TYPE_QUANTITY,
    QUERY_ANSWER_TYPE_TIME,
    QUERY_CURRENTNESS_CURRENT,
    QUERY_CURRENTNESS_MIXED,
    QUERY_TEMPORAL_CURRENT,
    QUERY_TEMPORAL_LATEST,
    QUERY_TEMPORAL_RANGE,
    STALE_CARD_STATUSES,
)
from dmf.models.memory import QueryFrame, RetrievedEvidence


_STOP_WORDS = COMMON_STOP_WORDS
_QUERY_SPECIFIC_STOP_WORDS = QUERY_SPECIFIC_STOP_WORDS
_STALE_CUES = STALE_TEMPORAL_CUES
_ACK_SOCIAL_CUES = {
    "thanks",
    "thank you",
    "ok",
    "okay",
    "sure",
    "got it",
    "sounds good",
    "you're welcome",
    "you are welcome",
}


@dataclass(frozen=True)
class AnswerabilityDiagnostics:
    """Serializable feature payload attached to each reranked candidate.
    
    Args:
        score: See the function signature and surrounding type hints.
        features: See the function signature and surrounding type hints.
        penalties: See the function signature and surrounding type hints.
        baseline_rank: See the function signature and surrounding type hints.
        baseline_key: See the function signature and surrounding type hints.
        notes: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    score: float
    features: dict[str, float]
    penalties: dict[str, float]
    baseline_rank: int
    baseline_key: dict[str, float | int]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic payload.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "score": self.score,
            "features": dict(sorted(self.features.items())),
            "penalties": dict(sorted(self.penalties.items())),
            "baseline_rank": self.baseline_rank,
            "baseline_key": dict(self.baseline_key),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AnswerabilityRankerConfig:
    """Weights for deterministic answerability reranking.
    
    Args:
        feature_weights: See the function signature and surrounding type hints.
        penalty_weights: See the function signature and surrounding type hints.
        tie_epsilon: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    feature_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "semantic_similarity": 1.0,
            "entity_overlap": 1.0,
            "subject_match": 1.2,
            "predicate_match": 1.15,
            "query_specificity_match": 1.2,
            "temporal_compatibility": 0.8,
            "current_state_compatibility": 0.9,
            "active_not_superseded": 1.0,
            "answer_span_likelihood": 1.0,
            "evidence_chain_availability": 0.4,
        }
    )
    penalty_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "stale_fact_penalty": 1.1,
            "ack_social_penalty": 0.9,
            "topic_only_penalty": 1.2,
        }
    )
    tie_epsilon: float = 1e-9


class AnswerabilityFeatureExtractor:
    """Extract explainable answerability features from one evidence candidate.
    
    Args:
        None.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def extract(
        self,
        query: QueryFrame,
        candidate: RetrievedEvidence,
        *,
        baseline_rank: int = 0,
    ) -> AnswerabilityDiagnostics:
        """Return weighted diagnostics without mutating ``candidate``.
        
        Args:
            query: See the function signature and surrounding type hints.
            candidate: See the function signature and surrounding type hints.
            baseline_rank: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        text = _candidate_text(candidate)
        card = _card_payload(candidate)
        query_temporal = _query_temporal_interpretation(query)
        candidate_temporal = _candidate_temporal_interpretation(candidate, text, card)
        features = {
            "semantic_similarity": _bounded(candidate.semantic_score),
            "entity_overlap": _entity_overlap(query, candidate, text),
            "subject_match": _subject_match(query, candidate, text, card),
            "predicate_match": _predicate_match(query, candidate, text, card),
            "query_specificity_match": _query_specificity_match(
                query,
                candidate,
                text,
                card,
            ),
            "temporal_compatibility": _temporal_compatibility(
                query,
                candidate,
                text,
                card,
                query_temporal,
                candidate_temporal,
            ),
            "current_state_compatibility": _current_state_compatibility(
                query,
                candidate,
                text,
                card,
                query_temporal,
                candidate_temporal,
            ),
            "active_not_superseded": _active_not_superseded(candidate, card),
            "answer_span_likelihood": _answer_span_likelihood(
                query,
                candidate,
                text,
                card,
                candidate_temporal,
            ),
            "evidence_chain_availability": _evidence_chain_availability(candidate, card),
        }
        penalties = {
            "stale_fact_penalty": _stale_fact_penalty(candidate, text, card),
            "ack_social_penalty": _ack_social_penalty(candidate, text),
            "topic_only_penalty": _topic_only_penalty(query, candidate, text, card),
        }
        return AnswerabilityDiagnostics(
            score=0.0,
            features=features,
            penalties=penalties,
            baseline_rank=baseline_rank,
            baseline_key=_baseline_key(candidate),
            notes=_diagnostic_notes(features, penalties),
        )


class AnswerabilityRanker:
    """Rerank candidates by deterministic answerability features.
    
        The input order is treated as the existing baseline rank. Equal
        answerability scores therefore preserve the prior rank key exactly.
    
    Args:
        extractor: See the function signature and surrounding type hints.
        config: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(
        self,
        *,
        extractor: AnswerabilityFeatureExtractor | None = None,
        config: AnswerabilityRankerConfig | None = None,
    ) -> None:
        self._extractor = extractor or AnswerabilityFeatureExtractor()
        self._config = config or AnswerabilityRankerConfig()

    def rerank(
        self,
        query: QueryFrame,
        candidates: Iterable[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """Return candidates enriched with diagnostics and answerability order.
        
        Args:
            query: See the function signature and surrounding type hints.
            candidates: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        enriched: list[tuple[float, int, RetrievedEvidence]] = []
        for baseline_rank, candidate in enumerate(candidates):
            diagnostics = self._extractor.extract(
                query,
                candidate,
                baseline_rank=baseline_rank,
            )
            score = self._score(diagnostics)
            diagnostic_payload = AnswerabilityDiagnostics(
                score=score,
                features=diagnostics.features,
                penalties=diagnostics.penalties,
                baseline_rank=diagnostics.baseline_rank,
                baseline_key=diagnostics.baseline_key,
                notes=diagnostics.notes,
            ).to_dict()
            enriched.append((
                score,
                baseline_rank,
                _with_answerability_diagnostics(candidate, diagnostic_payload),
            ))
        return [
            candidate
            for _, _, candidate in sorted(
                enriched,
                key=lambda item: (-_score_bucket(item[0], self._config.tie_epsilon), item[1]),
            )
        ]

    def _score(self, diagnostics: AnswerabilityDiagnostics) -> float:
        feature_score = sum(
            diagnostics.features.get(name, 0.0) * weight
            for name, weight in self._config.feature_weights.items()
        )
        penalty_score = sum(
            diagnostics.penalties.get(name, 0.0) * weight
            for name, weight in self._config.penalty_weights.items()
        )
        return round(feature_score - penalty_score, 6)


def rerank_answerable_evidence(
    query: QueryFrame,
    candidates: Iterable[RetrievedEvidence],
    *,
    config: AnswerabilityRankerConfig | None = None,
) -> list[RetrievedEvidence]:
    """Convenience wrapper for the default deterministic answerability ranker.
    
    Args:
        query: See the function signature and surrounding type hints.
        candidates: See the function signature and surrounding type hints.
        config: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    return AnswerabilityRanker(config=config).rerank(query, candidates)


def _with_answerability_diagnostics(
    candidate: RetrievedEvidence,
    diagnostics: dict[str, Any],
) -> RetrievedEvidence:
    features = {
        **candidate.answerability_features,
        "answerability_diagnostics": diagnostics,
        "answerability_score": diagnostics["score"],
    }
    return RetrievedEvidence(
        evidence_id=candidate.evidence_id,
        evidence_type=candidate.evidence_type,
        source=candidate.source,
        semantic_score=candidate.semantic_score,
        symbolic_score=candidate.symbolic_score,
        temporal_score=candidate.temporal_score,
        answerability_features=features,
        render_payload=candidate.render_payload,
        provenance=candidate.provenance,
    )


def _entity_overlap(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
) -> float:
    query_values = _query_entity_values(query)
    if not query_values:
        return 0.5
    candidate_text = _normalize(text)
    hits = sum(1 for value in query_values if value and value in candidate_text)
    if not hits:
        return 0.0
    return min(1.0, hits / len(query_values))


def _subject_match(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
) -> float:
    subjects = [_normalize(value) for value in query.subject_focus if value]
    if not subjects:
        return 0.5
    card_subject = _normalize(str(card.get("subject", ""))) if card else ""
    candidate_text = _normalize(text)
    matched = any(
        value == card_subject or value in candidate_text
        for value in subjects
    )
    return 1.0 if matched else 0.0


def _predicate_match(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
) -> float:
    predicates = _predicate_terms(query.predicate_focus)
    if not predicates:
        return 0.5
    candidate_tokens = _focus_tokens(
        " ".join(
            [
                str(candidate.answerability_features.get("kind", "")),
                text,
            ]
        )
    )
    if card:
        candidate_tokens.update(
            _focus_tokens(
                " ".join(
                    [
                        str(card.get("kind", "")),
                        str(card.get("predicate", "")),
                        str(card.get("object", "")),
                    ]
                )
            )
        )
    return 1.0 if predicates & candidate_tokens else 0.0


def _query_specificity_match(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
) -> float:
    query_tokens = _query_specific_tokens(query)
    if not query_tokens:
        return 0.5
    candidate_tokens = _focus_tokens(text)
    if card:
        candidate_tokens.update(
            _focus_tokens(
                " ".join(
                    [
                        str(card.get("predicate", "")),
                        str(card.get("object", "")),
                        *[str(value) for value in card.get("surface_forms", [])],
                    ]
                )
            )
        )
    return _overlap_ratio(query_tokens, candidate_tokens)


def _temporal_compatibility(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
    query_temporal: TemporalInterpretation,
    candidate_temporal: TemporalInterpretation,
) -> float:
    status = _validity_status(candidate, card)
    has_time_support = _candidate_has_time_support(candidate, card, candidate_temporal)
    if query_temporal.temporal_mode == "future":
        if candidate_temporal.temporal_mode == "future" or candidate_temporal.event_status == "planned":
            return 1.0
        if candidate_temporal.temporal_mode == "past" or candidate_temporal.event_status == "completed":
            return 0.0
        return 0.6 if has_time_support else 0.25
    if query_temporal.temporal_mode == "past":
        if candidate_temporal.temporal_mode == "future" or candidate_temporal.event_status == "planned":
            return 0.0
        if status in {CARD_STATUS_SUPERSEDED, CARD_STATUS_CORRECTED}:
            return 1.0
        if candidate_temporal.temporal_mode == "past" or candidate_temporal.event_status == "completed":
            return 1.0
        if candidate_temporal.temporal_mode == "current":
            return 0.35
        return 0.7 if has_time_support else 0.3
    if query.historical_vs_current == QUERY_CURRENTNESS_CURRENT or query.temporal_intent in {
        QUERY_TEMPORAL_CURRENT,
        QUERY_TEMPORAL_LATEST,
    }:
        if status in STALE_CARD_STATUSES:
            return 0.0
        if _has_any(_normalize(text), _STALE_CUES):
            return 0.2
        if candidate_temporal.temporal_mode == "past":
            return 0.2
        return 1.0 if candidate_temporal.temporal_mode == "current" else 0.8
    if query.temporal_intent == QUERY_TEMPORAL_RANGE or query_temporal.temporal_mode == "range":
        return 1.0 if has_time_support else 0.4
    if query.answer_type == QUERY_ANSWER_TYPE_TIME:
        if candidate_temporal.temporal_mode == "future" and query_temporal.temporal_mode == "past":
            return 0.0
        if candidate_temporal.temporal_mode == "past" and query_temporal.temporal_mode == "future":
            return 0.0
        if candidate_temporal.has_absolute_time or candidate_temporal.has_relative_time:
            return 1.0
        return 0.7 if has_time_support else 0.2
    return 0.7 if has_time_support else 0.5


def _current_state_compatibility(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
    query_temporal: TemporalInterpretation,
    candidate_temporal: TemporalInterpretation,
) -> float:
    is_current_query = (
        query.historical_vs_current in {QUERY_CURRENTNESS_CURRENT, QUERY_CURRENTNESS_MIXED}
        or query.temporal_intent in {QUERY_TEMPORAL_CURRENT, QUERY_TEMPORAL_LATEST}
        or query_temporal.temporal_mode == "current"
    )
    if not is_current_query:
        return 0.5
    status = _validity_status(candidate, card)
    if status in STALE_CARD_STATUSES:
        return 0.0
    kind = _card_kind(candidate, card)
    if kind == CARD_KIND_CURRENT_STATE or candidate_temporal.temporal_mode == "current":
        return 1.0
    if candidate_temporal.temporal_mode == "past":
        return 0.0
    return 0.7


def _active_not_superseded(
    candidate: RetrievedEvidence,
    card: Mapping[str, Any] | None,
) -> float:
    status = _validity_status(candidate, card)
    return 0.0 if status in STALE_CARD_STATUSES else 1.0


def _answer_span_likelihood(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
    candidate_temporal: TemporalInterpretation,
) -> float:
    normalized = _normalize(text)
    tokens = _tokens(text)
    predicate_supported = _predicate_match(query, candidate, text, card) >= 1.0
    specificity_match = _query_specificity_match(query, candidate, text, card)
    has_card_object = bool(card and str(card.get("object", "")).strip())
    if query.answer_type == QUERY_ANSWER_TYPE_PERSON:
        return 1.0 if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text) else 0.2
    if query.answer_type == QUERY_ANSWER_TYPE_PLACE:
        if any(cue in normalized for cue in (" in ", " at ", " from ")):
            return 1.0
        return 0.7 if has_card_object and (predicate_supported or specificity_match > 0.0) else 0.2
    if query.answer_type == QUERY_ANSWER_TYPE_TIME:
        if candidate_temporal.has_absolute_time or candidate_temporal.has_relative_time:
            return 1.0 if predicate_supported or specificity_match > 0.0 else 0.6
        if has_card_object and (predicate_supported or specificity_match > 0.25):
            return 0.55
        return 0.1
    if query.answer_type == QUERY_ANSWER_TYPE_QUANTITY:
        return 1.0 if re.search(r"\b\d+(?:\.\d+)?\b", normalized) else 0.2
    if query.answer_type == QUERY_ANSWER_TYPE_BOOLEAN:
        return 0.8 if tokens else 0.0
    if has_card_object:
        if predicate_supported or specificity_match > 0.0:
            return 1.0
        return 0.35
    return 0.8 if len(tokens) >= 3 else 0.2


def _topic_only_penalty(
    query: QueryFrame,
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
) -> float:
    query_tokens = _query_specific_tokens(query)
    if not query_tokens and not query.predicate_focus:
        return 0.0
    subject_match = _subject_match(query, candidate, text, card)
    entity_overlap = _entity_overlap(query, candidate, text)
    predicate_match = _predicate_match(query, candidate, text, card)
    specificity_match = _query_specificity_match(query, candidate, text, card)
    if max(predicate_match, specificity_match) > 0.0:
        return 0.0
    if subject_match > 0.0 or entity_overlap > 0.0:
        return 1.0 if query.answer_type == QUERY_ANSWER_TYPE_TIME else 0.8
    return 0.0


def _evidence_chain_availability(
    candidate: RetrievedEvidence,
    card: Mapping[str, Any] | None,
) -> float:
    score = 0.0
    if candidate.render_payload:
        score += 0.35
    if candidate.provenance.get("source_record_id"):
        score += 0.35
    if card and candidate.provenance.get("card_id"):
        score += 0.2
    if candidate.provenance.get("source_turn") is not None:
        score += 0.1
    return round(min(1.0, score), 6)


def _stale_fact_penalty(
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
) -> float:
    status = _validity_status(candidate, card)
    if status in STALE_CARD_STATUSES:
        return 1.0
    normalized = _normalize(text)
    return 0.7 if _has_any(normalized, _STALE_CUES) else 0.0


def _ack_social_penalty(candidate: RetrievedEvidence, text: str) -> float:
    normalized = _normalize(text)
    role = _normalize(str(candidate.provenance.get("speaker_role", "")))
    if role == "assistant" and _has_any(normalized, _ACK_SOCIAL_CUES):
        return 1.0
    if len(_tokens(text)) <= 3 and _has_any(normalized, _ACK_SOCIAL_CUES):
        return 0.8
    return 0.0


def _diagnostic_notes(
    features: Mapping[str, float],
    penalties: Mapping[str, float],
) -> list[str]:
    notes: list[str] = []
    notes.extend(f"{name}=low" for name, value in features.items() if value <= 0.0)
    notes.extend(f"{name}=active" for name, value in penalties.items() if value > 0.0)
    return notes


def _baseline_key(candidate: RetrievedEvidence) -> dict[str, float | int]:
    return {
        "semantic_score": candidate.semantic_score or 0.0,
        "symbolic_score": candidate.symbolic_score or 0.0,
        "temporal_score": candidate.temporal_score or 0.0,
    }


def _candidate_text(candidate: RetrievedEvidence) -> str:
    card = _card_payload(candidate)
    if card is not None:
        pieces = [
            card.get("kind"),
            card.get("subject"),
            card.get("predicate"),
            card.get("object"),
            *card.get("surface_forms", []),
        ]
        return " ".join(str(piece) for piece in pieces if piece)
    record = candidate.render_payload.get("record")
    if isinstance(record, Mapping):
        provenance = record.get("provenance", {})
        provenance_role = (
            provenance.get("role")
            if isinstance(provenance, Mapping)
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
    text = candidate.answerability_features.get("text")
    return str(text) if text is not None else ""


def _card_payload(candidate: RetrievedEvidence) -> Mapping[str, Any] | None:
    card = candidate.render_payload.get("card")
    return card if isinstance(card, Mapping) else None


def _validity_status(
    candidate: RetrievedEvidence,
    card: Mapping[str, Any] | None,
) -> str:
    value = candidate.answerability_features.get("validity_status")
    if value:
        return _normalize(str(value))
    if card:
        validity = card.get("validity", {})
        if isinstance(validity, Mapping):
            return _normalize(str(validity.get("status", CARD_STATUS_ACTIVE)))
    return CARD_STATUS_ACTIVE


def _card_kind(
    candidate: RetrievedEvidence,
    card: Mapping[str, Any] | None,
) -> str:
    value = candidate.answerability_features.get("kind")
    if value:
        return _normalize(str(value))
    if card:
        return _normalize(str(card.get("kind", "")))
    return ""


def _query_temporal_interpretation(query: QueryFrame) -> TemporalInterpretation:
    interpretation = interpretation_from_query_frame_filters(query.filters)
    if interpretation is None:
        interpretation = interpret_temporal_text(query.query_text)
    if query.temporal_intent == QUERY_TEMPORAL_RANGE and interpretation.temporal_mode == "unspecified":
        interpretation = TemporalInterpretation(
            temporal_mode="range",
            event_status=interpretation.event_status,
            has_absolute_time=interpretation.has_absolute_time,
            has_relative_time=interpretation.has_relative_time,
            time_anchor_text=interpretation.time_anchor_text,
        )
    if query.historical_vs_current == QUERY_CURRENTNESS_CURRENT and interpretation.temporal_mode == "unspecified":
        interpretation = TemporalInterpretation(
            temporal_mode=QUERY_TEMPORAL_CURRENT,
            event_status="state",
            has_absolute_time=interpretation.has_absolute_time,
            has_relative_time=interpretation.has_relative_time,
            time_anchor_text=interpretation.time_anchor_text,
        )
    if query.historical_vs_current == "historical" and interpretation.temporal_mode == "unspecified":
        interpretation = TemporalInterpretation(
            temporal_mode="past",
            event_status="completed",
            has_absolute_time=interpretation.has_absolute_time,
            has_relative_time=interpretation.has_relative_time,
            time_anchor_text=interpretation.time_anchor_text,
        )
    return interpretation


def _candidate_temporal_interpretation(
    candidate: RetrievedEvidence,
    text: str,
    card: Mapping[str, Any] | None,
) -> TemporalInterpretation:
    return interpret_temporal_text(
        text,
        kind=_card_kind(candidate, card),
        validity_status=_validity_status(candidate, card),
    )


def _candidate_has_time_support(
    candidate: RetrievedEvidence,
    card: Mapping[str, Any] | None,
    interpretation: TemporalInterpretation,
) -> bool:
    return bool(
        interpretation.has_absolute_time
        or interpretation.has_relative_time
        or candidate.temporal_score is not None
        or _time_anchor(card)
    )


def _time_anchor(card: Mapping[str, Any] | None) -> bool:
    if not card:
        return False
    anchor = card.get("time_anchor", {})
    return isinstance(anchor, Mapping) and any(anchor.values())


def _query_entity_values(query: QueryFrame) -> list[str]:
    values = [*query.entities, *query.subject_focus]
    for aliases in query.aliases.values():
        values.extend(str(alias) for alias in aliases)
    return _dedupe(_normalize(value) for value in values if len(_normalize(value)) >= 3)


def _query_specific_tokens(query: QueryFrame) -> set[str]:
    blocked = {
        token
        for value in [*query.entities, *query.subject_focus]
        for token in lemma_terms(value, extra_stop_words=_QUERY_SPECIFIC_STOP_WORDS)
        if token
    }
    tokens = _focus_tokens(" ".join([query.query_text, *query.predicate_focus]))
    return {
        token
        for token in tokens
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


def _normalize(value: str) -> str:
    return normalize_text(value)


def _predicate_terms(values: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        terms.update(_focus_tokens(value))
    return {term for term in terms if term}


def _has_any(normalized: str, cues: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(cue)}\b", normalized) for cue in cues)


def _bounded(value: float | None) -> float:
    if value is None:
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _score_bucket(score: float, epsilon: float) -> int:
    return int(round(score / epsilon))


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return min(1.0, len(overlap) / len(left))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
