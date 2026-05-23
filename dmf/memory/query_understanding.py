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

"""Deterministic query-understanding for future memory retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from dmf.memory.lexical_normalization import (
    dedupe_casefold,
    entity_spans,
    normalize_text,
    predicate_terms,
)
from dmf.memory.constants import (
    CURRENT_TEMPORAL_CUES,
    HISTORICAL_TEMPORAL_CUES,
    LATEST_TEMPORAL_CUES,
    RANGE_TEMPORAL_CUES,
)
from dmf.memory.temporal_interpretation import (
    TemporalInterpretation,
    interpret_temporal_text,
)
from dmf.models.analysis import AnalysisReport
from dmf.models.memory import QueryFrame


class QueryAnalyzer(Protocol):
    """Minimal protocol for reusing the existing NLP engine when supplied.
    
    Args:
        None.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def analyze_interaction(self, text: str, is_system: bool = False) -> AnalysisReport:
        """Return a deterministic analysis report for one text span.
        
        Args:
            text: See the function signature and surrounding type hints.
            is_system: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """


CURRENT_CUES = CURRENT_TEMPORAL_CUES
HISTORICAL_CUES = HISTORICAL_TEMPORAL_CUES
LATEST_CUES = LATEST_TEMPORAL_CUES
RANGE_CUES = RANGE_TEMPORAL_CUES
NEGATION_CUES = (
    "not",
    "no longer",
    "never",
    "without",
    "don't",
    "do not",
    "doesn't",
    "does not",
    "can't",
    "cannot",
    "isn't",
    "is not",
    "aren't",
    "are not",
    "wasn't",
    "was not",
)
BOOLEAN_STARTERS = (
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "should",
    "was",
    "were",
    "will",
    "would",
)
QUESTION_CUES = set(BOOLEAN_STARTERS) | {
    "describe",
    "how",
    "list",
    "summarize",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
PREDICATE_STOP_WORDS = QUESTION_CUES | {
    "a",
    "an",
    "at",
    "by",
    "for",
    "from",
    "her",
    "his",
    "i",
    "in",
    "it",
    "me",
    "mine",
    "my",
    "of",
    "on",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "we",
    "your",
}
STOP_ENTITIES = {
    "Did",
    "Do",
    "Does",
    "Has",
    "Have",
    "How",
    "Is",
    "List",
    "Tell",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "Would",
}
ENTITY_PREFIX_NOISE = {value.casefold() for value in QUESTION_CUES | set(BOOLEAN_STARTERS)}
PREDICATE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prefer", ("prefer", "preference", "favorite", "like", "likes", "liked")),
    ("live", ("live", "lives", "living", "address", "located", "location")),
    ("work", ("work", "works", "job", "role", "employer", "company")),
    ("name", ("name", "called", "call")),
    ("birth_date", ("birthday", "born", "birth date")),
    (
        "relation",
        ("friend", "manager", "sister", "brother", "wife", "husband", "partner"),
    ),
    ("contact", ("email", "phone", "number", "contact")),
    ("event", ("happen", "happened", "scheduled", "booked", "visited", "met")),
    ("constraint", ("avoid", "must", "should", "allowed", "forbidden", "cannot")),
    ("correction", ("correct", "correction", "actually", "instead")),
)
SUBJECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:for|about|of|does|did|is|was|are|were)\s+"
        r"([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3})"
    ),
    re.compile(r"\bmy\s+([a-z][\w'-]*(?:\s+[a-z][\w'-]*){0,2})"),
)


class QueryUnderstandingParser:
    """Rule-based query parser with no LLM or benchmark-specific categories.
    
    Args:
        analyzer: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(self, analyzer: QueryAnalyzer | None = None) -> None:
        self._analyzer = analyzer

    def parse(
        self,
        query_text: str,
        *,
        query_embedding: Sequence[float] | None = None,
    ) -> QueryFrame:
        """Build a stable ``QueryFrame`` from raw query text.
        
                Embeddings are accepted as an injected value rather than computed
                here, keeping the parser pure and independent from embedding
                providers, caches, credentials, and retrieval orchestration.
        
        Args:
            query_text: See the function signature and surrounding type hints.
            query_embedding: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        text = query_text.strip()
        analysis = self._analyze(text)
        entities = self._extract_entities(text, analysis)
        subject_focus = self._extract_subject_focus(text, entities)
        predicate_focus = self._extract_predicate_focus(text, entities)
        temporal_profile = interpret_temporal_text(text, analysis=analysis)
        temporal_intent = self._extract_temporal_intent(text, analysis, temporal_profile)
        historical_vs_current = self._extract_currentness_focus(
            temporal_intent,
            text,
            analysis,
            temporal_profile,
        )
        polarity = self._extract_polarity(text, analysis)
        filters = self._extract_filters(
            text,
            temporal_intent=temporal_intent,
            historical_vs_current=historical_vs_current,
            polarity=polarity,
            temporal_profile=temporal_profile,
        )

        return QueryFrame(
            query_text=text,
            query_embedding=None
            if query_embedding is None
            else [float(value) for value in query_embedding],
            entities=entities,
            aliases=self._build_aliases(entities),
            subject_focus=subject_focus,
            predicate_focus=predicate_focus,
            answer_type=self._extract_answer_type(text),
            temporal_intent=temporal_intent,
            historical_vs_current=historical_vs_current,
            polarity=polarity,
            filters=filters,
        )

    def _analyze(self, text: str) -> AnalysisReport | None:
        if self._analyzer is None or not text:
            return None
        return self._analyzer.analyze_interaction(text)

    def _extract_entities(
        self,
        text: str,
        analysis: AnalysisReport | None,
    ) -> list[str]:
        candidates: list[str] = []
        if analysis is not None:
            for entity in analysis.raw_metadata.get("entities", []):
                if isinstance(entity, dict):
                    candidates.append(str(entity.get("text", "")))

        candidates.extend(entity_spans(text))
        candidates.extend(re.findall(r'"([^"]+)"|\'([^\']+)\'', text))
        candidates.extend(
            match.group(0)
            for match in re.finditer(
                r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3}\b",
                text,
            )
        )
        flat_candidates = [
            self._clean_span(part)
            for candidate in candidates
            for part in (candidate if isinstance(candidate, tuple) else (candidate,))
        ]
        return self._dedupe(
            value
            for value in flat_candidates
            if value and value not in STOP_ENTITIES and not value.isdigit()
        )

    def _extract_subject_focus(self, text: str, entities: Sequence[str]) -> list[str]:
        subjects = list(entities)
        for pattern in SUBJECT_PATTERNS:
            for match in pattern.finditer(text):
                value = self._clean_span(match.group(1))
                if value and value not in STOP_ENTITIES:
                    subjects.append(value)
        if re.search(r"\b(my|me|i|mine)\b", text, re.IGNORECASE):
            subjects.append("user")
        return self._dedupe(subjects)

    def _extract_predicate_focus(
        self,
        text: str,
        entities: Sequence[str],
    ) -> list[str]:
        normalized = normalize_text(text)
        predicates: list[str] = []
        for predicate, cues in PREDICATE_CUES:
            if any(self._contains_phrase(normalized, cue) for cue in cues):
                predicates.append(predicate)
        if predicates and "event" not in predicates:
            return self._dedupe(predicates)
        return self._dedupe([*predicates, *self._extract_surface_predicates(text, entities)])

    def _extract_answer_type(self, text: str) -> str:
        normalized = self._normalize(text)
        first_word = normalized.split(" ", 1)[0] if normalized else ""
        if normalized.startswith("who ") or normalized.startswith("whose "):
            return "person"
        if normalized.startswith("where "):
            return "place"
        if normalized.startswith("when ") or " what time " in f" {normalized} ":
            return "time"
        if normalized.startswith(("how many ", "how much ", "how old ")):
            return "quantity"
        if normalized.startswith("why "):
            return "reason"
        if (
            normalized.startswith(("list ", "which "))
            or " what are " in f" {normalized} "
        ):
            return "list"
        if first_word in BOOLEAN_STARTERS:
            return "boolean"
        if normalized.startswith(("what ", "tell me ", "describe ", "summarize ")):
            return "fact"
        return "unknown"

    def _extract_temporal_intent(
        self,
        text: str,
        analysis: AnalysisReport | None,
        temporal_profile: TemporalInterpretation,
    ) -> str:
        normalized = self._normalize(text)
        if self._has_any(normalized, LATEST_CUES):
            return "latest"
        if temporal_profile.temporal_mode == "range":
            return "temporal_range"
        if temporal_profile.temporal_mode == "current" and temporal_profile.event_status in {
            "ongoing",
            "state",
        }:
            return "current"
        if self._has_any(normalized, CURRENT_CUES):
            return "current"
        if temporal_profile.temporal_mode == "past" and temporal_profile.event_status == "state":
            return "historical"
        if self._has_any(normalized, HISTORICAL_CUES):
            return "historical"
        if self._has_any(normalized, RANGE_CUES):
            return "temporal_range"
        if analysis is not None:
            signals = analysis.signals
            if signals.is_current_state:
                return "current"
            if signals.is_past_state:
                return "historical"
            if signals.temporal_markers:
                return "temporal_range"
        return "none"

    def _extract_currentness_focus(
        self,
        temporal_intent: str,
        text: str,
        analysis: AnalysisReport | None,
        temporal_profile: TemporalInterpretation,
    ) -> str:
        normalized = self._normalize(text)
        has_current = temporal_intent in {"current", "latest"} or self._has_any(
            normalized,
            CURRENT_CUES + LATEST_CUES,
        )
        has_historical = temporal_intent == "historical" or self._has_any(
            normalized,
            HISTORICAL_CUES,
        )
        if temporal_profile.temporal_mode == "current":
            has_current = True
        if temporal_profile.temporal_mode == "past":
            has_historical = True
        if analysis is not None:
            has_current = has_current or analysis.signals.is_current_state
            has_historical = has_historical or analysis.signals.is_past_state
        if has_current and has_historical:
            return "mixed"
        if has_current:
            return "current"
        if has_historical:
            return "historical"
        return "unspecified"

    def _extract_polarity(
        self,
        text: str,
        analysis: AnalysisReport | None,
    ) -> str:
        normalized = self._normalize(text)
        if self._has_any(normalized, NEGATION_CUES):
            return "negative"
        if analysis is not None and analysis.signals.has_negation:
            return "negative"
        return "positive"

    def _extract_filters(
        self,
        text: str,
        *,
        temporal_intent: str,
        historical_vs_current: str,
        polarity: str,
        temporal_profile: TemporalInterpretation,
    ) -> dict[str, Any]:
        normalized = self._normalize(text)
        filters: dict[str, Any] = {}
        temporal_cues = self._matched_cues(
            normalized,
            CURRENT_CUES + HISTORICAL_CUES + LATEST_CUES + RANGE_CUES,
        )
        if temporal_cues:
            filters["temporal_cues"] = temporal_cues
        if temporal_intent != "none":
            filters["temporal_intent"] = temporal_intent
        if historical_vs_current != "unspecified":
            filters["historical_vs_current"] = historical_vs_current
        filters["temporal_mode"] = temporal_profile.temporal_mode
        filters["event_status"] = temporal_profile.event_status
        if temporal_profile.has_absolute_time:
            filters["has_absolute_time"] = True
        if temporal_profile.has_relative_time:
            filters["has_relative_time"] = True
        if temporal_profile.time_anchor_text:
            filters["time_anchor_text"] = temporal_profile.time_anchor_text
        if polarity == "negative":
            filters["polarity"] = polarity
            filters["negation_cues"] = self._matched_cues(normalized, NEGATION_CUES)
        return filters

    def _build_aliases(self, entities: Sequence[str]) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        for entity in entities:
            values = [entity.casefold()]
            parts = entity.split()
            if len(parts) > 1:
                values.extend(part.casefold() for part in parts)
            aliases[entity] = self._dedupe(values)
        return aliases

    def _matched_cues(self, normalized: str, cues: Iterable[str]) -> list[str]:
        return self._dedupe(cue for cue in cues if self._contains_phrase(normalized, cue))

    def _extract_surface_predicates(
        self,
        text: str,
        entities: Sequence[str],
    ) -> list[str]:
        blocked_tokens = {
            token
            for value in [*entities, *CURRENT_CUES, *HISTORICAL_CUES, *LATEST_CUES]
            for token in predicate_terms(value)
        }
        return self._dedupe(
            value
            for value in predicate_terms(text, blocked_terms=blocked_tokens)
            if value not in PREDICATE_STOP_WORDS and value not in blocked_tokens
        )

    def _has_any(self, normalized: str, cues: Iterable[str]) -> bool:
        return any(self._contains_phrase(normalized, cue) for cue in cues)

    def _contains_phrase(self, normalized: str, cue: str) -> bool:
        return re.search(rf"\b{re.escape(cue)}\b", normalized) is not None

    def _normalize(self, text: str) -> str:
        return normalize_text(text)

    def _clean_span(self, value: str | tuple[str, ...]) -> str:
        if isinstance(value, tuple):
            value = next((part for part in value if part), "")
        cleaned = re.sub(r"\s+", " ", str(value).strip(" .,:;!?\"'")).strip()
        cleaned = re.sub(r"'s\b", "", cleaned).strip()
        parts = cleaned.split()
        while len(parts) > 1 and parts[0].casefold() in ENTITY_PREFIX_NOISE:
            parts = parts[1:]
        return " ".join(parts)

    def _dedupe(self, values: Iterable[str]) -> list[str]:
        return dedupe_casefold(values)


def parse_query_frame(
    query_text: str,
    *,
    query_embedding: Sequence[float] | None = None,
    analyzer: QueryAnalyzer | None = None,
) -> QueryFrame:
    """Parse a query with the default deterministic query-understanding parser.
    
    Args:
        query_text: See the function signature and surrounding type hints.
        query_embedding: See the function signature and surrounding type hints.
        analyzer: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    return QueryUnderstandingParser(analyzer=analyzer).parse(
        query_text,
        query_embedding=query_embedding,
    )
