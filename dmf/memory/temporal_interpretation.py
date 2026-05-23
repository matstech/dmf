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

"""Shared deterministic temporal interpretation for structured retrieval.

The goal of this module is to keep temporal reasoning in one place instead of
spreading additional English cue lists across query parsing and reranking.
It prefers existing NLP analysis when available and falls back to a small,
centralised spaCy-based interpretation pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from dmf.models.constants import STALE_CARD_STATUSES
from dmf.models.analysis import AnalysisReport
from dmf.utils.constants import DEFAULT_SPACY_MODEL


_TIME_ENTITY_LABELS = frozenset({"DATE", "TIME"})
_RELATIVE_PAST_LEMMAS = frozenset(
    {
        "ago",
        "earlier",
        "last",
        "previous",
        "yesterday",
    }
)
_RELATIVE_FUTURE_LEMMAS = frozenset(
    {
        "later",
        "next",
        "soon",
        "tomorrow",
        "upcoming",
    }
)
_RELATIVE_TIME_LEMMAS = _RELATIVE_PAST_LEMMAS | _RELATIVE_FUTURE_LEMMAS | {"today"}
_CURRENT_MARKER_LEMMAS = frozenset({"current", "currently", "latest", "now", "recent"})
_PAST_MARKER_LEMMAS = frozenset({"before", "former", "formerly", "old", "past", "previous"})
_RANGE_MARKER_LEMMAS = frozenset({"between", "during", "from", "since", "timeline", "until"})
_PLAN_LEMMAS = frozenset({"arrange", "book", "intend", "plan", "schedule"})
_STALE_STATUSES = STALE_CARD_STATUSES


@dataclass(frozen=True)
class TemporalInterpretation:
    """Deterministic temporal profile for a query or one evidence span.
    
    Args:
        temporal_mode: See the function signature and surrounding type hints.
        event_status: See the function signature and surrounding type hints.
        has_absolute_time: See the function signature and surrounding type hints.
        has_relative_time: See the function signature and surrounding type hints.
        time_anchor_text: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    temporal_mode: str = "unspecified"
    event_status: str = "unknown"
    has_absolute_time: bool = False
    has_relative_time: bool = False
    time_anchor_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "temporal_mode": self.temporal_mode,
            "event_status": self.event_status,
            "has_absolute_time": self.has_absolute_time,
            "has_relative_time": self.has_relative_time,
            "time_anchor_text": self.time_anchor_text,
        }


def interpret_temporal_text(
    text: str,
    *,
    analysis: AnalysisReport | None = None,
    kind: str | None = None,
    validity_status: str | None = None,
) -> TemporalInterpretation:
    """Return a deterministic temporal interpretation for one text span.
    
    Args:
        text: See the function signature and surrounding type hints.
        analysis: See the function signature and surrounding type hints.
        kind: See the function signature and surrounding type hints.
        validity_status: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    doc = _doc_for_text(text)
    time_entities = _time_entities(doc, analysis)
    has_relative_time = any(_span_is_relative(entity) for entity in time_entities)
    has_absolute_time = any(not _span_is_relative(entity) for entity in time_entities)
    time_anchor_text = next((entity.text.strip() for entity in time_entities if entity.text.strip()), None)

    stale_status = str(validity_status or "").casefold() in _STALE_STATUSES
    current_signal = bool(analysis and analysis.signals.is_current_state)
    past_signal = bool(analysis and analysis.signals.is_past_state)

    future_plan = _is_future_plan(doc)
    past_completed = stale_status or past_signal or _is_past_completed(doc)
    current_state = (
        (kind or "").casefold() == "current_state"
        and not stale_status
    ) or current_signal or _has_marker(doc, _CURRENT_MARKER_LEMMAS)
    temporal_range = _has_marker(doc, _RANGE_MARKER_LEMMAS)
    progressive_ongoing = _is_progressive_ongoing(doc) and not future_plan

    event_status = "unknown"
    if (kind or "").casefold() == "current_state":
        event_status = "state"
    elif future_plan:
        event_status = "planned"
    elif past_completed:
        event_status = "completed"
    elif current_state:
        event_status = "state"
    elif progressive_ongoing:
        event_status = "ongoing"

    if temporal_range:
        temporal_mode = "range"
    elif future_plan:
        temporal_mode = "future"
    elif past_completed or _has_marker(doc, _PAST_MARKER_LEMMAS):
        temporal_mode = "past"
    elif current_state or progressive_ongoing:
        temporal_mode = "current"
    elif has_absolute_time or has_relative_time:
        temporal_mode = "past" if has_relative_time else "range"
    else:
        temporal_mode = "unspecified"

    return TemporalInterpretation(
        temporal_mode=temporal_mode,
        event_status=event_status,
        has_absolute_time=has_absolute_time,
        has_relative_time=has_relative_time,
        time_anchor_text=time_anchor_text,
    )


def interpretation_from_query_frame_filters(filters: dict[str, Any]) -> TemporalInterpretation | None:
    """Hydrate a temporal interpretation from serialized query filters.
    
    Args:
        filters: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    mode = str(filters.get("temporal_mode", "")).strip()
    status = str(filters.get("event_status", "")).strip()
    if not mode and not status:
        return None
    anchor = filters.get("time_anchor_text")
    return TemporalInterpretation(
        temporal_mode=mode or "unspecified",
        event_status=status or "unknown",
        has_absolute_time=bool(filters.get("has_absolute_time", False)),
        has_relative_time=bool(filters.get("has_relative_time", False)),
        time_anchor_text=str(anchor) if isinstance(anchor, str) and anchor.strip() else None,
    )


@lru_cache(maxsize=4096)
def _doc_for_text(text: str) -> Doc:
    return _nlp()(text.strip())


@lru_cache(maxsize=1)
def _nlp() -> Language:
    try:
        return spacy.load(DEFAULT_SPACY_MODEL)
    except OSError:
        return spacy.blank("en")


def _time_entities(doc: Doc, analysis: AnalysisReport | None) -> tuple[Span, ...]:
    if analysis is not None:
        spans = _analysis_time_entities(doc, analysis)
        if spans:
            return spans
    return tuple(entity for entity in doc.ents if entity.label_ in _TIME_ENTITY_LABELS)


def _analysis_time_entities(doc: Doc, analysis: AnalysisReport) -> tuple[Span, ...]:
    values: list[Span] = []
    raw_entities = analysis.raw_metadata.get("entities", [])
    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("label", "")).upper() not in _TIME_ENTITY_LABELS:
            continue
        text = str(entity.get("text", "")).strip()
        span = _match_span(doc, text)
        if span is not None:
            values.append(span)
    return tuple(values)


def _match_span(doc: Doc, text: str) -> Span | None:
    if not text:
        return None
    lowered = text.casefold()
    for entity in doc.ents:
        if entity.text.casefold() == lowered:
            return entity
    start = doc.text.casefold().find(lowered)
    if start >= 0:
        return doc.char_span(start, start + len(text))
    return None


def _span_is_relative(span: Span) -> bool:
    return any(_token_lemma(token) in _RELATIVE_TIME_LEMMAS for token in span)


def _is_future_plan(doc: Doc) -> bool:
    if any(token.tag_ == "MD" and _token_lemma(token) in {"shall", "will"} for token in doc):
        return True
    if any(
        entity.label_ in _TIME_ENTITY_LABELS
        and any(_token_lemma(token) in _RELATIVE_FUTURE_LEMMAS for token in entity)
        for entity in doc.ents
    ):
        return True
    return any(
        _planning_token_has_future_scope(token) or _is_going_to_future(token)
        for token in doc
    )


def _is_past_completed(doc: Doc) -> bool:
    if any(_aux_is_past(token) for token in doc):
        return True
    if any(_verb_is_past(token) for token in doc):
        return True
    return any(
        entity.label_ in _TIME_ENTITY_LABELS
        and any(_token_lemma(token) in _RELATIVE_PAST_LEMMAS for token in entity)
        for entity in doc.ents
    )


def _is_progressive_ongoing(doc: Doc) -> bool:
    return any(
        token.tag_ == "VBG"
        and any(child.dep_ in {"aux", "auxpass"} and _token_lemma(child) == "be" for child in token.children)
        for token in doc
    )


def _has_marker(doc: Doc, markers: frozenset[str]) -> bool:
    return any(_token_lemma(token) in markers for token in doc)


def _is_going_to_future(token: Token) -> bool:
    if _token_lemma(token) != "go":
        return False
    if token.tag_ not in {"VB", "VBG", "VBP", "VBZ"}:
        return False
    if token.i + 1 >= len(token.doc) or _token_lemma(token.doc[token.i + 1]) != "to":
        return False
    return any(child.dep_ in {"xcomp", "ccomp"} and child.pos_ == "VERB" for child in token.children)


def _planning_token_has_future_scope(token: Token) -> bool:
    if _token_lemma(token) not in _PLAN_LEMMAS:
        return False
    if _token_is_past(token):
        return False
    if any(
        _aux_is_past(child) for child in token.children if child.dep_ in {"aux", "auxpass"}
    ):
        return False
    if any(
        child.dep_ in {"xcomp", "ccomp", "advcl"} and child.pos_ == "VERB"
        for child in token.children
    ):
        return True
    return any(
        child.dep_ == "prep"
        and any(
            grandchild.dep_ == "pcomp" and grandchild.pos_ == "VERB"
            for grandchild in child.children
        )
        for child in token.children
    )


def _aux_is_past(token: Token) -> bool:
    if token.dep_ not in {"aux", "auxpass"}:
        return False
    if token.tag_ == "VBD":
        return True
    return "Past" in token.morph.get("Tense")


def _verb_is_past(token: Token) -> bool:
    if token.pos_ != "VERB":
        return False
    if token.tag_ in {"VBD", "VBN"}:
        return True
    return "Past" in token.morph.get("Tense")


def _token_is_past(token: Token) -> bool:
    if _verb_is_past(token):
        return True
    return any(
        _aux_is_past(child)
        for child in token.children
        if child.dep_ in {"aux", "auxpass"}
    )


def _token_lemma(token: Token) -> str:
    lemma = (token.lemma_ or token.text).strip().casefold()
    return lemma
