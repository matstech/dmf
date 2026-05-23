"""Deterministic projection from raw memory entries to auxiliary cards."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from dmf.models.constants import (
    CARD_KIND_CONSTRAINT,
    CARD_KIND_CORRECTION,
    CARD_KIND_CURRENT_STATE,
    CARD_KIND_EVENT,
    CARD_KIND_FACT,
    CARD_KIND_PREFERENCE,
    CARD_KIND_RELATION,
    CARD_STATUS_ACTIVE,
    CARD_STATUS_CORRECTED,
    CARD_STATUS_INVALIDATED,
    CARD_STATUS_SUPERSEDED,
    UNKNOWN_VALUE,
)
from dmf.models.memory import (
    MemoryEntry,
    MemoryCard,
    MemoryCardProvenance,
    MemoryCardTimeAnchor,
    MemoryCardValidity,
)

_EVENT_VERBS = {
    "booked",
    "bought",
    "completed",
    "finished",
    "met",
    "moved",
    "scheduled",
    "started",
    "visited",
}
_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(.+?)\s+is\s+my\s+(.+?)\b", re.IGNORECASE), "is_my"),
    (re.compile(r"\bmy\s+(.+?)\s+is\s+(.+?)\b", re.IGNORECASE), "my_is"),
)


class MemoryCardProjector:
    """Conservative, benchmark-agnostic memory-card projector.

    Args:
        None.

    Returns:
        Projector instance that derives auxiliary cards from raw memory entries.

    Raises:
        None.

    Warning:
        The projector intentionally favors precision over recall. Missing a card
        is preferable to creating authoritative structured memory from weak
        conversational evidence.
    """

    def project(self, entry: MemoryEntry) -> list[MemoryCard]:
        """Project one raw memory entry into zero or more structured cards.

        Args:
            entry: Working-memory entry to project.

        Returns:
            Zero or more memory cards derived from stable signals.

        Raises:
            ValueError: If projected card fields violate the card schema.
        """
        if not self._is_projectable(entry):
            return []

        spec = self._classify(entry)
        if spec is None:
            return []

        kind, subject, predicate, obj, qualifiers, confidence = spec
        return [
            MemoryCard(
                card_id=f"card:{entry.record_id}:0",
                kind=kind,
                subject=subject,
                predicate=predicate,
                object=obj,
                qualifiers=qualifiers,
                time_anchor=MemoryCardTimeAnchor(
                    turn_id=entry.interaction_id,
                    absolute_time=entry.timestamp,
                    relative_order=entry.interaction_id,
                ),
                validity=self._validity_for(entry),
                provenance=MemoryCardProvenance(
                    source_record_id=entry.record_id,
                    speaker_role=entry.provenance.role or UNKNOWN_VALUE,
                    source_turn=entry.provenance.source_turn,
                    interaction=entry.provenance,
                ),
                confidence=confidence,
                surface_forms=self._surface_forms(entry, obj),
            )
        ]

    def _is_projectable(self, entry: MemoryEntry) -> bool:
        """Return False for interactions that should not become cards."""
        text = entry.text.strip()
        return bool(text) and not entry.report.is_query_like and not entry.report.is_ack_like

    def _classify(
        self,
        entry: MemoryEntry,
    ) -> tuple[str, str, str, str, dict[str, Any], float] | None:
        """Return a normalized card spec or None when confidence is too low."""
        report = entry.report
        signals = report.signals
        topic_identity = report.topic_identity
        topic_value = report.topic_value

        if topic_identity and topic_value:
            parts = topic_identity.split("|")
            topic_kind = parts[0]
            predicate = parts[1] if len(parts) > 1 else topic_kind
            subject = parts[2] if len(parts) > 2 else self._default_subject(entry)
            qualifiers = self._base_qualifiers(entry)

            if signals.is_correction or entry.provenance.is_user_correction:
                return (CARD_KIND_CORRECTION, subject, predicate, topic_value, qualifiers, 0.86)
            if signals.is_constraint:
                return (CARD_KIND_CONSTRAINT, subject, predicate, topic_value, qualifiers, 0.84)
            if signals.is_preference:
                return (CARD_KIND_PREFERENCE, subject, predicate, topic_value, qualifiers, 0.84)
            if signals.is_current_state:
                return (CARD_KIND_CURRENT_STATE, subject, predicate, topic_value, qualifiers, 0.82)
            if topic_kind == CARD_KIND_RELATION:
                return (CARD_KIND_RELATION, subject, predicate, topic_value, qualifiers, 0.78)
            if topic_kind == CARD_KIND_FACT:
                return (CARD_KIND_FACT, subject, predicate, topic_value, qualifiers, 0.74)

        relation = self._relation_from_text(entry)
        if relation is not None:
            subject, predicate, obj = relation
            return (CARD_KIND_RELATION, subject, predicate, obj, self._base_qualifiers(entry), 0.66)

        if self._looks_like_simple_event(entry):
            return (
                CARD_KIND_EVENT,
                self._default_subject(entry),
                "mentions_event",
                entry.text.strip(),
                self._base_qualifiers(entry),
                0.58,
            )

        return None

    def _validity_for(self, entry: MemoryEntry) -> MemoryCardValidity:
        """Build minimal validity state from existing lineage/provenance."""
        status = CARD_STATUS_ACTIVE
        if entry.provenance.corrected_by_user:
            status = CARD_STATUS_INVALIDATED
        elif entry.report.signals.is_correction or entry.provenance.is_user_correction:
            status = CARD_STATUS_CORRECTED

        return MemoryCardValidity(
            status=status,
            supersedes=list(entry.lineage.supersedes),
            invalidates=list(entry.lineage.invalidates),
            corrects=list(entry.lineage.corrects),
        )

    def _base_qualifiers(self, entry: MemoryEntry) -> dict[str, Any]:
        """Carry stable, non-authoritative qualifiers from existing analysis."""
        signals = entry.report.signals
        qualifiers: dict[str, Any] = {}
        if signals.has_negation:
            qualifiers["negated"] = True
        if signals.is_past_state:
            qualifiers["temporal_state"] = "past"
        if signals.is_current_state:
            qualifiers["temporal_state"] = "current"
        if signals.temporal_markers:
            qualifiers["temporal_markers"] = list(signals.temporal_markers)
        return qualifiers

    def _surface_forms(self, entry: MemoryEntry, obj: str) -> list[str]:
        """Return compact source forms useful for audit and debugging."""
        forms = [entry.text.strip()]
        if obj and obj not in forms:
            forms.append(obj)
        return forms

    def _default_subject(self, entry: MemoryEntry) -> str:
        """Use the speaker role as the safest available subject fallback."""
        return entry.provenance.role or UNKNOWN_VALUE

    def _relation_from_text(self, entry: MemoryEntry) -> tuple[str, str, str] | None:
        """Extract very narrow relation patterns without full claim parsing."""
        text = entry.text.strip()
        for pattern, predicate in _RELATION_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            left, right = (self._clean_span(value) for value in match.groups())
            if left and right:
                if predicate == "my_is":
                    return (self._default_subject(entry), left, right)
                return (left, predicate, right)
        return None

    def _looks_like_simple_event(self, entry: MemoryEntry) -> bool:
        """Return True for conservative event-like declarative statements."""
        if entry.report.signals.is_preference or entry.report.signals.is_constraint:
            return False
        words = {word.casefold().strip(".,:;!?") for word in entry.text.split()}
        return bool(words & _EVENT_VERBS)

    def _clean_span(self, value: str) -> str:
        """Normalize a short regex capture into a card field."""
        return re.sub(r"\s+", " ", value.strip(" .,:;!?")).strip()


def project_memory_cards(entry: MemoryEntry) -> list[MemoryCard]:
    """Project one memory entry with the default deterministic projector.

    Args:
        entry: Working-memory entry to project.

    Returns:
        Zero or more memory cards.

    Raises:
        ValueError: If projected card fields violate the card schema.
    """
    return MemoryCardProjector().project(entry)


def mark_superseded(card: MemoryCard, *, superseded_by: list[str]) -> MemoryCard:
    """Return a copy of a card marked as superseded.

    Args:
        card: Existing card to update immutably.
        superseded_by: Record or card identifiers that replace this card.

    Returns:
        Copy of ``card`` with superseded validity metadata.

    Raises:
        ValueError: If the replacement validity state violates the card schema.
    """
    validity = MemoryCardValidity(
        status=CARD_STATUS_SUPERSEDED,
        supersedes=list(card.validity.supersedes),
        superseded_by=list(superseded_by),
        invalidates=list(card.validity.invalidates),
        corrects=list(card.validity.corrects),
    )
    return dataclasses.replace(card, validity=validity)
