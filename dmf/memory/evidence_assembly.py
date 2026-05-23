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

"""Final cutoff, evidence expansion, and prompt rendering for retrieval evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from dmf.memory.constants import (
    CONTEXT_METADATA_SEPARATOR,
    UNIX_TIMESTAMP_MAX,
    UNIX_TIMESTAMP_MIN,
    UTC_RENDER_FORMAT,
)
from dmf.models.constants import (
    CARD_KIND_CURRENT_STATE,
    CARD_STATUS_ACTIVE,
    EVIDENCE_TYPE_CARD_BACKED_TURN,
    EVIDENCE_TYPE_RAW_TURN,
)
from dmf.models.memory import MemoryCard, QueryFrame, RetrievedEvidence
from dmf.models.raw_ltm import RawLTMRecord
from dmf.utils.config_loader import RetrievalSettings

_SUPPORT_RELATION_LABELS: dict[str, str] = {
    "source": "source",
    "historical_link": "history",
    "supersession_link": "newer",
    "neighbor_before": "prev",
    "neighbor_after": "next",
    "retrieved": "retrieved",
}


class RawEvidenceSource(Protocol):
    """Source that can expose canonical raw records for evidence expansion.

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
class EvidenceAssemblyConfig:
    """Opt-in final assembly controls for the structured retrieval stack.

    Args:
        final_recall_limit: Maximum number of reranked winners to keep.
        max_support_turns_per_card: Maximum raw support turns attached per card.
        include_superseded_when_historical: Whether historical queries may show
            superseded support.
        include_neighbor_turns: Whether adjacent raw turns are included as local
            context.

    Returns:
        Immutable evidence assembly configuration.

    Raises:
        ValueError: If a limit is outside the supported range.
    """

    final_recall_limit: int = 5
    max_support_turns_per_card: int = 3
    include_superseded_when_historical: bool = True
    include_neighbor_turns: bool = False

    def __post_init__(self) -> None:
        """Reject invalid cutoff and support-turn budgets.

        Returns:
            None.

        Raises:
            ValueError: If a configured limit is invalid.
        """
        if self.final_recall_limit < 0:
            raise ValueError("final_recall_limit must be non-negative")
        if self.max_support_turns_per_card < 1:
            raise ValueError("max_support_turns_per_card must be at least 1")

    @classmethod
    def from_retrieval_settings(
        cls,
        settings: RetrievalSettings,
    ) -> EvidenceAssemblyConfig:
        """Build assembly config from parsed DMF retrieval settings.

        Args:
            settings: Parsed retrieval settings.

        Returns:
            Evidence assembly config with matching values.

        Raises:
            ValueError: If a setting violates local validation.
        """
        return cls(
            final_recall_limit=settings.final_recall_limit,
            max_support_turns_per_card=settings.max_support_turns_per_card,
            include_superseded_when_historical=(
                settings.include_superseded_when_historical
            ),
            include_neighbor_turns=settings.include_neighbor_turns,
        )


def apply_final_cutoff(
    candidates: Iterable[RetrievedEvidence],
    *,
    final_recall_limit: int,
) -> list[RetrievedEvidence]:
    """Apply the explicit post-rerank final recall cutoff.

    Args:
        candidates: Reranked evidence stream.
        final_recall_limit: Maximum number of items to keep.

    Returns:
        Evidence list truncated to the configured limit.

    Raises:
        ValueError: If ``final_recall_limit`` is negative.
    """
    if final_recall_limit < 0:
        raise ValueError("final_recall_limit must be non-negative")
    if final_recall_limit == 0:
        return []
    return list(candidates)[:final_recall_limit]


def assemble_final_evidence(
    reranked_candidates: Iterable[RetrievedEvidence],
    *,
    raw_records: Iterable[RawLTMRecord] | RawEvidenceSource = (),
    query: QueryFrame | None = None,
    config: EvidenceAssemblyConfig | None = None,
) -> list[RetrievedEvidence]:
    """Cut off reranked evidence and expand winning cards into raw support turns.

    Args:
        reranked_candidates: Evidence ordered by final answerability score.
        raw_records: Raw-record source used for support expansion.
        query: Optional parsed query used to decide historical support behavior.
        config: Optional assembly configuration.

    Returns:
        Final evidence with support metadata attached where available.

    Raises:
        ValueError: If config construction receives invalid limits.
        Backend-specific exceptions may surface from ``raw_records.read_all``.
    """
    resolved = config or EvidenceAssemblyConfig()
    winners = apply_final_cutoff(
        reranked_candidates,
        final_recall_limit=resolved.final_recall_limit,
    )
    return expand_card_evidence(
        winners,
        raw_records=raw_records,
        query=query,
        config=resolved,
    )


def expand_card_evidence(
    candidates: Iterable[RetrievedEvidence],
    *,
    raw_records: Iterable[RawLTMRecord] | RawEvidenceSource = (),
    query: QueryFrame | None = None,
    config: EvidenceAssemblyConfig | None = None,
) -> list[RetrievedEvidence]:
    """Expand final winners with auditable raw support when configured.

    Args:
        candidates: Final evidence winners.
        raw_records: Raw-record source used for support expansion.
        query: Optional parsed query used to decide historical support behavior.
        config: Optional assembly configuration.

    Returns:
        Evidence with card/raw support payloads attached.

    Raises:
        Backend-specific exceptions may surface from ``raw_records.read_all``.
    """
    resolved = config or EvidenceAssemblyConfig()
    index = _RawRecordIndex(_read_records(raw_records))
    expanded: list[RetrievedEvidence] = []
    for candidate in candidates:
        if candidate.evidence_type == EVIDENCE_TYPE_RAW_TURN:
            source = _raw_record_from_candidate(candidate, index=index)
            if source is None:
                expanded.append(candidate)
                continue
            support = _support_turns_for_raw(source, index=index, config=resolved)
            expanded.append(_with_raw_support(candidate, source=source, support=support))
            continue
        if candidate.evidence_type != "card":
            expanded.append(candidate)
            continue
        card = _card_from_candidate(candidate)
        if card is None:
            expanded.append(candidate)
            continue
        support = _support_turns_for_card(
            card,
            index=index,
            query=query,
            config=resolved,
        )
        expanded.append(_with_card_support(candidate, card=card, support=support))
    return expanded


def render_evidence_context(
    evidence: Iterable[RetrievedEvidence],
    *,
    include_structured_evidence: bool = True,
) -> str:
    """Render final evidence as prompt-ready recalled context blocks.

    Args:
        evidence: Final evidence items to render.
        include_structured_evidence: Whether to include compact structured rows
            before raw support.

    Returns:
        Prompt-ready context string. Empty evidence renders as an empty string.

    Raises:
        None.
    """
    items = list(evidence)
    parts: list[str] = []
    if include_structured_evidence:
        structured_lines = _structured_lines(items)
        if structured_lines:
            parts.append("=== STRUCTURED EVIDENCE ===")
            parts.extend(structured_lines)
            parts.append("")

    raw_lines = _raw_support_lines(
        items,
        include_retrieved_records=not include_structured_evidence,
    )
    if raw_lines:
        parts.append("=== RAW SUPPORTING EVIDENCE ===")
        parts.extend(raw_lines)

    return "\n".join(parts).strip()


class _RawRecordIndex:
    """Small in-memory lookup over raw records for deterministic expansion."""

    def __init__(self, records: list[RawLTMRecord]) -> None:
        self.by_id = {record.record_id: record for record in records}
        self.by_turn = {record.interaction_id: record for record in records}

    def get(self, record_id: str) -> RawLTMRecord | None:
        """Return a raw record by stable id.
        
        Args:
            record_id: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return self.by_id.get(record_id)

    def neighbors(self, record: RawLTMRecord) -> list[tuple[str, RawLTMRecord]]:
        """Return immediate local neighbors when present.
        
        Args:
            record: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        neighbors: list[tuple[str, RawLTMRecord]] = []
        before = self.by_turn.get(record.interaction_id - 1)
        after = self.by_turn.get(record.interaction_id + 1)
        if before is not None:
            neighbors.append(("neighbor_before", before))
        if after is not None:
            neighbors.append(("neighbor_after", after))
        return neighbors


def _read_records(
    raw_records: Iterable[RawLTMRecord] | RawEvidenceSource,
) -> list[RawLTMRecord]:
    if hasattr(raw_records, "read_all"):
        return list(raw_records.read_all())
    return list(raw_records)


def _card_from_candidate(candidate: RetrievedEvidence) -> MemoryCard | None:
    payload = candidate.render_payload.get("card")
    if not isinstance(payload, dict):
        return None
    return MemoryCard.from_dict(payload)


def _support_turns_for_card(
    card: MemoryCard,
    *,
    index: _RawRecordIndex,
    query: QueryFrame | None,
    config: EvidenceAssemblyConfig,
) -> list[dict[str, Any]]:
    support: list[dict[str, Any]] = []
    source = index.get(card.provenance.source_record_id)
    if source is not None:
        _append_support(support, relation="source", record=source)

    if _include_historical_edges(query, config):
        for record_id in [*card.validity.supersedes, *card.validity.corrects]:
            record = index.get(record_id)
            if record is not None:
                _append_support(support, relation="historical_link", record=record)

    for record_id in [*card.validity.superseded_by, *card.validity.invalidates]:
        record = index.get(record_id)
        if record is not None:
            _append_support(support, relation="supersession_link", record=record)

    if config.include_neighbor_turns and source is not None:
        for relation, record in index.neighbors(source):
            _append_support(support, relation=relation, record=record)

    return support[: config.max_support_turns_per_card]


def _raw_record_from_candidate(
    candidate: RetrievedEvidence,
    *,
    index: _RawRecordIndex,
) -> RawLTMRecord | None:
    record_payload = candidate.render_payload.get("record")
    record_id = ""
    if isinstance(record_payload, dict):
        record_id = str(record_payload.get("record_id", ""))
    record_id = record_id or str(candidate.provenance.get("source_record_id", ""))

    indexed = index.get(record_id)
    if indexed is not None:
        return indexed

    if not isinstance(record_payload, dict):
        return None
    try:
        return RawLTMRecord.from_dict(record_payload)
    except (KeyError, TypeError, ValueError):
        return None


def _support_turns_for_raw(
    source: RawLTMRecord,
    *,
    index: _RawRecordIndex,
    config: EvidenceAssemblyConfig,
) -> list[dict[str, Any]]:
    if not config.include_neighbor_turns:
        return []

    support: list[dict[str, Any]] = []
    for relation, record in index.neighbors(source):
        _append_support(support, relation=relation, record=record)
    return support[: config.max_support_turns_per_card]


def _append_support(
    support: list[dict[str, Any]],
    *,
    relation: str,
    record: RawLTMRecord,
) -> None:
    if any(item["record"]["record_id"] == record.record_id for item in support):
        return
    support.append({"relation": relation, "record": record.to_dict()})


def _include_historical_edges(
    query: QueryFrame | None,
    config: EvidenceAssemblyConfig,
) -> bool:
    if not config.include_superseded_when_historical:
        return False
    if query is None:
        return True
    return query.historical_vs_current in {"historical", "mixed"}


def _with_card_support(
    candidate: RetrievedEvidence,
    *,
    card: MemoryCard,
    support: list[dict[str, Any]],
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=candidate.evidence_id,
        evidence_type=EVIDENCE_TYPE_CARD_BACKED_TURN,
        source=candidate.source,
        semantic_score=candidate.semantic_score,
        symbolic_score=candidate.symbolic_score,
        temporal_score=candidate.temporal_score,
        answerability_features=dict(candidate.answerability_features),
        render_payload={
            **candidate.render_payload,
            "card": card.to_dict(),
            "support_records": support,
        },
        provenance={
            **candidate.provenance,
            "card_id": card.card_id,
            "source_record_id": card.provenance.source_record_id,
            "support_record_ids": [
                item["record"]["record_id"]
                for item in support
                if isinstance(item.get("record"), dict)
            ],
        },
    )


def _with_raw_support(
    candidate: RetrievedEvidence,
    *,
    source: RawLTMRecord,
    support: list[dict[str, Any]],
) -> RetrievedEvidence:
    if not support:
        return candidate

    support_record_ids = [
        item["record"]["record_id"]
        for item in support
        if isinstance(item.get("record"), dict)
    ]
    return RetrievedEvidence(
        evidence_id=candidate.evidence_id,
        evidence_type=candidate.evidence_type,
        source=candidate.source,
        semantic_score=candidate.semantic_score,
        symbolic_score=candidate.symbolic_score,
        temporal_score=candidate.temporal_score,
        answerability_features=dict(candidate.answerability_features),
        render_payload={
            **candidate.render_payload,
            "support_records": support,
        },
        provenance={
            **candidate.provenance,
            "source_record_id": source.record_id,
            "support_record_ids": [source.record_id, *support_record_ids],
        },
    )


def _structured_lines(items: list[RetrievedEvidence]) -> list[str]:
    lines: list[str] = []
    rendered_count = 0
    for item in items:
        block = _structured_item_block(item, label=f"R{rendered_count + 1}")
        if not block:
            continue
        _append_block(lines, block)
        rendered_count += 1
    return lines


def _raw_support_lines(
    items: list[RetrievedEvidence],
    *,
    include_retrieved_records: bool,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    support_index = 0
    for item in items:
        support = item.render_payload.get("support_records")
        if isinstance(support, list):
            for support_item in support:
                if isinstance(support_item, dict):
                    record = support_item.get("record")
                    relation = str(support_item.get("relation", "support"))
                    appended = _append_raw_line(
                        lines,
                        seen,
                        label=f"S{support_index + 1}",
                        record=record,
                        relation=relation,
                    )
                    if appended:
                        support_index += 1
            continue
        if not include_retrieved_records:
            continue
        record = item.render_payload.get("record")
        appended = _append_raw_line(
            lines,
            seen,
            label=f"S{support_index + 1}",
            record=record,
            relation="retrieved",
        )
        if appended:
            support_index += 1
    return lines


def _append_raw_line(
    lines: list[str],
    seen: set[str],
    *,
    label: str,
    record: Any,
    relation: str,
) -> bool:
    if not isinstance(record, dict):
        return False
    record_id = str(record.get("record_id", ""))
    if not record_id or record_id in seen:
        return False
    text = str(record.get("text", "")).strip()
    if not text:
        return False
    seen.add(record_id)
    metadata = _support_metadata_parts(record, relation=relation)
    _append_block(lines, _render_prompt_block(label=label, metadata=metadata, text=text))
    return True


def _structured_item_block(
    item: RetrievedEvidence,
    *,
    label: str,
) -> list[str]:
    card_payload = item.render_payload.get("card")
    if isinstance(card_payload, dict):
        text = _card_prompt_text(card_payload)
        if not text:
            return []
        metadata = _card_metadata_parts(item, card_payload)
        return _render_prompt_block(label=label, metadata=metadata, text=text)

    record = item.render_payload.get("record")
    if not isinstance(record, dict):
        return []
    text = str(record.get("text", "")).strip()
    if not text:
        return []
    metadata = _record_metadata_parts(record, prefer_time_over_turn=True)
    return _render_prompt_block(label=label, metadata=metadata, text=text)


def _card_prompt_text(card_payload: dict[str, Any]) -> str:
    surface_forms = card_payload.get("surface_forms")
    if isinstance(surface_forms, list):
        for surface in surface_forms:
            text = str(surface).strip()
            if text:
                return text
    subject = str(card_payload.get("subject", "")).strip()
    predicate = str(card_payload.get("predicate", "")).strip()
    obj = str(card_payload.get("object", "")).strip()
    return " ".join(piece for piece in [subject, predicate, obj] if piece)


def _card_metadata_parts(
    item: RetrievedEvidence,
    card_payload: dict[str, Any],
) -> list[str]:
    support_record = _primary_support_record(item)
    role = _first_non_empty(
        card_payload.get("provenance", {}).get("speaker_role"),
        support_record.get("role") if support_record else None,
    )
    absolute_time = _first_non_empty(
        card_payload.get("time_anchor", {}).get("absolute_time"),
        support_record.get("created_at") if support_record else None,
    )
    turn_value = _first_non_empty(
        card_payload.get("provenance", {}).get("source_turn"),
        card_payload.get("time_anchor", {}).get("turn_id"),
        support_record.get("interaction_id") if support_record else None,
    )
    state = _state_from_card_payload(card_payload)
    return _metadata_parts(
        role=role,
        time_value=absolute_time,
        turn_value=turn_value,
        state=state,
        prefer_time_over_turn=absolute_time is not None,
    )


def _primary_support_record(item: RetrievedEvidence) -> dict[str, Any] | None:
    support = item.render_payload.get("support_records")
    if not isinstance(support, list):
        return None
    preferred: dict[str, Any] | None = None
    for support_item in support:
        if not isinstance(support_item, dict):
            continue
        record = support_item.get("record")
        if not isinstance(record, dict):
            continue
        if support_item.get("relation") == "source":
            return record
        if preferred is None:
            preferred = record
    return preferred


def _record_metadata_parts(
    record: dict[str, Any],
    *,
    state: str | None = None,
    extra_parts: list[str] | None = None,
    prefer_time_over_turn: bool = False,
) -> list[str]:
    role = _first_non_empty(record.get("role"), record.get("provenance", {}).get("role"))
    turn_value = _first_non_empty(
        record.get("provenance", {}).get("source_turn"),
        record.get("interaction_id"),
    )
    metadata = _metadata_parts(
        role=role,
        time_value=record.get("created_at"),
        turn_value=turn_value,
        state=state,
        prefer_time_over_turn=prefer_time_over_turn,
    )
    if extra_parts:
        metadata.extend(part for part in extra_parts if part)
    return metadata


def _support_metadata_parts(
    record: dict[str, Any],
    *,
    relation: str,
) -> list[str]:
    metadata = [_SUPPORT_RELATION_LABELS.get(relation, relation.replace("_", " "))]
    locator = _locator_part(
        time_value=record.get("created_at"),
        turn_value=_first_non_empty(
            record.get("provenance", {}).get("source_turn"),
            record.get("interaction_id"),
        ),
        prefer_time_over_turn=False,
    )
    if relation == "retrieved":
        role = _string_or_none(
            _first_non_empty(record.get("role"), record.get("provenance", {}).get("role"))
        )
        if role:
            metadata.append(role)
    if locator is not None:
        metadata.append(locator)
    return metadata


def _state_from_card_payload(card_payload: dict[str, Any]) -> str | None:
    kind = str(card_payload.get("kind", "")).strip()
    validity = str(card_payload.get("validity", {}).get("status", "")).strip()
    if kind == CARD_KIND_CURRENT_STATE and validity in {"", CARD_STATUS_ACTIVE}:
        return "current"
    return None


def _metadata_parts(
    *,
    role: object = None,
    time_value: object = None,
    turn_value: object = None,
    state: str | None = None,
    prefer_time_over_turn: bool = False,
) -> list[str]:
    metadata: list[str] = []
    role_text = _string_or_none(role)
    if role_text:
        metadata.append(role_text)
    locator = _locator_part(
        time_value=time_value,
        turn_value=turn_value,
        prefer_time_over_turn=prefer_time_over_turn,
    )
    if locator is not None:
        metadata.append(locator)
    if state:
        metadata.append(state)
    return metadata


def _locator_part(
    *,
    time_value: object = None,
    turn_value: object = None,
    prefer_time_over_turn: bool,
) -> str | None:
    formatted_turn = _format_turn_value(turn_value)
    formatted_time = _format_time_value(time_value)
    if prefer_time_over_turn and formatted_time is not None:
        return f"time={formatted_time}"
    if formatted_turn is not None:
        return f"turn={formatted_turn}"
    if formatted_time is not None:
        return f"time={formatted_time}"
    return None


def _render_prompt_block(
    *,
    label: str,
    metadata: list[str],
    text: str,
) -> list[str]:
    header = f"[{label}]"
    if metadata:
        header = f"{header} {CONTEXT_METADATA_SEPARATOR.join(metadata)}"
    lines = [header]
    lines.extend(text.strip().splitlines())
    return lines


def _append_block(lines: list[str], block: list[str]) -> None:
    if not block:
        return
    if lines:
        lines.append("")
    lines.extend(block)


def _format_time_value(value: object) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if UNIX_TIMESTAMP_MIN <= numeric <= UNIX_TIMESTAMP_MAX:
            return datetime.fromtimestamp(numeric, UTC).strftime(UTC_RENDER_FORMAT)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.3f}".rstrip("0").rstrip(".")
    text = _string_or_none(value)
    return text


def _format_turn_value(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return _string_or_none(value)


def _first_non_empty(*values: object) -> object | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
