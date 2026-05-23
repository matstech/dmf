"""Raw long-term-memory contracts.

The stored unit is intentionally minimal: it keeps only stable interaction
data that should remain valid even when the NLP engine evolves.

Conversational interpretation such as signals, topic identity, topic value,
and any score derived from a specific runtime context must be recomputed at
recall time and therefore does not belong in this archival model.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from dmf.models.analysis import AnalysisReport, InteractionProvenance


@dataclass(frozen=True)
class RawLTMRecord:
    """Stable raw interaction record archived in long-term memory.

    Args:
        record_id: Stable source-record identifier.
        interaction_id: Runtime interaction sequence number.
        role: Speaker role attached to the original interaction.
        text: Original interaction text.
        created_at: Unix timestamp captured when the memory entry was created.
        provenance: Structured source metadata.

    Returns:
        A frozen dataclass containing only source-level archival data.

    Raises:
        None.

    Warning:
        Derived interpretation such as signals, topic metadata, pruning scores,
        or rank hints is deliberately excluded so recall can re-run current NLP
        rules against stable raw text.
    """

    record_id: str
    interaction_id: int
    role: str
    text: str
    created_at: float
    provenance: InteractionProvenance

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the raw record.

        Returns:
            Dictionary containing the stable raw record fields.

        Raises:
            None.
        """
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """Return the raw record as a JSON string.

        Returns:
            Pretty-printed JSON representation.

        Raises:
            TypeError: If a future field is not JSON serialisable.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RawLTMRecord:
        """Hydrate a raw record from a previously serialised dict.

        Args:
            payload: Dictionary produced by ``to_dict``.

        Returns:
            Reconstructed raw LTM record.

        Raises:
            KeyError: If a required payload key is missing.
            TypeError: If ``provenance`` cannot hydrate ``InteractionProvenance``.
        """
        return cls(
            record_id=payload["record_id"],
            interaction_id=payload["interaction_id"],
            role=payload["role"],
            text=payload["text"],
            created_at=payload["created_at"],
            provenance=InteractionProvenance(**payload["provenance"]),
        )


@dataclass(frozen=True)
class RawRecallHit:
    """Direct result of raw semantic search before conversational re-analysis.

    Args:
        record: Archived raw record returned by the backend.
        similarity_score: Optional similarity score where larger means better.
        distance: Optional backend distance where smaller means better.
        rank_hint: Optional backend rank before deterministic reranking.
        source: Retrieval channel label.

    Returns:
        A frozen dataclass describing a backend raw-search hit.

    Raises:
        None.
    """

    record: RawLTMRecord
    similarity_score: float | None = None
    distance: float | None = None
    rank_hint: int | None = None
    source: str = "ltm_raw"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the raw hit.

        Returns:
            Dictionary containing the raw record and retrieval metadata.

        Raises:
            None.
        """
        return {
            "record": self.record.to_dict(),
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "rank_hint": self.rank_hint,
            "source": self.source,
        }


@dataclass
class ContextualizedRecallCandidate:
    """Raw recall hit after deterministic recall-time NLP and ranking.

    Args:
        record: Archived raw record that produced the candidate.
        report: Current ``AnalysisReport`` computed at recall time.
        similarity_score: Optional backend similarity score.
        distance: Optional backend distance.
        rank_hint: Optional backend rank before deterministic reranking.
        recall_score: Optional deterministic reranking score.
        suppressed: Whether the candidate was excluded from rendered context.
        suppression_reason: Machine-readable explanation when suppressed.
        source: Retrieval channel label.

    Returns:
        Mutable candidate object enriched by recall-time analysis.

    Raises:
        None.

    Warning:
        Conversational interpretation is kept in ``report`` to avoid duplicating
        topic or signal fields outside the canonical ``AnalysisReport`` surface.
    """

    record: RawLTMRecord
    report: AnalysisReport
    similarity_score: float | None = None
    distance: float | None = None
    rank_hint: int | None = None
    recall_score: float | None = None
    suppressed: bool = False
    suppression_reason: str | None = None
    source: str = "ltm_raw"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the candidate.

        Returns:
            Dictionary containing raw record, recall-time report, and ranking
            diagnostics.

        Raises:
            None.
        """
        return {
            "record": self.record.to_dict(),
            "report": self.report.to_dict(),
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "rank_hint": self.rank_hint,
            "recall_score": self.recall_score,
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "source": self.source,
        }
