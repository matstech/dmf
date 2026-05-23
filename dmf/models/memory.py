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

"""Memory data contracts for active context, cards, and retrieval evidence.

A ``MemoryEntry`` is the atomic unit stored in the ``TemporalMemory``
deque.  It groups together everything the orchestrator needs to make
eviction decisions without touching the original source material again:

  - The message text (for placeholder injection and LTM archival).
  - The AnalysisReport (carries Ω, status, and all NLP signals).
  - The embedding vector (keeps geometry in-sync with the deque).
  - The interaction counter at creation time (to compute Δn later).
  - A pre-calculated token count (so budget checks are O(1) reads).
  - A Unix timestamp (for LTM audit trails and ordering).

Design notes
------------
- **Not frozen**: The eviction manager may need to stamp runtime state
  in-place. A mutable dataclass avoids creating copies on every state change.
- **vector is np.ndarray**: Kept on the entry — not reconstructed from
  the AnalysisReport — because the report does not store the raw vector.
  The InteractionMatrix retains its own deque copy for geometry; the
  MemoryEntry copy is the authoritative record for LTM serialisation.
- **token_count is int**: Pre-calculated by the orchestrator via
  ``tiktoken`` at insertion time so ``get_total_tokens()`` is a pure
  ``sum()`` with no encoding overhead in the hot path.
- **timestamp is float**: Unix epoch (``time.time()``) for compatibility
  with standard logging and archival metadata schemas.  Not used for any
  decay calculation — all decay is interaction-count-based.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dmf.models.analysis import AnalysisReport, InteractionProvenance, MemoryLineage
from dmf.models.constants import (
    CARD_STATUS_ACTIVE,
    QUERY_ANSWER_TYPE_UNKNOWN,
    QUERY_CURRENTNESS_UNSPECIFIED,
    QUERY_POLARITY_POSITIVE,
    QUERY_TEMPORAL_NONE,
    UNKNOWN_VALUE,
    VALID_CARD_KINDS,
    VALID_CARD_STATUSES,
    VALID_EVIDENCE_TYPES,
    VALID_QUERY_ANSWER_TYPES,
    VALID_QUERY_CURRENTNESS,
    VALID_QUERY_POLARITIES,
    VALID_QUERY_TEMPORAL_INTENTS,
)
from dmf.models.raw_ltm import RawLTMRecord


@dataclass(frozen=True)
class QueryFrame:
    """Deterministic, serialisable query-understanding input for retrieval.
    
        ``query_embedding`` is optional by design. Query parsing is pure and
        should not call embedding providers or mutate retrieval state; callers
        can inject an already-computed vector when the future retriever needs it.
    
    Args:
        query_text: See the function signature and surrounding type hints.
        query_embedding: See the function signature and surrounding type hints.
        entities: See the function signature and surrounding type hints.
        aliases: See the function signature and surrounding type hints.
        subject_focus: See the function signature and surrounding type hints.
        predicate_focus: See the function signature and surrounding type hints.
        answer_type: See the function signature and surrounding type hints.
        temporal_intent: See the function signature and surrounding type hints.
        historical_vs_current: See the function signature and surrounding type hints.
        polarity: See the function signature and surrounding type hints.
        filters: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        ValueError: Raised by class validation or initialization.
    """

    query_text: str
    query_embedding: list[float] | None = None
    entities: list[str] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    subject_focus: list[str] = field(default_factory=list)
    predicate_focus: list[str] = field(default_factory=list)
    answer_type: str = QUERY_ANSWER_TYPE_UNKNOWN
    temporal_intent: str = QUERY_TEMPORAL_NONE
    historical_vs_current: str = QUERY_CURRENTNESS_UNSPECIFIED
    polarity: str = QUERY_POLARITY_POSITIVE
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate public enum-like fields used by downstream retrieval."""
        if self.answer_type not in VALID_QUERY_ANSWER_TYPES:
            raise ValueError(f"Unsupported query answer type: {self.answer_type!r}")
        if self.temporal_intent not in VALID_QUERY_TEMPORAL_INTENTS:
            raise ValueError(
                f"Unsupported query temporal intent: {self.temporal_intent!r}"
            )
        if self.historical_vs_current not in VALID_QUERY_CURRENTNESS:
            raise ValueError(
                f"Unsupported query currentness focus: {self.historical_vs_current!r}"
            )
        if self.polarity not in VALID_QUERY_POLARITIES:
            raise ValueError(f"Unsupported query polarity: {self.polarity!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serialisable representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "query_text": self.query_text,
            "query_embedding": (
                None if self.query_embedding is None else list(self.query_embedding)
            ),
            "entities": list(self.entities),
            "aliases": {
                key: list(values)
                for key, values in sorted(self.aliases.items(), key=lambda item: item[0])
            },
            "subject_focus": list(self.subject_focus),
            "predicate_focus": list(self.predicate_focus),
            "answer_type": self.answer_type,
            "temporal_intent": self.temporal_intent,
            "historical_vs_current": self.historical_vs_current,
            "polarity": self.polarity,
            "filters": dict(self.filters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QueryFrame:
        """Hydrate from a previously serialised query-frame payload.
        
        Args:
            payload: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        embedding = payload.get("query_embedding")
        return cls(
            query_text=payload["query_text"],
            query_embedding=None if embedding is None else [float(v) for v in embedding],
            entities=list(payload.get("entities", [])),
            aliases={
                str(key): list(values)
                for key, values in payload.get("aliases", {}).items()
            },
            subject_focus=list(payload.get("subject_focus", [])),
            predicate_focus=list(payload.get("predicate_focus", [])),
            answer_type=payload.get("answer_type", QUERY_ANSWER_TYPE_UNKNOWN),
            temporal_intent=payload.get("temporal_intent", QUERY_TEMPORAL_NONE),
            historical_vs_current=payload.get("historical_vs_current", QUERY_CURRENTNESS_UNSPECIFIED),
            polarity=payload.get("polarity", QUERY_POLARITY_POSITIVE),
            filters=dict(payload.get("filters", {})),
        )


@dataclass(frozen=True)
class MemoryCardTimeAnchor:
    """Deterministic temporal anchor for a projected memory card.
    
    Args:
        session_id: See the function signature and surrounding type hints.
        turn_id: See the function signature and surrounding type hints.
        absolute_time: See the function signature and surrounding type hints.
        relative_order: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    session_id: str | None = None
    turn_id: int | None = None
    absolute_time: float | None = None
    relative_order: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryCardTimeAnchor:
        """Hydrate from a serialised payload.
        
        Args:
            payload: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return cls(
            session_id=payload.get("session_id"),
            turn_id=payload.get("turn_id"),
            absolute_time=payload.get("absolute_time"),
            relative_order=payload.get("relative_order"),
        )


@dataclass(frozen=True)
class MemoryCardValidity:
    """Validity state and lineage edges for an auxiliary memory card.
    
    Args:
        status: See the function signature and surrounding type hints.
        supersedes: See the function signature and surrounding type hints.
        superseded_by: See the function signature and surrounding type hints.
        invalidates: See the function signature and surrounding type hints.
        corrects: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        ValueError: Raised by class validation or initialization.
    """

    status: str = CARD_STATUS_ACTIVE
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)
    corrects: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject unknown validity statuses early."""
        if self.status not in VALID_CARD_STATUSES:
            raise ValueError(f"Unsupported memory-card validity status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryCardValidity:
        """Hydrate from a serialised payload.
        
        Args:
            payload: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return cls(
            status=payload.get("status", CARD_STATUS_ACTIVE),
            supersedes=list(payload.get("supersedes", [])),
            superseded_by=list(payload.get("superseded_by", [])),
            invalidates=list(payload.get("invalidates", [])),
            corrects=list(payload.get("corrects", [])),
        )


@dataclass(frozen=True)
class MemoryCardProvenance:
    """Source linkage for an auxiliary memory card.
    
    Args:
        source_record_id: See the function signature and surrounding type hints.
        speaker_role: See the function signature and surrounding type hints.
        source_turn: See the function signature and surrounding type hints.
        interaction: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    source_record_id: str
    speaker_role: str
    source_turn: int | None = None
    interaction: InteractionProvenance = field(default_factory=InteractionProvenance)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "source_record_id": self.source_record_id,
            "speaker_role": self.speaker_role,
            "source_turn": self.source_turn,
            "interaction": dataclasses.asdict(self.interaction),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryCardProvenance:
        """Hydrate from a serialised payload.
        
        Args:
            payload: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return cls(
            source_record_id=payload["source_record_id"],
            speaker_role=payload.get("speaker_role", UNKNOWN_VALUE),
            source_turn=payload.get("source_turn"),
            interaction=InteractionProvenance(**payload.get("interaction", {})),
        )


@dataclass(frozen=True)
class MemoryCard:
    """Deterministic structured projection of one raw memory entry.
    
    Args:
        card_id: See the function signature and surrounding type hints.
        kind: See the function signature and surrounding type hints.
        subject: See the function signature and surrounding type hints.
        predicate: See the function signature and surrounding type hints.
        object: See the function signature and surrounding type hints.
        qualifiers: See the function signature and surrounding type hints.
        time_anchor: See the function signature and surrounding type hints.
        validity: See the function signature and surrounding type hints.
        provenance: See the function signature and surrounding type hints.
        confidence: See the function signature and surrounding type hints.
        surface_forms: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        ValueError: Raised by class validation or initialization.
    """

    card_id: str
    kind: str
    subject: str
    predicate: str
    object: str
    qualifiers: dict[str, Any] = field(default_factory=dict)
    time_anchor: MemoryCardTimeAnchor = field(default_factory=MemoryCardTimeAnchor)
    validity: MemoryCardValidity = field(default_factory=MemoryCardValidity)
    provenance: MemoryCardProvenance = field(
        default_factory=lambda: MemoryCardProvenance(
            source_record_id="",
            speaker_role=UNKNOWN_VALUE,
        )
    )
    confidence: float = 0.0
    surface_forms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the stable public card schema."""
        if self.kind not in VALID_CARD_KINDS:
            raise ValueError(f"Unsupported memory-card kind: {self.kind!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("memory-card confidence must be within [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "card_id": self.card_id,
            "kind": self.kind,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "qualifiers": self.qualifiers,
            "time_anchor": self.time_anchor.to_dict(),
            "validity": self.validity.to_dict(),
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "surface_forms": list(self.surface_forms),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MemoryCard:
        """Hydrate from a serialised payload.
        
        Args:
            payload: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return cls(
            card_id=payload["card_id"],
            kind=payload["kind"],
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            qualifiers=dict(payload.get("qualifiers", {})),
            time_anchor=MemoryCardTimeAnchor.from_dict(payload.get("time_anchor", {})),
            validity=MemoryCardValidity.from_dict(payload.get("validity", {})),
            provenance=MemoryCardProvenance.from_dict(payload["provenance"]),
            confidence=float(payload.get("confidence", 0.0)),
            surface_forms=list(payload.get("surface_forms", [])),
        )


@dataclass(frozen=True)
class RetrievedEvidence:
    """Uniform, serialisable envelope for multi-channel recall candidates.
    
    Args:
        evidence_id: See the function signature and surrounding type hints.
        evidence_type: See the function signature and surrounding type hints.
        source: See the function signature and surrounding type hints.
        semantic_score: See the function signature and surrounding type hints.
        symbolic_score: See the function signature and surrounding type hints.
        temporal_score: See the function signature and surrounding type hints.
        answerability_features: See the function signature and surrounding type hints.
        render_payload: See the function signature and surrounding type hints.
        provenance: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        ValueError: Raised by class validation or initialization.
    """

    evidence_id: str
    evidence_type: str
    source: str
    semantic_score: float | None = None
    symbolic_score: float | None = None
    temporal_score: float | None = None
    answerability_features: dict[str, Any] = field(default_factory=dict)
    render_payload: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the stable evidence schema."""
        if self.evidence_type not in VALID_EVIDENCE_TYPES:
            raise ValueError(f"Unsupported evidence type: {self.evidence_type!r}")
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serialisable representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "source": self.source,
            "semantic_score": self.semantic_score,
            "symbolic_score": self.symbolic_score,
            "temporal_score": self.temporal_score,
            "answerability_features": dict(self.answerability_features),
            "render_payload": dict(self.render_payload),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrievedEvidence:
        """Hydrate from a previously serialised evidence payload.
        
        Args:
            payload: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return cls(
            evidence_id=payload["evidence_id"],
            evidence_type=payload["evidence_type"],
            source=payload["source"],
            semantic_score=_optional_float(payload.get("semantic_score")),
            symbolic_score=_optional_float(payload.get("symbolic_score")),
            temporal_score=_optional_float(payload.get("temporal_score")),
            answerability_features=dict(payload.get("answerability_features", {})),
            render_payload=dict(payload.get("render_payload", {})),
            provenance=dict(payload.get("provenance", {})),
        )


def _optional_float(value: Any) -> float | None:
    """Coerce optional score fields while preserving missing values."""
    return None if value is None else float(value)



@dataclass
class MemoryEntry:
    """Single slot in the Working Memory deque.
    
        Created once per interaction by ``TemporalMemory.add_interaction`` and
        retained until evicted by pressure pruning or passive hard-kill.
    
        In the current model this object is also the runtime ``source_record``:
        it preserves the full interaction text plus the metadata needed for
        raw archival and recall-time reinterpretation.
    
        Attributes
        ----------
        interaction_id : int
            Monotonically increasing counter assigned at insertion time.
            Δn for any entry is computed as ``current_counter − entry.interaction_id``.
            Never reused within a session.
        text : str
            Original message text as received by the pipeline.  Stored verbatim
            so eviction can archive the full text to LTM without reprocessing.
        report : AnalysisReport
            Feature envelope produced by the analysis and scoring pipeline. The
            ``survival_score`` field carries the static Ω; ``status`` carries
            the categorical tier. Both are stamped by ``ScoringEngine`` before
            the entry is created.
        vector : np.ndarray
            L2-normalised embedding vector produced by ``EmbeddingEngine``.
            Shape: ``(vector_dim,)`` — typically 384 for bge-small-en-v1.5.
            Stored here for LTM serialisation; the ``InteractionMatrix`` keeps
            its own deque copy for live centroid tracking.
        token_count : int
            Number of tokens in ``text`` as counted by the ``cl100k_base``
            tiktoken encoder.  Pre-calculated at insertion time so
            ``get_total_tokens()`` is a pure ``sum()`` in the hot path.
        timestamp : float
            Unix epoch timestamp (``time.time()``) recorded at insertion.
            Used for LTM audit ordering and structured log emission.
            **Not** used for any decay calculation — all decay is Δn-based.
        provenance : InteractionProvenance
            Structured provenance metadata mirrored from ``report.provenance``.
    
    Args:
        interaction_id: See the function signature and surrounding type hints.
        text: See the function signature and surrounding type hints.
        report: See the function signature and surrounding type hints.
        vector: See the function signature and surrounding type hints.
        token_count: See the function signature and surrounding type hints.
        timestamp: See the function signature and surrounding type hints.
        lineage: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    interaction_id: int
    text: str
    report: AnalysisReport
    vector: np.ndarray
    token_count: int
    timestamp: float

    lineage: MemoryLineage = field(default_factory=MemoryLineage, compare=False)


    @property
    def omega(self) -> float:
        """Static Survival Score Ω from the AnalysisReport.
        
                Returns 0.0 when the report has not yet been scored (survival_score
                is None), providing a safe default for decay calculations.
        
        Raises:
            None.
        """
        return self.report.survival_score or 0.0

    @property
    def status(self):
        """SurvivalStatus tier from the AnalysisReport (may be None).
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return self.report.status

    @property
    def provenance(self) -> InteractionProvenance:
        """Structured provenance metadata mirrored from the AnalysisReport.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return self.report.provenance

    @property
    def record_id(self) -> str:
        """Stable source-record identifier derived from ``interaction_id``.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return f"record:{self.interaction_id}"


    def to_source_record_dict(self) -> dict[str, Any]:
        """Return the canonical source-record representation.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "record_id": self.record_id,
            "interaction_id": self.interaction_id,
            "role": self.provenance.role,
            "text": self.text,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "lineage": self.lineage.to_dict(),
            "provenance": dataclasses.asdict(self.provenance),
            "signals": dataclasses.asdict(self.report.signals),
            "topic_identity": self.report.topic_identity,
            "topic_value": self.report.topic_value,
            "report": self.report.to_dict(),
        }

    def to_raw_ltm_record(self) -> RawLTMRecord:
        """Return the stable raw-LTM archival representation.
        
                This projection intentionally strips all derived runtime signals
                and scores. The raw-LTM store keeps only source-level data so that
                future recall can recompute interpretation with the then-current
                NLP engine.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return RawLTMRecord(
            record_id=self.record_id,
            interaction_id=self.interaction_id,
            role=self.provenance.role or UNKNOWN_VALUE,
            text=self.text,
            created_at=self.timestamp,
            provenance=self.provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.
        
                The ``np.ndarray`` embedding vector is converted to a plain Python
                ``list[float]`` so the result can be passed directly to
                ``json.dumps`` without a custom encoder.  The ``AnalysisReport``
                is serialised via its own ``to_dict()`` method, which handles
                ``SurvivalStatus`` enum conversion.
        
                Keys
                ----
                interaction_id : int
                text           : str
                token_count    : int
                timestamp      : float   — Unix epoch
                omega          : float   — static Survival Score Ω
                status         : str | None — SurvivalStatus value string
                vector         : list[float] — L2-normalised embedding
                report         : dict   — full AnalysisReport via report.to_dict()
                provenance     : dict   — top-level mirror of ``report.provenance``
                                 for easier inspection in LTM artifacts.
                source_record  : dict   — canonical source-record projection kept
                                 separate from any downstream archival view.
        
                Returns
                -------
                dict[str, Any]
                    Flat/nested dict safe for ``json.dumps``.
        
        Raises:
            None.
        """
        return {
            "record_id": self.record_id,
            "interaction_id": self.interaction_id,
            "text": self.text,
            "token_count": self.token_count,
            "timestamp": self.timestamp,
            "omega": self.omega,
            "status": self.status.value if self.status is not None else None,
            "vector": self.vector.tolist(),
            "report": self.report.to_dict(),
            "provenance": dataclasses.asdict(self.provenance),
            "lineage": self.lineage.to_dict(),
            "source_record": self.to_source_record_dict(),
        }
