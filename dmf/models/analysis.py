"""Core domain data structures for the Deterministic Memory Framework.

AnalysisReport is the unified feature envelope produced by the pipeline for
every interaction processed. It carries:

  - Three NLP signals: Information Density (ID), Sentiment (|S|), and Entity
    Protection (E).
  - One geometric signal: Semantic Divergence (D) — how far the
    interaction's embedding departs from the recent context centroid.

All four signals are consumed by the scoring engine.

Every instance must support full JSON serialization so it can be:
  - persisted as metadata alongside archived interactions
  - emitted as a structured log event
  - inspected offline for benchmarking
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dmf.models.status import SurvivalStatus



@dataclass
class MemoryLineage:
    """Structured relationships between records.
    
    Args:
        supersedes: See the function signature and surrounding type hints.
        conflicts_with: See the function signature and surrounding type hints.
        corrects: See the function signature and surrounding type hints.
        invalidates: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    supersedes: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    corrects: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        """Return a JSON-serialisable representation of the lineage.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return dataclasses.asdict(self)

    def has_relationships(self) -> bool:
        """Return True when at least one lineage edge is present.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return any(self.to_dict().values())



@dataclass
class InteractionProvenance:
    """Structured provenance metadata for one interaction.
    
    Args:
        role: See the function signature and surrounding type hints.
        source_turn: See the function signature and surrounding type hints.
        is_user_correction: See the function signature and surrounding type hints.
        is_preference_update: See the function signature and surrounding type hints.
        is_constraint: See the function signature and surrounding type hints.
        derived_from_model: See the function signature and surrounding type hints.
        corrected_by_user: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    role: str | None = None
    source_turn: int | None = None
    is_user_correction: bool = False
    is_preference_update: bool = False
    is_constraint: bool = False
    derived_from_model: bool = False
    corrected_by_user: bool = False



@dataclass
class InteractionSignals:
    """Structured conversational and pragmatic signals for one interaction.
    
        These signals are intentionally compositional rather than taxonomic: the
        same interaction can express preference, correction, temporal status, and
        negation simultaneously. Boolean fields capture explicit pragmatic cues
        while continuous fields reserve space for later scoring/reranking work.
    
    Args:
        is_current_state: See the function signature and surrounding type hints.
        is_past_state: See the function signature and surrounding type hints.
        is_preference: See the function signature and surrounding type hints.
        is_constraint: See the function signature and surrounding type hints.
        is_correction: See the function signature and surrounding type hints.
        has_negation: See the function signature and surrounding type hints.
        has_replacement: See the function signature and surrounding type hints.
        operational_weight: See the function signature and surrounding type hints.
        personal_relevance: See the function signature and surrounding type hints.
        quantitative_relevance: See the function signature and surrounding type hints.
        task_relevance: See the function signature and surrounding type hints.
        temporal_markers: See the function signature and surrounding type hints.
        cue_phrases: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    is_current_state: bool = False
    is_past_state: bool = False
    is_preference: bool = False
    is_constraint: bool = False
    is_correction: bool = False
    has_negation: bool = False
    has_replacement: bool = False
    operational_weight: float = 0.0
    personal_relevance: float = 0.0
    quantitative_relevance: float = 0.0
    task_relevance: float = 0.0
    temporal_markers: list[str] = field(default_factory=list)
    cue_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the signals.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return dataclasses.asdict(self)



@dataclass
class AnalysisReport:
    """NLP feature envelope for a single conversation interaction.
    
        Produced by the NLP Engine and consumed by the Scoring Engine. All
        fields are mandatory except `raw_metadata`, which defaults to an empty
        dict to avoid mutable default argument issues (PEP 557).
    
        Attributes
        ----------
        info_density : float
            Information Density ($ID$) — ratio of semantic lemmas
            (NOUN, VERB, ADJ, PROPN) to total token count. Range: [0.0, 1.0].
        sentiment_abs : float
            Absolute VADER compound sentiment score ($|S|$). Range: [0.0, 1.0].
        entity_count : int
            Count of named entities ($E$) recognised by spaCy's standard NER
            (PERSON, ORG, GPE, …). Non-negative integer.
        is_system_prompt : bool
            True when this report describes the system prompt interaction.
            Drives the `analyze_system_prompt` branch in NLPConfig.
        latency_ms : float
            Wall-clock time taken by the NLP pipeline to produce this report,
            in milliseconds. Populated from ExecutionLatencyTimer.
        semantic_divergence : float
            Geometric divergence ($D$) of this interaction's embedding from the
            sliding-window centroid: D = 1 − cosine_similarity(vector, centroid).
            Range: [0.0, 2.0]. 0.0 = perfectly aligned with context; 2.0 =
            diametrically opposed. Defaults to 0.0 for the first interaction
            (no prior context) and for skipped system-prompt interactions.
            Populated by InteractionPipeline; NLPEngine alone leaves
            it at the default.
        survival_score : float | None
            Continuous Survival Score Ω ∈ (0, 1) computed by ScoringEngine
            None until ``ScoringEngine.calculate_score`` has been called on
            this report. Stamped in-place by the engine immediately after
            rounding so callers always have the authoritative value on the
            report itself.
        status : SurvivalStatus | None
            Categorical tier derived from Ω after scoring:
            HEALTHY (Ω > 0.6), UNSTABLE (0.3 < Ω ≤ 0.6), CRITICAL (Ω ≤ 0.3).
            None until ``ScoringEngine.calculate_score`` stamps it.  None is
            the correct sentinel for "not yet scored" — defaulting to CRITICAL
            would conflate unscored reports with genuinely low-value ones.
        raw_metadata : dict[str, Any]
            Audit-level detail: POS tag counts, raw token list, intermediate
            scores, and any other data needed for post-execution inspection.
            Defaults to an empty dict when not supplied.
        provenance : InteractionProvenance
            Structured provenance metadata kept separate from raw NLP audit
            detail. Defaults to an empty ``InteractionProvenance``.
        signals : InteractionSignals
            Structured conversational/pragmatic cues kept separate from the raw
            audit trail so downstream policy can consume a stable schema.
        topic_identity : str | None
            Stable topic field extracted from the interaction when enough
            structure is available. Used for memory-level conflict policies.
        topic_value : str | None
            Value expressed for ``topic_identity`` in this interaction, when a
            stable value can be extracted.
        is_query_like : bool
            True when the interaction is primarily a query/request rather than
            memory content. This is derived in the language adapter so memory
            policy can stay language-agnostic.
        is_ack_like : bool
            True when the interaction is primarily an acknowledgement or light
            reformulation rather than a new memory-bearing statement.
    
    Args:
        info_density: See the function signature and surrounding type hints.
        sentiment_abs: See the function signature and surrounding type hints.
        entity_count: See the function signature and surrounding type hints.
        is_system_prompt: See the function signature and surrounding type hints.
        latency_ms: See the function signature and surrounding type hints.
        semantic_divergence: See the function signature and surrounding type hints.
        survival_score: See the function signature and surrounding type hints.
        status: See the function signature and surrounding type hints.
        raw_metadata: See the function signature and surrounding type hints.
        provenance: See the function signature and surrounding type hints.
        signals: See the function signature and surrounding type hints.
        topic_identity: See the function signature and surrounding type hints.
        topic_value: See the function signature and surrounding type hints.
        is_query_like: See the function signature and surrounding type hints.
        is_ack_like: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    info_density: float
    sentiment_abs: float
    entity_count: int
    is_system_prompt: bool
    latency_ms: float
    semantic_divergence: float = 0.0
    survival_score: float | None = None
    status: SurvivalStatus | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: InteractionProvenance = field(default_factory=InteractionProvenance)
    signals: InteractionSignals = field(default_factory=InteractionSignals)
    topic_identity: str | None = None
    topic_value: str | None = None
    is_query_like: bool = False
    is_ack_like: bool = False


    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation of this report.
        
                Uses dataclasses.asdict() for a deep copy — safe to mutate the
                result without affecting the original report. Enum fields are
                converted to their ``.value`` string so the output is directly
                JSON-serialisable without a custom encoder.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        d = dataclasses.asdict(self)
        # Convert any Enum instances left by asdict's deepcopy to plain
        # values so json.dumps works without a custom encoder.
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.value
        return d

    def to_json(self) -> str:
        """Return a JSON string representation of this report.
        
                Produces human-readable output (indent=2) suitable for structured
                logging and metadata persistence.
                The output is guaranteed to be parseable by json.loads().
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return json.dumps(self.to_dict(), indent=2)
