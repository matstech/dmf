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

"""Universal configuration loader for DMF settings.

Parses ``dmf_settings.toml`` once at startup and exposes a single
immutable ``DMFConfig`` object that is injected into every DMF component
as a dependency.

Architecture
------------
``DMFConfig`` is a tree of frozen dataclasses — one leaf dataclass per TOML
section.  This mirrors the TOML structure exactly so that readers can map
any config field to its TOML source without indirection:

    DMFConfig
    ├── nlp      : NLPSettings              ← [nlp]
    ├── scoring  : ScoringWeightsSettings   ← [scoring_weights]
    ├── decay    : TemporalDecaySettings    ← [temporal_decay]
    ├── tiers    : MemoryTiersSettings      ← [memory_tiers]
    ├── capacity : CapacitySettings         ← [capacity]
    └── pruning_priority : PruningPrioritySettings ← [pruning_priority]

Parsing
-------
Uses ``tomllib`` from the Python 3.11+ standard library for TOML decoding.
Pydantic validates and coerces the raw TOML tree at the loader boundary before
the public frozen dataclasses are constructed.

Validation
----------
Each leaf dataclass is ``frozen=True`` to enforce immutability after load.
Type coercion and shape validation happen before runtime components receive
the config, so bad TOML values fail at startup rather than silently producing
wrong results at runtime.

Usage
-----
::

    from dmf.utils.config_loader import load_dmf_config, DMFConfig

    cfg: DMFConfig = load_dmf_config()           # default path
    tm = TemporalMemory.from_dmf_config(cfg)
    engine = ScoringEngine.from_dmf_config(cfg)

Integration with existing configs
----------------------------------
``DMFConfig`` does NOT replace ``ScoringConfig``, ``DecayConfig``, or
``VectorConfig``.  Those remain the internal typed contracts used by the
component constructors and all existing unit tests.  The ``from_dmf_config``
factory classmethods on each component translate ``DMFConfig`` → the
appropriate internal config object transparently.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dmf.utils.constants import (
    DEFAULT_DECAY_HARD_KILL_THRESHOLD,
    DEFAULT_DECAY_INERTIA_STRENGTH,
    DEFAULT_DECAY_LAMBDA_BASE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LTM_CARDS_COLLECTION_NAME,
    DEFAULT_LTM_CARDS_PATH,
    DEFAULT_LTM_CHROMA_AUTH_TOKEN_ENV,
    DEFAULT_LTM_CHROMA_DATABASE,
    DEFAULT_LTM_CHROMA_HOST,
    DEFAULT_LTM_CHROMA_MODE,
    DEFAULT_LTM_CHROMA_PATH,
    DEFAULT_LTM_CHROMA_PORT,
    DEFAULT_LTM_CHROMA_SSL,
    DEFAULT_LTM_CHROMA_TENANT,
    DEFAULT_LTM_COLLECTION_NAME,
    DEFAULT_LTM_DISTANCE_THRESHOLD,
    DEFAULT_LTM_QDRANT_MODE,
    DEFAULT_LTM_RECALL_LIMIT,
    DEFAULT_LTM_STORAGE_PATH,
    DEFAULT_PRUNING_FREQUENCY,
    DEFAULT_PRUNING_RHO_CONSTRAINT,
    DEFAULT_PRUNING_RHO_CORRECTION,
    DEFAULT_PRUNING_RHO_CURRENT_STATE,
    DEFAULT_PRUNING_RHO_PREFERENCE,
    DEFAULT_PRUNING_RHO_REPLACEMENT,
    DEFAULT_PRUNING_SUPERSEDED_PAST_PENALTY,
    DEFAULT_RETRIEVAL_CARD_PREFETCH_K,
    DEFAULT_RETRIEVAL_FINAL_RECALL_LIMIT,
    DEFAULT_RETRIEVAL_MAX_SUPPORT_TURNS_PER_CARD,
    DEFAULT_RETRIEVAL_RAW_PREFETCH_K,
    DEFAULT_RETRIEVAL_SYMBOLIC_LOOKUP_K,
    DEFAULT_SCORING_ALPHA_DENSITY,
    DEFAULT_SCORING_BETA_ENTITIES,
    DEFAULT_SCORING_CONSTRAINT_BOOST,
    DEFAULT_SCORING_CORRECTED_BY_USER_PENALTY,
    DEFAULT_SCORING_DELTA_TECHNICAL,
    DEFAULT_SCORING_ENTITY_CAP,
    DEFAULT_SCORING_ETA_CONSTRAINT,
    DEFAULT_SCORING_ETA_CORRECTION,
    DEFAULT_SCORING_ETA_CURRENT_STATE,
    DEFAULT_SCORING_ETA_PAST_STATE,
    DEFAULT_SCORING_ETA_PREFERENCE,
    DEFAULT_SCORING_ETA_REPLACEMENT,
    DEFAULT_SCORING_GAMMA_SENTIMENT,
    DEFAULT_SCORING_LAMBDA_OPERATIONAL,
    DEFAULT_SCORING_MIN_SOCIAL_SCORE,
    DEFAULT_SCORING_PREFERENCE_UPDATE_BOOST,
    DEFAULT_SCORING_SIGMOID_MIDPOINT,
    DEFAULT_SCORING_SOCIAL_THRESHOLD,
    DEFAULT_SCORING_USER_CORRECTION_BOOST,
    DEFAULT_SETTINGS_PATH,
    DEFAULT_SPACY_MODEL,
    DEFAULT_TIER_CRITICAL_MAX,
    DEFAULT_TIER_HEALTHY_MIN,
    DEFAULT_TIER_UNSTABLE_MAX,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_VECTOR_DIM,
    DEFAULT_WINDOW_SIZE,
    LTM_BACKEND_CHROMA,
    LTM_BACKEND_FILE,
    LTM_BACKEND_QDRANT,
    LTM_CHROMA_MODE_SERVER,
    SUPPORTED_LTM_BACKENDS,
    SUPPORTED_LTM_CHROMA_MODES,
    SUPPORTED_LTM_QDRANT_MODES,
)

# Default path: project root / dmf_settings.toml
# Resolved relative to this file's location so it works from any cwd.
#   __file__ = .../dmf/utils/config_loader.py
#   .parents[2] = project root
_DEFAULT_TOML: Path = DEFAULT_SETTINGS_PATH


def _validate_memory_tiers(tiers: MemoryTiersSettings) -> None:
    """Validate memory tier thresholds loaded from TOML.

    The DMF currently classifies runtime memory using ``critical_max`` and
    ``healthy_min``. ``unstable_max`` remains part of the public config
    contract and should stay ordered between them for reporting and future
    use, so the loader rejects invalid or self-contradictory tier layouts.

    Raises
    ------
    ValueError
        If thresholds fall outside ``[0.0, 1.0]`` or are not strictly
        ordered as ``critical_max < unstable_max <= healthy_min``.
    """
    if not 0.0 <= tiers.critical_max <= 1.0:
        raise ValueError(
            "memory_tiers.critical_max must be within [0.0, 1.0]"
        )
    if not 0.0 <= tiers.unstable_max <= 1.0:
        raise ValueError(
            "memory_tiers.unstable_max must be within [0.0, 1.0]"
        )
    if not 0.0 <= tiers.healthy_min <= 1.0:
        raise ValueError(
            "memory_tiers.healthy_min must be within [0.0, 1.0]"
        )
    if tiers.critical_max >= tiers.unstable_max:
        raise ValueError(
            "memory_tiers.critical_max must be strictly lower than "
            "memory_tiers.unstable_max"
        )
    if tiers.unstable_max > tiers.healthy_min:
        raise ValueError(
            "memory_tiers.unstable_max must be lower than or equal to "
            "memory_tiers.healthy_min"
        )


def _validate_ltm_settings(ltm: LTMSettings) -> None:
    """Validate common LTM settings and the active backend-specific options."""
    supported = SUPPORTED_LTM_BACKENDS
    if ltm.storage_type not in supported:
        joined = ", ".join(sorted(supported))
        raise ValueError(
            f"ltm.storage_type must be one of {{{joined}}}; "
            f"got {ltm.storage_type!r}"
        )

    if ltm.recall_limit < 0:
        raise ValueError("ltm.recall_limit must be non-negative")
    if not 0.0 <= ltm.distance_threshold <= 2.0:
        raise ValueError("ltm.distance_threshold must be within [0.0, 2.0]")

    chroma_is_active = ltm.enabled and ltm.storage_type == LTM_BACKEND_CHROMA
    qdrant_is_active = ltm.enabled and ltm.storage_type == LTM_BACKEND_QDRANT

    if qdrant_is_active:
        supported_qdrant_modes = SUPPORTED_LTM_QDRANT_MODES
        if ltm.qdrant_mode not in supported_qdrant_modes:
            joined = ", ".join(sorted(supported_qdrant_modes))
            raise ValueError(
                f"ltm.qdrant_mode must be one of {{{joined}}}; "
                f"got {ltm.qdrant_mode!r}"
            )

    if not chroma_is_active:
        return

    supported_chroma_modes = SUPPORTED_LTM_CHROMA_MODES
    if ltm.chroma_mode not in supported_chroma_modes:
        joined = ", ".join(sorted(supported_chroma_modes))
        raise ValueError(
            f"ltm.chroma_mode must be one of {{{joined}}}; "
            f"got {ltm.chroma_mode!r}"
        )

    if ltm.chroma_auth_token_env and not ltm.chroma_auth_token_env.strip():
        raise ValueError(
            "ltm.chroma_auth_token_env must not contain only whitespace"
        )

    if ltm.chroma_mode != LTM_CHROMA_MODE_SERVER:
        return

    if not ltm.chroma_host.strip():
        raise ValueError("ltm.chroma_host must not be empty in server mode")
    if not 1 <= ltm.chroma_port <= 65535:
        raise ValueError(
            "ltm.chroma_port must be between 1 and 65535 in server mode"
        )
    if not ltm.chroma_tenant.strip():
        raise ValueError("ltm.chroma_tenant must not be empty in server mode")
    if not ltm.chroma_database.strip():
        raise ValueError("ltm.chroma_database must not be empty in server mode")


def _validate_retrieval_settings(retrieval: RetrievalSettings) -> None:
    """Validate retrieval-stage budgets and evidence rendering limits."""
    if retrieval.card_prefetch_k < 0:
        raise ValueError("retrieval.card_prefetch_k must be non-negative")
    if retrieval.raw_prefetch_k < 0:
        raise ValueError("retrieval.raw_prefetch_k must be non-negative")
    if retrieval.symbolic_lookup_k < 0:
        raise ValueError("retrieval.symbolic_lookup_k must be non-negative")
    if retrieval.final_recall_limit < 0:
        raise ValueError("retrieval.final_recall_limit must be non-negative")
    if retrieval.max_support_turns_per_card < 1:
        raise ValueError(
            "retrieval.max_support_turns_per_card must be greater than or equal to 1"
        )



@dataclass(frozen=True)
class NLPSettings:
    """Settings from the ``[nlp]`` TOML section.
    
        Attributes
        ----------
        spacy_model : str
            spaCy pipeline name (e.g. ``"en_core_web_sm"``).
        model_name : str
            FastEmbed model identifier.
        vector_dim : int
            Native output dimension of the embedding model.
    
    Args:
        spacy_model: See the function signature and surrounding type hints.
        model_name: See the function signature and surrounding type hints.
        vector_dim: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    spacy_model: str = DEFAULT_SPACY_MODEL
    model_name: str = DEFAULT_EMBEDDING_MODEL
    vector_dim: int = DEFAULT_VECTOR_DIM


@dataclass(frozen=True)
class ScoringWeightsSettings:
    """Settings from the ``[scoring_weights]`` TOML section.
    
        Field names follow the signal each weight amplifies, making the TOML
        self-documenting.  The ``from_dmf_config`` factory on ``ScoringEngine``
        maps these to the internal ``ScoringConfig`` Greek-letter fields:
    
        ======================== ========================
        TOML field               ScoringConfig field
        ======================== ========================
        alpha_density            alpha   (ID weight)
        beta_entities            gamma   (entity weight)
        gamma_sentiment          beta    (sentiment weight)
        delta_technical          delta   (divergence penalty)
        sigmoid_midpoint         x0
        entity_cap               e_cap
        social_threshold         social_threshold
        min_social_score         min_social_score
        ======================== ========================
    
        Attributes
        ----------
        alpha_density : float
            Weight for Information Density (ID). Default: 3.0.
        beta_entities : float
            Weight for normalised Entity count (E_norm). Default: 2.0.
        gamma_sentiment : float
            Weight for absolute Sentiment (|S|). Default: 1.5.
        delta_technical : float
            Weight for Semantic Divergence (D). Negative → penalty. Default: -1.5.
        sigmoid_midpoint : float
            Sigmoid shift x₀. Default: 1.5.
        entity_cap : int
            Entity saturation cap E_cap. Default: 5.
        social_threshold : float
            Social floor activation threshold. Default: 0.4.
        min_social_score : float
            Minimum Ω guaranteed by the Social Floor. Default: 0.25.
        lambda_operational : float
            Global multiplier for the operational pre-sigmoid channel.
            Default: 0.75.
        eta_constraint : float
            Weight for ``signals.is_constraint``. Default: 1.2.
        eta_preference : float
            Weight for ``signals.is_preference``. Default: 0.7.
        eta_current_state : float
            Weight for ``signals.is_current_state``. Default: 0.6.
        eta_correction : float
            Weight for ``signals.is_correction``. Default: 0.9.
        eta_replacement : float
            Weight for ``signals.has_replacement``. Default: 0.5.
        eta_past_state : float
            Weight for ``signals.is_past_state``. Default: 0.0.
        user_correction_boost : float
            Pre-sigmoid provenance contribution for user corrections. Default: 0.0.
        preference_update_boost : float
            Pre-sigmoid provenance contribution for preference updates. Default: 0.0.
        constraint_boost : float
            Pre-sigmoid provenance contribution for constraints. Default: 0.0.
        corrected_by_user_penalty : float
            Pre-sigmoid provenance penalty for content later corrected by the user.
            Default: 0.0.
    
    Args:
        alpha_density: See the function signature and surrounding type hints.
        beta_entities: See the function signature and surrounding type hints.
        gamma_sentiment: See the function signature and surrounding type hints.
        delta_technical: See the function signature and surrounding type hints.
        sigmoid_midpoint: See the function signature and surrounding type hints.
        entity_cap: See the function signature and surrounding type hints.
        social_threshold: See the function signature and surrounding type hints.
        min_social_score: See the function signature and surrounding type hints.
        lambda_operational: See the function signature and surrounding type hints.
        eta_constraint: See the function signature and surrounding type hints.
        eta_preference: See the function signature and surrounding type hints.
        eta_current_state: See the function signature and surrounding type hints.
        eta_correction: See the function signature and surrounding type hints.
        eta_replacement: See the function signature and surrounding type hints.
        eta_past_state: See the function signature and surrounding type hints.
        user_correction_boost: See the function signature and surrounding type hints.
        preference_update_boost: See the function signature and surrounding type hints.
        constraint_boost: See the function signature and surrounding type hints.
        corrected_by_user_penalty: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    alpha_density: float = DEFAULT_SCORING_ALPHA_DENSITY
    beta_entities: float = DEFAULT_SCORING_BETA_ENTITIES
    gamma_sentiment: float = DEFAULT_SCORING_GAMMA_SENTIMENT
    delta_technical: float = DEFAULT_SCORING_DELTA_TECHNICAL
    sigmoid_midpoint: float = DEFAULT_SCORING_SIGMOID_MIDPOINT
    entity_cap: int = DEFAULT_SCORING_ENTITY_CAP
    social_threshold: float = DEFAULT_SCORING_SOCIAL_THRESHOLD
    min_social_score: float = DEFAULT_SCORING_MIN_SOCIAL_SCORE
    lambda_operational: float = DEFAULT_SCORING_LAMBDA_OPERATIONAL
    eta_constraint: float = DEFAULT_SCORING_ETA_CONSTRAINT
    eta_preference: float = DEFAULT_SCORING_ETA_PREFERENCE
    eta_current_state: float = DEFAULT_SCORING_ETA_CURRENT_STATE
    eta_correction: float = DEFAULT_SCORING_ETA_CORRECTION
    eta_replacement: float = DEFAULT_SCORING_ETA_REPLACEMENT
    eta_past_state: float = DEFAULT_SCORING_ETA_PAST_STATE
    user_correction_boost: float = DEFAULT_SCORING_USER_CORRECTION_BOOST
    preference_update_boost: float = DEFAULT_SCORING_PREFERENCE_UPDATE_BOOST
    constraint_boost: float = DEFAULT_SCORING_CONSTRAINT_BOOST
    corrected_by_user_penalty: float = DEFAULT_SCORING_CORRECTED_BY_USER_PENALTY


@dataclass(frozen=True)
class TemporalDecaySettings:
    """Settings from the ``[temporal_decay]`` TOML section.
    
        Attributes
        ----------
        lambda_base : float
            Base decay rate λ. Default: 0.035.
        inertia_strength : float
            Inertia coefficient η ∈ [0, 1). Default: 0.5.
        hard_kill_threshold : float
            Absolute Ω_eff floor below which entries are hard-killed. Default: 0.05.
    
    Args:
        lambda_base: See the function signature and surrounding type hints.
        inertia_strength: See the function signature and surrounding type hints.
        hard_kill_threshold: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    lambda_base: float = DEFAULT_DECAY_LAMBDA_BASE
    inertia_strength: float = DEFAULT_DECAY_INERTIA_STRENGTH
    hard_kill_threshold: float = DEFAULT_DECAY_HARD_KILL_THRESHOLD


@dataclass(frozen=True)
class MemoryTiersSettings:
    """Settings from the ``[memory_tiers]`` TOML section.
    
        Defines the Ω_eff boundaries between CRITICAL, UNSTABLE, and HEALTHY
        tiers.  These thresholds drive pruning priority in ``TemporalMemory``.
    
        Attributes
        ----------
        critical_max : float
            Ω_eff ≤ this → CRITICAL (highest eviction priority). Default: 0.3.
        unstable_max : float
            critical_max < Ω_eff ≤ unstable_max → UNSTABLE. Default: 0.6.
        healthy_min : float
            Ω_eff > this → HEALTHY (protected from budget pressure;
            protected from budget pressure and left to normal decay/cleanup). Default: 0.6.
    
    Args:
        critical_max: See the function signature and surrounding type hints.
        unstable_max: See the function signature and surrounding type hints.
        healthy_min: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    critical_max: float = DEFAULT_TIER_CRITICAL_MAX
    unstable_max: float = DEFAULT_TIER_UNSTABLE_MAX
    healthy_min: float = DEFAULT_TIER_HEALTHY_MIN


@dataclass(frozen=True)
class CapacitySettings:
    """Settings from the ``[capacity]`` TOML section.
    
        Attributes
        ----------
        token_budget : int
            Maximum total tiktoken tokens in the active context window. Default: 4096.
        pruning_frequency_x : int
            Periodic cleanup runs every this many turns.  The ``_x`` suffix is
            present in the TOML key to avoid naming collisions in some TOML
            parser versions. Default: 5.
        window_size : int
            Maximum vectors in the ``InteractionMatrix`` sliding window. Default: 10.
    
    Args:
        token_budget: See the function signature and surrounding type hints.
        pruning_frequency_x: See the function signature and surrounding type hints.
        window_size: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    token_budget: int = DEFAULT_TOKEN_BUDGET
    pruning_frequency_x: int = DEFAULT_PRUNING_FREQUENCY
    window_size: int = DEFAULT_WINDOW_SIZE


@dataclass(frozen=True)
class PruningPrioritySettings:
    """Settings from the ``[pruning_priority]`` TOML section.
    
        These bonuses affect only budget-pressure pruning in ``TemporalMemory``.
        Lower effective pruning scores are evicted first.
    
        Attributes
        ----------
        rho_constraint : float
            Retention bonus for explicit constraints. Default: 0.2.
        rho_preference : float
            Retention bonus for preferences. Default: 0.1.
        rho_current_state : float
            Retention bonus for current-state updates. Default: 0.1.
        rho_correction : float
            Retention bonus for corrections. Default: 0.15.
        rho_replacement : float
            Retention bonus for replacement patterns. Default: 0.08.
        superseded_past_penalty : float
            Penalty applied when an older active entry is topic-superseded by
            a newer one. Default: 0.35.
    
    Args:
        rho_constraint: See the function signature and surrounding type hints.
        rho_preference: See the function signature and surrounding type hints.
        rho_current_state: See the function signature and surrounding type hints.
        rho_correction: See the function signature and surrounding type hints.
        rho_replacement: See the function signature and surrounding type hints.
        superseded_past_penalty: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    rho_constraint: float = DEFAULT_PRUNING_RHO_CONSTRAINT
    rho_preference: float = DEFAULT_PRUNING_RHO_PREFERENCE
    rho_current_state: float = DEFAULT_PRUNING_RHO_CURRENT_STATE
    rho_correction: float = DEFAULT_PRUNING_RHO_CORRECTION
    rho_replacement: float = DEFAULT_PRUNING_RHO_REPLACEMENT
    superseded_past_penalty: float = DEFAULT_PRUNING_SUPERSEDED_PAST_PENALTY


@dataclass(frozen=True)
class LTMSettings:
    """Settings from the ``[ltm]`` TOML section.
    
        Controls the Long-Term Memory persistence backend used by
        ``TemporalMemory.from_dmf_config`` when constructing the ``LTMHook``.
    
        Attributes
        ----------
        storage_type : str
            Backend identifier.
            ``"file"``   → ``FileLTMHook``  (JSONL audit trail, write-only).
            ``"chroma"`` → ``ChromaLTMHook`` (vector store with active recall).
            ``"qdrant"`` → Qdrant vector store (local in-memory mode).
            ``"null"``   → ``NullLTMHook`` (silent discard, for tests).
            Default: ``"file"``.
        storage_path : str
            Path to the JSONL archive file (for ``storage_type="file"``),
            relative to the working directory.  The parent directory is
            created automatically.  Default: ``"data/ltm_archive.jsonl"``.
        chroma_path : str
            Persist directory for the ChromaDB vector store
            (for ``storage_type="chroma"``).  Created automatically.
            Default: ``"data/ltm_chroma"``.
        chroma_mode : str
            Chroma connection mode: ``"embedded"`` or ``"server"``.
        qdrant_mode : str
            Qdrant connection mode. Only volatile ``"memory"`` is currently
            supported.
        chroma_host : str
            Chroma server hostname.
        chroma_port : int
            Chroma server HTTP port.
        chroma_ssl : bool
            Whether to use HTTPS for a Chroma server connection.
        chroma_tenant : str
            Chroma tenant used by embedded and server clients.
        chroma_database : str
            Chroma database used by embedded and server clients.
        chroma_auth_token_env : str
            Optional environment-variable name containing a server Bearer token.
        collection_name : str
            Raw-record vector collection name. Changing this creates an
            independent namespace for vector-backed LTM sessions or benchmarks.
            Default: ``"dmf_memory"``.
        recall_limit : int
            Maximum number of raw records to retrieve per ``search_raw()`` call
            (passed to ``DecayConfig.ltm_recall_limit``).  Default: 5.
        distance_threshold : float
            Cosine-distance ceiling for recalled raw records
            (passed to ``DecayConfig.ltm_threshold``).  Default: 0.7.
        enabled : bool
            When ``False``, ``from_dmf_config`` falls back to ``NullLTMHook``
            regardless of ``storage_type``.  Useful for test environments that
            must not touch the filesystem.  Default: ``True``.
        cards_enabled : bool
            Enables the auxiliary structured memory-card JSONL index for
            file-backed raw LTM archival. Default: ``False``.
        cards_path : str
            Path to the auxiliary memory-card JSONL archive.
            Default: ``"data/ltm_cards.jsonl"``.
        cards_collection_name : str
            Structured memory-card vector collection name for vector-backed LTM.
            Default: ``"dmf_cards"``.
    
    Args:
        storage_type: See the function signature and surrounding type hints.
        storage_path: See the function signature and surrounding type hints.
        chroma_path: See the function signature and surrounding type hints.
        chroma_mode: See the function signature and surrounding type hints.
        qdrant_mode: See the function signature and surrounding type hints.
        chroma_host: See the function signature and surrounding type hints.
        chroma_port: See the function signature and surrounding type hints.
        chroma_ssl: See the function signature and surrounding type hints.
        chroma_tenant: See the function signature and surrounding type hints.
        chroma_database: See the function signature and surrounding type hints.
        chroma_auth_token_env: See the function signature and surrounding type hints.
        collection_name: See the function signature and surrounding type hints.
        recall_limit: See the function signature and surrounding type hints.
        distance_threshold: See the function signature and surrounding type hints.
        enabled: See the function signature and surrounding type hints.
        cards_enabled: See the function signature and surrounding type hints.
        cards_path: See the function signature and surrounding type hints.
        cards_collection_name: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    storage_type: str = LTM_BACKEND_FILE
    storage_path: str = DEFAULT_LTM_STORAGE_PATH
    chroma_path: str = DEFAULT_LTM_CHROMA_PATH
    collection_name: str = DEFAULT_LTM_COLLECTION_NAME
    recall_limit: int = DEFAULT_LTM_RECALL_LIMIT
    distance_threshold: float = DEFAULT_LTM_DISTANCE_THRESHOLD
    enabled: bool = True
    cards_enabled: bool = False
    cards_path: str = DEFAULT_LTM_CARDS_PATH
    cards_collection_name: str = DEFAULT_LTM_CARDS_COLLECTION_NAME
    qdrant_mode: str = DEFAULT_LTM_QDRANT_MODE
    chroma_mode: str = DEFAULT_LTM_CHROMA_MODE
    chroma_host: str = DEFAULT_LTM_CHROMA_HOST
    chroma_port: int = DEFAULT_LTM_CHROMA_PORT
    chroma_ssl: bool = DEFAULT_LTM_CHROMA_SSL
    chroma_tenant: str = DEFAULT_LTM_CHROMA_TENANT
    chroma_database: str = DEFAULT_LTM_CHROMA_DATABASE
    chroma_auth_token_env: str = DEFAULT_LTM_CHROMA_AUTH_TOKEN_ENV


@dataclass(frozen=True)
class RetrievalSettings:
    """Settings from the optional ``[retrieval]`` TOML section.
    
        These gates and budgets are for the opt-in structured retrieval stack.
        They do not alter ``ltm.recall_limit`` or the existing public final
        retrieval path until the public API is explicitly stabilized.
    
    Args:
        card_prefetch_k: See the function signature and surrounding type hints.
        raw_prefetch_k: See the function signature and surrounding type hints.
        symbolic_lookup_k: See the function signature and surrounding type hints.
        final_recall_limit: See the function signature and surrounding type hints.
        max_support_turns_per_card: See the function signature and surrounding type hints.
        include_superseded_when_historical: See the function signature and surrounding type hints.
        include_neighbor_turns: See the function signature and surrounding type hints.
        enable_raw_semantic: See the function signature and surrounding type hints.
        enable_raw_lexical: See the function signature and surrounding type hints.
        enable_card_semantic: See the function signature and surrounding type hints.
        enable_card_symbolic: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    card_prefetch_k: int = DEFAULT_RETRIEVAL_CARD_PREFETCH_K
    raw_prefetch_k: int = DEFAULT_RETRIEVAL_RAW_PREFETCH_K
    symbolic_lookup_k: int = DEFAULT_RETRIEVAL_SYMBOLIC_LOOKUP_K
    final_recall_limit: int = DEFAULT_RETRIEVAL_FINAL_RECALL_LIMIT
    max_support_turns_per_card: int = DEFAULT_RETRIEVAL_MAX_SUPPORT_TURNS_PER_CARD
    include_superseded_when_historical: bool = True
    include_neighbor_turns: bool = False
    enable_raw_semantic: bool = True
    enable_raw_lexical: bool = True
    enable_card_semantic: bool = True
    enable_card_symbolic: bool = True



@dataclass(frozen=True)
class DMFConfig:
    """Immutable, fully-parsed representation of ``dmf_settings.toml``.
    
        Constructed by ``load_dmf_config()`` and injected into DMF components
        via their ``from_dmf_config()`` factory classmethods.  Never mutated
        after construction.
    
        Attributes
        ----------
        nlp : NLPSettings
            Parsed ``[nlp]`` section.
        scoring : ScoringWeightsSettings
            Parsed ``[scoring_weights]`` section.
        decay : TemporalDecaySettings
            Parsed ``[temporal_decay]`` section.
        tiers : MemoryTiersSettings
            Parsed ``[memory_tiers]`` section.
        capacity : CapacitySettings
            Parsed ``[capacity]`` section.
        pruning_priority : PruningPrioritySettings
            Parsed ``[pruning_priority]`` section.
        ltm : LTMSettings
            Parsed ``[ltm]`` section.
        retrieval : RetrievalSettings
            Parsed ``[retrieval]`` section for opt-in candidate generation.
    
    Args:
        nlp: See the function signature and surrounding type hints.
        scoring: See the function signature and surrounding type hints.
        decay: See the function signature and surrounding type hints.
        tiers: See the function signature and surrounding type hints.
        capacity: See the function signature and surrounding type hints.
        pruning_priority: See the function signature and surrounding type hints.
        ltm: See the function signature and surrounding type hints.
        retrieval: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    nlp: NLPSettings = dataclass_field(default_factory=NLPSettings)
    scoring: ScoringWeightsSettings = dataclass_field(default_factory=ScoringWeightsSettings)
    decay: TemporalDecaySettings = dataclass_field(default_factory=TemporalDecaySettings)
    tiers: MemoryTiersSettings = dataclass_field(default_factory=MemoryTiersSettings)
    capacity: CapacitySettings = dataclass_field(default_factory=CapacitySettings)
    pruning_priority: PruningPrioritySettings = dataclass_field(default_factory=PruningPrioritySettings)
    ltm: LTMSettings = dataclass_field(default_factory=LTMSettings)
    retrieval: RetrievalSettings = dataclass_field(default_factory=RetrievalSettings)


class _ConfigSectionModel(BaseModel):
    """Pydantic base for TOML sections before conversion to public dataclasses."""

    model_config = ConfigDict(extra="ignore")


class _NLPSettingsModel(_ConfigSectionModel):
    spacy_model: str = NLPSettings.spacy_model
    model_name: str = NLPSettings.model_name
    vector_dim: int = NLPSettings.vector_dim


class _ScoringWeightsSettingsModel(_ConfigSectionModel):
    alpha_density: float = ScoringWeightsSettings.alpha_density
    beta_entities: float = ScoringWeightsSettings.beta_entities
    gamma_sentiment: float = ScoringWeightsSettings.gamma_sentiment
    delta_technical: float = ScoringWeightsSettings.delta_technical
    sigmoid_midpoint: float = ScoringWeightsSettings.sigmoid_midpoint
    entity_cap: int = ScoringWeightsSettings.entity_cap
    social_threshold: float = ScoringWeightsSettings.social_threshold
    min_social_score: float = ScoringWeightsSettings.min_social_score
    lambda_operational: float = ScoringWeightsSettings.lambda_operational
    eta_constraint: float = ScoringWeightsSettings.eta_constraint
    eta_preference: float = ScoringWeightsSettings.eta_preference
    eta_current_state: float = ScoringWeightsSettings.eta_current_state
    eta_correction: float = ScoringWeightsSettings.eta_correction
    eta_replacement: float = ScoringWeightsSettings.eta_replacement
    eta_past_state: float = ScoringWeightsSettings.eta_past_state
    user_correction_boost: float = ScoringWeightsSettings.user_correction_boost
    preference_update_boost: float = ScoringWeightsSettings.preference_update_boost
    constraint_boost: float = ScoringWeightsSettings.constraint_boost
    corrected_by_user_penalty: float = ScoringWeightsSettings.corrected_by_user_penalty


class _TemporalDecaySettingsModel(_ConfigSectionModel):
    lambda_base: float = TemporalDecaySettings.lambda_base
    inertia_strength: float = TemporalDecaySettings.inertia_strength
    hard_kill_threshold: float = TemporalDecaySettings.hard_kill_threshold


class _MemoryTiersSettingsModel(_ConfigSectionModel):
    critical_max: float = MemoryTiersSettings.critical_max
    unstable_max: float = MemoryTiersSettings.unstable_max
    healthy_min: float = MemoryTiersSettings.healthy_min


class _CapacitySettingsModel(_ConfigSectionModel):
    token_budget: int = CapacitySettings.token_budget
    pruning_frequency_x: int = CapacitySettings.pruning_frequency_x
    window_size: int = CapacitySettings.window_size


class _PruningPrioritySettingsModel(_ConfigSectionModel):
    rho_constraint: float = PruningPrioritySettings.rho_constraint
    rho_preference: float = PruningPrioritySettings.rho_preference
    rho_current_state: float = PruningPrioritySettings.rho_current_state
    rho_correction: float = PruningPrioritySettings.rho_correction
    rho_replacement: float = PruningPrioritySettings.rho_replacement
    superseded_past_penalty: float = PruningPrioritySettings.superseded_past_penalty


class _LTMSettingsModel(_ConfigSectionModel):
    storage_type: str = LTMSettings.storage_type
    storage_path: str = LTMSettings.storage_path
    chroma_path: str = LTMSettings.chroma_path
    qdrant_mode: str = LTMSettings.qdrant_mode
    chroma_mode: str = LTMSettings.chroma_mode
    chroma_host: str = LTMSettings.chroma_host
    chroma_port: int = LTMSettings.chroma_port
    chroma_ssl: bool = LTMSettings.chroma_ssl
    chroma_tenant: str = LTMSettings.chroma_tenant
    chroma_database: str = LTMSettings.chroma_database
    chroma_auth_token_env: str = LTMSettings.chroma_auth_token_env
    collection_name: str = LTMSettings.collection_name
    recall_limit: int = LTMSettings.recall_limit
    distance_threshold: float = LTMSettings.distance_threshold
    enabled: bool = LTMSettings.enabled
    cards_enabled: bool = LTMSettings.cards_enabled
    cards_path: str = LTMSettings.cards_path
    cards_collection_name: str = LTMSettings.cards_collection_name


class _RetrievalSettingsModel(_ConfigSectionModel):
    card_prefetch_k: int = RetrievalSettings.card_prefetch_k
    raw_prefetch_k: int = RetrievalSettings.raw_prefetch_k
    symbolic_lookup_k: int = RetrievalSettings.symbolic_lookup_k
    final_recall_limit: int = RetrievalSettings.final_recall_limit
    max_support_turns_per_card: int = RetrievalSettings.max_support_turns_per_card
    include_superseded_when_historical: bool = RetrievalSettings.include_superseded_when_historical
    include_neighbor_turns: bool = RetrievalSettings.include_neighbor_turns
    enable_raw_semantic: bool = RetrievalSettings.enable_raw_semantic
    enable_raw_lexical: bool = RetrievalSettings.enable_raw_lexical
    enable_card_semantic: bool = RetrievalSettings.enable_card_semantic
    enable_card_symbolic: bool = RetrievalSettings.enable_card_symbolic


class _DMFConfigModel(BaseModel):
    """Pydantic representation of the raw TOML tree.

    The public runtime contract remains the frozen dataclass ``DMFConfig``.
    Pydantic is kept at the loader boundary to centralize type coercion and
    preserve lightweight dataclasses on hot runtime paths.
    """

    model_config = ConfigDict(extra="ignore")

    nlp: _NLPSettingsModel = Field(default_factory=_NLPSettingsModel)
    scoring_weights: _ScoringWeightsSettingsModel = Field(
        default_factory=_ScoringWeightsSettingsModel
    )
    temporal_decay: _TemporalDecaySettingsModel = Field(
        default_factory=_TemporalDecaySettingsModel
    )
    memory_tiers: _MemoryTiersSettingsModel = Field(
        default_factory=_MemoryTiersSettingsModel
    )
    capacity: _CapacitySettingsModel = Field(default_factory=_CapacitySettingsModel)
    pruning_priority: _PruningPrioritySettingsModel = Field(
        default_factory=_PruningPrioritySettingsModel
    )
    ltm: _LTMSettingsModel = Field(default_factory=_LTMSettingsModel)
    retrieval: _RetrievalSettingsModel = Field(default_factory=_RetrievalSettingsModel)


def _dataclass_from_model(dataclass_type: type, model: BaseModel):
    """Hydrate one public dataclass from a validated Pydantic section."""
    return dataclass_type(**model.model_dump())


def _coerce_raw_config(raw: dict) -> DMFConfig:
    """Validate raw TOML data with Pydantic and return the public dataclass tree."""
    try:
        model = _DMFConfigModel.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    return DMFConfig(
        nlp=_dataclass_from_model(NLPSettings, model.nlp),
        scoring=_dataclass_from_model(
            ScoringWeightsSettings,
            model.scoring_weights,
        ),
        decay=_dataclass_from_model(
            TemporalDecaySettings,
            model.temporal_decay,
        ),
        tiers=_dataclass_from_model(MemoryTiersSettings, model.memory_tiers),
        capacity=_dataclass_from_model(CapacitySettings, model.capacity),
        pruning_priority=_dataclass_from_model(
            PruningPrioritySettings,
            model.pruning_priority,
        ),
        ltm=_dataclass_from_model(LTMSettings, model.ltm),
        retrieval=_dataclass_from_model(RetrievalSettings, model.retrieval),
    )


def load_dmf_config(path: Path | str | None = None) -> DMFConfig:
    """Parse ``dmf_settings.toml`` and return an immutable ``DMFConfig``.

    Args:
        path: TOML file path. ``None`` resolves to ``dmf_settings.toml`` in the
            project root, using this module location instead of the current
            working directory.

    Returns:
        Fully populated, immutable configuration object.

    Raises:
        FileNotFoundError: If the TOML file does not exist at the resolved path.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        TypeError: If a required field has an incompatible type.
        ValueError: If a parsed field violates runtime configuration invariants.

    Warning:
        The function does not cache its result. Callers that need process-wide
        reuse should store the returned object explicitly.
    """
    resolved = Path(path) if path is not None else _DEFAULT_TOML

    with open(resolved, "rb") as fh:
        raw: dict = tomllib.load(fh)

    cfg = _coerce_raw_config(raw)

    _validate_memory_tiers(cfg.tiers)
    _validate_ltm_settings(cfg.ltm)
    _validate_retrieval_settings(cfg.retrieval)

    return cfg
