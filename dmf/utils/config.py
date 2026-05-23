"""Configuration dataclasses for the Deterministic Memory Framework.

Each DMF component receives its own typed configuration object in its
constructor. Using ``frozen=True`` enforces immutability: once a component is
initialized, its base parameters cannot change during the session's execution,
preventing accidental mid-session drift.

Configs
-------
NLPConfig      : spaCy model, system-prompt gating.
VectorConfig   : FastEmbed model, window size.
ScoringConfig  : Survival Score weights (α, β, γ, δ), sigmoid
                 midpoint (x₀), entity saturation cap (E_cap).
DecayConfig    : temporal decay rate (λ), inertia strength (η),
                 hard-kill threshold, token budget, pruning cadence.
PruningPriorityConfig : pressure-based eviction bonuses for
                        operationally important memories.
"""

from __future__ import annotations

from dataclasses import dataclass

from dmf.utils.constants import (
    DEFAULT_DECAY_HARD_KILL_THRESHOLD,
    DEFAULT_DECAY_INERTIA_STRENGTH,
    DEFAULT_DECAY_LAMBDA_BASE,
    DEFAULT_EMBEDDING_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LTM_DISTANCE_THRESHOLD,
    DEFAULT_LTM_RECALL_LIMIT,
    DEFAULT_PRUNING_FREQUENCY,
    DEFAULT_PRUNING_RHO_CONSTRAINT,
    DEFAULT_PRUNING_RHO_CORRECTION,
    DEFAULT_PRUNING_RHO_CURRENT_STATE,
    DEFAULT_PRUNING_RHO_PREFERENCE,
    DEFAULT_PRUNING_RHO_REPLACEMENT,
    DEFAULT_PRUNING_SUPERSEDED_PAST_PENALTY,
    DEFAULT_RETRIEVAL_CARD_PREFETCH_K,
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
    DEFAULT_SPACY_MODEL,
    DEFAULT_TIER_CRITICAL_MAX,
    DEFAULT_TIER_HEALTHY_MIN,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_VECTOR_DIM,
    DEFAULT_WINDOW_SIZE,
)


@dataclass(frozen=True)
class NLPConfig:
    """Immutable configuration for the NLP Engine.
    
        Passed to the NLP Engine at construction time and never mutated
        afterwards. All fields have sensible defaults so callers only need
        to override what differs from the standard profile.
    
        Attributes
        ----------
        spacy_model : str
            Name of the spaCy model used for tokenization, POS tagging, and
            NER. Defaults to "en_core_web_sm" (English, CPU-optimized, small
            footprint).
        analyze_system_prompt : bool
            When True, the NLP Engine processes the system prompt and extracts
            its embedding as a Semantic Anchor for the geometry module.
            Defaults to False to avoid style-content pollution.
    
    Args:
        spacy_model: See the function signature and surrounding type hints.
        analyze_system_prompt: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    spacy_model: str = DEFAULT_SPACY_MODEL
    analyze_system_prompt: bool = False



@dataclass(frozen=True)
class VectorConfig:
    """Immutable configuration for the Embedding Engine.
    
        Passed to EmbeddingEngine at construction time and never mutated
        afterwards. All fields have sensible defaults for local CPU inference.
    
        Attributes
        ----------
        model_name : str
            FastEmbed model identifier. Defaults to "BAAI/bge-small-en-v1.5",
            a CPU-optimized model whose native output dimension is 384.
        vector_dim : int
            Expected output dimension of the embedding model. Defaults to 384,
            matching the native output of bge-small-en-v1.5. Projection to 256
            is deferred to the geometry layer.
        cache_dir : str
            Local filesystem path where FastEmbed caches downloaded model
            weights. Defaults to "models/embeddings".
        window_size : int
            Maximum number of recent interaction vectors retained by the
            InteractionMatrix sliding window. When the window is full, the
            oldest vector is evicted automatically (FIFO via collections.deque).
            Defaults to 10.
    
    Args:
        model_name: See the function signature and surrounding type hints.
        vector_dim: See the function signature and surrounding type hints.
        cache_dir: See the function signature and surrounding type hints.
        window_size: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    vector_dim: int = DEFAULT_VECTOR_DIM
    cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR
    window_size: int = DEFAULT_WINDOW_SIZE



@dataclass(frozen=True)
class ScoringConfig:
    """Immutable configuration for the Scoring Engine.
    
        Stores the legacy content weights (α, β, γ, δ), the sigmoid midpoint
        (x₀), and the entity saturation cap (E_cap) used by the Survival
        Score formula.
    
        The final score keeps a single sigmoid and adds a new operational
        pre-sigmoid term:
    
            z  = α·ID + β·|S| + γ·E_norm + δ·D
            z_op = λ_op · (
                η_constraint·constraint +
                η_preference·preference +
                η_current_state·current_state +
                η_correction·correction +
                η_replacement·replacement +
                η_past_state·past_state
            )
            z_total = z + z_op + z_provenance
            Ω  = σ(z_total − x₀) = 1 / (1 + exp(−(z_total − x₀)))
    
        where E_norm = min(E, E_cap) / E_cap.
    
        Default values are calibrated so that an "average" interaction
        (ID ≈ 0.40, |S| ≈ 0.15, E_norm ≈ 0.20, D ≈ 0.15) scores Ω ≈ 0.5,
        giving the sigmoid maximum discriminatory power in the typical range.
    
        Attributes
        ----------
        alpha : float
            Weight for Information Density (ID). Dominant positive signal —
            content-rich messages should survive pruning. Default: 3.0.
        beta : float
            Weight for absolute Sentiment (|S|). Moderate positive signal —
            emotional content has social memory value but should not override
            informational density. Default: 1.5.
        gamma : float
            Weight for normalised Entity count (E_norm). Strong supplementary
            signal — named entities are factual anchors that preserve
            conversation coherence. Default: 2.0.
        delta : float
            Weight for Semantic Divergence (D). Must be negative so that
            context drift penalises survival. Default: −1.5.
        x0 : float
            Sigmoid midpoint — the value of z at which Ω = 0.5. Calibrated
            to the expected weighted sum of an average interaction. Default: 1.5.
        e_cap : int
            Entity saturation cap. Raw entity count E is normalised to
            [0, 1] via min(E, e_cap) / e_cap before weighting. Messages with
            e_cap or more entities receive the maximum entity contribution.
            Default: 5.
        social_threshold : float
            If the raw Ω falls below this value **and** the message is
            identified as a social cue (short + keyword match), the Social
            Floor activates and boosts Ω to at least ``min_social_score``.
            Default: 0.4.
        min_social_score : float
            Minimum Survival Score guaranteed by the Social Floor for
            recognised social cues. Prevents trivially short rapport
            messages ("Thanks!", "Ok") from falling into CRITICAL and
            being pruned, which would create coherence gaps for the LLM.
            Default: 0.25.
        critical_threshold : float
            Ω ≤ this value maps to `CRITICAL`. Default: 0.3.
        healthy_threshold : float
            Ω > this value maps to `HEALTHY`. Default: 0.6.
        lambda_operational : float
            Global multiplier applied to the operational/pragmatic signal
            channel before the final sigmoid. Default: 0.75.
        eta_constraint : float
            Weight for ``signals.is_constraint`` in the operational channel.
            Default: 1.2.
        eta_preference : float
            Weight for ``signals.is_preference`` in the operational channel.
            Default: 0.7.
        eta_current_state : float
            Weight for ``signals.is_current_state`` in the operational channel.
            Default: 0.6.
        eta_correction : float
            Weight for ``signals.is_correction`` in the operational channel.
            Default: 0.9.
        eta_replacement : float
            Weight for ``signals.has_replacement`` in the operational channel.
            Default: 0.5.
        eta_past_state : float
            Weight for ``signals.is_past_state`` in the operational channel.
            Neutral by default: historical state should not be preferred, but
            conflict resolution belongs in memory policies rather than in a
            strong negative score. Default: 0.0.
        user_correction_boost : float
            Pre-sigmoid provenance contribution applied when
            ``provenance.is_user_correction`` is true. Default: 0.0.
        preference_update_boost : float
            Pre-sigmoid provenance contribution applied when
            ``provenance.is_preference_update`` is true. Default: 0.0.
        constraint_boost : float
            Pre-sigmoid provenance contribution applied when
            ``provenance.is_constraint`` is true. Default: 0.0.
        corrected_by_user_penalty : float
            Pre-sigmoid provenance penalty applied when
            ``provenance.corrected_by_user`` is true. Default: 0.0.
    
    Args:
        alpha: See the function signature and surrounding type hints.
        beta: See the function signature and surrounding type hints.
        gamma: See the function signature and surrounding type hints.
        delta: See the function signature and surrounding type hints.
        x0: See the function signature and surrounding type hints.
        e_cap: See the function signature and surrounding type hints.
        social_threshold: See the function signature and surrounding type hints.
        min_social_score: See the function signature and surrounding type hints.
        critical_threshold: See the function signature and surrounding type hints.
        healthy_threshold: See the function signature and surrounding type hints.
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

    alpha: float = DEFAULT_SCORING_ALPHA_DENSITY
    beta: float = DEFAULT_SCORING_GAMMA_SENTIMENT
    gamma: float = DEFAULT_SCORING_BETA_ENTITIES
    delta: float = DEFAULT_SCORING_DELTA_TECHNICAL
    x0: float = DEFAULT_SCORING_SIGMOID_MIDPOINT
    e_cap: int = DEFAULT_SCORING_ENTITY_CAP
    social_threshold: float = DEFAULT_SCORING_SOCIAL_THRESHOLD
    min_social_score: float = DEFAULT_SCORING_MIN_SOCIAL_SCORE
    critical_threshold: float = DEFAULT_TIER_CRITICAL_MAX
    healthy_threshold: float = DEFAULT_TIER_HEALTHY_MIN
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
class DecayConfig:
    """Immutable configuration for temporal decay.
    
        Controls how the static Survival Score Ω decays over time (measured
        in interaction turns, not wall-clock seconds) to produce the Effective
        Survival Score:
    
            μ       = 1 − η·Ω
            Ω_eff   = Ω · exp(−λ · μ · Δn)
    
        The decay is **interaction-count-based** (Δn = number of newer turns),
        not wall-clock-based.  This guarantees deterministic, reproducible
        memory states: the same message sequence always produces the same
        pruning decisions, regardless of execution timing.
    
        Default values are calibrated so that:
          - A HEALTHY anchor (Ω = 0.85) survives ~141 turns.
          - A CRITICAL noise message (Ω = 0.15) is hard-killed in ~34 turns.
          - The survival gap between tiers widens over time (inertia effect).
    
        Attributes
        ----------
        lambda_decay : float
            Baseline decay rate (λ).  Higher values increase conversational
            pressure — messages are forgotten faster.  The base half-life
            (at μ = 1, i.e. Ω = 0) is ln(2) / λ ≈ 19.8 turns.
            Default: 0.035.
        inertia_strength : float
            Inertia coefficient (η ∈ [0, 1)).  Controls how much the original
            Ω shields a message from decay.  η = 0 means uniform decay for all
            messages.  η = 0.5 means a message with Ω = 0.80 decays at 60% of
            the base rate.  Must be strictly < 1.0 to guarantee that μ > 0 and
            every message eventually decays.
            Default: 0.5.
        hard_kill_threshold : float
            Absolute floor (Ω_kill).  When Ω_eff drops below this value the
            interaction is physically evicted from Working Memory and archived
            to Long-Term Memory. Set low enough that pressure-based
            pruning (argmin) handles most evictions before passive hard-kill
            fires.
            Default: 0.05.
        token_budget : int
            Maximum total tokens allowed in the active context window.  When
            the sum of all retained interaction tokens exceeds this limit,
            pressure-based pruning evicts the lowest effective pruning score
            interaction until the budget is met.
            Default: 4096.
        pruning_frequency : int
            Number of new interaction turns between periodic cleanup sweeps.
            Every ``pruning_frequency`` turns the buffer manager scans for
            hard-kill candidates and enforces the token budget.  A value of
            1 means "check every turn" (most responsive but highest overhead);
            5 means "check every 5 turns" (batched, lower overhead).
            Default: 5.
    
    Args:
        lambda_decay: See the function signature and surrounding type hints.
        inertia_strength: See the function signature and surrounding type hints.
        hard_kill_threshold: See the function signature and surrounding type hints.
        token_budget: See the function signature and surrounding type hints.
        pruning_frequency: See the function signature and surrounding type hints.
        critical_threshold: See the function signature and surrounding type hints.
        healthy_threshold: See the function signature and surrounding type hints.
        ltm_recall_limit: See the function signature and surrounding type hints.
        ltm_threshold: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    lambda_decay: float = DEFAULT_DECAY_LAMBDA_BASE
    inertia_strength: float = DEFAULT_DECAY_INERTIA_STRENGTH
    hard_kill_threshold: float = DEFAULT_DECAY_HARD_KILL_THRESHOLD
    token_budget: int = DEFAULT_TOKEN_BUDGET
    pruning_frequency: int = DEFAULT_PRUNING_FREQUENCY

    critical_threshold: float = DEFAULT_TIER_CRITICAL_MAX
    healthy_threshold: float = DEFAULT_TIER_HEALTHY_MIN

    ltm_recall_limit: int = DEFAULT_LTM_RECALL_LIMIT
    ltm_threshold: float = DEFAULT_LTM_DISTANCE_THRESHOLD


@dataclass(frozen=True)
class RetrievalConfig:
    """Opt-in phase-3 candidate-generation prefetch settings.
    
        These values are intentionally separate from ``DecayConfig.ltm_recall_limit``.
        They size the standalone candidate pool and do not change the final
        evidence cutoff used by the structured retrieval facade.
    
    Args:
        card_prefetch_k: See the function signature and surrounding type hints.
        raw_prefetch_k: See the function signature and surrounding type hints.
        symbolic_lookup_k: See the function signature and surrounding type hints.
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
    enable_raw_semantic: bool = True
    enable_raw_lexical: bool = True
    enable_card_semantic: bool = True
    enable_card_symbolic: bool = True



@dataclass(frozen=True)
class PruningPriorityConfig:
    """Retention bonuses used only during budget-pressure pruning.
    
        ``TemporalMemory.prune_to_budget`` computes a pressure score per entry:
    
            pruning_bonus = (
                ρ_constraint   · is_constraint +
                ρ_preference   · is_preference +
                ρ_current_state· is_current_state +
                ρ_correction   · is_correction +
                ρ_replacement  · has_replacement
            )
    
            effective_pruning_score = Ω_eff + pruning_bonus
    
        Lower scores are evicted first. These bonuses do not change tier
        classification, temporal decay, periodic hard-kill, or LTM recall.
    
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
            Retention bonus for replacement patterns such as ``not X but Y``.
            Default: 0.08.
        superseded_past_penalty : float
            Penalty applied during budget-pressure pruning when an older entry
            is semantically superseded by a newer entry on the same topic.
            Lower effective pruning scores are evicted first, so this value is
            subtracted from ``effective_pruning_score``. Default: 0.35.
    
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
