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

"""Survival Score (Ω) calculation for analysed interactions.

This module transforms the raw signals carried by an ``AnalysisReport`` into
a single scalar Survival Score Ω ∈ (0, 1) that determines how likely an
interaction is to be retained in working memory.

Formula
-------
    E_norm = min(E, E_cap) / E_cap
    z_content = α·ID + β·|S| + γ·E_norm + δ·D
    z_operational = λ_op · (
        η_constraint·constraint +
        η_preference·preference +
        η_current_state·current_state +
        η_correction·correction +
        η_replacement·replacement +
        η_past_state·past_state
    )
    z_provenance = configured provenance contributions
    z_total = z_content + z_operational + z_provenance
    Ω      = σ(z_total − x₀) = 1 / (1 + exp(−(z_total − x₀)))

Social Floor
------------
If Ω < social_threshold AND the message is a recognised social cue
(short message with a keyword match), Ω is boosted to at least
min_social_score. This prevents trivially short rapport messages
("Thanks!", "Ok") from being pruned, which would create coherence
gaps for the LLM.

Detection heuristic (CPU-first, deterministic):
    1. Word count ≤ _SOCIAL_CUE_MAX_WORDS  (terse message)
    2. At least one word ∈ _SOCIAL_CUE_KEYWORDS  (social vocabulary)

CPU-First Guarantee
-------------------
The entire calculation is pure floating-point arithmetic + string
operations — no I/O, no model inference, no GPU.
"""

from __future__ import annotations

import math

from dmf.models.analysis import AnalysisReport
from dmf.models.status import classify_survival_status
from dmf.utils.config import ScoringConfig
from dmf.utils.config_loader import DMFConfig



_SOCIAL_CUE_MAX_WORDS: int = 6
"""Maximum word count for a message to qualify as a social cue."""

_STRIP_CHARS: str = ".,!?;:'\"()-…–—"
"""Punctuation stripped from each word before keyword matching."""

_SOCIAL_CUE_KEYWORDS: frozenset[str] = frozenset({
    "thanks", "thank", "thx", "grazie", "merci", "appreciated",
    "hi", "hello", "hey", "bye", "goodbye", "ciao",
    "ok", "okay", "understood", "noted", "sure", "alright", "gotcha",
    "yes", "yeah", "yep", "absolutely", "exactly", "indeed",
    "nice", "great", "cool", "awesome", "perfect", "wonderful", "excellent",
    "sorry", "apologies", "pardon",
    "please", "welcome",
})



class ScoringEngine:
    """Computes the Survival Score Ω for a single AnalysisReport.
    
        Constructed once with an immutable ScoringConfig and reused across
        all interactions in a session.
    
        Attributes
        ----------
        _config : ScoringConfig
            Immutable weight and sigmoid parameters. Read once at construction
            time and never mutated.
    
    Args:
        config: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(self, config: ScoringConfig) -> None:
        """Initialise the Scoring Engine with the given configuration.

        Parameters
        ----------
        config : ScoringConfig
            Frozen dataclass carrying α, β, γ, δ, x₀, E_cap, and
            social floor parameters.
        """
        self._config: ScoringConfig = config


    @classmethod
    def from_dmf_config(cls, config: DMFConfig) -> ScoringEngine:
        """Construct a ``ScoringEngine`` from a universal ``DMFConfig``.
        
                Translates the human-readable TOML field names (``alpha_density``,
                ``beta_entities``, …) to the internal ``ScoringConfig`` Greek-letter
                fields, preserving the existing constructor signature and all tests.
        
                Translation map
                ---------------
                ======================== ======================== ===================
                TOML key                 DMFConfig path           ScoringConfig field
                ======================== ======================== ===================
                scoring_weights.alpha_density    scoring.alpha_density    alpha
                scoring_weights.beta_entities    scoring.beta_entities    gamma
                scoring_weights.gamma_sentiment  scoring.gamma_sentiment  beta
                scoring_weights.delta_technical  scoring.delta_technical  delta
                scoring_weights.sigmoid_midpoint scoring.sigmoid_midpoint x0
                scoring_weights.entity_cap       scoring.entity_cap       e_cap
                scoring_weights.social_threshold scoring.social_threshold social_threshold
                scoring_weights.min_social_score scoring.min_social_score min_social_score
                ======================== ======================== ===================
        
                Note: ``beta_entities`` maps to ``ScoringConfig.gamma`` (entity
                weight) and ``gamma_sentiment`` maps to ``ScoringConfig.beta``
                (sentiment weight).  The TOML uses signal-descriptive names;
                the internal config uses Greek letters matching the formula.
        
                Parameters
                ----------
                config : DMFConfig
                    Fully populated config object returned by ``load_dmf_config()``.
        
                Returns
                -------
                ScoringEngine
                    Fully initialised instance.
        
        Args:
            config: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        scoring_cfg = ScoringConfig(
            alpha=config.scoring.alpha_density,
            beta=config.scoring.gamma_sentiment,
            gamma=config.scoring.beta_entities,
            delta=config.scoring.delta_technical,
            x0=config.scoring.sigmoid_midpoint,
            e_cap=config.scoring.entity_cap,
            social_threshold=config.scoring.social_threshold,
            min_social_score=config.scoring.min_social_score,
            critical_threshold=config.tiers.critical_max,
            healthy_threshold=config.tiers.healthy_min,
            lambda_operational=config.scoring.lambda_operational,
            eta_constraint=config.scoring.eta_constraint,
            eta_preference=config.scoring.eta_preference,
            eta_current_state=config.scoring.eta_current_state,
            eta_correction=config.scoring.eta_correction,
            eta_replacement=config.scoring.eta_replacement,
            eta_past_state=config.scoring.eta_past_state,
            user_correction_boost=config.scoring.user_correction_boost,
            preference_update_boost=config.scoring.preference_update_boost,
            constraint_boost=config.scoring.constraint_boost,
            corrected_by_user_penalty=config.scoring.corrected_by_user_penalty,
        )
        return cls(config=scoring_cfg)


    def calculate_score(
        self,
        report: AnalysisReport,
        text: str = "",
    ) -> float:
        """Compute the Survival Score Ω for *report*.
        
                Steps
                -----
                1. Normalise entity count: E_norm = min(E, E_cap) / E_cap.
                2. Compute weighted content logit.
                3. Compute weighted operational logit from conversational signals.
                4. Compute provenance logit from structured provenance metadata.
                5. Apply a single shifted sigmoid to the total logit.
                6. Social Floor: if Ω < social_threshold and *text* is a social
                   cue, boost Ω to at least min_social_score.
                7. Round to 4 decimal places.
                8. Stamp *report* in-place: set ``report.survival_score = Ω`` and
                   classify ``report.status`` using runtime thresholds so the report
                   always carries its own authoritative score and tier.
        
                Parameters
                ----------
                report : AnalysisReport
                    Fully populated report carrying ID, |S|, E, and D.
                text : str
                    Original interaction text. Required for Social Floor
                    detection (keyword matching). When empty the Social Floor
                    is skipped, preserving backward compatibility.
        
                Returns
                -------
                float
                    Survival Score Ω ∈ (0, 1), rounded to 4 decimal places.
        
        Args:
            report: See the function signature and surrounding type hints.
            text: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        cfg = self._config

        entity_norm = min(report.entity_count, cfg.e_cap) / cfg.e_cap

        z_content = _calculate_content_logit(report, entity_norm, cfg)

        z_operational = _calculate_operational_logit(report, cfg)

        z_provenance = _calculate_provenance_logit(report, cfg)

        omega = _sigmoid((z_content + z_operational + z_provenance) - cfg.x0)

        if text and omega < cfg.social_threshold and _is_social_cue(text):
            omega = max(omega, cfg.min_social_score)

        omega = round(omega, 4)

        report.survival_score = omega
        report.status = classify_survival_status(
            omega=omega,
            critical_threshold=cfg.critical_threshold,
            healthy_threshold=cfg.healthy_threshold,
        )

        return omega




def _calculate_content_logit(
    report: AnalysisReport,
    entity_norm: float,
    cfg: ScoringConfig,
) -> float:
    """Return the legacy pre-sigmoid weighted score."""
    return (
        cfg.alpha * report.info_density
        + cfg.beta * report.sentiment_abs
        + cfg.gamma * entity_norm
        + cfg.delta * report.semantic_divergence
    )


def _calculate_operational_logit(
    report: AnalysisReport,
    cfg: ScoringConfig,
) -> float:
    """Return the weighted conversational-signal contribution."""
    signals = report.signals
    raw_operational = (
        cfg.eta_constraint * float(signals.is_constraint)
        + cfg.eta_preference * float(signals.is_preference)
        + cfg.eta_current_state * float(signals.is_current_state)
        + cfg.eta_correction * float(signals.is_correction)
        + cfg.eta_replacement * float(signals.has_replacement)
        + cfg.eta_past_state * float(signals.is_past_state)
    )
    return cfg.lambda_operational * raw_operational


def _calculate_provenance_logit(
    report: AnalysisReport,
    cfg: ScoringConfig,
) -> float:
    """Return the weighted structured-provenance contribution."""
    provenance = report.provenance
    z_provenance = 0.0

    if provenance.is_user_correction:
        z_provenance += cfg.user_correction_boost
    if provenance.is_preference_update:
        z_provenance += cfg.preference_update_boost
    if provenance.is_constraint:
        z_provenance += cfg.constraint_boost
    if provenance.corrected_by_user:
        z_provenance -= cfg.corrected_by_user_penalty

    return z_provenance

def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid: σ(x) = 1 / (1 + exp(−x)).

    Uses math.exp (not numpy) for scalar precision and zero-dependency
    overhead. Handles large negative x gracefully — math.exp(large
    positive) returns inf, and 1 / (1 + inf) == 0.0 in IEEE 754.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # For x < 0, use the numerically stable form to avoid overflow
    # in exp(-x) when x is a large negative number.
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)

def _is_social_cue(text: str) -> bool:
    """Detect whether *text* is a short social / rapport message.

    Heuristic (two-condition gate):
      1. The message has at most _SOCIAL_CUE_MAX_WORDS words.
      2. At least one word (after punctuation stripping) belongs to the
         _SOCIAL_CUE_KEYWORDS vocabulary.

    Returns False for empty text or messages longer than the word cap,
    even if they contain social keywords — long messages should be scored
    on their actual information content.
    """
    words = text.lower().split()
    if not words or len(words) > _SOCIAL_CUE_MAX_WORDS:
        return False
    cleaned = {w.strip(_STRIP_CHARS) for w in words} - {""}
    return bool(cleaned & _SOCIAL_CUE_KEYWORDS)
