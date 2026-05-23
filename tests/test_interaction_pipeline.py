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

"""
tests/test_interaction_pipeline.py
-----------------------------------
Integration tests for dmf/core/pipeline.py — InteractionPipeline.

These tests exercise the full Module 1 + Module 2 pipeline end-to-end using
real components (spaCy, VADER, FastEmbed). The FastEmbed model
(BAAI/bge-small-en-v1.5, ~24 MB) is loaded once via a module-scoped fixture
and cached locally for subsequent runs.

Coverage
--------
  AnalysisReport contract
    - Returns an AnalysisReport instance.
    - semantic_divergence field is a float.
    - NLP fields (info_density, sentiment_abs, entity_count) are populated.
    - latency_ms is positive (NLP timer wired correctly).

  Semantic divergence — edge cases
    - First interaction: D = 0.0 (no prior context).
    - Skipped system prompt: D = 0.0 (embedding and geometry bypassed).

  Semantic divergence — ordering
    - Topic-consistent follow-up has lower D than an out-of-context interjection.

  System-prompt gating
    - analyze_system_prompt=False (default): system prompt skipped, D = 0.0.
    - analyze_system_prompt=True: system prompt fully analysed, D = 0.0 on first.

  Serialization
    - semantic_divergence round-trips through to_dict() and to_json().
"""

from __future__ import annotations

import json

import pytest

from dmf.runtime.pipeline import InteractionPipeline
from dmf.models.analysis import AnalysisReport, InteractionProvenance
from dmf.utils.config_loader import load_dmf_config
from dmf.utils.config import NLPConfig, VectorConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline() -> InteractionPipeline:
    """Module-scoped pipeline with default configs.

    spaCy (en_core_web_sm) loads at construction; FastEmbed loads lazily on
    the first get_embedding() call. Both are cached for the full test module.
    The InteractionMatrix accumulates state across all tests that use this
    fixture — tests that require an isolated matrix use a fresh pipeline.
    """
    return InteractionPipeline(
        nlp_config=NLPConfig(),
        vector_config=VectorConfig(),
    )


def _fresh_pipeline() -> InteractionPipeline:
    """Return a new pipeline with an empty InteractionMatrix."""
    return InteractionPipeline(
        nlp_config=NLPConfig(),
        vector_config=VectorConfig(),
    )


# ---------------------------------------------------------------------------
# AnalysisReport contract
# ---------------------------------------------------------------------------

class TestReportContract:
    """The pipeline must return a well-formed, fully enriched AnalysisReport."""

    def test_returns_analysis_report_instance(self, pipeline: InteractionPipeline) -> None:
        """analyze_interaction() must return an AnalysisReport."""
        report = pipeline.analyze_interaction("The server processed the request.")
        assert isinstance(report, AnalysisReport)

    def test_semantic_divergence_is_a_float(self, pipeline: InteractionPipeline) -> None:
        """semantic_divergence must be a Python float."""
        report = pipeline.analyze_interaction("The database connection pool is exhausted.")
        assert isinstance(report.semantic_divergence, float)

    def test_nlp_fields_are_populated(self, pipeline: InteractionPipeline) -> None:
        """NLP-derived fields must pass through from NLPEngine unchanged."""
        text = "Apple unveiled a new MacBook Pro at its headquarters in Cupertino."
        report = pipeline.analyze_interaction(text)
        # Technical text with named entities → non-zero NLP signals
        assert report.info_density > 0.0
        assert report.entity_count > 0
        assert report.latency_ms > 0.0

    def test_latency_ms_is_positive(self, pipeline: InteractionPipeline) -> None:
        """latency_ms must be > 0 — the ExecutionLatencyTimer must be wired."""
        report = pipeline.analyze_interaction("Processing latency must be measured.")
        assert report.latency_ms > 0.0

    def test_is_system_prompt_flag_is_forwarded(self, pipeline: InteractionPipeline) -> None:
        """is_system_prompt=False must be forwarded correctly by the pipeline."""
        report = pipeline.analyze_interaction(
            "The network interface failed to initialise.", is_system=False
        )
        assert report.is_system_prompt is False

    def test_explicit_provenance_is_forwarded_to_report(self) -> None:
        p = _fresh_pipeline()
        provenance = InteractionProvenance(
            role="user",
            source_turn=7,
            is_user_correction=True,
        )

        report = p.analyze_interaction(
            "No, I prefer risotto.",
            provenance=provenance,
        )

        assert report.provenance == provenance

    def test_from_dmf_config_overrides_system_prompt_gating(self) -> None:
        cfg = load_dmf_config()
        p = InteractionPipeline.from_dmf_config(
            cfg,
            analyze_system_prompt=True,
        )

        assert p._nlp_config.spacy_model == cfg.nlp.spacy_model
        assert p._nlp_config.analyze_system_prompt is True
        assert p._vector_config.vector_dim == cfg.nlp.vector_dim
        assert p._vector_config.window_size == cfg.capacity.window_size

    def test_analyze_interaction_with_vector_returns_embedding(self) -> None:
        p = _fresh_pipeline()

        report, vector = p.analyze_interaction_with_vector(
            "The cache invalidation routine updates Redis keys.",
        )

        assert isinstance(report, AnalysisReport)
        assert vector is not None
        assert vector.shape == (p._vector_config.vector_dim,)

    def test_analyze_interaction_with_vector_returns_none_for_skipped_system(self) -> None:
        p = _fresh_pipeline()

        report, vector = p.analyze_interaction_with_vector(
            "You are a precise assistant.",
            is_system=True,
        )

        assert report.raw_metadata.get("skipped") is True
        assert vector is None


# ---------------------------------------------------------------------------
# Semantic divergence — edge cases
# ---------------------------------------------------------------------------

class TestDivergenceEdgeCases:
    """Edge-case contracts for semantic_divergence."""

    def test_first_interaction_has_zero_divergence(self) -> None:
        """Empty matrix → no prior context → D must be exactly 0.0."""
        p = _fresh_pipeline()
        report = p.analyze_interaction("Initialising the embedding pipeline.")
        assert report.semantic_divergence == 0.0

    def test_skipped_system_prompt_has_zero_divergence(self) -> None:
        """Gated system prompt → embedding bypassed → D must remain 0.0."""
        p = _fresh_pipeline()
        report = p.analyze_interaction(
            "You are a helpful assistant.", is_system=True
        )
        # NLPConfig.analyze_system_prompt defaults to False → gated
        assert report.raw_metadata.get("skipped") is True
        assert report.semantic_divergence == 0.0

    def test_skipped_system_prompt_does_not_update_matrix(self) -> None:
        """A gated system prompt must not add a vector to the InteractionMatrix.

        After one skipped system prompt, the next real interaction must still
        see an empty matrix and return D = 0.0 (first-call edge case).
        """
        p = _fresh_pipeline()
        p.analyze_interaction("You are a precise assistant.", is_system=True)
        report = p.analyze_interaction("Tell me about cloud computing.")
        # The matrix had no vectors before this call → D = 0.0
        assert report.semantic_divergence == 0.0

    def test_skipped_system_prompt_keeps_explicit_provenance(self) -> None:
        p = _fresh_pipeline()
        provenance = InteractionProvenance(role="system", source_turn=0)

        report = p.analyze_interaction(
            "You are a precise assistant.",
            is_system=True,
            provenance=provenance,
        )

        assert report.raw_metadata.get("skipped") is True
        assert report.provenance == provenance

    def test_skipped_system_prompt_keeps_empty_signal_shape(self) -> None:
        p = _fresh_pipeline()

        report = p.analyze_interaction(
            "You are a precise assistant.",
            is_system=True,
        )

        assert report.raw_metadata.get("skipped") is True
        assert report.signals.is_current_state is False
        assert report.signals.is_preference is False
        assert report.raw_metadata["signals"]["is_current_state"] is False
        assert report.raw_metadata["signals"]["is_constraint"] is False
        assert report.raw_metadata["signal_evidence"] == []
        assert report.topic_identity is None
        assert report.topic_value is None
        assert report.raw_metadata["topic_identity"] is None
        assert report.raw_metadata["topic_value"] is None


# ---------------------------------------------------------------------------
# Semantic divergence — ordering
# ---------------------------------------------------------------------------

class TestDivergenceOrdering:
    """The relative ordering of D scores must reflect semantic distance."""

    def test_on_topic_follow_up_has_lower_divergence_than_topic_switch(self) -> None:
        """D(on-topic) < D(off-topic) after a coherent context is established.

        Strategy
        --------
        1. Seed the matrix with several machine-learning sentences to build
           a stable context centroid.
        2. Add an on-topic sentence (still about ML) → record D_similar.
        3. Reset and repeat seeding, then add an unrelated sentence
           (cooking recipe) → record D_outlier.
        4. Assert D_outlier > D_similar.
        """
        ml_texts = [
            "Neural networks learn representations through backpropagation.",
            "Gradient descent optimises the loss function over many epochs.",
            "Convolutional layers extract spatial features from image tensors.",
        ]

        # --- similar follow-up ---
        p1 = _fresh_pipeline()
        for t in ml_texts:
            p1.analyze_interaction(t)
        report_similar = p1.analyze_interaction(
            "Transformer architectures use self-attention to process sequences."
        )

        # --- out-of-context interjection ---
        p2 = _fresh_pipeline()
        for t in ml_texts:
            p2.analyze_interaction(t)
        report_outlier = p2.analyze_interaction(
            "Sauté the onions in olive oil until golden brown, then add garlic."
        )

        assert report_outlier.semantic_divergence > report_similar.semantic_divergence

    def test_identical_follow_up_has_near_zero_divergence(self) -> None:
        """Repeating the same sentence must produce D ≈ 0.

        After the first call (D = 0.0 by edge case), the centroid IS the
        vector. Adding the same vector again → cos(θ) = 1 → D ≈ 0.
        """
        p = _fresh_pipeline()
        text = "The function returns the cached embedding for this token."
        p.analyze_interaction(text)          # first → D=0.0, seeds matrix
        report = p.analyze_interaction(text)  # second → centroid == vector
        assert report.semantic_divergence < 0.05


# ---------------------------------------------------------------------------
# System-prompt gating with analyze_system_prompt=True
# ---------------------------------------------------------------------------

class TestSystemPromptEnabled:
    """When analyze_system_prompt=True, system prompts flow through the full pipeline."""

    def test_enabled_system_prompt_is_fully_analysed(self) -> None:
        """analyze_system_prompt=True must produce non-zero NLP fields."""
        p = InteractionPipeline(
            nlp_config=NLPConfig(analyze_system_prompt=True),
            vector_config=VectorConfig(),
        )
        report = p.analyze_interaction(
            "You are a knowledgeable and concise assistant.", is_system=True
        )
        assert report.is_system_prompt is True
        assert report.info_density > 0.0
        # First interaction → D = 0.0 even when fully analysed
        assert report.semantic_divergence == 0.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """semantic_divergence must survive to_dict() and to_json() round-trips."""

    def test_semantic_divergence_present_in_to_dict(self, pipeline: InteractionPipeline) -> None:
        """to_dict() must include the semantic_divergence key."""
        report = pipeline.analyze_interaction("Memory allocation failed unexpectedly.")
        assert "semantic_divergence" in report.to_dict()

    def test_semantic_divergence_round_trips_through_to_json(
        self, pipeline: InteractionPipeline
    ) -> None:
        """semantic_divergence must survive a to_json() → json.loads() round-trip."""
        report = pipeline.analyze_interaction("The scheduler queued four pending tasks.")
        parsed = json.loads(report.to_json())
        assert "semantic_divergence" in parsed
        assert isinstance(parsed["semantic_divergence"], float)
