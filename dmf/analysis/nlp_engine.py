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

"""Deterministic NLP feature extraction from raw interaction text.

Transforms a conversation interaction into an AnalysisReport containing:

  - Three numerical signals consumed by the scoring engine:
    Information Density (ID), Absolute Sentiment (|S|), Entity Protection (E).
  - One structured conversational-signal block extracted through a language
    adapter and kept separate from the legacy numeric metrics.

No LLMs are involved. All computation is deterministic and CPU-bound.
"""

from __future__ import annotations

import spacy
from spacy.language import Language
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from dmf.analysis.constants import SEMANTIC_POS_TAGS
from dmf.models.analysis import AnalysisReport, InteractionSignals
from dmf.analysis.signals import EnglishSignalAdapter
from dmf.utils.config import NLPConfig
from dmf.utils.perf import ExecutionLatencyTimer



# POS tags that carry semantic weight in the Information Density formula.
# Only these four universal tags (NOUN, VERB, ADJ, PROPN) are considered
# "meaningful" — determiners, conjunctions, punctuation, etc. are excluded
# because they contribute syntax but not information content.
# Returned when analysis is skipped (system prompt with flag disabled)
# or when the input text is empty. Neutral weights do not distort scoring.
NEUTRAL_INFO_DENSITY: float = 0.0
NEUTRAL_SENTIMENT_ABS: float = 0.0
NEUTRAL_ENTITY_COUNT: int = 0



class NLPEngine:
    """Deterministic NLP feature extractor for DMF interactions.
    
        Loads a spaCy model and VADER once at construction time and reuses
        them across all calls, avoiding per-interaction model loading.
    
        Attributes are prefixed with _ to signal they are implementation
        details; callers interact only through analyze_interaction().
    
    Args:
        config: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(self, config: NLPConfig) -> None:
        """Initialise the NLP Engine with the provided configuration.

        Parameters
        ----------
        config : NLPConfig
            Immutable configuration object. spacy_model and
            analyze_system_prompt are read once here and never changed.

        Raises
        ------
        OSError
            If the spaCy model named in config.spacy_model is not installed.
            Run: python -m spacy download <model_name>
        """
        self._config: NLPConfig = config
        self._nlp_model: Language = spacy.load(config.spacy_model)
        self._sentiment_analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
        self._latency_timer: ExecutionLatencyTimer = ExecutionLatencyTimer()
        self._signal_adapter = EnglishSignalAdapter(self._nlp_model)


    def analyze_interaction(
        self,
        text: str,
        is_system: bool = False,
    ) -> AnalysisReport:
        """Extract NLP features from a single interaction text.
        
                When is_system=True and config.analyze_system_prompt=False, the
                system prompt is intentionally skipped and neutral weights (0.0)
                are returned. This prevents the system prompt's style (e.g.
                "be assertive") from polluting content-based scoring.
        
                Parameters
                ----------
                text : str
                    Raw interaction content to analyse.
                is_system : bool
                    True if this text comes from the system prompt role.
        
                Returns
                -------
                AnalysisReport
                    Populated with ID, |S|, E, latency_ms, and raw_metadata.
        
        Args:
            text: See the function signature and surrounding type hints.
            is_system: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        if is_system and not self._config.analyze_system_prompt:
            return self._build_skipped_system_prompt_report()

        with self._latency_timer:
            doc = self._nlp_model(text)
            info_density = self._calculate_information_density_score(doc)
            sentiment_abs = self._calculate_absolute_sentiment_score(text)
            entity_count = self._calculate_entity_count(doc)
            pos_counts = self._extract_pos_counts(doc)
            entity_details = self._extract_entity_details(doc)
            extraction = self._signal_adapter.extract(text, doc)
            extraction_metadata = extraction.metadata_fields()

        return AnalysisReport(
            info_density=info_density,
            sentiment_abs=sentiment_abs,
            entity_count=entity_count,
            is_system_prompt=is_system,
            latency_ms=self._latency_timer.elapsed_ms,
            topic_identity=extraction_metadata["topic_identity"],
            topic_value=extraction_metadata["topic_value"],
            is_query_like=extraction_metadata["is_query_like"],
            is_ack_like=extraction_metadata["is_ack_like"],
            raw_metadata={
                "pos_counts": pos_counts,
                "token_count": len(doc),
                "entities": entity_details,
                "signals": self._signals_to_metadata(extraction.signals),
                **extraction_metadata,
            },
            signals=extraction.signals,
        )


    def _calculate_information_density_score(
        self,
        doc: spacy.tokens.Doc,
    ) -> float:
        """Compute ID as the ratio of semantic tokens to total tokens.

        Returns 0.0 for empty documents to avoid division-by-zero; this
        neutral weight is safe to pass downstream to the Scoring Engine.
        """
        total_token_count = len(doc)
        if total_token_count == 0:
            return NEUTRAL_INFO_DENSITY

        semantic_token_count = sum(
            1 for token in doc if token.pos_ in SEMANTIC_POS_TAGS
        )
        return semantic_token_count / total_token_count

    def _calculate_absolute_sentiment_score(self, text: str) -> float:
        """Compute |S| as the absolute value of VADER's compound score.

        Returns 0.0 for blank text — VADER returns 0.0 compound for empty
        strings anyway, but the explicit guard documents intent clearly.
        """
        if not text.strip():
            return NEUTRAL_SENTIMENT_ABS

        vader_scores = self._sentiment_analyzer.polarity_scores(text)
        return abs(vader_scores["compound"])

    def _calculate_entity_count(self, doc: spacy.tokens.Doc) -> int:
        """Count the total number of named entities found by spaCy NER."""
        return len(doc.ents)


    def _extract_pos_counts(
        self,
        doc: spacy.tokens.Doc,
    ) -> dict[str, int]:
        """Build a frequency map of universal POS tags for the document.

        Stored in raw_metadata for audit and offline analysis — allows
        reconstruction of the ID calculation without re-running spaCy.
        """
        pos_counts: dict[str, int] = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        return pos_counts

    def _extract_entity_details(
        self,
        doc: spacy.tokens.Doc,
    ) -> list[dict[str, str]]:
        """Extract entity text and label for every NER span in the document.

        Each entry is {"text": <surface form>, "label": <NER type>}.
        Stored in raw_metadata so the entity list is auditable without
        re-running the spaCy pipeline.
        """
        return [
            {"text": entity.text, "label": entity.label_}
            for entity in doc.ents
        ]

    def _signals_to_metadata(self, signals: InteractionSignals) -> dict[str, object]:
        """Return a JSON-friendly metadata snapshot of the interaction signals."""
        return {
            "is_current_state": signals.is_current_state,
            "is_past_state": signals.is_past_state,
            "is_preference": signals.is_preference,
            "is_constraint": signals.is_constraint,
            "is_correction": signals.is_correction,
            "has_negation": signals.has_negation,
            "has_replacement": signals.has_replacement,
            "operational_weight": signals.operational_weight,
            "personal_relevance": signals.personal_relevance,
            "quantitative_relevance": signals.quantitative_relevance,
            "task_relevance": signals.task_relevance,
            "temporal_markers": signals.temporal_markers,
            "cue_phrases": signals.cue_phrases,
        }


    def _build_skipped_system_prompt_report(self) -> AnalysisReport:
        """Return a zero-weight report for system prompts that are not analysed.

        Using explicit neutral constants keeps the Scoring Engine's formula
        numerically stable — multiplying by 0.0 ID/sentiment won't produce
        NaN or unexpectedly high scores.
        """
        empty_signals = InteractionSignals()
        return AnalysisReport(
            info_density=NEUTRAL_INFO_DENSITY,
            sentiment_abs=NEUTRAL_SENTIMENT_ABS,
            entity_count=NEUTRAL_ENTITY_COUNT,
            is_system_prompt=True,
            latency_ms=0.0,
            signals=empty_signals,
            topic_identity=None,
            topic_value=None,
            is_query_like=False,
            is_ack_like=False,
            raw_metadata={
                "skipped": True,
                "reason": "analyze_system_prompt=False in NLPConfig",
                "signals": self._signals_to_metadata(empty_signals),
                "signal_evidence": [],
                "topic_identity": None,
                "topic_value": None,
                "is_query_like": False,
                "is_ack_like": False,
            },
        )
