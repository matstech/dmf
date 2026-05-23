"""Pipeline coordinator for interaction analysis.

InteractionPipeline is the single entry point for processing a conversation
interaction through the full DMF feature-extraction pipeline. It coordinates
three independent components without coupling them to each other:

    NLPEngine           → ID, |S|, E
    EmbeddingEngine     → unit vector
    InteractionMatrix   → D

This coordinator keeps each component focused on its own responsibility while
providing callers with a single, fully-enriched ``AnalysisReport`` that carries
all signals required by the scoring engine.

Execution flow for each interaction
------------------------------------
1. NLPEngine.analyze_interaction() — extracts ID, |S|, E, latency_ms.
2. Early-return if the interaction was gated (system prompt with
   analyze_system_prompt=False): semantic_divergence stays at 0.0.
3. EmbeddingEngine.get_embedding() — produces an L2-normalised vector.
   The model is loaded lazily on the first call.
4. InteractionMatrix.add_vector() — computes D against the previous
   L2-normalised centroid, then updates the sliding window.
5. dataclasses.replace() — stamps semantic_divergence onto the report
   without mutating the original NLPEngine output.

CPU-First Guarantee
-------------------
All three components operate exclusively on CPU:
  - spaCy en_core_web_sm and VADER are CPU-only by design.
  - FastEmbed TextEmbedding defaults to CPU inference (no GPU required).
  - numpy operations are single-process CPU arithmetic.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from dmf.analysis.embedding_engine import EmbeddingEngine
from dmf.analysis.geometry import InteractionMatrix
from dmf.models.analysis import AnalysisReport, InteractionProvenance
from dmf.analysis.nlp_engine import NLPEngine
from dmf.utils.config import NLPConfig, VectorConfig
from dmf.utils.config_loader import DMFConfig



class InteractionPipeline:
    """Full-pipeline coordinator for a single conversation session.
    
        Owns one instance each of NLPEngine, EmbeddingEngine, and
        InteractionMatrix for the lifetime of the session. Callers interact
        only through analyze_interaction(); the internal routing is opaque.
    
        Attributes
        ----------
        _nlp_config : NLPConfig
            Immutable NLP configuration forwarded to NLPEngine.
        _vector_config : VectorConfig
            Immutable vector configuration forwarded to EmbeddingEngine and
            InteractionMatrix.
        _nlp_engine : NLPEngine
            Extracts ID, |S|, E from raw text via spaCy + VADER.
        _embedding_engine : EmbeddingEngine
            Converts text to an L2-normalised dense vector.
            FastEmbed model is loaded lazily on the first get_embedding() call.
        _interaction_matrix : InteractionMatrix
            Maintains the sliding-window centroid and computes D.
    
    Args:
        nlp_config: See the function signature and surrounding type hints.
        vector_config: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(
        self,
        nlp_config: NLPConfig,
        vector_config: VectorConfig,
    ) -> None:
        """Initialise the pipeline by constructing all component engines.

        All three components are constructed here. Only NLPEngine triggers I/O
        at construction time (loading the spaCy model). EmbeddingEngine defers
        FastEmbed loading to the first get_embedding() call.

        Parameters
        ----------
        nlp_config : NLPConfig
            Configuration for the NLP Engine (spaCy model, system-prompt
            gating flag).
        vector_config : VectorConfig
            Configuration for the Embedding Engine and InteractionMatrix
            (FastEmbed model, cache directory, window size).
        """
        self._nlp_config: NLPConfig = nlp_config
        self._vector_config: VectorConfig = vector_config

        self._nlp_engine: NLPEngine = NLPEngine(config=nlp_config)
        self._embedding_engine: EmbeddingEngine = EmbeddingEngine(config=vector_config)
        self._interaction_matrix: InteractionMatrix = InteractionMatrix(config=vector_config)

    @classmethod
    def from_dmf_config(
        cls,
        config: DMFConfig,
        *,
        analyze_system_prompt: bool | None = None,
    ) -> InteractionPipeline:
        """Construct an ``InteractionPipeline`` from the universal ``DMFConfig``.
        
                Parameters
                ----------
                config : DMFConfig
                    Fully populated config object returned by ``load_dmf_config()``.
                analyze_system_prompt : bool | None
                    Optional override for ``NLPConfig.analyze_system_prompt``.
                    When ``None``, the ``NLPConfig`` default is preserved.
        
        Args:
            config: See the function signature and surrounding type hints.
            analyze_system_prompt: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        nlp_kwargs: dict[str, object] = {"spacy_model": config.nlp.spacy_model}
        if analyze_system_prompt is not None:
            nlp_kwargs["analyze_system_prompt"] = analyze_system_prompt

        nlp_cfg = NLPConfig(**nlp_kwargs)
        vector_cfg = VectorConfig(
            model_name=config.nlp.model_name,
            vector_dim=config.nlp.vector_dim,
            window_size=config.capacity.window_size,
        )
        return cls(
            nlp_config=nlp_cfg,
            vector_config=vector_cfg,
        )


    def analyze_interaction(
        self,
        text: str,
        is_system: bool = False,
        provenance: InteractionProvenance | None = None,
    ) -> AnalysisReport:
        """Process *text* through the full pipeline and return an enriched report.
        
                Steps
                -----
                1. NLP extraction (always runs, except for gated system prompts).
                2. Early return if the interaction was skipped by NLPEngine gating:
                   ``semantic_divergence`` remains at its default of 0.0.
                3. Embedding: convert text to a unit vector via FastEmbed.
                4. Geometry: compute divergence D and update the sliding window.
                5. Stamp ``semantic_divergence`` onto the report and return.
        
                Parameters
                ----------
                text : str
                    Raw interaction text (user message, assistant turn, etc.).
                is_system : bool
                    True when processing the system prompt. Combined with
                    ``NLPConfig.analyze_system_prompt``, controls whether the
                    system prompt is gated or fully analysed.
                provenance : InteractionProvenance | None
                    Optional structured provenance metadata supplied by the caller.
                    The pipeline copies it onto the returned report without adding
                    adapter-specific heuristics.
        
                Returns
                -------
                AnalysisReport
                    Fully enriched report with ID, |S|, E, D, latency_ms, and
                    raw_metadata populated.
        
        Args:
            text: See the function signature and surrounding type hints.
            is_system: See the function signature and surrounding type hints.
            provenance: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        report, _ = self.analyze_interaction_with_vector(
            text,
            is_system=is_system,
            provenance=provenance,
        )
        return report

    def analyze_interaction_with_vector(
        self,
        text: str,
        is_system: bool = False,
        provenance: InteractionProvenance | None = None,
    ) -> tuple[AnalysisReport, np.ndarray | None]:
        """Process *text* and also return the canonical embedding used.
        
                This is the public boundary for callers that need both the enriched
                ``AnalysisReport`` and the exact vector stamped into the session
                geometry. It avoids re-embedding the same text out-of-band.
        
                Returns
                -------
                tuple[AnalysisReport, np.ndarray | None]
                    ``(report, vector)`` for normal turns; ``vector`` is ``None``
                    when the interaction was skipped by system-prompt gating.
        
        Args:
            text: See the function signature and surrounding type hints.
            is_system: See the function signature and surrounding type hints.
            provenance: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        report = self._nlp_engine.analyze_interaction(text, is_system=is_system)

        if report.raw_metadata.get("skipped"):
            return (
                dataclasses.replace(
                    report,
                    provenance=provenance or InteractionProvenance(),
                ),
                None,
            )

        vector = self._embedding_engine.get_embedding(text)
        divergence = self._interaction_matrix.add_vector(vector)

        return (
            dataclasses.replace(
            report,
            semantic_divergence=divergence,
            provenance=provenance or InteractionProvenance(),
            ),
            vector,
        )
