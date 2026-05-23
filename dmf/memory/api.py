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

"""Public memory facade exposed to agent/runtime callers."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from dmf.models.memory import RetrievedEvidence

if TYPE_CHECKING:
    from dmf.analysis.embedding_engine import EmbeddingEngine
    from dmf.memory.candidate_generation import CandidateGenerationConfig
    from dmf.memory.evidence_assembly import EvidenceAssemblyConfig
    from dmf.memory.temporal_memory import TemporalMemory
    from dmf.utils.config_loader import DMFConfig


class Memory:
    """Small public facade over the internal temporal/LTM memory runtime.

    Callers should use this wrapper instead of reaching into
    ``TemporalMemory`` directly. The facade speaks in application terms:
    text queries in, prompt context or recalled candidates out.

    Args:
        temporal_memory: Active-memory runtime that owns pruning, LTM hooks, and
            context rendering.
        embedding_engine: Embedding provider used to vectorize text queries.
        candidate_generation_config: Optional structured retrieval settings.
        assembly_config: Optional evidence assembly settings.

    Returns:
        Facade instance for runtime memory operations.

    Raises:
        None.
    """

    def __init__(
        self,
        temporal_memory: TemporalMemory,
        embedding_engine: EmbeddingEngine,
        candidate_generation_config: CandidateGenerationConfig | None = None,
        assembly_config: EvidenceAssemblyConfig | None = None,
    ) -> None:
        self._temporal_memory = temporal_memory
        self._embedding_engine = embedding_engine
        self._candidate_generation_config = candidate_generation_config
        self._assembly_config = assembly_config

    @classmethod
    def from_dmf_config(
        cls,
        config: DMFConfig,
        temporal_memory: TemporalMemory,
        embedding_engine: EmbeddingEngine,
    ) -> Memory:
        """Construct a ``Memory`` facade from a fully-parsed ``DMFConfig``.

        Args:
            config: Parsed DMF configuration tree.
            temporal_memory: Active-memory runtime to wrap.
            embedding_engine: Embedding provider used for query vectors.

        Returns:
            Fully wired memory facade with structured retrieval enabled.

        Raises:
            AttributeError: If the supplied config does not expose the expected
                ``retrieval`` settings.
        """
        from dmf.memory.candidate_generation import CandidateGenerationConfig
        from dmf.memory.evidence_assembly import EvidenceAssemblyConfig

        r = config.retrieval
        candidate_generation_config = CandidateGenerationConfig(
            card_prefetch_k=r.card_prefetch_k,
            raw_prefetch_k=r.raw_prefetch_k,
            symbolic_lookup_k=r.symbolic_lookup_k,
            enable_card_semantic=r.enable_card_semantic,
            enable_card_symbolic=r.enable_card_symbolic,
            enable_raw_semantic=r.enable_raw_semantic,
            enable_raw_lexical=r.enable_raw_lexical,
        )
        assembly_config = EvidenceAssemblyConfig(
            final_recall_limit=r.final_recall_limit,
            max_support_turns_per_card=r.max_support_turns_per_card,
            include_superseded_when_historical=r.include_superseded_when_historical,
            include_neighbor_turns=r.include_neighbor_turns,
        )
        return cls(
            temporal_memory,
            embedding_engine,
            candidate_generation_config=candidate_generation_config,
            assembly_config=assembly_config,
        )

    def retrieve(self, query_text: str) -> list[RetrievedEvidence]:
        """Return final retrieved evidence for one query text.

        Args:
            query_text: Text query to parse, embed, retrieve, and rerank.

        Returns:
            Final evidence after candidate generation, answerability reranking,
            and evidence assembly.

        Raises:
            RuntimeError: If the structured retrieval stack was not configured.
            Backend-specific exceptions may propagate from the configured LTM
            hook.
        """
        if self._candidate_generation_config is None or self._assembly_config is None:
            raise RuntimeError(
                "Memory.retrieve() requires structured stack components. "
                "Use Memory.from_dmf_config() to construct Memory with the new retrieval stack."
            )

        from dmf.memory.answerability_rerank import rerank_answerable_evidence
        from dmf.memory.candidate_generation import CandidateGenerator
        from dmf.memory.evidence_assembly import assemble_final_evidence
        from dmf.memory.query_understanding import parse_query_frame

        query_vector = self._embed_query_text(query_text)
        query_frame = parse_query_frame(
            query_text,
            analyzer=getattr(self._temporal_memory, "_nlp_engine", None),
        )
        query_frame = dataclasses.replace(
            query_frame,
            query_embedding=query_vector.tolist(),
        )

        ltm_hook = self._temporal_memory.ltm_hook
        active_raw_records = getattr(
            self._temporal_memory,
            "get_active_raw_records",
            lambda: [],
        )()
        raw_records = [
            *ltm_hook.read_all(),
            *active_raw_records,
        ]
        if hasattr(ltm_hook, "card_store") and ltm_hook.card_store is not None:
            cards = ltm_hook.card_store.read_all()
        else:
            cards = []

        generator = CandidateGenerator(
            cards=cards,
            ltm_hook=ltm_hook,
            raw_records=raw_records,
            config=self._candidate_generation_config,
        )
        pool = generator.generate(query_frame)
        reranked = rerank_answerable_evidence(query_frame, pool.candidates)
        return assemble_final_evidence(
            reranked,
            raw_records=raw_records,
            query=query_frame,
            config=self._assembly_config,
        )

    def render_context(self, query_text: str) -> str:
        """Return a rendered prompt-ready context string for one query.

        Args:
            query_text: Text query to retrieve and render.

        Returns:
            Prompt-ready structured evidence context.

        Raises:
            RuntimeError: If the structured retrieval stack was not configured.
            Backend-specific exceptions may propagate from the configured LTM
            hook.
        """
        from dmf.memory.evidence_assembly import render_evidence_context

        evidence = self.retrieve(query_text)
        return render_evidence_context(evidence)

    def _embed_query_text(self, query_text: str) -> np.ndarray:
        """Return the canonical DMF embedding for one query text."""
        return self._embedding_engine.get_embedding(query_text)
