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

"""Memory layer: active-memory lifecycle and LTM backends."""

from dmf.memory.api import Memory
from dmf.memory.answerability_rerank import (
    AnswerabilityDiagnostics,
    AnswerabilityFeatureExtractor,
    AnswerabilityRanker,
    AnswerabilityRankerConfig,
    rerank_answerable_evidence,
)
from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.memory.card_store import JsonlMemoryCardStore
from dmf.memory.candidate_generation import (
    CandidateGenerationConfig,
    CandidateGenerator,
    CandidatePool,
    CardSemanticRetriever,
    CardSymbolicRetriever,
    DeterministicCardSemanticRetriever,
    HardFilterContext,
    RawLexicalRetriever,
    RawSemanticRetriever,
)
from dmf.memory.evidence_assembly import (
    EvidenceAssemblyConfig,
    RawEvidenceSource,
    apply_final_cutoff,
    assemble_final_evidence,
    expand_card_evidence,
    render_evidence_context,
)
from dmf.memory.ltm_engine import FileLTMHook
from dmf.memory.query_understanding import QueryUnderstandingParser, parse_query_frame
from dmf.memory.temporal_memory import TemporalMemory

__all__ = [
    "Memory",
    "AnswerabilityDiagnostics",
    "AnswerabilityFeatureExtractor",
    "AnswerabilityRanker",
    "AnswerabilityRankerConfig",
    "rerank_answerable_evidence",
    "ChromaLTMHook",
    "JsonlMemoryCardStore",
    "CandidateGenerationConfig",
    "CandidateGenerator",
    "CandidatePool",
    "CardSemanticRetriever",
    "CardSymbolicRetriever",
    "DeterministicCardSemanticRetriever",
    "HardFilterContext",
    "RawLexicalRetriever",
    "RawSemanticRetriever",
    "EvidenceAssemblyConfig",
    "RawEvidenceSource",
    "apply_final_cutoff",
    "assemble_final_evidence",
    "expand_card_evidence",
    "render_evidence_context",
    "FileLTMHook",
    "QueryUnderstandingParser",
    "TemporalMemory",
    "parse_query_frame",
]
