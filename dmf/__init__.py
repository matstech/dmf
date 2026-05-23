"""Deterministic Memory Framework package."""

from dmf.analysis import EmbeddingEngine, InteractionMatrix, NLPEngine, ScoringEngine
from dmf.memory import ChromaLTMHook, FileLTMHook, Memory, TemporalMemory
from dmf.models import AnalysisReport
from dmf.runtime import InteractionPipeline

# $versifyr:template={{ (printf "__version__ = \"%s\""  .version) }}$
__version__ = "0.1.0"
__author__ = "matstech"
__email__ = "matteo.stabile2@gmail.com"

__all__ = [
    "AnalysisReport",
    "ChromaLTMHook",
    "EmbeddingEngine",
    "FileLTMHook",
    "InteractionMatrix",
    "InteractionPipeline",
    "Memory",
    "NLPEngine",
    "ScoringEngine",
    "TemporalMemory",
]
