"""Conversational signal extraction helpers."""

from dmf.analysis.signals.base import SignalAdapter, SignalEvidence, SignalExtractionResult
from dmf.analysis.signals.english import EnglishSignalAdapter

__all__ = [
    "SignalAdapter",
    "SignalEvidence",
    "SignalExtractionResult",
    "EnglishSignalAdapter",
]
