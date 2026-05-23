"""Core contracts for conversational signal extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from spacy.tokens import Doc

from dmf.models.analysis import InteractionSignals

SignalEvidence = dict[str, Any]


@dataclass(frozen=True)
class SignalExtractionResult:
    """Canonical output of a language-specific signal extraction pass.
    
        The public shape stays flat because downstream callers already consume
        these fields directly. Internally, callers should treat the payload as
        three semantic groups:
    
        - conversational `signals`
        - pragmatic flags (`is_query_like`, `is_ack_like`)
        - topic fields (`topic_identity`, `topic_value`)
    
    Args:
        signals: See the function signature and surrounding type hints.
        signal_evidence: See the function signature and surrounding type hints.
        topic_identity: See the function signature and surrounding type hints.
        topic_value: See the function signature and surrounding type hints.
        is_query_like: See the function signature and surrounding type hints.
        is_ack_like: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    signals: InteractionSignals
    signal_evidence: list[SignalEvidence]
    topic_identity: str | None = None
    topic_value: str | None = None
    is_query_like: bool = False
    is_ack_like: bool = False

    def pragmatic_flags(self) -> dict[str, bool]:
        """Return the pragmatic flags consumed by memory policy.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "is_query_like": self.is_query_like,
            "is_ack_like": self.is_ack_like,
        }

    def topic_fields(self) -> dict[str, str | None]:
        """Return the extracted topic payload as a small semantic group.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "topic_identity": self.topic_identity,
            "topic_value": self.topic_value,
        }

    def metadata_fields(self) -> dict[str, Any]:
        """Return the flat extraction metadata stored in `AnalysisReport.raw_metadata`.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            **self.topic_fields(),
            **self.pragmatic_flags(),
            "signal_evidence": self.signal_evidence,
        }


class SignalAdapter(Protocol):
    """Language-specific extractor for conversational/pragmatic signals.
    
    Args:
        None.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def extract(
        self,
        text: str,
        doc: Doc,
    ) -> SignalExtractionResult:
        """Return normalised signals, evidence, and optional topic fields.
        
        Args:
            text: See the function signature and surrounding type hints.
            doc: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
