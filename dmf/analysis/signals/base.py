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
