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

"""Tests for Memory.retrieve() and Memory.render_context() structured stack methods."""

from __future__ import annotations

import numpy as np
import pytest

from dmf.memory.api import Memory
from dmf.models.memory import QueryFrame
from dmf.memory.temporal_memory import TemporalMemory
from dmf.models.analysis import AnalysisReport, InteractionProvenance, InteractionSignals
from dmf.models.ltm_hook import NullLTMHook
from dmf.utils.config_loader import DMFConfig


class _FakeEmbeddingEngine:
    """Minimal embedding engine that returns a zero vector without loading any model."""

    def get_embedding(self, text: str) -> np.ndarray:  # noqa: ARG002
        return np.zeros(768, dtype=float)


def _make_memory_legacy() -> Memory:
    """Build a legacy Memory (no structured stack)."""
    tm = TemporalMemory(ltm_hook=NullLTMHook())
    return Memory(tm, _FakeEmbeddingEngine())


def _make_memory_structured() -> Memory:
    """Build a Memory with the structured retrieval stack using all defaults."""
    tm = TemporalMemory(ltm_hook=NullLTMHook())
    config = DMFConfig()
    return Memory.from_dmf_config(config, tm, _FakeEmbeddingEngine())


def test_retrieve_raises_without_structured_stack() -> None:
    """Memory(tm, ee).retrieve() must raise RuntimeError when stack is absent."""
    memory = _make_memory_legacy()

    with pytest.raises(RuntimeError, match="Memory.retrieve()"):
        memory.retrieve("What did I say about my job?")


def test_render_context_raises_without_structured_stack() -> None:
    """Memory(tm, ee).render_context() must raise RuntimeError when stack is absent."""
    memory = _make_memory_legacy()

    with pytest.raises(RuntimeError, match="Memory.retrieve()"):
        memory.render_context("What did I say about my job?")


def test_retrieve_returns_empty_list_with_empty_store() -> None:
    """retrieve() returns [] when no records or cards exist in the store."""
    memory = _make_memory_structured()

    result = memory.retrieve("What is my favourite food?")

    assert isinstance(result, list)
    assert result == []


def test_retrieve_searches_visible_active_records() -> None:
    """Structured retrieve() must not wait for active entries to be LTM-archived."""
    memory = _make_memory_structured()
    report = AnalysisReport(
        info_density=0.5,
        sentiment_abs=0.0,
        entity_count=1,
        is_system_prompt=False,
        latency_ms=0.0,
        provenance=InteractionProvenance(role="alice", source_turn=1),
        signals=InteractionSignals(is_preference=True),
    )
    memory._temporal_memory.add_interaction(
        "I prefer green tea in the afternoon.",
        report,
        np.zeros(768, dtype=float),
    )

    result = memory.retrieve("What tea does Alice prefer?")

    assert [item.evidence_id for item in result] == ["record:0"]
    assert result[0].render_payload["record"]["text"] == (
        "I prefer green tea in the afternoon."
    )


def test_render_context_returns_empty_string_with_empty_store() -> None:
    """render_context() returns an empty string when the store is empty."""
    memory = _make_memory_structured()

    result = memory.render_context("What is my favourite food?")

    assert isinstance(result, str)
    assert result.strip() == ""


def test_retrieve_reuses_temporal_memory_analyzer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured retrieve() forwards the runtime NLP analyzer into query parsing."""
    memory = _make_memory_structured()
    analyzer = object()
    memory._temporal_memory._nlp_engine = analyzer
    observed: dict[str, object] = {}

    def fake_parse_query_frame(
        query_text: str,
        *,
        query_embedding=None,  # noqa: ANN001
        analyzer=None,  # noqa: ANN001
    ) -> QueryFrame:
        observed["query_text"] = query_text
        observed["query_embedding"] = query_embedding
        observed["analyzer"] = analyzer
        return QueryFrame(query_text=query_text)

    monkeypatch.setattr(
        "dmf.memory.query_understanding.parse_query_frame",
        fake_parse_query_frame,
    )

    result = memory.retrieve("What is my favourite food?")

    assert result == []
    assert observed["query_text"] == "What is my favourite food?"
    assert observed["query_embedding"] is None
    assert observed["analyzer"] is analyzer

