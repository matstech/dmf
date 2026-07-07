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

"""Opt-in roundtrip tests against the Chroma server from compose.chroma.yml."""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest

from dmf.memory.ltm_hooks import ChromaLTMHook
from dmf.memory.ltm_hooks.chroma_client import (
    ChromaConnectionConfig,
    ChromaConnectionMode,
)
from dmf.models.analysis import AnalysisReport
from dmf.models.memory import MemoryEntry
from dmf.models.status import SurvivalStatus
from dmf.utils.constants import (
    DEFAULT_LTM_CHROMA_DATABASE,
    DEFAULT_LTM_CHROMA_TENANT,
)

pytestmark = pytest.mark.integration


def _embedding(_: str) -> np.ndarray:
    """Return a fixed vector without loading or downloading an embedding model."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _entry() -> MemoryEntry:
    report = AnalysisReport(
        info_density=0.8,
        sentiment_abs=0.0,
        entity_count=1,
        is_system_prompt=False,
        latency_ms=0.0,
        survival_score=0.9,
        status=SurvivalStatus.HEALTHY,
    )
    return MemoryEntry(
        interaction_id=1,
        text="The integration test stores a deterministic Chroma record.",
        report=report,
        vector=_embedding(""),
        token_count=8,
        timestamp=1_700_000_000.0,
    )


def test_chroma_server_roundtrip() -> None:
    """Bootstrap a server client and exercise the complete raw-record lifecycle."""
    connection = ChromaConnectionConfig(
        mode=ChromaConnectionMode.SERVER,
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
        tenant=DEFAULT_LTM_CHROMA_TENANT,
        database=DEFAULT_LTM_CHROMA_DATABASE,
    )
    hook = ChromaLTMHook(
        collection_name=f"dmf_integration_{uuid4().hex}",
        distance_threshold=0.01,
        embed_text=_embedding,
        connection=connection,
    )

    try:
        assert hook.count() == 0

        entry = _entry()
        hook.archive(entry)

        assert hook.count() == 1
        assert [record.record_id for record in hook.read_all()] == ["record:1"]

        hits = hook.search_raw([1.0, 0.0, 0.0], k=1)
        assert len(hits) == 1
        assert hits[0].record.record_id == "record:1"
        assert hits[0].record.text == entry.text

        hook.clear()
        assert hook.count() == 0
        assert hook.read_all() == []
    finally:
        hook.clear()
