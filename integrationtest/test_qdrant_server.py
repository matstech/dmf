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

"""Opt-in multi-client roundtrip test for compose.qdrant.yml."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from dmf.memory.ltm_hooks import QdrantLTMHook
from dmf.memory.ltm_hooks.qdrant_client import (
    QdrantConnectionConfig,
    QdrantConnectionMode,
    build_qdrant_client,
)
from dmf.models.analysis import AnalysisReport, InteractionSignals
from dmf.models.memory import MemoryEntry
from dmf.models.status import SurvivalStatus
from dmf.utils.config import VectorConfig

pytestmark = pytest.mark.integration

_VECTOR_CONFIG = VectorConfig(vector_dim=3)


def _embedding(_: str) -> np.ndarray:
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
        signals=InteractionSignals(is_preference=True),
        topic_identity="preference|prefer",
        topic_value="green tea",
    )
    return MemoryEntry(
        interaction_id=1,
        text="I prefer green tea.",
        report=report,
        vector=_embedding(""),
        token_count=4,
        timestamp=1_700_000_000.0,
    )


def test_qdrant_server_persists_across_clients(tmp_path: Path) -> None:
    suffix = uuid4().hex
    raw_collection = f"dmf_integration_raw_{suffix}"
    cards_collection = f"dmf_integration_cards_{suffix}"
    connection = QdrantConnectionConfig(
        mode=QdrantConnectionMode.SERVER,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    client_a = build_qdrant_client(connection)
    client_b = build_qdrant_client(connection)

    try:
        hook_a = QdrantLTMHook(
            collection_name=raw_collection,
            cards_collection_name=cards_collection,
            distance_threshold=0.01,
            vector_config=_VECTOR_CONFIG,
            embed_text=_embedding,
            cards_enabled=True,
            cards_path=tmp_path / "cards.jsonl",
            client=client_a,
        )
        entry = _entry()
        hook_a.archive(entry)

        hook_b = QdrantLTMHook(
            collection_name=raw_collection,
            cards_collection_name=cards_collection,
            distance_threshold=0.01,
            vector_config=_VECTOR_CONFIG,
            embed_text=_embedding,
            cards_enabled=True,
            cards_path=tmp_path / "cards-b.jsonl",
            client=client_b,
        )

        assert hook_b.count() == 1
        assert hook_b.count_cards() == 1
        assert [record.record_id for record in hook_b.read_all()] == ["record:1"]
        assert hook_b.search_raw([1.0, 0.0, 0.0], k=1)[0].record.text == entry.text
        assert hook_b.search_cards([1.0, 0.0, 0.0], k=1)[0].record.text == entry.text

        hook_b.clear()
        assert hook_a.count() == 0
        assert hook_a.count_cards() == 0
    finally:
        for client in (client_a, client_b):
            for collection_name in (raw_collection, cards_collection):
                if client.collection_exists(collection_name):
                    client.delete_collection(collection_name)
            client.close()
