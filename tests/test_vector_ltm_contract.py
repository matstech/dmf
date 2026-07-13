# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

"""Shared vector-LTM contract for Chroma and Qdrant backends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from dmf.memory.ltm_hooks.chroma_hook import ChromaLTMHook
from dmf.memory.ltm_hooks.qdrant_hook import QdrantLTMHook
from dmf.models.analysis import AnalysisReport, InteractionProvenance
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawRecallHit
from dmf.models.status import SurvivalStatus
from dmf.utils.config import VectorConfig


_EMBEDDINGS: dict[str, list[float]] = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "edge": [0.8, 0.6],
}


def _embed(text: str) -> np.ndarray:
    return np.array(_EMBEDDINGS[text], dtype=np.float32)


def _entry(interaction_id: int, text: str, vector: list[float]) -> MemoryEntry:
    return MemoryEntry(
        interaction_id=interaction_id,
        text=text,
        report=AnalysisReport(
            info_density=0.7,
            sentiment_abs=0.1,
            entity_count=1,
            is_system_prompt=False,
            latency_ms=0.0,
            survival_score=0.82,
            status=SurvivalStatus.HEALTHY,
            provenance=InteractionProvenance(role="user", source_turn=interaction_id),
        ),
        vector=np.array(vector, dtype=np.float32),
        token_count=1,
        timestamp=float(interaction_id),
    )


@pytest.fixture(params=["chroma", "qdrant"])
def vector_ltm_hook(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Callable[..., object]:
    backend = str(request.param)

    def build(*, distance_threshold: float = 1.0) -> object:
        vector_config = VectorConfig(vector_dim=2)
        if backend == "chroma":
            return ChromaLTMHook(
                collection_name=f"contract_{backend}",
                persist_directory=tmp_path / backend,
                distance_threshold=distance_threshold,
                vector_config=vector_config,
                embed_text=_embed,
            )
        return QdrantLTMHook(
            collection_name=f"contract_{backend}",
            distance_threshold=distance_threshold,
            vector_config=vector_config,
            embed_text=_embed,
            client=QdrantClient(":memory:"),
        )

    return build


def test_round_trips_raw_records(vector_ltm_hook: Callable[..., object]) -> None:
    hook = vector_ltm_hook()

    hook.archive(_entry(7, "alpha", [1.0, 0.0]))

    records = hook.read_all()
    assert [record.record_id for record in records] == ["record:7"]
    assert records[0].text == "alpha"
    assert records[0].role == "user"
    assert records[0].provenance.source_turn == 7


def test_archive_is_idempotent(vector_ltm_hook: Callable[..., object]) -> None:
    hook = vector_ltm_hook()
    entry = _entry(3, "alpha", [1.0, 0.0])

    hook.archive(entry)
    hook.archive(entry)

    assert hook.count() == 1
    assert [record.record_id for record in hook.read_all()] == ["record:3"]


def test_known_vector_ranking_and_threshold_edge(
    vector_ltm_hook: Callable[..., object],
) -> None:
    hook = vector_ltm_hook(distance_threshold=0.200001)
    hook.archive(_entry(1, "alpha", [1.0, 0.0]))
    hook.archive(_entry(2, "beta", [0.0, 1.0]))
    hook.archive(_entry(3, "edge", [0.8, 0.6]))

    hits = hook.search_raw([1.0, 0.0], k=3)

    assert [hit.record.record_id for hit in hits] == ["record:1", "record:3"]
    assert [hit.rank_hint for hit in hits] == [0, 1]
    assert hits[0].similarity_score == pytest.approx(1.0)
    assert hits[0].distance == pytest.approx(0.0)
    assert hits[1].similarity_score == pytest.approx(0.8, abs=1e-6)
    assert hits[1].distance == pytest.approx(0.2, abs=1e-6)


def test_raw_recall_hit_shape(vector_ltm_hook: Callable[..., object]) -> None:
    hook = vector_ltm_hook()
    hook.archive(_entry(1, "alpha", [1.0, 0.0]))

    hit = hook.search_raw([1.0, 0.0], k=1)[0]

    assert isinstance(hit, RawRecallHit)
    assert hit.source == "ltm_raw"
    assert hit.record.record_id == "record:1"
    assert isinstance(hit.similarity_score, float)
    assert isinstance(hit.distance, float)
    assert hit.rank_hint == 0


def test_read_all_count_and_clear_contract(
    vector_ltm_hook: Callable[..., object],
) -> None:
    hook = vector_ltm_hook()
    hook.archive(_entry(20, "beta", [0.0, 1.0]))
    hook.archive(_entry(10, "alpha", [1.0, 0.0]))
    hook.archive(_entry(30, "edge", [0.8, 0.6]))

    assert hook.count() == 3
    assert [record.interaction_id for record in hook.read_all()] == [10, 20, 30]

    hook.clear()

    assert hook.count() == 0
    assert hook.read_all() == []
