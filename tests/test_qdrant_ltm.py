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

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from dmf.memory.ltm_hooks.qdrant_hook import (
    QdrantLTMHook,
    _card_point_id,
    _raw_point_id,
)
from dmf.models.analysis import AnalysisReport
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawRecallHit
from dmf.models.status import SurvivalStatus
from dmf.utils.config import VectorConfig


def _make_entry(
    interaction_id: int,
    text: str,
    vector: list[float],
    timestamp: float | None = None,
) -> MemoryEntry:
    report = AnalysisReport(
        info_density=0.7,
        sentiment_abs=0.1,
        entity_count=2,
        is_system_prompt=False,
        latency_ms=1.0,
        survival_score=0.85,
        status=SurvivalStatus.HEALTHY,
    )
    return MemoryEntry(
        interaction_id=interaction_id,
        text=text,
        report=report,
        vector=np.array(vector, dtype=np.float32),
        token_count=6,
        timestamp=float(interaction_id if timestamp is None else timestamp),
    )


def _hook(
    *,
    collection_name: str = "test_raw",
    distance_threshold: float = 1.0,
    client: QdrantClient | None = None,
) -> QdrantLTMHook:
    return QdrantLTMHook(
        collection_name=collection_name,
        distance_threshold=distance_threshold,
        vector_config=VectorConfig(vector_dim=2),
        embed_text=lambda text: np.array(_EMBEDDINGS[text], dtype=np.float32),
        client=client or QdrantClient(":memory:"),
    )


_EMBEDDINGS = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "edge": [0.8, 0.6],
    "negative": [-1.0, 0.0],
}


def test_constructor_creates_collection_with_cosine_vector_config() -> None:
    client = QdrantClient(":memory:")

    hook = _hook(collection_name="created", client=client)

    info = client.get_collection("created")
    assert hook._client is client
    assert info.config.params.vectors.size == 2
    assert info.config.params.vectors.distance == models.Distance.COSINE


def test_constructor_reuses_existing_collection_without_destruction() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="shared",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    point = models.PointStruct(
        id=_raw_point_id("existing"),
        vector=[1.0, 0.0],
        payload={"raw_record": _make_entry(1, "alpha", [1.0, 0.0]).to_raw_ltm_record().to_dict()},
    )
    client.upsert(collection_name="shared", points=[point], wait=True)

    _hook(collection_name="shared", client=client)

    assert client.count(collection_name="shared", exact=True).count == 1


@pytest.mark.parametrize(
    ("vectors_config", "match"),
    [
        (
            models.VectorParams(size=3, distance=models.Distance.COSINE),
            "size=3",
        ),
        (
            models.VectorParams(size=2, distance=models.Distance.DOT),
            "Distance.DOT|Dot",
        ),
        (
            {"named": models.VectorParams(size=2, distance=models.Distance.COSINE)},
            "named vectors",
        ),
    ],
)
def test_constructor_rejects_incompatible_collections(
    vectors_config: Any,
    match: str,
) -> None:
    client = QdrantClient(":memory:")
    client.create_collection(collection_name="bad", vectors_config=vectors_config)

    with pytest.raises(ValueError, match=match):
        _hook(collection_name="bad", client=client)

    assert client.collection_exists("bad")


def test_constructor_rejects_non_positive_vector_dimension() -> None:
    with pytest.raises(ValueError, match="vector_dim > 0"):
        QdrantLTMHook(
            collection_name="invalid_dim",
            vector_config=VectorConfig(vector_dim=0),
            client=QdrantClient(":memory:"),
        )


def test_point_ids_are_stable_and_separate_by_record_type() -> None:
    assert _raw_point_id("record:7") == _raw_point_id("record:7")
    assert _card_point_id("record:7") == _card_point_id("record:7")
    assert _raw_point_id("record:7") != _card_point_id("record:7")


def test_archive_is_idempotent_and_preserves_raw_payload_mapping() -> None:
    hook = _hook()
    entry = _make_entry(7, "alpha", [1.0, 0.0])

    hook.archive(entry)
    hook.archive(entry)

    assert hook.count() == 1
    records = hook.read_all()
    assert [record.record_id for record in records] == ["record:7"]
    assert records[0].text == "alpha"


def test_search_raw_returns_ranking_threshold_and_scores() -> None:
    hook = _hook(distance_threshold=0.4)
    hook.archive(_make_entry(1, "alpha", [1.0, 0.0]))
    hook.archive(_make_entry(2, "beta", [0.0, 1.0]))
    hook.archive(_make_entry(3, "edge", [0.8, 0.6]))

    hits = hook.search_raw([1.0, 0.0], k=3)

    assert [hit.record.text for hit in hits] == ["alpha", "edge"]
    assert [hit.rank_hint for hit in hits] == [0, 1]
    assert hits[0].similarity_score == pytest.approx(1.0)
    assert hits[0].distance == pytest.approx(0.0)
    assert hits[1].similarity_score == pytest.approx(0.8)
    assert hits[1].distance == pytest.approx(0.2)


def test_search_raw_includes_threshold_edge() -> None:
    hook = _hook(distance_threshold=0.2)
    hook.archive(_make_entry(3, "edge", [0.8, 0.6]))

    hits = hook.search_raw([1.0, 0.0], k=1)

    assert [hit.record.text for hit in hits] == ["edge"]
    assert hits[0].similarity_score == pytest.approx(0.8)


def test_search_raw_does_not_clamp_negative_score() -> None:
    hook = _hook(distance_threshold=2.5)
    hook.archive(_make_entry(4, "negative", [-1.0, 0.0]))

    hits = hook.search_raw([1.0, 0.0], k=1)

    assert hits == [
        RawRecallHit(
            record=hits[0].record,
            similarity_score=pytest.approx(-1.0),
            distance=pytest.approx(2.0),
            rank_hint=0,
            source="ltm_raw",
        )
    ]
    assert hits[0].record.text == "negative"


def test_search_raw_validates_query_dimension() -> None:
    hook = _hook()

    with pytest.raises(ValueError, match="query vector dimension mismatch"):
        hook.search_raw([1.0, 0.0, 0.0], k=1)


def test_search_raw_handles_empty_collection_and_non_positive_k() -> None:
    hook = _hook()
    hook._client.count = lambda **kwargs: pytest.fail("count should not be called")  # type: ignore[method-assign]
    hook._client.query_points = lambda **kwargs: pytest.fail("query should not be called")  # type: ignore[method-assign]

    assert hook.search_raw([1.0, 0.0], k=0) == []

    hook = _hook()
    assert hook.search_raw([1.0, 0.0], k=3) == []


def test_archive_validates_embedding_dimension_before_upsert() -> None:
    hook = QdrantLTMHook(
        collection_name="bad_embedding",
        vector_config=VectorConfig(vector_dim=2),
        embed_text=lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        client=QdrantClient(":memory:"),
    )

    with pytest.raises(ValueError, match="raw record vector dimension mismatch"):
        hook.archive(_make_entry(1, "alpha", [1.0, 0.0]))

    assert hook.count() == 0


def test_search_raw_skips_malformed_payloads() -> None:
    client = QdrantClient(":memory:")
    hook = _hook(client=client)
    hook.archive(_make_entry(1, "alpha", [1.0, 0.0]))
    client.upsert(
        collection_name="test_raw",
        points=[
            models.PointStruct(
                id=_raw_point_id("malformed"),
                vector=[0.9, 0.1],
                payload={"raw_record": {"record_id": "malformed"}},
            )
        ],
        wait=True,
    )

    hits = hook.search_raw([1.0, 0.0], k=5)

    assert [hit.record.record_id for hit in hits] == ["record:1"]


def test_read_all_uses_pages_and_orders_records() -> None:
    hook = _hook()
    for entry in [
        _make_entry(300, "alpha", [1.0, 0.0]),
        _make_entry(1, "beta", [0.0, 1.0]),
        _make_entry(2, "edge", [0.8, 0.6]),
    ]:
        hook.archive(entry)

    records = hook.read_all()

    assert [record.interaction_id for record in records] == [1, 2, 300]


def test_count_and_clear_preserve_collection() -> None:
    hook = _hook()
    hook.archive(_make_entry(1, "alpha", [1.0, 0.0]))

    assert hook.count() == 1
    hook.clear()

    assert hook.count() == 0
    assert hook._client.collection_exists("test_raw")


def test_backend_errors_are_propagated() -> None:
    hook = _hook()

    def fail_query(**kwargs: Any) -> object:
        raise RuntimeError("backend failed")

    hook._client.query_points = fail_query  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="backend failed"):
        hook.search_raw([1.0, 0.0], k=1)
