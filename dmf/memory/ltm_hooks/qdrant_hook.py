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

"""Raw-record Qdrant LTM adapter."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from dmf.memory.card_projection import MemoryCardProjector
from dmf.memory.card_store import JsonlMemoryCardStore
from dmf.memory.ltm_hooks.codecs import (
    build_card_payload,
    build_raw_payload,
    raw_record_from_payload,
)
from dmf.memory.ltm_hooks.qdrant_client import (
    QdrantConnectionConfig,
    build_qdrant_client,
)
from dmf.memory.ltm_hooks.vector_types import (
    cosine_similarity_to_distance,
    distance_threshold_to_min_similarity,
    validate_vector_dimension,
)
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit
from dmf.utils.config import VectorConfig
from dmf.utils.constants import (
    DEFAULT_LTM_CARDS_COLLECTION_NAME,
    DEFAULT_LTM_CARDS_PATH,
    DEFAULT_LTM_COLLECTION_NAME,
    DEFAULT_LTM_DISTANCE_THRESHOLD,
)

if TYPE_CHECKING:
    import numpy as np

_QDRANT_POINT_NAMESPACE = uuid.UUID("42e1c69e-6a1d-5f58-a650-fbd3d8a7466f")


def _raw_point_id(record_id: str) -> str:
    """Return the deterministic Qdrant point UUID for a raw record."""
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, f"raw:{record_id}"))


def _card_point_id(card_id: str) -> str:
    """Return the deterministic Qdrant point UUID for a projected card."""
    return str(uuid.uuid5(_QDRANT_POINT_NAMESPACE, f"card:{card_id}"))


# Point IDs use a stable UUIDv5 namespace to keep archive upserts idempotent
# and to keep raw records distinct from projected cards.
class QdrantLTMHook:
    """In-memory Qdrant vector LTM store with raw-record retrieval."""

    def __init__(
        self,
        collection_name: str = DEFAULT_LTM_COLLECTION_NAME,
        distance_threshold: float = DEFAULT_LTM_DISTANCE_THRESHOLD,
        vector_config: VectorConfig | None = None,
        embed_text: Callable[[str], np.ndarray] | None = None,
        cards_enabled: bool = False,
        cards_path: Path | str | None = None,
        card_store: JsonlMemoryCardStore | None = None,
        cards_collection_name: str = DEFAULT_LTM_CARDS_COLLECTION_NAME,
        connection: QdrantConnectionConfig | None = None,
        client: object | None = None,
    ) -> None:
        if cards_enabled and cards_collection_name == collection_name:
            raise ValueError("Qdrant raw and card collections must use distinct names")

        self._collection_name = collection_name
        self._cards_collection_name = cards_collection_name
        self._distance_threshold = distance_threshold
        self._lock = threading.Lock()
        self._vector_config = vector_config or VectorConfig()
        self._embed_text = embed_text
        self._embedding_engine = None
        self._cards_enabled = cards_enabled
        self._card_store = card_store
        if self._card_store is None and cards_enabled:
            self._card_store = JsonlMemoryCardStore(cards_path or DEFAULT_LTM_CARDS_PATH)

        if self._vector_config.vector_dim <= 0:
            raise ValueError(
                f"Qdrant collection {collection_name!r} expected vector_dim > 0, "
                f"observed {self._vector_config.vector_dim}"
            )

        self._client = client if client is not None else build_qdrant_client(
            connection or QdrantConnectionConfig()
        )
        self._ensure_collection(self._collection_name)
        if cards_enabled:
            self._ensure_collection(self._cards_collection_name)
        self._card_projector = MemoryCardProjector()

    def archive(self, entry: MemoryEntry) -> None:
        """Index one evicted raw interaction record into Qdrant."""
        models = _qdrant_models()
        raw_record = entry.to_raw_ltm_record()
        vector = self._embed_text_payload(raw_record.text)
        validate_vector_dimension(
            vector,
            self._vector_config.vector_dim,
            field="raw record",
        )
        point = models.PointStruct(
            id=_raw_point_id(raw_record.record_id),
            vector=vector,
            payload=build_raw_payload(raw_record),
        )
        card_points = []
        if self._cards_enabled:
            source_vector = entry.vector.tolist()
            validate_vector_dimension(
                source_vector,
                self._vector_config.vector_dim,
                field="card source",
            )
            for card in self._card_projector.project(entry):
                card_points.append(
                    models.PointStruct(
                        id=_card_point_id(card.card_id),
                        vector=source_vector,
                        payload=build_card_payload(card),
                    )
                )

        with self._lock:
            self._client.upsert(
                collection_name=self._collection_name,
                points=[point],
                wait=True,
            )
            if card_points:
                self._client.upsert(
                    collection_name=self._cards_collection_name,
                    points=card_points,
                    wait=True,
                )
        if self._card_store is not None:
            self._card_store.archive(entry)

    def search_raw(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[RawRecallHit]:
        """Retrieve top-k raw records by Qdrant cosine similarity."""
        if k <= 0:
            return []

        validate_vector_dimension(
            query_vector,
            self._vector_config.vector_dim,
            field="query",
        )
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
            score_threshold=distance_threshold_to_min_similarity(
                self._distance_threshold
            ),
        )

        hits: list[RawRecallHit] = []
        for idx, point in enumerate(response.points):
            payload = point.payload
            if not isinstance(payload, Mapping):
                continue
            try:
                record = raw_record_from_payload(payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            score = float(point.score)
            hits.append(
                RawRecallHit(
                    record=record,
                    similarity_score=score,
                    distance=cosine_similarity_to_distance(score),
                    rank_hint=idx,
                )
            )
        return hits

    def search_cards(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[RawRecallHit]:
        """Retrieve source raw records for the top-k matching projected cards."""
        if not self._cards_enabled:
            return []
        if k <= 0:
            return []

        validate_vector_dimension(
            query_vector,
            self._vector_config.vector_dim,
            field="query",
        )
        response = self._client.query_points(
            collection_name=self._cards_collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
            score_threshold=distance_threshold_to_min_similarity(
                self._distance_threshold
            ),
        )

        valid_candidates: list[tuple[int, str, float]] = []
        for idx, point in enumerate(response.points):
            payload = point.payload
            if not isinstance(payload, Mapping):
                continue
            source_record_id = payload.get("source_record_id")
            if not isinstance(source_record_id, str) or not source_record_id:
                continue
            valid_candidates.append((idx, source_record_id, float(point.score)))

        if not valid_candidates:
            return []

        source_ids = list(
            dict.fromkeys(source_id for _, source_id, _ in valid_candidates)
        )
        raw_points = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[_raw_point_id(source_id) for source_id in source_ids],
            with_payload=True,
            with_vectors=False,
        )
        records_by_id: dict[str, RawLTMRecord] = {}
        for point in raw_points:
            payload = point.payload
            if not isinstance(payload, Mapping):
                continue
            try:
                record = raw_record_from_payload(payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            records_by_id[record.record_id] = record

        hits: list[RawRecallHit] = []
        for idx, source_id, score in valid_candidates:
            record = records_by_id.get(source_id)
            if record is None:
                continue
            hits.append(
                RawRecallHit(
                    record=record,
                    similarity_score=score,
                    distance=cosine_similarity_to_distance(score),
                    rank_hint=idx,
                )
            )
        return hits

    def count_cards(self) -> int:
        """Return the number of indexed card records, or zero when disabled."""
        if not self._cards_enabled:
            return 0
        return int(
            self._client.count(
                collection_name=self._cards_collection_name,
                exact=True,
            ).count
        )

    def read_all(self) -> list[RawLTMRecord]:
        """Return all archived raw records ordered by source identity."""
        records: list[RawLTMRecord] = []
        offset = None
        while True:
            page, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in page:
                payload = point.payload
                if not isinstance(payload, Mapping):
                    continue
                try:
                    records.append(raw_record_from_payload(payload))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            if next_offset is None:
                break
            offset = next_offset

        records.sort(key=lambda record: (record.interaction_id, record.record_id))
        return records

    def count(self) -> int:
        """Return the number of indexed raw records in the collection."""
        return int(
            self._client.count(
                collection_name=self._collection_name,
                exact=True,
            ).count
        )

    def clear(self) -> None:
        """Delete all raw points while preserving the Qdrant collection."""
        models = _qdrant_models()
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(filter=models.Filter()),
            wait=True,
        )

    @property
    def card_store(self) -> JsonlMemoryCardStore | None:
        """Auxiliary JSONL card audit store, when configured."""
        return self._card_store

    def _ensure_collection(self, collection_name: str) -> None:
        models = _qdrant_models()
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_config.vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )
        self._validate_collection(collection_name)

    def _validate_collection(self, collection_name: str) -> None:
        models = _qdrant_models()
        collection = self._client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        expected = (
            f"single vector size={self._vector_config.vector_dim}, "
            f"distance={models.Distance.COSINE}"
        )

        if isinstance(vectors, Mapping):
            raise ValueError(
                f"Qdrant collection {collection_name!r} is incompatible: "
                f"expected {expected}, observed named vectors {vectors!r}"
            )

        observed_size = getattr(vectors, "size", None)
        observed_distance = getattr(vectors, "distance", None)
        if (
            observed_size != self._vector_config.vector_dim
            or observed_distance != models.Distance.COSINE
        ):
            observed = f"size={observed_size}, distance={observed_distance}"
            raise ValueError(
                f"Qdrant collection {collection_name!r} is incompatible: "
                f"expected {expected}, observed {observed}"
            )

    def _embed_text_payload(self, text: str) -> list[float]:
        """Return the vector used to index one raw record."""
        vector = self._get_embedder()(text)
        return vector.tolist()

    def _get_embedder(self) -> Callable[[str], np.ndarray]:
        """Return the text embedder, lazily initialising the default engine."""
        if self._embed_text is not None:
            return self._embed_text

        if self._embedding_engine is None:
            from dmf.analysis.embedding_engine import EmbeddingEngine  # noqa: PLC0415

            self._embedding_engine = EmbeddingEngine(self._vector_config)

        return self._embedding_engine.get_embedding


def _qdrant_models() -> object:
    """Import Qdrant models lazily so module import does not require the extra."""
    from qdrant_client import models  # noqa: PLC0415

    return models


__all__ = ["QdrantLTMHook"]
