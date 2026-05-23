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

"""Raw-record ChromaDB LTM adapter.

`ChromaLTMHook` persists evicted interactions as raw records and exposes a
CPU-only semantic-search surface through `search_raw()`. Conversational
interpretation is intentionally deferred to recall-time NLP in
`TemporalMemory`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import chromadb
from chromadb.config import Settings

from dmf.memory.card_projection import MemoryCardProjector
from dmf.memory.card_store import JsonlMemoryCardStore
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit
from dmf.utils.config import VectorConfig
from dmf.utils.constants import (
    DEFAULT_LTM_CARDS_COLLECTION_NAME,
    DEFAULT_LTM_CHROMA_PATH,
    DEFAULT_LTM_COLLECTION_NAME,
    DEFAULT_LTM_DISTANCE_THRESHOLD,
)

if TYPE_CHECKING:
    import numpy as np


class ChromaLTMHook:
    """Persistent vector LTM store with raw-record retrieval via ChromaDB.

    Args:
        collection_name: Chroma collection used for raw records.
        persist_directory: Directory where Chroma persists local state.
        distance_threshold: Maximum cosine distance accepted for recall hits.
        vector_config: Optional embedding configuration for lazy indexing.
        embed_text: Optional embedding function override.
        cards_enabled: Whether to index projected auxiliary cards.
        cards_path: Optional JSONL audit path for projected cards.
        card_store: Optional prebuilt JSONL card store.
        cards_collection_name: Chroma collection used for projected cards.

    Returns:
        Chroma-backed LTM hook instance.

    Raises:
        OSError: If the persistence directory cannot be created.
        ChromaDB exceptions may surface during client or collection creation.

    Warning:
        Chroma cosine distance is lower-is-better. The public
        ``similarity_score`` exposed by hits is derived as ``1.0 - distance``
        only for deterministic ordering diagnostics.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_LTM_COLLECTION_NAME,
        persist_directory: Path | str = DEFAULT_LTM_CHROMA_PATH,
        distance_threshold: float = DEFAULT_LTM_DISTANCE_THRESHOLD,
        vector_config: VectorConfig | None = None,
        embed_text: Callable[[str], np.ndarray] | None = None,
        cards_enabled: bool = False,
        cards_path: Path | str | None = None,
        card_store: JsonlMemoryCardStore | None = None,
        cards_collection_name: str = DEFAULT_LTM_CARDS_COLLECTION_NAME,
    ) -> None:
        self._distance_threshold = distance_threshold
        self._lock = threading.Lock()
        self._vector_config = vector_config or VectorConfig()
        self._embed_text = embed_text
        self._embedding_engine = None
        self._cards_enabled = cards_enabled
        self._card_store = card_store
        if self._card_store is None and cards_enabled:
            default_cards_path = Path(persist_directory) / "ltm_cards.jsonl"
            self._card_store = JsonlMemoryCardStore(cards_path or default_cards_path)

        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        self._persist_directory: Path = persist_path

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._cards_collection = None
        if cards_enabled:
            self._cards_collection = self._client.get_or_create_collection(
                name=cards_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self._card_projector = MemoryCardProjector()

    def archive(self, entry: MemoryEntry) -> None:
        """Index one evicted raw interaction record into ChromaDB.

        Args:
            entry: Working-memory entry selected for archival.

        Returns:
            None.

        Raises:
            ChromaDB exceptions may surface during upsert.
            TypeError: If raw-record or card metadata cannot be serialised.
        """
        raw_record = entry.to_raw_ltm_record()
        metadata = {
            "raw_record": json.dumps(raw_record.to_dict(), ensure_ascii=False),
            "record_id": raw_record.record_id,
            "raw_interaction_id": raw_record.interaction_id,
            "raw_role": raw_record.role,
            "raw_created_at": raw_record.created_at,
        }

        with self._lock:
            self._collection.upsert(
                ids=[raw_record.record_id],
                embeddings=[self._embed_text_payload(raw_record.text)],
                documents=[raw_record.text],
                metadatas=[metadata],
            )
            if getattr(self, "_cards_collection", None) is not None:
                cards = self._card_projector.project(entry)
                source_vector = entry.vector.tolist()
                for card in cards:
                    card_text = " ".join(
                        piece for piece in [card.kind, card.subject, card.predicate, card.object]
                        if piece
                    )
                    card_metadata = {
                        "card": json.dumps(card.to_dict(), ensure_ascii=False),
                        "card_id": card.card_id,
                        "source_record_id": card.provenance.source_record_id,
                        "kind": card.kind,
                    }
                    self._cards_collection.upsert(
                        ids=[card.card_id],
                        embeddings=[source_vector],
                        documents=[card_text],
                        metadatas=[card_metadata],
                    )
        if self._card_store is not None:
            self._card_store.archive(entry)

    def search_raw(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[RawRecallHit]:
        """Retrieve the top-k most relevant raw records by vector similarity.

        Args:
            query_vector: Query embedding in the same vector space as indexed
                raw records.
            k: Maximum number of raw hits requested.

        Returns:
            Raw hits under the configured distance threshold.

        Raises:
            ChromaDB exceptions may surface during query execution.
        """
        if k <= 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["metadatas", "distances"],
        )

        metadatas: list[dict[str, object]] = results["metadatas"][0]
        distances: list[float] = results["distances"][0]

        hits: list[RawRecallHit] = []
        for idx, (meta, dist) in enumerate(zip(metadatas, distances)):
            if dist > self._distance_threshold:
                continue
            try:
                record = self._deserialize_raw_record(meta)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            hits.append(
                RawRecallHit(
                    record=record,
                    distance=float(dist),
                    similarity_score=1.0 - float(dist),
                    rank_hint=idx,
                )
            )
        return hits

    def search_cards(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[RawRecallHit]:
        """Retrieve the top-k most relevant card hits by vector similarity.

        Each returned :class:`RawRecallHit` points to the *raw source record*
        that the matching card was projected from.  If the source record is no
        longer present in the main collection the card hit is silently skipped.

        Args:
            query_vector: Query embedding in the same vector space as indexed
                cards.
            k: Maximum number of card hits requested.

        Returns:
            Raw hits corresponding to the source records of matching cards.

        Raises:
            ChromaDB exceptions may surface during the initial card query.
        """
        if self._cards_collection is None:
            return []
        if k <= 0:
            return []

        results = self._cards_collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["metadatas", "distances"],
        )

        metadatas: list[dict[str, object]] = results["metadatas"][0]
        distances: list[float] = results["distances"][0]

        hits: list[RawRecallHit] = []
        for idx, (meta, dist) in enumerate(zip(metadatas, distances)):
            if dist > self._distance_threshold:
                continue
            source_record_id = meta.get("source_record_id")
            if not source_record_id:
                continue
            try:
                raw_results = self._collection.get(
                    ids=[str(source_record_id)],
                    include=["metadatas"],
                )
            except Exception:  # noqa: BLE001
                continue
            raw_metas = raw_results.get("metadatas") or []
            if not raw_metas:
                continue
            try:
                record = self._deserialize_raw_record(raw_metas[0])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            hits.append(
                RawRecallHit(
                    record=record,
                    distance=float(dist),
                    similarity_score=1.0 - float(dist),
                    rank_hint=idx,
                )
            )
        return hits

    def count_cards(self) -> int:
        """Return the number of indexed card records.

        Returns:
            Indexed card count, or 0 when cards are disabled.

        Raises:
            ChromaDB exceptions may surface when counting an enabled collection.
        """
        if self._cards_collection is None:
            return 0
        return self._cards_collection.count()

    def read_all(self) -> list[RawLTMRecord]:
        """Return all archived raw records ordered by interaction_id ascending.

        Fetches every document from the main collection using
        ``include=["metadatas"]`` and deserialises each entry via
        ``_deserialize_raw_record``.  Records that fail deserialisation are
        silently skipped.

        Returns:
            Deserialisable raw records sorted by interaction id.

        Raises:
            ChromaDB exceptions may surface during collection reads.

        Warning:
            Malformed records are skipped to keep recall resilient to partial
            writes or older metadata schemas.
        """
        results = self._collection.get(include=["metadatas"])
        metadatas: list[dict[str, object]] = results.get("metadatas") or []
        records: list[RawLTMRecord] = []
        for meta in metadatas:
            try:
                records.append(self._deserialize_raw_record(meta))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        records.sort(key=lambda r: r.interaction_id)
        return records

    def count(self) -> int:
        """Return the number of indexed raw records in the collection.

        Returns:
            Indexed raw-record count.

        Raises:
            ChromaDB exceptions may surface during count.
        """
        return self._collection.count()

    def clear(self) -> None:
        """Delete all indexed records from the raw-record collection.

        Returns:
            None.

        Raises:
            ChromaDB exceptions may surface during deletion.
        """
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)

    @property
    def card_store(self) -> JsonlMemoryCardStore | None:
        """Auxiliary JSONL card audit store, when configured.

        Returns:
            Configured card store or ``None``.

        Raises:
            None.
        """
        return self._card_store

    def _deserialize_raw_record(self, meta: dict[str, object]) -> RawLTMRecord:
        """Hydrate a raw-LTM record from Chroma metadata."""
        raw_payload = meta["raw_record"]
        if not isinstance(raw_payload, str) or not raw_payload:
            raise ValueError("Missing raw_record metadata")
        return RawLTMRecord.from_dict(json.loads(raw_payload))

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
