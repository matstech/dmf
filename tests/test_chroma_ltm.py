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

import json
import threading

import numpy as np
import pytest

from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.models.analysis import AnalysisReport
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit
from dmf.models.status import SurvivalStatus
from dmf.memory.temporal_memory import TemporalMemory
from dmf.utils.config_loader import DMFConfig, LTMSettings


class _FakeCollection:
    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self._query_result = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)

    def count(self) -> int:
        return len(self._query_result["documents"][0])

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return self._query_result

    def get(self, include=None):  # noqa: ARG002
        return {"ids": []}

    def delete(self, ids):  # noqa: ARG002
        return None


class _FakeCardStore:
    def __init__(self) -> None:
        self.archived: list[MemoryEntry] = []

    def archive(self, entry: MemoryEntry) -> list:
        self.archived.append(entry)
        return []


def _make_entry() -> MemoryEntry:
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
        interaction_id=7,
        text="Alice booked three tickets to Paris.",
        report=report,
        vector=np.array([0.0, 1.0], dtype=np.float32),
        token_count=6,
        timestamp=123.0,
    )


class TestChromaLTMHook:
    def test_archive_indexes_raw_record_text_and_metadata(self) -> None:
        collection = _FakeCollection()
        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7
        hook._lock = threading.Lock()
        hook._vector_config = None
        hook._embedding_engine = None
        hook._card_store = None
        embedded_texts: list[str] = []
        hook._embed_text = lambda text: embedded_texts.append(text) or np.array([0.3, 0.4], dtype=np.float32)

        entry = _make_entry()
        hook.archive(entry)

        assert embedded_texts == [entry.text]
        upsert = collection.upsert_calls[0]
        assert upsert["ids"] == ["record:7"]
        assert upsert["documents"] == [entry.text]
        assert upsert["embeddings"][0] == pytest.approx([0.3, 0.4])
        assert json.loads(upsert["metadatas"][0]["raw_record"])["text"] == entry.text
        assert upsert["metadatas"][0]["record_id"] == "record:7"
        assert upsert["metadatas"][0]["raw_interaction_id"] == 7
        assert upsert["metadatas"][0]["raw_role"] == "unknown"
        assert upsert["metadatas"][0]["raw_created_at"] == 123.0

    def test_archive_persists_auxiliary_cards_when_configured(self) -> None:
        collection = _FakeCollection()
        card_store = _FakeCardStore()
        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7
        hook._lock = threading.Lock()
        hook._vector_config = None
        hook._embedding_engine = None
        hook._card_store = card_store
        hook._embed_text = lambda text: np.array([0.3, 0.4], dtype=np.float32)  # noqa: ARG005

        entry = _make_entry()
        hook.archive(entry)

        assert len(collection.upsert_calls) == 1
        assert card_store.archived == [entry]

    def test_archive_does_not_create_cards_when_disabled(self) -> None:
        collection = _FakeCollection()
        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7
        hook._lock = threading.Lock()
        hook._vector_config = None
        hook._embedding_engine = None
        hook._card_store = None
        hook._embed_text = lambda text: np.array([0.3, 0.4], dtype=np.float32)  # noqa: ARG005

        hook.archive(_make_entry())

        assert len(collection.upsert_calls) == 1
        assert hook.card_store is None

    def test_search_raw_returns_raw_recall_hits(self) -> None:
        collection = _FakeCollection()
        collection._query_result = {
            "documents": [["Use SQLite for this task."]],
            "metadatas": [[
                {
                    "raw_record": json.dumps(
                        {
                            "record_id": "record:7",
                            "interaction_id": 7,
                            "role": "assistant",
                            "text": "Use SQLite for this task.",
                            "created_at": 123.0,
                            "provenance": {
                                "role": "assistant",
                                "source_turn": 12,
                                "is_user_correction": False,
                                "is_preference_update": False,
                                "is_constraint": False,
                                "derived_from_model": True,
                                "corrected_by_user": False,
                            },
                        }
                    )
                }
            ]],
            "distances": [[0.2]],
        }

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7

        hits = hook.search_raw([0.1, 0.2], k=1)

        assert hits == [
            RawRecallHit(
                record=hits[0].record,
                similarity_score=0.8,
                distance=0.2,
                rank_hint=0,
                source="ltm_raw",
            )
        ]
        assert hits[0].record.record_id == "record:7"
        assert hits[0].record.text == "Use SQLite for this task."
        assert collection.query_calls[0]["query_embeddings"] == [[0.1, 0.2]]

    def test_search_raw_ignores_configured_card_store(self) -> None:
        collection = _FakeCollection()
        collection._query_result = {
            "documents": [["Use SQLite for this task."]],
            "metadatas": [[
                {
                    "raw_record": json.dumps(
                        {
                            "record_id": "record:7",
                            "interaction_id": 7,
                            "role": "assistant",
                            "text": "Use SQLite for this task.",
                            "created_at": 123.0,
                            "provenance": {
                                "role": "assistant",
                                "source_turn": 12,
                                "is_user_correction": False,
                                "is_preference_update": False,
                                "is_constraint": False,
                                "derived_from_model": True,
                                "corrected_by_user": False,
                            },
                        }
                    )
                }
            ]],
            "distances": [[0.2]],
        }
        card_store = _FakeCardStore()
        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7
        hook._card_store = card_store

        hits = hook.search_raw([0.1, 0.2], k=1)

        assert len(hits) == 1
        assert isinstance(hits[0], RawRecallHit)
        assert card_store.archived == []

    def test_search_raw_does_not_call_count_before_query(self) -> None:
        collection = _FakeCollection()
        collection._query_result = {
            "documents": [["Use SQLite for this task."]],
            "metadatas": [[
                {
                    "raw_record": json.dumps(
                        {
                            "record_id": "record:7",
                            "interaction_id": 7,
                            "role": "assistant",
                            "text": "Use SQLite for this task.",
                            "created_at": 123.0,
                            "provenance": {
                                "role": "assistant",
                                "source_turn": 12,
                                "is_user_correction": False,
                                "is_preference_update": False,
                                "is_constraint": False,
                                "derived_from_model": True,
                                "corrected_by_user": False,
                            },
                        }
                    )
                }
            ]],
            "distances": [[0.2]],
        }
        collection.count = lambda: (_ for _ in ()).throw(AssertionError("count() should not be called"))  # type: ignore[assignment]

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7

        hits = hook.search_raw([0.1, 0.2], k=1)

        assert len(hits) == 1
        assert collection.query_calls[0]["n_results"] == 1

    def test_search_raw_skips_records_without_raw_metadata(self) -> None:
        collection = _FakeCollection()
        collection._query_result = {
            "documents": [["Use SQLite for this task."]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        }

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7

        assert hook.search_raw([0.1, 0.2], k=1) == []

    def test_search_raw_skips_records_over_distance_threshold(self) -> None:
        collection = _FakeCollection()
        collection._query_result = {
            "documents": [["Use SQLite for this task."]],
            "metadatas": [[
                {
                    "raw_record": json.dumps(
                        {
                            "record_id": "record:7",
                            "interaction_id": 7,
                            "role": "assistant",
                            "text": "Use SQLite for this task.",
                            "created_at": 123.0,
                            "provenance": {"role": "assistant", "source_turn": 12},
                        }
                    )
                }
            ]],
            "distances": [[0.9]],
        }

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7

        assert hook.search_raw([0.1, 0.2], k=1) == []

    def test_clear_deletes_existing_ids(self) -> None:
        collection = _FakeCollection()
        deleted: list[list[str]] = []
        collection.get = lambda include=None: {"ids": ["record:1", "record:2"]}  # type: ignore[assignment]
        collection.delete = lambda ids: deleted.append(ids)  # type: ignore[assignment]

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection

        hook.clear()

        assert deleted == [["record:1", "record:2"]]

    def test_from_dmf_config_passes_card_settings_to_chroma_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        captured: dict = {}

        def fake_init(self, **kwargs) -> None:  # noqa: ANN001
            captured.update(kwargs)

        monkeypatch.setattr(ChromaLTMHook, "__init__", fake_init)
        cfg = DMFConfig(
            ltm=LTMSettings(
                storage_type="chroma",
                chroma_path=str(tmp_path / "chroma"),
                collection_name="test_collection",
                distance_threshold=0.42,
                cards_enabled=True,
                cards_path=str(tmp_path / "cards.jsonl"),
            )
        )

        tm = TemporalMemory.from_dmf_config(cfg)

        assert isinstance(tm._ltm_hook, ChromaLTMHook)
        assert captured["collection_name"] == "test_collection"
        assert captured["persist_directory"] == str(tmp_path / "chroma")
        assert captured["distance_threshold"] == 0.42
        assert captured["cards_enabled"] is True
        assert captured["cards_path"] == str(tmp_path / "cards.jsonl")

    def test_from_dmf_config_passes_cards_collection_name_to_chroma_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        captured: dict = {}

        def fake_init(self, **kwargs) -> None:  # noqa: ANN001
            captured.update(kwargs)

        monkeypatch.setattr(ChromaLTMHook, "__init__", fake_init)
        cfg = DMFConfig(
            ltm=LTMSettings(
                storage_type="chroma",
                chroma_path=str(tmp_path / "chroma"),
                cards_enabled=True,
                cards_collection_name="my_cards",
            )
        )

        TemporalMemory.from_dmf_config(cfg)

        assert captured["cards_collection_name"] == "my_cards"


class TestChromaLTMHookCards:
    """Tests for the Chroma-backed card semantic index (Phase 6a)."""

    def _make_hook_with_fake_collections(
        self,
        *,
        cards_enabled: bool = True,
        distance_threshold: float = 0.7,
    ) -> tuple["ChromaLTMHook", "_FakeCollection", "_FakeCollection | None"]:
        """Build a ChromaLTMHook with injected fake collections."""
        from dmf.memory.card_projection import MemoryCardProjector

        main_collection = _FakeCollection()
        cards_collection = _FakeCollection() if cards_enabled else None

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = main_collection
        hook._distance_threshold = distance_threshold
        hook._lock = threading.Lock()
        hook._vector_config = None
        hook._embedding_engine = None
        hook._card_store = None
        hook._cards_enabled = cards_enabled
        hook._cards_collection = cards_collection
        hook._card_projector = MemoryCardProjector()
        hook._embed_text = lambda text: np.array([0.3, 0.4], dtype=np.float32)  # noqa: ARG005

        return hook, main_collection, cards_collection

    def test_search_cards_returns_empty_when_collection_is_none(self) -> None:
        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._cards_collection = None
        hook._distance_threshold = 0.7

        assert hook.search_cards([0.1, 0.2]) == []

    def test_count_cards_returns_zero_when_collection_is_none(self) -> None:
        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._cards_collection = None

        assert hook.count_cards() == 0

    def test_search_cards_on_disabled_hook_returns_empty(self) -> None:
        hook, _, _ = self._make_hook_with_fake_collections(cards_enabled=False)

        assert hook.search_cards([0.1, 0.2]) == []

    def test_archive_upserts_card_into_cards_collection(self) -> None:
        hook, main_col, cards_col = self._make_hook_with_fake_collections(cards_enabled=True)
        assert cards_col is not None

        entry = _make_entry()
        hook.archive(entry)

        # The main raw record must always be upserted
        assert len(main_col.upsert_calls) == 1

        # A projectable entry ("Alice booked three tickets to Paris.") should
        # produce at least one card upserted to the cards collection.
        assert len(cards_col.upsert_calls) >= 1
        card_upsert = cards_col.upsert_calls[0]
        meta = card_upsert["metadatas"][0]
        assert "card" in meta
        assert meta["source_record_id"] == "record:7"

    def test_count_cards_increments_after_archive(self) -> None:
        hook, _main, cards_col = self._make_hook_with_fake_collections(cards_enabled=True)
        assert cards_col is not None

        # Patch count() to reflect upserted items
        def _dynamic_count() -> int:
            return len(cards_col.upsert_calls)

        cards_col.count = _dynamic_count  # type: ignore[method-assign]

        assert hook.count_cards() == 0
        hook.archive(_make_entry())
        assert hook.count_cards() > 0

    def test_search_cards_returns_hits_pointing_to_source_raw_record(self) -> None:
        raw_record_dict = {
            "record_id": "record:7",
            "interaction_id": 7,
            "role": "user",
            "text": "Alice booked three tickets to Paris.",
            "created_at": 123.0,
            "provenance": {
                "role": "user",
                "source_turn": 7,
                "is_user_correction": False,
                "is_preference_update": False,
                "is_constraint": False,
                "derived_from_model": False,
                "corrected_by_user": False,
            },
        }
        card_meta = {
            "card": json.dumps({"card_id": "card:record:7:0"}),
            "card_id": "card:record:7:0",
            "source_record_id": "record:7",
            "kind": "event",
        }

        hook, main_col, cards_col = self._make_hook_with_fake_collections(cards_enabled=True)
        assert cards_col is not None

        # Configure the cards collection to return one card hit
        cards_col._query_result = {
            "metadatas": [[card_meta]],
            "distances": [[0.15]],
        }
        # Configure the main collection to resolve the source raw record
        main_col.get = lambda ids, include=None: {  # type: ignore[assignment]
            "metadatas": [{"raw_record": json.dumps(raw_record_dict)}]
        }

        hits = hook.search_cards([0.3, 0.4], k=1)

        assert len(hits) == 1
        assert hits[0].record.record_id == "record:7"
        assert hits[0].distance == pytest.approx(0.15)
        assert hits[0].similarity_score == pytest.approx(0.85)

    def test_search_cards_skips_card_when_source_record_not_found(self) -> None:
        card_meta = {
            "card": json.dumps({"card_id": "card:record:99:0"}),
            "card_id": "card:record:99:0",
            "source_record_id": "record:99",
            "kind": "fact",
        }

        hook, main_col, cards_col = self._make_hook_with_fake_collections(cards_enabled=True)
        assert cards_col is not None

        cards_col._query_result = {
            "metadatas": [[card_meta]],
            "distances": [[0.10]],
        }
        # Source record not found: return empty metadatas
        main_col.get = lambda ids, include=None: {"metadatas": []}  # type: ignore[assignment]

        hits = hook.search_cards([0.3, 0.4], k=1)

        assert hits == []

    def test_search_cards_skips_hits_over_distance_threshold(self) -> None:
        card_meta = {
            "card": json.dumps({"card_id": "card:record:7:0"}),
            "card_id": "card:record:7:0",
            "source_record_id": "record:7",
            "kind": "event",
        }

        hook, _main, cards_col = self._make_hook_with_fake_collections(
            cards_enabled=True, distance_threshold=0.5
        )
        assert cards_col is not None

        cards_col._query_result = {
            "metadatas": [[card_meta]],
            "distances": [[0.8]],  # above threshold
        }

        hits = hook.search_cards([0.3, 0.4], k=1)

        assert hits == []


# ---------------------------------------------------------------------------
# Helper: a raw-record dict matching RawLTMRecord.from_dict expectations
# ---------------------------------------------------------------------------

def _raw_record_dict(interaction_id: int, text: str = "hello") -> dict:
    return {
        "record_id": f"record:{interaction_id}",
        "interaction_id": interaction_id,
        "role": "assistant",
        "text": text,
        "created_at": float(interaction_id),
        "provenance": {
            "role": "assistant",
            "source_turn": interaction_id,
            "is_user_correction": False,
            "is_preference_update": False,
            "is_constraint": False,
            "derived_from_model": True,
            "corrected_by_user": False,
        },
    }


class TestChromaLTMHookReadAll:
    """Tests for ChromaLTMHook.read_all()."""

    def _hook_with_get_result(self, metadatas: list[dict]) -> ChromaLTMHook:
        """Build a hook whose collection.get() returns the given metadatas."""
        collection = _FakeCollection()
        collection.get = lambda include=None: {"ids": [], "metadatas": metadatas}  # type: ignore[assignment]

        hook = ChromaLTMHook.__new__(ChromaLTMHook)
        hook._collection = collection
        hook._distance_threshold = 0.7
        hook._lock = threading.Lock()
        return hook

    def test_read_all_returns_empty_when_collection_is_empty(self) -> None:
        hook = self._hook_with_get_result([])

        result = hook.read_all()

        assert result == []

    def test_read_all_returns_all_records(self) -> None:
        meta_a = {"raw_record": json.dumps(_raw_record_dict(1, "first"))}
        meta_b = {"raw_record": json.dumps(_raw_record_dict(2, "second"))}
        hook = self._hook_with_get_result([meta_a, meta_b])

        result = hook.read_all()

        assert len(result) == 2
        assert all(isinstance(r, RawLTMRecord) for r in result)
        texts = {r.text for r in result}
        assert texts == {"first", "second"}

    def test_read_all_orders_by_interaction_id_ascending(self) -> None:
        # Intentionally provide records in reverse order
        metas = [
            {"raw_record": json.dumps(_raw_record_dict(interaction_id=i))}
            for i in [5, 1, 3]
        ]
        hook = self._hook_with_get_result(metas)

        result = hook.read_all()

        assert [r.interaction_id for r in result] == [1, 3, 5]

    def test_read_all_skips_malformed_entries(self) -> None:
        metas = [
            {"raw_record": "not-valid-json"},
            {"raw_record": json.dumps(_raw_record_dict(7, "valid"))},
        ]
        hook = self._hook_with_get_result(metas)

        result = hook.read_all()

        assert len(result) == 1
        assert result[0].interaction_id == 7

    def test_read_all_skips_entries_missing_raw_record_key(self) -> None:
        metas = [
            {},  # no raw_record key
            {"raw_record": json.dumps(_raw_record_dict(3, "ok"))},
        ]
        hook = self._hook_with_get_result(metas)

        result = hook.read_all()

        assert len(result) == 1
        assert result[0].text == "ok"
