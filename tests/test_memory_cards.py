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
import time
from pathlib import Path

import numpy as np

from dmf.memory.card_projection import mark_superseded, project_memory_cards
from dmf.memory.card_store import JsonlMemoryCardStore
from dmf.memory.ltm_hooks import FileLTMHook
from dmf.memory.temporal_memory import TemporalMemory
from dmf.models.analysis import (
    AnalysisReport,
    InteractionProvenance,
    InteractionSignals,
    MemoryLineage,
)
from dmf.models.memory import MemoryEntry
from dmf.models.memory import MemoryCard
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit
from dmf.utils.config_loader import DMFConfig, LTMSettings


def _entry(
    *,
    text: str = "I prefer afternoon meetings.",
    interaction_id: int = 7,
    role: str = "user",
    topic_identity: str | None = "preference|prefer",
    topic_value: str | None = "afternoon_meetings",
    signals: InteractionSignals | None = None,
    provenance: InteractionProvenance | None = None,
    lineage: MemoryLineage | None = None,
    is_query_like: bool = False,
    is_ack_like: bool = False,
) -> MemoryEntry:
    return MemoryEntry(
        interaction_id=interaction_id,
        text=text,
        report=AnalysisReport(
            info_density=0.5,
            sentiment_abs=0.0,
            entity_count=1,
            is_system_prompt=False,
            latency_ms=0.0,
            survival_score=0.8,
            signals=signals or InteractionSignals(is_preference=True),
            provenance=provenance or InteractionProvenance(role=role, source_turn=12),
            topic_identity=topic_identity,
            topic_value=topic_value,
            is_query_like=is_query_like,
            is_ack_like=is_ack_like,
        ),
        vector=np.array([1.0, 0.0], dtype=float),
        token_count=len(text.split()),
        timestamp=1710000000.5,
        lineage=lineage or MemoryLineage(),
    )


def test_projects_preference_card_with_source_provenance() -> None:
    cards = project_memory_cards(_entry())

    assert len(cards) == 1
    card = cards[0]
    assert card.card_id == "card:record:7:0"
    assert card.kind == "preference"
    assert card.subject == "user"
    assert card.predicate == "prefer"
    assert card.object == "afternoon_meetings"
    assert card.validity.status == "active"
    assert card.provenance.source_record_id == "record:7"
    assert card.provenance.speaker_role == "user"
    assert card.provenance.source_turn == 12
    assert card.time_anchor.turn_id == 7


def test_projects_available_benchmark_agnostic_primitives() -> None:
    cases = [
        (
            _entry(
                topic_identity="state|live",
                topic_value="rome",
                signals=InteractionSignals(is_current_state=True),
            ),
            "current_state",
        ),
        (
            _entry(
                topic_identity="constraint|use",
                topic_value="external_api",
                signals=InteractionSignals(is_constraint=True, has_negation=True),
            ),
            "constraint",
        ),
        (
            _entry(
                topic_identity="preference|prefer",
                topic_value="coffee",
                signals=InteractionSignals(is_correction=True),
            ),
            "correction",
        ),
        (
            _entry(
                text="Alice is my manager.",
                topic_identity=None,
                topic_value=None,
                signals=InteractionSignals(),
            ),
            "relation",
        ),
        (
            _entry(
                text="I booked flights to Paris.",
                topic_identity=None,
                topic_value=None,
                signals=InteractionSignals(),
            ),
            "event",
        ),
    ]

    assert [project_memory_cards(entry)[0].kind for entry, _ in cases] == [
        expected for _, expected in cases
    ]


def test_projection_skips_queries_and_ack_like_turns() -> None:
    assert project_memory_cards(_entry(is_query_like=True)) == []
    assert project_memory_cards(_entry(is_ack_like=True)) == []


def test_memory_card_serialization_round_trips() -> None:
    card = project_memory_cards(_entry())[0]

    parsed = MemoryCard.from_dict(json.loads(json.dumps(card.to_dict())))

    assert parsed == card


def test_validity_helper_supports_superseded_cards() -> None:
    card = project_memory_cards(_entry())[0]

    superseded = mark_superseded(card, superseded_by=["card:record:8:0"])

    assert superseded.validity.status == "superseded"
    assert superseded.validity.superseded_by == ["card:record:8:0"]
    assert card.validity.status == "active"


def test_projection_maps_correction_lineage_to_validity() -> None:
    entry = _entry(
        signals=InteractionSignals(is_correction=True),
        lineage=MemoryLineage(corrects=["record:3"], invalidates=["record:3"]),
    )

    card = project_memory_cards(entry)[0]

    assert card.validity.status == "corrected"
    assert card.validity.corrects == ["record:3"]
    assert card.validity.invalidates == ["record:3"]


def test_jsonl_card_store_persists_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cards.jsonl"
    store = JsonlMemoryCardStore(path)

    cards = store.archive(_entry(text="I prefer tea.", topic_value="tea"))

    assert path.exists()
    assert store.read_all() == cards
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["provenance"]["source_record_id"] == "record:7"
    assert payload["surface_forms"] == ["I prefer tea.", "tea"]


def test_jsonl_card_store_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "cards.jsonl"
    store = JsonlMemoryCardStore(path, enabled=False)

    assert store.archive(_entry()) == []
    assert not path.exists()


def test_file_ltm_hook_can_persist_raw_and_auxiliary_cards(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    cards_path = tmp_path / "cards.jsonl"
    hook = FileLTMHook(raw_path, cards_enabled=True, cards_path=cards_path)

    hook.archive(_entry())

    assert len(raw_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(cards_path.read_text(encoding="utf-8").splitlines()) == 1
    assert hook.card_store is not None
    assert hook.search_raw([1.0, 0.0]) == []


def test_temporal_memory_config_disable_leaves_no_card_store(tmp_path: Path) -> None:
    cfg = DMFConfig(
        ltm=LTMSettings(
            storage_type="file",
            storage_path=str(tmp_path / "raw.jsonl"),
            cards_enabled=False,
            cards_path=str(tmp_path / "cards.jsonl"),
        )
    )

    tm = TemporalMemory.from_dmf_config(cfg)

    assert isinstance(tm._ltm_hook, FileLTMHook)
    assert tm._ltm_hook.card_store is None


def test_temporal_memory_config_can_enable_jsonl_card_store(tmp_path: Path) -> None:
    cards_path = tmp_path / "cards.jsonl"
    cfg = DMFConfig(
        ltm=LTMSettings(
            storage_type="file",
            storage_path=str(tmp_path / "raw.jsonl"),
            cards_enabled=True,
            cards_path=str(cards_path),
        )
    )

    tm = TemporalMemory.from_dmf_config(cfg)

    assert isinstance(tm._ltm_hook, FileLTMHook)
    assert tm._ltm_hook.card_store is not None
    assert tm._ltm_hook.card_store.path == cards_path


def test_auxiliary_cards_do_not_change_final_raw_recall(tmp_path: Path) -> None:
    class RecallingFileHook(FileLTMHook):
        def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
            return [
                RawRecallHit(
                    record=RawLTMRecord(
                        record_id="record:1",
                        interaction_id=1,
                        role="user",
                        text="I prefer coffee.",
                        created_at=time.time(),
                        provenance=InteractionProvenance(role="user"),
                    ),
                    similarity_score=0.9,
                )
            ]

    hook = RecallingFileHook(tmp_path / "raw.jsonl", cards_enabled=True)
    tm = TemporalMemory(ltm_hook=hook)

    assert [hit.record.record_id for hit in tm.get_raw_recall_hits(np.array([1.0]))] == [
        "record:1"
    ]
