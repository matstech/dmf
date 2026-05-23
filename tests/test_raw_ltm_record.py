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

"""
tests/test_raw_ltm_record.py
----------------------------
Unit tests for the raw-LTM archival contract.
"""

from __future__ import annotations

import json

import numpy as np

from dmf.models.analysis import AnalysisReport, InteractionProvenance, InteractionSignals
from dmf.models.raw_ltm import (
    ContextualizedRecallCandidate,
    RawLTMRecord,
    RawRecallHit,
)
from dmf.models.memory import MemoryEntry


def _make_record() -> RawLTMRecord:
    return RawLTMRecord(
        record_id="record:7",
        interaction_id=7,
        role="assistant",
        text="Use SQLite for this task.",
        created_at=1710000000.5,
        provenance=InteractionProvenance(
            role="assistant",
            source_turn=12,
            derived_from_model=True,
        ),
    )


def test_to_dict_returns_only_stable_raw_fields() -> None:
    record = _make_record()

    assert record.to_dict() == {
        "record_id": "record:7",
        "interaction_id": 7,
        "role": "assistant",
        "text": "Use SQLite for this task.",
        "created_at": 1710000000.5,
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


def test_to_dict_does_not_include_derived_recall_fields() -> None:
    record = _make_record()
    payload = record.to_dict()

    assert "signals" not in payload
    assert "topic_identity" not in payload
    assert "topic_value" not in payload
    assert "omega" not in payload


def test_to_json_round_trips() -> None:
    record = _make_record()

    parsed = json.loads(record.to_json())

    assert parsed == record.to_dict()


def test_from_dict_rehydrates_provenance() -> None:
    payload = _make_record().to_dict()

    record = RawLTMRecord.from_dict(payload)

    assert record.record_id == "record:7"
    assert record.provenance == InteractionProvenance(
        role="assistant",
        source_turn=12,
        derived_from_model=True,
    )


def test_memory_entry_projects_to_raw_ltm_record() -> None:
    entry = MemoryEntry(
        interaction_id=7,
        text="Use SQLite for this task.",
        report=AnalysisReport(
            info_density=0.4,
            sentiment_abs=0.0,
            entity_count=0,
            is_system_prompt=False,
            latency_ms=1.0,
            topic_identity="preference|prefer",
            topic_value="sqlite",
            signals=InteractionSignals(is_preference=True),
            provenance=InteractionProvenance(
                role="assistant",
                source_turn=12,
                derived_from_model=True,
            ),
        ),
        vector=np.array([0.0, 1.0], dtype=np.float32),
        token_count=5,
        timestamp=1710000000.5,
    )

    record = entry.to_raw_ltm_record()

    assert record == RawLTMRecord(
        record_id="record:7",
        interaction_id=7,
        role="assistant",
        text="Use SQLite for this task.",
        created_at=1710000000.5,
        provenance=InteractionProvenance(
            role="assistant",
            source_turn=12,
            derived_from_model=True,
        ),
    )


def test_raw_recall_hit_serializes_search_fields_without_reanalysis() -> None:
    hit = RawRecallHit(
        record=_make_record(),
        similarity_score=0.83,
        distance=0.17,
        rank_hint=2,
    )

    assert hit.to_dict() == {
        "record": _make_record().to_dict(),
        "similarity_score": 0.83,
        "distance": 0.17,
        "rank_hint": 2,
        "source": "ltm_raw",
    }


def test_contextualized_candidate_keeps_topic_metadata_inside_report() -> None:
    candidate = ContextualizedRecallCandidate(
        record=_make_record(),
        report=AnalysisReport(
            info_density=0.4,
            sentiment_abs=0.0,
            entity_count=0,
            is_system_prompt=False,
            latency_ms=1.0,
            topic_identity="preference|prefer",
            topic_value="coffee",
            signals=InteractionSignals(
                is_current_state=True,
                is_preference=True,
            ),
        ),
        similarity_score=0.91,
        recall_score=0.88,
    )

    payload = candidate.to_dict()

    assert payload["record"]["record_id"] == "record:7"
    assert payload["report"]["topic_identity"] == "preference|prefer"
    assert payload["report"]["topic_value"] == "coffee"
    assert payload["similarity_score"] == 0.91
    assert payload["recall_score"] == 0.88
    assert "topic_identity" not in payload
    assert "topic_value" not in payload
