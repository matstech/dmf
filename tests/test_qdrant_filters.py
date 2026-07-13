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

import pytest

from dmf.memory.ltm_hooks.chroma_filters import build_chroma_where
from dmf.memory.ltm_hooks.qdrant_filters import build_qdrant_filter
from dmf.models.recall_filter import RecallFilter


def test_recall_filter_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="interaction_id_min"):
        RecallFilter(interaction_id_min=3, interaction_id_max=2)

    with pytest.raises(ValueError, match="created_at_min"):
        RecallFilter(created_at_min=2.0, created_at_max=1.0)


def test_recall_filter_normalizes_and_rejects_invalid_strings() -> None:
    assert RecallFilter(record_ids=(" record:1 ",)).record_ids == ("record:1",)

    with pytest.raises(ValueError, match="non-empty"):
        RecallFilter(roles=(" ",))

    with pytest.raises(ValueError, match="unique"):
        RecallFilter(record_ids=("record:1", "record:1"))

    with pytest.raises(ValueError, match="Unsupported card_kinds"):
        RecallFilter(card_kinds=("unsupported",))


def test_empty_filter_produces_no_backend_predicate() -> None:
    assert build_qdrant_filter(None, target="raw") is None
    assert build_qdrant_filter(RecallFilter(), target="raw") is None
    assert build_chroma_where(None, target="raw") is None
    assert build_chroma_where(RecallFilter(), target="raw") is None


def test_qdrant_raw_filter_combines_must_must_not_and_ranges() -> None:
    qfilter = build_qdrant_filter(
        RecallFilter(
            record_ids=("record:1", "record:2"),
            excluded_record_ids=("record:9",),
            roles=("assistant", "user"),
            interaction_id_min=3,
            interaction_id_max=7,
            created_at_min=100.0,
            created_at_max=200.0,
        ),
        target="raw",
    )

    assert qfilter is not None
    assert [condition.key for condition in qfilter.must] == [
        "record_id",
        "role",
        "interaction_id",
        "created_at",
    ]
    assert qfilter.must[0].match.any == ["record:1", "record:2"]
    assert qfilter.must[1].match.any == ["assistant", "user"]
    assert qfilter.must[2].range.gte == 3
    assert qfilter.must[2].range.lte == 7
    assert qfilter.must[3].range.gte == 100.0
    assert qfilter.must[3].range.lte == 200.0
    assert [condition.key for condition in qfilter.must_not] == ["record_id"]
    assert qfilter.must_not[0].match.any == ["record:9"]


def test_qdrant_card_filter_uses_source_and_card_metadata_fields() -> None:
    qfilter = build_qdrant_filter(
        RecallFilter(
            record_ids=("record:1",),
            excluded_record_ids=("record:9",),
            roles=("user",),
            interaction_id_min=1,
            created_at_max=300.0,
            card_kinds=("preference",),
        ),
        target="card",
    )

    assert qfilter is not None
    assert [condition.key for condition in qfilter.must] == [
        "source_record_id",
        "raw_role",
        "raw_interaction_id",
        "raw_created_at",
        "kind",
    ]
    assert qfilter.must[0].match.any == ["record:1"]
    assert qfilter.must[-1].match.any == ["preference"]
    assert [condition.key for condition in qfilter.must_not] == ["source_record_id"]


def test_chroma_where_uses_and_for_combined_predicates() -> None:
    where = build_chroma_where(
        RecallFilter(
            record_ids=("record:1",),
            excluded_record_ids=("record:9",),
            roles=("assistant",),
            interaction_id_min=3,
            interaction_id_max=7,
            created_at_min=100.0,
            created_at_max=200.0,
        ),
        target="raw",
    )

    assert where == {
        "$and": [
            {"record_id": {"$in": ["record:1"]}},
            {"record_id": {"$nin": ["record:9"]}},
            {"raw_role": {"$in": ["assistant"]}},
            {"raw_interaction_id": {"$gte": 3}},
            {"raw_interaction_id": {"$lte": 7}},
            {"raw_created_at": {"$gte": 100.0}},
            {"raw_created_at": {"$lte": 200.0}},
        ]
    }


def test_chroma_card_where_uses_source_record_and_kind() -> None:
    assert build_chroma_where(
        RecallFilter(record_ids=("record:1",), card_kinds=("preference",)),
        target="card",
    ) == {
        "$and": [
            {"source_record_id": {"$in": ["record:1"]}},
            {"kind": {"$in": ["preference"]}},
        ]
    }
