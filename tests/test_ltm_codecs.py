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

import pytest

from dmf.memory.ltm_hooks.codecs import (
    LTM_PAYLOAD_SCHEMA_VERSION,
    LTM_RECORD_TYPE_CARD,
    LTM_RECORD_TYPE_RAW,
    build_card_payload,
    build_raw_payload,
    raw_record_from_payload,
)
from dmf.models.analysis import InteractionProvenance
from dmf.models.memory import (
    MemoryCard,
    MemoryCardProvenance,
    MemoryCardTimeAnchor,
    MemoryCardValidity,
)
from dmf.models.raw_ltm import RawLTMRecord


def _raw_record() -> RawLTMRecord:
    return RawLTMRecord(
        record_id="record:7",
        interaction_id=7,
        role="user",
        text="Alice prefers green tea.",
        created_at=123.0,
        provenance=InteractionProvenance(role="user", source_turn=7),
    )


def _card() -> MemoryCard:
    return MemoryCard(
        card_id="card:record:7:0",
        kind="preference",
        subject="Alice",
        predicate="prefer",
        object="green tea",
        qualifiers={"time": "afternoon"},
        time_anchor=MemoryCardTimeAnchor(relative_order=7, turn_id=7),
        validity=MemoryCardValidity(status="active"),
        provenance=MemoryCardProvenance(
            source_record_id="record:7",
            speaker_role="user",
            source_turn=7,
        ),
        confidence=0.8,
        surface_forms=["Alice prefers green tea."],
    )


def test_raw_payload_round_trips_from_mapping() -> None:
    record = _raw_record()
    payload = build_raw_payload(record)

    assert payload["schema_version"] == LTM_PAYLOAD_SCHEMA_VERSION
    assert payload["record_type"] == LTM_RECORD_TYPE_RAW
    assert payload["record_id"] == "record:7"
    assert payload["interaction_id"] == 7
    assert payload["role"] == "user"
    assert payload["created_at"] == 123.0
    assert raw_record_from_payload(payload) == record


def test_raw_payload_round_trips_from_json_string() -> None:
    record = _raw_record()
    payload = build_raw_payload(record)
    payload["raw_record"] = json.dumps(payload["raw_record"])

    assert raw_record_from_payload(payload) == record


def test_raw_payload_missing_key_propagates_key_error() -> None:
    with pytest.raises(KeyError):
        raw_record_from_payload({})


def test_raw_payload_wrong_type_propagates_type_error() -> None:
    with pytest.raises(TypeError):
        raw_record_from_payload({"raw_record": 123})


def test_raw_payload_invalid_json_propagates_decode_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        raw_record_from_payload({"raw_record": "not-valid-json"})


def test_card_payload_contains_complete_card_metadata() -> None:
    card = _card()
    payload = build_card_payload(card)

    assert payload["schema_version"] == LTM_PAYLOAD_SCHEMA_VERSION
    assert payload["record_type"] == LTM_RECORD_TYPE_CARD
    assert payload["card_id"] == "card:record:7:0"
    assert payload["source_record_id"] == "record:7"
    assert payload["kind"] == "preference"
    assert payload["card"] == card.to_dict()
