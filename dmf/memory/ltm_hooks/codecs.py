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

"""Shared payload codecs for LTM backend adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from dmf.models.memory import MemoryCard
from dmf.models.raw_ltm import RawLTMRecord

LTM_PAYLOAD_SCHEMA_VERSION = 1
LTM_RECORD_TYPE_RAW = "raw"
LTM_RECORD_TYPE_CARD = "card"


def build_raw_payload(record: RawLTMRecord) -> dict[str, object]:
    """Return the backend-neutral raw-record payload."""
    return {
        "schema_version": LTM_PAYLOAD_SCHEMA_VERSION,
        "record_type": LTM_RECORD_TYPE_RAW,
        "record_id": record.record_id,
        "interaction_id": record.interaction_id,
        "role": record.role,
        "created_at": record.created_at,
        "raw_record": record.to_dict(),
    }


def raw_record_from_payload(payload: Mapping[str, object]) -> RawLTMRecord:
    """Hydrate a raw record from a backend payload.

    ``raw_record`` may be a mapping, as used by document payload stores, or a
    JSON string, as used by existing Chroma metadata.
    """
    raw_payload = payload["raw_record"]
    if isinstance(raw_payload, str):
        raw_payload = json.loads(raw_payload)
    if not isinstance(raw_payload, Mapping):
        raise TypeError("raw_record payload must be a mapping or JSON string")
    return RawLTMRecord.from_dict(dict(raw_payload))


def build_card_payload(card: MemoryCard) -> dict[str, object]:
    """Return the backend-neutral projected-card payload."""
    card_payload: dict[str, Any] = card.to_dict()
    return {
        "schema_version": LTM_PAYLOAD_SCHEMA_VERSION,
        "record_type": LTM_RECORD_TYPE_CARD,
        "card_id": card.card_id,
        "source_record_id": card.provenance.source_record_id,
        "kind": card.kind,
        "card": card_payload,
    }
