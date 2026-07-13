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

"""Translate backend-neutral recall filters to Chroma metadata predicates."""

from __future__ import annotations

from typing import Any, Literal

from dmf.models.recall_filter import RecallFilter

FilterTarget = Literal["raw", "card"]


def build_chroma_where(
    recall_filter: RecallFilter | None,
    *,
    target: FilterTarget,
) -> dict[str, Any] | None:
    """Return a Chroma ``where`` mapping for the target collection, or ``None``."""
    if recall_filter is None or recall_filter.is_empty:
        return None

    conditions: list[dict[str, Any]] = []
    record_field = "record_id" if target == "raw" else "source_record_id"
    if recall_filter.record_ids:
        conditions.append({record_field: {"$in": list(recall_filter.record_ids)}})
    if recall_filter.excluded_record_ids:
        conditions.append(
            {record_field: {"$nin": list(recall_filter.excluded_record_ids)}}
        )
    if recall_filter.roles:
        conditions.append({"raw_role": {"$in": list(recall_filter.roles)}})
    if recall_filter.interaction_id_min is not None:
        conditions.append(
            {"raw_interaction_id": {"$gte": recall_filter.interaction_id_min}}
        )
    if recall_filter.interaction_id_max is not None:
        conditions.append(
            {"raw_interaction_id": {"$lte": recall_filter.interaction_id_max}}
        )
    if recall_filter.created_at_min is not None:
        conditions.append({"raw_created_at": {"$gte": recall_filter.created_at_min}})
    if recall_filter.created_at_max is not None:
        conditions.append({"raw_created_at": {"$lte": recall_filter.created_at_max}})
    if recall_filter.card_kinds:
        conditions.append({"kind": {"$in": list(recall_filter.card_kinds)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


__all__ = ["build_chroma_where"]
