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

"""Translate backend-neutral recall filters to Qdrant predicates."""

from __future__ import annotations

from typing import Literal

from dmf.models.recall_filter import RecallFilter

FilterTarget = Literal["raw", "card"]


def build_qdrant_filter(
    recall_filter: RecallFilter | None,
    *,
    target: FilterTarget,
) -> object | None:
    """Return a Qdrant Filter for the target collection, or ``None``."""
    if recall_filter is None or recall_filter.is_empty:
        return None

    models = _qdrant_models()
    must = []
    must_not = []

    record_field = "record_id" if target == "raw" else "source_record_id"
    role_field = "role" if target == "raw" else "raw_role"
    interaction_field = "interaction_id" if target == "raw" else "raw_interaction_id"
    created_at_field = "created_at" if target == "raw" else "raw_created_at"
    if recall_filter.record_ids:
        must.append(_match_any(models, record_field, recall_filter.record_ids))
    if recall_filter.excluded_record_ids:
        must_not.append(
            _match_any(models, record_field, recall_filter.excluded_record_ids)
        )
    if recall_filter.roles:
        must.append(_match_any(models, role_field, recall_filter.roles))
    if (
        recall_filter.interaction_id_min is not None
        or recall_filter.interaction_id_max is not None
    ):
        must.append(
            _range(
                models,
                interaction_field,
                gte=recall_filter.interaction_id_min,
                lte=recall_filter.interaction_id_max,
            )
        )
    if recall_filter.created_at_min is not None or recall_filter.created_at_max is not None:
        must.append(
            _range(
                models,
                created_at_field,
                gte=recall_filter.created_at_min,
                lte=recall_filter.created_at_max,
            )
        )
    if recall_filter.card_kinds:
        must.append(_match_any(models, "kind", recall_filter.card_kinds))

    return models.Filter(must=must or None, must_not=must_not or None)


def _match_any(models: object, key: str, values: tuple[str, ...]) -> object:
    return models.FieldCondition(
        key=key,
        match=models.MatchAny(any=list(values)),
    )


def _range(
    models: object,
    key: str,
    *,
    gte: int | float | None,
    lte: int | float | None,
) -> object:
    return models.FieldCondition(
        key=key,
        range=models.Range(gte=gte, lte=lte),
    )


def _qdrant_models() -> object:
    from qdrant_client import models  # noqa: PLC0415

    return models


__all__ = ["build_qdrant_filter"]
