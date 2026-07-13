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

"""Backend-neutral recall filtering contract."""

from __future__ import annotations

from dataclasses import dataclass

from dmf.models.constants import VALID_CARD_KINDS


@dataclass(frozen=True)
class RecallFilter:
    """Backend-neutral metadata filter for raw and card recall.

    String sequences are stripped and stored as tuples. Empty strings and
    duplicate values are rejected so backend translators receive deterministic
    predicates.
    """

    record_ids: tuple[str, ...] = ()
    excluded_record_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    interaction_id_min: int | None = None
    interaction_id_max: int | None = None
    created_at_min: float | None = None
    created_at_max: float | None = None
    card_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate ranges and normalize string sequences."""
        object.__setattr__(
            self,
            "record_ids",
            _normalize_unique_strings(self.record_ids, field="record_ids"),
        )
        object.__setattr__(
            self,
            "excluded_record_ids",
            _normalize_unique_strings(
                self.excluded_record_ids,
                field="excluded_record_ids",
            ),
        )
        object.__setattr__(
            self,
            "roles",
            _normalize_unique_strings(self.roles, field="roles"),
        )
        card_kinds = _normalize_unique_strings(self.card_kinds, field="card_kinds")
        unsupported_kinds = sorted(set(card_kinds) - VALID_CARD_KINDS)
        if unsupported_kinds:
            raise ValueError(f"Unsupported card_kinds: {unsupported_kinds!r}")
        object.__setattr__(self, "card_kinds", card_kinds)

        _validate_range(
            minimum=self.interaction_id_min,
            maximum=self.interaction_id_max,
            field="interaction_id",
        )
        _validate_range(
            minimum=self.created_at_min,
            maximum=self.created_at_max,
            field="created_at",
        )

    @property
    def is_empty(self) -> bool:
        """Return whether the filter contains no backend predicate."""
        return not (
            self.record_ids
            or self.excluded_record_ids
            or self.roles
            or self.interaction_id_min is not None
            or self.interaction_id_max is not None
            or self.created_at_min is not None
            or self.created_at_max is not None
            or self.card_kinds
        )


def _normalize_unique_strings(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{field} values must be strings")
        text = value.strip()
        if not text:
            raise ValueError(f"{field} values must be non-empty strings")
        if text in seen:
            raise ValueError(f"{field} values must be unique")
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _validate_range(
    *,
    minimum: int | float | None,
    maximum: int | float | None,
    field: str,
) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{field}_min cannot be greater than {field}_max")


__all__ = ["RecallFilter"]
