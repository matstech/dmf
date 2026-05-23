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

"""Long-term-memory archival contracts.

The runtime depends on this protocol instead of a concrete storage backend so
pruning can be tested without ChromaDB, LanceDB, or filesystem side effects.
Raw archived records are intentionally reinterpreted at recall time; storing
derived NLP signals here would make old archives drift when extraction rules
change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit

@runtime_checkable
class LTMHook(Protocol):
    """Interface for sending evicted interactions to Long-Term Memory.

    Args:
        entry: Evicted working-memory entry accepted by ``archive``.
        query_vector: Query embedding accepted by ``search_raw``.
        k: Maximum number of raw recall hits requested.

    Returns:
        Implementations return raw recall hits from ``search_raw`` and all
        archived raw records from ``read_all``.

    Raises:
        Backend-specific exceptions may surface from concrete implementations.
        The protocol itself does not require a specific exception type.
    """

    def archive(self, entry: MemoryEntry) -> None:
        """Persist an evicted interaction to Long-Term Memory.

        Args:
            entry: Working-memory entry selected for archival.

        Returns:
            None.

        Raises:
            Backend-specific exceptions may be raised by concrete stores.
        """
        ...

    def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:
        """Retrieve the top-k most relevant raw recall hits from storage.

        Args:
            query_vector: Dense embedding used by the backend for similarity
                search.
            k: Maximum number of hits to return.

        Returns:
            Raw recall hits ordered by backend relevance.

        Raises:
            Backend-specific exceptions may be raised by concrete stores.
        """
        ...

    def read_all(self) -> list[RawLTMRecord]:
        """Return all archived raw records in insertion order.

        Returns:
            Raw records ordered by archival insertion.

        Raises:
            Backend-specific exceptions may be raised by concrete stores.
        """
        ...

class NullLTMHook:
    """No-op LTM hook for environments without a real storage backend.

    Args:
        None.

    Returns:
        Instances satisfy ``LTMHook`` and intentionally discard all writes.

    Raises:
        None.
    """

    def archive(self, entry: MemoryEntry) -> None:
        """Accept and discard the evicted entry.

        Args:
            entry: Working-memory entry selected for archival.

        Returns:
            None.

        Raises:
            None.
        """
        pass

    def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:
        """Return no raw hits.

        Args:
            query_vector: Ignored query embedding.
            k: Ignored hit limit.

        Returns:
            Empty list.

        Raises:
            None.
        """
        return []

    def read_all(self) -> list[RawLTMRecord]:
        """Return no archived raw records.

        Returns:
            Empty list.

        Raises:
            None.
        """
        return []
