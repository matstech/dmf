"""Append-only raw-record LTM archive.

`FileLTMHook` is the simplest concrete `LTMHook`: it writes one
`RawLTMRecord` per JSONL line and exposes no retrieval index.
Semantic search belongs to vector-backed stores such as `ChromaLTMHook`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from dmf.memory.card_store import JsonlMemoryCardStore
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import RawLTMRecord, RawRecallHit
from dmf.utils.constants import DEFAULT_LTM_RECALL_LIMIT, DEFAULT_TEXT_ENCODING


class FileLTMHook:
    """Append-only JSONL archive for raw long-term-memory records.

    Args:
        storage_path: Filesystem path for the raw-record JSONL archive.
        cards_enabled: Whether to project auxiliary memory cards on archive.
        cards_path: Optional JSONL path for auxiliary cards.
        card_store: Optional prebuilt card store. Takes precedence over
            ``cards_enabled`` and ``cards_path``.

    Returns:
        File-backed LTM hook instance.

    Raises:
        OSError: If the archive parent directory cannot be created.

    Warning:
        ``search_raw`` intentionally returns an empty list. This backend is
        archival-only; semantic retrieval requires a vector-backed hook.
    """

    def __init__(
        self,
        storage_path: Path | str,
        *,
        cards_enabled: bool = False,
        cards_path: Path | str | None = None,
        card_store: JsonlMemoryCardStore | None = None,
    ) -> None:
        self._path: Path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._card_store = card_store
        if self._card_store is None and cards_enabled:
            self._card_store = JsonlMemoryCardStore(
                cards_path or self._path.with_suffix(".cards.jsonl")
            )

    def archive(self, entry: MemoryEntry) -> None:
        """Append one raw record and optional auxiliary cards.

        Args:
            entry: Working-memory entry selected for archival.

        Returns:
            None.

        Raises:
            OSError: If the archive cannot be written.
            TypeError: If the raw record cannot be JSON-serialised.
        """
        line = json.dumps(entry.to_raw_ltm_record().to_dict(), ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding=DEFAULT_TEXT_ENCODING) as fh:
                fh.write(line + "\n")
        if self._card_store is not None:
            self._card_store.archive(entry)

    def search_raw(self, query_vector: list[float], k: int = DEFAULT_LTM_RECALL_LIMIT) -> list[RawRecallHit]:
        """Return no raw search hits for this archival-only backend.

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
        """Return all archived raw records in insertion order.

        Reads the JSONL archive line by line and deserialises each record.
        Returns an empty list when the archive file does not yet exist.

        Returns:
            Raw records in file order.

        Raises:
            OSError: If the archive exists but cannot be read.
            json.JSONDecodeError: If a non-empty line is not valid JSON.
            KeyError: If a raw-record payload is missing required fields.
        """
        if not self._path.exists():
            return []
        records: list[RawLTMRecord] = []
        with self._lock:
            with open(self._path, encoding=DEFAULT_TEXT_ENCODING) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(RawLTMRecord.from_dict(json.loads(line)))
        return records

    @property
    def path(self) -> Path:
        """Filesystem path of the JSONL archive.

        Returns:
            Configured raw-record archive path.

        Raises:
            None.
        """
        return self._path

    @property
    def card_store(self) -> JsonlMemoryCardStore | None:
        """Auxiliary JSONL card audit store, when configured.

        Returns:
            Configured card store or ``None``.

        Raises:
            None.
        """
        return self._card_store
