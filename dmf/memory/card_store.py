"""JSONL audit persistence for auxiliary memory cards."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from dmf.memory.card_projection import MemoryCardProjector
from dmf.models.memory import MemoryCard, MemoryEntry
from dmf.utils.constants import DEFAULT_TEXT_ENCODING


class JsonlMemoryCardStore:
    """Append-only JSONL audit archive for structured auxiliary memory cards.

    This is a backend-neutral persistence layer for projected cards. It can
    be used alongside raw file or raw Chroma LTM archival, but it is not a
    Chroma card index and does not provide semantic card retrieval.

    Args:
        storage_path: Filesystem path for the JSONL card archive.
        enabled: Whether projection and persistence should run.
        projector: Optional projector override for tests or custom extraction.

    Returns:
        Card store instance.

    Raises:
        OSError: If the parent directory cannot be created.
    """

    def __init__(
        self,
        storage_path: Path | str,
        *,
        enabled: bool = True,
        projector: MemoryCardProjector | None = None,
    ) -> None:
        self._path = Path(storage_path)
        self._enabled = enabled
        self._projector = projector or MemoryCardProjector()
        self._lock = threading.Lock()
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def archive(self, entry: MemoryEntry) -> list[MemoryCard]:
        """Project and persist cards for one raw memory entry.

        Args:
            entry: Working-memory entry to project into cards.

        Returns:
            Persisted cards, or an empty list when disabled or unprojectable.

        Raises:
            OSError: If the JSONL archive cannot be written.
            TypeError: If a projected card cannot be JSON-serialised.
        """
        if not self._enabled:
            return []

        cards = self._projector.project(entry)
        if not cards:
            return []

        with self._lock:
            with open(self._path, "a", encoding=DEFAULT_TEXT_ENCODING) as fh:
                for card in cards:
                    fh.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        return cards

    def read_all(self) -> list[MemoryCard]:
        """Read all persisted cards from the JSONL archive.

        Returns:
            Persisted cards in file order.

        Raises:
            OSError: If the archive exists but cannot be read.
            json.JSONDecodeError: If a non-empty line is not valid JSON.
            KeyError: If a card payload is missing required fields.
        """
        if not self._path.exists():
            return []
        cards: list[MemoryCard] = []
        with open(self._path, encoding=DEFAULT_TEXT_ENCODING) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    cards.append(MemoryCard.from_dict(json.loads(stripped)))
        return cards

    @property
    def path(self) -> Path:
        """Filesystem path of the card JSONL archive.

        Returns:
            Configured JSONL path.

        Raises:
            None.
        """
        return self._path

    @property
    def enabled(self) -> bool:
        """Whether card persistence is enabled.

        Returns:
            True when archive writes are enabled.

        Raises:
            None.
        """
        return self._enabled
