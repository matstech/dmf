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

"""Construction of Qdrant clients for the LTM backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from dmf.utils.constants import (
    DEFAULT_LTM_QDRANT_HOST,
    DEFAULT_LTM_QDRANT_PORT,
    DEFAULT_LTM_QDRANT_SSL,
    DEFAULT_LTM_QDRANT_TIMEOUT,
    LTM_QDRANT_MODE_MEMORY,
    LTM_QDRANT_MODE_SERVER,
)

_QDRANT_EXTRA_INSTALL_MESSAGE = (
    "Install the Qdrant backend with: pip install 'dmf-memory[qdrant]'"
)


class QdrantConnectionMode(str, Enum):
    """Supported Qdrant deployment modes."""

    MEMORY = LTM_QDRANT_MODE_MEMORY
    SERVER = LTM_QDRANT_MODE_SERVER


@dataclass(frozen=True)
class QdrantConnectionConfig:
    """Connection parameters used by :func:`build_qdrant_client`."""

    mode: QdrantConnectionMode = QdrantConnectionMode.MEMORY
    host: str = DEFAULT_LTM_QDRANT_HOST
    port: int = DEFAULT_LTM_QDRANT_PORT
    ssl: bool = DEFAULT_LTM_QDRANT_SSL
    api_key: str | None = field(default=None, repr=False)
    timeout: int = DEFAULT_LTM_QDRANT_TIMEOUT

    def __post_init__(self) -> None:
        """Validate parameters used by a direct server connection."""
        if self.mode != QdrantConnectionMode.SERVER:
            return
        if not self.host.strip():
            raise ValueError("Qdrant server host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Qdrant server port must be between 1 and 65535")
        if self.timeout <= 0:
            raise ValueError("Qdrant server timeout must be greater than zero")


def build_qdrant_client(connection: QdrantConnectionConfig) -> object:
    """Build a Qdrant client for the requested deployment mode."""
    if connection.mode == QdrantConnectionMode.MEMORY:
        QdrantClient = _qdrant_client_class()
        return QdrantClient(":memory:")

    if connection.mode == QdrantConnectionMode.SERVER:
        QdrantClient = _qdrant_client_class()
        return QdrantClient(
            host=connection.host,
            port=connection.port,
            https=connection.ssl,
            api_key=connection.api_key,
            timeout=connection.timeout,
        )

    raise ValueError(f"Unsupported Qdrant connection mode: {connection.mode!r}")


def _qdrant_client_class() -> type:
    """Load the optional client dependency with an actionable failure."""
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        if exc.name == "qdrant_client":
            raise ModuleNotFoundError(_QDRANT_EXTRA_INSTALL_MESSAGE) from exc
        raise
    return QdrantClient


__all__ = [
    "QdrantConnectionConfig",
    "QdrantConnectionMode",
    "build_qdrant_client",
]
